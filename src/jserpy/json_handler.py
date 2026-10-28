import base64
import datetime
import json
import sys
import typing
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from functools import lru_cache, partial
from itertools import repeat
from types import GenericAlias, MappingProxyType, UnionType
from typing import Any, TypeVar, Generic, cast, Type, get_origin, get_args, Sequence, Union, Tuple
# from typing import Union as UnionType
from enum import Enum
from pathlib import Path, PurePath

from typing_inspect import is_optional_type, get_origin

from jserpy.json_handler_utils import Dataclass, is_primitive
from jserpy.json_typing import JSON
from jserpy.jsonable import Jsonable

# Define types
T = TypeVar('T')
J = TypeVar('J', bound=Union[Jsonable, Dataclass])
JJ = TypeVar('JJ', bound=Jsonable)
DC = TypeVar('DC', bound=dataclass)
Fallback = Callable[[Any], Any]


def _type_name(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


_NESTED_TOO_DEEPLY = "JSON structure is nested too deeply to serialize"
_NESTED_TOO_DEEPLY_TO_DESERIALIZE = "JSON structure is nested too deeply to deserialize"
# Bound of the explicit stack of `_SerializationContext.prepare`, for values that never stop nesting (e.g. a handler
# that returns a new container of its own type). Well above what the stdlib encoder handles: about 10_000 levels up to
# 3.13 and about 100_000 on 3.14 (8 MiB stack).
_MAX_PREPARE_DEPTH = 200_000
# Bound of the explicit stack of `_deserialize`, for input that never stops nesting (a cyclic dict or list). The same
# as `_MAX_PREPARE_DEPTH`: a typed level takes one or two entries (e.g. a union and a dataclass), and `json.loads`
# parses about 10_000 levels up to 3.13 and about 100_000 on 3.14.
_MAX_DESERIALIZE_DEPTH = 200_000

# Kinds of `_Frame`; `_PENDING` stands for a prepared (or deserialized) value that is not ready yet (see
# `_SerializationContext.prepare` and `_deserialize`)
_MAPPING_FRAME = 0
_LIST_FRAME = 1
_TUPLE_FRAME = 2
_HANDLER_FRAME = 3
_PENDING = object()


_NATIVE_JSON_TYPES = frozenset((str, int, float, bool, type(None)))


def _normalize_annotation(cls: Any) -> Any:
    """Return the annotation that decides how a value is deserialized, with the wrappers around it removed.

    `None` stands for NoneType (PEP 484), `NewType` for its supertype, `Annotated[X, ...]` and `Final[X]` for X (a bare
    `Final` says nothing, so it stands for Any). They nest (`Final[Annotated[UserId, ...]]`). Everything else, such as
    `Optional[UserId]` or `list[UserId]`, is returned as it is: the wrappers inside it are removed when the members of
    the union or the items of the container are entered.
    """
    while not isinstance(cls, type):
        if cls is None:
            return type(None)
        # not isinstance(cls, typing.NewType): typing_extensions has a NewType of its own before Python 3.10
        supertype = getattr(cls, "__supertype__", None)
        if supertype is not None:
            cls = supertype
        elif typing.get_origin(cls) in (typing.Annotated, typing.Final):
            cls = typing.get_args(cls)[0]
        elif cls is typing.Final:
            return Any
        else:
            return cls
    return cls


def _is_jsonable_primitive(value: Any) -> bool:
    """Whether `value` is a `Jsonable` that subclasses str, int or float.

    The stdlib encoder writes such a value as the primitive, without calling `default`: it needs to reach its handler
    before that.
    """
    return (
        type(value) not in _NATIVE_JSON_TYPES
        and isinstance(value, (str, int, float))
        and isinstance(value, Jsonable)
    )


class _Frame:
    """An entry of the explicit stack of `_SerializationContext.prepare`.

    A container frame holds the iterator over the source container and the prepared items so far (`key` is the
    converted key of the mapping item in progress). A handler frame keeps a value tracked while the result of its
    registered handler is prepared.
    """

    __slots__ = ("kind", "source", "items", "prepared", "key")

    def __init__(self, kind: int, source: Any, prepared: Any = None):
        self.kind = kind
        self.source = source
        self.items: Any = None
        self.prepared = prepared
        self.key: str | None = None


class _SerializationContext:
    """State of a single serialization call.

    `prepare` normalizes containers (mapping keys, tuples, sets, ...) before they reach `json.dumps`, and tracks
    the containers currently being converted so that cycles fail predictably. Error messages never include the
    serialized values, only type names.
    """

    def __init__(self, *, fallback: Fallback | None, allow_nan: bool):
        self.fallback = fallback
        self.allow_nan = allow_nan
        self._active_ids: set[int] = set()

    @contextmanager
    def tracking(self, value: Any):
        value_id = id(value)
        if value_id in self._active_ids:
            raise ValueError("Cyclic reference detected during JSON serialization")

        self._active_ids.add(value_id)
        try:
            yield
        finally:
            self._active_ids.remove(value_id)

    def prepare(self, value: Any) -> Any:
        # A depth-first walk with an explicit stack instead of recursion, so that the nesting depth is limited only by
        # the stdlib encoder, not by Python frames. Values are visited, tracked and checked in the same order as by a
        # recursive walk.
        stack: list[_Frame] = []
        try:
            result = self._enter(value, stack)
            while stack:
                frame = stack[-1]
                if result is not _PENDING:
                    # `result` is the prepared form of what the top frame waits for
                    if frame.kind == _HANDLER_FRAME:
                        stack.pop()
                        self._active_ids.remove(id(frame.source))
                        continue
                    if frame.kind == _MAPPING_FRAME:
                        frame.prepared[frame.key] = result
                    else:
                        frame.prepared.append(result)
                result = self._fill(frame, stack)
        except BaseException:
            for frame in stack:
                self._active_ids.discard(id(frame.source))
            raise
        return result

    def _enter(self, value: Any, stack: list[_Frame]) -> Any:
        """Return the prepared form of `value`, or `_PENDING` after pushing a frame for a container to fill."""
        while True:
            if id(value) in self._active_ids:
                raise ValueError("Cyclic reference detected during JSON serialization")

            if not isinstance(value, (Mapping, list, tuple)) and not _is_jsonable_primitive(value):
                return value

            if len(stack) >= _MAX_PREPARE_DEPTH:
                raise ValueError(_NESTED_TOO_DEEPLY)

            handler = _get_handler(type(value))
            if handler is not None and handler not in (ListJsonSerializingHandler, TupleJsonSerializingHandler):
                if handler is SetJsonSerializingHandler:
                    return self.prepare_set(value)
                # As in `prepare_with_handler`: `value` stays tracked until its serialized form is prepared
                stack.append(_Frame(_HANDLER_FRAME, value))
                self._active_ids.add(id(value))
                value = self._serialize_with_handler(value, handler)
                continue

            if isinstance(value, Mapping):
                frame = _Frame(_MAPPING_FRAME, value, {})
            elif isinstance(value, list):
                frame = _Frame(_LIST_FRAME, value, [])
            else:
                frame = _Frame(_TUPLE_FRAME, value, [])
            stack.append(frame)
            self._active_ids.add(id(value))
            frame.items = iter(value.items() if frame.kind == _MAPPING_FRAME else value)
            return _PENDING

    def _fill(self, frame: _Frame, stack: list[_Frame]) -> Any:
        """Prepare the remaining items of the container `frame`, the top of `stack`.

        Return the prepared container once all items are done, or the result of `_enter` for an item that pushed
        frames of its own (the item is stored when that result is ready).
        """
        prepared = frame.prepared
        if frame.kind == _MAPPING_FRAME:
            for key, item in frame.items:
                converted_key = self.convert_key(key)
                if converted_key in prepared:
                    raise ValueError("JSON mapping key collision detected")
                result = self._enter(item, stack)
                if stack[-1] is not frame:
                    frame.key = converted_key
                    return result
                prepared[converted_key] = result
        else:
            for item in frame.items:
                result = self._enter(item, stack)
                if stack[-1] is not frame:
                    return result
                prepared.append(result)

        stack.pop()
        self._active_ids.remove(id(frame.source))
        return tuple(prepared) if frame.kind == _TUPLE_FRAME else prepared

    def convert_key(self, key: Any) -> str:
        if isinstance(key, Enum):
            enum_value = EnumJsonSerializingHandler.serialize(key)
            if enum_value is None:
                return "null"
            return self.convert_key(enum_value)

        if isinstance(key, tuple):
            return _encode_json(
                key,
                context=self,
                ensure_ascii=True,
                allow_nan=self.allow_nan,
                indent=None,
                separators=None,
                sort_keys=False,
            )

        if isinstance(key, str):
            return str.__str__(key)

        if isinstance(key, bool):
            return "true" if key else "false"

        if isinstance(key, int):
            return int.__repr__(key)

        if isinstance(key, float):
            return json.dumps(key, allow_nan=self.allow_nan)

        raise TypeError(f"Unsupported JSON mapping key type: {_type_name(key)}")

    def prepare_set(self, value: set[Any] | frozenset[Any]) -> list[JSON]:
        # Sets have no order: sort the items by their canonical (compact, sorted-keys) JSON text, not by repr().
        normalized_items: list[tuple[str, JSON]] = []
        with self.tracking(value):
            for item in value:
                canonical = _encode_json(
                    item,
                    context=self,
                    ensure_ascii=True,
                    allow_nan=self.allow_nan,
                    indent=None,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                normalized_items.append((canonical, cast(JSON, json.loads(canonical))))

        normalized_items.sort(key=lambda item: item[0])
        return [item for _, item in normalized_items]

    def prepare_with_handler(
        self,
        value: Any,
        handler: type["JsonSerializingHandler[Any]"],
    ) -> Any:
        if handler is SetJsonSerializingHandler:
            return self.prepare_set(value)

        with self.tracking(value):
            return self.prepare(self._serialize_with_handler(value, handler))

    @staticmethod
    def _serialize_with_handler(value: Any, handler: type["JsonSerializingHandler[Any]"]) -> Any:
        serialized = handler.serialize(value)
        if serialized is value:
            raise ValueError(
                "Cyclic reference returned by registered JSON handler for "
                f"{_type_name(value)}"
            )
        return serialized


def convert_tuple_key(t: tuple) -> str:
    serialized_tuple = serialize_json(t)
    return serialized_tuple


def _restore_tuple_key(key: str, cls: Union[Type[Tuple], GenericAlias]) -> tuple:
    try:
        deserialized_obj = json.loads(key)
    except ValueError:
        # Not JSON text (a JSONDecodeError holds the key itself). Raised outside the handler, so that the original is
        # neither chained nor kept as `__context__`.
        pass
    else:
        return deserialize_json(deserialized_obj, cls)
    raise TypeError(f"Expected a JSON array as the mapping key for {cls}, got text that is not JSON")


def _restore_enum_key(key: str, cls: Type[T]) -> T:
    try:
        return EnumJsonSerializingHandler.deserialize(key, cls)
    except TypeError:
        pass

    # The key text of a member is that of its value (see `_SerializationContext.convert_key`): the value itself for a
    # str, but "1" for 1, "true" for True, "[0, 0]" for (0, 0)
    context = _SerializationContext(fallback=None, allow_nan=True)
    for member in cls:
        try:
            member_key = context.convert_key(member)
        except (TypeError, ValueError):
            continue
        if member_key == key:
            return member
    raise TypeError(f"Invalid value for {cls}, got {type(key).__name__}")


def _reconstruct_key(key: str, cls: Type[T]) -> T:
    # issubclass() takes the origin: it rejects a parameterized `typing.Tuple[...]` (and, from Python 3.13, `tuple[...]`)
    cls_origin = cast(type, get_origin(cls))
    if cls_origin is None:
        cls_origin = cls

    if issubclass(cls_origin, Enum):
        return _restore_enum_key(key, cls)

    if issubclass(cls_origin, tuple):
        return _restore_tuple_key(key, cls)

    if not issubclass(cls_origin, str):
        raise ValueError(f"Couldn't de-convert JSON mapping key to its original type: {cls}")

    return key


def _deserialize_keys(obj: Any, keys_cls: Type[T]) -> Any:
    keys_cls = _normalize_annotation(keys_cls)
    if keys_cls is Any:
        # nothing to restore the keys to: they stay the strings of the JSON object
        return obj
    if isinstance(obj, list):
        return [_deserialize_keys(item, keys_cls) for item in obj]
    if not isinstance(obj, dict):
        return obj

    return {_reconstruct_key(k, keys_cls): v for k, v in obj.items()}


# Steps of deserializing a container: a generator that yields `(data, cls)` for every nested value, receives the
# deserialized value back (or has the error raised at the `yield`), and returns the deserialized container
_DeserializeSteps = Generator[tuple[Any, Any], Any, Any]


def _dict_value_steps(data: dict, value_cls: Any) -> _DeserializeSteps:
    values = {}
    for key, value in data.items():
        values[key] = yield value, value_cls
    return values


def _require_json_array(data: Any, cls: Any) -> None:
    # A str, bytes or dict is iterable too, and would be split into characters, byte values or keys. Raise
    # TypeError (like the other handlers) so that unions try their next member; name types, never the value.
    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Expected a JSON array for {cls}, got {type(data).__name__}")


# What a failed conversion such as `Enum(...)`, `Decimal(...)`, `fromisoformat` or `b64decode` raises
_CONVERSION_ERRORS = (ValueError, TypeError, ArithmeticError)


def _convert(data: Any, cls: Any, convert: Callable[[Any], T]) -> T:
    """Return `convert(data)`, raising a TypeError that names types only if the conversion fails.

    The original errors often echo the value (e.g. "'x' is not a valid Color") and are not always TypeErrors, which
    unions need to try their next member. The original is dropped, not chained or kept as `__context__`.
    """
    try:
        return convert(data)
    except _CONVERSION_ERRORS:
        pass
    raise TypeError(f"Invalid value for {cls}, got {type(data).__name__}")


_JSON_CONTAINER_TYPES = frozenset((dict, list))


def _copy_json(value: Any) -> Any:
    """Copy the JSON containers in `value`, so that the result shares no mutable container with it.

    Used where deserialization passes input through as is. Only exact dicts and lists (what `json.loads` builds) are
    copied, at any depth; anything else, such as scalars, tuples, dict or list subclasses and other objects, is kept by
    reference along with what it contains. Iterative, so the nesting depth is not limited by recursion; repeated and
    cyclic references are preserved.
    """
    if type(value) is not dict and type(value) is not list:
        return value

    root = value.copy()
    copies = {id(value): root}
    pending = [root]
    while pending:
        container = pending.pop()
        if type(container) is dict:
            entries = container.items()
        elif _JSON_CONTAINER_TYPES.isdisjoint(map(type, container)):
            # a list of scalars only (checked at C speed): its shallow copy is complete
            continue
        else:
            entries = enumerate(container)
        for key, item in entries:
            item_type = type(item)
            if item_type is dict or item_type is list:
                item_copy = copies.get(id(item))
                if item_copy is None:
                    item_copy = copies[id(item)] = item.copy()
                    pending.append(item_copy)
                container[key] = item_copy
    return root


# Abstract base class for JSON serializing handlers
class JsonSerializingHandler(Generic[T], ABC):
    # Handlers of containers define `deserialize_steps(data, cls) -> _DeserializeSteps`, which `deserialize_json`
    # runs instead of `deserialize` so that nested values cost no Python frames (see `_deserialize`)
    deserialize_steps: Callable[[Any, Any], _DeserializeSteps] | None = None

    @staticmethod
    @abstractmethod
    def serialize(obj: T) -> JSON:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def deserialize(data: JSON, cls: type[T]) -> T:
        raise NotImplementedError()


class NumpyTypeJsonSerializingHandler(JsonSerializingHandler):
    """NumPy scalars. NumPy is imported lazily (see jserpy._numpy)."""

    @staticmethod
    def serialize(obj: Any) -> JSON:
        from jserpy._numpy import serialize_numpy
        return serialize_numpy(obj, "scalar")

    @staticmethod
    def deserialize(data: JSON, cls: type) -> Any:
        from jserpy._numpy import deserialize_numpy
        if data is None:
            # np.float64(None) is nan, np.bool_(None) False, ...: null is not a NumPy scalar (so Optional[...] gets it)
            raise TypeError(f"Expected a JSON value of a NumPy scalar for {cls}, got NoneType")
        # copied, since np.object_(x) returns x itself
        return _convert(_copy_json(data), cls, partial(deserialize_numpy, cls=cls, kind="scalar"))


# Handler for Jsonable objects
class JsonableSerializingHandler(JsonSerializingHandler[JJ]):

    @staticmethod
    def serialize(obj: JJ) -> JSON:
        return obj.to_json()

    @staticmethod
    def deserialize(data: JSON, cls: type[JJ]) -> JJ:
        return cls.from_json(data)


def _is_empty_tuple_annotation(cls: Any) -> bool:
    """Whether `cls` is `tuple[()]` (or `Tuple[()]`), a tuple of length 0, and not a bare `tuple`.

    `get_args` of `tuple[()]` is `((),)` before Python 3.11 and `()` from 3.11, which is what a bare tuple has too.
    """
    generic_args = get_args(cls)
    if generic_args:
        return generic_args == ((),)
    return cls is not tuple and cls is not Tuple and typing.get_origin(cls) is tuple


class TupleJsonSerializingHandler(JsonSerializingHandler[tuple]):

    @staticmethod
    def serialize(obj: tuple) -> JSON:
        return list(obj)

    @staticmethod
    def deserialize(data: JSON, cls: type[tuple]) -> tuple:
        return _run_deserialize_steps(TupleJsonSerializingHandler.deserialize_steps(data, cls))

    @staticmethod
    def deserialize_steps(data: JSON, cls: type[tuple]) -> _DeserializeSteps:
        _require_json_array(data, cls)
        if _is_empty_tuple_annotation(cls):
            if data:
                raise TypeError(f"Expected a JSON array of length 0 for {cls}, got length {len(data)}")
            return ()
        generic_args_types = get_args(cls)
        if not generic_args_types:
            return tuple(_copy_json(data if type(data) is list else list(data)))

        if len(generic_args_types) == 2 and generic_args_types[1] is Ellipsis:
            item_types = repeat(generic_args_types[0])
        elif len(data) == len(generic_args_types):
            item_types = generic_args_types
        elif len(generic_args_types) == 1:
            item_types = repeat(generic_args_types[0])
        else:
            raise TypeError(
                f"Expected a JSON array of length {len(generic_args_types)} for {cls}, got length {len(data)}"
            )

        items = []
        for item, item_type in zip(data, item_types):
            items.append((yield item, item_type))
        return tuple(items)


class ListJsonSerializingHandler(JsonSerializingHandler[list]):

    @staticmethod
    def serialize(obj: list) -> JSON:
        return obj

    @staticmethod
    def deserialize(data: JSON, cls: type[list]) -> list:
        return _run_deserialize_steps(ListJsonSerializingHandler.deserialize_steps(data, cls))

    @staticmethod
    def deserialize_steps(data: JSON, cls: type[list]) -> _DeserializeSteps:
        generic_args = get_args(cls)
        if not generic_args and data is None:
            # an unparameterized list passes null through (e.g. a missing dataclass field), as before
            return data
        _require_json_array(data, cls)
        if not generic_args:
            return _copy_json(data if type(data) is list else list(data))

        generic_type = generic_args[0]
        items = []
        for item in data:
            items.append((yield item, generic_type))
        return items


class GenericJsonSerializingHandler(JsonSerializingHandler[GenericAlias]):

    @staticmethod
    def serialize(obj: GenericAlias) -> JSON:
        raise NotImplementedError()

    @staticmethod
    def deserialize(data: JSON, cls: type[GenericAlias]) -> T:
        pass


class SetJsonSerializingHandler(JsonSerializingHandler[set[Any] | frozenset[Any]]):
    """set / frozenset: serialized as a JSON array in a deterministic order (see `_SerializationContext.prepare_set`)."""

    @staticmethod
    def serialize(obj: set[Any] | frozenset[Any]) -> JSON:
        return list(obj)

    @staticmethod
    def deserialize(data: JSON, cls: type[set[Any] | frozenset[Any]]) -> set[Any] | frozenset[Any]:
        return _run_deserialize_steps(SetJsonSerializingHandler.deserialize_steps(data, cls))

    @staticmethod
    def deserialize_steps(data: JSON, cls: type[set[Any] | frozenset[Any]]) -> _DeserializeSteps:
        _require_json_array(data, cls)
        cls_origin = get_origin(cls) or cls
        generic_args = get_args(cls)
        values = data
        if generic_args:
            item_type = generic_args[0]
            values = []
            for item in data:
                values.append((yield item, item_type))
        return cls_origin(values)


class MappingJsonSerializingHandler(JsonSerializingHandler):
    """Handles MappingProxyType (and Mapping annotations): serialized as a plain JSON object."""

    @staticmethod
    def serialize(obj: Mapping) -> JSON:
        # Keys and nested values are converted by `_SerializationContext.prepare`.
        return cast(JSON, dict(obj))

    @staticmethod
    def deserialize(data: JSON, cls: type[Mapping]) -> Mapping:
        return _run_deserialize_steps(MappingJsonSerializingHandler.deserialize_steps(data, cls))

    @staticmethod
    def deserialize_steps(data: JSON, cls: type[Mapping]) -> _DeserializeSteps:
        if not isinstance(data, dict):
            raise TypeError(f"Expected a JSON object for {cls}, got {type(data).__name__}")
        cls_origin = get_origin(cls) or cls
        generic_args = get_args(cls)
        if generic_args:
            key_cls, value_cls = generic_args
            data = _deserialize_keys(data, key_cls)
            data = yield from _dict_value_steps(data, value_cls)
        else:
            # the values are passed through as is: copy them, so that the result does not alias the input
            data = _copy_json(data if type(data) is dict else dict(data))
        if issubclass(cls_origin, MappingProxyType):
            return MappingProxyType(data)
        return data


class LiteralJsonSerializingHandler(JsonSerializingHandler):

    @staticmethod
    def serialize(obj: Any) -> JSON:
        return obj

    @staticmethod
    def deserialize(data: JSON, cls: Any) -> Any:
        allowed = get_args(cls)
        # compare type too, so that e.g. 1 / True do not match each other
        if any(type(data) is type(value) and data == value for value in allowed):
            return data
        raise TypeError(f"Expected one of the allowed literals {allowed}, got {type(data).__name__}")


class UnionJsonSerializingHandler(JsonSerializingHandler):

    @staticmethod
    def serialize(obj: Any) -> JSON:
        return str(obj)

    @staticmethod
    def deserialize(data: JSON, cls: type[UnionType]) -> T:
        cls_list = get_args(cls)
        return deserialize_multi_cls_from_json(data, cls_list)


@lru_cache(maxsize=None)
def _resolved_field_types(cls: type) -> dict[str, Any]:
    """Resolve string annotations (e.g. `from __future__ import annotations`) once per class."""
    try:
        return typing.get_type_hints(cls)
    except RecursionError:
        # raised when called near the recursion limit, not because of `cls`: propagate it instead of caching `{}`
        raise
    except Exception:
        return {}


def _has_forward_reference(annotation: Any) -> bool:
    """Whether `annotation` holds a string or `ForwardRef`, e.g. `Optional["Node"]` or `list["Node"]`.

    The type arguments of `Literal` (values) and the metadata of `Annotated` are not annotations; they are skipped.
    Iterative, since annotations can nest as deeply as the data.
    """
    pending = [annotation]
    while pending:
        item = pending.pop()
        if isinstance(item, (str, typing.ForwardRef)):
            return True
        origin = typing.get_origin(item)
        if origin is typing.Literal:
            continue
        if origin is typing.Annotated:
            pending.append(typing.get_args(item)[0])
            continue
        pending.extend(typing.get_args(item))
    return False


def _resolve_forward_references(annotation: Any, globalns: dict[str, Any], localns: dict[str, Any]) -> Any:
    """Evaluate the string and `ForwardRef` arguments inside `annotation`.

    `typing.get_type_hints` leaves the strings in builtin generics such as `list["Node"]` before Python 3.11. A
    reference that cannot be evaluated is left as it is, which fails when the field is deserialized.
    """
    if isinstance(annotation, (str, typing.ForwardRef)):
        expression = annotation if isinstance(annotation, str) else annotation.__forward_arg__
        try:
            evaluated = eval(expression, globalns, localns)
        except Exception:
            return annotation
        if isinstance(evaluated, (str, typing.ForwardRef)):
            return annotation
        return _resolve_forward_references(evaluated, globalns, localns)

    args = typing.get_args(annotation)
    origin = typing.get_origin(annotation)
    if not args or origin is typing.Literal:
        return annotation
    if origin is typing.Annotated:
        # only the first argument is an annotation, the others are metadata
        resolved = _resolve_forward_references(args[0], globalns, localns)
        return annotation if resolved is args[0] else typing.Annotated[(resolved, *args[1:])]
    resolved_args = tuple(_resolve_forward_references(arg, globalns, localns) for arg in args)
    if all(resolved is arg for resolved, arg in zip(resolved_args, args)):
        return annotation
    try:
        if isinstance(annotation, GenericAlias):
            return GenericAlias(typing.get_origin(annotation), resolved_args)
        if typing.get_origin(annotation) is Union or isinstance(annotation, UnionType):
            return Union[resolved_args]
        return annotation.copy_with(resolved_args)
    except Exception:
        return annotation


@lru_cache(maxsize=None)
def _dataclass_field_types(cls: type) -> dict[str, Any]:
    """The annotation of each field of the dataclass `cls`, with the references to classes by name resolved once."""
    resolved_types = _resolved_field_types(cls)
    module = sys.modules.get(getattr(cls, "__module__", None))
    globalns = getattr(module, "__dict__", {})
    localns = {cls.__name__: cls} if hasattr(cls, "__name__") else {}

    field_types = {}
    for field in fields(cls):
        field_type = field.type
        if isinstance(field_type, str) or _has_forward_reference(field_type):
            field_type = resolved_types.get(field.name, field_type)
        if _has_forward_reference(field_type):
            field_type = _resolve_forward_references(field_type, globalns, localns)
        field_types[field.name] = field_type
    return field_types


# Handler for dataclasses with recursive deserialization
class DataclassJsonSerializingHandler(JsonSerializingHandler[DC]):

    @staticmethod
    def serialize(obj: DC) -> JSON:
        # Not dataclasses.asdict: it deep-copies values and fails on non-copyable ones (e.g. MappingProxyType).
        # Nested values are converted by `_SerializationContext.prepare` and the encoder.
        return cast(JSON, {field.name: getattr(obj, field.name) for field in fields(obj)})

    @staticmethod
    def deserialize(data: JSON, cls: type[DC]) -> DC:
        return _run_deserialize_steps(DataclassJsonSerializingHandler.deserialize_steps(data, cls))

    @staticmethod
    def deserialize_steps(data: JSON, cls: type[DC]) -> _DeserializeSteps:
        if not isinstance(data, Mapping):
            # also for a dataclass without fields, which would otherwise never look at its data
            raise TypeError(f"Expected a JSON object for {cls}, got {type(data).__name__}")
        kwargs = {}
        not_init = {}
        field_types = _dataclass_field_types(cls)
        # Iterate through each field of the dataclass
        for field in fields(cls):
            field_name = field.name
            field_type = field_types[field_name]
            try:
                field_value = data[field_name]
            except KeyError as e:
                if not field.init:
                    # e.g. computed in `__post_init__`: keeps what the constructor made of it
                    continue
                if not is_optional_type(field_type):
                    pass
                    # LOG.warning(f"Couldn't find a required field named {field_name} of dataclass {cls} "
                    #           f"in the data object with keys: {list(data.keys())}")
                field_value = None

            deserialized_value = yield field_value, field_type
            if field.init:
                kwargs[field_name] = deserialized_value
            else:
                not_init[field_name] = deserialized_value

        instance = cls(**kwargs)
        for field_name, deserialized_value in not_init.items():
            # `__init__` does not take these; set them like the constructor does (also for frozen and slots classes)
            object.__setattr__(instance, field_name, deserialized_value)
        return instance


# Handler for numpy arrays
class NumpyJsonSerializingHandler(JsonSerializingHandler):

    @staticmethod
    def serialize(obj: Any) -> JSON:
        from jserpy._numpy import serialize_numpy
        return serialize_numpy(obj, "array")

    @staticmethod
    def deserialize(data: JSON, cls: type) -> Any:
        from jserpy._numpy import deserialize_numpy
        if data is None:
            # np.array(None) is a 0-d object array: null is not an array (so Optional[...] gets it)
            raise TypeError(f"Expected a JSON array for {cls}, got NoneType")
        # a ragged array raises ValueError in NumPy >= 1.24
        convert = partial(deserialize_numpy, cls=cls, kind="array")
        array = _convert(data, cls, convert)
        if array.dtype == object:
            # an object array holds the input's own dicts and lists: build it from a copy instead
            array = _convert(_copy_json(data), cls, convert)
        return array


class BytesJsonSerializingHandler(JsonSerializingHandler):

    @staticmethod
    def serialize(obj: bytes) -> JSON:
        # Encode bytes to Base64 string
        base64_str = base64.b64encode(obj).decode('ascii')
        return base64_str

    @staticmethod
    def deserialize(data: JSON, cls: type[bytes]) -> bytes:
        # Decode Base64 string back to bytes (of the subclass, if that is what is asked for)
        decoded = _convert(data, cls, base64.b64decode)
        return decoded if cls is bytes else _convert(decoded, cls, cls)


class DatetimeJsonSerializingHandler(JsonSerializingHandler):
    """Convert datetime (or date, time) object to ISO 8601 string and back to datetime (or date, time) object."""

    @staticmethod
    def serialize(obj: datetime.datetime) -> JSON:
        return obj.isoformat()

    @staticmethod
    def deserialize(data: JSON, cls: type[datetime.datetime]) -> T:
        return _convert(data, cls, cls.fromisoformat)


class DecimalJsonSerializingHandler(JsonSerializingHandler[Decimal]):
    """Decimal <-> string, so that precision and trailing zeroes survive."""

    @staticmethod
    def serialize(obj: Decimal) -> JSON:
        return str(obj)

    @staticmethod
    def deserialize(data: JSON, cls: type[Decimal]) -> Decimal:
        return _convert(data, cls, cls)


def _is_json_of(value: Any, data: JSON) -> bool:
    """Whether the parsed JSON `data` is what `value` is written as: a tuple is an array, at any depth."""
    if isinstance(value, tuple):
        return type(data) is list and len(value) == len(data) and all(map(_is_json_of, value, data))
    return value == data


# Handler for Enum objects
class EnumJsonSerializingHandler(JsonSerializingHandler):

    @staticmethod
    def serialize(obj: T) -> JSON:
        return obj.value

    @staticmethod
    def deserialize(data: JSON, cls: type[T]) -> T:
        try:
            return cls(data)
        except _CONVERSION_ERRORS:
            pass

        if type(data) is list:
            # the value of a member may be a tuple, which was written as an array, and a list is not equal to it
            for member in cls:
                if _is_json_of(member.value, data):
                    return member
        # raised outside the `except`, so that the original error (it echoes the value) is not kept as `__context__`
        raise TypeError(f"Invalid value for {cls}, got {type(data).__name__}")


def is_json_primitive(data: JSON) -> bool:
    if isinstance(data, bytes):
        return False

    return is_primitive(data)


class PathJsonSerializingHandler(JsonSerializingHandler):
    @staticmethod
    def serialize(obj: T) -> JSON:
        return str(obj)

    @staticmethod
    def deserialize(data: JSON, cls: type[T]) -> T:
        # `cls(data)`, so that a PureWindowsPath stays one and a Path subclass keeps its class (PurePath itself makes
        # the pure path of the running platform)
        return _convert(data, cls, cls)



# Register the handlers for each type
_handlers: dict[type[T], type[JsonSerializingHandler[T]]] = {
    tuple: TupleJsonSerializingHandler,
    list: ListJsonSerializingHandler,
    set: SetJsonSerializingHandler,
    frozenset: SetJsonSerializingHandler,
    UnionType: UnionJsonSerializingHandler,
    typing.Union: UnionJsonSerializingHandler,
    bytes: BytesJsonSerializingHandler,
    datetime.datetime: DatetimeJsonSerializingHandler,
    datetime.date: DatetimeJsonSerializingHandler,
    datetime.time: DatetimeJsonSerializingHandler,
    Decimal: DecimalJsonSerializingHandler,
    Path: PathJsonSerializingHandler,
    MappingProxyType: MappingJsonSerializingHandler,
    Mapping: MappingJsonSerializingHandler,
    typing.Literal: LiteralJsonSerializingHandler,
}


# The types whose handlers serve their subclasses as well (found along the MRO, the most specific base first: a
# datetime is a date too)
_SUBCLASS_HANDLED_TYPES = frozenset((datetime.datetime, datetime.date, datetime.time, Decimal, bytes))


def is_valid_class(cls: Type[T]) -> bool:
    if cls is Any:
        return False
    if cls is UnionType:
        return False
    if cls is typing.Union:
        return False
    if isinstance(cls, TypeVar):
        return False

    return True


def _get_handler(cls: type[T]) -> typing.Optional[type[JsonSerializingHandler]]:
    # typing.get_origin (unlike typing_inspect's) recognizes `X | Y` unions, incl. ones containing generics
    cls_origin = cast(type, typing.get_origin(cls))
    if cls_origin is None:
        cls_origin = cls

    handler = _handlers.get(cls_origin)
    if handler is not None:
        return handler

    if not is_valid_class(cls_origin):
        return None

    if issubclass(cls_origin, Jsonable):
        return JsonableSerializingHandler

    if issubclass(cls_origin, PurePath):
        return PathJsonSerializingHandler

    if issubclass(cls_origin, Enum):
        return EnumJsonSerializingHandler

    if is_dataclass(cls_origin):
        return DataclassJsonSerializingHandler

    if issubclass(cls_origin, list):
        return ListJsonSerializingHandler

    if issubclass(cls_origin, tuple):
        return TupleJsonSerializingHandler

    if issubclass(cls_origin, (set, frozenset)):
        return SetJsonSerializingHandler

    from jserpy._numpy import get_numpy_kind

    numpy_kind = get_numpy_kind(cls_origin)
    if numpy_kind == "array":
        return NumpyJsonSerializingHandler
    if numpy_kind == "scalar":
        return NumpyTypeJsonSerializingHandler

    for base in cls_origin.__mro__:
        if base in _SUBCLASS_HANDLED_TYPES:
            return _handlers[base]

    return None


class CustomEncoder(json.JSONEncoder):

    def __init__(self, *args: Any, _jserpy_context: _SerializationContext | None = None, **kwargs: Any):
        self._jserpy_context = _jserpy_context or _SerializationContext(
            fallback=None,
            allow_nan=kwargs.get("allow_nan", True),
        )
        super().__init__(*args, **kwargs)

    def default(self, obj: Any):
        # Precedence: a registered handler first, the caller's fallback only for values no handler supports.
        handler = _get_handler(type(obj))
        if handler is not None:
            return self._jserpy_context.prepare_with_handler(obj, handler)

        fallback = self._jserpy_context.fallback
        if fallback is not None:
            with self._jserpy_context.tracking(obj):
                converted = fallback(obj)
                if converted is obj:
                    raise ValueError(
                        f"Fallback returned the original unsupported value for {_type_name(obj)}"
                    )
                # The result may itself contain supported (or unsupported) values: process it recursively.
                return self._jserpy_context.prepare(converted)

        return super().default(obj)


def _encode_json(
    obj: Any,
    *,
    context: _SerializationContext,
    ensure_ascii: bool,
    allow_nan: bool,
    indent: int | str | None,
    separators: tuple[str, str] | None,
    sort_keys: bool,
) -> str:
    prepared = context.prepare(obj)
    try:
        return json.dumps(
            prepared,
            cls=CustomEncoder,
            ensure_ascii=ensure_ascii,
            allow_nan=allow_nan,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
            _jserpy_context=context,
        )
    except (TypeError, ValueError, RecursionError) as error:
        _drop_encoder_notes(error)
        if str(error) == "Circular reference detected":
            raise ValueError("Cyclic reference detected during JSON serialization") from error
        raise


_ENCODER_NOTE_PREFIX = "when serializing "


def _drop_encoder_notes(error: BaseException) -> None:
    """Remove the notes that the `json` module (Python 3.14) adds to an error raised while it encodes.

    They name the dict keys and list positions on the path to the value, e.g. "when serializing dict item 'k'", and
    would leak the data into otherwise data-free errors. Notes of anyone else are kept.
    """
    notes = getattr(error, "__notes__", None)
    if not notes:
        return
    kept = [note for note in notes if not (isinstance(note, str) and note.startswith(_ENCODER_NOTE_PREFIX))]
    if kept:
        error.__notes__ = kept
    else:
        del error.__notes__


def serialize_json(
    obj: Any,
    *,
    ensure_ascii: bool = True,
    allow_nan: bool = True,
    indent: int | str | None = None,
    separators: tuple[str, str] | None = None,
    sort_keys: bool = False,
    fallback: Fallback | None = None,
) -> str:
    """Serialize `obj` to a JSON string.

    The keyword-only arguments are forwarded to `json.dumps`. `fallback` is called for a value that no registered
    handler supports; its result is serialized recursively. With no extra arguments the output is identical to
    previous versions.

    Raises ValueError for a cyclic reference, and for a structure nested more deeply than the `json` module can encode
    (including a handler or fallback that recurses without bound).
    """
    context = _SerializationContext(fallback=fallback, allow_nan=allow_nan)
    try:
        return _encode_json(
            obj,
            context=context,
            ensure_ascii=ensure_ascii,
            allow_nan=allow_nan,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )
    except RecursionError as error:
        # jserpy's own walk is iterative, but the stdlib encoder still recurses (in C, through `CustomEncoder.default`
        # for handler-converted values, and in Python with `indent` before 3.13), as do the encoding of nested sets
        # and user handlers or fallbacks. Report one documented error instead of a RecursionError that depends on the
        # caller's stack depth.
        raise ValueError(_NESTED_TOO_DEEPLY) from error


def serialize_json_as_obj(obj: Any, *, fallback: Fallback | None = None) -> "JSON":
    """Convert `obj` to a detached, JSON-compatible tree (dicts, lists, str, int, float, bool, None).

    Equal to `json.loads(serialize_json(obj, fallback=fallback))`, for any JSON root, and raises ValueError in the same
    cases.
    """
    text = serialize_json(obj, fallback=fallback)
    try:
        return cast(JSON, json.loads(text))
    except RecursionError as error:
        raise ValueError(_NESTED_TOO_DEEPLY) from error


def serialize_json_as_dict(obj: Any, *, fallback: Fallback | None = None) -> "JSON":
    """Backward-compatible alias of `serialize_json_as_obj` (the root is not necessarily a dict)."""
    return serialize_json_as_obj(obj, fallback=fallback)


# Whether a `deserialize_json` call is in progress in this context (see `deserialize_json`)
_deserializing: ContextVar[bool] = ContextVar("jserpy_deserializing", default=False)


def deserialize_json(data: JSON, cls: type[T]) -> T:
    """Deserialize `data`, a JSON value as returned by `json.loads`, to an instance of `cls`.

    Raises TypeError when `data` does not match `cls`, and ValueError for a structure nested too deeply: more deeply
    than the Python recursion limit allows through `Jsonable.from_json` implementations that call `deserialize_json`
    themselves, or beyond what the `json` module parses (for example cyclic input).
    """
    if _deserializing.get():
        # Called by a handler or a `Jsonable.from_json` during an outer call: a RecursionError passes through them
        # unchanged, as before, and the outermost call reports it
        return _deserialize(data, cls)

    token = _deserializing.set(True)
    try:
        return _deserialize(data, cls)
    except RecursionError as error:
        raise ValueError(_NESTED_TOO_DEEPLY_TO_DESERIALIZE) from error
    finally:
        _deserializing.reset(token)


def _deserialize(data: JSON, cls: Any) -> Any:
    # A depth-first walk with an explicit stack of the containers' `_DeserializeSteps` instead of recursion, so that
    # the nesting depth costs no Python frames. Values are deserialized, and errors raised and caught (e.g. by unions),
    # in the same order as by recursive calls.
    stack: list[_DeserializeSteps] = []
    result = _enter_deserialization(data, cls, stack)
    if result is _PENDING:
        result = _run_deserialize_steps_stack(stack)
    return result


def _run_deserialize_steps(steps: _DeserializeSteps) -> Any:
    return _run_deserialize_steps_stack([steps])


def _run_deserialize_steps_stack(stack: list[_DeserializeSteps]) -> Any:
    """Run the steps on `stack` to the end and return the result of the bottom ones."""
    result: Any = None  # sent to the steps on top: None starts new steps
    error: BaseException | None = None  # raised in the steps on top instead
    while stack:
        steps = stack[-1]
        try:
            if error is None:
                data, cls = steps.send(result)
            else:
                data, cls = steps.throw(error)
        except StopIteration as stop:
            stack.pop()
            result, error = stop.value, None
            continue
        except BaseException as raised:
            stack.pop()
            error = raised
            continue

        error = None
        try:
            result = _enter_deserialization(data, cls, stack)
        except BaseException as raised:
            error = raised
        else:
            if result is _PENDING:
                result = None

    if error is not None:
        raise error
    return result


def _enter_deserialization(data: JSON, cls: Any, stack: list[_DeserializeSteps]) -> Any:
    """Return `data` deserialized as `cls`, or `_PENDING` after pushing the steps of a container onto `stack`."""
    cls = _normalize_annotation(cls)

    if cls is Any:
        return _copy_json(data)

    if cls is float and type(data) is int:
        # JSON from non-Python encoders may write whole floats as ints (one too large for a float is a mismatch)
        return _convert(data, cls, float)

    if cls is type(None):
        if data is not None:
            raise TypeError(f"Expected null, got {type(data).__name__}")
        return None

    handler = _get_handler(cls)
    if handler is UnionJsonSerializingHandler:
        return _enter_union_deserialization(data, get_args(cls), stack)
    if handler is not None:
        deserialize_steps = handler.deserialize_steps
        if deserialize_steps is None:
            return handler.deserialize(data, cls)
        return _push_deserialize_steps(deserialize_steps(data, cls), stack)

    dict_cls = typing.get_origin(cls) or cls
    if isinstance(dict_cls, type) and issubclass(dict_cls, dict) and data is not None:
        # dict[K, V], a bare dict or typing.Dict, Counter, a dict subclass: a JSON object, like Mapping and unlike a
        # scalar or an array, which unions need to refuse. (null stays: it is what a missing dataclass field is.)
        if not isinstance(data, dict):
            raise TypeError(f"Expected a JSON object for {cls}, got {type(data).__name__}")
        generic_args = get_args(cls)
        if len(generic_args) == 1 and issubclass(dict_cls, Counter):
            # Counter[K]: the counts are ints
            generic_args = (generic_args[0], int)
        if generic_args:
            key_cls, value_cls = generic_args
            data = _deserialize_keys(data, key_cls)
            return _push_deserialize_steps(_dict_value_steps(data, value_cls), stack)
        # a bare dict or typing.Dict is passed through as is: copied below

    if isinstance(data, dict):
        if isinstance(cls, TypeVar):
            # a type parameter that no type argument replaced (e.g. the T of a plain `Box`): only a scalar passes
            raise TypeError(f"Unsupported type for deserialization: {cls}")
        if isinstance(cls, type) and issubclass(cls, (str, int, float)):
            # a JSON object is no text or number (bool is an int): raise TypeError so that a union moves on to a
            # member that takes objects, such as a dataclass. A scalar of the wrong kind is not checked (see below).
            raise TypeError(f"Expected a JSON scalar for {cls}, got {type(data).__name__}")

        # passed through as is (Any, object, a class without handler, ...): copy, so that the result does not alias the
        # input
        return _copy_json(data)

    if is_json_primitive(data):
        return data

    raise TypeError(f"Unsupported type for deserialization: {cls}")


def _push_deserialize_steps(steps: _DeserializeSteps, stack: list[_DeserializeSteps]) -> Any:
    if len(stack) >= _MAX_DESERIALIZE_DEPTH:
        raise ValueError(_NESTED_TOO_DEEPLY_TO_DESERIALIZE)
    stack.append(steps)
    return _PENDING


def _enter_union_deserialization(data: JSON, cls_list: Sequence[Any], stack: list[_DeserializeSteps]) -> Any:
    """Like `_enter_deserialization` with `_multi_cls_steps`, but members that are not containers are tried here.

    Most unions (e.g. `Optional[str]`) then need no steps of their own; they are pushed only under a container member.
    """
    for index, cls in enumerate(cls_list):
        depth = len(stack)
        try:
            result = _enter_deserialization(data, cls, stack)
        except TypeError:
            continue
        if result is _PENDING:
            # The member's steps run later: the steps under them try the remaining members if they raise TypeError
            steps = _multi_cls_steps(data, cls_list, running=index)
            next(steps)
            stack.insert(depth, steps)
        return result

    raise _no_matching_cls(data, cls_list)


def deserialize_multi_cls_from_json(data: JSON, cls_list: Sequence[type[T]]) -> T:
    return _run_deserialize_steps(_multi_cls_steps(data, cls_list))


def _multi_cls_steps(data: JSON, cls_list: Sequence[Any], running: int = -1) -> _DeserializeSteps:
    """Try the members of `cls_list` in order until one does not raise TypeError.

    With `running`, the steps of that member are on the stack above these ones already: the first `yield`, reached by
    priming these steps with `next()`, receives its result.
    """
    if running >= 0:
        try:
            return (yield)
        except TypeError:
            pass
    for cls in cls_list[running + 1:]:
        try:
            return (yield data, cls)
        except TypeError:
            continue

    raise _no_matching_cls(data, cls_list)


def _no_matching_cls(data: JSON, cls_list: Sequence[Any]) -> TypeError:
    return TypeError(f"Unsupported types for deserialization: {cls_list}, got {type(data).__name__}")

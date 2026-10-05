import base64
import datetime
import json
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal
from functools import lru_cache, partial
from types import GenericAlias, MappingProxyType, UnionType
from typing import Any, TypeVar, Generic, cast, Type, get_origin, get_args, Sequence, Union, Tuple
# from typing import Union as UnionType
from enum import Enum
from pathlib import Path, PurePath

from typing_inspect import is_optional_type, is_generic_type, get_origin

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
        if id(value) in self._active_ids:
            raise ValueError("Cyclic reference detected during JSON serialization")

        if isinstance(value, (Mapping, list, tuple)):
            handler = _get_handler(type(value))
            native_container_handlers = {
                ListJsonSerializingHandler,
                TupleJsonSerializingHandler,
            }
            if handler is not None and handler not in native_container_handlers:
                return self.prepare_with_handler(value, handler)

        if isinstance(value, Mapping):
            with self.tracking(value):
                return self._prepare_mapping(value)

        if isinstance(value, list):
            with self.tracking(value):
                return [self.prepare(item) for item in value]

        if isinstance(value, tuple):
            with self.tracking(value):
                return tuple(self.prepare(item) for item in value)

        return value

    def _prepare_mapping(self, value: Mapping[Any, Any]) -> dict[str, Any]:
        prepared: dict[str, Any] = {}
        for key, item in value.items():
            converted_key = self.convert_key(key)
            if converted_key in prepared:
                raise ValueError("JSON mapping key collision detected")
            prepared[converted_key] = self.prepare(item)
        return prepared

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
            serialized = handler.serialize(value)
            if serialized is value:
                raise ValueError(
                    "Cyclic reference returned by registered JSON handler for "
                    f"{_type_name(value)}"
                )
            return self.prepare(serialized)


def convert_tuple_key(t: tuple) -> str:
    serialized_tuple = serialize_json(t)
    return serialized_tuple


def _restore_tuple_key(key: str, cls: Union[Type[Tuple], GenericAlias]) -> tuple:
    deserialized_obj = json.loads(key)
    deserialized_key = deserialize_json(deserialized_obj, cls)
    return deserialized_key


def _reconstruct_key(key: str, cls: Type[T]) -> T:
    if issubclass(cls, Enum):
        return EnumJsonSerializingHandler.deserialize(key, cls)

    cls_origin = cast(type, get_origin(cls))
    if cls_origin is None:
        cls_origin = cls

    if issubclass(cls_origin, tuple):
        return _restore_tuple_key(key, cls)

    if not issubclass(cls_origin, str):
        raise ValueError(f"Couldn't de-convert JSON mapping key to its original type: {cls}")

    return key


def _deserialize_keys(obj: Any, keys_cls: Type[T]) -> Any:
    if isinstance(obj, list):
        return [_deserialize_keys(item, keys_cls) for item in obj]
    if not isinstance(obj, dict):
        return obj

    return {_reconstruct_key(k, keys_cls): v for k, v in obj.items()}


# Abstract base class for JSON serializing handlers
class JsonSerializingHandler(Generic[T], ABC):

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
        return deserialize_numpy(data, cls, "scalar")


# Handler for Jsonable objects
class JsonableSerializingHandler(JsonSerializingHandler[JJ]):

    @staticmethod
    def serialize(obj: JJ) -> JSON:
        return obj.to_json()

    @staticmethod
    def deserialize(data: JSON, cls: type[JJ]) -> JJ:
        return cls.from_json(data)


class TupleJsonSerializingHandler(JsonSerializingHandler[tuple]):

    @staticmethod
    def serialize(obj: tuple) -> JSON:
        return list(obj)

    @staticmethod
    def deserialize(data: JSON, cls: type[tuple]) -> tuple:
        # if not isinstance(cls, GenericAlias):
        #     return tuple(data)

        generic_args_types = get_args(cls)
        if not generic_args_types:
            return tuple(data)

        if len(generic_args_types) == 2 and generic_args_types[1] is Ellipsis:
            _deserialize = partial(deserialize_json, cls=generic_args_types[0])
            return tuple(map(_deserialize, data))

        if len(data) == len(generic_args_types):
            tuple_data = (deserialize_json(item, arg_cls) for item, arg_cls in zip(data, generic_args_types))
            return tuple(tuple_data)

        elif len(generic_args_types) == 1:
            generic_type = generic_args_types[0]
            _deserialize = partial(deserialize_json, cls=generic_type)
            return tuple(map(_deserialize, data))


class ListJsonSerializingHandler(JsonSerializingHandler[list]):

    @staticmethod
    def serialize(obj: list) -> JSON:
        return obj

    @staticmethod
    def deserialize(data: JSON, cls: type[list]) -> list:
        # if not isinstance(cls, GenericAlias):
        #     return data

        if not is_generic_type(cls):
            return data

        cls_generic = cast(GenericAlias, cls)
        generic_type = get_args(cls_generic)[0]
        _deserialize = partial(deserialize_json, cls=generic_type)
        return list(map(_deserialize, data))


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
        cls_origin = get_origin(cls) or cls
        generic_args = get_args(cls)
        values = data
        if generic_args:
            item_type = generic_args[0]
            values = [deserialize_json(item, item_type) for item in data]
        return cls_origin(values)


class MappingJsonSerializingHandler(JsonSerializingHandler):
    """Handles MappingProxyType (and Mapping annotations): serialized as a plain JSON object."""

    @staticmethod
    def serialize(obj: Mapping) -> JSON:
        # Keys and nested values are converted by `_SerializationContext.prepare`.
        return cast(JSON, dict(obj))

    @staticmethod
    def deserialize(data: JSON, cls: type[Mapping]) -> Mapping:
        if not isinstance(data, dict):
            raise TypeError(f"Expected a JSON object for {cls}, got {type(data).__name__}")
        cls_origin = get_origin(cls) or cls
        generic_args = get_args(cls)
        if generic_args:
            key_cls, value_cls = generic_args
            if key_cls is not Any:
                data = _deserialize_keys(data, key_cls)
            data = {k: deserialize_json(v, value_cls) for k, v in data.items()}
        if issubclass(cls_origin, MappingProxyType):
            return MappingProxyType(dict(data))
        return dict(data)


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
        raise TypeError(f"Value {data!r} is not one of the allowed literals {allowed}")


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
    except Exception:
        return {}


# Handler for dataclasses with recursive deserialization
class DataclassJsonSerializingHandler(JsonSerializingHandler[DC]):

    @staticmethod
    def serialize(obj: DC) -> JSON:
        # Not dataclasses.asdict: it deep-copies values and fails on non-copyable ones (e.g. MappingProxyType).
        # Nested values are converted by `_SerializationContext.prepare` and the encoder.
        return cast(JSON, {field.name: getattr(obj, field.name) for field in fields(obj)})

    @staticmethod
    def deserialize(data: JSON, cls: type[DC]) -> DC:
        kwargs = {}
        resolved_types = _resolved_field_types(cls)
        # Iterate through each field of the dataclass
        for field in fields(cls):
            field_name = field.name
            field_type = field.type
            if isinstance(field_type, str):
                field_type = resolved_types.get(field_name, field_type)
            try:
                field_value = data[field_name]
            except KeyError as e:
                if not is_optional_type(field_type):
                    pass
                    # LOG.warning(f"Couldn't find a required field named {field_name} of dataclass {cls} "
                    #           f"in the data object with keys: {list(data.keys())}")
                field_value = None

            deserialized_value = deserialize_json(field_value, field_type)
            kwargs[field_name] = deserialized_value

        return cls(**kwargs)


# Handler for numpy arrays
class NumpyJsonSerializingHandler(JsonSerializingHandler):

    @staticmethod
    def serialize(obj: Any) -> JSON:
        from jserpy._numpy import serialize_numpy
        return serialize_numpy(obj, "array")

    @staticmethod
    def deserialize(data: JSON, cls: type) -> Any:
        from jserpy._numpy import deserialize_numpy
        return deserialize_numpy(data, cls, "array")


class BytesJsonSerializingHandler(JsonSerializingHandler):

    @staticmethod
    def serialize(obj: bytes) -> JSON:
        # Encode bytes to Base64 string
        base64_str = base64.b64encode(obj).decode('ascii')
        return base64_str

    @staticmethod
    def deserialize(data: JSON, cls: type[bytes]) -> bytes:
        # Decode Base64 string back to bytes
        return base64.b64decode(data)


class DatetimeJsonSerializingHandler(JsonSerializingHandler):
    """Convert datetime (or date, time) object to ISO 8601 string and back to datetime (or date, time) object."""

    @staticmethod
    def serialize(obj: datetime.datetime) -> JSON:
        return obj.isoformat()

    @staticmethod
    def deserialize(data: JSON, cls: type[datetime.datetime]) -> T:
        return cls.fromisoformat(data)


class DecimalJsonSerializingHandler(JsonSerializingHandler[Decimal]):
    """Decimal <-> string, so that precision and trailing zeroes survive."""

    @staticmethod
    def serialize(obj: Decimal) -> JSON:
        return str(obj)

    @staticmethod
    def deserialize(data: JSON, cls: type[Decimal]) -> Decimal:
        return cls(data)


# Handler for Enum objects
class EnumJsonSerializingHandler(JsonSerializingHandler):

    @staticmethod
    def serialize(obj: T) -> JSON:
        return obj.value

    @staticmethod
    def deserialize(data: JSON, cls: type[T]) -> T:
        return cls(data)


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
        from pathlib import Path
        return Path(data)



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
    except ValueError as error:
        if str(error) == "Circular reference detected":
            raise ValueError("Cyclic reference detected during JSON serialization") from error
        raise


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
    """
    context = _SerializationContext(fallback=fallback, allow_nan=allow_nan)
    return _encode_json(
        obj,
        context=context,
        ensure_ascii=ensure_ascii,
        allow_nan=allow_nan,
        indent=indent,
        separators=separators,
        sort_keys=sort_keys,
    )


def serialize_json_as_obj(obj: Any, *, fallback: Fallback | None = None) -> "JSON":
    """Convert `obj` to a detached, JSON-compatible tree (dicts, lists, str, int, float, bool, None).

    Equal to `json.loads(serialize_json(obj, fallback=fallback))`, for any JSON root.
    """
    return cast(JSON, json.loads(serialize_json(obj, fallback=fallback)))


def serialize_json_as_dict(obj: Any, *, fallback: Fallback | None = None) -> "JSON":
    """Backward-compatible alias of `serialize_json_as_obj` (the root is not necessarily a dict)."""
    return serialize_json_as_obj(obj, fallback=fallback)


def deserialize_json(data: JSON, cls: type[T]) -> T:
    if cls is Any:
        return data

    if cls is float and type(data) is int:
        # JSON from non-Python encoders may write whole floats as ints
        return float(data)

    if cls is type(None):
        if data is not None:
            raise TypeError(f"Expected null, got {type(data).__name__}")
        return None

    handler = _get_handler(cls)
    if handler is not None:
        return handler.deserialize(data, cls)

    if isinstance(data, dict):
        if is_generic_type(cls):
            origin_type = get_origin(cls)
            if issubclass(origin_type, dict):
                key_cls, value_cls = get_args(cls)
                data = _deserialize_keys(data, key_cls)
                data = {k: deserialize_json(v, value_cls) for k, v in data.items()}

        elif issubclass(cls, Jsonable):
            return JsonableSerializingHandler.deserialize(data, cls)

        elif is_dataclass(cls):
            return DataclassJsonSerializingHandler.deserialize(data, cls)

        return data

    if is_json_primitive(data):
        return data

    raise TypeError(f"Unsupported type for deserialization: {cls}")


def deserialize_multi_cls_from_json(data: JSON, cls_list: Sequence[type[T]]) -> T:
    for cls in cls_list:
        try:
            return deserialize_json(data, cls)
        except TypeError:
            continue

    raise TypeError(f"Unsupported types for deserialization: {cls_list}")

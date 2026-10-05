import base64
import datetime
import json
import typing
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
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


def convert_tuple_key(t: tuple) -> str:
    serialized_tuple = serialize_json(t)
    return serialized_tuple


def _restore_tuple_key(key: str, cls: Union[Type[Tuple], GenericAlias]) -> tuple:
    deserialized_obj = json.loads(key)
    deserialized_key = deserialize_json(deserialized_obj, cls)
    return deserialized_key


def _convert_key(key: Any) -> str:
    if isinstance(key, Enum):
        return EnumJsonSerializingHandler.serialize(key)

    if isinstance(key, tuple):
        return convert_tuple_key(key)

    if not isinstance(key, (str, int, float, bool)):
        raise ValueError(f"Couldn't convert key of the given type to str: {type(key)} ({key=})")

    return key


def _reconstruct_key(key: str, cls: Type[T]) -> T:
    if issubclass(cls, Enum):
        return EnumJsonSerializingHandler.deserialize(key, cls)

    cls_origin = cast(type, get_origin(cls))
    if cls_origin is None:
        cls_origin = cls

    if issubclass(cls_origin, tuple):
        return _restore_tuple_key(key, cls)

    if not issubclass(cls_origin, str):
        raise ValueError(f"Couldn't de-convert key to its original type: '{key}' -> {cls}")

    return key


def _serialize_keys(obj: Any) -> Any:
    if isinstance(obj, list):
        return [_serialize_keys(item) for item in obj]
    if not isinstance(obj, dict):
        return obj

    return {_convert_key(k): _serialize_keys(v) for k, v in obj.items()}


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


class MappingJsonSerializingHandler(JsonSerializingHandler):
    """Handles MappingProxyType (and Mapping annotations): serialized as a plain JSON object."""

    @staticmethod
    def serialize(obj: Mapping) -> JSON:
        return _serialize_keys(dict(obj))

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
        # Nested values are converted by the encoder.
        data_dict = {field.name: getattr(obj, field.name) for field in fields(obj)}
        return _serialize_keys(data_dict)

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
    """Convert datetime (or date) object to ISO 8601 string and back to datetime (or date) object."""

    @staticmethod
    def serialize(obj: datetime.datetime) -> JSON:
        return obj.isoformat()

    @staticmethod
    def deserialize(data: JSON, cls: type[datetime.datetime]) -> T:
        return cls.fromisoformat(data)


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
    UnionType: UnionJsonSerializingHandler,
    typing.Union: UnionJsonSerializingHandler,
    bytes: BytesJsonSerializingHandler,
    datetime.datetime: DatetimeJsonSerializingHandler,
    datetime.date: DatetimeJsonSerializingHandler,
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

    from jserpy._numpy import get_numpy_kind

    numpy_kind = get_numpy_kind(cls_origin)
    if numpy_kind == "array":
        return NumpyJsonSerializingHandler
    if numpy_kind == "scalar":
        return NumpyTypeJsonSerializingHandler


class CustomEncoder(json.JSONEncoder):

    def default(self, obj: Any):
        handler = _get_handler(type(obj))
        if handler is not None:
            return handler.serialize(obj)

        return super().default(obj)


def serialize_json(obj: Any) -> str:
    converted_obj = _serialize_keys(obj)
    return json.dumps(converted_obj, cls=CustomEncoder)


def serialize_json_as_dict(obj: Any) -> dict[str, JSON]:
    serialized = serialize_json(obj)
    return json.loads(serialized)


def deserialize_json(data: JSON, cls: type[T]) -> T:
    if cls is Any:
        return data

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

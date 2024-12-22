from dataclasses import Field
from typing import Protocol, ClassVar, Any


_PRIMITIVES = (bool, str, int, float, type(None))


def is_primitive(value: Any, inheritance: bool = False) -> bool:
    if inheritance:
        return isinstance(value, _PRIMITIVES)

    return type(value) in _PRIMITIVES


class Dataclass(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

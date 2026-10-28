"""User-defined objects (``Jsonable`` and ``fallback=``) and the encoding options, on objects from trivial to complex.

Three parts:

* ``Jsonable`` implementations: every JSON kind ``to_json`` can return, results that need further conversion, positions
  (field, list, dict value, Optional/Union, set, tuple, mapping key), nesting, hierarchies, subclasses of the builtin
  containers, generics, and the failure modes (``to_json`` returning itself, never-ending ``to_json``, ``from_json``
  receiving unexpected JSON).
* ``fallback=``: converting instances of classes without a handler, recursively, at any position, with the precedence
  rules and the failure modes.
* The encoding options on one complex dataclass tree: goldens on a small tree, and a full option matrix checked against
  a strong oracle (the options change the text, never the data).

Tests named ``*_limitation`` document current behavior that is not a promise. Tests marked ``xfail(strict=True)`` pin
what a user would expect and the library does not do yet: the ``to_json`` of a ``Jsonable`` that subclasses ``str``,
``int`` or ``float`` is ignored.
"""

import itertools
import json
from collections import UserDict
from dataclasses import dataclass
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from functools import singledispatch
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union

import pytest

from jserpy import (
    deserialize_json,
    serialize_json,
    serialize_json_as_dict,
    serialize_json_as_obj,
)
from jserpy.jsonable import Jsonable

from objects_support import assert_round_trip, assert_strictly_equal, to_json_value


# ---------------------------------------------------------------------------------------------------------------------
# Shared enums and dataclasses
# ---------------------------------------------------------------------------------------------------------------------


class Label(Enum):
    ALPHA = "alpha"
    BETA = "beta"


class Priority(Enum):
    LOW = 1
    HIGH = 2


class Status(Enum):
    PENDING = "pending"
    PAID = "paid"


@dataclass
class Customer:
    name: str
    vip: bool = False


@dataclass
class Holder:
    item: Any
    items: list[Any]


@dataclass(frozen=True)
class Inner:
    when: datetime
    kind: Label


# ---------------------------------------------------------------------------------------------------------------------
# Jsonable implementations
# ---------------------------------------------------------------------------------------------------------------------


def _same(left: Any, right: Any) -> bool:
    try:
        assert_strictly_equal(left, right)
    except AssertionError:
        return False
    return True


def _expect(data: Any, kind: type) -> None:
    # What a careful `from_json` does, so that a union moves on to its next member
    if not isinstance(data, kind):
        raise TypeError(f"expected a JSON value of type {kind.__name__}")


def assert_same_json(actual: Any, expected: Any) -> None:
    # `==` cannot tell 1 from 1.0 or True, nor the order of the keys of an object; the text of both can
    assert json.dumps(actual) == json.dumps(expected)


class Value(Jsonable):
    """Base of the Jsonable test types: equal when the type and the state (compared strictly) are equal."""

    def _state(self) -> Any:
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        return type(other) is type(self) and _same(self._state(), other._state())

    def __hash__(self) -> int:
        return hash((type(self).__qualname__, repr(self._state())))

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self._state()!r}"


class Point(Value):
    """To a JSON object."""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def _state(self):
        return (self.x, self.y)

    def to_json(self):
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_json(cls, data):
        _expect(data, dict)
        return cls(data["x"], data["y"])


class Coordinate(Value):
    """To a JSON array."""

    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon

    def _state(self):
        return (self.lat, self.lon)

    def to_json(self):
        return [self.lat, self.lon]

    @classmethod
    def from_json(cls, data):
        _expect(data, list)
        lat, lon = data
        return cls(lat, lon)


class Scalar(Value):
    """To a JSON primitive. `from_json` accepts any JSON value, like the simplest `cls(data)` implementation."""

    def __init__(self, value: Any):
        self.value = value

    def _state(self):
        return (self.value,)

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, data):
        return cls(data)


class Meters(Scalar):
    pass


class Slug(Scalar):
    pass


class Count(Scalar):
    pass


class Enabled(Scalar):
    pass


class Unset(Scalar):
    pass


class Booking(Value):
    """`to_json` returns values that still need conversion; `from_json` delegates to `deserialize_json`."""

    def __init__(self, when, status, price, customer, receipt, labels):
        self.when = when
        self.status = status
        self.price = price
        self.customer = customer
        self.receipt = receipt
        self.labels = labels

    def _state(self):
        return (self.when, self.status, self.price, self.customer, self.receipt, self.labels)

    def to_json(self):
        return {
            "when": self.when,
            "status": self.status,
            "price": self.price,
            "customer": self.customer,
            "receipt": self.receipt,
            "labels": self.labels,
        }

    @classmethod
    def from_json(cls, data):
        return cls(
            deserialize_json(data["when"], datetime),
            deserialize_json(data["status"], Status),
            deserialize_json(data["price"], Decimal),
            deserialize_json(data["customer"], Customer),
            deserialize_json(data["receipt"], bytes),
            deserialize_json(data["labels"], frozenset[Label]),
        )


class Ledger(Value):
    """`to_json` returns a structure with Enum keys and tuple keys."""

    def __init__(self, by_label: dict, by_cell: dict):
        self.by_label = by_label
        self.by_cell = by_cell

    def _state(self):
        return (self.by_label, self.by_cell)

    def to_json(self):
        return {"by_label": self.by_label, "by_cell": self.by_cell}

    @classmethod
    def from_json(cls, data):
        return cls(
            deserialize_json(data["by_label"], dict[Label, int]),
            deserialize_json(data["by_cell"], dict[tuple[int, int], str]),
        )


class Place(Value):
    """Level 3 of `Atlas`: holds a `Point`, restored by calling `Point.from_json` directly."""

    def __init__(self, name: str, point: Point):
        self.name = name
        self.point = point

    def _state(self):
        return (self.name, self.point)

    def to_json(self):
        return {"name": self.name, "point": self.point}

    @classmethod
    def from_json(cls, data):
        return cls(data["name"], Point.from_json(data["point"]))


class Trip(Value):
    """Level 2 of `Atlas`: a list of `Place`, restored through `deserialize_json`."""

    def __init__(self, title: str, places: list):
        self.title = title
        self.places = places

    def _state(self):
        return (self.title, self.places)

    def to_json(self):
        return {"title": self.title, "places": self.places}

    @classmethod
    def from_json(cls, data):
        return cls(data["title"], deserialize_json(data["places"], list[Place]))


class Atlas(Value):
    """Level 1: Atlas -> Trip -> Place -> Point."""

    def __init__(self, trips: dict):
        self.trips = trips

    def _state(self):
        return (self.trips,)

    def to_json(self):
        return {"trips": self.trips}

    @classmethod
    def from_json(cls, data):
        return cls(deserialize_json(data["trips"], dict[str, Trip]))


class Tree(Value):
    """A recursive Jsonable of any depth and breadth."""

    def __init__(self, label: str, children: list = ()):
        self.label = label
        self.children = list(children)

    def _state(self):
        return (self.label, self.children)

    def to_json(self):
        return {"label": self.label, "children": self.children}

    @classmethod
    def from_json(cls, data):
        return cls(data["label"], [deserialize_json(child, Tree) for child in data["children"]])


class Shape(Value):
    """Root of a hierarchy: `to_json` and `from_json` are inherited, `from_json` dispatches on the "kind" entry."""

    kind = "shape"

    def _fields(self) -> dict:
        raise NotImplementedError

    def _state(self):
        return (self.kind, self._fields())

    def to_json(self):
        return {"kind": self.kind, **self._fields()}

    @classmethod
    def from_json(cls, data):
        target = SHAPES[data["kind"]] if cls is Shape else cls
        return target._from_fields(data)


class Circle(Shape):
    kind = "circle"

    def __init__(self, radius: float):
        self.radius = radius

    def _fields(self):
        return {"radius": self.radius}

    @classmethod
    def _from_fields(cls, data):
        return cls(data["radius"])


class Rect(Shape):
    kind = "rect"

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def _fields(self):
        return {"width": self.width, "height": self.height}

    @classmethod
    def _from_fields(cls, data):
        return cls(data["width"], data["height"])


class Square(Rect):
    """A subclass of a subclass: inherits `_fields` (and so the JSON shape) but builds itself from one side."""

    kind = "square"

    def __init__(self, side: int):
        super().__init__(side, side)

    @classmethod
    def _from_fields(cls, data):
        return cls(data["width"])


class LabeledCircle(Circle):
    """Extends the JSON of its parent through `super()`."""

    kind = "labeled-circle"

    def __init__(self, radius: float, label: str):
        super().__init__(radius)
        self.label = label

    def _fields(self):
        return {**super()._fields(), "label": self.label}

    @classmethod
    def _from_fields(cls, data):
        return cls(data["radius"], data["label"])


SHAPES = {shape.kind: shape for shape in (Circle, Rect, Square, LabeledCircle)}


@dataclass(frozen=True)
class Temperature(Jsonable):
    """Both a dataclass and a Jsonable: the Jsonable handler is consulted first, so the JSON is a string."""

    degrees: float
    unit: str = "C"

    def to_json(self):
        return f"{self.degrees}{self.unit}"

    @classmethod
    def from_json(cls, data):
        _expect(data, str)
        return cls(float(data[:-1]), data[-1])


@dataclass
class Reading2:
    temperature: Temperature
    history: list[Temperature]


@dataclass
class Route:
    start: Point
    stops: list[Point]
    named: dict[str, Point]
    visited: frozenset[Point]
    legs: tuple[Point, Point]
    path: tuple[Coordinate, ...]
    end: Optional[Point] = None


@dataclass
class Waypoint:
    at: Point | Coordinate | None


@dataclass
class Tagged:
    label: str
    slug: Optional[Slug] = None


class Carrier(Value):
    """Carries any value inside the JSON that `to_json` returns."""

    def __init__(self, value: Any):
        self.value = value

    def _state(self):
        return (self.value,)

    def to_json(self):
        return {"carried": [self.value]}

    @classmethod
    def from_json(cls, data):
        return cls(data["carried"][0])


class Recorder(Jsonable):
    """Records the JSON values `from_json` receives."""

    received: list = []

    def to_json(self):
        return None

    @classmethod
    def from_json(cls, data):
        cls.received.append(data)
        return cls()


@pytest.fixture
def received():
    Recorder.received.clear()
    yield Recorder.received
    Recorder.received.clear()


class Linked(Jsonable):
    """To {"peer": <peer>}: wiring the peers up makes cycles that only exist through `to_json`."""

    def __init__(self, peer: Optional["Linked"] = None):
        self.peer = peer

    def to_json(self):
        return {"peer": self.peer}

    @classmethod
    def from_json(cls, data):
        return cls()

    def __repr__(self):
        return "LINKED_PAYLOAD_MUST_NOT_LEAK"


class SelfReturning(Jsonable):
    def to_json(self):
        return self

    @classmethod
    def from_json(cls, data):
        return cls()

    def __repr__(self):
        return "SELF_RETURNING_PAYLOAD_MUST_NOT_LEAK"


class SetWithSelf(Jsonable):
    """`to_json` returns a set holding the instance itself: the cycle runs through a set."""

    def to_json(self):
        return {self}

    @classmethod
    def from_json(cls, data):
        return cls()

    def __repr__(self):
        return "SET_PAYLOAD_MUST_NOT_LEAK"


class Endless(Jsonable):
    """`to_json` returns a new instance each time: a new id, so never a cycle, only the depth limit ends it."""

    def to_json(self):
        return Endless()

    @classmethod
    def from_json(cls, data):
        return cls()


class EndlessInList(Jsonable):
    def to_json(self):
        return [EndlessInList()]

    @classmethod
    def from_json(cls, data):
        return cls()


class ConversionError(Exception):
    pass


class ExplodingJsonable(Jsonable):
    def to_json(self):
        raise ConversionError("to_json failed")

    @classmethod
    def from_json(cls, data):
        return cls()


class StrictSlug(Value):
    """A `from_json` that validates its input the usual way: ValueError."""

    def __init__(self, text: str):
        self.text = text

    def _state(self):
        return (self.text,)

    def to_json(self):
        return self.text

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, str) or not data.isidentifier():
            raise ValueError("not a slug")
        return cls(data)


class LenientPoint(Jsonable):
    """A `from_json` written the usual way with `.get`, which fails on anything but a dict."""

    def __init__(self, x):
        self.x = x

    def to_json(self):
        return {"x": self.x}

    @classmethod
    def from_json(cls, data):
        return cls(data.get("x"))

    def __repr__(self):
        return f"LenientPoint({self.x!r})"


class NamedStr(str, Jsonable):
    def to_json(self):
        return {"value": str(self)}

    @classmethod
    def from_json(cls, data):
        return cls(data["value"])


class NamedInt(int, Jsonable):
    def to_json(self):
        return {"value": int(self)}

    @classmethod
    def from_json(cls, data):
        return cls(data["value"])


class NamedFloat(float, Jsonable):
    def to_json(self):
        return {"value": float(self)}

    @classmethod
    def from_json(cls, data):
        return cls(data["value"])


class Transparent(Value):
    """`to_json` returns the wrapped value itself, to be converted by JSerPy; `from_json` restores it as `inner_type`."""

    inner_type: Any = None

    def __init__(self, inner: Any):
        self.inner = inner

    def _state(self):
        return (self.inner,)

    def to_json(self):
        return self.inner

    @classmethod
    def from_json(cls, data):
        return cls(deserialize_json(data, cls.inner_type))


def wrapper_of(inner_type: Any) -> type:
    return type("Wrapper", (Transparent,), {"inner_type": inner_type})


class JsonableDict(dict, Jsonable):
    def to_json(self):
        return {"wrapped": dict(self)}

    @classmethod
    def from_json(cls, data):
        return cls(data["wrapped"])


class JsonableList(list, Jsonable):
    def to_json(self):
        return {"wrapped": list(self)}

    @classmethod
    def from_json(cls, data):
        return cls(data["wrapped"])


class JsonableTuple(tuple, Jsonable):
    def to_json(self):
        return {"wrapped": list(self)}

    @classmethod
    def from_json(cls, data):
        return cls(data["wrapped"])


class JsonableSet(set, Jsonable):
    def to_json(self):
        return {"wrapped": sorted(self)}

    @classmethod
    def from_json(cls, data):
        return cls(data["wrapped"])


class JsonableFrozenset(frozenset, Jsonable):
    def to_json(self):
        return {"wrapped": sorted(self)}

    @classmethod
    def from_json(cls, data):
        return cls(data["wrapped"])


class JsonableBytes(bytes, Jsonable):
    def to_json(self):
        return {"wrapped": list(self)}

    @classmethod
    def from_json(cls, data):
        return cls(data["wrapped"])


class JsonablePath(PurePosixPath, Jsonable):
    def to_json(self):
        return {"wrapped": str(self)}

    @classmethod
    def from_json(cls, data):
        return cls(data["wrapped"])


T = TypeVar("T")


class Box(Value, Generic[T]):
    """A generic Jsonable: the annotation `Box[int]` reaches `Box.from_json`, which does not look at the argument."""

    def __init__(self, content: Any):
        self.content = content

    def _state(self):
        return (self.content,)

    def to_json(self):
        return {"content": self.content}

    @classmethod
    def from_json(cls, data):
        return cls(data["content"])


class Journaled(Jsonable):
    """Appends its name to the `journal` it was given whenever `to_json` runs. Hashable, to sit in sets."""

    def __init__(self, name: str, journal: list):
        self.name = name
        self.journal = journal

    def to_json(self):
        self.journal.append(self.name)
        return {"name": self.name}

    @classmethod
    def from_json(cls, data):
        return cls(data["name"], [])

    def __eq__(self, other):
        return type(other) is Journaled and other.name == self.name

    def __hash__(self):
        return hash(self.name)


class Greeting(Enum):
    HELLO = "שלום"
    WAVE = "👋"


class Opaque:
    """A class JSerPy has no handler for. The repr carries no memory address."""

    def __init__(self, tag: Any):
        self.tag = tag

    def __repr__(self):
        return f"Opaque({self.tag!r})"


class OpaqueCarrier(Value):
    """A Jsonable whose JSON contains an unsupported value."""

    def __init__(self, tag):
        self.tag = tag

    def _state(self):
        return (self.tag,)

    def to_json(self):
        return {"carried": Opaque(self.tag)}

    @classmethod
    def from_json(cls, data):
        return cls(data["carried"]["opaque"])


# ---------------------------------------------------------------------------------------------------------------------
# Jsonable: every JSON kind, at the root and nested
# ---------------------------------------------------------------------------------------------------------------------

JSONABLE_SHAPES = [
    pytest.param(Point(1, -2), Point, '{"x": 1, "y": -2}', id="object"),
    pytest.param(Coordinate(32.5, 34.75), Coordinate, "[32.5, 34.75]", id="array"),
    pytest.param(Slug("a b"), Slug, '"a b"', id="string"),
    pytest.param(Count(0), Count, "0", id="int-zero"),
    pytest.param(Meters(1.0), Meters, "1.0", id="float"),
    pytest.param(Enabled(False), Enabled, "false", id="bool"),
    pytest.param(Unset(None), Unset, "null", id="null"),
]


@pytest.mark.parametrize(("value", "cls", "text"), JSONABLE_SHAPES)
def test_jsonable_serializes_to_what_to_json_returns_as_root(value, cls, text):
    assert serialize_json(value) == text
    assert_same_json(serialize_json_as_obj(value), json.loads(text))
    assert_same_json(serialize_json_as_dict(value), json.loads(text))
    assert_round_trip(value, cls)


@pytest.mark.parametrize(("value", "cls", "text"), JSONABLE_SHAPES)
def test_jsonable_nested_in_containers(value, cls, text):
    assert serialize_json([value, value]) == f"[{text}, {text}]"
    assert_round_trip([value, value], list[cls])

    assert serialize_json({"k": value}) == f'{{"k": {text}}}'
    assert_round_trip({"k": value}, dict[str, cls])

    assert serialize_json((value, value)) == f"[{text}, {text}]"
    assert_round_trip((value, value), tuple[cls, cls])

    assert serialize_json({"deep": [{"k": (value,)}]}) == f'{{"deep": [{{"k": [{text}]}}]}}'
    assert_round_trip({"deep": [{"k": (value,)}]}, dict[str, list[dict[str, tuple[cls]]]])


def test_jsonable_returning_a_structure_is_converted_recursively():
    booking = Booking(
        datetime(2024, 3, 1, 9, 30),
        Status.PAID,
        Decimal("19.90"),
        Customer("Dana", True),
        b"JSerPy",
        frozenset({Label.BETA, Label.ALPHA}),
    )

    assert serialize_json(booking) == (
        '{"when": "2024-03-01T09:30:00", "status": "paid", "price": "19.90", '
        '"customer": {"name": "Dana", "vip": true}, "receipt": "SlNlclB5", "labels": ["alpha", "beta"]}'
    )
    assert_round_trip(booking, Booking)
    assert_round_trip([booking], list[Booking])


def test_jsonable_returning_enum_keys_and_tuple_keys():
    ledger = Ledger({Label.BETA: 2, Label.ALPHA: 1}, {(2, 3): "y", (0, 1): "x"})

    # json.dumps of both: unlike ==, it tells the order of the keys
    assert json.dumps(to_json_value(ledger)) == json.dumps(
        {"by_label": {"beta": 2, "alpha": 1}, "by_cell": {"[2, 3]": "y", "[0, 1]": "x"}}
    )
    assert_round_trip(ledger, Ledger)


wrap_int_tuple = wrapper_of(tuple[int, ...])
wrap_point = wrapper_of(Point)

TRANSPARENT_JSONABLES = [
    pytest.param(wrapper_of(Priority)(Priority.HIGH), "2", id="enum"),
    pytest.param(wrapper_of(datetime)(datetime(2024, 1, 2, 3, 4, 5)), '"2024-01-02T03:04:05"', id="datetime"),
    pytest.param(wrapper_of(date)(date(2024, 1, 2)), '"2024-01-02"', id="date"),
    pytest.param(wrapper_of(time)(time(1, 2, 3)), '"01:02:03"', id="time"),
    pytest.param(wrapper_of(Decimal)(Decimal("1.50")), '"1.50"', id="decimal-keeps-trailing-zero"),
    pytest.param(wrapper_of(bytes)(b"ab"), '"YWI="', id="bytes"),
    pytest.param(wrapper_of(frozenset[int])(frozenset({3, 1, 2})), "[1, 2, 3]", id="frozenset"),
    pytest.param(wrapper_of(set[int])({3, 1, 2}), "[1, 2, 3]", id="set"),
    pytest.param(wrap_int_tuple((3, 1, 2)), "[3, 1, 2]", id="tuple"),
    pytest.param(wrapper_of(Customer)(Customer("Dana", True)), '{"name": "Dana", "vip": true}', id="dataclass"),
    pytest.param(wrapper_of(MappingProxyType)(MappingProxyType({"k": 1})), '{"k": 1}', id="mappingproxy"),
    pytest.param(wrap_point(Point(1, 2)), '{"x": 1, "y": 2}', id="another-jsonable"),
    pytest.param(
        wrapper_of(wrap_point)(wrap_point(Point(1, 2))), '{"x": 1, "y": 2}', id="jsonable-in-jsonable-in-jsonable"
    ),
]


@pytest.mark.parametrize(("value", "text"), TRANSPARENT_JSONABLES)
def test_jsonable_returning_a_supported_value_serializes_like_that_value(value, text):
    cls = type(value)

    assert serialize_json(value) == text
    assert serialize_json([value, {"k": value}]) == f'[{text}, {{"k": {text}}}]'
    assert_round_trip(value, cls)
    assert_round_trip({"k": [value]}, dict[str, list[cls]])


def test_jsonable_returning_a_path_serializes_it_as_a_string():
    # Restoring is left out: the Path handler builds a platform path, not the PurePosixPath that went in
    assert serialize_json(wrapper_of(PurePosixPath)(PurePosixPath("docs/ספר.txt")), ensure_ascii=False) == (
        '"docs/ספר.txt"'
    )


CONTAINER_SUBCLASSES = [
    pytest.param(JsonableDict(a=1, b=2), '{"wrapped": {"a": 1, "b": 2}}', id="dict"),
    pytest.param(JsonableDict(), '{"wrapped": {}}', id="empty-dict"),
    pytest.param(JsonableList([3, 1, 2]), '{"wrapped": [3, 1, 2]}', id="list"),
    pytest.param(JsonableList(), '{"wrapped": []}', id="empty-list"),
    pytest.param(JsonableTuple((3, 1, 2)), '{"wrapped": [3, 1, 2]}', id="tuple"),
    pytest.param(JsonableTuple(), '{"wrapped": []}', id="empty-tuple"),
    pytest.param(JsonableSet({3, 1, 2}), '{"wrapped": [1, 2, 3]}', id="set"),
    pytest.param(JsonableFrozenset({3, 1, 2}), '{"wrapped": [1, 2, 3]}', id="frozenset"),
    pytest.param(JsonableBytes(b"ab"), '{"wrapped": [97, 98]}', id="bytes"),
    pytest.param(JsonablePath("docs/a.txt"), '{"wrapped": "docs/a.txt"}', id="path"),
]


@pytest.mark.parametrize(("value", "text"), CONTAINER_SUBCLASSES)
def test_jsonable_that_subclasses_a_builtin_container_is_written_by_to_json(value, text):
    cls = type(value)

    # The Jsonable handler is consulted before the ones for dict, list, tuple, set, bytes and paths
    assert serialize_json(value) == text
    assert serialize_json([value]) == f"[{text}]"
    assert serialize_json({"k": value}) == f'{{"k": {text}}}'
    assert_round_trip(value, cls)
    assert_round_trip([value, value], list[cls])


@pytest.mark.parametrize(
    ("cls", "value", "text"),
    [
        pytest.param(NamedStr, "x", '{"value": "x"}', id="str-subclass"),
        pytest.param(NamedInt, 3, '{"value": 3}', id="int-subclass"),
        pytest.param(NamedFloat, 3.5, '{"value": 3.5}', id="float-subclass"),
    ],
)
def test_from_json_of_a_jsonable_subclassing_a_primitive_is_used(cls, value, text):
    # The other direction of test_to_json_of_a_jsonable_subclassing_a_primitive_is_used, which is a known defect
    restored = deserialize_json(json.loads(text), cls)

    assert_strictly_equal(restored, cls(value))
    assert_strictly_equal(deserialize_json(json.loads(f"[{text}]"), list[cls]), [cls(value)])


def test_generic_jsonable_is_restored_through_its_parameterized_annotation():
    assert serialize_json(Box(5)) == '{"content": 5}'
    assert_round_trip(Box(5), Box[int])
    assert_round_trip([Box("a"), Box("b")], list[Box[str]])
    assert_round_trip({"k": Box(1.5)}, dict[str, Box[float]])
    assert_round_trip((Box(1), None), tuple[Box[int], Optional[Box[int]]])
    assert_round_trip(None, Optional[Box[int]])


@pytest.mark.parametrize("serializer", [serialize_json, serialize_json_as_obj], ids=["text", "intermediate"])
def test_to_json_runs_once_per_occurrence_in_document_order(serializer):
    journal = []
    a, b, c, d, e = (Journaled(name, journal) for name in "abcde")

    serializer({"x": [a, {"y": b}], "z": Holder(c, [d]), "w": (e,)})

    assert journal == list("abcde")


def test_to_json_runs_once_per_set_member_tuple_key_member_and_repeated_instance():
    journal = []
    a, b, c = Journaled("a", journal), Journaled("b", journal), Journaled("c", journal)

    serialize_json({a, b})
    assert sorted(journal) == ["a", "b"]  # the iteration order of a set is not defined

    journal.clear()
    serialize_json({(c, 1): 0})
    assert journal == ["c"]

    journal.clear()
    serialize_json([a, a])
    assert journal == ["a", "a"]


# ---------------------------------------------------------------------------------------------------------------------
# Jsonable: positions in other objects
# ---------------------------------------------------------------------------------------------------------------------


def make_route(**changes):
    values = dict(
        start=Point(0, 0),
        stops=[Point(2, 1), Point(1, 5), Point(10, 0)],
        named={"home": Point(0, 0), "work": Point(3, 4)},
        visited=frozenset({Point(2, 1), Point(10, 0), Point(1, 5)}),
        legs=(Point(0, 0), Point(1, 5)),
        path=(Coordinate(1.5, 2.5), Coordinate(3.0, -4.0)),
        end=Point(9, 9),
    )
    values.update(changes)
    return Route(**values)


def test_jsonable_in_dataclass_fields_lists_dicts_sets_and_tuples():
    route = make_route()
    # json.dumps of both: unlike ==, it tells 3.0 from 3 and the order of the keys
    assert json.dumps(to_json_value(route)) == json.dumps(
        {
            "start": {"x": 0, "y": 0},
            "stops": [{"x": 2, "y": 1}, {"x": 1, "y": 5}, {"x": 10, "y": 0}],
            "named": {"home": {"x": 0, "y": 0}, "work": {"x": 3, "y": 4}},
            # the set is ordered by the text of its members: ...,"y":5} < ...:10,... < ...:2,...
            "visited": [{"x": 1, "y": 5}, {"x": 10, "y": 0}, {"x": 2, "y": 1}],
            "legs": [{"x": 0, "y": 0}, {"x": 1, "y": 5}],
            "path": [[1.5, 2.5], [3.0, -4.0]],
            "end": {"x": 9, "y": 9},
        }
    )
    assert_round_trip(route, Route)
    assert_round_trip(make_route(end=None, stops=[], named={}, visited=frozenset()), Route)


def test_optional_jsonable_field_missing_from_the_data_is_none():
    data = to_json_value(make_route())
    del data["end"]

    restored = deserialize_json(data, Route)

    assert restored.end is None
    assert_strictly_equal(restored, make_route(end=None))


def test_set_of_jsonables_is_ordered_by_text_independent_of_insertion_order():
    points = [Point(2, 1), Point(1, 5), Point(10, 0)]
    expected = '[{"x": 1, "y": 5}, {"x": 10, "y": 0}, {"x": 2, "y": 1}]'

    assert serialize_json(frozenset(points)) == expected
    assert serialize_json(set(reversed(points))) == expected
    assert_round_trip(set(points), set[Point])
    assert_round_trip(frozenset(points), frozenset[Point])


@pytest.mark.parametrize(
    ("members", "cls", "text"),
    [
        pytest.param(
            [Coordinate(2.0, 1.0), Coordinate(10.5, 0.0), Coordinate(1.5, 9.0)],
            Coordinate,
            "[[1.5, 9.0], [10.5, 0.0], [2.0, 1.0]]",
            id="arrays",
        ),
        pytest.param([Slug("b"), Slug("ab"), Slug("a")], Slug, '["a", "ab", "b"]', id="strings"),
        pytest.param([Count(2), Count(10), Count(1)], Count, "[1, 10, 2]", id="ints"),
        pytest.param([Enabled(True), Enabled(False)], Enabled, "[false, true]", id="bools"),
        pytest.param([Unset(None)], Unset, "[null]", id="null"),
    ],
)
def test_sets_of_jsonables_of_every_json_kind_are_ordered_by_the_text_of_their_members(members, cls, text):
    # "[1.5" < "[10.5" < "[2.0", "a" < "ab" < "b", "1" < "10" < "2", "false" < "true"
    for ordering in (members, members[::-1]):
        assert serialize_json(set(ordering)) == text
        assert serialize_json(frozenset(ordering)) == text
    assert_round_trip(set(members), set[cls])
    assert_round_trip(frozenset(members), frozenset[cls])


@pytest.mark.parametrize(
    ("at", "expected"),
    [
        pytest.param(Point(1, 2), {"at": {"x": 1, "y": 2}}, id="first-member"),
        pytest.param(Coordinate(1.5, 2.5), {"at": [1.5, 2.5]}, id="second-member"),
        pytest.param(None, {"at": None}, id="none-member"),
    ],
)
def test_jsonable_in_a_union_field_is_picked_by_the_member_that_accepts_the_data(at, expected):
    waypoint = Waypoint(at)

    assert to_json_value(waypoint) == expected
    assert_round_trip(waypoint, Waypoint)


def test_jsonable_in_optional_list_items_and_dict_values():
    points = [Point(1, 2), None, Point(3, 4)]
    assert serialize_json(points) == '[{"x": 1, "y": 2}, null, {"x": 3, "y": 4}]'
    assert_round_trip(points, list[Optional[Point]])
    assert_round_trip({"a": None, "b": Point(0, 0)}, dict[str, Optional[Point]])
    assert_round_trip(None, Optional[Point])
    assert_round_trip(Point(5, 6), Optional[Point])


def test_union_of_jsonables_and_a_primitive_at_the_root():
    options = list[Union[Point, Coordinate, str]]
    values = [Point(1, 2), Coordinate(1.5, 2.5), "plain"]

    assert serialize_json(values) == '[{"x": 1, "y": 2}, [1.5, 2.5], "plain"]'
    assert_round_trip(values, options)


def test_jsonable_inside_a_tuple_mapping_key_is_serialized_as_json_text():
    assert serialize_json({(Point(1, 2), "a"): 3}) == '{"[{\\"x\\": 1, \\"y\\": 2}, \\"a\\"]": 3}'


@pytest.mark.parametrize(
    ("value", "cls"),
    [
        pytest.param(
            {(Point(1, 2), "a"): 3, (Point(0, 0), "b"): 4}, dict[tuple[Point, str], int], id="jsonable-member"
        ),
        pytest.param({(1, 2): "x", (3, 4): "y"}, dict[tuple[int, int], str], id="ints"),
        pytest.param({(1, 2): "x", (3, 4): "y"}, Dict[Tuple[int, int], str], id="typing-spelling"),
    ],
)
def test_parameterized_tuple_keys_round_trip(value, cls):
    assert_round_trip(value, cls)


def test_same_jsonable_instance_twice_is_not_a_cycle():
    point = Point(1, 2)
    value = {"a": point, "b": [point, (point,)]}

    assert serialize_json(value) == (
        '{"a": {"x": 1, "y": 2}, "b": [{"x": 1, "y": 2}, [{"x": 1, "y": 2}]]}'
    )
    shared = Linked()
    assert serialize_json([shared, shared]) == '[{"peer": null}, {"peer": null}]'


def canonical_text(point: Point) -> str:
    return json.dumps(point.to_json(), sort_keys=True, separators=(",", ":"))


def test_large_collections_of_jsonables():
    points = [Point(x, y) for x in range(-5, 40, 3) for y in range(0, 60, 4)]  # 15 x 15 distinct points

    assert_round_trip(points, list[Point])
    assert_round_trip(tuple(points), tuple[Point, ...])
    assert_round_trip({f"p{index}": point for index, point in enumerate(points)}, dict[str, Point])

    # A set is written in the order of the text of its members, which is not the numeric order of the coordinates
    by_text = sorted(points, key=canonical_text)
    assert by_text != sorted(points, key=lambda point: (point.x, point.y))
    assert serialize_json(set(points)) == serialize_json(by_text)
    assert serialize_json(frozenset(reversed(points))) == serialize_json(by_text)
    restored = assert_round_trip(set(points), set[Point])
    assert len(restored) == len(points) == 225


# ---------------------------------------------------------------------------------------------------------------------
# Jsonable: nesting, hierarchies, and who wins against a dataclass
# ---------------------------------------------------------------------------------------------------------------------


def make_atlas():
    return Atlas(
        {
            "short": Trip("Walk", [Place("Gate", Point(0, 0))]),
            "long": Trip("Tour", [Place("A", Point(1, 2)), Place("B", Point(3, 4))]),
            "empty": Trip("Nothing", []),
        }
    )


def test_jsonable_nested_three_levels_deep():
    atlas = make_atlas()

    assert serialize_json(atlas) == (
        '{"trips": {"short": {"title": "Walk", "places": [{"name": "Gate", "point": {"x": 0, "y": 0}}]}, '
        '"long": {"title": "Tour", "places": [{"name": "A", "point": {"x": 1, "y": 2}}, '
        '{"name": "B", "point": {"x": 3, "y": 4}}]}, "empty": {"title": "Nothing", "places": []}}}'
    )
    assert_round_trip(atlas, Atlas)
    assert_round_trip([atlas, atlas], list[Atlas])


def test_recursive_jsonable_small_tree():
    tree = Tree("root", [Tree("a", [Tree("a1")]), Tree("b")])

    assert serialize_json(tree) == (
        '{"label": "root", "children": [{"label": "a", "children": [{"label": "a1", "children": []}]}, '
        '{"label": "b", "children": []}]}'
    )
    assert_round_trip(tree, Tree)


def make_complete_tree(depth: int, label: str = "n") -> Tree:
    if depth == 0:
        return Tree(label)
    return Tree(label, [make_complete_tree(depth - 1, label + "0"), make_complete_tree(depth - 1, label + "1")])


def make_chain(depth: int) -> Tree:
    tree = Tree("leaf")
    for level in range(depth):
        tree = Tree(f"level-{level}", [tree])
    return tree


def test_recursive_jsonable_wide_tree():
    tree = make_complete_tree(5)

    def count(node):
        return 1 + sum(count(child) for child in node["children"])

    assert count(to_json_value(tree)) == 63  # 2**6 - 1
    assert_round_trip(tree, Tree)


def test_recursive_jsonable_deep_chain():
    chain = make_chain(40)

    parsed = to_json_value(chain)
    depth = 0
    while parsed["children"]:
        parsed = parsed["children"][0]
        depth += 1
    assert (depth, parsed["label"]) == (40, "leaf")
    assert_round_trip(chain, Tree)


def test_hierarchy_inherits_to_json_and_from_json_and_restores_exact_subclasses():
    shapes = [Circle(1.5), Rect(2, 3), Square(4), LabeledCircle(0.5, "small")]

    assert serialize_json(shapes) == (
        '[{"kind": "circle", "radius": 1.5}, {"kind": "rect", "width": 2, "height": 3}, '
        '{"kind": "square", "width": 4, "height": 4}, '
        '{"kind": "labeled-circle", "radius": 0.5, "label": "small"}]'
    )
    restored = assert_round_trip(shapes, list[Shape])
    assert [type(shape) for shape in restored] == [Circle, Rect, Square, LabeledCircle]


def test_hierarchy_in_other_positions_and_with_a_subclass_as_target():
    assert_round_trip({"a": Square(1), "b": Circle(2.0)}, dict[str, Shape])
    assert_round_trip(Square(5), Optional[Shape])
    assert_round_trip(None, Optional[Shape])

    data = to_json_value(LabeledCircle(2.0, "big"))
    restored = deserialize_json(data, LabeledCircle)
    assert type(restored) is LabeledCircle
    assert restored == LabeledCircle(2.0, "big")


def test_jsonable_that_is_also_a_dataclass_serializes_through_to_json():
    assert serialize_json(Temperature(21.5)) == '"21.5C"'
    assert serialize_json(Temperature(-3.0, "F")) == '"-3.0F"'

    reading = Reading2(Temperature(21.5), [Temperature(1.0), Temperature(-3.0, "F")])
    assert to_json_value(reading) == {"temperature": "21.5C", "history": ["1.0C", "-3.0F"]}


def test_jsonable_that_is_also_a_dataclass_deserializes_through_from_json():
    restored = assert_round_trip(
        Reading2(Temperature(21.5), [Temperature(1.0), Temperature(-3.0, "F")]), Reading2
    )

    assert type(restored.temperature) is Temperature
    assert deserialize_json("40.0K", Temperature) == Temperature(40.0, "K")
    # the dataclass handler would accept an object: from_json decides, and it wants a string
    with pytest.raises(TypeError, match="expected a JSON value of type str"):
        deserialize_json({"degrees": 21.5, "unit": "C"}, Temperature)


# ---------------------------------------------------------------------------------------------------------------------
# Jsonable: from_json receives whatever JSON is there
# ---------------------------------------------------------------------------------------------------------------------

JSON_VALUES = [
    pytest.param({"a": [1, {"b": None}]}, id="object"),
    pytest.param({}, id="empty-object"),
    pytest.param([1, [2, 3]], id="array"),
    pytest.param([], id="empty-array"),
    pytest.param("text", id="string"),
    pytest.param(7, id="int"),
    pytest.param(2.5, id="float"),
    pytest.param(True, id="bool"),
    pytest.param(None, id="null"),
]


@pytest.mark.parametrize("data", JSON_VALUES)
def test_from_json_receives_every_kind_of_json_value(data, received):
    restored = deserialize_json(data, Recorder)

    assert type(restored) is Recorder
    assert len(received) == 1
    assert_strictly_equal(received[0], data)


def test_from_json_receives_the_value_of_a_nested_position(received):
    deserialize_json({"x": [1, 2], "y": 3}, dict[str, Recorder])
    deserialize_json([None, "s"], list[Recorder])

    assert_strictly_equal(received, [[1, 2], 3, None, "s"])


def test_dict_shaped_jsonable_rejects_other_shapes_with_the_error_of_its_own_from_json():
    for data in (None, [1, 2], "x", 3):
        with pytest.raises(TypeError, match="expected a JSON value of type dict"):
            deserialize_json(data, Point)
    with pytest.raises(TypeError, match="expected a JSON value of type list"):
        deserialize_json({"x": 1}, Coordinate)


def test_a_jsonable_union_member_that_rejects_the_data_with_typeerror_lets_the_next_member_try():
    assert deserialize_json(None, Optional[Point]) is None
    assert deserialize_json("s", Union[Point, str]) == "s"
    assert deserialize_json([1.5, 2.5], Union[Point, Coordinate]) == Coordinate(1.5, 2.5)
    with pytest.raises(TypeError, match="Unsupported types for deserialization"):
        deserialize_json(5, Union[Point, Coordinate])


# ---------------------------------------------------------------------------------------------------------------------
# Jsonable: failure modes
# ---------------------------------------------------------------------------------------------------------------------

SELF_RETURNING_POSITIONS = [
    pytest.param(lambda item: item, id="root"),
    pytest.param(lambda item: [item], id="list-item"),
    pytest.param(lambda item: {"k": item}, id="dict-value"),
    pytest.param(lambda item: Holder(item, []), id="dataclass-field"),
    pytest.param(lambda item: [{"a": (item,)}], id="deeply-nested"),
    pytest.param(lambda item: {item}, id="set-member"),
]


@pytest.mark.parametrize("wrap", SELF_RETURNING_POSITIONS)
@pytest.mark.parametrize("serializer", [serialize_json, serialize_json_as_obj], ids=["text", "intermediate"])
def test_to_json_returning_self_fails_predictably(wrap, serializer):
    with pytest.raises(ValueError, match="(?i)cyc") as caught:
        serializer(wrap(SelfReturning()))

    assert "PAYLOAD_MUST_NOT_LEAK" not in str(caught.value)


def test_to_json_containing_the_jsonable_itself_is_a_cycle():
    node = Linked()
    node.peer = node

    with pytest.raises(ValueError, match="(?i)cyc") as caught:
        serialize_json(node)
    assert "PAYLOAD_MUST_NOT_LEAK" not in str(caught.value)


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda item: item, id="root"),
        pytest.param(lambda item: [{"k": item}], id="nested"),
        pytest.param(lambda item: Holder(item, []), id="dataclass-field"),
    ],
)
def test_a_cycle_that_runs_through_a_set_fails_with_a_value_error(wrap):
    # The members of a set are encoded one by one, each on its own, so the repeat is not noticed: the cycle ends at the
    # depth limit instead. Either way it is a ValueError, and the message names no value.
    with pytest.raises(ValueError, match="(?i)cyc|nested too deeply") as caught:
        serialize_json(wrap(SetWithSelf()))

    assert "SET_PAYLOAD_MUST_NOT_LEAK" not in str(caught.value)


def test_two_jsonables_returning_each_other_are_a_cycle():
    first, second = Linked(), Linked()
    first.peer, second.peer = second, first

    with pytest.raises(ValueError, match="(?i)cyc") as caught:
        serialize_json([first])
    assert "PAYLOAD_MUST_NOT_LEAK" not in str(caught.value)


def test_jsonable_chain_without_a_cycle_ends_in_null():
    first, second, third = Linked(), Linked(), Linked()
    first.peer, second.peer = second, third

    assert serialize_json(first) == '{"peer": {"peer": {"peer": null}}}'


@pytest.mark.parametrize("depth", [1, 10, 100])
def test_jsonable_chain_of_any_depth_serializes_to_nested_objects(depth):
    node = None
    for _ in range(depth):
        node = Linked(node)

    assert serialize_json(node) == '{"peer": ' * depth + "null" + "}" * depth


@pytest.mark.parametrize(
    "make", [Endless, EndlessInList], ids=["returns-new-instance", "returns-list-with-new-instance"]
)
def test_to_json_that_never_stops_nesting_fails_with_a_bounded_error(make):
    with pytest.raises(ValueError, match="nested too deeply") as caught:
        serialize_json(make())

    assert isinstance(caught.value.__cause__, RecursionError)


def test_unsupported_value_inside_a_to_json_result_is_a_typeerror_without_a_fallback():
    with pytest.raises(TypeError, match="Object of type Opaque is not JSON serializable"):
        serialize_json([OpaqueCarrier("a")])
    with pytest.raises(TypeError, match="Object of type Opaque is not JSON serializable"):
        serialize_json_as_obj({"k": OpaqueCarrier("a")})


def test_to_json_that_never_stops_nesting_fails_inside_a_structure_too():
    with pytest.raises(ValueError, match="nested too deeply"):
        serialize_json_as_obj({"a": [Endless()]})


@pytest.mark.parametrize("wrap", SELF_RETURNING_POSITIONS)
def test_error_raised_by_to_json_propagates_unchanged(wrap):
    with pytest.raises(ConversionError) as caught:
        serialize_json(wrap(ExplodingJsonable()))

    # Python 3.13+ adds a note about the position to the exception: the arguments are what the callback raised
    assert caught.value.args == ("to_json failed",)


# ---------------------------------------------------------------------------------------------------------------------
# Jsonable: limitations and defects
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "data", "error", "message"),
    [
        pytest.param(Union[StrictSlug, str], "not a slug!", ValueError, "not a slug", id="validation-error"),
        pytest.param(
            Optional[LenientPoint],
            None,
            AttributeError,
            "'NoneType' object has no attribute 'get'",
            id="get-on-null-in-an-optional",
        ),
    ],
)
def test_from_json_errors_other_than_typeerror_escape_unions_limitation(cls, data, error, message):
    # Documents current behavior, not a promise: only TypeError makes a union try its next member, so a `from_json`
    # that validates with ValueError (the usual choice), or fails on null with AttributeError, ends the whole union.
    with pytest.raises(error, match=message):
        deserialize_json(data, cls)


def test_union_with_a_validating_jsonable_accepts_data_it_validates():
    assert deserialize_json("slug", Union[StrictSlug, str]) == StrictSlug("slug")


def test_null_for_optional_jsonable_reaches_from_json_limitation():
    # Documents current behavior, not a promise: Optional[X] is Union[X, None], X is tried first, and a Jsonable
    # receives every JSON value, null included. A tolerant `from_json` such as Slug's turns null into an object
    # instead of None (a `from_json` that raises TypeError for null, like Point's, gives None; see below).
    assert_strictly_equal(deserialize_json(None, Optional[Slug]), Slug(None))
    assert_strictly_equal(deserialize_json({"label": "x", "slug": None}, Tagged), Tagged("x", Slug(None)))
    assert_strictly_equal(deserialize_json([None, "a"], list[Optional[Slug]]), [Slug(None), Slug("a")])


def test_null_for_a_union_that_lists_none_first_is_none():
    # The workaround for the limitation above: with the members the other way round, null is None
    assert deserialize_json(None, Union[None, Slug]) is None
    assert deserialize_json("a", Union[None, Slug]) == Slug("a")


def test_jsonable_mapping_key_is_rejected_limitation():
    # Documents current behavior, not a promise: only str, bool, int, float, Enum and tuple keys are supported, and
    # the fallback does not apply to keys either.
    calls = []

    with pytest.raises(TypeError, match="Unsupported JSON mapping key type"):
        serialize_json({Point(1, 2): "v"})
    with pytest.raises(TypeError, match="Unsupported JSON mapping key type"):
        serialize_json({Point(1, 2): "v"}, fallback=lambda value: calls.append(value) or "key")
    assert calls == []


@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(NamedStr("x"), '{"value": "x"}', id="str-subclass"),
        pytest.param(NamedInt(3), '{"value": 3}', id="int-subclass"),
        pytest.param(NamedFloat(3.5), '{"value": 3.5}', id="float-subclass"),
    ],
)
def test_to_json_of_a_jsonable_subclassing_a_primitive_is_used(value, text):
    assert serialize_json(value) == text
    assert serialize_json([value]) == f"[{text}]"
    assert serialize_json(Holder(value, [])) == f'{{"item": {text}, "items": []}}'


# ---------------------------------------------------------------------------------------------------------------------
# fallback=
# ---------------------------------------------------------------------------------------------------------------------


class Secret:
    def __repr__(self):
        return "SECRET_PAYLOAD_MUST_NOT_LEAK"


class OpaqueDict(dict):
    """Unsupported-looking, but a dict: it is a mapping, so the fallback is not for it."""


class OpaqueList(list):
    pass


def as_marker(value):
    return {"opaque": value.tag}


def recording(convert):
    calls = []

    def fallback(value):
        calls.append(value)
        return convert(value)

    return fallback, calls


def never_called(value):
    raise AssertionError(f"the fallback received a supported {type(value).__name__}")


FALLBACK_POSITIONS = [
    pytest.param(Opaque("a"), '{"opaque": "a"}', id="root"),
    pytest.param([Opaque("a"), 1], '[{"opaque": "a"}, 1]', id="list-item"),
    pytest.param({"k": Opaque("a")}, '{"k": {"opaque": "a"}}', id="dict-value"),
    pytest.param((Opaque("a"), "x"), '[{"opaque": "a"}, "x"]', id="tuple-item"),
    pytest.param(
        Holder(Opaque("a"), [Opaque("b")]),
        '{"item": {"opaque": "a"}, "items": [{"opaque": "b"}]}',
        id="dataclass-fields",
    ),
    pytest.param({Opaque("a")}, '[{"opaque": "a"}]', id="set-member"),
    pytest.param(frozenset({Opaque("a")}), '[{"opaque": "a"}]', id="frozenset-member"),
    pytest.param(
        MappingProxyType({"k": Opaque("a")}), '{"k": {"opaque": "a"}}', id="mappingproxy-value"
    ),
    pytest.param(
        UserDict({"k": [Opaque("a")]}), '{"k": [{"opaque": "a"}]}', id="mapping-implementation-value"
    ),
    pytest.param(
        {"a": [{"b": (Opaque("a"),)}]}, '{"a": [{"b": [{"opaque": "a"}]}]}', id="deeply-nested"
    ),
    pytest.param(OpaqueCarrier("a"), '{"carried": {"opaque": "a"}}', id="inside-jsonable-result"),
    pytest.param(
        {"x": Holder({"y": [OpaqueCarrier("a")]}, [])},
        '{"x": {"item": {"y": [{"carried": {"opaque": "a"}}]}, "items": []}}',
        id="through-dataclass-dict-list-jsonable",
    ),
]


@pytest.mark.parametrize(("value", "text"), FALLBACK_POSITIONS)
def test_fallback_converts_an_unsupported_instance_at_any_position(value, text):
    assert serialize_json(value, fallback=as_marker) == text
    assert_same_json(serialize_json_as_obj(value, fallback=as_marker), json.loads(text))
    assert_same_json(serialize_json_as_dict(value, fallback=as_marker), json.loads(text))


FALLBACK_RESULTS = [
    pytest.param(lambda value: {"tag": value.tag, "n": 1}, '{"tag": "t", "n": 1}', id="dict"),
    pytest.param(lambda value: f"<{value.tag}>", '"<t>"', id="str"),
    pytest.param(lambda value: [value.tag, 2, None], '["t", 2, null]', id="list"),
    pytest.param(lambda value: (value.tag, 2.5), '["t", 2.5]', id="tuple"),
    pytest.param(lambda value: 42, "42", id="int"),
    pytest.param(lambda value: None, "null", id="none"),
    pytest.param(lambda value: False, "false", id="bool"),
    pytest.param(lambda value: {}, "{}", id="empty-dict"),
    pytest.param(lambda value: [], "[]", id="empty-list"),
    pytest.param(lambda value: Priority.HIGH, "2", id="enum"),
    pytest.param(lambda value: b"JSerPy", '"SlNlclB5"', id="bytes"),
    pytest.param(lambda value: Decimal("1.50"), '"1.50"', id="decimal"),
    pytest.param(lambda value: datetime(2024, 1, 2, 3, 4, 5), '"2024-01-02T03:04:05"', id="datetime"),
    pytest.param(lambda value: PurePosixPath("a/b"), '"a/b"', id="path"),
    pytest.param(lambda value: frozenset({3, 1, 2}), "[1, 2, 3]", id="frozenset"),
    pytest.param(lambda value: Point(1, 2), '{"x": 1, "y": 2}', id="jsonable"),
    pytest.param(
        lambda value: Inner(datetime(2024, 1, 2), Label.BETA),
        '{"when": "2024-01-02T00:00:00", "kind": "beta"}',
        id="dataclass",
    ),
    pytest.param(
        lambda value: {Label.BETA: 1, (1, 2): 3, 4: 5}, '{"beta": 1, "[1, 2]": 3, "4": 5}', id="converted-keys"
    ),
    pytest.param(lambda value: MappingProxyType({"k": [value.tag]}), '{"k": ["t"]}', id="mappingproxy"),
]


@pytest.mark.parametrize(("convert", "text"), FALLBACK_RESULTS)
def test_fallback_may_return_any_kind_of_json_value(convert, text):
    value = Opaque("t")

    assert serialize_json(value, fallback=convert) == text
    assert serialize_json([value], fallback=convert) == f"[{text}]"
    assert serialize_json({"k": value}, fallback=convert) == f'{{"k": {text}}}'
    assert_same_json(serialize_json_as_obj(value, fallback=convert), json.loads(text))


def test_fallback_is_called_once_per_occurrence_in_document_order():
    fallback, calls = recording(as_marker)
    value = [Opaque("a"), {"k": Opaque("b"), "j": [Opaque("c")]}, Holder(Opaque("d"), [Opaque("e")]), (Opaque("f"),)]

    text = serialize_json(value, fallback=fallback)

    assert [call.tag for call in calls] == ["a", "b", "c", "d", "e", "f"]
    assert text == (
        '[{"opaque": "a"}, {"k": {"opaque": "b"}, "j": [{"opaque": "c"}]}, '
        '{"item": {"opaque": "d"}, "items": [{"opaque": "e"}]}, [{"opaque": "f"}]]'
    )


def test_same_unsupported_instance_twice_is_not_a_cycle():
    shared = Opaque("s")
    fallback, calls = recording(as_marker)

    text = serialize_json({"a": shared, "b": [shared]}, fallback=fallback)

    assert text == '{"a": {"opaque": "s"}, "b": [{"opaque": "s"}]}'
    assert len(calls) == 2


def chain(value):
    # Every step returns a different kind of container holding the next unsupported value
    if value.tag == "a":
        return {"b": Opaque("b"), "when": datetime(2024, 1, 2, 3, 4, 5)}
    if value.tag == "b":
        return [Opaque("c"), PurePosixPath("docs/readme.md")]
    if value.tag == "c":
        return (Opaque("d"), Decimal("1.50"))
    if value.tag == "d":
        return {Opaque("e")}
    return f"end:{value.tag}"


def test_fallback_result_with_more_unsupported_values_is_converted_recursively_four_levels_deep():
    fallback, calls = recording(chain)

    text = serialize_json(Opaque("a"), fallback=fallback)

    assert text == (
        '{"b": [[["end:e"], "1.50"], "docs/readme.md"], "when": "2024-01-02T03:04:05"}'
    )
    assert [call.tag for call in calls] == ["a", "b", "c", "d", "e"]


def test_fallback_chain_inside_a_structure_gives_the_same_result_through_every_entry_point():
    value = {"items": [Opaque("a")]}
    inner = '{"b": [[["end:e"], "1.50"], "docs/readme.md"], "when": "2024-01-02T03:04:05"}'

    assert serialize_json(value, fallback=chain) == f'{{"items": [{inner}]}}'
    assert_same_json(serialize_json_as_obj(value, fallback=chain), json.loads(f'{{"items": [{inner}]}}'))


def test_fallback_recursion_with_the_same_class_at_every_level():
    def descend(value):
        return {"depth": value.tag, "next": Opaque(value.tag + 1)} if value.tag < 3 else "bottom"

    fallback, calls = recording(descend)

    assert serialize_json(Opaque(0), fallback=fallback) == (
        '{"depth": 0, "next": {"depth": 1, "next": {"depth": 2, "next": "bottom"}}}'
    )
    assert [call.tag for call in calls] == [0, 1, 2, 3]


def test_fallback_recursion_sixty_levels_deep():
    def descend(value):
        return {"level": value.tag, "next": Opaque(value.tag + 1)} if value.tag < 60 else None

    fallback, calls = recording(descend)

    parsed = json.loads(serialize_json(Opaque(0), fallback=fallback))

    levels = []
    while parsed is not None:
        levels.append(parsed["level"])
        parsed = parsed["next"]
    assert levels == list(range(60))
    assert len(calls) == 61


def test_set_of_unsupported_values_is_ordered_by_the_text_of_the_converted_values():
    members = [Opaque(10), Opaque(2), Opaque(1)]

    # "1" < "10" < "2", whatever the insertion order and the hashes
    assert serialize_json(set(members), fallback=lambda value: value.tag) == "[1, 10, 2]"
    assert serialize_json(frozenset(reversed(members)), fallback=lambda value: value.tag) == "[1, 10, 2]"


def test_set_mixing_supported_and_converted_members_is_ordered_by_text():
    members = {Opaque("x"), 5, "s"}

    # the texts "fb-x" < "s" < 5, since the quote sorts before the digit
    assert serialize_json(members, fallback=lambda value: f"fb-{value.tag}") == '["fb-x", "s", 5]'


def test_set_of_unsupported_values_converted_to_objects():
    members = {Opaque("b"), Opaque("c"), Opaque("a")}

    assert serialize_json(members, fallback=as_marker) == (
        '[{"opaque": "a"}, {"opaque": "b"}, {"opaque": "c"}]'
    )


def test_sets_nested_in_sets_convert_each_unsupported_member_once():
    fallback, calls = recording(lambda value: value.tag)
    members = frozenset({frozenset({Opaque(1)}), frozenset({Opaque(2), Opaque(3)}), frozenset()})

    # the members are ordered by their text: "[1]" < "[2,3]" < "[]", since "]" sorts after the digits
    assert serialize_json(members, fallback=fallback) == "[[1], [2, 3], []]"
    assert sorted(call.tag for call in calls) == [1, 2, 3]  # the iteration order of a set is not defined


class PlainNode:
    """A plain class (no dataclass, no Jsonable) with attributes, the kind `fallback=vars` is meant for."""

    def __init__(self, name: str, children: list = (), parent: Any = None):
        self.name = name
        self.children = list(children)
        self.parent = parent


def test_vars_as_the_fallback_converts_a_graph_of_plain_objects():
    shared = PlainNode("shared")
    root = PlainNode("root", [PlainNode("a"), shared, shared])

    def leaf(name):
        return f'{{"name": "{name}", "children": [], "parent": null}}'

    # the attributes come in the order of their assignment; the shared child appears twice, which is not a cycle
    expected = f'{{"name": "root", "children": [{leaf("a")}, {leaf("shared")}, {leaf("shared")}], "parent": null}}'
    assert serialize_json(root, fallback=vars) == expected
    assert_same_json(serialize_json_as_obj(root, fallback=vars), json.loads(expected))


@pytest.mark.parametrize(
    "convert",
    [
        pytest.param(vars, id="the-attribute-dict-itself"),
        pytest.param(lambda node: dict(vars(node)), id="copy-of-the-attributes"),
        pytest.param(
            lambda node: {**vars(node), "children": list(node.children)}, id="copy-with-a-copy-of-the-children"
        ),
    ],
)
def test_parent_back_reference_through_a_fallback_is_a_cycle(convert):
    root = PlainNode("root")
    root.children.append(PlainNode("child", parent=root))

    with pytest.raises(ValueError, match="(?i)cyc"):
        serialize_json(root, fallback=convert)


class Alternating(Jsonable):
    """`to_json` carries an unsupported value, the fallback (see `unwind`) carries the next Alternating."""

    def __init__(self, steps: int):
        self.steps = steps

    def to_json(self):
        return {"steps": self.steps, "next": Opaque(self.steps)}

    @classmethod
    def from_json(cls, data):
        return cls(data["steps"])


def unwind(value):
    return [Alternating(value.tag - 1)] if value.tag > 0 else "end"


def test_jsonable_and_fallback_hand_over_to_each_other_to_any_depth():
    assert serialize_json(Alternating(3), fallback=unwind) == (
        '{"steps": 3, "next": [{"steps": 2, "next": [{"steps": 1, "next": [{"steps": 0, "next": "end"}]}]}]}'
    )
    # the keys of a set member are always sorted, see test_set_members_that_are_objects_always_have_sorted_keys_limitation
    assert serialize_json([{Alternating(0)}], fallback=unwind) == '[[{"next": "end", "steps": 0}]]'


# precedence: a registered handler (or a mapping, list or tuple) always wins over the fallback


def make_supported_values():
    return {
        "dataclass": Inner(datetime(2024, 2, 29, 12, 30), Label.ALPHA),
        "set": {3, 1, 2},
        "frozenset": frozenset({"b", "a"}),
        "tuple": (1, "x"),
        "path": PurePosixPath("a/b.txt"),
        "datetime": datetime(2024, 2, 29, 12, 30, 15),
        "date": date(2024, 2, 29),
        "time": time(7, 5, 3),
        "decimal": Decimal("1E+3"),
        "bytes": b"JSerPy",
        "enum": Priority.HIGH,
        "jsonable": Point(1, 2),
        "mappingproxy": MappingProxyType({"k": 1}),
        "mapping": UserDict({"k": 2}),
        "dict-subclass": OpaqueDict(k=3),
        "list-subclass": OpaqueList([4]),
    }


SUPPORTED_JSON = {
    "dataclass": {"when": "2024-02-29T12:30:00", "kind": "alpha"},
    "set": [1, 2, 3],
    "frozenset": ["a", "b"],
    "tuple": [1, "x"],
    "path": "a/b.txt",
    "datetime": "2024-02-29T12:30:15",
    "date": "2024-02-29",
    "time": "07:05:03",
    "decimal": "1E+3",
    "bytes": "SlNlclB5",
    "enum": 2,
    "jsonable": {"x": 1, "y": 2},
    "mappingproxy": {"k": 1},
    "mapping": {"k": 2},
    "dict-subclass": {"k": 3},
    "list-subclass": [4],
}


@pytest.mark.parametrize("serializer", [serialize_json, serialize_json_as_obj], ids=["text", "intermediate"])
def test_handlers_take_precedence_over_the_fallback_in_one_structure(serializer):
    value = make_supported_values()
    value["unsupported"] = Opaque("only")
    fallback, calls = recording(as_marker)

    result = serializer(value, fallback=fallback)

    parsed = json.loads(result) if isinstance(result, str) else result
    assert_same_json(parsed, {**SUPPORTED_JSON, "unsupported": {"opaque": "only"}})
    assert [call.tag for call in calls] == ["only"]


def test_fallback_is_not_needed_when_nothing_is_unsupported():
    value = make_supported_values()

    assert_same_json(json.loads(serialize_json(value, fallback=never_called)), SUPPORTED_JSON)
    assert_same_json(json.loads(serialize_json(value)), SUPPORTED_JSON)


# failure modes


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda item: item, id="root"),
        pytest.param(lambda item: [item], id="list-item"),
        pytest.param(lambda item: {"k": {"j": item}}, id="dict-value"),
        pytest.param(lambda item: Holder(item, []), id="dataclass-field"),
        pytest.param(lambda item: {item}, id="set-member"),
    ],
)
@pytest.mark.parametrize("serializer", [serialize_json, serialize_json_as_obj], ids=["text", "intermediate"])
def test_fallback_returning_the_same_object_fails_predictably(wrap, serializer):
    with pytest.raises(ValueError, match="(?i)fallback") as caught:
        serializer(wrap(Secret()), fallback=lambda value: value)

    assert "SECRET_PAYLOAD_MUST_NOT_LEAK" not in str(caught.value)


@pytest.mark.parametrize(
    "convert",
    [
        pytest.param(lambda value: {"self": value}, id="mapping-with-the-value"),
        pytest.param(lambda value: [[value]], id="nested-list-with-the-value"),
        pytest.param(lambda value: Holder(value, []), id="dataclass-with-the-value"),
    ],
)
def test_fallback_result_containing_the_original_value_is_a_cycle(convert):
    with pytest.raises(ValueError, match="(?i)cyc") as caught:
        serialize_json([Secret()], fallback=convert)

    assert "SECRET_PAYLOAD_MUST_NOT_LEAK" not in str(caught.value)


def test_fallback_alternating_between_two_instances_is_a_cycle():
    first, second = Opaque("first"), Opaque("second")

    with pytest.raises(ValueError, match="(?i)cyc"):
        serialize_json(first, fallback=lambda value: second if value is first else first)


def test_fallback_returning_a_set_with_the_original_value_fails_with_a_value_error():
    # As for a Jsonable: the members of a set are encoded on their own, so this ends at the depth limit
    with pytest.raises(ValueError, match="(?i)cyc|nested too deeply") as caught:
        serialize_json([Secret()], fallback=lambda value: {value})

    assert "SECRET_PAYLOAD_MUST_NOT_LEAK" not in str(caught.value)


def test_fallback_returning_a_new_unsupported_value_each_time_fails_with_a_bounded_error():
    with pytest.raises(ValueError, match="nested too deeply") as caught:
        serialize_json(Opaque("endless"), fallback=lambda value: Opaque("endless"))

    assert isinstance(caught.value.__cause__, RecursionError)


def test_fallback_returning_a_cyclic_structure_fails_predictably():
    cyclic = {"k": []}
    cyclic["k"].append(cyclic)

    with pytest.raises(ValueError, match="(?i)cyc"):
        serialize_json({"inside": Opaque("x")}, fallback=lambda value: cyclic)


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda item: item, id="root"),
        pytest.param(lambda item: [{"k": item}], id="nested"),
        pytest.param(lambda item: Holder(item, []), id="dataclass-field"),
        pytest.param(lambda item: {item}, id="set-member"),
    ],
)
@pytest.mark.parametrize("serializer", [serialize_json, serialize_json_as_obj], ids=["text", "intermediate"])
def test_error_raised_by_the_fallback_propagates_unchanged(wrap, serializer):
    def fallback(value):
        raise ConversionError(f"cannot convert {value.tag}")

    with pytest.raises(ConversionError) as caught:
        serializer(wrap(Opaque("boom")), fallback=fallback)

    assert caught.value.args == ("cannot convert boom",)


def test_fallback_raising_valueerror_or_typeerror_keeps_its_own_message():
    def fail_with(error):
        def fallback(value):
            raise error

        return fallback

    with pytest.raises(ValueError) as value_error:
        serialize_json([Opaque(1)], fallback=fail_with(ValueError("my own value problem")))
    with pytest.raises(TypeError) as type_error:
        serialize_json([Opaque(1)], fallback=fail_with(TypeError("my own type problem")))

    assert value_error.value.args == ("my own value problem",)
    assert type_error.value.args == ("my own type problem",)


KEYS_WITHOUT_SUPPORT = [
    pytest.param(lambda: Opaque("k"), id="custom-instance"),
    pytest.param(lambda: 1j, id="complex"),
    pytest.param(lambda: int, id="type-object"),
    pytest.param(lambda: Inner(datetime(2024, 1, 1), Label.ALPHA), id="dataclass"),
]


@pytest.mark.parametrize("make_key", KEYS_WITHOUT_SUPPORT)
def test_fallback_is_not_applied_to_mapping_keys(make_key):
    fallback, calls = recording(lambda value: "converted")

    for value in ({make_key(): 1}, Holder({make_key(): 1}, []), [{"k": {make_key(): 1}}]):
        with pytest.raises(TypeError, match="Unsupported JSON mapping key type"):
            serialize_json(value, fallback=fallback)
        with pytest.raises(TypeError, match="Unsupported JSON mapping key type"):
            serialize_json(value)

    assert calls == []


def test_fallback_inside_a_tuple_mapping_key_is_applied_limitation():
    # Documents current behavior, not a promise: a key that is itself an unsupported value is a TypeError, but the
    # members of a tuple key are serialized like any other value, so the fallback converts them.
    assert serialize_json({(Opaque("a"), 1): "v"}, fallback=lambda value: value.tag) == '{"[\\"a\\", 1]": "v"}'


# unsupported standard types


def make_generator():
    return (n * n for n in range(4))


def helper_function():
    return None


UNSUPPORTED_STANDARD_VALUES = [
    pytest.param(lambda: complex(1, -2), "complex", lambda c: [c.real, c.imag], "[1.0, -2.0]", id="complex"),
    pytest.param(
        lambda: range(2, 10, 3),
        "range",
        lambda r: {"start": r.start, "stop": r.stop, "step": r.step},
        '{"start": 2, "stop": 10, "step": 3}',
        id="range",
    ),
    pytest.param(lambda: slice(1, None, 2), "slice", lambda s: [s.start, s.stop, s.step], "[1, null, 2]", id="slice"),
    pytest.param(lambda: int, "type", lambda t: t.__qualname__, '"int"', id="builtin-type"),
    pytest.param(lambda: Opaque, "type", lambda t: t.__qualname__, '"Opaque"', id="class"),
    pytest.param(  # the metaclass is called EnumMeta before Python 3.11 and EnumType since
        lambda: Label, "Enum(Type|Meta)", lambda t: [member.value for member in t], '["alpha", "beta"]', id="enum-class"
    ),
    pytest.param(lambda: Inner, "type", lambda t: t.__qualname__, '"Inner"', id="dataclass-class"),
    pytest.param(lambda: helper_function, "function", lambda f: f.__name__, '"helper_function"', id="function"),
    pytest.param(lambda: (lambda: 1), "function", lambda f: f.__name__, '"<lambda>"', id="lambda"),
    pytest.param(make_generator, "generator", list, "[0, 1, 4, 9]", id="generator"),
    pytest.param(lambda: iter([1, 2]), "list_iterator", list, "[1, 2]", id="list-iterator"),
    pytest.param(lambda: map(str, [1, 2]), "map", list, '["1", "2"]', id="map"),
    pytest.param(lambda: zip("ab", [1, 2]), "zip", list, '[["a", 1], ["b", 2]]', id="zip"),
    pytest.param(lambda: {"a": 1}.keys(), "dict_keys", list, '["a"]', id="dict-keys"),
    pytest.param(lambda: {"a": 1}.values(), "dict_values", list, "[1]", id="dict-values"),
    pytest.param(lambda: {"a": 1}.items(), "dict_items", list, '[["a", 1]]', id="dict-items"),
    pytest.param(lambda: object(), "object", lambda o: "an object", '"an object"', id="object"),
]


@pytest.mark.parametrize(("make", "type_name", "convert", "text"), UNSUPPORTED_STANDARD_VALUES)
def test_unsupported_standard_values_raise_typeerror_without_a_fallback(make, type_name, convert, text):
    for value in (make(), [make()], {"k": make()}, Holder(make(), [])):
        with pytest.raises(TypeError, match=f"Object of type {type_name} is not JSON serializable"):
            serialize_json(value)
        with pytest.raises(TypeError, match=f"Object of type {type_name} is not JSON serializable"):
            serialize_json_as_obj(value)


@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(lambda item: {item}, id="set-member"),
        pytest.param(lambda item: frozenset({item}), id="frozenset-member"),
        pytest.param(lambda item: (item, 1), id="tuple-item"),
        pytest.param(lambda item: MappingProxyType({"k": item}), id="mappingproxy-value"),
        pytest.param(lambda item: UserDict({"k": [item]}), id="mapping-implementation-value"),
        pytest.param(lambda item: {("k", item): 1}, id="tuple-mapping-key-member"),
        pytest.param(lambda item: Carrier(item), id="jsonable-result"),
    ],
)
@pytest.mark.parametrize("serializer", [serialize_json, serialize_json_as_obj], ids=["text", "intermediate"])
def test_unsupported_value_at_any_position_is_a_typeerror_that_names_the_type_only(wrap, serializer):
    with pytest.raises(TypeError, match="Object of type Secret is not JSON serializable") as caught:
        serializer(wrap(Secret()))

    assert "SECRET_PAYLOAD_MUST_NOT_LEAK" not in str(caught.value)


@pytest.mark.parametrize(("make", "type_name", "convert", "text"), UNSUPPORTED_STANDARD_VALUES)
def test_unsupported_standard_values_are_converted_by_a_fallback(make, type_name, convert, text):
    assert serialize_json(make(), fallback=convert) == text
    assert serialize_json([make()], fallback=convert) == f"[{text}]"
    assert serialize_json({"k": make()}, fallback=convert) == f'{{"k": {text}}}'
    assert serialize_json(Holder(make(), []), fallback=convert) == f'{{"item": {text}, "items": []}}'
    assert_same_json(serialize_json_as_obj(make(), fallback=convert), json.loads(text))


def test_singledispatch_fallback_converts_each_unsupported_type_its_own_way():
    @singledispatch
    def fallback(value):
        raise TypeError(f"no converter for {type(value).__name__}")

    fallback.register(complex, lambda value: [value.real, value.imag])
    fallback.register(range, lambda value: list(value))
    fallback.register(Opaque, as_marker)

    value = {"c": 1j, "r": range(3), "o": [Opaque("x")]}

    assert serialize_json(value, fallback=fallback) == '{"c": [0.0, 1.0], "r": [0, 1, 2], "o": [{"opaque": "x"}]}'
    with pytest.raises(TypeError) as caught:
        serialize_json([1j, slice(1)], fallback=fallback)
    assert caught.value.args == ("no converter for slice",)


# ---------------------------------------------------------------------------------------------------------------------
# Encoding options on a complex object
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    sensor: str
    value: float


@dataclass
class Spot:
    name: str
    lat: float
    lon: float


@dataclass
class Task:
    title: str
    priority: Priority
    due: datetime
    labels: frozenset[Label]
    done: bool = False
    note: Optional[str] = None


@dataclass
class Project:
    name: str
    home: Spot
    tasks: list[Task]
    budget: Decimal


@dataclass
class Dashboard:
    # The field order is deliberately not alphabetical
    title: str
    projects: list[Project]
    # Variable length: the singleton dashboard has a key of one element, the others have two
    grid: dict[tuple[int, ...], float]
    index: dict[Label, list[str]]
    readings: frozenset[Reading]
    primes: set[int]
    ratios: tuple[float, ...]
    created: date
    opens: time
    blob: bytes
    meta: dict[str, Any]


def make_dashboard() -> Dashboard:
    return Dashboard(
        title="Dashboard שלום 😀",
        projects=[
            Project(
                name='Ωmega "quoted" \\ back',
                home=Spot("Tel Aviv", 32.0853, 34.7818),
                tasks=[
                    Task(
                        "Write ספר",
                        Priority.HIGH,
                        datetime(2024, 2, 29, 12, 30),
                        frozenset({Label.BETA, Label.ALPHA}),
                        done=True,
                    ),
                    Task(
                        "Plan",
                        Priority.LOW,
                        datetime(2024, 3, 1, 8, 0, 0, 250000),
                        frozenset(),
                        note="two\nlines\ttabbed\u2028end",
                    ),
                ],
                budget=Decimal("1250.50"),
            ),
            Project("Empty", Spot("Nowhere", 0.0, -0.0), [], Decimal("1E+3")),
        ],
        grid={(10, -2): -0.25, (0, 1): 1.5, (0, 0): 0.0},
        index={Label.BETA: ["b2", "b1"], Label.ALPHA: []},
        readings=frozenset({Reading("t2", 2.5), Reading("t1", 1.0), Reading("t10", 0.1)}),
        primes={2, 3, 5, 7, 11, 13},
        ratios=(0.1, 1e16, 1e-07, -0.0, 100.0),
        created=date(2024, 2, 29),
        opens=time(7, 5, 3),
        blob=b"JSerPy",
        meta={
            "z": [1, 2.0, True, None, "x"],
            "שם": {"nested": {"deep": [[], {}, [[]]]}},
            "big": 2**70,
            "quote": 'He said "hi"\n\x01',
        },
    )


DASHBOARD_JSON = {
    "title": "Dashboard שלום 😀",
    "projects": [
        {
            "name": 'Ωmega "quoted" \\ back',
            "home": {"name": "Tel Aviv", "lat": 32.0853, "lon": 34.7818},
            "tasks": [
                {
                    "title": "Write ספר",
                    "priority": 2,
                    "due": "2024-02-29T12:30:00",
                    "labels": ["alpha", "beta"],
                    "done": True,
                    "note": None,
                },
                {
                    "title": "Plan",
                    "priority": 1,
                    "due": "2024-03-01T08:00:00.250000",
                    "labels": [],
                    "done": False,
                    "note": "two\nlines\ttabbed\u2028end",
                },
            ],
            "budget": "1250.50",
        },
        {
            "name": "Empty",
            "home": {"name": "Nowhere", "lat": 0.0, "lon": -0.0},
            "tasks": [],
            "budget": "1E+3",
        },
    ],
    "grid": {"[10, -2]": -0.25, "[0, 1]": 1.5, "[0, 0]": 0.0},
    "index": {"beta": ["b2", "b1"], "alpha": []},
    # ordered by the text of the members: ..."t1"... < ..."t10"... < ..."t2"...
    "readings": [
        {"sensor": "t1", "value": 1.0},
        {"sensor": "t10", "value": 0.1},
        {"sensor": "t2", "value": 2.5},
    ],
    # ordered as text: "11" < "13" < "2" < "3" < "5" < "7"
    "primes": [11, 13, 2, 3, 5, 7],
    "ratios": [0.1, 1e16, 1e-07, -0.0, 100.0],
    "created": "2024-02-29",
    "opens": "07:05:03",
    "blob": "SlNlclB5",
    "meta": {
        "z": [1, 2.0, True, None, "x"],
        "שם": {"nested": {"deep": [[], {}, [[]]]}},
        "big": 1180591620717411303424,
        "quote": 'He said "hi"\n\x01',
    },
}


def test_complex_object_json_shape():
    parsed = to_json_value(make_dashboard())

    assert parsed == DASHBOARD_JSON
    # == cannot tell 1 from 1.0 or -0.0 from 0.0, nor the order of the keys: the text of both can
    assert json.dumps(parsed) == json.dumps(DASHBOARD_JSON)


def test_complex_object_round_trip():
    restored = assert_round_trip(make_dashboard(), Dashboard)

    assert restored.grid == {(10, -2): -0.25, (0, 1): 1.5, (0, 0): 0.0}
    assert restored.readings == make_dashboard().readings
    assert restored.projects[0].tasks[1].note == "two\nlines\ttabbed\u2028end"


def test_complex_object_in_other_containers_round_trips():
    dashboards = [make_dashboard(), make_dashboard()]

    assert_round_trip(dashboards, list[Dashboard])
    assert_round_trip({"first": dashboards[0]}, dict[str, Dashboard])
    assert_round_trip((dashboards[0], None), tuple[Dashboard, Optional[Dashboard]])


def make_empty_dashboard() -> Dashboard:
    return Dashboard("", [], {}, {}, frozenset(), set(), (), date(1, 1, 1), time(0, 0), b"", {})


EMPTY_DASHBOARD_JSON = {
    "title": "",
    "projects": [],
    "grid": {},
    "index": {},
    "readings": [],
    "primes": [],
    "ratios": [],
    "created": "0001-01-01",
    "opens": "00:00:00",
    "blob": "",
    "meta": {},
}


def make_singleton_dashboard() -> Dashboard:
    return Dashboard(
        "x",
        [
            Project(
                "p",
                Spot("s", 1.0, 2.0),
                [Task("t", Priority.LOW, datetime(2000, 1, 1), frozenset({Label.ALPHA}))],
                Decimal("0"),
            )
        ],
        {(1,): 1.0},
        {Label.ALPHA: ["a"]},
        frozenset({Reading("s", 0.0)}),
        {7},
        (1.0,),
        date(2000, 1, 1),
        time(0, 0, 0),
        b"\x00",
        {"k": None},
    )


SINGLETON_DASHBOARD_JSON = {
    "title": "x",
    "projects": [
        {
            "name": "p",
            "home": {"name": "s", "lat": 1.0, "lon": 2.0},
            "tasks": [
                {
                    "title": "t",
                    "priority": 1,
                    "due": "2000-01-01T00:00:00",
                    "labels": ["alpha"],
                    "done": False,
                    "note": None,
                }
            ],
            "budget": "0",
        }
    ],
    "grid": {"[1]": 1.0},
    "index": {"alpha": ["a"]},
    "readings": [{"sensor": "s", "value": 0.0}],
    "primes": [7],
    "ratios": [1.0],
    "created": "2000-01-01",
    "opens": "00:00:00",
    "blob": "AA==",
    "meta": {"k": None},
}


def make_large_dashboard(projects: int = 60) -> Dashboard:
    meta: dict = {"leaf": True}
    for level in range(60):
        meta = {"level": level, "inner": meta}
    return Dashboard(
        title="large",
        projects=[
            Project(
                f"project-{number}",
                Spot(f"spot-{number}", number * 0.25, -number * 0.5),
                [
                    Task(
                        f"task-{number}-{index}",
                        Priority.HIGH if index % 2 else Priority.LOW,
                        datetime(2024, 1 + number % 12, 1 + index, 8, index),
                        frozenset(label for bit, label in enumerate(Label) if (index >> bit) & 1),
                        done=index % 2 == 0,
                        note=None if index % 2 else f"n{number}",
                    )
                    for index in range(5)
                ],
                Decimal(f"{number}.{number % 100:02d}"),
            )
            for number in range(projects)
        ],
        grid={(row, column): row * 0.5 - column for row in range(20) for column in range(20)},
        index={label: [f"{label.value}-{item}" for item in range(30)] for label in Label},
        readings=frozenset(Reading(f"s{item}", item / 4) for item in range(150)),
        primes=set(range(1, 301)),
        ratios=tuple(item / 8 for item in range(100)),
        created=date(2024, 12, 31),
        opens=time(23, 59, 59),
        blob=bytes(range(256)),
        meta=meta,
    )


def test_empty_object_json_shape():
    parsed = to_json_value(make_empty_dashboard())

    assert parsed == EMPTY_DASHBOARD_JSON
    assert json.dumps(parsed) == json.dumps(EMPTY_DASHBOARD_JSON)
    assert serialize_json(make_empty_dashboard(), indent=2) == (
        '{\n  "title": "",\n  "projects": [],\n  "grid": {},\n  "index": {},\n  "readings": [],\n  "primes": [],\n'
        '  "ratios": [],\n  "created": "0001-01-01",\n  "opens": "00:00:00",\n  "blob": "",\n  "meta": {}\n}'
    )
    assert_round_trip(make_empty_dashboard(), Dashboard)


def test_singleton_object_json_shape():
    parsed = to_json_value(make_singleton_dashboard())

    assert parsed == SINGLETON_DASHBOARD_JSON
    assert json.dumps(parsed) == json.dumps(SINGLETON_DASHBOARD_JSON)
    assert_round_trip(make_singleton_dashboard(), Dashboard)


def test_large_object_round_trip_and_structure():
    dashboard = make_large_dashboard()

    restored = assert_round_trip(dashboard, Dashboard)
    parsed = to_json_value(dashboard)

    assert len(restored.projects) == 60 and len(restored.grid) == 400
    assert [len(project["tasks"]) for project in parsed["projects"]] == [5] * 60
    # sets are written in the order of the text of their members: "1" < "10" < "100" < "101" < ... < "2"
    assert parsed["primes"] == sorted(range(1, 301), key=str)
    assert [reading["sensor"] for reading in parsed["readings"]] == sorted(
        (f"s{item}" for item in range(150)), key=lambda sensor: json.dumps(sensor)
    )
    depth, meta = 0, parsed["meta"]
    while "inner" in meta:
        depth, meta = depth + 1, meta["inner"]
    assert (depth, meta) == (60, {"leaf": True})


INDENTS = [None, 0, 2, 4, "\t"]
SEPARATORS = [None, (",", ":"), (", ", ": ")]
INDENT_IDS = {None: "no-indent", 0: "indent-0", 2: "indent-2", 4: "indent-4", "\t": "indent-tab"}
SEPARATOR_IDS = {None: "default-sep", (",", ":"): "compact", (", ", ": "): "spaced"}

OPTION_COMBINATIONS = [
    pytest.param(
        {
            "indent": indent,
            "separators": separators,
            "sort_keys": sort_keys,
            "ensure_ascii": ensure_ascii,
            "allow_nan": allow_nan,
        },
        id="-".join(
            [
                INDENT_IDS[indent],
                SEPARATOR_IDS[separators],
                "sorted" if sort_keys else "unsorted",
                "ascii" if ensure_ascii else "unicode",
                "nan-ok" if allow_nan else "no-nan",
            ]
        ),
    )
    for indent, separators, sort_keys, ensure_ascii, allow_nan in itertools.product(
        INDENTS, SEPARATORS, [False, True], [True, False], [True, False]
    )
]


@pytest.mark.parametrize("options", OPTION_COMBINATIONS)
def test_options_change_the_text_but_not_the_data(options):
    dashboard = make_dashboard()

    text = serialize_json(dashboard, **options)

    assert_strictly_equal(json.loads(text), json.loads(serialize_json(dashboard)))
    assert text.isascii() is options["ensure_ascii"]  # the object holds Hebrew and an emoji


@pytest.mark.parametrize("options", OPTION_COMBINATIONS)
def test_options_match_json_dumps_of_the_plain_tree(options):
    dashboard = make_dashboard()

    # The plain tree is what a consumer sees; encoding it with the same options must give the same text
    assert serialize_json(dashboard, **options) == json.dumps(serialize_json_as_obj(dashboard), **options)


@pytest.mark.parametrize("options", OPTION_COMBINATIONS)
def test_complex_object_round_trips_with_every_option_combination(options):
    assert_round_trip(make_dashboard(), Dashboard, **options)


def assert_keys_sorted_everywhere(value):
    if isinstance(value, dict):
        assert list(value) == sorted(value)
        for item in value.values():
            assert_keys_sorted_everywhere(item)
    elif isinstance(value, list):
        for item in value:
            assert_keys_sorted_everywhere(item)


@pytest.mark.parametrize("separators", SEPARATORS, ids=SEPARATOR_IDS.get)
@pytest.mark.parametrize("indent", INDENTS, ids=INDENT_IDS.get)
def test_sort_keys_sorts_every_nesting_level_and_leaves_sets_alone(indent, separators):
    dashboard = make_dashboard()
    unsorted_parsed = to_json_value(dashboard, indent=indent, separators=separators, sort_keys=False)

    parsed = to_json_value(dashboard, indent=indent, separators=separators, sort_keys=True)

    assert_keys_sorted_everywhere(parsed)
    assert parsed == unsorted_parsed
    assert list(parsed)[:3] == ["blob", "created", "grid"]
    assert list(parsed["meta"]) == ["big", "quote", "z", "שם"]
    assert list(parsed["grid"]) == ["[0, 0]", "[0, 1]", "[10, -2]"]
    assert list(parsed["index"]) == ["alpha", "beta"]
    assert list(parsed["projects"][0]["tasks"][0]) == ["done", "due", "labels", "note", "priority", "title"]
    # arrays keep their order: list items stay where they are, set members stay in the order of their text
    assert [project["name"] for project in parsed["projects"]] == ["Ωmega \"quoted\" \\ back", "Empty"]
    assert parsed["primes"] == [11, 13, 2, 3, 5, 7]
    assert parsed["readings"] == unsorted_parsed["readings"] == DASHBOARD_JSON["readings"]
    assert parsed["ratios"] == DASHBOARD_JSON["ratios"]


def test_without_sort_keys_dictionaries_keep_their_declared_and_insertion_order():
    parsed = to_json_value(make_dashboard())

    assert list(parsed) == [
        "title", "projects", "grid", "index", "readings", "primes", "ratios", "created", "opens", "blob", "meta",
    ]
    assert list(parsed["meta"]) == ["z", "שם", "big", "quote"]
    assert list(parsed["grid"]) == ["[10, -2]", "[0, 1]", "[0, 0]"]
    assert list(parsed["index"]) == ["beta", "alpha"]
    assert list(parsed["projects"][0]["tasks"][0]) == ["title", "priority", "due", "labels", "done", "note"]


def test_set_order_does_not_depend_on_insertion_order_or_hashes():
    words = ["pear", "apple", "fig", "kiwi", "banana"]
    expected = '["apple", "banana", "fig", "kiwi", "pear"]'

    for ordering in (words, list(reversed(words)), sorted(words, key=len)):
        assert serialize_json(set(ordering)) == expected
        assert serialize_json(frozenset(ordering)) == expected
        assert serialize_json(set(ordering), sort_keys=True) == expected
        assert serialize_json(set(ordering), indent=1) == '[\n "apple",\n "banana",\n "fig",\n "kiwi",\n "pear"\n]'


@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_set_of_non_ascii_strings_is_ordered_by_its_escaped_text_limitation(ensure_ascii):
    words = {"z", "A", "אבג", "a"}

    # Documents current behavior, not a promise: the members are ordered by their canonical text, which always
    # escapes non-ASCII characters (whatever ensure_ascii is): "A" < "\u05d0\u05d1\u05d2" < "a" < "z", since the
    # backslash (0x5c) sorts between the capitals and the lowercase letters. What is a promise: the order is the
    # same for both values of ensure_ascii.
    text = serialize_json(words, ensure_ascii=ensure_ascii)
    assert json.loads(text) == ["A", "אבג", "a", "z"]
    assert json.loads(serialize_json(frozenset(words), sort_keys=True, ensure_ascii=ensure_ascii)) == [
        "A", "אבג", "a", "z",
    ]


@dataclass(frozen=True)
class Backwards:
    second: int
    first: int


def test_set_members_that_are_objects_always_have_sorted_keys_limitation():
    # Documents current behavior, not a promise: set members are ordered by their canonical (sorted-keys) text and the
    # output uses that canonical form, so the keys of the members are sorted even without sort_keys. A list keeps the
    # declared order.
    members = {Backwards(2, 1)}

    assert serialize_json(members) == '[{"first": 1, "second": 2}]'
    assert serialize_json(list(members)) == '[{"second": 2, "first": 1}]'


@pytest.mark.parametrize("separators", SEPARATORS, ids=SEPARATOR_IDS.get)
@pytest.mark.parametrize("indent", INDENTS, ids=INDENT_IDS.get)
def test_ensure_ascii_escapes_or_keeps_non_ascii_characters(indent, separators):
    dashboard = make_dashboard()

    escaped = serialize_json(dashboard, indent=indent, separators=separators, ensure_ascii=True)
    unicode_text = serialize_json(dashboard, indent=indent, separators=separators, ensure_ascii=False)

    assert escaped.isascii()
    assert "Dashboard \\u05e9\\u05dc\\u05d5\\u05dd \\ud83d\\ude00" in escaped  # the emoji as a surrogate pair
    assert '"\\u05e9\\u05dd"' in escaped  # the Hebrew key
    assert "\\u2028" in escaped
    assert "\\u0001" in escaped  # control characters are escaped either way

    assert not unicode_text.isascii()
    assert "Dashboard שלום 😀" in unicode_text
    # json.dumps leaves U+2028 (a line separator in JavaScript) as it is when ensure_ascii is False
    assert '"שם"' in unicode_text
    assert "\u2028" in unicode_text
    assert "\\u0001" in unicode_text
    assert "\\u05e9" not in unicode_text

    assert_same_json(json.loads(escaped), DASHBOARD_JSON)
    assert_same_json(json.loads(unicode_text), DASHBOARD_JSON)


@dataclass
class Mini:
    name: str
    tags: frozenset[Label]
    when: datetime
    grid: dict[tuple[int, int], float]
    extra: dict[str, dict[str, int]]
    empty: list[int]


def make_mini() -> Mini:
    return Mini(
        "שלום",
        frozenset({Label.BETA, Label.ALPHA}),
        datetime(2024, 2, 29, 12, 30),
        {(1, 2): 0.5},
        {"b": {"y": 2, "x": 1}, "a": {}},
        [],
    )


# The text of `make_mini()` with indent: (depth, line) with the default separators
MINI_LINES = [
    (0, "{"),
    (1, '"name": "\\u05e9\\u05dc\\u05d5\\u05dd",'),
    (1, '"tags": ['),
    (2, '"alpha",'),
    (2, '"beta"'),
    (1, "],"),
    (1, '"when": "2024-02-29T12:30:00",'),
    (1, '"grid": {'),
    (2, '"[1, 2]": 0.5'),
    (1, "},"),
    (1, '"extra": {'),
    (2, '"b": {'),
    (3, '"y": 2,'),
    (3, '"x": 1'),
    (2, "},"),
    (2, '"a": {}'),
    (1, "},"),
    (1, '"empty": []'),
    (0, "}"),
]

MINI_COMPACT_LINE = (
    '{"name": "\\u05e9\\u05dc\\u05d5\\u05dd", "tags": ["alpha", "beta"], "when": "2024-02-29T12:30:00", '
    '"grid": {"[1, 2]": 0.5}, "extra": {"b": {"y": 2, "x": 1}, "a": {}}, "empty": []}'
)


def test_default_text_is_one_line_with_default_separators():
    assert serialize_json(make_mini()) == MINI_COMPACT_LINE
    assert serialize_json(make_mini(), indent=None) == MINI_COMPACT_LINE


@pytest.mark.parametrize(
    ("indent", "unit"),
    [
        pytest.param(0, "", id="zero"),
        pytest.param(1, " ", id="one"),
        pytest.param(2, "  ", id="two"),
        pytest.param(4, "    ", id="four"),
        pytest.param("\t", "\t", id="tab"),
        pytest.param("--", "--", id="string"),
    ],
)
def test_indent_puts_every_value_on_its_own_line(indent, unit):
    expected = "\n".join(unit * depth + line for depth, line in MINI_LINES)

    assert serialize_json(make_mini(), indent=indent) == expected


def test_indent_text_has_no_trailing_newline_and_parses_to_the_default_data():
    for indent in (0, 2, "\t"):
        text = serialize_json(make_mini(), indent=indent)
        assert not text.endswith("\n")
        assert json.loads(text) == json.loads(serialize_json(make_mini()))


def test_compact_separators_on_one_line():
    assert serialize_json(make_mini(), separators=(",", ":")) == (
        # the tuple key keeps its own text: options change the layout, not the data
        '{"name":"\\u05e9\\u05dc\\u05d5\\u05dd","tags":["alpha","beta"],"when":"2024-02-29T12:30:00",'
        '"grid":{"[1, 2]":0.5},"extra":{"b":{"y":2,"x":1},"a":{}},"empty":[]}'
    )


def test_spaced_separators_equal_the_default_without_indent():
    assert serialize_json(make_mini(), separators=(", ", ": ")) == MINI_COMPACT_LINE


def test_separators_combined_with_indent():
    value = {"a": [1], "b": {}}

    assert serialize_json(value, indent=1, separators=(", ", ": ")) == '{\n "a": [\n  1\n ], \n "b": {}\n}'
    assert serialize_json(value, indent=2, separators=(",", ":")) == '{\n  "a":[\n    1\n  ],\n  "b":{}\n}'
    assert serialize_json(value, indent=2, separators=(" ,", " : ")) == '{\n  "a" : [\n    1\n  ] ,\n  "b" : {}\n}'


def test_sort_keys_orders_the_converted_keys_of_mixed_types():
    # json.dumps cannot sort keys of mixed types; the keys are texts by then
    value = {10: 1, 9: 3, "x": 7, False: 6, Label.BETA: 5, (1, 2): 4, 2.5: 2}

    assert serialize_json(value) == '{"10": 1, "9": 3, "x": 7, "false": 6, "beta": 5, "[1, 2]": 4, "2.5": 2}'
    assert serialize_json(value, sort_keys=True) == (
        '{"10": 1, "2.5": 2, "9": 3, "[1, 2]": 4, "beta": 5, "false": 6, "x": 7}'
    )


def test_sort_keys_golden_on_the_small_object():
    assert serialize_json(make_mini(), sort_keys=True) == (
        '{"empty": [], "extra": {"a": {}, "b": {"x": 1, "y": 2}}, "grid": {"[1, 2]": 0.5}, '
        '"name": "\\u05e9\\u05dc\\u05d5\\u05dd", "tags": ["alpha", "beta"], "when": "2024-02-29T12:30:00"}'
    )


def test_ensure_ascii_false_golden_on_the_small_object():
    assert serialize_json(make_mini(), ensure_ascii=False) == (
        '{"name": "שלום", "tags": ["alpha", "beta"], "when": "2024-02-29T12:30:00", "grid": {"[1, 2]": 0.5}, '
        '"extra": {"b": {"y": 2, "x": 1}, "a": {}}, "empty": []}'
    )


def test_ensure_ascii_golden_for_every_kind_of_character():
    value = {"hebrew": "שלום", "emoji": "😀", "plain": 'a"b\\c\n\t\x01/'}

    assert serialize_json(value) == (
        r'{"hebrew": "\u05e9\u05dc\u05d5\u05dd", "emoji": "\ud83d\ude00", "plain": "a\"b\\c\n\t\u0001/"}'
    )
    assert serialize_json(value, ensure_ascii=False) == (
        '{"hebrew": "שלום", "emoji": "😀", "plain": "a\\"b\\\\c\\n\\t\\u0001/"}'
    )


@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_non_ascii_enum_values_enum_keys_paths_and_fallback_results_follow_ensure_ascii(ensure_ascii):
    value = {
        "by_greeting": {Greeting.HELLO: [Greeting.WAVE], Greeting.WAVE: [Greeting.HELLO]},
        "path": PurePosixPath("תיקייה/קובץ.txt"),
        "greetings": {Greeting.WAVE, Greeting.HELLO},
        "converted": Opaque("אבג 🙂"),
    }
    expected = {
        "by_greeting": {"שלום": ["👋"], "👋": ["שלום"]},
        "path": "תיקייה/קובץ.txt",
        # ordered by the escaped text: "\\u05e9..." < "\\ud83d...", whatever ensure_ascii is
        "greetings": ["שלום", "👋"],
        "converted": "אבג 🙂",
    }

    text = serialize_json(value, ensure_ascii=ensure_ascii, fallback=lambda opaque: opaque.tag)

    assert text == json.dumps(expected, ensure_ascii=ensure_ascii)
    assert text.isascii() is ensure_ascii
    assert_round_trip(value["by_greeting"], dict[Greeting, list[Greeting]], ensure_ascii=ensure_ascii)
    assert_round_trip(value["greetings"], set[Greeting], ensure_ascii=ensure_ascii)


def test_tuple_key_text_does_not_depend_on_the_encoding_options_limitation():
    # Documents current behavior, not a promise: the exact text of a tuple key is not specified. What is pinned is that
    # it is the same for every option, as the options change the layout of the text and never the data. (The round
    # trips below hold whatever the text is.)
    value = {("שלום", 1): "v"}
    keys = {
        tuple(json.loads(serialize_json(value, **options)))
        for options in (
            {},
            {"ensure_ascii": False},
            {"separators": (",", ":")},
            {"indent": 2, "sort_keys": True},
        )
    }

    # The JSON text of the tuple, written the same way for every option (escapes included)
    assert keys == {('["\\u05e9\\u05dc\\u05d5\\u05dd", 1]',)}
    assert_round_trip(value, dict[tuple[str, int], str], ensure_ascii=False, separators=(",", ":"))
    assert_round_trip(value, dict[tuple[str, int], str], indent=2, sort_keys=True)


# allow_nan

NON_FINITE = [
    pytest.param(float("nan"), "NaN", id="nan"),
    pytest.param(float("inf"), "Infinity", id="inf"),
    pytest.param(float("-inf"), "-Infinity", id="negative-inf"),
]


@dataclass
class Sample:
    label: str
    values: list[float]


NON_FINITE_POSITIONS = [
    pytest.param(lambda x: [1.0, x], "[1.0, TOKEN]", id="list"),
    pytest.param(lambda x: {"a": {"b": x}}, '{"a": {"b": TOKEN}}', id="dict-value"),
    pytest.param(lambda x: Sample("s", [0.5, x]), '{"label": "s", "values": [0.5, TOKEN]}', id="dataclass-field"),
    pytest.param(lambda x: (1.0, x), "[1.0, TOKEN]", id="tuple-item"),
    pytest.param(lambda x: frozenset({x}), "[TOKEN]", id="set-member"),
    pytest.param(lambda x: {x: 1}, '{"TOKEN": 1}', id="float-mapping-key"),
    pytest.param(lambda x: {(1, x): 1}, '{"[1, TOKEN]": 1}', id="tuple-mapping-key"),
    pytest.param(lambda x: Carrier(x), '{"carried": [TOKEN]}', id="jsonable-result"),
    pytest.param(
        lambda x: [{"a": [Sample("s", [x])]}],
        '[{"a": [{"label": "s", "values": [TOKEN]}]}]',
        id="deeply-nested",
    ),
    pytest.param(lambda x: {"k": Opaque(x)}, '{"k": [TOKEN]}', id="fallback-result"),
]


@pytest.mark.parametrize(("make", "template"), NON_FINITE_POSITIONS)
@pytest.mark.parametrize(("number", "token"), NON_FINITE)
def test_non_finite_floats_are_written_as_javascript_tokens_by_default(make, template, number, token):
    text = serialize_json(make(number), fallback=lambda value: [value.tag])

    assert text == template.replace("TOKEN", token)


@pytest.mark.parametrize(("make", "template"), NON_FINITE_POSITIONS)
@pytest.mark.parametrize(("number", "token"), NON_FINITE)
def test_allow_nan_false_rejects_a_non_finite_float_at_any_depth(make, template, number, token):
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        serialize_json(make(number), allow_nan=False, fallback=lambda value: [value.tag])


# a sample, every seventh of the 120 combinations: it still contains every value of every option
@pytest.mark.parametrize("options", OPTION_COMBINATIONS[::7])
def test_allow_nan_false_changes_nothing_when_all_floats_are_finite(options):
    dashboard = make_dashboard()

    assert serialize_json(dashboard, **{**options, "allow_nan": False}) == serialize_json(
        dashboard, **{**options, "allow_nan": True}
    )


def test_non_finite_floats_round_trip_by_default():
    values = [1.0, float("nan"), float("inf"), float("-inf"), -0.0]

    assert_round_trip(values, list[float])
    assert_round_trip(Sample("s", values), Sample)
    assert_round_trip({"k": values}, dict[str, list[float]], indent=2, sort_keys=True)


# ---------------------------------------------------------------------------------------------------------------------
# Options on user-defined objects
# ---------------------------------------------------------------------------------------------------------------------


def make_workspace():
    return {
        "zeta": make_atlas(),
        "origin": Point(0, 0),
        "booking": Booking(
            datetime(2024, 3, 1, 9, 30),
            Status.PAID,
            Decimal("19.90"),
            Customer("דנה", True),
            b"JSerPy",
            frozenset({Label.BETA, Label.ALPHA}),
        ),
        "ledger": Ledger({Label.BETA: 2, Label.ALPHA: 1}, {(2, 3): "y", (0, 1): "x"}),
        "device": Opaque("sensor-7"),
        "devices": {Opaque("b"), Opaque("a")},
    }


def convert_opaque(value):
    return {"tag": value.tag, "seen": datetime(2024, 1, 2, 3, 4, 5), "emoji": "😀"}


WORKSPACE_COMBINATIONS = [
    pytest.param(
        {"indent": indent, "sort_keys": sort_keys, "ensure_ascii": ensure_ascii},
        id=f"{INDENT_IDS[indent]}-{'sorted' if sort_keys else 'unsorted'}-{'ascii' if ensure_ascii else 'unicode'}",
    )
    for indent, sort_keys, ensure_ascii in itertools.product(INDENTS, [False, True], [True, False])
]


@pytest.mark.parametrize("options", WORKSPACE_COMBINATIONS)
def test_options_apply_to_jsonable_results_and_fallback_results(options):
    workspace = make_workspace()

    text = serialize_json(workspace, fallback=convert_opaque, **options)
    plain = serialize_json_as_obj(workspace, fallback=convert_opaque)

    assert text == json.dumps(plain, **options)
    parsed = json.loads(text)
    assert parsed == json.loads(serialize_json(workspace, fallback=convert_opaque))
    if options["sort_keys"]:
        assert_keys_sorted_everywhere(parsed)
    else:
        assert list(parsed) == ["zeta", "origin", "booking", "ledger", "device", "devices"]
        assert list(parsed["device"]) == ["tag", "seen", "emoji"]
        assert list(parsed["booking"]["customer"]) == ["name", "vip"]
    assert text.isascii() is options["ensure_ascii"]
    assert ("דנה" in text) is (not options["ensure_ascii"])
    assert [device["tag"] for device in parsed["devices"]] == ["a", "b"]


def test_workspace_json_shape_with_a_fallback():
    parsed = to_json_value(make_workspace(), fallback=convert_opaque)

    assert parsed["origin"] == {"x": 0, "y": 0}
    assert parsed["booking"]["customer"] == {"name": "דנה", "vip": True}
    assert parsed["ledger"] == {"by_label": {"beta": 2, "alpha": 1}, "by_cell": {"[2, 3]": "y", "[0, 1]": "x"}}
    assert parsed["device"] == {"tag": "sensor-7", "seen": "2024-01-02T03:04:05", "emoji": "😀"}
    assert [device["tag"] for device in parsed["devices"]] == ["a", "b"]
    assert len(parsed["zeta"]["trips"]) == 3


@pytest.mark.parametrize("options", WORKSPACE_COMBINATIONS)
@pytest.mark.parametrize(
    "make",
    [
        pytest.param(make_empty_dashboard, id="empty"),
        pytest.param(make_singleton_dashboard, id="singleton"),
        pytest.param(make_large_dashboard, id="large"),
    ],
)
def test_options_on_trivial_and_large_objects(make, options):
    dashboard = make()

    text = serialize_json(dashboard, **options)

    assert text == json.dumps(serialize_json_as_obj(dashboard), **options)
    assert_strictly_equal(json.loads(text), json.loads(serialize_json(dashboard)))
    assert_round_trip(dashboard, Dashboard, **options)
    if options["sort_keys"]:
        assert_keys_sorted_everywhere(json.loads(text))

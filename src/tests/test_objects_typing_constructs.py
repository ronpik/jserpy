"""Typing constructs as deserialization targets: Optional, Union, Literal, Any and the wrappers around them.

``deserialize_json(data, cls)`` is driven by the target type, so this module throws every kind of annotation at it
(typing and builtin spellings, PEP 604 unions, forward references), as the root target and nested in containers and
dataclasses. A union is first-match-wins: the first member that does not raise TypeError decides, so member order is
part of what is pinned. Each case is checked from both sides: the JSON shape of the Python value, and a strict round
trip back to the exact type.

All classes are defined at module level and annotations are strings (``from __future__ import annotations``), so
forward references to classes defined further down are part of the point.

Tests marked ``xfail(strict=True)`` assert what a user would expect and currently fail (library defects). Tests named
``*_limitation`` document current behavior of ambiguous or deliberately lenient corners and are not a promise.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import datetime
import enum
import json
import typing
from dataclasses import dataclass, make_dataclass
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import (
    Annotated,
    Any,
    ClassVar,
    Dict,
    Final,
    ForwardRef,
    FrozenSet,
    Generic,
    List,
    Literal,
    Mapping,
    MutableMapping,
    NamedTuple,
    NewType,
    Optional,
    Set,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
)

import pytest

from jserpy import deserialize_json, serialize_json
from objects_support import assert_round_trip, assert_strictly_equal, to_json_value

if typing.TYPE_CHECKING:
    from fractions import Fraction  # imported for type checkers only: not resolvable at runtime

NOON = datetime.datetime(2026, 10, 27, 12, 0)
DAY = datetime.date(2026, 10, 27)
SENTINEL = "TYPING_PAYLOAD_MUST_NOT_LEAK"


# --- models ---------------------------------------------------------------------------------------------------------


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Level(enum.IntEnum):
    LOW = 1
    HIGH = 2


class Mode(str, enum.Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass(frozen=True)
class Point:
    x: int
    y: int


# members told apart by a Literal discriminator
@dataclass(frozen=True)
class Cat:
    kind: Literal["cat"]
    lives: int


@dataclass(frozen=True)
class Dog:
    kind: Literal["dog"]
    good: bool


@dataclass(frozen=True)
class Bird:
    kind: Literal["bird"]
    wingspan: float | None


# the same shape; only the discriminator differs
@dataclass(frozen=True)
class Circle:
    kind: Literal["circle"]
    size: int


@dataclass(frozen=True)
class Square:
    kind: Literal["square"]
    size: int


# no discriminator, disjoint field names
@dataclass(frozen=True)
class Swimmer:
    fins: int


@dataclass(frozen=True)
class Flyer:
    wings: int


# no discriminator, identical shapes
@dataclass(frozen=True)
class Meters:
    value: float


@dataclass(frozen=True)
class Feet:
    value: float


# no discriminator, disjoint field names whose types refuse null: an absent field is passed on as None, so each of
# these rejects data that lacks its field (ByNum, whose int field takes anything, can only come last)
@dataclass(frozen=True)
class ByColor:
    color: Color


@dataclass(frozen=True)
class ByDate:
    when: datetime.date


@dataclass(frozen=True)
class ByTags:
    tags: list[str]


@dataclass(frozen=True)
class ByNum:
    num: int


def _kind_class(index: int) -> type:
    return make_dataclass(f"Kind{index}", [("kind", Literal[f"k{index}"]), ("n", int)], frozen=True)


KIND_CLASSES = [_kind_class(index) for index in range(12)]
WIDE_UNION = Union[tuple(KIND_CLASSES)]


@dataclass(frozen=True)
class Schedule:
    slot: tuple[datetime.date, datetime.date] | tuple[datetime.date, str]
    others: list[tuple[datetime.date, datetime.date] | tuple[datetime.date, str]]


# module-level aliases, referred to by name from string annotations
IntOrStr = Union[int, str]
Scores = dict[str, list[float]]
MaybePoints = Optional[list[Point]]
Palette = Literal["dark", "light"] | None


@dataclass(frozen=True)
class Aliased:
    ident: IntOrStr
    scores: Scores
    points: MaybePoints
    palette: Palette


class NamedPair(NamedTuple):
    when: datetime.date
    count: int


class Settings(TypedDict):
    when: datetime.date
    mode: Mode


@dataclass(frozen=True)
class Zoo:
    star: Cat | Dog | Bird
    runner_up: Optional[Cat | Dog]
    residents: list[Cat | Dog | Bird]
    by_name: dict[str, Cat | Dog | Bird]


@dataclass(frozen=True)
class Profile:
    name: str
    nickname: Optional[str]
    age: int | None
    tags: list[str] | None
    home: Optional[Path]
    joined: Optional[datetime.date]
    favorite: Optional[Color]


@dataclass(frozen=True)
class Envelope:
    ident: int | str
    payload: list[int] | dict[str, int] | None
    stamp: datetime.date | datetime.datetime | None
    ratio: Decimal | str
    levels: tuple[Level | Color | None, ...]


@dataclass(frozen=True)
class Job:
    mode: Literal["fast", "slow"]
    retries: Literal[0, 1, 3]
    dry_run: Literal[True]
    flag: Literal[0, 1]
    note: Literal["ok", "failed"] | None
    stages: tuple[Literal["build", "test"], ...]
    outcome: dict[str, Literal["pass", "fail"]]


@dataclass
class Document:
    title: str
    meta: Any
    items: list[Any]
    extra: dict[str, Any]
    maybe: Any | None


# deliberately spelled with the typing module's aliases
@dataclass
class TypingSpelled:
    names: List[str]
    scores: Dict[str, Tuple[int, ...]]
    pair: Tuple[int, Color]
    uniq: Set[int]
    frozen: FrozenSet[str]
    lookup: Mapping[str, List[Point]]
    maybe: Optional[List[Tuple[int, ...]]]
    either: Union[List[int], Dict[str, int]]


UserId = NewType("UserId", int)
Slug = NewType("Slug", str)


@dataclass(frozen=True)
class Account:
    uid: UserId
    slug: Slug
    friends: list[UserId]
    boss: Optional[UserId]


@dataclass(frozen=True)
class Measured:
    count: Annotated[int, "non-negative"]
    ratio: Annotated[float, "percent"]
    labels: Annotated[list[str], "ordered"]
    when: Annotated[datetime.date, "utc"]
    maybe: Annotated[Optional[Level], "optional"]
    nested: list[Annotated[Color, "element"]]


RealMeasured = make_dataclass(
    "RealMeasured", [("count", Annotated[int, "non-negative"]), ("when", Annotated[datetime.date, "utc"])]
)


T = TypeVar("T")


@dataclass(frozen=True)
class Box(Generic[T]):
    item: T
    items: list[T]


@dataclass(frozen=True)
class HoldsBox:
    box: Box[Point]


@dataclass(frozen=True)
class HoldsIntBox:
    box: Box[int]


@dataclass(frozen=True)
class WithClassVar:
    name: str
    version: ClassVar[int] = 3
    aliases: ClassVar[tuple[str, ...]] = ("a", "b")


@dataclass(frozen=True)
class WithFinal:
    name: str
    limit: Final[int] = 10


@dataclass(frozen=True)
class WithUnimportable:
    name: str
    ratio: Fraction


# a Library refers to Book and Author, which are defined further down
@dataclass(frozen=True)
class Library:
    name: str
    featured: Optional[Book]
    shelf: list[Book]
    by_isbn: dict[str, Book] | None
    pair: tuple[Book, Optional[Book]]
    quoted: "list[Book]"
    spelled: List["Book"]
    optional_quoted: Optional["Book"]


@dataclass(frozen=True)
class Book:
    title: str
    year: int
    author: Optional[Author]


@dataclass(frozen=True)
class Author:
    name: str
    born: datetime.date | None
    books: tuple[Book, ...]


# quotes inside builtin generics are resolved by typing.get_type_hints only since Python 3.11
@dataclass(frozen=True)
class InnerQuoted:
    books: list["Book"]
    by_title: dict[str, "Book"]
    trio: tuple["Book", ...]
    maybe: Optional[list["Book"]]


@dataclass
class TreeNode:
    value: int
    children: list[TreeNode]
    label: Optional[str]


@dataclass
class Link:
    value: int
    next: Link | None


# refers to itself through a dict and through a tuple of optionals
@dataclass
class Folder:
    name: str
    subfolders: dict[str, Folder]
    files: frozenset[str]
    pinned: tuple[Folder | None, ...]


@dataclass(frozen=True)
class Sensor:
    ident: str
    kind: Literal["temp", "humidity"]
    readings: tuple[float | None, ...]
    last_seen: datetime.datetime | None
    calibration: dict[str, Decimal] | None


@dataclass(frozen=True)
class Site:
    name: str
    sensors: dict[str, list[Sensor]]
    tags: frozenset[str]
    extra: Any
    parent: Optional[Site]


# --- helpers --------------------------------------------------------------------------------------------------------


def _annotation_id(annotation: object) -> str:
    if isinstance(annotation, type) and not typing.get_args(annotation):
        return annotation.__name__
    return repr(annotation)


def round_trip_as(value: Any, annotation: Any) -> Any:
    """Serialize ``value`` to JSON and deserialize it as ``annotation`` without any strict comparison."""
    return deserialize_json(to_json_value(value), annotation)


def _json_containers(value: Any) -> list[Any]:
    """Every dict and list reachable from ``value`` through dicts, lists, tuples, sets and dataclass fields."""
    found = []
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, (dict, list)):
            found.append(current)
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple, set, frozenset)):
            stack.extend(current)
        elif dataclasses.is_dataclass(current) and not isinstance(current, type):
            stack.extend(getattr(current, field.name) for field in dataclasses.fields(current))
    return found


def assert_no_shared_containers(result: Any, source: Any) -> None:
    source_ids = {id(container) for container in _json_containers(source)}
    shared = [container for container in _json_containers(result) if id(container) in source_ids]
    assert not shared, f"{len(shared)} mutable container(s) of the result are the input's own objects"


def _shared_in_dict_value() -> tuple[Any, Any]:
    shared = [1, 2]
    return {"b": shared}, {"a": shared}


def _shared_in_nested_list() -> tuple[Any, Any]:
    shared = [1, 2]
    return [shared], {"a": [shared]}


def _shared_in_dataclass_field() -> tuple[Any, Any]:
    shared = [1, 2]
    return Document("t", shared, [], {}, None), [shared]


def _shared_inside_tuple() -> tuple[Any, Any]:
    shared = {"k": 1}
    return {"x": (shared,)}, {"y": shared}


@pytest.mark.parametrize(
    "build",
    [_shared_in_dict_value, _shared_in_nested_list, _shared_in_dataclass_field, _shared_inside_tuple],
    ids=["dict-value", "list-in-list", "dataclass-field", "inside-tuple"],
)
def test_aliasing_helper_detects_a_shared_container(build):
    result, source = build()

    with pytest.raises(AssertionError, match=r"input's own objects"):
        assert_no_shared_containers(result, source)


def test_aliasing_helper_accepts_equal_but_distinct_containers():
    assert_no_shared_containers({"a": [1, {"b": []}]}, {"a": [1, {"b": []}]})


def _pets() -> list[Cat | Dog | Bird]:
    return [Dog("dog", True), Cat("cat", 9), Bird("bird", None), Cat("cat", 1), Bird("bird", 1.5)]


PET_JSON = [
    {"kind": "dog", "good": True},
    {"kind": "cat", "lives": 9},
    {"kind": "bird", "wingspan": None},
    {"kind": "cat", "lives": 1},
    {"kind": "bird", "wingspan": 1.5},
]


# --- Optional[X] and X | None ---------------------------------------------------------------------------------------

OPTIONAL_CASES = [
    # (base annotation, a non-null value, its JSON text)
    pytest.param(int, 5, "5", id="int"),
    pytest.param(int, 0, "0", id="int-zero"),
    pytest.param(str, "", '""', id="empty-str"),
    pytest.param(bool, False, "false", id="bool-false"),
    pytest.param(float, 0.5, "0.5", id="float"),
    pytest.param(float, 0.0, "0.0", id="float-zero"),
    pytest.param(float, -0.0, "-0.0", id="float-negative-zero"),
    pytest.param(float, float("inf"), "Infinity", id="float-infinity"),
    pytest.param(float, float("nan"), "NaN", id="float-nan"),
    pytest.param(int, 10**30, "1000000000000000000000000000000", id="int-beyond-64-bits"),
    pytest.param(Color, Color.GREEN, '"green"', id="enum"),
    pytest.param(Level, Level.HIGH, "2", id="int-enum"),
    pytest.param(datetime.datetime, NOON, '"2026-10-27T12:00:00"', id="datetime"),
    pytest.param(datetime.date, DAY, '"2026-10-27"', id="date"),
    pytest.param(datetime.time, datetime.time(8, 30), '"08:30:00"', id="time"),
    pytest.param(Decimal, Decimal("1.50"), '"1.50"', id="decimal"),
    pytest.param(bytes, b"hi", '"aGk="', id="bytes"),
    pytest.param(Path, Path("report.txt"), '"report.txt"', id="path"),
    pytest.param(list[int], [1, 2], "[1, 2]", id="list"),
    pytest.param(list[int], [], "[]", id="empty-list"),
    pytest.param(tuple[int, ...], (1, 2), "[1, 2]", id="tuple"),
    pytest.param(tuple[int, str], (1, "a"), '[1, "a"]', id="pair"),
    pytest.param(set[int], {3, 1, 2}, "[1, 2, 3]", id="set"),
    pytest.param(frozenset[str], frozenset({"b", "a"}), '["a", "b"]', id="frozenset"),
    pytest.param(dict[str, int], {"a": 1}, '{"a": 1}', id="dict"),
    pytest.param(dict[str, int], {}, "{}", id="empty-dict"),
    pytest.param(Point, Point(1, 2), '{"x": 1, "y": 2}', id="dataclass"),
]

OPTIONAL_SPELLINGS = {
    "Optional": lambda base: Optional[base],
    "PEP604": lambda base: base | None,
    "Union-None-first": lambda base: Union[None, base],
}


@pytest.mark.parametrize("spelling", OPTIONAL_SPELLINGS)
@pytest.mark.parametrize(("base", "value", "text"), OPTIONAL_CASES)
def test_optional_non_null_value_round_trips(spelling, base, value, text):
    annotation = OPTIONAL_SPELLINGS[spelling](base)

    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


@pytest.mark.parametrize("spelling", OPTIONAL_SPELLINGS)
@pytest.mark.parametrize("base", list({case.values[0]: None for case in OPTIONAL_CASES}), ids=_annotation_id)
def test_optional_null_round_trips_to_none(spelling, base):
    annotation = OPTIONAL_SPELLINGS[spelling](base)

    assert serialize_json(None) == "null"
    assert_round_trip(None, annotation)


NESTED_OPTIONALS = [
    pytest.param(Optional[Optional[int]], id="Optional[Optional[int]]"),
    pytest.param(Optional[Union[int, None]], id="Optional[Union[int, None]]"),
    pytest.param(Optional[int | None], id="Optional[int | None]"),
    pytest.param(Union[int, None, None], id="Union[int, None, None]"),
    pytest.param(Union[Optional[int], Optional[int]], id="Union[Optional[int], Optional[int]]"),
    pytest.param(int | None | None, id="int | None | None"),
]


@pytest.mark.parametrize("annotation", NESTED_OPTIONALS)
@pytest.mark.parametrize(("value", "text"), [(7, "7"), (None, "null"), (0, "0")], ids=["int", "null", "zero"])
def test_nested_optionals_flatten_to_one_optional(annotation, value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


@pytest.mark.parametrize(
    ("annotation", "data", "expected"),
    [
        pytest.param(Optional[float], 5, 5.0, id="Optional[float]"),
        pytest.param(float | None, 0, 0.0, id="float|None-zero"),
        pytest.param(Union[float, str], 5, 5.0, id="float|str"),
        pytest.param(Union[None, float], -3, -3.0, id="None|float-negative"),
        pytest.param(list[float | None], [1, None, 2.5], [1.0, None, 2.5], id="list-of-float|None"),
        pytest.param(dict[str, tuple[float, ...]], {"a": [1, 2]}, {"a": (1.0, 2.0)}, id="dict-of-float-tuples"),
        pytest.param(Optional[tuple[float, int]], [1, 2], (1.0, 2), id="Optional-pair-float-int"),
    ],
)
def test_whole_numbers_are_coerced_to_float_inside_unions_and_containers(annotation, data, expected):
    # JSON from non-Python encoders may write 1.0 as 1; the int must not come back as an int
    assert_strictly_equal(deserialize_json(data, annotation), expected)


def test_optional_dataclass_fields_hold_values():
    profile = Profile("Ann", "annie", 31, ["a", "b"], Path("ann.txt"), DAY, Color.RED)

    assert serialize_json(profile) == (
        '{"name": "Ann", "nickname": "annie", "age": 31, "tags": ["a", "b"], "home": "ann.txt", '
        '"joined": "2026-10-27", "favorite": "red"}'
    )
    assert_round_trip(profile, Profile)


def test_optional_dataclass_fields_hold_null():
    profile = Profile("Ann", None, None, None, None, None, None)

    assert serialize_json(profile) == (
        '{"name": "Ann", "nickname": null, "age": null, "tags": null, "home": null, "joined": null, '
        '"favorite": null}'
    )
    assert_round_trip(profile, Profile)


def test_optional_dataclass_fields_absent_from_the_data_become_none():
    restored = deserialize_json({"name": "Ann"}, Profile)

    assert_strictly_equal(restored, Profile("Ann", None, None, None, None, None, None))


def test_optional_dataclass_fields_ignore_extra_keys():
    data = {**json.loads(serialize_json(Profile("Ann", None, 5, [], None, None, None))), "surprise": [1, 2]}

    assert_strictly_equal(deserialize_json(data, Profile), Profile("Ann", None, 5, [], None, None, None))


def test_optional_empty_containers_are_not_confused_with_null():
    profile = Profile("", "", 0, [], None, None, None)

    restored = assert_round_trip(profile, Profile)

    assert restored.tags == [] and restored.nickname == "" and restored.age == 0


# --- explicit NoneType members --------------------------------------------------------------------------------------

NONE_TYPE_FORMS = [
    pytest.param(Union[int, type(None)], id="Union[int, NoneType]"),
    pytest.param(Union[type(None), int], id="Union[NoneType, int]"),
    pytest.param(int | type(None), id="int | NoneType"),
    pytest.param(type(None) | int, id="NoneType | int"),
    pytest.param(Optional[int], id="Optional[int]"),
]


@pytest.mark.parametrize("annotation", NONE_TYPE_FORMS)
@pytest.mark.parametrize(("value", "text"), [(5, "5"), (None, "null")], ids=["int", "null"])
def test_union_with_explicit_none_type_round_trips(annotation, value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


def test_none_type_alone_accepts_only_null():
    assert deserialize_json(None, type(None)) is None
    with pytest.raises(TypeError, match=r"Expected null, got int"):
        deserialize_json(5, type(None))


def test_none_type_inside_a_container_is_strict():
    assert_strictly_equal(deserialize_json([None, None], list[type(None)]), [None, None])
    with pytest.raises(TypeError, match=r"Expected null, got int"):
        deserialize_json([None, 1], list[type(None)])


@pytest.mark.parametrize(
    ("value", "text"), [([1, 2], "[1, 2]"), (None, "null"), ([], "[]")], ids=["list", "null", "empty-list"]
)
def test_none_type_first_in_a_union_with_a_container(value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, Union[type(None), list[int]])


# --- union member order: the first member that does not raise TypeError wins ------------------------------------------

UNION_DECODE_CASES = [
    # (annotation, JSON value, expected restored value)
    # a string is not a sequence: a list member rejects it, whichever the order
    pytest.param(Union[list[str], str], "abc", "abc", id="list|str-str"),
    pytest.param(Union[list[str], str], "", "", id="list|str-empty-str"),
    pytest.param(Union[list[str], str], ["a", "b"], ["a", "b"], id="list|str-list"),
    pytest.param(Union[str, list[str]], "abc", "abc", id="str|list-str"),
    pytest.param(Union[str, list[str]], ["a", "b"], ["a", "b"], id="str|list-list"),
    pytest.param(Union[tuple[str, ...], str], ["a"], ("a",), id="tuple|str-list"),
    pytest.param(Union[tuple[str, ...], str], "a", "a", id="tuple|str-str"),
    # enums: the invalid value falls through to the next member, or is claimed by str if that comes first
    pytest.param(Union[Color, str], "red", Color.RED, id="Color|str-valid"),
    pytest.param(Union[Color, str], "purple", "purple", id="Color|str-invalid"),
    pytest.param(Union[str, Color], "red", "red", id="str|Color-valid"),
    pytest.param(Union[Level, Color], 2, Level.HIGH, id="Level|Color-level"),
    pytest.param(Union[Level, Color], "blue", Color.BLUE, id="Level|Color-color"),
    pytest.param(Union[Mode, str], "fast", Mode.FAST, id="str-enum|str-valid"),
    pytest.param(Union[Mode, str], "warp", "warp", id="str-enum|str-invalid"),
    pytest.param(Union[Level, int], 2, Level.HIGH, id="Level|int-member"),
    pytest.param(Union[Level, int], 9, 9, id="Level|int-not-a-member"),
    pytest.param(Union[int, Level], 2, 2, id="int|Level-int-claims-it"),
    # primitives are never validated, so the first primitive member claims every primitive
    pytest.param(Union[int, str], 5, 5, id="int|str-int"),
    pytest.param(Union[int, str], "x", "x", id="int|str-str"),
    pytest.param(Union[str, int], 5, 5, id="str|int-int"),
    pytest.param(Union[str, int], "x", "x", id="str|int-str"),
    pytest.param(Union[int, bool], True, True, id="int|bool-bool"),
    pytest.param(Union[int, bool], 1, 1, id="int|bool-int"),
    pytest.param(Union[int, float], 1, 1, id="int|float-whole"),
    pytest.param(Union[int, float], 1.5, 1.5, id="int|float-fraction"),
    pytest.param(Union[float, int], 1, 1.0, id="float|int-whole-is-coerced"),
    pytest.param(Union[float, int], 1.5, 1.5, id="float|int-fraction"),
    # strings that look like something else
    pytest.param(Union[datetime.datetime, str], "2026-10-27T12:00:00", NOON, id="datetime|str-iso"),
    pytest.param(Union[datetime.datetime, str], "not a date", "not a date", id="datetime|str-text"),
    pytest.param(Union[str, datetime.datetime], "2026-10-27T12:00:00", "2026-10-27T12:00:00", id="str|datetime-iso"),
    pytest.param(Union[datetime.date, datetime.datetime], "2026-10-27", DAY, id="date|datetime-date-text"),
    pytest.param(
        Union[datetime.date, datetime.datetime], "2026-10-27T12:00:00", NOON, id="date|datetime-datetime-text"
    ),
    # datetime.fromisoformat accepts a bare date, so a leading datetime member turns dates into midnights
    pytest.param(
        Union[datetime.datetime, datetime.date],
        "2026-10-27",
        datetime.datetime(2026, 10, 27, 0, 0),
        id="datetime|date-date-text-becomes-datetime",
    ),
    pytest.param(Union[bytes, str], "aGk=", b"hi", id="bytes|str-base64"),
    pytest.param(Union[bytes, str], "hello", "hello", id="bytes|str-not-base64"),
    pytest.param(Union[str, bytes], "aGk=", "aGk=", id="str|bytes-base64"),
    pytest.param(Union[Decimal, str], "1.50", Decimal("1.50"), id="Decimal|str-number"),
    pytest.param(Union[Decimal, str], "n/a", "n/a", id="Decimal|str-text"),
    pytest.param(Union[str, Decimal], "1.50", "1.50", id="str|Decimal-number"),
    pytest.param(Union[Path, str], "report.txt", Path("report.txt"), id="Path|str"),
    pytest.param(Union[str, Path], "report.txt", "report.txt", id="str|Path"),
    pytest.param(Union[Path, int], 5, 5, id="Path|int-int"),
    # containers of different shapes
    pytest.param(Union[tuple[int, int], dict[str, int], list[int]], [1, 2], (1, 2), id="tuple|dict|list-pair"),
    pytest.param(Union[tuple[int, int], dict[str, int], list[int]], [1, 2, 3], [1, 2, 3], id="tuple|dict|list-triple"),
    pytest.param(Union[tuple[int, int], dict[str, int], list[int]], [], [], id="tuple|dict|list-empty"),
    pytest.param(Union[tuple[int, int], dict[str, int], list[int]], {"a": 1}, {"a": 1}, id="tuple|dict|list-object"),
    pytest.param(Union[list[int], dict[str, int], tuple[int, int]], [1, 2], [1, 2], id="list|dict|tuple-pair"),
    pytest.param(Union[list[Color], dict[str, Color]], ["red"], [Color.RED], id="list|dict-of-enum-list"),
    pytest.param(Union[list[Color], dict[str, Color]], {"k": "blue"}, {"k": Color.BLUE}, id="list|dict-of-enum-object"),
    pytest.param(Union[Point, list[Point]], {"x": 1, "y": 2}, Point(1, 2), id="one-or-many-one"),
    pytest.param(Union[Point, list[Point]], [{"x": 1, "y": 2}], [Point(1, 2)], id="one-or-many-many"),
    pytest.param(Union[list[Point], Point], {"x": 1, "y": 2}, Point(1, 2), id="many-or-one-one"),
    pytest.param(Union[list[Point], Point], [{"x": 1, "y": 2}], [Point(1, 2)], id="many-or-one-many"),
    pytest.param(Union[tuple[str, ...], frozenset[str]], ["b", "a"], ("b", "a"), id="tuple|frozenset-first-wins"),
    pytest.param(Union[frozenset[str], tuple[str, ...]], ["b", "a"], frozenset({"a", "b"}), id="frozenset|tuple"),
]


@pytest.mark.parametrize(("annotation", "data", "expected"), UNION_DECODE_CASES)
def test_union_selects_the_first_member_that_accepts(annotation, data, expected):
    assert_strictly_equal(deserialize_json(data, annotation), expected)


@pytest.mark.parametrize(("annotation", "data", "expected"), UNION_DECODE_CASES)
def test_union_decoding_agrees_with_the_pep_604_spelling(annotation, data, expected):
    members = typing.get_args(annotation)
    pep_604 = members[0]
    for member in members[1:]:
        pep_604 = pep_604 | member

    assert_strictly_equal(deserialize_json(data, pep_604), expected)


UNION_ROUND_TRIP_CASES = [
    # (annotation, value, JSON text); every value is restored to the very same type
    pytest.param(Union[list[str], str], "abc", '"abc"', id="list|str-str"),
    pytest.param(Union[list[str], str], ["a", "b"], '["a", "b"]', id="list|str-list"),
    pytest.param(Union[Color, str], Color.RED, '"red"', id="Color|str-enum"),
    pytest.param(Union[Color, str], "purple", '"purple"', id="Color|str-str"),
    pytest.param(Union[Level, Color], Level.LOW, "1", id="Level|Color-level"),
    pytest.param(Union[Level, Color], Color.GREEN, '"green"', id="Level|Color-color"),
    pytest.param(Union[Mode, str], Mode.SLOW, '"slow"', id="str-enum|str-enum"),
    pytest.param(Union[Mode, str], "warp", '"warp"', id="str-enum|str-str"),
    pytest.param(Union[Level, int], Level.LOW, "1", id="Level|int-enum"),
    pytest.param(Union[Level, int], 9, "9", id="Level|int-int"),
    pytest.param(Union[int, str], 5, "5", id="int|str-int"),
    pytest.param(Union[int, str], "5", '"5"', id="int|str-digit-str"),
    pytest.param(Union[str, int], 5, "5", id="str|int-int"),
    pytest.param(Union[int, bool], True, "true", id="int|bool-bool"),
    pytest.param(Union[bool, int], 1, "1", id="bool|int-int"),
    pytest.param(Union[bool, int], False, "false", id="bool|int-bool"),
    pytest.param(Union[float, int], 1.5, "1.5", id="float|int-float"),
    pytest.param(Union[datetime.datetime, str], NOON, '"2026-10-27T12:00:00"', id="datetime|str-datetime"),
    pytest.param(Union[datetime.datetime, str], "not a date", '"not a date"', id="datetime|str-str"),
    pytest.param(Union[datetime.date, datetime.datetime], DAY, '"2026-10-27"', id="date|datetime-date"),
    pytest.param(Union[datetime.date, datetime.datetime], NOON, '"2026-10-27T12:00:00"', id="date|datetime-datetime"),
    pytest.param(Union[bytes, str], b"hi", '"aGk="', id="bytes|str-bytes"),
    pytest.param(Union[bytes, str], "hello", '"hello"', id="bytes|str-str"),
    pytest.param(Union[Decimal, str], Decimal("1.50"), '"1.50"', id="Decimal|str-decimal"),
    pytest.param(Union[Decimal, str], "n/a", '"n/a"', id="Decimal|str-str"),
    pytest.param(Union[Path, str], Path("report.txt"), '"report.txt"', id="Path|str-path"),
    pytest.param(Union[tuple[int, int], list[int]], (1, 2), "[1, 2]", id="tuple|list-pair"),
    pytest.param(Union[tuple[int, int], list[int]], [1, 2, 3], "[1, 2, 3]", id="tuple|list-triple"),
    pytest.param(Union[tuple[int, int], dict[str, int], list[int]], {"a": 1}, '{"a": 1}', id="tuple|dict|list-dict"),
    pytest.param(Union[list[Color], dict[str, Color]], {"k": Color.BLUE}, '{"k": "blue"}', id="list|dict-of-enum"),
    pytest.param(Union[Point, list[Point]], Point(1, 2), '{"x": 1, "y": 2}', id="one-or-many-one"),
    pytest.param(Union[Point, list[Point]], [Point(1, 2)], '[{"x": 1, "y": 2}]', id="one-or-many-many"),
    pytest.param(Union[frozenset[str], tuple[str, ...]], frozenset({"b", "a"}), '["a", "b"]', id="frozenset|tuple"),
]


@pytest.mark.parametrize(("annotation", "value", "text"), UNION_ROUND_TRIP_CASES)
def test_union_values_round_trip_to_the_same_type(annotation, value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


@pytest.mark.parametrize(
    ("annotation", "data"),
    [
        pytest.param(Union[list[int], tuple[int, int]], "abc", id="sequences-str"),
        pytest.param(Union[list[int], tuple[int, int]], {"a": 1}, id="sequences-object"),
        pytest.param(Union[list[int], tuple[int, int]], 5, id="sequences-int"),
        pytest.param(Union[list[int], Literal["x"]], 5, id="list|literal-int"),
        pytest.param(Union[Color, Literal["x"]], "purple", id="enum|literal-str"),
        pytest.param(Union[Color, Level], [1], id="enums-list"),
        pytest.param(Union[Decimal, datetime.date], "n/a", id="decimal|date-text"),
        pytest.param(Union[bytes, Decimal], ["x"], id="bytes|decimal-list"),
        pytest.param(Union[Cat, Dog], [1], id="dataclasses-list"),
        pytest.param(Union[Cat, Dog], 5, id="dataclasses-int"),
        pytest.param(Union[Cat, Dog], "cat", id="dataclasses-str"),
        pytest.param(Union[Cat, Dog], None, id="dataclasses-null"),
        pytest.param(Union[tuple[int, int], tuple[int, int, int]], [1], id="tuples-wrong-length"),
    ],
)
def test_union_without_a_matching_member_raises_type_error(annotation, data):
    with pytest.raises(TypeError, match=rf"Unsupported types for deserialization.*got {type(data).__name__}"):
        deserialize_json(data, annotation)


def test_union_mismatch_message_does_not_leak_the_data():
    with pytest.raises(TypeError) as caught:
        deserialize_json(SENTINEL, Union[list[str], tuple[str, str]])

    assert SENTINEL not in str(caught.value)
    assert SENTINEL not in repr(caught.value)


# --- unions of dataclasses --------------------------------------------------------------------------------------------

PET_UNION_FORMS = [
    pytest.param(Union[Cat, Dog, Bird], id="Union"),
    pytest.param(Cat | Dog | Bird, id="PEP604"),
    pytest.param(Union[Bird, Dog, Cat], id="Union-reversed"),
    pytest.param(Optional[Union[Dog, Cat, Bird]], id="Optional-Union"),
]


@pytest.mark.parametrize("annotation", PET_UNION_FORMS)
@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(Cat("cat", 9), '{"kind": "cat", "lives": 9}', id="cat"),
        pytest.param(Dog("dog", True), '{"kind": "dog", "good": true}', id="dog"),
        pytest.param(Bird("bird", None), '{"kind": "bird", "wingspan": null}', id="bird-null"),
        pytest.param(Bird("bird", 1.5), '{"kind": "bird", "wingspan": 1.5}', id="bird"),
    ],
)
def test_union_discriminated_by_a_literal_field_restores_each_member(annotation, value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


@pytest.mark.parametrize(
    "annotation", [Union[Circle, Square], Union[Square, Circle]], ids=["circle-first", "square-first"]
)
@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(Circle("circle", 3), '{"kind": "circle", "size": 3}', id="circle"),
        pytest.param(Square("square", 4), '{"kind": "square", "size": 4}', id="square"),
    ],
)
def test_literal_discriminator_separates_otherwise_identical_shapes(annotation, value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"kind": "fish", "lives": 9}, id="unknown-discriminator"),
        pytest.param({"kind": "Cat", "lives": 9}, id="wrong-case"),
        pytest.param({"lives": 9}, id="missing-discriminator"),
        pytest.param({"kind": None, "lives": 9}, id="null-discriminator"),
        pytest.param({"kind": ["cat"]}, id="array-discriminator"),
    ],
)
def test_literal_discriminated_union_rejects_an_unknown_discriminator(data):
    with pytest.raises(TypeError, match=r"Unsupported types for deserialization"):
        deserialize_json(data, Union[Cat, Dog, Bird])


def test_discriminated_member_ignores_the_keys_of_its_siblings():
    data = {"kind": "dog", "good": True, "lives": 9, "wingspan": 2.0}

    assert_strictly_equal(deserialize_json(data, Union[Cat, Dog, Bird]), Dog("dog", True))


def test_mixed_list_of_discriminated_members_keeps_order_and_types():
    pets = _pets()

    assert serialize_json(pets) == json.dumps(PET_JSON)
    restored = assert_round_trip(pets, list[Cat | Dog | Bird])

    assert [type(pet) for pet in restored] == [Dog, Cat, Bird, Cat, Bird]


def test_discriminated_members_in_tuple_and_dict_containers():
    pets = _pets()
    as_tuple = tuple(pets[:3])
    as_dict = {"b": pets[0], "a": pets[1]}

    assert serialize_json(as_tuple) == json.dumps(PET_JSON[:3])
    assert serialize_json(as_dict) == json.dumps({"b": PET_JSON[0], "a": PET_JSON[1]})
    assert_round_trip(as_tuple, tuple[Cat | Dog | Bird, ...])
    assert_round_trip(as_dict, dict[str, Cat | Dog])


def test_zoo_of_discriminated_unions_in_dataclass_fields():
    pets = _pets()
    zoo = Zoo(
        star=pets[2],
        runner_up=pets[1],
        residents=pets,
        by_name={"rex": pets[0], "tom": pets[1], "tweety": pets[4]},
    )

    assert serialize_json(zoo) == json.dumps(
        {
            "star": PET_JSON[2],
            "runner_up": PET_JSON[1],
            "residents": PET_JSON,
            "by_name": {"rex": PET_JSON[0], "tom": PET_JSON[1], "tweety": PET_JSON[4]},
        }
    )
    assert_round_trip(zoo, Zoo)


def test_zoo_with_an_empty_optional_and_no_residents():
    zoo = Zoo(star=Dog("dog", False), runner_up=None, residents=[], by_name={})

    assert serialize_json(zoo) == (
        '{"star": {"kind": "dog", "good": false}, "runner_up": null, "residents": [], "by_name": {}}'
    )
    assert_round_trip(zoo, Zoo)


def test_zoo_rejects_a_member_with_an_unknown_discriminator():
    data = json.loads(serialize_json(Zoo(Cat("cat", 1), None, [Cat("cat", 2)], {})))
    data["residents"][0]["kind"] = "fish"

    with pytest.raises(TypeError, match=r"Unsupported types for deserialization"):
        deserialize_json(data, Zoo)


def test_union_of_identically_shaped_dataclasses_picks_the_first_member_limitation():
    # Documents current behavior, not a promise: with nothing to tell the members apart the first one wins, so a
    # Feet does not survive a round trip through Union[Meters, Feet].
    feet = Feet(2.5)

    assert serialize_json(feet) == '{"value": 2.5}'
    assert_strictly_equal(round_trip_as(feet, Union[Meters, Feet]), Meters(2.5))
    assert_strictly_equal(round_trip_as(feet, Union[Feet, Meters]), Feet(2.5))


def test_union_of_dataclasses_with_disjoint_field_names_picks_the_first_member_limitation():
    # Documents current behavior, not a promise: a field absent from the data becomes None (also for a required
    # field), so the first dataclass accepts any object and the data of the second one is silently dropped.
    flyer = Flyer(2)

    assert serialize_json(flyer) == '{"wings": 2}'
    assert_strictly_equal(round_trip_as(flyer, Union[Swimmer, Flyer]), Swimmer(None))
    assert_strictly_equal(round_trip_as(flyer, Union[Flyer, Swimmer]), flyer)


@pytest.mark.parametrize(
    ("annotation", "value", "text", "claimed_by_first_member"),
    [
        pytest.param(Union[bytes, str], "abcd", '"abcd"', b"i\xb7\x1d", id="bytes|str-base64-looking"),
        pytest.param(Union[Decimal, str], "1.50", '"1.50"', Decimal("1.50"), id="Decimal|str-number-looking"),
        pytest.param(
            Union[datetime.datetime, str],
            "2026-10-27",
            '"2026-10-27"',
            datetime.datetime(2026, 10, 27),
            id="datetime|str-date-looking",
        ),
        pytest.param(Union[Path, str], "report.txt", '"report.txt"', Path("report.txt"), id="Path|str-any-text"),
        pytest.param(Union[Color, str], "red", '"red"', Color.RED, id="Color|str-member-value"),
    ],
)
def test_str_value_that_looks_like_the_first_member_is_converted_limitation(
    annotation, value, text, claimed_by_first_member
):
    # Documents current behavior, not a promise: the JSON text of a str and of the first member is the same, so the
    # str value cannot come back as a str.
    assert serialize_json(value) == text
    assert_strictly_equal(round_trip_as(value, annotation), claimed_by_first_member)


# A primitive annotation (str, int, ...) hands a JSON object through unvalidated, so a primitive member in front of a
# dataclass or a typed dict claims every object: "id or embedded object" unions only work with the object first.
@pytest.mark.parametrize(
    ("annotation", "data"),
    [
        pytest.param(Union[str, Point], {"x": 1, "y": 2}, id="str|dataclass"),
        pytest.param(Union[int, Point], {"x": 1, "y": 2}, id="int|dataclass"),
        pytest.param(Union[float, Point], {"x": 1, "y": 2}, id="float|dataclass"),
        pytest.param(Union[str, dict[str, tuple[int, int]]], {"a": [1, 2]}, id="str|dict-of-tuples"),
    ],
)
def test_primitive_member_before_an_object_member_claims_the_object_limitation(annotation, data):
    # Documents current behavior, not a promise: the raw JSON object comes back (as an independent copy), unconverted.
    restored = deserialize_json(data, annotation)

    assert_strictly_equal(restored, data)
    assert_no_shared_containers(restored, data)


@pytest.mark.parametrize(
    ("annotation", "data", "expected"),
    [
        pytest.param(Union[Point, str], {"x": 1, "y": 2}, Point(1, 2), id="dataclass|str-object"),
        pytest.param(Union[Point, str], "p1", "p1", id="dataclass|str-id"),
        pytest.param(Union[Point, int], 7, 7, id="dataclass|int-id"),
        pytest.param(Union[dict[str, tuple[int, int]], list[int]], {"a": [1, 2]}, {"a": (1, 2)}, id="dict|list-object"),
    ],
)
def test_object_member_before_a_primitive_member_gives_id_or_object_unions(annotation, data, expected):
    assert_strictly_equal(deserialize_json(data, annotation), expected)


# A parameterized dict, unlike list/tuple/set/Mapping, takes a JSON scalar as is instead of raising TypeError, so it
# also claims the scalars that a later union member should get.
@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="dict[K, V] returns a JSON scalar unchanged instead of raising TypeError (Mapping[K, V] raises)",
)
@pytest.mark.parametrize("data", ["abc", 5, 1.5, True], ids=["str", "int", "float", "bool"])
def test_parameterized_dict_annotation_rejects_a_scalar(data):
    with pytest.raises(TypeError):
        deserialize_json(data, dict[str, int])


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="dict[K, V] returns a JSON scalar unchanged instead of raising TypeError, so it claims the str",
)
def test_parameterized_dict_member_does_not_claim_a_string_wanted_by_a_later_member():
    assert deserialize_json("red", Union[dict[str, int], Color]) is Color.RED


@pytest.mark.parametrize("data", ["abc", 5, 1.5, True], ids=["str", "int", "float", "bool"])
def test_mapping_annotation_rejects_a_scalar(data):
    with pytest.raises(TypeError, match=r"Expected a JSON object for .*, got " + type(data).__name__):
        deserialize_json(data, Mapping[str, int])


def test_mapping_member_before_a_scalar_member_selects_by_shape():
    # the strict spelling of the same union: Mapping raises TypeError for a scalar
    assert deserialize_json("red", Union[Mapping[str, int], Color]) is Color.RED
    assert deserialize_json({"a": 1}, Union[Mapping[str, int], Color]) == {"a": 1}


# --- unions of dataclasses told apart by their field names -----------------------------------------------------------


@pytest.mark.parametrize(
    "annotation",
    [
        Union[ByColor, ByDate, ByTags, ByNum],
        Union[ByTags, ByDate, ByColor, ByNum],
        Optional[Union[ByDate, ByTags, ByNum]],
    ],
    ids=["color-first", "tags-first", "optional-without-color"],
)
@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(ByDate(DAY), '{"when": "2026-10-27"}', id="date"),
        pytest.param(ByTags(["a", "b"]), '{"tags": ["a", "b"]}', id="tags"),
        pytest.param(ByTags([]), '{"tags": []}', id="empty-tags"),
        pytest.param(ByNum(1), '{"num": 1}', id="num"),
    ],
)
def test_union_of_dataclasses_with_disjoint_null_refusing_fields_restores_each_member(annotation, value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


def test_union_of_dataclasses_with_disjoint_fields_restores_the_enum_member():
    value = ByColor(Color.BLUE)

    assert serialize_json(value) == '{"color": "blue"}'
    assert_round_trip(value, Union[ByColor, ByDate, ByTags, ByNum])
    assert_round_trip([value, ByDate(DAY), ByNum(0)], list[ByColor | ByDate | ByNum])


# --- union backtracking: a member that fails after consuming part of the data hands the whole data on ----------------

DAY_AFTER = datetime.date(2026, 10, 28)

BACKTRACKING_CASES = [
    # (annotation, value, JSON text); the first member converts a prefix of the data, then fails
    pytest.param(
        Union[tuple[datetime.date, datetime.date], tuple[datetime.date, str]],
        (DAY, "soon"),
        '["2026-10-27", "soon"]',
        id="tuple-second-item-is-no-date",
    ),
    pytest.param(
        Union[tuple[datetime.date, datetime.date], tuple[datetime.date, str]],
        (DAY, DAY_AFTER),
        '["2026-10-27", "2026-10-28"]',
        id="tuple-both-dates",
    ),
    pytest.param(
        Union[dict[str, Decimal], dict[str, str]],
        {"x": "1.5", "y": "n/a"},
        '{"x": "1.5", "y": "n/a"}',
        id="dict-nothing-converted-leaks-into-the-fallback",
    ),
    pytest.param(
        Union[dict[str, Decimal], dict[str, str]],
        {"x": Decimal("1.5"), "y": Decimal("2")},
        '{"x": "1.5", "y": "2"}',
        id="dict-all-decimals",
    ),
    pytest.param(
        Union[tuple[Color, Level], tuple[Color, Color]],
        (Color.RED, Color.BLUE),
        '["red", "blue"]',
        id="tuple-second-position-is-a-color",
    ),
    pytest.param(
        Union[tuple[Color, Level], tuple[Color, Color]],
        (Color.RED, Level.LOW),
        '["red", 1]',
        id="tuple-second-position-is-a-level",
    ),
    pytest.param(
        Union[list[list[tuple[Color, Level]]], list[list[tuple[Color, Color]]]],
        [[(Color.RED, Color.BLUE)], [(Color.BLUE, Color.RED), (Color.GREEN, Color.GREEN)]],
        '[[["red", "blue"]], [["blue", "red"], ["green", "green"]]]',
        id="failure-three-levels-down",
    ),
    pytest.param(
        Union[list[list[tuple[Color, Level]]], list[list[tuple[Color, Color]]]],
        [[(Color.RED, Level.LOW)], []],
        '[[["red", 1]], []]',
        id="first-member-accepts-everything",
    ),
    pytest.param(
        list[Union[list[Color], list[Level]]],
        [[Color.RED, Color.BLUE], [Level.LOW, Level.HIGH], []],
        '[["red", "blue"], [1, 2], []]',
        id="each-element-picks-its-own-member",
    ),
    pytest.param(
        dict[str, Union[tuple[datetime.date, datetime.date], tuple[datetime.date, str]]],
        {"a": (DAY, "soon"), "b": (DAY, DAY_AFTER)},
        '{"a": ["2026-10-27", "soon"], "b": ["2026-10-27", "2026-10-28"]}',
        id="each-dict-value-picks-its-own-member",
    ),
    pytest.param(
        Union[dict[Color, int], dict[str, int]],
        {Color.RED: 1, Color.GREEN: 2},
        '{"red": 1, "green": 2}',
        id="enum-keys-restored",
    ),
    pytest.param(
        Union[dict[Color, int], dict[str, int]],
        {"red": 1, "purple": 2},
        '{"red": 1, "purple": 2}',
        id="one-key-is-no-color-so-all-stay-str",
    ),
]


@pytest.mark.parametrize(("annotation", "value", "text"), BACKTRACKING_CASES)
def test_union_backtracking_restores_the_member_that_accepts_all_the_data(annotation, value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


def test_union_backtracking_inside_dataclass_fields():
    value = Schedule(
        slot=(DAY, "soon"),
        others=[(DAY, DAY_AFTER), (DAY_AFTER, "later"), (DAY, DAY)],
    )

    assert serialize_json(value) == (
        '{"slot": ["2026-10-27", "soon"], "others": [["2026-10-27", "2026-10-28"], ["2026-10-28", "later"], '
        '["2026-10-27", "2026-10-27"]]}'
    )
    assert_round_trip(value, Schedule)


def test_union_failing_in_every_member_raises_and_leaves_the_library_usable():
    annotation = Union[list[list[tuple[Color, Level]]], list[list[tuple[Color, Color]]]]

    with pytest.raises(TypeError, match=r"Unsupported types for deserialization"):
        deserialize_json([[["red", "blue"]], [["red", "purple"]]], annotation)

    assert_strictly_equal(deserialize_json([[["red", "blue"]]], annotation), [[(Color.RED, Color.BLUE)]])


# --- a wide union of dataclasses --------------------------------------------------------------------------------------


@pytest.mark.parametrize("index", range(len(KIND_CLASSES)), ids=lambda index: f"member-{index}")
def test_wide_union_restores_every_member(index):
    value = KIND_CLASSES[index](f"k{index}", index * 10)

    assert serialize_json(value) == f'{{"kind": "k{index}", "n": {index * 10}}}'
    restored = assert_round_trip(value, WIDE_UNION)

    assert type(restored) is KIND_CLASSES[index]


def test_wide_union_in_a_mixed_list_and_as_optional():
    values = [KIND_CLASSES[index % 12](f"k{index % 12}", index) for index in range(40, 0, -1)]

    assert_round_trip(values, list[WIDE_UNION])
    assert_round_trip(values[0], Optional[WIDE_UNION])
    assert_round_trip(None, Optional[WIDE_UNION])


def test_wide_union_rejects_an_unknown_discriminator():
    with pytest.raises(TypeError, match=r"Unsupported types for deserialization"):
        deserialize_json({"kind": "k12", "n": 1}, WIDE_UNION)


# --- unions nested in containers and dataclass fields -----------------------------------------------------------------

NESTED_UNION_CASES = [
    # (annotation, value, JSON text)
    pytest.param(list[int | str | None], [1, "two", None, 3], '[1, "two", null, 3]', id="list-of-int|str|None"),
    pytest.param(list[int | str | None], [], "[]", id="empty-list-of-int|str|None"),
    pytest.param(list[int | str | None], [None, None], "[null, null]", id="list-of-nulls"),
    pytest.param(
        dict[str, Optional[list[Optional[int]]]],
        {"a": [1, None, 3], "b": None, "c": []},
        '{"a": [1, null, 3], "b": null, "c": []}',
        id="dict-of-optional-lists-of-optional-int",
    ),
    pytest.param(tuple[int | str, ...], (1, "a", 2), '[1, "a", 2]', id="homogeneous-tuple-of-int|str"),
    pytest.param(tuple[int | str, ...], (), "[]", id="empty-tuple-of-int|str"),
    pytest.param(tuple[int | None, str | None], (None, "x"), '[null, "x"]', id="pair-of-optionals"),
    pytest.param(list[list[int] | None], [[1], None, []], "[[1], null, []]", id="list-of-optional-lists"),
    pytest.param(dict[str, int] | list[str], {"a": 1}, '{"a": 1}', id="dict|list-dict"),
    pytest.param(dict[str, int] | list[str], ["a", "b"], '["a", "b"]', id="dict|list-list"),
    pytest.param(list[int] | None, [1], "[1]", id="list|None-list"),
    pytest.param(list[int] | None, None, "null", id="list|None-null"),
    pytest.param(
        list[Color | datetime.date | None],
        [Color.RED, DAY, None],
        '["red", "2026-10-27", null]',
        id="list-of-enum|date|None",
    ),
    pytest.param(
        list[Level | Color | None],
        [Level.HIGH, Color.BLUE, None],
        '[2, "blue", null]',
        id="list-of-int-enum|enum|None",
    ),
    pytest.param(
        dict[str, Decimal | Point | None],
        {"a": Decimal("2.50"), "b": Point(1, 2), "c": None},
        '{"a": "2.50", "b": {"x": 1, "y": 2}, "c": null}',
        id="dict-of-decimal|dataclass|None",
    ),
    pytest.param(
        Optional[tuple[Optional[tuple[int, ...]], ...]],
        ((1, 2), None, ()),
        "[[1, 2], null, []]",
        id="optional-tuple-of-optional-tuples",
    ),
    # enum keys, restored from the JSON keys, with union values
    pytest.param(
        dict[Color, Level | None],
        {Color.RED: Level.LOW, Color.BLUE: None},
        '{"red": 1, "blue": null}',
        id="enum-keys-int-enum|None-values",
    ),
    pytest.param(
        Optional[dict[Color, list[Point | None]]],
        {Color.GREEN: [Point(1, 2), None], Color.RED: []},
        '{"green": [{"x": 1, "y": 2}, null], "red": []}',
        id="optional-enum-keys-lists-of-optional-dataclasses",
    ),
    pytest.param(dict[Mode, int], {Mode.FAST: 1, Mode.SLOW: 2}, '{"fast": 1, "slow": 2}', id="str-enum-keys"),
    pytest.param(
        MappingProxyType[str, list[Color] | None],
        MappingProxyType({"a": [Color.RED], "b": None}),
        '{"a": ["red"], "b": null}',
        id="mapping-proxy-of-optional-lists",
    ),
    pytest.param(
        Optional[Mapping[Color, Point | None]],
        {Color.RED: None, Color.BLUE: Point(0, 0)},
        '{"red": null, "blue": {"x": 0, "y": 0}}',
        id="optional-Mapping-with-enum-keys",
    ),
    # sets are ordered by the canonical JSON text of their items, whatever the item types are
    pytest.param(set[int | str], {1, "1", 2}, '["1", 1, 2]', id="set-of-int|str"),
    pytest.param(
        frozenset[Level | Color], frozenset({Level.LOW, Color.RED}), '["red", 1]', id="frozenset-of-int-enum|enum"
    ),
    pytest.param(
        Optional[frozenset[Point | None]],
        frozenset({Point(1, 2), None}),
        '[null, {"x": 1, "y": 2}]',
        id="optional-frozenset-of-optional-dataclasses",
    ),
    pytest.param(set[int | None], set(), "[]", id="empty-set-of-int|None"),
]


@pytest.mark.parametrize(("annotation", "value", "text"), NESTED_UNION_CASES)
def test_unions_nested_in_containers_round_trip(annotation, value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


def test_union_fields_of_a_dataclass_hold_each_alternative():
    envelope = Envelope(
        ident="a-1",
        payload={"n": 1},
        stamp=NOON,
        ratio=Decimal("0.50"),
        levels=(Level.LOW, Color.GREEN, None),
    )

    assert serialize_json(envelope) == json.dumps(
        {
            "ident": "a-1",
            "payload": {"n": 1},
            "stamp": "2026-10-27T12:00:00",
            "ratio": "0.50",
            "levels": [1, "green", None],
        }
    )
    assert_round_trip(envelope, Envelope)


def test_union_fields_of_a_dataclass_hold_the_other_alternatives():
    envelope = Envelope(ident=7, payload=[1, 2], stamp=DAY, ratio="n/a", levels=())

    assert serialize_json(envelope) == json.dumps(
        {"ident": 7, "payload": [1, 2], "stamp": "2026-10-27", "ratio": "n/a", "levels": []}
    )
    assert_round_trip(envelope, Envelope)


def test_union_fields_of_a_dataclass_hold_null():
    envelope = Envelope(ident="", payload=None, stamp=None, ratio=Decimal("0"), levels=(None,))

    assert serialize_json(envelope) == json.dumps(
        {"ident": "", "payload": None, "stamp": None, "ratio": "0", "levels": [None]}
    )
    assert_round_trip(envelope, Envelope)


def test_large_union_list_round_trips():
    values = [index if index % 3 == 0 else (str(index) if index % 3 == 1 else None) for index in range(900)]

    assert serialize_json(values) == json.dumps(values)
    assert_round_trip(values, list[int | str | None])


def test_wide_dict_of_optional_lists_round_trips():
    value = {
        f"key{index:03d}": (None if index % 5 == 0 else [None if i % 2 else i for i in range(index % 4)])
        for index in range(300)
    }

    assert serialize_json(value) == json.dumps(value)
    assert_round_trip(value, dict[str, Optional[list[Optional[int]]]])


def test_wide_list_of_discriminated_members_round_trips():
    pets = [_pets()[index % 5] for index in range(300)]

    assert serialize_json(pets) == json.dumps([PET_JSON[index % 5] for index in range(300)])
    assert_round_trip(pets, list[Cat | Dog | Bird])


# --- Literal --------------------------------------------------------------------------------------------------------

LITERAL_ACCEPT_CASES = [
    # (annotation, value, JSON text)
    pytest.param(Literal["a", "b"], "a", '"a"', id="str-first"),
    pytest.param(Literal["a", "b"], "b", '"b"', id="str-second"),
    pytest.param(Literal[""], "", '""', id="empty-str"),
    pytest.param(Literal["é"], "é", '"\\u00e9"', id="non-ascii-str"),
    pytest.param(Literal[1, 2, 3], 2, "2", id="int"),
    pytest.param(Literal[0], 0, "0", id="zero"),
    pytest.param(Literal[-1], -1, "-1", id="negative"),
    pytest.param(Literal[True], True, "true", id="true"),
    pytest.param(Literal[False], False, "false", id="false"),
    pytest.param(Literal[True, False], False, "false", id="both-bools"),
    pytest.param(Literal[None], None, "null", id="none"),
    pytest.param(Literal["a", None], None, "null", id="str-or-none-none"),
    pytest.param(Literal["a", None], "a", '"a"', id="str-or-none-str"),
    pytest.param(Literal["a", 1, True, None], "a", '"a"', id="mixed-str"),
    pytest.param(Literal["a", 1, True, None], 1, "1", id="mixed-int"),
    pytest.param(Literal["a", 1, True, None], True, "true", id="mixed-bool"),
    pytest.param(Literal["a", 1, True, None], None, "null", id="mixed-none"),
    pytest.param(Literal[Literal["a", "b"], "c"], "c", '"c"', id="nested-literal-outer"),
    pytest.param(Literal[Literal["a", "b"], "c"], "a", '"a"', id="nested-literal-inner"),
]


@pytest.mark.parametrize(("annotation", "value", "text"), LITERAL_ACCEPT_CASES)
def test_literal_accepts_each_allowed_value_with_its_exact_type(annotation, value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


LITERAL_REJECT_CASES = [
    # bool and int are different literals although True == 1 and False == 0
    pytest.param(Literal[1], True, id="1-rejects-True"),
    pytest.param(Literal[0], False, id="0-rejects-False"),
    pytest.param(Literal[True], 1, id="True-rejects-1"),
    pytest.param(Literal[False], 0, id="False-rejects-0"),
    pytest.param(Literal[True, False], 1, id="both-bools-reject-1"),
    pytest.param(Literal[0, 1, 3], True, id="ints-reject-True"),
    pytest.param(Literal[1], 1.0, id="1-rejects-1.0"),
    pytest.param(Literal[2], "2", id="int-rejects-digit-str"),
    pytest.param(Literal["2"], 2, id="digit-str-rejects-int"),
    pytest.param(Literal["a"], "A", id="case-sensitive"),
    pytest.param(Literal["a", "b"], "c", id="other-str"),
    pytest.param(Literal["a", "b"], None, id="str-literal-rejects-null"),
    pytest.param(Literal[""], None, id="empty-str-rejects-null"),
    pytest.param(Literal["a"], ["a"], id="rejects-array"),
    pytest.param(Literal["a"], {"a": 1}, id="rejects-object"),
    pytest.param(Literal[1, 2], 3, id="other-int"),
    pytest.param(Literal[None], "null", id="none-rejects-the-word-null"),
    pytest.param(Literal[None], 0, id="none-rejects-zero"),
    pytest.param(Literal[None], False, id="none-rejects-false"),
    pytest.param(Literal["a", 1, True, None], False, id="mixed-rejects-False"),
    pytest.param(Literal["a", 1, True, None], 0, id="mixed-rejects-0"),
    pytest.param(Literal["a", 1, True, None], "1", id="mixed-rejects-digit-str"),
    pytest.param(Literal["a", 1, True, None], 1.0, id="mixed-rejects-float"),
]


@pytest.mark.parametrize(("annotation", "data"), LITERAL_REJECT_CASES)
def test_literal_rejects_a_look_alike_value(annotation, data):
    with pytest.raises(TypeError, match=rf"allowed literals .*, got {type(data).__name__}$"):
        deserialize_json(data, annotation)


@pytest.mark.parametrize(
    ("annotation", "value", "text"),
    [
        pytest.param(Optional[Literal["a", "b"]], "a", '"a"', id="Optional-str"),
        pytest.param(Optional[Literal["a", "b"]], None, "null", id="Optional-null"),
        pytest.param(Literal["a", "b"] | None, "b", '"b"', id="PEP604-str"),
        pytest.param(Literal["a", "b"] | None, None, "null", id="PEP604-null"),
        pytest.param(Optional[Literal[True]], True, "true", id="Optional-True"),
        pytest.param(Optional[Literal[0]], 0, "0", id="Optional-zero"),
        pytest.param(Literal["auto", "manual"] | Path, "auto", '"auto"', id="literal|Path-literal"),
        pytest.param(Literal["auto", "manual"] | Path, Path("other.txt"), '"other.txt"', id="literal|Path-path"),
        pytest.param(Literal["auto"] | list[int], "auto", '"auto"', id="literal|list-literal"),
        pytest.param(Literal["auto"] | list[int], [1, 2], "[1, 2]", id="literal|list-list"),
        pytest.param(Literal[1, 2] | Color, 2, "2", id="int-literal|enum-int"),
        pytest.param(Literal[1, 2] | Color, Color.RED, '"red"', id="int-literal|enum-enum"),
        pytest.param(Literal[True] | Level, True, "true", id="True-literal|int-enum-bool"),
        pytest.param(Literal[True] | Level, Level.LOW, "1", id="True-literal|int-enum-int"),
    ],
)
def test_literal_inside_a_union_round_trips(annotation, value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


@pytest.mark.parametrize(
    ("annotation", "value", "text"),
    [
        pytest.param(Literal["a"] | Literal["b"], "a", '"a"', id="two-literals-first"),
        pytest.param(Literal["a"] | Literal["b"], "b", '"b"', id="two-literals-second"),
        pytest.param(Union[Literal[1], Literal[True]], True, "true", id="1-or-True-takes-the-bool"),
        pytest.param(Union[Literal[1], Literal[True]], 1, "1", id="1-or-True-takes-the-int"),
        pytest.param(Union[Literal["a"], Literal["b"], None], None, "null", id="two-literals-or-None"),
        pytest.param(list[Literal["a"] | Literal["b"]], ["b", "a", "b"], '["b", "a", "b"]', id="list-of-two-literals"),
    ],
)
def test_union_of_separate_literals_round_trips(annotation, value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


@pytest.mark.parametrize(
    ("annotation", "data"),
    [
        pytest.param(Literal["a"] | Literal["b"], "c", id="two-literals-other"),
        pytest.param(Union[Literal[1], Literal[True]], 0, id="1-or-True-rejects-0"),
        pytest.param(Union[Literal[1], Literal[True]], 1.0, id="1-or-True-rejects-1.0"),
        pytest.param(Optional[Literal["a", "b"]], "c", id="Optional-other"),
        pytest.param(Optional[Literal[True]], 1, id="Optional-True-rejects-1"),
        pytest.param(Optional[Literal[0]], False, id="Optional-zero-rejects-False"),
        pytest.param(Literal["auto"] | list[int], "manual", id="literal|list-other-str"),
    ],
)
def test_literal_inside_a_union_rejects_values_of_no_member(annotation, data):
    with pytest.raises(TypeError, match=r"Unsupported types for deserialization"):
        deserialize_json(data, annotation)


def _job() -> Job:
    return Job(
        mode="fast",
        retries=3,
        dry_run=True,
        flag=0,
        note=None,
        stages=("build", "test"),
        outcome={"unit": "pass", "lint": "fail"},
    )


JOB_JSON = {
    "mode": "fast",
    "retries": 3,
    "dry_run": True,
    "flag": 0,
    "note": None,
    "stages": ["build", "test"],
    "outcome": {"unit": "pass", "lint": "fail"},
}


def test_literal_dataclass_fields_round_trip():
    job = _job()

    assert serialize_json(job) == json.dumps(JOB_JSON)
    restored = assert_round_trip(job, Job)

    assert type(restored.flag) is int and restored.dry_run is True and type(restored.stages) is tuple


def test_literal_dataclass_fields_with_every_alternative():
    job = Job("slow", 0, True, 1, "failed", (), {})

    assert serialize_json(job) == json.dumps(
        {
            "mode": "slow",
            "retries": 0,
            "dry_run": True,
            "flag": 1,
            "note": "failed",
            "stages": [],
            "outcome": {},
        }
    )
    assert_round_trip(job, Job)


LITERAL_MISMATCH = r"Expected one of the allowed literals .*, got "


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        pytest.param("mode", "medium", LITERAL_MISMATCH + "str", id="mode-other"),
        pytest.param("mode", None, LITERAL_MISMATCH + "NoneType", id="mode-null"),
        pytest.param("retries", 2, LITERAL_MISMATCH + "int", id="retries-other"),
        pytest.param("retries", "3", LITERAL_MISMATCH + "str", id="retries-digit-str"),
        pytest.param("retries", True, LITERAL_MISMATCH + "bool", id="retries-bool"),
        pytest.param("dry_run", 1, LITERAL_MISMATCH + "int", id="dry_run-int"),
        pytest.param("dry_run", False, LITERAL_MISMATCH + "bool", id="dry_run-false"),
        pytest.param("flag", True, LITERAL_MISMATCH + "bool", id="flag-bool"),
        pytest.param("flag", False, LITERAL_MISMATCH + "bool", id="flag-false"),
        pytest.param("flag", 2, LITERAL_MISMATCH + "int", id="flag-other"),
        pytest.param("note", "oops", r"Unsupported types for deserialization", id="note-other"),
        pytest.param("stages", ["deploy"], LITERAL_MISMATCH + "str", id="stages-other"),
        pytest.param("stages", ["build", "deploy"], LITERAL_MISMATCH + "str", id="stages-one-bad"),
        pytest.param("stages", "build", r"Expected a JSON array", id="stages-str-not-array"),
        pytest.param("outcome", {"unit": "skip"}, LITERAL_MISMATCH + "str", id="outcome-other"),
    ],
)
def test_literal_dataclass_field_rejects_an_invalid_value(field, bad, message):
    data = {**JOB_JSON, field: bad}

    with pytest.raises(TypeError, match=message):
        deserialize_json(data, Job)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        pytest.param("mode", LITERAL_MISMATCH + "NoneType", id="mode"),
        pytest.param("retries", LITERAL_MISMATCH + "NoneType", id="retries"),
        pytest.param("dry_run", LITERAL_MISMATCH + "NoneType", id="dry_run"),
        pytest.param("flag", LITERAL_MISMATCH + "NoneType", id="flag"),
        pytest.param("stages", r"Expected a JSON array", id="stages"),
    ],
)
def test_missing_literal_field_is_rejected_because_none_is_not_allowed(field, message):
    data = {key: value for key, value in JOB_JSON.items() if key != field}

    with pytest.raises(TypeError, match=message):
        deserialize_json(data, Job)


def test_missing_optional_literal_field_becomes_none():
    data = {key: value for key, value in JOB_JSON.items() if key != "note"}

    assert deserialize_json(data, Job).note is None


@pytest.mark.parametrize(
    ("annotation", "value", "text"),
    [
        pytest.param(list[Literal[0, 1]], [0, 1, 1, 0], "[0, 1, 1, 0]", id="int-flags"),
        pytest.param(list[Literal[True, False]], [True, False], "[true, false]", id="bool-flags"),
    ],
)
def test_literal_elements_keep_the_bool_int_distinction(annotation, value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


@pytest.mark.parametrize(
    ("annotation", "data"),
    [
        pytest.param(list[Literal[0, 1]], [0, True], id="int-flags-reject-True"),
        pytest.param(list[Literal[0, 1]], [False], id="int-flags-reject-False"),
        pytest.param(list[Literal[True, False]], [1], id="bool-flags-reject-1"),
        pytest.param(list[Literal[True, False]], [True, 0], id="bool-flags-reject-0"),
    ],
)
def test_literal_elements_reject_the_look_alike_of_the_other_kind(annotation, data):
    with pytest.raises(TypeError, match=LITERAL_MISMATCH):
        deserialize_json(data, annotation)


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(dict[Literal["a", "b"], int], id="dict-Literal-keys"),
        pytest.param(Mapping[Literal["a", "b"], int], id="Mapping-Literal-keys"),
        pytest.param(dict[Optional[str], int], id="dict-Optional-keys"),
        pytest.param(dict[str | int, int], id="dict-union-keys"),
    ],
)
def test_mapping_keys_annotated_with_a_literal_or_union_are_refused_limitation(annotation):
    # Documents current behavior, not a promise: JSON keys are restored only for str, Enum and tuple key annotations,
    # and any other form (including ones a str key would satisfy) fails with a TypeError raised by issubclass().
    assert serialize_json({"a": 1}) == '{"a": 1}'
    with pytest.raises(TypeError):
        deserialize_json({"a": 1}, annotation)


def test_literal_with_enum_members_cannot_restore_from_json_limitation():
    # Documents current behavior, not a promise: the enum serializes by value ("red"), but the Literal handler only
    # compares the JSON value with the enum members themselves, so its own output is rejected.
    annotation = Literal[Color.RED, Color.GREEN]

    assert serialize_json(Color.RED) == '"red"'
    with pytest.raises(TypeError, match=r"allowed literals"):
        deserialize_json("red", annotation)
    assert deserialize_json(Color.RED, annotation) is Color.RED  # an enum member passed in directly is accepted


# --- Any --------------------------------------------------------------------------------------------------------------

JSON_SAMPLE = {
    "id": 7,
    "ratio": 0.5,
    "ok": True,
    "off": False,
    "none": None,
    "zero": 0,
    "zero_float": 0.0,
    "text": "tête",
    "list": [1, "two", [3, [4, {"five": [5, None]}]], {}],
    "nested": {"a": {"b": {"c": [None, False, 0, 0.0, ""]}}},
    "empty": {},
    "empty_list": [],
}

JSON_LIST_SAMPLE = [1, "a", None, [2, {"b": [3, 4.5]}], {"c": {}}, True, [], 0.0]

ANY_OBJECT_FORMS = [
    pytest.param(Any, id="Any"),
    pytest.param(dict[str, Any], id="dict[str, Any]"),
    pytest.param(Dict[str, Any], id="Dict[str, Any]"),
    pytest.param(Mapping[str, Any], id="Mapping[str, Any]"),
    pytest.param(Optional[Any], id="Optional[Any]"),
    pytest.param(Any | None, id="Any | None"),
    pytest.param(Optional[dict[str, Any]], id="Optional[dict[str, Any]]"),
    pytest.param(dict[str, Any] | list[Any], id="dict | list of Any"),
    pytest.param(Union[Any, int], id="Union[Any, int]"),
    pytest.param(dict, id="dict"),
    pytest.param(Mapping[Any, Any], id="Mapping[Any, Any]"),
]

ANY_LIST_FORMS = [
    pytest.param(Any, id="Any"),
    pytest.param(list[Any], id="list[Any]"),
    pytest.param(List[Any], id="List[Any]"),
    pytest.param(Optional[list[Any]], id="Optional[list[Any]]"),
    pytest.param(list[Any] | None, id="list[Any] | None"),
    pytest.param(list, id="list"),
    pytest.param(List, id="List"),
    pytest.param(Optional[Any], id="Optional[Any]"),
]


@pytest.mark.parametrize("annotation", ANY_OBJECT_FORMS)
def test_any_object_round_trips_with_exact_scalar_types(annotation):
    assert serialize_json(JSON_SAMPLE) == json.dumps(JSON_SAMPLE)
    restored = assert_round_trip(JSON_SAMPLE, annotation)

    assert type(restored) is dict
    assert type(restored["zero_float"]) is float and type(restored["zero"]) is int and restored["off"] is False


@pytest.mark.parametrize("annotation", ANY_LIST_FORMS)
def test_any_list_round_trips_with_exact_scalar_types(annotation):
    assert serialize_json(JSON_LIST_SAMPLE) == json.dumps(JSON_LIST_SAMPLE)
    restored = assert_round_trip(JSON_LIST_SAMPLE, annotation)

    assert type(restored) is list


@pytest.mark.parametrize("annotation", ANY_OBJECT_FORMS)
def test_any_object_result_shares_no_container_with_the_input(annotation):
    parsed = json.loads(serialize_json(JSON_SAMPLE))

    restored = deserialize_json(parsed, annotation)

    assert_no_shared_containers(restored, parsed)
    parsed["list"][2][1][1]["five"].append("late")
    parsed["nested"]["a"]["b"]["c"].clear()
    parsed["empty"]["new"] = 1
    assert_strictly_equal(restored, JSON_SAMPLE)


@pytest.mark.parametrize("annotation", ANY_LIST_FORMS)
def test_any_list_result_shares_no_container_with_the_input(annotation):
    parsed = json.loads(serialize_json(JSON_LIST_SAMPLE))

    restored = deserialize_json(parsed, annotation)

    assert_no_shared_containers(restored, parsed)
    parsed[3][1]["b"].append("late")
    parsed[4]["c"]["new"] = 1
    assert_strictly_equal(restored, JSON_LIST_SAMPLE)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(5, "5", id="int"),
        pytest.param("s", '"s"', id="str"),
        pytest.param(None, "null", id="null"),
        pytest.param(True, "true", id="bool"),
        pytest.param(1.5, "1.5", id="float"),
        pytest.param([], "[]", id="empty-list"),
        pytest.param({}, "{}", id="empty-dict"),
        pytest.param([1, [2, [3, []]]], "[1, [2, [3, []]]]", id="nested-list"),
        pytest.param({"a": {"b": {"c": None}}}, '{"a": {"b": {"c": null}}}', id="nested-dict"),
    ],
)
def test_any_root_accepts_every_kind_of_json_value(value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, Any)


def _document() -> Document:
    return Document(
        title="t",
        meta={"deep": [1, {"k": [None, True, 2.5]}], "flag": False},
        items=[1, "a", None, {"z": []}, [[], {}]],
        extra={"a": {"b": {"c": {}}}, "list": [{"x": 1}]},
        maybe=None,
    )


DOCUMENT_JSON = {
    "title": "t",
    "meta": {"deep": [1, {"k": [None, True, 2.5]}], "flag": False},
    "items": [1, "a", None, {"z": []}, [[], {}]],
    "extra": {"a": {"b": {"c": {}}}, "list": [{"x": 1}]},
    "maybe": None,
}


def test_any_dataclass_fields_hold_deeply_nested_json():
    document = _document()

    assert serialize_json(document) == json.dumps(DOCUMENT_JSON)
    assert_round_trip(document, Document)


@pytest.mark.parametrize(
    ("any_value", "text"),
    [
        pytest.param(7, "7", id="int"),
        pytest.param("s", '"s"', id="str"),
        pytest.param([1, 2], "[1, 2]", id="list"),
        pytest.param({"k": None}, '{"k": null}', id="dict"),
        pytest.param(True, "true", id="bool"),
    ],
)
def test_any_dataclass_fields_hold_scalars_and_containers(any_value, text):
    document = Document("t", any_value, [], {}, any_value)

    assert serialize_json(document) == f'{{"title": "t", "meta": {text}, "items": [], "extra": {{}}, "maybe": {text}}}'
    assert_round_trip(document, Document)


def test_any_dataclass_result_shares_no_container_with_the_input():
    parsed = json.loads(serialize_json(_document()))

    restored = deserialize_json(parsed, Document)

    assert_no_shared_containers(restored, parsed)
    parsed["meta"]["deep"][1]["k"].append("late")
    parsed["items"][3]["z"].append(1)
    parsed["extra"]["a"]["b"]["c"]["new"] = 1
    parsed["extra"]["list"][0]["x"] = 2
    assert_strictly_equal(restored, _document())


def test_mutating_the_restored_any_value_leaves_the_input_untouched():
    parsed = json.loads(serialize_json(_document()))
    snapshot = json.dumps(parsed)

    restored = deserialize_json(parsed, Document)
    restored.meta["deep"][1]["k"].append("late")
    restored.extra["a"]["b"]["c"]["new"] = 1
    restored.items.append("late")

    assert json.dumps(parsed) == snapshot


def test_any_field_restores_json_forms_of_rich_values():
    # Any cannot know the original type: rich values serialize to their JSON forms and come back as exactly those.
    document = Document(
        title="t",
        meta={
            "when": NOON,
            "price": Decimal("1.50"),
            "color": Color.RED,
            "tags": {"b", "a"},
            "pair": (1, 2),
            "raw": b"hi",
            "file": Path("f.txt"),
        },
        items=[Decimal("2.50"), DAY, Level.HIGH],
        extra={"frozen": frozenset({"y", "x"})},
        maybe=Color.BLUE,
    )
    expected_json = {
        "title": "t",
        "meta": {
            "when": "2026-10-27T12:00:00",
            "price": "1.50",
            "color": "red",
            "tags": ["a", "b"],
            "pair": [1, 2],
            "raw": "aGk=",
            "file": "f.txt",
        },
        "items": ["2.50", "2026-10-27", 2],
        "extra": {"frozen": ["x", "y"]},
        "maybe": "blue",
    }

    assert serialize_json(document) == json.dumps(expected_json)
    restored = deserialize_json(to_json_value(document), Document)

    assert_strictly_equal(
        restored,
        Document("t", expected_json["meta"], expected_json["items"], expected_json["extra"], "blue"),
    )


@pytest.mark.parametrize("data", [{"a": [1, {"b": None}]}, 5, "s", None, True, 1.5], ids=repr)
def test_object_annotation_hands_objects_and_scalars_through_limitation(data):
    # Documents current behavior, not a promise: `object` has no handler, so it is treated like any unknown class
    # (a JSON object or scalar is handed through), but unlike Any it refuses an array.
    assert_strictly_equal(deserialize_json(data, object), data)


def test_object_annotation_refuses_arrays_limitation():
    # Documents current behavior, not a promise: see above; Any is the annotation that accepts every JSON value.
    with pytest.raises(TypeError, match=r"Unsupported type for deserialization: <class 'object'>"):
        deserialize_json([1, [2, {"a": None}]], object)
    assert_strictly_equal(deserialize_json([1, [2, {"a": None}]], Any), [1, [2, {"a": None}]])


# --- typing spellings versus builtin generics -------------------------------------------------------------------------

SPELLING_CASES = [
    # (typing spelling, builtin spelling, JSON value, expected)
    pytest.param(List[int], list[int], [1, 2], [1, 2], id="list"),
    pytest.param(
        List[Tuple[int, str]], list[tuple[int, str]], [[1, "a"], [2, "b"]], [(1, "a"), (2, "b")], id="list-of-pairs"
    ),
    pytest.param(Dict[str, int], dict[str, int], {"a": 1}, {"a": 1}, id="dict"),
    pytest.param(Dict[str, List[int]], dict[str, list[int]], {"a": [1, 2]}, {"a": [1, 2]}, id="dict-of-lists"),
    pytest.param(
        Dict[str, Tuple[int, ...]], dict[str, tuple[int, ...]], {"a": [1, 2]}, {"a": (1, 2)}, id="dict-of-tuples"
    ),
    pytest.param(Tuple[int, ...], tuple[int, ...], [1, 2, 3], (1, 2, 3), id="homogeneous-tuple"),
    pytest.param(Tuple[int, str, bool], tuple[int, str, bool], [1, "a", True], (1, "a", True), id="tuple"),
    pytest.param(Set[int], set[int], [3, 1], {1, 3}, id="set"),
    pytest.param(FrozenSet[str], frozenset[str], ["a"], frozenset({"a"}), id="frozenset"),
    pytest.param(
        Mapping[str, Tuple[int, int]],
        collections.abc.Mapping[str, tuple[int, int]],
        {"a": [1, 2]},
        {"a": (1, 2)},
        id="Mapping",
    ),
    pytest.param(Optional[List[int]], list[int] | None, [1], [1], id="optional-list"),
    pytest.param(Optional[List[int]], list[int] | None, None, None, id="optional-list-null"),
    pytest.param(
        Union[List[int], Dict[str, int]], list[int] | dict[str, int], {"a": 1}, {"a": 1}, id="list-or-dict"
    ),
    pytest.param(
        Dict[str, Optional[Set[int]]],
        dict[str, set[int] | None],
        {"a": [1], "b": None},
        {"a": {1}, "b": None},
        id="optional-set-values",
    ),
    pytest.param(
        List[Dict[str, Tuple[Color, ...]]],
        list[dict[str, tuple[Color, ...]]],
        [{"p": ["red", "blue"]}],
        [{"p": (Color.RED, Color.BLUE)}],
        id="deep",
    ),
]


@pytest.mark.parametrize(("typing_form", "builtin_form", "data", "expected"), SPELLING_CASES)
def test_typing_and_builtin_spellings_behave_alike(typing_form, builtin_form, data, expected):
    assert_strictly_equal(deserialize_json(data, typing_form), expected)
    assert_strictly_equal(deserialize_json(data, builtin_form), expected)


def test_typing_spelled_dataclass_round_trips():
    value = TypingSpelled(
        names=["a", "b"],
        scores={"s": (1, 2)},
        pair=(3, Color.RED),
        uniq={2, 1},
        frozen=frozenset({"y", "x"}),
        lookup={"home": [Point(1, 2)]},
        maybe=[(1,), ()],
        either={"k": 1},
    )

    assert serialize_json(value) == json.dumps(
        {
            "names": ["a", "b"],
            "scores": {"s": [1, 2]},
            "pair": [3, "red"],
            "uniq": [1, 2],
            "frozen": ["x", "y"],
            "lookup": {"home": [{"x": 1, "y": 2}]},
            "maybe": [[1], []],
            "either": {"k": 1},
        }
    )
    assert_round_trip(value, TypingSpelled)


def test_typing_spelled_dataclass_with_every_optional_empty():
    value = TypingSpelled([], {}, (0, Color.BLUE), set(), frozenset(), {}, None, [1])

    assert_round_trip(value, TypingSpelled)


BARE_CASES = [
    # (bare annotation, JSON value, expected); a bare container takes the JSON value as is, only the outer type changes
    pytest.param(List, [1, "a", [2]], [1, "a", [2]], id="typing.List"),
    pytest.param(Tuple, [1, "a"], (1, "a"), id="typing.Tuple"),
    pytest.param(Set, [1, 2], {1, 2}, id="typing.Set"),
    pytest.param(FrozenSet, [1, 2], frozenset({1, 2}), id="typing.FrozenSet"),
    pytest.param(Mapping, {"a": [1]}, {"a": [1]}, id="typing.Mapping"),
    pytest.param(list, [1, "a", [2]], [1, "a", [2]], id="list"),
    pytest.param(tuple, [1, "a"], (1, "a"), id="tuple"),
    pytest.param(tuple, [1, [2, 3]], (1, [2, 3]), id="tuple-keeps-inner-list"),
    pytest.param(set, ["a", "b"], {"a", "b"}, id="set"),
    pytest.param(frozenset, ["a", "b"], frozenset({"a", "b"}), id="frozenset"),
    pytest.param(dict, {"a": [1, {"b": None}]}, {"a": [1, {"b": None}]}, id="dict"),
    pytest.param(tuple, [], (), id="empty-tuple"),
    pytest.param(list, [], [], id="empty-list"),
    pytest.param(dict, {}, {}, id="empty-dict"),
]


@pytest.mark.parametrize(("annotation", "data", "expected"), BARE_CASES)
def test_bare_container_annotations(annotation, data, expected):
    assert_strictly_equal(deserialize_json(data, annotation), expected)


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="bare typing.Dict raises ValueError (unpacking its missing type arguments); bare dict and Mapping pass",
)
def test_bare_typing_dict_passes_the_object_through():
    assert_strictly_equal(deserialize_json({"a": [1]}, Dict), {"a": [1]})


@pytest.mark.xfail(
    strict=True,
    raises=(TypeError, ValueError),  # which one depends on the Python version: Any is a class from 3.11
    reason="dict[Any, V] raises (key restoration cannot handle Any) unlike Mapping[Any, V]",
)
@pytest.mark.parametrize(
    "annotation",
    [dict[Any, int], Dict[Any, int], Dict[Any, Any]],
    ids=["dict[Any, int]", "Dict[Any, int]", "Dict[Any, Any]"],
)
def test_dict_with_any_keys_keeps_the_json_keys(annotation):
    assert_strictly_equal(deserialize_json({"a": 1}, annotation), {"a": 1})


def test_mapping_with_any_keys_keeps_the_json_keys():
    assert_strictly_equal(deserialize_json({"a": 1}, Mapping[Any, int]), {"a": 1})


@pytest.mark.parametrize(
    "annotation",
    [
        MutableMapping[str, tuple[int, ...]],
        collections.abc.MutableMapping[str, tuple[int, ...]],
    ],
    ids=["typing", "abc"],
)
def test_mutable_mapping_annotation_leaves_the_values_unconverted_limitation(annotation):
    # Documents current behavior, not a promise: only the exact Mapping has a handler. Any other mapping annotation
    # (MutableMapping, UserDict, ChainMap, ...) hands the JSON object through as is, as a copy, with its list values.
    data = {"a": [1, 2]}

    restored = deserialize_json(data, annotation)

    assert_strictly_equal(restored, {"a": [1, 2]})
    assert_no_shared_containers(restored, data)
    assert_strictly_equal(deserialize_json(data, Mapping[str, tuple[int, ...]]), {"a": (1, 2)})  # what Mapping does


@pytest.mark.parametrize(
    "annotation",
    [
        typing.Sequence[int],
        typing.Sequence,
        collections.abc.Sequence[int],
        typing.Iterable[int],
        typing.Collection[int],
    ],
    ids=["typing.Sequence[int]", "typing.Sequence", "abc.Sequence[int]", "typing.Iterable[int]", "typing.Collection"],
)
def test_abstract_sequence_annotations_are_unsupported_limitation(annotation):
    # Documents current behavior, not a promise: unlike Mapping, the abstract sequence types have no handler, so even a
    # plain JSON array is refused (loudly, with a TypeError).
    with pytest.raises(TypeError, match=r"Unsupported type for deserialization"):
        deserialize_json([1, 2], annotation)


@pytest.mark.parametrize("annotation", [Type[int], type[int]], ids=["typing.Type[int]", "type[int]"])
def test_type_annotations_are_not_interpreted_limitation(annotation):
    # Documents current behavior, not a promise: JSON cannot hold a class, so there is nothing to restore; objects and
    # scalars are handed through like for any annotation without a handler, and an array is refused.
    assert_strictly_equal(deserialize_json({"a": 1}, annotation), {"a": 1})
    assert_strictly_equal(deserialize_json("int", annotation), "int")
    with pytest.raises(TypeError, match=r"Unsupported type for deserialization"):
        deserialize_json([1], annotation)


def test_named_tuple_comes_back_as_a_plain_tuple_limitation():
    # Documents current behavior, not a promise: a NamedTuple is a tuple subclass, so it serializes as a plain array,
    # and the target is handled like a bare `tuple`: the result is no NamedPair and its items are not converted.
    value = NamedPair(DAY, 3)

    assert serialize_json(value) == '["2026-10-27", 3]'
    assert_strictly_equal(deserialize_json(to_json_value(value), NamedPair), ("2026-10-27", 3))
    assert_strictly_equal(deserialize_json([["2026-10-27", 3]], list[NamedPair]), [("2026-10-27", 3)])


def test_typed_dict_values_are_handed_through_unconverted_limitation():
    # Documents current behavior, not a promise: a TypedDict is a plain dict at run time and the annotations of its keys
    # are not read, so the date and the enum stay strings.
    value = Settings(when=DAY, mode=Mode.FAST)

    assert serialize_json(value) == '{"when": "2026-10-27", "mode": "fast"}'
    assert_strictly_equal(deserialize_json(to_json_value(value), Settings), {"when": "2026-10-27", "mode": "fast"})


# --- NewType, Annotated, Final, ClassVar, TypeVar, Generic ------------------------------------------------------------

_NEW_TYPE_REASON = "NewType is not unwrapped to its supertype (TypeError: issubclass() arg 1 must be a class)"
_NEW_TYPE_XFAIL = pytest.mark.xfail(strict=True, raises=TypeError, reason=_NEW_TYPE_REASON)


def test_new_type_values_serialize_as_their_supertype():
    account = Account(UserId(7), Slug("ann"), [UserId(1), UserId(2)], None)

    assert serialize_json(account) == '{"uid": 7, "slug": "ann", "friends": [1, 2], "boss": null}'


@_NEW_TYPE_XFAIL
@pytest.mark.parametrize(
    ("annotation", "data", "expected"),
    [
        pytest.param(UserId, 7, 7, id="int"),
        pytest.param(Slug, "ann", "ann", id="str"),
        pytest.param(list[UserId], [1, 2], [1, 2], id="list"),
        pytest.param(Optional[UserId], 7, 7, id="optional"),
        pytest.param(dict[str, UserId], {"a": 1}, {"a": 1}, id="dict-values"),
    ],
)
def test_new_type_target_behaves_like_its_supertype(annotation, data, expected):
    assert_strictly_equal(deserialize_json(data, annotation), expected)


@_NEW_TYPE_XFAIL
def test_new_type_dataclass_fields_round_trip():
    account = Account(UserId(7), Slug("ann"), [UserId(1), UserId(2)], UserId(3))

    assert_round_trip(account, Account)


def test_annotated_fields_of_a_string_annotated_dataclass_use_the_wrapped_type():
    # typing.get_type_hints, which resolves the string annotations, strips the Annotated metadata
    value = Measured(3, 0.5, ["a", "b"], DAY, Level.LOW, [Color.RED, Color.BLUE])

    assert serialize_json(value) == json.dumps(
        {
            "count": 3,
            "ratio": 0.5,
            "labels": ["a", "b"],
            "when": "2026-10-27",
            "maybe": 1,
            "nested": ["red", "blue"],
        }
    )
    assert_round_trip(value, Measured)


def test_annotated_optional_field_accepts_null():
    value = Measured(0, 0.0, [], DAY, None, [])

    assert_round_trip(value, Measured)


# Python 3.14 raises TypeError (issubclass() arg 1 must be a class); on 3.10, where Annotated is a plain class, the JSON
# value is passed through unconverted, which the strict comparison reports. Only cases that fail on both are listed:
# Annotated[int, ...] "works" on 3.10 because primitives are never validated.
_ANNOTATED_XFAIL = pytest.mark.xfail(
    strict=True,
    raises=(TypeError, AssertionError),
    reason="Annotated[X, ...] is not unwrapped to X outside string annotations",
)


@_ANNOTATED_XFAIL
@pytest.mark.parametrize(
    ("annotation", "data", "expected"),
    [
        pytest.param(Annotated[list[int], "meta"], [1, 2], [1, 2], id="list"),
        pytest.param(Annotated[Color, "meta"], "red", Color.RED, id="enum"),
        pytest.param(Annotated[datetime.date, "meta"], "2026-10-27", DAY, id="date"),
        pytest.param(list[Annotated[Color, "meta"]], ["red"], [Color.RED], id="list-element"),
        pytest.param(Optional[Annotated[Color, "meta"]], "red", Color.RED, id="optional-enum"),
    ],
)
def test_annotated_target_behaves_like_the_wrapped_type(annotation, data, expected):
    assert_strictly_equal(deserialize_json(data, annotation), expected)


def test_annotated_field_with_a_real_annotation_serializes_as_its_wrapped_type():
    assert serialize_json(RealMeasured(count=1, when=DAY)) == '{"count": 1, "when": "2026-10-27"}'


@_ANNOTATED_XFAIL
def test_annotated_field_with_a_real_annotation_round_trips():
    assert_round_trip(RealMeasured(count=1, when=DAY), RealMeasured)


def test_class_var_is_neither_serialized_nor_deserialized():
    value = WithClassVar("x")

    assert serialize_json(value) == '{"name": "x"}'
    restored = deserialize_json({"name": "x", "version": 99, "aliases": ["z"]}, WithClassVar)

    assert_strictly_equal(restored, value)
    assert WithClassVar.version == 3 and WithClassVar.aliases == ("a", "b")


def test_final_field_serializes_as_its_value():
    assert serialize_json(WithFinal("x")) == '{"name": "x", "limit": 10}'


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Final[X] is not unwrapped to X (TypeError: issubclass() arg 1 must be a class)",
)
def test_final_field_round_trips():
    assert_round_trip(WithFinal("x", 12), WithFinal)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(Box(1, [2, 3]), '{"item": 1, "items": [2, 3]}', id="int"),
        pytest.param(Box("a", ["b"]), '{"item": "a", "items": ["b"]}', id="str"),
        pytest.param(Box(None, []), '{"item": null, "items": []}', id="null"),
        pytest.param(Box(1.5, [None, 2.5]), '{"item": 1.5, "items": [null, 2.5]}', id="float"),
        pytest.param(Box(True, [False]), '{"item": true, "items": [false]}', id="bool"),
    ],
)
def test_unparameterized_generic_dataclass_round_trips_scalar_values(value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, Box)


def test_generic_dataclass_serializes_its_runtime_values():
    assert serialize_json(Box(Point(1, 2), [Point(3, 4)])) == (
        '{"item": {"x": 1, "y": 2}, "items": [{"x": 3, "y": 4}]}'
    )
    assert serialize_json(Box(Color.RED, [Color.BLUE])) == '{"item": "red", "items": ["blue"]}'


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"item": [1, 2], "items": []}, id="array"),
        pytest.param({"item": {"a": 1}, "items": []}, id="object"),
        pytest.param({"item": 1, "items": [{"a": 1}]}, id="object-element"),
    ],
)
def test_type_variable_fields_refuse_structured_json_limitation(data):
    # Documents current behavior, not a promise: a TypeVar is never resolved, and only JSON scalars pass through it.
    with pytest.raises(TypeError):
        deserialize_json(data, Box)


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="a parameterized generic dataclass (Box[int]) is rejected: dataclasses.fields() does not take the alias",
)
@pytest.mark.parametrize(
    ("annotation", "data", "expected"),
    [
        pytest.param(Box[int], {"item": 1, "items": [2, 3]}, Box(1, [2, 3]), id="Box[int]"),
        pytest.param(Box[str], {"item": "a", "items": []}, Box("a", []), id="Box[str]"),
        pytest.param(Optional[Box[int]], {"item": 1, "items": [2]}, Box(1, [2]), id="Optional[Box[int]]"),
        pytest.param(list[Box[int]], [{"item": 1, "items": []}], [Box(1, [])], id="list[Box[int]]"),
        pytest.param(
            HoldsIntBox, {"box": {"item": 1, "items": [2]}}, HoldsIntBox(Box(1, [2])), id="field-Box[int]"
        ),
    ],
)
def test_parameterized_generic_dataclass_can_be_deserialized(annotation, data, expected):
    assert_strictly_equal(deserialize_json(data, annotation), expected)


@pytest.mark.parametrize(
    ("annotation", "data"),
    [
        pytest.param(Box[Point], {"item": {"x": 1, "y": 2}, "items": [{"x": 3, "y": 4}]}, id="Box[Point]"),
        pytest.param(Optional[Box[Color]], {"item": "red", "items": ["blue"]}, id="Optional[Box[Color]]"),
        pytest.param(HoldsBox, {"box": {"item": {"x": 1, "y": 2}, "items": []}}, id="field-Box[Point]"),
    ],
)
def test_type_arguments_of_a_generic_dataclass_are_not_substituted_limitation(annotation, data):
    # Documents current behavior, not a promise: nothing maps the T of Box[Point] onto the TypeVar fields, so a
    # Box[Point] cannot be restored (today it is refused already because the parameterized alias is rejected, see the
    # xfail above). It is refused loudly, never half-restored with raw dicts in place of Points.
    with pytest.raises(TypeError):
        deserialize_json(data, annotation)


# --- forward references and string annotations ------------------------------------------------------------------------


def _books() -> tuple[Book, Book, Book]:
    austen = Author("Austen", datetime.date(1775, 12, 16), ())
    emma = Book("Emma", 1815, austen)
    dune = Book("Dune", 1965, None)
    return emma, dune, Book("Dune", 1965, Author("Herbert", datetime.date(1920, 10, 8), (dune,)))


EMMA_JSON = {
    "title": "Emma",
    "year": 1815,
    "author": {"name": "Austen", "born": "1775-12-16", "books": []},
}
DUNE_JSON = {"title": "Dune", "year": 1965, "author": None}
DUNE_BY_HERBERT_JSON = {
    "title": "Dune",
    "year": 1965,
    "author": {"name": "Herbert", "born": "1920-10-08", "books": [DUNE_JSON]},
}


def test_forward_referenced_mutually_recursive_dataclasses_round_trip():
    emma, dune, dune_by_herbert = _books()

    assert serialize_json(dune_by_herbert) == json.dumps(DUNE_BY_HERBERT_JSON)
    restored = assert_round_trip(dune_by_herbert, Book)

    assert type(restored.author) is Author and type(restored.author.books[0]) is Book
    assert_round_trip(emma.author, Author)


def test_library_with_every_spelling_of_a_forward_reference_round_trips():
    emma, dune, dune_by_herbert = _books()
    library = Library(
        name="Main",
        featured=dune_by_herbert,
        shelf=[emma, dune],
        by_isbn={"978-0": emma},
        pair=(dune, None),
        quoted=[emma],
        spelled=[dune],
        optional_quoted=None,
    )

    assert serialize_json(library) == json.dumps(
        {
            "name": "Main",
            "featured": DUNE_BY_HERBERT_JSON,
            "shelf": [EMMA_JSON, DUNE_JSON],
            "by_isbn": {"978-0": EMMA_JSON},
            "pair": [DUNE_JSON, None],
            "quoted": [EMMA_JSON],
            "spelled": [DUNE_JSON],
            "optional_quoted": None,
        }
    )
    restored = assert_round_trip(library, Library)

    assert type(restored.pair) is tuple and type(restored.shelf) is list and type(restored.by_isbn["978-0"]) is Book


def test_library_with_empty_optionals_and_containers():
    library = Library("Empty", None, [], None, (Book("A", 1, None), Book("B", 2, None)), [], [], Book("C", 3, None))

    assert_round_trip(library, Library)


def test_forward_referenced_field_rejects_a_wrong_shape():
    data = {
        "name": "Main",
        "featured": None,
        "shelf": "Emma",
        "by_isbn": None,
        "pair": [DUNE_JSON, None],
        "quoted": [],
        "spelled": [],
        "optional_quoted": None,
    }

    with pytest.raises(TypeError, match=r"Expected a JSON array for list\[.*Book\], got str"):
        deserialize_json(data, Library)


def _quoted_names_in_builtin_generics_resolve() -> bool:
    # typing.get_type_hints, which jserpy relies on, leaves list["Book"] unresolved on older Pythons (3.10)
    try:
        return typing.get_args(typing.get_type_hints(InnerQuoted)["books"]) == (Book,)
    except Exception:
        return False


@pytest.mark.skipif(
    not _quoted_names_in_builtin_generics_resolve(),
    reason="typing.get_type_hints does not resolve quoted names inside builtin generics on this Python",
)
def test_quoted_names_inside_builtin_generic_annotations_are_resolved():
    emma, dune, _ = _books()
    value = InnerQuoted(books=[emma], by_title={"Dune": dune}, trio=(dune, emma, dune), maybe=[dune])

    assert serialize_json(value) == json.dumps(
        {
            "books": [EMMA_JSON],
            "by_title": {"Dune": DUNE_JSON},
            "trio": [DUNE_JSON, EMMA_JSON, DUNE_JSON],
            "maybe": [DUNE_JSON],
        }
    )
    assert_round_trip(value, InnerQuoted)


@pytest.mark.parametrize(
    ("annotation", "data"),
    [
        pytest.param(list["Point"], [{"x": 1, "y": 2}], id="list-of-quoted-name"),
        pytest.param(Optional["Point"], {"x": 1, "y": 2}, id="Optional-quoted-name"),
        pytest.param("Point", {"x": 1, "y": 2}, id="bare-str"),
        pytest.param(ForwardRef("Point"), {"x": 1, "y": 2}, id="ForwardRef"),
    ],
)
def test_unresolved_forward_reference_target_is_refused_limitation(annotation, data):
    # Documents current behavior, not a promise: only the annotations of dataclass fields are resolved (against the
    # class's module); a string given to deserialize_json directly has no namespace to be resolved in.
    with pytest.raises(TypeError):
        deserialize_json(data, annotation)


def test_dataclass_with_an_unresolvable_annotation_cannot_be_deserialized_limitation():
    # Documents current behavior, not a promise: one name that typing.get_type_hints cannot resolve (here, imported
    # for type checkers only) leaves every field annotation of the class as a string, which is refused with a
    # (rather obscure) TypeError. Serializing is unaffected.
    value = WithUnimportable("x", "1/2")

    assert serialize_json(value) == '{"name": "x", "ratio": "1/2"}'
    with pytest.raises(TypeError):
        deserialize_json({"name": "x", "ratio": "1/2"}, WithUnimportable)


def _tree() -> TreeNode:
    return TreeNode(
        1,
        [TreeNode(2, [TreeNode(4, [], None)], "b"), TreeNode(3, [], None)],
        "root",
    )


def test_self_referential_dataclass_round_trips():
    assert serialize_json(_tree()) == json.dumps(
        {
            "value": 1,
            "children": [
                {"value": 2, "children": [{"value": 4, "children": [], "label": None}], "label": "b"},
                {"value": 3, "children": [], "label": None},
            ],
            "label": "root",
        }
    )
    assert_round_trip(_tree(), TreeNode)


def _wide_tree(depth: int, counter: list[int]) -> tuple[TreeNode, dict[str, Any]]:
    counter[0] += 1
    value = counter[0]
    if depth == 0:
        return TreeNode(value, [], None), {"value": value, "children": [], "label": None}
    built = [_wide_tree(depth - 1, counter) for _ in range(3)]
    label = f"n{value}"
    return (
        TreeNode(value, [node for node, _ in built], label),
        {"value": value, "children": [data for _, data in built], "label": label},
    )


def test_wide_self_referential_tree_round_trips():
    tree, expected_json = _wide_tree(4, [0])  # 1 + 3 + 9 + 27 + 81 nodes

    assert serialize_json(tree) == json.dumps(expected_json)
    assert_round_trip(tree, TreeNode)


def test_long_chain_of_optional_self_references_round_trips():
    chain = None
    expected_json = None
    for value in range(80, 0, -1):
        chain = Link(value, chain)
        expected_json = {"value": value, "next": expected_json}

    assert serialize_json(chain) == json.dumps(expected_json)
    assert_round_trip(chain, Link)


def test_self_referential_folder_through_dict_tuple_and_frozenset_fields():
    leaf = Folder("leaf", {}, frozenset(), ())
    docs = Folder("docs", {"leaf": leaf}, frozenset({"b.txt", "a.txt"}), (None, leaf))
    root = Folder("root", {"docs": docs, "empty": Folder("empty", {}, frozenset(), ())}, frozenset(), (docs, None))

    leaf_json = {"name": "leaf", "subfolders": {}, "files": [], "pinned": []}
    docs_json = {
        "name": "docs",
        "subfolders": {"leaf": leaf_json},
        "files": ["a.txt", "b.txt"],
        "pinned": [None, leaf_json],
    }
    empty_json = {"name": "empty", "subfolders": {}, "files": [], "pinned": []}
    assert serialize_json(root) == json.dumps(
        {
            "name": "root",
            "subfolders": {"docs": docs_json, "empty": empty_json},
            "files": [],
            "pinned": [docs_json, None],
        }
    )
    assert_round_trip(root, Folder)


def test_type_aliases_named_in_string_annotations_are_resolved():
    value = Aliased(ident="a-1", scores={"k": [1.5, 2.0], "e": []}, points=[Point(1, 2)], palette="dark")

    assert serialize_json(value) == (
        '{"ident": "a-1", "scores": {"k": [1.5, 2.0], "e": []}, "points": [{"x": 1, "y": 2}], "palette": "dark"}'
    )
    assert_round_trip(value, Aliased)


def test_type_aliases_hold_their_other_alternatives():
    value = Aliased(ident=7, scores={}, points=None, palette=None)

    assert serialize_json(value) == '{"ident": 7, "scores": {}, "points": null, "palette": null}'
    assert_round_trip(value, Aliased)


def test_type_alias_rejects_a_value_outside_its_literal():
    data = {"ident": 7, "scores": {}, "points": None, "palette": "sepia"}

    with pytest.raises(TypeError, match=r"Unsupported types for deserialization"):
        deserialize_json(data, Aliased)


# --- deeply nested typing constructs ----------------------------------------------------------------------------------


def _nested_optional_lists(depth: int, innermost: Any) -> Any:
    annotation = innermost
    for _ in range(depth):
        annotation = Optional[list[annotation]]
    return annotation


def _nested_optional_dicts(depth: int, innermost: Any) -> Any:
    annotation = innermost
    for _ in range(depth):
        annotation = Optional[dict[str, annotation]]
    return annotation


@pytest.mark.parametrize("depth", [1, 2, 10, 60])
def test_annotation_of_optional_lists_nested_deeply(depth):
    value: Any = 7
    for _ in range(depth):
        value = [value]

    assert serialize_json(value) == "[" * depth + "7" + "]" * depth
    assert_round_trip(value, _nested_optional_lists(depth, int))


@pytest.mark.parametrize("depth", [1, 2, 10, 60])
def test_annotation_of_optional_lists_nested_deeply_with_null_at_the_bottom(depth):
    value: Any = None
    for _ in range(depth):
        value = [value]

    assert serialize_json(value) == "[" * depth + "null" + "]" * depth
    assert_round_trip(value, _nested_optional_lists(depth, Optional[int]))


@pytest.mark.parametrize("depth", [1, 2, 10, 60])
def test_annotation_of_optional_dicts_nested_deeply(depth):
    value: Any = Color.GREEN
    for _ in range(depth):
        value = {"k": value}

    assert serialize_json(value) == '{"k": ' * depth + '"green"' + "}" * depth
    assert_round_trip(value, _nested_optional_dicts(depth, Color))


def test_annotation_alternating_containers_and_unions_with_null_at_every_level():
    annotation = Optional[dict[str, Optional[list[Optional[tuple[Optional[int], ...]]]]]]
    value = {"a": [(1, None), None, ()], "b": None, "c": []}
    text = '{"a": [[1, null], null, []], "b": null, "c": []}'

    assert serialize_json(value) == text
    assert_round_trip(value, annotation)
    assert_round_trip(None, annotation)
    assert_round_trip({}, annotation)


def test_dict_of_lists_of_tuples_of_unions_and_literals():
    annotation = dict[str, list[tuple[int | None, Literal["a", "b"] | Path]]]
    value = {
        "x": [(1, "a"), (None, Path("data.txt")), (2, "b")],
        "y": [],
        "z": [(None, "b")],
    }

    assert serialize_json(value) == '{"x": [[1, "a"], [null, "data.txt"], [2, "b"]], "y": [], "z": [[null, "b"]]}'
    restored = assert_round_trip(value, annotation)

    assert type(restored["x"][1][1]) is type(Path()) and restored["x"][0][1] == "a"


def test_literal_member_wins_over_a_path_with_the_same_text():
    # Union order again: "a" is both an allowed literal and a valid path, and the Literal member comes first.
    annotation = list[Literal["a", "b"] | Path]

    assert_strictly_equal(deserialize_json(["a", "c"], annotation), ["a", Path("c")])
    assert_strictly_equal(deserialize_json(["a", "c"], list[Path | Literal["a", "b"]]), [Path("a"), Path("c")])


def test_list_of_dicts_of_tuples_of_enum_and_decimal_unions():
    annotation = list[dict[str, tuple[Color | None, Decimal | str]]]
    value = [{"p": (Color.RED, Decimal("2.50")), "q": (None, "n/a")}, {}]

    assert serialize_json(value) == '[{"p": ["red", "2.50"], "q": [null, "n/a"]}, {}]'
    assert_round_trip(value, annotation)


def test_optional_dict_of_optional_lists_of_tuples_with_frozensets():
    annotation = Optional[
        dict[str, Optional[list[tuple[Level | Color, frozenset[datetime.date | None]]]]]
    ]
    value = {
        "k": [(Level.LOW, frozenset({DAY, None})), (Color.BLUE, frozenset())],
        "n": None,
    }

    # the frozenset is ordered by the canonical JSON text of its items: the quoted date before null
    assert serialize_json(value) == '{"k": [[1, ["2026-10-27", null]], ["blue", []]], "n": null}'
    assert_round_trip(value, annotation)
    assert_round_trip(None, annotation)


def test_heterogeneous_tuple_of_containers_of_unions():
    annotation = tuple[int, list[tuple[str, Optional[datetime.date]]], dict[str, Literal[1, 2] | None]]
    value = (3, [("a", DAY), ("b", None)], {"x": 1, "y": None, "z": 2})

    assert serialize_json(value) == '[3, [["a", "2026-10-27"], ["b", null]], {"x": 1, "y": null, "z": 2}]'
    assert_round_trip(value, annotation)


def test_deep_union_rejects_a_mismatch_at_the_innermost_level():
    annotation = dict[str, list[tuple[int | None, Literal["a", "b"] | list[int]]]]

    with pytest.raises(TypeError, match=r"Unsupported types for deserialization.*got str"):
        deserialize_json({"x": [[1, "a"], [None, "zzz"]]}, annotation)  # "zzz" is no literal and no array


def test_deep_union_rejects_a_wrong_tuple_length():
    annotation = dict[str, list[tuple[int | None, Literal["a", "b"] | Path]]]

    with pytest.raises(TypeError, match=r"Expected a JSON array of length 2 .*, got length 3"):
        deserialize_json({"x": [[1, "a", "extra"]]}, annotation)


def _site(parent: Site | None) -> Site:
    sensor = Sensor("s1", "temp", (20.5, None, 21.0), NOON, {"offset": Decimal("0.10")})
    quiet = Sensor("s2", "humidity", (), None, None)
    return Site(
        name="north",
        sensors={"roof": [sensor, quiet], "cellar": []},
        tags=frozenset({"b", "a"}),
        extra={"notes": [1, {"k": None}], "ratio": 0.5},
        parent=parent,
    )


SENSOR_JSON = {
    "ident": "s1",
    "kind": "temp",
    "readings": [20.5, None, 21.0],
    "last_seen": "2026-10-27T12:00:00",
    "calibration": {"offset": "0.10"},
}
QUIET_JSON = {"ident": "s2", "kind": "humidity", "readings": [], "last_seen": None, "calibration": None}


def test_nested_dataclasses_with_literals_unions_decimals_and_any():
    site = _site(None)

    assert serialize_json(site) == json.dumps(
        {
            "name": "north",
            "sensors": {"roof": [SENSOR_JSON, QUIET_JSON], "cellar": []},
            "tags": ["a", "b"],
            "extra": {"notes": [1, {"k": None}], "ratio": 0.5},
            "parent": None,
        }
    )
    assert_round_trip(site, Site)


def test_nested_dataclass_tree_through_an_optional_forward_reference():
    site = _site(_site(_site(None)))

    restored = assert_round_trip(site, Site)

    assert restored.parent.parent.parent is None and type(restored.parent.parent) is Site

"""Containers and mapping keys: lists, tuples, sets, mappings, and which dict keys survive a round trip.

Every case is checked from both sides: the JSON a consumer of the text sees (derived by hand or with the stdlib
``json`` module, never from jserpy's own output) and a strict round trip back to the annotated type
(``objects_support.assert_round_trip`` compares exact types, so ``1`` / ``1.0`` / ``True`` and ``tuple`` / ``list`` /
``set`` / ``frozenset`` cannot be mistaken for each other).

Conventions:

* A test named ``*_limitation`` documents current, undocumented behavior. It is not a promise.
* A strict ``xfail`` is a library defect: the test asserts what a user would expect and starts failing (as an
  unexpected pass) the day the library is fixed.
"""

import copy
import functools
import json
import math
import typing
from collections import ChainMap, Counter, OrderedDict, UserDict, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, make_dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum, IntEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, Optional, Union

import pytest

from jserpy import deserialize_json, serialize_json, serialize_json_as_obj
from objects_support import assert_round_trip, assert_strictly_equal, to_json_value


# --- shared fixtures ---------------------------------------------------------------------------------------------


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Shade(Enum):
    """A second enum with the same values as ``Color.RED``, to collide with it."""

    RED = "red"


class Level(Enum):
    LOW = 1
    HIGH = 2


class Digit(Enum):
    ONE = "1"


class Priority(IntEnum):
    LOW = 1
    HIGH = 2


class Switch(Enum):
    ON = True


class Ratio(Enum):
    HALF = 0.5


class Cell(Enum):
    ORIGIN = (0, 0)


class Blank(Enum):
    NOTHING = None


class Inner(Enum):
    X = "x"


class Wrapper(Enum):
    WRAPPED = Inner.X


class Label(str, Enum):
    A = "a"


class Shout(str):
    pass


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass
class Span:
    bounds: tuple[int, int]


@dataclass
class Holder:
    payload: Any


@dataclass
class Grid:
    rows: list[list[float]]
    labels: dict[str, list[int]]


@dataclass
class Board:
    cells: dict[tuple[int, int], str]


@dataclass(frozen=True)
class YX:
    """Declares ``y`` before ``x``, the reverse of alphabetical order."""

    y: int
    x: int


class Vec(NamedTuple):
    x: int
    y: int


class Stamp(NamedTuple):
    day: date
    tags: frozenset[str]


class Playlist(list):
    pass


class Tags(set):
    pass


class FrozenTags(frozenset):
    pass


@dataclass(frozen=True)
class Tagged:
    """Hashable, with container fields of its own, so that it can be a set member."""

    name: str
    tags: frozenset[str]
    path: tuple[int, ...]


class FrozenMap(Mapping):
    """An immutable, hashable mapping: the only kind of mapping that can be a member of a set."""

    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __hash__(self):
        return len(self._data)


class LeakyKey:
    def __repr__(self):
        return "KEY_SENTINEL"

    __str__ = __repr__


def _xfail(reason):
    return pytest.mark.xfail(strict=True, reason=reason)


def _canonical(json_value):
    """How a set member is ordered: its compact, sorted-keys JSON text (computed with the stdlib only)."""
    return json.dumps(json_value, sort_keys=True, separators=(",", ":"))


def _ordered_members(*json_members):
    """JSON members de-duplicated and sorted the way a set is written."""
    unique = {_canonical(member): member for member in json_members}
    return [unique[text] for text in sorted(unique)]


def _insertion_orders(members):
    """The same members in three different orders: ascending, descending and strided."""
    ordered = list(members)
    step = next(candidate for candidate in (7, 11, 13, 17) if math.gcd(candidate, len(ordered)) == 1)
    strided = [ordered[(index * step) % len(ordered)] for index in range(len(ordered))]
    return [ordered, ordered[::-1], strided]


# --- lists -------------------------------------------------------------------------------------------------------


LIST_GOLDENS = [
    pytest.param([], "[]", id="empty"),
    pytest.param([None], "[null]", id="singleton-null"),
    pytest.param([[]], "[[]]", id="list-with-an-empty-list"),
    pytest.param([[], []], "[[], []]", id="empty-rows"),
    pytest.param([1, "a", None, True, 2.5], '[1, "a", null, true, 2.5]', id="heterogeneous"),
    pytest.param([[1]], "[[1]]", id="matrix-1x1"),
    pytest.param(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        "[[1, 2, 3], [4, 5, 6], [7, 8, 9]]",
        id="matrix-3x3",
    ),
    pytest.param([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], "[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]", id="matrix-2x5"),
    pytest.param([[1], [], [2, 3, 4], [5, 6]], "[[1], [], [2, 3, 4], [5, 6]]", id="ragged"),
    pytest.param([{"a": 1}, {}, {"b": [2]}], '[{"a": 1}, {}, {"b": [2]}]', id="list-of-dicts"),
    pytest.param([[[1, 2], [3]], [[4]], []], "[[[1, 2], [3]], [[4]], []]", id="three-levels"),
    pytest.param([1.0, -0.0, 1e100, 1e-7], "[1.0, -0.0, 1e+100, 1e-07]", id="floats"),
    pytest.param([Point(1, 2), Point(3, 4)], '[{"x": 1, "y": 2}, {"x": 3, "y": 4}]', id="list-of-dataclasses"),
    pytest.param(Playlist([1, 2]), "[1, 2]", id="list-subclass"),
]


@pytest.mark.parametrize(("value", "text"), LIST_GOLDENS)
def test_list_serialization_golden(value, text):
    assert serialize_json(value) == text
    assert_strictly_equal(to_json_value(value), json.loads(text))


LIST_ROUND_TRIPS = [
    pytest.param([], list[int], id="empty"),
    pytest.param([], list, id="empty-bare"),
    pytest.param([7], list[int], id="singleton"),
    pytest.param(["a", "b", "c"], list[str], id="homogeneous"),
    pytest.param([True, False], list[bool], id="bools"),
    pytest.param([1, "a", None, True, 2.5, [1, 2], {"k": [3]}], list[Any], id="heterogeneous-any"),
    pytest.param([1, "a", None, True, 2.5, [1, 2], {"k": [3]}], list, id="heterogeneous-bare"),
    pytest.param([[1], [], [2, 3, 4], [5, 6]], list[list[int]], id="ragged"),
    pytest.param([[1]], list[list[int]], id="matrix-1x1"),
    pytest.param([[1.5, 2.0, 0.0], [-0.0, 1e100, 1e-7]], list[list[float]], id="matrix-2x3-floats"),
    pytest.param([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], list[list[int]], id="matrix-2x5"),
    pytest.param([[], []], list[list[int]], id="empty-rows"),
    pytest.param([[[1, 2], [3]], [[4]], []], list[list[list[int]]], id="three-levels"),
    pytest.param([{"a": 1}, {}, {"b": 2}], list[dict[str, int]], id="list-of-dicts"),
    pytest.param([None, 1, None], list[Optional[int]], id="optional-items"),
    pytest.param([None, (1, "a")], list[Optional[tuple[int, str]]], id="optional-tuple-items"),
    pytest.param([1, [2, 3], []], list[Union[int, list[int]]], id="int-or-list-items"),
    pytest.param([Color.RED, Color.BLUE], list[Color], id="enums"),
    pytest.param([Point(1, 2), Point(3, 4)], list[Point], id="dataclasses"),
    pytest.param([[Point(1, 2)], []], list[list[Point]], id="lists-of-dataclasses"),
    pytest.param([(1, "a"), (2, "b")], list[tuple[int, str]], id="list-of-tuples"),
    pytest.param([{1, 2}, set()], list[set[int]], id="list-of-sets"),
    pytest.param([frozenset(), frozenset({"a"})], list[frozenset[str]], id="list-of-frozensets"),
    pytest.param(["a"], typing.List[str], id="typing-List"),
    pytest.param([[1], [2]], typing.List[typing.List[int]], id="typing-List-of-List"),
]


@pytest.mark.parametrize(("value", "annotation"), LIST_ROUND_TRIPS)
def test_list_round_trip(value, annotation):
    assert_round_trip(value, annotation)


def test_list_subclass_is_restored_as_a_plain_list():
    restored = deserialize_json(to_json_value(Playlist([1, 2])), list[int])

    assert_strictly_equal(restored, [1, 2])


def test_any_members_come_back_as_their_json_form_limitation():
    # Documents current behavior, not a promise: deserialization follows the annotation, and `Any` (like a bare `list`
    # or `dict`) has nothing to rebuild a tuple, set, enum or Decimal from, so the JSON value is returned as is.
    value = [(1, 2), {3}, Color.RED, Decimal("1.5"), {"k": (4,)}]
    expected = [[1, 2], [3], "red", "1.5", {"k": [4]}]

    assert_strictly_equal(deserialize_json(to_json_value(value), list[Any]), expected)
    assert_strictly_equal(deserialize_json(to_json_value(value), list), expected)


def test_large_list_and_matrix_round_trip():
    numbers = list(range(5000))
    matrix = [[row * 0.25 + col for col in range(40)] for row in range(40)]

    assert serialize_json(numbers) == json.dumps(numbers)
    assert serialize_json(matrix) == json.dumps(matrix)
    assert_round_trip(numbers, list[int])
    assert_round_trip(matrix, list[list[float]])


def test_dataclass_with_list_fields_has_declaration_order_and_round_trips():
    grid = Grid(rows=[[1.5, 2.0], [], [0.0]], labels={"odd": [1, 3], "none": []})
    expected = {"rows": [[1.5, 2.0], [], [0.0]], "labels": {"odd": [1, 3], "none": []}}

    assert serialize_json(grid) == json.dumps(expected)
    assert_round_trip(grid, Grid)


@pytest.mark.parametrize(
    ("data", "annotation", "message"),
    [
        pytest.param(["LEAK", "LEAK"], list[list[int]], "Expected a JSON array", id="strings-for-lists"),
        pytest.param([["LEAK"]], list[dict[str, int]], "Expected a JSON object", id="list-for-dict"),
        pytest.param({"a": ["LEAK"]}, list[list[int]], "Expected a JSON array", id="object-for-list"),
        pytest.param(
            [["LEAK"], {"LEAK": "LEAK"}], list[list[str]], "Expected a JSON array", id="object-among-lists"
        ),
        pytest.param(
            [[["LEAK"], 1]], list[list[list[str]]], "Expected a JSON array", id="number-three-levels-down"
        ),
    ],
)
def test_list_with_the_wrong_shape_raises_type_error_without_the_data(data, annotation, message):
    with pytest.raises(TypeError, match=message) as caught:
        deserialize_json(data, annotation)

    assert "LEAK" not in str(caught.value)


def test_integers_become_floats_inside_containers():
    restored = deserialize_json({"a": [1, 2.5], "b": []}, dict[str, list[float]])

    assert_strictly_equal(restored, {"a": [1.0, 2.5], "b": []})
    assert_strictly_equal(deserialize_json([1, 2], set[float]), {1.0, 2.0})
    assert_strictly_equal(deserialize_json([[1, 2]], tuple[tuple[float, float], ...]), ((1.0, 2.0),))


# --- tuples ------------------------------------------------------------------------------------------------------


TUPLE_GOLDENS = [
    pytest.param((), "[]", id="empty"),
    pytest.param((1,), "[1]", id="singleton"),
    pytest.param((1, "a", 2.5), '[1, "a", 2.5]', id="heterogeneous"),
    pytest.param(((1, 2), (3,), ()), "[[1, 2], [3], []]", id="nested-tuples"),
    pytest.param(([1, 2], [3]), "[[1, 2], [3]]", id="tuple-of-lists"),
    pytest.param([(1, 2), (3, 4)], "[[1, 2], [3, 4]]", id="list-of-tuples"),
    pytest.param(((("a", 1), ("b", 2)), None), '[[["a", 1], ["b", 2]], null]', id="pairs-and-null"),
    pytest.param(Vec(1, 2), "[1, 2]", id="named-tuple"),
]


@pytest.mark.parametrize(("value", "text"), TUPLE_GOLDENS)
def test_tuple_serialization_golden(value, text):
    assert serialize_json(value) == text


TUPLE_ROUND_TRIPS = [
    pytest.param((), tuple[int, ...], id="empty-variadic"),
    pytest.param((), tuple, id="empty-bare"),
    pytest.param((), tuple[()], id="empty-tuple-annotation"),
    pytest.param((5,), tuple[int], id="singleton"),
    pytest.param((5,), tuple[int, ...], id="singleton-variadic"),
    pytest.param((1, "a", 2.5), tuple[int, str, float], id="heterogeneous"),
    pytest.param((1, "a", 2.0), tuple[int, str, float], id="float-stays-float"),
    pytest.param((1, 2, 3, 4, 5), tuple[int, ...], id="variadic"),
    pytest.param(((1, 2), (3, 4)), tuple[tuple[int, int], ...], id="tuple-of-pairs"),
    pytest.param(((1, ("a", 2.5)), (2, ("b", 0.5))), tuple[tuple[int, tuple[str, float]], ...], id="nested-fixed"),
    pytest.param(([1, 2], [3]), tuple[list[int], ...], id="tuple-of-lists"),
    pytest.param([(1, 2), (3, 4)], list[tuple[int, int]], id="list-of-tuples"),
    pytest.param(({"a": 1},), tuple[dict[str, int], ...], id="tuple-of-dicts"),
    pytest.param((frozenset({1}), {2}), tuple[frozenset[int], set[int]], id="tuple-of-sets"),
    pytest.param((Point(1, 2), Point(3, 4)), tuple[Point, ...], id="tuple-of-dataclasses"),
    pytest.param((Color.RED, 1), tuple[Color, int], id="enum-and-int"),
    pytest.param((None, 1), tuple[Optional[int], int], id="optional-member"),
    pytest.param((1, "a"), typing.Tuple[int, str], id="typing-Tuple"),
    pytest.param((1, 2, 3), typing.Tuple[int, ...], id="typing-Tuple-variadic"),
    pytest.param((1, "a", [2, 3], {"k": 4}), tuple, id="bare-with-json-members"),
]


@pytest.mark.parametrize(("value", "annotation"), TUPLE_ROUND_TRIPS)
def test_tuple_round_trip(value, annotation):
    assert_round_trip(value, annotation)


@pytest.mark.parametrize(
    ("data", "annotation", "message"),
    [
        pytest.param(["LEAK"], tuple[int, str], "length 2", id="short-pair"),
        pytest.param(["LEAK", "a", 2.5, 4], tuple[int, str, float], "length 3", id="long-triple"),
        pytest.param([], tuple[int, int], "length 2", id="empty-for-a-pair"),
        pytest.param([["LEAK", 2], ["LEAK"]], tuple[tuple[int, int], ...], "length 2", id="short-inner-pair"),
        pytest.param([["LEAK", "LEAK", "LEAK"]], list[tuple[int, int]], "length 2", id="long-pair-in-a-list"),
        pytest.param({"p": ["LEAK"]}, dict[str, tuple[int, int]], "length 2", id="short-pair-in-a-dict"),
        # a union reports that no member matched, not why each one failed
        pytest.param(["LEAK"], Optional[tuple[int, int]], "Unsupported types for deserialization", id="optional-pair"),
        pytest.param(
            ["LEAK", "LEAK"],
            tuple[int, str, float] | None,
            "Unsupported types for deserialization",
            id="optional-triple",
        ),
    ],
)
def test_tuple_of_the_wrong_length_raises_type_error_without_the_data(data, annotation, message):
    with pytest.raises(TypeError, match=message) as caught:
        deserialize_json(data, annotation)

    assert "LEAK" not in str(caught.value)


def test_fixed_tuple_length_error_reports_both_lengths():
    with pytest.raises(TypeError, match=r"\b3\b.*\b2\b"):
        deserialize_json([1, "a"], tuple[int, str, float])


def test_tuple_dataclass_field_of_the_wrong_length_raises_type_error():
    assert_round_trip(Span((1, 2)), Span)

    for bounds in ([1], [1, 2, 3]):
        with pytest.raises(TypeError, match="length 2"):
            deserialize_json({"bounds": bounds}, Span)


def test_fixed_tuple_with_a_wrongly_shaped_member_raises_type_error():
    with pytest.raises(TypeError, match="Expected a JSON array for"):
        deserialize_json(["x", 1], tuple[list[int], int])


def test_empty_tuple_annotation_rejects_a_non_empty_array():
    with pytest.raises(TypeError):
        deserialize_json([1], tuple[()])


@pytest.mark.parametrize("annotation", [tuple[()], typing.Tuple[()]], ids=["tuple", "typing-Tuple"])
def test_empty_tuple_annotation_is_a_fixed_length_of_zero(annotation):
    # unlike the bare tuple, which takes any array; get_args() of both is () from Python 3.11
    assert_strictly_equal(deserialize_json([], annotation), ())
    for data in ([1], [None], ["x", 2]):
        with pytest.raises(TypeError, match="length 0"):
            deserialize_json(data, annotation)

    assert_strictly_equal(deserialize_json([1, 2], tuple), (1, 2))
    assert_strictly_equal(deserialize_json([1, 2], typing.Tuple), (1, 2))
    assert_strictly_equal(deserialize_json([], Union[tuple[()], list[int]]), ())
    assert_strictly_equal(deserialize_json([1], Union[tuple[()], list[int]]), [1])


def test_bare_tuple_annotation_returns_inner_lists_limitation():
    # Documents current behavior, not a promise: with no member types there is nothing that could turn a JSON array
    # back into a tuple, so a tuple inside a bare `tuple` comes back as a list.
    restored = deserialize_json(to_json_value(((1, 2), "x")), tuple)

    assert_strictly_equal(restored, ([1, 2], "x"))


def test_named_tuple_restores_its_values_positionally():
    restored = deserialize_json(to_json_value(Vec(1, 2)), Vec)

    assert restored == (1, 2)
    assert tuple(restored) == (1, 2)


_NAMED_TUPLE_DEFECT = _xfail("a NamedTuple target restores a plain tuple (and ignores the field types)")

NAMED_TUPLE_TARGETS = [
    pytest.param(Vec(1, 2), Vec, id="root", marks=_NAMED_TUPLE_DEFECT),
    pytest.param([Vec(1, 2), Vec(3, 4)], list[Vec], id="list-of-named-tuples", marks=_NAMED_TUPLE_DEFECT),
    pytest.param({"a": Vec(0, 0)}, dict[str, Vec], id="dict-value", marks=_NAMED_TUPLE_DEFECT),
    pytest.param(Stamp(date(2024, 5, 17), frozenset({"a"})), Stamp, id="field-types", marks=_NAMED_TUPLE_DEFECT),
]


@pytest.mark.parametrize(("value", "annotation"), NAMED_TUPLE_TARGETS)
def test_named_tuple_target_restores_the_named_tuple(value, annotation):
    assert_round_trip(value, annotation)


# --- sets and frozensets -----------------------------------------------------------------------------------------


SET_GOLDENS = [
    pytest.param(set(), "[]", id="empty-set"),
    pytest.param(frozenset(), "[]", id="empty-frozenset"),
    pytest.param({5}, "[5]", id="singleton"),
    pytest.param({3, 1, 2}, "[1, 2, 3]", id="ints"),
    pytest.param({-1, 1, 0}, "[-1, 0, 1]", id="negative-sorts-first-as-text"),
    pytest.param({10, 9, 100, 1}, "[1, 10, 100, 9]", id="ints-sort-as-text"),
    pytest.param({"b", "a", "c"}, '["a", "b", "c"]', id="strs"),
    pytest.param({"", " ", "a"}, '[" ", "", "a"]', id="empty-and-blank-strs"),
    pytest.param({"b", "B", "a", "A"}, '["A", "B", "a", "b"]', id="upper-case-sorts-before-lower-case"),
    pytest.param({True, False}, "[false, true]", id="bools"),
    pytest.param({1, "a", None, 2, 3.5}, '["a", 1, 2, 3.5, null]', id="mixed-with-null"),
    pytest.param(
        {(2, "b"), (10, "a"), (2, "a")},
        '[[10, "a"], [2, "a"], [2, "b"]]',
        id="tuples",
    ),
    pytest.param(
        frozenset({frozenset({2, 1}), frozenset(), frozenset({3})}),
        "[[1, 2], [3], []]",
        id="frozenset-of-frozensets",
    ),
    pytest.param({Color.RED, Color.BLUE}, '["blue", "red"]', id="enums"),
    pytest.param({Point(2, 1), Point(1, 5)}, '[{"x": 1, "y": 5}, {"x": 2, "y": 1}]', id="frozen-dataclasses"),
    pytest.param({float("inf"), float("-inf"), 1.0}, "[-Infinity, 1.0, Infinity]", id="non-finite-floats"),
    pytest.param(Tags({"b", "a"}), '["a", "b"]', id="set-subclass"),
    pytest.param(FrozenTags({"b", "a"}), '["a", "b"]', id="frozenset-subclass"),
]


@pytest.mark.parametrize(("value", "text"), SET_GOLDENS)
def test_set_serialization_golden(value, text):
    # a set and a frozenset of the same members are written identically
    assert serialize_json(value) == text
    assert serialize_json(set(value)) == text
    assert serialize_json(frozenset(value)) == text


SET_ROUND_TRIPS = [
    pytest.param(set(), set[int], id="empty-set"),
    pytest.param(set(), set, id="empty-bare-set"),
    pytest.param(frozenset(), frozenset[str], id="empty-frozenset"),
    pytest.param({1}, set[int], id="singleton"),
    pytest.param({3, 1, 2}, set[int], id="ints"),
    pytest.param(frozenset({3, 1, 2}), frozenset[int], id="frozenset-of-ints"),
    pytest.param({"a", "b"}, set[str], id="strs"),
    pytest.param({"a", "b"}, typing.Set[str], id="typing-Set"),
    pytest.param(frozenset({"a"}), typing.FrozenSet[str], id="typing-FrozenSet"),
    pytest.param({True, False}, set[bool], id="bools"),
    pytest.param({1.5, 2.0, float("inf")}, set[float], id="floats"),
    pytest.param({1, "a"}, set[Union[int, str]], id="mixed-int-and-str"),
    pytest.param({None, 1, 2}, set[Optional[int]], id="none-member"),
    pytest.param({(1, "a"), (2, "b")}, set[tuple[int, str]], id="tuples"),
    pytest.param({((1, 2), "a")}, set[tuple[tuple[int, int], str]], id="nested-tuples"),
    pytest.param(
        frozenset({frozenset({1, 2}), frozenset(), frozenset({3})}),
        frozenset[frozenset[int]],
        id="frozenset-of-frozensets",
    ),
    pytest.param({frozenset({1}), frozenset({2, 3})}, set[frozenset[int]], id="set-of-frozensets"),
    pytest.param({Point(1, 2), Point(3, 4)}, set[Point], id="frozen-dataclasses"),
    pytest.param(frozenset({Point(1, 2)}), frozenset[Point], id="frozenset-of-dataclasses"),
    pytest.param(
        {Tagged("b", frozenset(), ()), Tagged("a", frozenset({"y", "x"}), (1, 2))},
        set[Tagged],
        id="dataclasses-with-container-fields",
    ),
    pytest.param({Color.RED, Color.BLUE}, set[Color], id="enums"),
    pytest.param({(Point(1, 2), "a")}, set[tuple[Point, str]], id="tuples-with-dataclasses"),
    pytest.param(Tags({"a", "b"}), Tags, id="set-subclass"),
    pytest.param(FrozenTags({"a", "b"}), FrozenTags, id="frozenset-subclass"),
]


@pytest.mark.parametrize(("value", "annotation"), SET_ROUND_TRIPS)
def test_set_round_trip(value, annotation):
    assert_round_trip(value, annotation)


def test_set_and_frozenset_target_types_are_preserved():
    data = to_json_value({1, 2})

    assert type(deserialize_json(data, set[int])) is set
    assert type(deserialize_json(data, frozenset[int])) is frozenset
    assert type(deserialize_json(data, set)) is set
    assert type(deserialize_json(data, frozenset)) is frozenset


def test_set_from_foreign_json_drops_duplicates_and_ignores_order():
    assert_strictly_equal(deserialize_json([3, 1, 2, 1, 3], set[int]), {1, 2, 3})
    assert_strictly_equal(deserialize_json([[2, 1], [1, 2]], set[tuple[int, int]]), {(2, 1), (1, 2)})


@pytest.mark.parametrize(
    ("data", "annotation"),
    [
        pytest.param([[1]], set[list[int]], id="list-members"),
        pytest.param([{"a": 1}], set, id="dict-members-in-a-bare-set"),
        pytest.param([[1, 2]], frozenset[set[int]], id="set-members"),
    ],
)
def test_unhashable_members_raise_type_error(data, annotation):
    # the message comes from Python and differs between versions, but always says "unhashable"
    with pytest.raises(TypeError, match="unhashable"):
        deserialize_json(data, annotation)


SET_FAMILIES = [
    pytest.param(list(range(150)), set[int], id="ints-150"),
    pytest.param([f"s{i}" for i in range(120)], set[str], id="strs-120"),
    pytest.param([(i % 7, f"v{i}") for i in range(90)], set[tuple[int, str]], id="tuples-90"),
    pytest.param(list(range(50)) + [str(i) for i in range(50)], set[Union[int, str]], id="ints-and-strs-100"),
    pytest.param([Point(i % 13, i) for i in range(100)], set[Point], id="dataclasses-100"),
    pytest.param([frozenset({i, i + 1}) for i in range(60)], set[frozenset[int]], id="frozensets-60"),
    pytest.param([0, 8, 16, 24, 32, 40, 48, 56], set[int], id="colliding-int-hashes"),
]


def _json_form(member):
    if isinstance(member, Point):
        return {"x": member.x, "y": member.y}
    if isinstance(member, tuple):
        return [_json_form(item) for item in member]
    if isinstance(member, frozenset):
        return sorted((_json_form(item) for item in member), key=_canonical)
    return member


@pytest.mark.parametrize(("members", "annotation"), SET_FAMILIES)
def test_set_text_is_independent_of_insertion_order(members, annotation):
    expected_text = json.dumps(sorted(map(_json_form, members), key=_canonical))

    texts = set()
    for order in _insertion_orders(members):
        built = set()
        for member in order:
            built.add(member)
        texts.add(serialize_json(built))
        texts.add(serialize_json(frozenset(order)))

    assert texts == {expected_text}
    assert_round_trip(set(members), annotation)
    assert_round_trip(frozenset(members), frozenset[typing.get_args(annotation)[0]])


def test_colliding_hash_slots_do_not_leak_iteration_order_into_the_text():
    # 0, 8, 16, ... share a slot in a small table, so the iteration order of the set depends on insertion order
    members = [0, 8, 16, 24, 32, 40, 48, 56]
    texts = {serialize_json(set(order)) for order in _insertion_orders(members)}
    texts |= {serialize_json(set(reversed(members))), serialize_json(set(members))}

    assert texts == {"[0, 16, 24, 32, 40, 48, 56, 8]"}


def test_sets_nested_in_other_containers_are_ordered_too():
    value = {
        "list": [{10, 9}, frozenset({"b", "a"}), set()],
        "tuple": ({2, 11}, {(2, 1), (10, 1)}),
        "dict": {"k": frozenset({Color.RED, Color.BLUE})},
        "holder": Holder({3, 20}),
    }
    expected = {
        "list": [[10, 9], ["a", "b"], []],
        "tuple": [[11, 2], [[10, 1], [2, 1]]],
        "dict": {"k": ["blue", "red"]},
        "holder": {"payload": [20, 3]},
    }

    assert serialize_json(value) == json.dumps(expected)


@dataclass
class SetFields:
    numbers: set[int]
    words: frozenset[str]
    pairs: set[tuple[int, int]]
    groups: dict[str, set[int]]
    rows: list[frozenset[int]]
    maybe: Optional[set[str]]


def test_set_fields_of_a_dataclass():
    value = SetFields(
        numbers={10, 9, 1},
        words=frozenset({"b", "a"}),
        pairs={(2, 1), (10, 1)},
        groups={"even": {4, 2}, "none": set()},
        rows=[frozenset({2, 11}), frozenset()],
        maybe=None,
    )
    expected = {
        "numbers": [1, 10, 9],
        "words": ["a", "b"],
        "pairs": [[10, 1], [2, 1]],
        "groups": {"even": [2, 4], "none": []},
        "rows": [[11, 2], []],
        "maybe": None,
    }

    assert serialize_json(value) == json.dumps(expected)
    assert_round_trip(value, SetFields)


def test_set_order_does_not_depend_on_encoding_options():
    value = {"z", "é", "\U0001f600", "a"}

    plain = to_json_value(value)
    assert to_json_value(value, ensure_ascii=False) == plain
    assert to_json_value(value, indent=2) == plain
    assert to_json_value(value, sort_keys=True, separators=(",", ":")) == plain
    assert_round_trip(value, set[str], ensure_ascii=False)


def test_set_text_honors_indent_and_compact_separators():
    value = {"a": [], "b": {}, "c": {2, 1}, "d": {(2, 1), (10, 1)}}

    assert serialize_json(value, indent=2) == json.dumps(
        {"a": [], "b": {}, "c": [1, 2], "d": [[10, 1], [2, 1]]}, indent=2
    )
    assert serialize_json({"s": {3, 1, 2}}, separators=(",", ":")) == '{"s":[1,2,3]}'


def test_set_with_a_non_finite_float_needs_allow_nan():
    assert serialize_json({float("inf"), 1.0}) == "[1.0, Infinity]"
    with pytest.raises(ValueError, match="Out of range float values"):
        serialize_json({float("inf"), 1.0}, allow_nan=False)


def test_sets_of_dataclasses_with_container_fields_are_canonical_all_the_way_down():
    members = {Tagged("b", frozenset(), ()), Tagged("a", frozenset({"y", "x", "z"}), (2, 1))}
    expected = [
        {"name": "a", "tags": ["x", "y", "z"], "path": [2, 1]},
        {"name": "b", "tags": [], "path": []},
    ]

    # compared as parsed JSON: the keys of a set member are sorted (see the *_limitation test below)
    assert to_json_value(members) == expected
    assert to_json_value(frozenset(members)) == expected


def test_set_of_non_ascii_members_orders_by_escaped_text_limitation():
    # Documents current behavior, not a promise: members are ordered by their ASCII-escaped canonical text, so "é"
    # ('"\\u00e9"') sorts before "z" ('"z"') although its code point is higher.
    assert to_json_value({"z", "é", "\U0001f600"}) == ["é", "\U0001f600", "z"]


def test_object_members_of_a_set_have_sorted_keys_limitation():
    # Documents current behavior, not a promise: a set member is re-parsed from its canonical (sorted-keys) text, so
    # a dataclass or a mapping inside a set loses its order of keys, unlike the same value inside a list.
    assert serialize_json([YX(1, 2)]) == '[{"y": 1, "x": 2}]'
    assert serialize_json({YX(1, 2)}) == '[{"x": 2, "y": 1}]'
    assert serialize_json([FrozenMap({"b": 1, "a": 2})]) == '[{"b": 1, "a": 2}]'
    assert serialize_json({FrozenMap({"b": 1, "a": 2})}) == '[{"a": 2, "b": 1}]'


# --- dict and the Mapping family ---------------------------------------------------------------------------------


DICT_GOLDENS = [
    pytest.param({}, "{}", id="empty"),
    pytest.param({"a": 1}, '{"a": 1}', id="singleton"),
    pytest.param({"z": 1, "a": 2, "m": 3}, '{"z": 1, "a": 2, "m": 3}', id="insertion-order"),
    pytest.param({"": 1}, '{"": 1}', id="empty-key"),
    pytest.param({" ": 1, "  ": 2}, '{" ": 1, "  ": 2}', id="whitespace-keys"),
    pytest.param({"a": {"b": {"c": {"d": 1}}}}, '{"a": {"b": {"c": {"d": 1}}}}', id="four-levels"),
    pytest.param({"a": [], "b": {}, "c": [{}]}, '{"a": [], "b": {}, "c": [{}]}', id="empty-containers"),
    pytest.param(
        {"": 1, " ": 2, "\t": 3, "é": 4, "שלום": 5, "\U0001f600": 6},
        '{"": 1, " ": 2, "\\t": 3, "\\u00e9": 4, "\\u05e9\\u05dc\\u05d5\\u05dd": 5, "\\ud83d\\ude00": 6}',
        id="unicode-keys-escaped",
    ),
    pytest.param(
        {'a"b': 1, "c\\d": 2, "e\nf": 3},
        '{"a\\"b": 1, "c\\\\d": 2, "e\\nf": 3}',
        id="quote-backslash-newline",
    ),
]


@pytest.mark.parametrize(("value", "text"), DICT_GOLDENS)
def test_dict_serialization_golden(value, text):
    assert serialize_json(value) == text == json.dumps(value)
    assert list(to_json_value(value)) == list(value)


def test_unicode_keys_are_written_raw_without_ensure_ascii():
    value = {"": 1, " ": 2, "\t": 3, "é": 4, "שלום": 5, "\U0001f600": 6}

    assert serialize_json(value, ensure_ascii=False) == (
        '{"": 1, " ": 2, "\\t": 3, "é": 4, "שלום": 5, "\U0001f600": 6}'
    )
    assert_round_trip(value, dict[str, int], ensure_ascii=False)


DICT_ROUND_TRIPS = [
    pytest.param({}, dict[str, int], id="empty"),
    pytest.param({}, dict, id="empty-bare"),
    pytest.param({"a": 1}, dict[str, int], id="singleton"),
    pytest.param(
        {"": 1, " ": 2, "\t": 3, "é": 4, "שלום": 5, "\U0001f600": 6, 'a"b': 7, "c\\d": 8},
        dict[str, int],
        id="awkward-keys",
    ),
    pytest.param({"a": 1, "b": 2}, typing.Dict[str, int], id="typing-Dict"),
    pytest.param({"a": 1.5, "b": [1, {"k": None}]}, dict[str, Any], id="values-any"),
    pytest.param({"a": 1.5, "b": [1, {"k": None}]}, dict, id="bare-dict"),
    pytest.param(
        {"a": [{"x": (1, 2), "y": ()}, {}], "b": []},
        dict[str, list[dict[str, tuple[int, ...]]]],
        id="dict-of-lists-of-dicts-of-tuples",
    ),
    pytest.param(
        {"a": {"b": {"c": {"d": 1, "e": 2}, "f": {}}, "g": {}}, "h": {}},
        dict[str, dict[str, dict[str, dict[str, int]]]],
        id="four-levels",
    ),
    pytest.param({"a": {1, 2}, "b": set()}, dict[str, set[int]], id="set-values"),
    pytest.param({"a": frozenset({"x"})}, dict[str, frozenset[str]], id="frozenset-values"),
    pytest.param({"p": Point(1, 2)}, dict[str, Point], id="dataclass-values"),
    pytest.param({"a": None, "b": 1}, dict[str, Optional[int]], id="optional-values"),
    pytest.param({"a": None, "b": []}, dict[str, Optional[list[int]]], id="optional-list-values"),
    pytest.param({"a": 1, "b": [2]}, dict[str, int | list[int]], id="int-or-list-values"),
    pytest.param({"a": 1}, Mapping[str, int], id="Mapping-annotation"),
    pytest.param({"a": [1, 2]}, typing.Mapping[str, list[int]], id="typing-Mapping"),
    pytest.param(MappingProxyType({"a": 1}), MappingProxyType[str, int], id="mappingproxy-typed"),
    pytest.param(MappingProxyType({"a": [1, {"b": 2}]}), MappingProxyType, id="mappingproxy-untyped"),
    pytest.param(
        MappingProxyType({"a": MappingProxyType({"b": [1, 2]})}),
        MappingProxyType[str, MappingProxyType[str, list[int]]],
        id="mappingproxy-of-mappingproxy",
    ),
    pytest.param(
        MappingProxyType({"a": (1, 2), "b": ()}),
        MappingProxyType[str, tuple[int, ...]],
        id="mappingproxy-of-tuples",
    ),
]


@pytest.mark.parametrize(("value", "annotation"), DICT_ROUND_TRIPS)
def test_dict_round_trip(value, annotation):
    assert_round_trip(value, annotation)


def test_restored_mappingproxy_is_read_only():
    restored = deserialize_json({"a": 1}, MappingProxyType[str, int])

    assert type(restored) is MappingProxyType
    with pytest.raises(TypeError):
        restored["b"] = 2


def test_wide_dict_keeps_every_key_in_order():
    value = {f"key-{index}": index for index in range(2000)}

    assert serialize_json(value) == json.dumps(value)
    restored = assert_round_trip(value, dict[str, int])
    assert list(restored) == list(value)


def test_insertion_order_survives_reordering_operations():
    ordered = OrderedDict([("b", 1), ("a", 2), ("c", 3)])
    ordered.move_to_end("b")
    reinserted = {"a": 1, "b": 2}
    del reinserted["a"]
    reinserted["a"] = 3

    assert serialize_json(ordered) == '{"a": 2, "c": 3, "b": 1}'
    assert serialize_json(reinserted) == '{"b": 2, "a": 3}'
    assert list(deserialize_json(to_json_value(ordered), dict[str, int])) == ["a", "c", "b"]


MAPPING_OBJECTS = [
    pytest.param(
        OrderedDict([("b", 1), ("a", 2)]), '{"b": 1, "a": 2}', dict[str, int], id="OrderedDict"
    ),
    pytest.param(
        defaultdict(list, {"b": [1], "a": []}), '{"b": [1], "a": []}', dict[str, list[int]], id="defaultdict"
    ),
    pytest.param(Counter("abca"), '{"a": 2, "b": 1, "c": 1}', dict[str, int], id="Counter"),
    pytest.param(UserDict({"z": 1, "a": 2}), '{"z": 1, "a": 2}', dict[str, int], id="UserDict"),
    pytest.param(
        UserDict({"inner": UserDict({"k": [1]})}),
        '{"inner": {"k": [1]}}',
        dict[str, dict[str, list[int]]],
        id="UserDict-of-UserDict",
    ),
    pytest.param(
        MappingProxyType({"b": 1, "a": 2}), '{"b": 1, "a": 2}', dict[str, int], id="MappingProxyType"
    ),
    pytest.param(FrozenMap({"b": 1, "a": 2}), '{"b": 1, "a": 2}', dict[str, int], id="custom-Mapping"),
]


@pytest.mark.parametrize(("value", "text", "annotation"), MAPPING_OBJECTS)
def test_mapping_objects_serialize_as_objects_and_restore_as_plain_dicts(value, text, annotation):
    assert serialize_json(value) == text

    restored = deserialize_json(json.loads(text), annotation)

    assert_strictly_equal(restored, json.loads(text))
    assert list(restored) == list(value)


def test_chain_map_serializes_the_visible_view():
    # The front map wins; only the parsed value is compared, as the key order of a ChainMap is not the point here
    value = ChainMap({"a": 1}, {"a": 9, "b": 2})

    assert to_json_value(value) == {"a": 1, "b": 2}
    assert_strictly_equal(deserialize_json(to_json_value(value), dict[str, int]), dict(value))


def test_mapping_objects_nest_inside_each_other():
    value = {
        "ordered": OrderedDict([("k", Counter("aab"))]),
        "chain": ChainMap({"x": UserDict({"y": 1})}),
        "proxy": MappingProxyType({"d": defaultdict(int, {"z": 1})}),
        "custom": FrozenMap({"m": FrozenMap({"n": [FrozenMap({})]})}),
    }
    expected = {
        "ordered": {"k": {"a": 2, "b": 1}},
        "chain": {"x": {"y": 1}},
        "proxy": {"d": {"z": 1}},
        "custom": {"m": {"n": [{}]}},
    }

    assert serialize_json(value) == json.dumps(expected)
    assert_strictly_equal(deserialize_json(to_json_value(value), dict[str, Any]), expected)


def test_a_custom_mapping_converts_its_keys_like_a_dict():
    value = FrozenMap({1: FrozenMap({(1, "a"): FrozenMap({Color.RED: [True]})}), 2.5: None})

    assert serialize_json(value) == '{"1": {"[1, \\"a\\"]": {"red": [true]}}, "2.5": null}'


@pytest.mark.parametrize(
    ("value", "annotation"),
    [
        pytest.param(OrderedDict([("b", 1), ("a", 2)]), OrderedDict[str, int], id="OrderedDict-typed"),
        pytest.param(OrderedDict([("b", 1), ("a", 2)]), OrderedDict, id="OrderedDict-bare"),
        pytest.param(Counter("abca"), Counter, id="Counter-bare"),
        pytest.param(defaultdict(list, {"a": [1]}), defaultdict[str, list[int]], id="defaultdict"),
        pytest.param(UserDict({"a": 1}), UserDict, id="UserDict"),
        pytest.param(ChainMap({"a": 1}, {"b": 2}), ChainMap[str, int], id="ChainMap"),
    ],
)
def test_mapping_subclass_targets_restore_a_plain_dict_limitation(value, annotation):
    # Documents current behavior, not a promise: whatever mapping class is annotated, a JSON object is restored as a
    # plain dict (like a list or tuple subclass target, unlike a set subclass). Building the subclass would be
    # reasonable for an OrderedDict or a Counter, but a defaultdict needs a factory and a ChainMap its maps, which
    # JSON does not carry. The order of the keys is kept.
    restored = deserialize_json(to_json_value(value), annotation)

    assert_strictly_equal(restored, dict(value))
    assert list(restored) == list(value)


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(Counter[str], id="Counter"),
        pytest.param(Optional[Counter[str]], id="optional-Counter"),
    ],
)
def test_counter_with_a_single_type_argument_is_restored(annotation):
    # Bare `Counter` restores {"a": 1} (see above), and Counter[str] is its natural annotation
    restored = deserialize_json({"a": 1}, annotation)

    assert restored == {"a": 1}
    assert isinstance(restored, dict)


def test_abstract_sequence_annotations_are_unsupported_limitation():
    # Documents current behavior, not a promise: only list, tuple, set and frozenset (and their typing aliases) have
    # handlers, so an abstract Sequence or Iterable annotation is rejected with TypeError.
    for annotation in (typing.Sequence[int], typing.Iterable[int], typing.Collection[int]):
        with pytest.raises(TypeError, match="Unsupported type for deserialization"):
            deserialize_json([1, 2], annotation)


# --- sort_keys and encoding options on nested containers ---------------------------------------------------------


def test_sort_keys_sorts_every_nested_object_but_no_array():
    value = {
        "b": [{"z": 1, "a": 2}],
        "a": ({"y": 1, "x": 2},),
        "c": MappingProxyType({"q": 1, "p": 2}),
        "d": OrderedDict([("n", 1), ("m", 2)]),
        "e": Holder({"v": 1, "u": 2}),
        "f": [3, 1, 2],
        "g": (3, 1, 2),
        "h": {2, 1},
    }
    plain = {
        "b": [{"z": 1, "a": 2}],
        "a": [{"y": 1, "x": 2}],
        "c": {"q": 1, "p": 2},
        "d": {"n": 1, "m": 2},
        "e": {"payload": {"v": 1, "u": 2}},
        "f": [3, 1, 2],
        "g": [3, 1, 2],
        "h": [1, 2],
    }

    assert serialize_json(value, sort_keys=True) == json.dumps(plain, sort_keys=True)
    assert serialize_json(value) == json.dumps(plain)


def test_sort_keys_orders_converted_keys_by_their_text():
    value = {10: "a", 2: "b", "x": "c", True: "d"}

    assert serialize_json(value) == '{"10": "a", "2": "b", "x": "c", "true": "d"}'
    assert serialize_json(value, sort_keys=True) == '{"10": "a", "2": "b", "true": "d", "x": "c"}'


def test_sort_keys_sorts_dataclass_fields_and_tuple_keys():
    assert serialize_json(Point(2, 1), sort_keys=True) == '{"x": 2, "y": 1}'
    assert serialize_json({(2, 1): "b", (1, 2): "a"}, sort_keys=True) == '{"[1, 2]": "a", "[2, 1]": "b"}'


OPTION_SETS = [
    pytest.param({}, id="defaults"),
    pytest.param({"sort_keys": True}, id="sort_keys"),
    pytest.param({"indent": 2}, id="indent"),
    pytest.param({"separators": (",", ":")}, id="compact"),
    pytest.param({"ensure_ascii": False}, id="ensure_ascii-false"),
    pytest.param({"allow_nan": False}, id="allow_nan-false"),
    pytest.param(
        {"sort_keys": True, "indent": "\t", "ensure_ascii": False, "separators": (",", ": ")}, id="combined"
    ),
]


@dataclass
class Shelf:
    label: str
    items: list[str]
    counts: dict[str, int]
    tags: set[str]
    codes: frozenset[int]
    size: tuple[int, int]
    route: tuple[Point, ...]
    by_color: dict[Color, list[Point]]
    ledger: dict[str, dict[str, list[int]]]
    pairs: list[tuple[str, int]]
    spare: list[str] | None
    view: MappingProxyType[str, tuple[int, ...]]


@dataclass
class Warehouse:
    name: str
    shelves: list[Shelf]
    by_label: dict[str, Shelf]
    unlabelled: tuple[Shelf, ...]
    index: dict[str, set[Point]]


def _shelf(label):
    return Shelf(
        label=label,
        items=["bolt", "nut"],
        counts={"bolt": 3, "nut": 0},
        tags={"b", "a"},
        codes=frozenset({10, 9}),
        size=(2, 3),
        route=(Point(0, 0), Point(1, 2)),
        by_color={Color.RED: [Point(1, 1)], Color.BLUE: []},
        ledger={"in": {"mon": [1, 2], "tue": []}, "out": {}},
        pairs=[("a", 1), ("b", 2)],
        spare=None,
        view=MappingProxyType({"w": (1, 2), "x": ()}),
    )


def _shelf_json(label):
    return {
        "label": label,
        "items": ["bolt", "nut"],
        "counts": {"bolt": 3, "nut": 0},
        "tags": ["a", "b"],
        "codes": [10, 9],
        "size": [2, 3],
        "route": [{"x": 0, "y": 0}, {"x": 1, "y": 2}],
        "by_color": {"red": [{"x": 1, "y": 1}], "blue": []},
        "ledger": {"in": {"mon": [1, 2], "tue": []}, "out": {}},
        "pairs": [["a", 1], ["b", 2]],
        "spare": None,
        "view": {"w": [1, 2], "x": []},
    }


def _warehouse():
    return Warehouse(
        name="Wé",
        shelves=[_shelf("s1"), _shelf("s2")],
        by_label={"s1": _shelf("s1")},
        unlabelled=(_shelf(""),),
        index={"origin": {Point(1, 0), Point(0, 0)}, "none": set()},
    )


def _warehouse_json():
    return {
        "name": "Wé",
        "shelves": [_shelf_json("s1"), _shelf_json("s2")],
        "by_label": {"s1": _shelf_json("s1")},
        "unlabelled": [_shelf_json("")],
        "index": {"origin": [{"x": 0, "y": 0}, {"x": 1, "y": 0}], "none": []},
    }


def test_shelf_with_every_container_kind_matches_the_hand_written_json():
    assert serialize_json(_shelf("s1")) == json.dumps(_shelf_json("s1"))
    assert_round_trip(_shelf("s1"), Shelf)


def test_warehouse_matches_the_hand_written_json_and_round_trips():
    assert serialize_json(_warehouse()) == json.dumps(_warehouse_json())
    assert_round_trip(_warehouse(), Warehouse)


@pytest.mark.parametrize("options", OPTION_SETS)
def test_warehouse_text_matches_the_stdlib_for_every_option_set(options):
    assert serialize_json(_warehouse(), **options) == json.dumps(_warehouse_json(), **options)
    assert_round_trip(_warehouse(), Warehouse, **options)


@pytest.mark.parametrize(
    ("wrap", "annotation"),
    [
        pytest.param(lambda w: [w], list[Warehouse], id="list"),
        pytest.param(lambda w: (w,), tuple[Warehouse, ...], id="tuple"),
        pytest.param(lambda w: {"w": w}, dict[str, Warehouse], id="dict"),
        pytest.param(lambda w: w, Optional[Warehouse], id="optional"),
        pytest.param(lambda w: {"k": [(w,)]}, dict[str, list[tuple[Warehouse, ...]]], id="deep"),
    ],
)
def test_warehouse_nested_in_another_container(wrap, annotation):
    assert_round_trip(wrap(_warehouse()), annotation)


def test_shelf_with_every_container_empty():
    empty = Shelf(
        label="",
        items=[],
        counts={},
        tags=set(),
        codes=frozenset(),
        size=(0, 0),
        route=(),
        by_color={},
        ledger={},
        pairs=[],
        spare=[],
        view=MappingProxyType({}),
    )
    expected = {
        "label": "",
        "items": [],
        "counts": {},
        "tags": [],
        "codes": [],
        "size": [0, 0],
        "route": [],
        "by_color": {},
        "ledger": {},
        "pairs": [],
        "spare": [],
        "view": {},
    }

    assert serialize_json(empty) == json.dumps(expected)
    assert serialize_json(empty, indent=2) == json.dumps(expected, indent=2)
    assert_round_trip(empty, Shelf)


def test_many_shelves_with_container_fields_round_trip():
    shelves = [_shelf(f"s{index}") for index in range(250)]
    expected = [_shelf_json(f"s{index}") for index in range(250)]

    assert serialize_json(shelves) == json.dumps(expected)
    assert_round_trip(shelves, list[Shelf])
    assert_round_trip({shelf.label: shelf for shelf in shelves}, dict[str, Shelf])


# --- mapping keys: serialization ---------------------------------------------------------------------------------


KEY_GOLDENS = [
    pytest.param("k", '"k"', id="str"),
    pytest.param(Shout("k"), '"k"', id="str-subclass"),
    pytest.param(0, '"0"', id="zero"),
    pytest.param(7, '"7"', id="int"),
    pytest.param(-3, '"-3"', id="negative-int"),
    pytest.param(10**20, '"100000000000000000000"', id="huge-int"),
    pytest.param(1.5, '"1.5"', id="float"),
    pytest.param(-2.25, '"-2.25"', id="negative-float"),
    pytest.param(100.0, '"100.0"', id="whole-float"),
    pytest.param(1e100, '"1e+100"', id="big-float"),
    pytest.param(1e-7, '"1e-07"', id="small-float"),
    pytest.param(1e16, '"1e+16"', id="float-that-needs-an-exponent"),
    pytest.param(0.1 + 0.2, '"0.30000000000000004"', id="float-needing-17-digits"),
    pytest.param(-0.0, '"-0.0"', id="negative-zero"),
    pytest.param(float("nan"), '"NaN"', id="nan"),
    pytest.param(float("inf"), '"Infinity"', id="inf"),
    pytest.param(float("-inf"), '"-Infinity"', id="negative-inf"),
    pytest.param(True, '"true"', id="true"),
    pytest.param(False, '"false"', id="false"),
    pytest.param(Color.RED, '"red"', id="str-enum"),
    pytest.param(Label.A, '"a"', id="str-mixin-enum"),
    pytest.param(Level.HIGH, '"2"', id="int-enum"),
    pytest.param(Priority.LOW, '"1"', id="IntEnum"),
    pytest.param(Switch.ON, '"true"', id="bool-enum"),
    pytest.param(Ratio.HALF, '"0.5"', id="float-enum"),
    pytest.param(Wrapper.WRAPPED, '"x"', id="enum-whose-value-is-an-enum"),
    pytest.param(Cell.ORIGIN, '"[0, 0]"', id="enum-whose-value-is-a-tuple"),
    pytest.param((), '"[]"', id="empty-tuple"),
    pytest.param((1,), '"[1]"', id="one-tuple"),
    pytest.param((1, "a"), '"[1, \\"a\\"]"', id="tuple"),
    pytest.param(((1, 2), (3,), ()), '"[[1, 2], [3], []]"', id="nested-tuples"),
    pytest.param((Color.RED, True, None, 1.5), '"[\\"red\\", true, null, 1.5]"', id="tuple-with-enum-bool-none-float"),
    pytest.param((Level.LOW, Priority.HIGH), '"[1, 2]"', id="tuple-with-int-enums"),
    pytest.param((-0.0, 1e100), '"[-0.0, 1e+100]"', id="tuple-with-floats"),
    pytest.param((float("inf"),), '"[Infinity]"', id="tuple-with-inf"),
    pytest.param(Vec(1, 2), '"[1, 2]"', id="named-tuple"),
    pytest.param((Point(1, 2), "a"), '"[{\\"x\\": 1, \\"y\\": 2}, \\"a\\"]"', id="tuple-with-dataclass"),
    pytest.param(((Color.RED,), 1), '"[[\\"red\\"], 1]"', id="tuple-with-nested-enum-tuple"),
    pytest.param((Cell.ORIGIN,), '"[[0, 0]]"', id="tuple-with-tuple-valued-enum"),
    pytest.param((Wrapper.WRAPPED,), '"[\\"x\\"]"', id="tuple-with-enum-valued-enum"),
    pytest.param((frozenset({10, 9}),), '"[[10, 9]]"', id="tuple-with-frozenset"),
    pytest.param(
        (date(2024, 5, 17), Decimal("1.50")), '"[\\"2024-05-17\\", \\"1.50\\"]"', id="tuple-with-date-and-decimal"
    ),
    pytest.param((datetime(2024, 5, 17, 8, 30),), '"[\\"2024-05-17T08:30:00\\"]"', id="tuple-with-datetime"),
    pytest.param((b"ab",), '"[\\"YWI=\\"]"', id="tuple-with-bytes"),
    pytest.param((PurePosixPath("a/b"),), '"[\\"a/b\\"]"', id="tuple-with-path"),
]


@pytest.mark.parametrize(("key", "key_text"), KEY_GOLDENS)
def test_key_text_golden(key, key_text):
    assert serialize_json({key: 1}) == "{" + key_text + ": 1}"


# A container position and the same position written as plain JSON, to place one mapping in
POSITIONS = [
    pytest.param(lambda m: m, lambda m: m, id="root"),
    pytest.param(lambda m: [m], lambda m: [m], id="list"),
    pytest.param(lambda m: (m, m), lambda m: [m, m], id="tuple"),
    pytest.param(lambda m: {"k": m}, lambda m: {"k": m}, id="dict-value"),
    pytest.param(lambda m: Holder(m), lambda m: {"payload": m}, id="dataclass-field"),
    pytest.param(lambda m: MappingProxyType({"k": m}), lambda m: {"k": m}, id="mappingproxy-value"),
    pytest.param(lambda m: {"a": [{"b": (m,)}]}, lambda m: {"a": [{"b": [m]}]}, id="deep"),
]


@pytest.mark.parametrize(("place", "place_expected"), POSITIONS)
@pytest.mark.parametrize(("key", "key_text"), KEY_GOLDENS)
def test_key_text_is_the_same_wherever_the_mapping_sits(key, key_text, place, place_expected):
    expected = place_expected({json.loads(key_text): 1})

    assert serialize_json(place({key: 1})) == json.dumps(expected)
    assert to_json_value(place({key: 1})) == expected


@pytest.mark.parametrize(
    ("key", "text"),
    [
        pytest.param(float("nan"), '{"NaN": 1}', id="nan"),
        pytest.param(float("inf"), '{"Infinity": 1}', id="inf"),
        pytest.param(float("-inf"), '{"-Infinity": 1}', id="negative-inf"),
        pytest.param((float("nan"),), '{"[NaN]": 1}', id="tuple-with-nan"),
        pytest.param((1, (float("inf"),)), '{"[1, [Infinity]]": 1}', id="tuple-with-nested-inf"),
    ],
)
def test_non_finite_float_keys_need_allow_nan(key, text):
    assert serialize_json({key: 1}) == text
    assert serialize_json({key: 1}, allow_nan=True) == text
    with pytest.raises(ValueError, match="Out of range float values"):
        serialize_json({key: 1}, allow_nan=False)


def test_keys_that_look_alike_but_differ_do_not_collide():
    value = {1: "a", "01": "b", 1.5: "c", "1.50": "d", False: "e", "False": "f", (1,): "g", "[1,]": "h"}

    assert serialize_json(value) == (
        '{"1": "a", "01": "b", "1.5": "c", "1.50": "d", "false": "e", "False": "f", "[1]": "g", "[1,]": "h"}'
    )


def test_a_fallback_result_gets_its_keys_converted_and_its_sets_ordered_too():
    def fallback(value):
        return {1: {10, 9}, (1, "a"): frozenset({"b", "a"}), Color.RED: [False]}

    assert serialize_json({"x": object()}, fallback=fallback) == (
        '{"x": {"1": [10, 9], "[1, \\"a\\"]": ["a", "b"], "red": [false]}}'
    )
    with pytest.raises(ValueError, match="(?i)colli"):
        serialize_json({"x": object()}, fallback=lambda value: {1: "a", "1": "b"})


def test_equal_key_text_at_different_places_is_not_a_collision():
    assert serialize_json([{1: "a"}, {"1": "b"}]) == '[{"1": "a"}, {"1": "b"}]'
    assert serialize_json({"1": {1: "x"}, 2: {"2": "y"}}) == '{"1": {"1": "x"}, "2": {"2": "y"}}'


def test_tuple_key_text_ignores_separators_and_ensure_ascii_limitation():
    # Documents current behavior, not a promise: the text of a tuple key is always written with the default
    # separators and ASCII escapes, whatever options the surrounding document uses (the key stays restorable).
    assert serialize_json({(1, 2): 1}, separators=(",", ":")) == '{"[1, 2]":1}'
    assert serialize_json({("é",): 1}, ensure_ascii=False) == '{"[\\"\\\\u00e9\\"]": 1}'


UNSUPPORTED_KEYS = [
    pytest.param(None, "NoneType", None, id="None"),
    pytest.param(b"KEY_SENTINEL", "bytes", "KEY_SENTINEL", id="bytes"),
    pytest.param(frozenset({"KEY_SENTINEL"}), "frozenset", "KEY_SENTINEL", id="frozenset"),
    pytest.param(object(), "object", None, id="object"),
    pytest.param(LeakyKey(), "LeakyKey", "KEY_SENTINEL", id="custom-object"),
    pytest.param(datetime(2024, 5, 17, 8, 30), "datetime", "2024-05-17", id="datetime"),
    pytest.param(date(2024, 5, 17), "date", "2024-05-17", id="date"),
    pytest.param(Decimal("1.50"), "Decimal", "1.50", id="Decimal"),
    pytest.param(PurePosixPath("secret/dir"), "PurePosixPath", "secret/dir", id="path"),
    pytest.param(Point(7123, 8), "Point", "7123", id="dataclass"),
    pytest.param(1 + 2j, "complex", "2j", id="complex"),
]


def _names_the_type(type_name):
    """The message ends with the module-qualified name of the key's type: ``...key type: datetime.date``."""
    return rf"key type: [\w.]*\.{type_name}$"


@pytest.mark.parametrize(("key", "type_name", "payload"), UNSUPPORTED_KEYS)
def test_unsupported_key_raises_type_error_naming_only_the_type(key, type_name, payload):
    with pytest.raises(TypeError, match=_names_the_type(type_name)) as caught:
        serialize_json({key: 1})

    if payload is not None:
        assert payload not in str(caught.value)


@pytest.mark.parametrize(("key", "type_name", "payload"), UNSUPPORTED_KEYS)
def test_unsupported_key_raises_wherever_the_mapping_sits(key, type_name, payload):
    for place in (
        lambda m: [m],
        lambda m: Holder(m),
        lambda m: MappingProxyType({"k": m}),
        lambda m: {"a": [{"b": (m,)}]},
        lambda m: {FrozenMap(m)},
    ):
        for serializer in (serialize_json, serialize_json_as_obj):
            with pytest.raises(TypeError, match=_names_the_type(type_name)) as caught:
                serializer(place({key: 1}))

            if payload is not None:
                assert payload not in str(caught.value)


def test_unsupported_member_of_a_tuple_key_raises_type_error_naming_only_the_type():
    with pytest.raises(TypeError, match="LeakyKey") as caught:
        serialize_json({(1, LeakyKey()): 1})

    assert "KEY_SENTINEL" not in str(caught.value)


COLLISIONS = [
    pytest.param({1: "LEFT_PAYLOAD", "1": "RIGHT_PAYLOAD"}, id="int-vs-str"),
    pytest.param({-7: "LEFT_PAYLOAD", "-7": "RIGHT_PAYLOAD"}, id="negative-int-vs-str"),
    pytest.param({10**20: "LEFT_PAYLOAD", "100000000000000000000": "RIGHT_PAYLOAD"}, id="huge-int-vs-str"),
    pytest.param({True: "LEFT_PAYLOAD", "true": "RIGHT_PAYLOAD"}, id="true-vs-str"),
    pytest.param({False: "LEFT_PAYLOAD", "false": "RIGHT_PAYLOAD"}, id="false-vs-str"),
    pytest.param({1.0: "LEFT_PAYLOAD", "1.0": "RIGHT_PAYLOAD"}, id="float-vs-str"),
    pytest.param({1e100: "LEFT_PAYLOAD", "1e+100": "RIGHT_PAYLOAD"}, id="big-float-vs-str"),
    pytest.param({-0.0: "LEFT_PAYLOAD", "-0.0": "RIGHT_PAYLOAD"}, id="negative-zero-vs-str"),
    pytest.param({float("nan"): "LEFT_PAYLOAD", "NaN": "RIGHT_PAYLOAD"}, id="nan-vs-str"),
    pytest.param({float("nan"): "LEFT_PAYLOAD", float("nan"): "RIGHT_PAYLOAD"}, id="two-nans"),
    pytest.param({Color.RED: "LEFT_PAYLOAD", "red": "RIGHT_PAYLOAD"}, id="enum-vs-its-value"),
    pytest.param({Color.RED: "LEFT_PAYLOAD", Shade.RED: "RIGHT_PAYLOAD"}, id="two-enums-with-one-value"),
    pytest.param({Level.LOW: "LEFT_PAYLOAD", Digit.ONE: "RIGHT_PAYLOAD"}, id="int-enum-vs-str-enum"),
    pytest.param({Priority.LOW: "LEFT_PAYLOAD", "1": "RIGHT_PAYLOAD"}, id="IntEnum-vs-str"),
    pytest.param({Switch.ON: "LEFT_PAYLOAD", "true": "RIGHT_PAYLOAD"}, id="bool-enum-vs-str"),
    pytest.param({Ratio.HALF: "LEFT_PAYLOAD", 0.5: "RIGHT_PAYLOAD"}, id="float-enum-vs-float"),
    pytest.param({Wrapper.WRAPPED: "LEFT_PAYLOAD", "x": "RIGHT_PAYLOAD"}, id="nested-enum-vs-str"),
    pytest.param({Blank.NOTHING: "LEFT_PAYLOAD", "null": "RIGHT_PAYLOAD"}, id="null-enum-vs-str"),
    pytest.param({(1, "a"): "LEFT_PAYLOAD", '[1, "a"]': "RIGHT_PAYLOAD"}, id="tuple-vs-str"),
    pytest.param({(Color.RED,): "LEFT_PAYLOAD", ("red",): "RIGHT_PAYLOAD"}, id="tuples-with-enum-vs-value"),
    pytest.param({(Switch.ON,): "LEFT_PAYLOAD", (True,): "RIGHT_PAYLOAD"}, id="tuples-with-enum-vs-bool"),
    pytest.param({(Level.LOW,): "LEFT_PAYLOAD", (1,): "RIGHT_PAYLOAD"}, id="tuples-with-int-enum-vs-int"),
    pytest.param({((Color.RED,), 1): "LEFT_PAYLOAD", (("red",), 1): "RIGHT_PAYLOAD"}, id="nested-tuples-with-enum"),
    pytest.param({"a": 1, 2: "LEFT_PAYLOAD", "2": "RIGHT_PAYLOAD"}, id="collision-after-other-keys"),
]


@pytest.mark.parametrize("serializer", [serialize_json, serialize_json_as_obj])
@pytest.mark.parametrize("value", COLLISIONS)
def test_colliding_keys_raise_value_error_without_payload(value, serializer):
    with pytest.raises(ValueError, match="(?i)colli") as caught:
        serializer(value)

    message = str(caught.value)
    assert "LEFT_PAYLOAD" not in message
    assert "RIGHT_PAYLOAD" not in message


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"sort_keys": True}, id="sort_keys"),
        pytest.param({"indent": 2}, id="indent"),
        pytest.param({"ensure_ascii": False}, id="ensure_ascii"),
        pytest.param({"allow_nan": False}, id="allow_nan"),
    ],
)
def test_collision_is_detected_under_every_encoding_option(options):
    with pytest.raises(ValueError, match="(?i)colli"):
        serialize_json({1: "a", "1": "b"}, **options)


COLLISION_PLACES = [
    pytest.param(lambda m: [m], id="list"),
    pytest.param(lambda m: (m,), id="tuple"),
    pytest.param(lambda m: {"k": m}, id="dict-value"),
    pytest.param(lambda m: Holder(m), id="dataclass-field"),
    pytest.param(lambda m: MappingProxyType({"k": m}), id="mappingproxy-value"),
    pytest.param(lambda m: MappingProxyType(m), id="mappingproxy-itself"),
    pytest.param(lambda m: OrderedDict(k=m), id="OrderedDict-value"),
    pytest.param(lambda m: UserDict(m), id="UserDict-itself"),
    pytest.param(lambda m: {"a": [{"b": (m,)}]}, id="deep"),
    pytest.param(lambda m: {FrozenMap(m)}, id="set-member"),
    pytest.param(lambda m: frozenset({(FrozenMap(m),)}), id="frozenset-of-tuple-with-member"),
    pytest.param(lambda m: {"k": {FrozenMap(m)}}, id="set-member-in-a-dict"),
]


@pytest.mark.parametrize("place", COLLISION_PLACES)
@pytest.mark.parametrize("serializer", [serialize_json, serialize_json_as_obj])
def test_collision_is_detected_wherever_the_mapping_sits(place, serializer):
    with pytest.raises(ValueError, match="(?i)colli") as caught:
        serializer(place({1: "LEFT_PAYLOAD", "1": "RIGHT_PAYLOAD"}))

    assert "PAYLOAD" not in str(caught.value)


# --- mapping keys: deserialization -------------------------------------------------------------------------------


KEY_RESTORES = [
    pytest.param({"a": 1}, dict[str, int], id="str"),
    pytest.param(
        {"": 1, " ": 2, "é": 3, "שלום": 4, '"': 5, "\\": 6, "\n": 7, "1": 8, "true": 9, "[1, 2]": 10},
        dict[str, int],
        id="awkward-str-keys",
    ),
    pytest.param({Color.RED: 1, Color.BLUE: 2}, dict[Color, int], id="enum"),
    pytest.param({Label.A: 1}, dict[Label, int], id="str-mixin-enum"),
    pytest.param(
        {Color.RED: {Color.GREEN: [Point(1, 2)]}}, dict[Color, dict[Color, list[Point]]], id="nested-enum-keys"
    ),
    pytest.param({Color.RED: 1}, Mapping[Color, int], id="Mapping-enum"),
    pytest.param(
        MappingProxyType({Color.RED: (1, 2)}), MappingProxyType[Color, tuple[int, ...]], id="mappingproxy-enum"
    ),
    pytest.param([{Color.RED: 1}, {}], list[dict[Color, int]], id="list-of-enum-dicts"),
    pytest.param({Color.RED: 1}, Optional[dict[Color, int]], id="optional-enum-dict"),
    pytest.param({(1, "a"): 1, (): 2}, dict[tuple, int], id="bare-tuple-key"),
]


@pytest.mark.parametrize(("value", "annotation"), KEY_RESTORES)
def test_keys_are_restored_to_their_original_type(value, annotation):
    assert_round_trip(value, annotation)


TUPLE_KEY_RESTORES = [
    pytest.param({(1, "a"): "x", (2, "b"): "y"}, dict[tuple[int, str], str], id="flat"),
    pytest.param({((1, 2), "a"): 1, ((3, 4), "b"): 2}, dict[tuple[tuple[int, int], str], int], id="nested"),
    pytest.param({(1, 2, 3): 1, (): 2, (9,): 3}, dict[tuple[int, ...], int], id="variadic"),
    pytest.param({(Color.RED, 1): 1}, dict[tuple[Color, int], int], id="enum-member"),
    pytest.param(
        {(None, True, 1.5): 1, (3, False, 0.0): 2}, dict[tuple[Optional[int], bool, float], int], id="none-bool-float"
    ),
    pytest.param({(Point(1, 2), "a"): 1}, dict[tuple[Point, str], int], id="dataclass-member"),
    pytest.param(
        {(date(2024, 5, 17), Decimal("1.50")): 1}, dict[tuple[date, Decimal], int], id="date-and-decimal-members"
    ),
    pytest.param(
        {(b"ab", datetime(2024, 5, 17, 8, 30)): 1}, dict[tuple[bytes, datetime], int], id="bytes-and-datetime"
    ),
    pytest.param({(frozenset({10, 9}), 1): 1}, dict[tuple[frozenset[int], int], int], id="frozenset-member"),
    pytest.param({((Color.RED,), 1): 1}, dict[tuple[tuple[Color], int], int], id="nested-enum-tuple"),
    pytest.param(MappingProxyType({(1, 2): "a"}), MappingProxyType[tuple[int, int], str], id="mappingproxy"),
    pytest.param({(1, 2): "a"}, Mapping[tuple[int, int], str], id="Mapping"),
    pytest.param({"outer": {(1, 2): 3}}, dict[str, dict[tuple[int, int], int]], id="dict-of-dicts"),
    pytest.param([{(1, 2): "a"}, {}], list[dict[tuple[int, int], str]], id="list-of-dicts"),
    pytest.param(Board({(0, 0): "a", (0, 1): "b"}), Board, id="dataclass-field"),
]


@pytest.mark.parametrize(("value", "annotation"), TUPLE_KEY_RESTORES)
def test_tuple_keys_are_restored_to_tuples(value, annotation):
    assert_round_trip(value, annotation)


@pytest.mark.parametrize("options", OPTION_SETS)
def test_tuple_keys_survive_every_option_set(options):
    value = {("é", 1): [1], ("a", 2): [2, 3]}

    assert_round_trip(value, dict[tuple[str, int], list[int]], **options)


def test_enum_keys_with_non_string_values_serialize_as_their_value_text():
    assert to_json_value({Level.LOW: "x", Level.HIGH: "y"}) == {"1": "x", "2": "y"}
    assert to_json_value({Priority.LOW: "x"}) == {"1": "x"}
    assert to_json_value({Switch.ON: 1, Ratio.HALF: 2, Cell.ORIGIN: 3}) == {"true": 1, "0.5": 2, "[0, 0]": 3}


@pytest.mark.parametrize(
    ("member", "enum_type"),
    [
        pytest.param(Level.LOW, Level, id="int-valued"),
        pytest.param(Priority.LOW, Priority, id="IntEnum"),
        pytest.param(Switch.ON, Switch, id="bool-valued"),
        pytest.param(Ratio.HALF, Ratio, id="float-valued"),
        pytest.param(Blank.NOTHING, Blank, id="null-valued"),
        pytest.param(Cell.ORIGIN, Cell, id="tuple-valued"),
    ],
)
def test_enum_keys_with_non_string_values_are_restored_as_members(member, enum_type):
    # a key is the text of the member value ("1", "true", "0.5", "null", "[0, 0]"): the member with that key text is
    # restored, not the member whose value equals the text
    restored = assert_round_trip({member: 1}, dict[enum_type, int])

    assert next(iter(restored)) is member


def test_enum_key_that_is_no_members_text_is_a_type_error_naming_the_enum_only():
    with pytest.raises(TypeError, match="Invalid value for .*, got str") as caught:
        deserialize_json({"KEY_SENTINEL": 1}, dict[Level, int])

    assert "KEY_SENTINEL" not in str(caught.value)
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert deserialize_json({"KEY_SENTINEL": 1}, Union[dict[Level, int], dict[str, int]]) == {"KEY_SENTINEL": 1}


def test_enum_keys_with_str_values_are_restored_from_their_text():
    assert_round_trip({Color.RED: 1}, dict[Color, int])
    assert_round_trip({Digit.ONE: 1}, dict[Digit, int])


def test_unknown_enum_key_raises_type_error_without_the_key():
    with pytest.raises(TypeError, match="Invalid value for") as caught:
        deserialize_json({"KEY_SENTINEL_NOT_A_COLOR": 1}, dict[Color, int])

    assert "KEY_SENTINEL_NOT_A_COLOR" not in str(caught.value)


@pytest.mark.parametrize(
    ("data", "annotation"),
    [
        pytest.param({"1": "a"}, dict[int, str], id="int"),
        pytest.param({"1.5": "a"}, dict[float, str], id="float"),
        pytest.param({"true": "a"}, dict[bool, str], id="bool"),
        pytest.param({"1": "a"}, typing.Dict[int, str], id="typing-Dict"),
        pytest.param({"1": "a"}, Mapping[int, str], id="Mapping"),
        pytest.param({"1": "a"}, MappingProxyType[int, str], id="mappingproxy"),
        pytest.param({"k": {"1": "a"}}, dict[str, dict[int, str]], id="nested"),
        pytest.param([{"1": "a"}], list[dict[int, str]], id="in-a-list"),
        pytest.param({"1": "a"}, Optional[dict[int, str]], id="optional"),
    ],
)
def test_non_string_scalar_keys_cannot_be_restored_limitation(data, annotation):
    # Documents current behavior, not a promise: a JSON object key is always a string and nothing says how to turn
    # it back into an int, float or bool, so jserpy refuses (with ValueError, which a Union does not catch) instead
    # of guessing. Annotate such keys as `str`, or use an Enum or a tuple key, to get them back.
    with pytest.raises(ValueError, match="(?i)convert"):
        deserialize_json(data, annotation)


def test_bare_tuple_key_with_a_nested_tuple_cannot_be_restored_limitation():
    # Documents current behavior, not a promise: with no member types, the inner tuple of the key comes back as a list
    # (see the bare `tuple` limitation above), which cannot be a dict key. `dict[tuple[tuple[int, int], str], V]`
    # restores the same key (see TUPLE_KEY_RESTORES).
    assert serialize_json({((1, 2), "a"): 1}) == '{"[[1, 2], \\"a\\"]": 1}'
    with pytest.raises(TypeError, match="unhashable"):
        deserialize_json({'[[1, 2], "a"]': 1}, dict[tuple, int])


@pytest.mark.parametrize(
    ("key", "error", "message"),
    [
        pytest.param("KEY_SENTINEL", TypeError, "not JSON", id="not-json"),
        pytest.param("5", TypeError, "Expected a JSON array", id="json-number"),
        pytest.param('{"KEY_SENTINEL": 1}', TypeError, "Expected a JSON array", id="json-object"),
    ],
)
def test_tuple_key_that_is_not_a_json_array_is_refused_without_the_key(key, error, message):
    # only a key that is a JSON array can be a tuple; one that is not JSON at all is a TypeError too
    with pytest.raises(error, match=message) as caught:
        deserialize_json({key: 1}, dict[tuple, int])

    assert "KEY_SENTINEL" not in str(caught.value)


def test_scalar_keys_come_back_as_strings_when_annotated_as_strings():
    value = {1: "a", 2.5: "b", False: "c"}

    restored = deserialize_json(to_json_value(value), dict[str, str])

    assert_strictly_equal(restored, {"1": "a", "2.5": "b", "false": "c"})


def test_keys_of_different_kinds_are_converted_at_every_depth():
    value = {1: {2.5: {True: {(1, 2): {Color.RED: "x", Color.BLUE: "y"}}}}, -1: {}}

    assert serialize_json(value) == (
        '{"1": {"2.5": {"true": {"[1, 2]": {"red": "x", "blue": "y"}}}}, "-1": {}}'
    )
    # the int, float, bool and tuple levels are annotated as str: they cannot be restored to their own types
    restored = deserialize_json(
        to_json_value(value), dict[str, dict[str, dict[str, dict[str, dict[Color, str]]]]]
    )
    assert_strictly_equal(
        restored,
        {"1": {"2.5": {"true": {"[1, 2]": {Color.RED: "x", Color.BLUE: "y"}}}}, "-1": {}},
    )


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(dict[Union[int, str], int], id="union"),
        pytest.param(dict[Optional[str], int], id="optional"),
        pytest.param(dict[Literal["a", "b"], int], id="literal"),
    ],
)
def test_key_annotations_that_are_not_classes_are_unsupported_limitation(annotation):
    # Documents current behavior, not a promise: key annotations are inspected with issubclass(), which rejects a
    # Union or a Literal with an internal TypeError ("issubclass() arg 1 must be a class").
    with pytest.raises(TypeError):
        deserialize_json({"a": 1}, annotation)


@pytest.mark.parametrize(
    ("data", "annotation", "expected"),
    [
        pytest.param([None, None], list[None], [None, None], id="list"),
        pytest.param([1, None], tuple[int, None], (1, None), id="tuple"),
        pytest.param({"a": None}, dict[str, None], {"a": None}, id="dict"),
        pytest.param([None], set[None], {None}, id="set"),
    ],
)
def test_none_as_a_type_argument_of_a_builtin_generic(data, annotation, expected):
    assert_strictly_equal(deserialize_json(data, annotation), expected)


@pytest.mark.parametrize(
    ("data", "annotation", "expected"),
    [
        pytest.param([1, None], typing.Tuple[int, None], (1, None), id="typing-Tuple"),
        pytest.param([1, None], tuple[int, type(None)], (1, None), id="NoneType"),
        pytest.param([1, None], list[Optional[int]], [1, None], id="Optional"),
        pytest.param({"a": None}, dict[str, Union[None, int]], {"a": None}, id="Union-with-None"),
    ],
)
def test_none_spelled_in_a_supported_way_restores(data, annotation, expected):
    assert_strictly_equal(deserialize_json(data, annotation), expected)


# --- unions of containers ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "annotation", "expected"),
    [
        pytest.param([1, 2], list[int] | dict[str, int] | None, [1, 2], id="list-member"),
        pytest.param({"a": 1}, list[int] | dict[str, int] | None, {"a": 1}, id="dict-member"),
        pytest.param(None, list[int] | dict[str, int] | None, None, id="none-member"),
        pytest.param([1, 1, 2], set[int] | dict[str, int], {1, 2}, id="set-member"),
        pytest.param([1, 2], tuple[int, int] | list[int], (1, 2), id="first-matching-member-wins"),
        pytest.param([1, 2, 3], tuple[int, int] | list[int], [1, 2, 3], id="falls-through-on-length"),
        pytest.param(
            [1, 2, 3], tuple[int, int] | tuple[int, int, int] | list[int], (1, 2, 3), id="second-of-three-members"
        ),
        pytest.param(
            [1, 2, 3, 4],
            tuple[int, int] | tuple[int, int, int] | list[int],
            [1, 2, 3, 4],
            id="third-of-three-members",
        ),
        pytest.param(
            [[1, 2], [1, 2, 3], [1, 2, 3, 4], []],
            list[tuple[int, int] | tuple[int, int, int] | list[int]],
            [(1, 2), (1, 2, 3), [1, 2, 3, 4], []],
            id="a-different-member-per-item",
        ),
        pytest.param({"red": 1}, Optional[dict[Color, int]], {Color.RED: 1}, id="enum-keyed-member"),
        pytest.param([[1], []], Union[list[list[int]], list[str]], [[1], []], id="nested-list-member"),
    ],
)
def test_union_of_containers_selects_the_member_matching_the_data(data, annotation, expected):
    assert_strictly_equal(deserialize_json(data, annotation), expected)


def test_union_of_containers_rejects_data_that_matches_no_member():
    with pytest.raises(TypeError):
        deserialize_json({"red": 1}, Union[list[int], tuple[int, int]])
    with pytest.raises(TypeError):
        deserialize_json({"purple": 1}, dict[Color, int] | list[int])


# --- depth and width ---------------------------------------------------------------------------------------------


def _nest(depth, kinds):
    """A value, its annotation and its JSON text, wrapped `depth` times with `kinds` in turn (the text is built by
    hand, so it is independent of jserpy)."""
    value, annotation, text = 7, int, "7"
    for level in range(depth):
        kind = kinds[level % len(kinds)]
        if kind == "list":
            value, annotation, text = [value], list[annotation], f"[{text}]"
        elif kind == "tuple":
            value, annotation, text = (value,), tuple[annotation, ...], f"[{text}]"
        elif kind == "frozenset":
            value, annotation, text = frozenset({value}), frozenset[annotation], f"[{text}]"
        elif kind == "dict":
            value, annotation, text = {"k": value}, dict[str, annotation], f'{{"k": {text}}}'
        else:
            proxy_type = MappingProxyType[str, annotation]
            value, annotation, text = MappingProxyType({"k": value}), proxy_type, f'{{"k": {text}}}'
    return value, annotation, text


NEST_KINDS = [
    pytest.param(("list",), id="lists"),
    pytest.param(("tuple",), id="tuples"),
    pytest.param(("dict",), id="dicts"),
    pytest.param(("frozenset",), id="frozensets"),
    pytest.param(("list", "dict"), id="lists-and-dicts"),
    pytest.param(("dict", "tuple", "list"), id="dicts-tuples-lists"),
    pytest.param(("tuple", "frozenset"), id="tuples-and-frozensets"),
    pytest.param(("proxy", "list"), id="proxies-and-lists"),
]


@pytest.mark.parametrize("kinds", NEST_KINDS)
@pytest.mark.parametrize("depth", [1, 2, 3, 8, 30])
def test_nesting_depth_text_and_round_trip(kinds, depth):
    value, annotation, text = _nest(depth, kinds)

    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


def test_wide_set_and_nested_wide_containers():
    members = set(range(1000))
    wide = {
        "rows": [list(range(50)) for _ in range(40)],
        "members": members,
        "pairs": [(i, str(i)) for i in range(300)],
    }
    expected = {
        "rows": [list(range(50)) for _ in range(40)],
        "members": sorted(members, key=str),
        "pairs": [[i, str(i)] for i in range(300)],
    }

    assert serialize_json(wide) == json.dumps(expected)
    assert_round_trip(wide["members"], set[int])
    assert_round_trip(wide["pairs"], list[tuple[int, str]])


# --- combinatorial families with an independent oracle -----------------------------------------------------------


class Case(NamedTuple):
    value: Any
    annotation: Any
    expected: Any  # the JSON value that a reader of the text must see, built by hand
    hashable: bool


def _leaf_pairs():
    return {
        "int": (Case(1, int, 1, True), Case(2, int, 2, True)),
        "str": (Case("a", str, "a", True), Case("", str, "", True)),
        "list": (Case([1, 2], list[int], [1, 2], False), Case([], list[int], [], False)),
        "tuple": (Case((1, 2), tuple[int, ...], [1, 2], True), Case((3,), tuple[int, ...], [3], True)),
        "set": (Case({9, 10}, set[int], [10, 9], False), Case({2, 11}, set[int], [11, 2], False)),
        "frozenset": (
            Case(frozenset({9, 10}), frozenset[int], [10, 9], True),
            Case(frozenset(), frozenset[int], [], True),
        ),
        "dict": (
            Case({"b": 1, "a": 2}, dict[str, int], {"b": 1, "a": 2}, False),
            Case({}, dict[str, int], {}, False),
        ),
    }


@functools.lru_cache(maxsize=None)
def _box_class(first_annotation, second_annotation):
    """One dataclass per pair of annotations, so equal wrappers share a class."""
    return make_dataclass("Box", [("first", first_annotation), ("second", second_annotation)])


def _wrap(kind, first, second):
    """`first` and `second` (two cases of one annotation) as members of a `kind` container, or None if impossible."""
    values = (first.value, second.value)
    expected = [first.expected, second.expected]
    hashable = first.hashable and second.hashable
    named = {"first": first.expected, "second": second.expected}
    if kind == "list":
        return Case(list(values), list[first.annotation], expected, False)
    if kind == "tuple":
        return Case(values, tuple[first.annotation, ...], expected, hashable)
    if kind == "pair":
        return Case(values, tuple[first.annotation, second.annotation], expected, hashable)
    if kind in ("set", "frozenset"):
        if not hashable:
            return None
        container = set if kind == "set" else frozenset
        return Case(container(values), container[first.annotation], _ordered_members(*expected), kind == "frozenset")
    if kind == "dict":
        return Case(dict(zip(("first", "second"), values)), dict[str, first.annotation], named, False)
    if kind == "mappingproxy":
        proxy = MappingProxyType(dict(zip(("first", "second"), values)))
        return Case(proxy, MappingProxyType[str, first.annotation], named, False)
    box = _box_class(first.annotation, second.annotation)
    return Case(box(*values), box, named, False)


WRAP_KINDS = ["list", "tuple", "pair", "set", "frozenset", "dict", "mappingproxy", "dataclass"]


def _pair_family():
    params = []
    for inner, (first, second) in _leaf_pairs().items():
        for outer in WRAP_KINDS:
            case = _wrap(outer, first, second)
            if case is not None:
                params.append(pytest.param(case, id=f"{outer}-of-{inner}"))
    return params


def _triple_family():
    params = []
    leaves = _leaf_pairs()
    for inner in ("int", "list", "frozenset", "dict"):
        first, second = leaves[inner]
        for middle in WRAP_KINDS:
            near, far = _wrap(middle, first, second), _wrap(middle, first, first)
            if near is None or far is None:
                continue
            for outer in WRAP_KINDS:
                case = _wrap(outer, near, far)
                if case is not None:
                    params.append(pytest.param(case, id=f"{outer}-of-{middle}-of-{inner}"))
    return params


PAIR_FAMILY = _pair_family()
TRIPLE_FAMILY = _triple_family()


def test_the_families_cover_the_container_pairs():
    assert len(PAIR_FAMILY) >= 45
    assert len(TRIPLE_FAMILY) >= 150


@pytest.mark.parametrize("case", PAIR_FAMILY + TRIPLE_FAMILY)
def test_family_serialization_matches_the_hand_built_json(case):
    assert to_json_value(case.value) == case.expected
    assert serialize_json(case.value) == json.dumps(case.expected)


@pytest.mark.parametrize("case", PAIR_FAMILY + TRIPLE_FAMILY)
def test_family_round_trip(case):
    assert_round_trip(case.value, case.annotation)


@pytest.mark.parametrize("case", PAIR_FAMILY)
@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"indent": 2}, id="indent"),
        pytest.param({"separators": (",", ":"), "sort_keys": True}, id="compact-sorted"),
    ],
)
def test_family_text_honors_options_like_the_stdlib(case, options):
    # sets are written in canonical order, which the hand-built `expected` already has
    assert serialize_json(case.value, **options) == json.dumps(case.expected, **options)


# --- JSON-native values: the oracle is the stdlib ----------------------------------------------------------------


NATIVE_VALUES = [
    pytest.param(None, Any, id="null"),
    pytest.param(True, bool, id="true"),
    pytest.param(12, int, id="int"),
    pytest.param(-0.0, float, id="negative-zero"),
    pytest.param("", str, id="empty-str"),
    pytest.param([], list[Any], id="empty-list"),
    pytest.param({}, dict[str, Any], id="empty-dict"),
    pytest.param([None, True, 1, 1.0, "1", [1], {"1": 1}], list[Any], id="every-json-type-in-a-list"),
    pytest.param(
        {"n": None, "b": False, "i": 0, "f": 0.5, "s": "x", "l": [], "d": {}},
        dict[str, Any],
        id="every-json-type-in-a-dict",
    ),
    pytest.param([[[[[[]]]]]], Any, id="six-levels-of-lists"),
    pytest.param({"a": {"b": {"c": {"d": {"e": {}}}}}}, Any, id="six-levels-of-dicts"),
    pytest.param([{"a": [{"b": [{"c": [None]}]}]}], Any, id="alternating"),
    pytest.param([1, 2, 3], list[int], id="natural-list-of-ints"),
    pytest.param({"a": [1.5, 2.5], "b": []}, dict[str, list[float]], id="natural-dict-of-lists-of-floats"),
    pytest.param([[1, 2], [3]], list[list[int]], id="natural-matrix"),
    pytest.param([[1], {"a": [2, {"b": []}]}], list, id="natural-bare-list"),
    pytest.param({"a": [1, {"b": []}], "c": {"d": {}}}, dict, id="natural-bare-dict"),
    pytest.param([{"id": 1, "tags": ["a"]}, {"id": 2, "tags": []}], list[dict[str, Any]], id="natural-records"),
    pytest.param([10**30, -(10**30), 5e-324, 1.7976931348623157e308], list[Union[int, float]], id="extreme-numbers"),
    pytest.param({"é": "ש", "\U0001f600": "\x00 "}, dict[str, str], id="unicode-and-control-characters"),
]


def _container_ids(value):
    found = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            found.add(id(current))
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            if isinstance(current, list):
                found.add(id(current))
            pending.extend(current)
    return found


def _mutate_everywhere(value):
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            pending.extend(list(current.values()))
            current["MUTATED"] = "MUTATED"
        elif isinstance(current, list):
            pending.extend(list(current))
            current.append("MUTATED")


@pytest.mark.parametrize(("value", "annotation"), NATIVE_VALUES)
def test_native_value_is_its_own_json(value, annotation):
    assert_strictly_equal(to_json_value(value), value)
    assert serialize_json(value) == json.dumps(value)


@pytest.mark.parametrize("options", OPTION_SETS)
@pytest.mark.parametrize(("value", "annotation"), NATIVE_VALUES)
def test_native_value_text_matches_the_stdlib_for_every_option_set(value, annotation, options):
    assert serialize_json(value, **options) == json.dumps(value, **options)


@pytest.mark.parametrize(("value", "annotation"), NATIVE_VALUES)
def test_native_value_round_trips_to_an_equal_value_that_shares_nothing(value, annotation):
    assert_round_trip(value, annotation)

    parsed = json.loads(json.dumps(value))
    snapshot = copy.deepcopy(parsed)
    restored = deserialize_json(parsed, annotation)

    assert_strictly_equal(restored, value)
    assert _container_ids(restored).isdisjoint(_container_ids(parsed))

    # mutating the result leaves the input alone, and mutating the input leaves the result alone
    _mutate_everywhere(restored)
    assert json.dumps(parsed) == json.dumps(snapshot)
    untouched = deserialize_json(parsed, annotation)
    _mutate_everywhere(parsed)
    assert_strictly_equal(untouched, value)


@pytest.mark.parametrize("annotation", [tuple, tuple[Any, ...], tuple[list[int], dict[str, Any]]], ids=repr)
def test_tuple_members_that_are_json_containers_do_not_alias_the_input(annotation):
    parsed = [[1, 2], {"a": [3, {"b": []}]}]
    snapshot = copy.deepcopy(parsed)

    restored = deserialize_json(parsed, annotation)

    assert type(restored) is tuple
    assert list(restored) == parsed
    assert _container_ids(restored).isdisjoint(_container_ids(parsed))
    _mutate_everywhere(restored)
    assert parsed == snapshot


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(Mapping, id="Mapping"),
        pytest.param(typing.Mapping, id="typing-Mapping"),
        pytest.param(Mapping[str, Any], id="Mapping-of-Any"),
        pytest.param(MappingProxyType, id="MappingProxyType"),
        pytest.param(MappingProxyType[str, Any], id="MappingProxyType-of-Any"),
    ],
)
def test_mapping_targets_do_not_alias_the_input(annotation):
    parsed = {"a": [1, {"b": []}], "c": {"d": {"e": 1}}}
    snapshot = copy.deepcopy(parsed)

    restored = deserialize_json(parsed, annotation)

    # a change anywhere in the input, top level included, must not show in the result
    _mutate_everywhere(parsed)
    assert json.dumps(dict(restored)) == json.dumps(snapshot)


def test_native_non_finite_floats_survive_when_allowed():
    value = [float("nan"), float("inf"), float("-inf"), {"x": float("nan")}]

    assert_round_trip(value, list[Any])
    with pytest.raises(ValueError):
        serialize_json(value, allow_nan=False)


class _Lcg:
    """A tiny deterministic generator: the algorithms of the random module may differ between Python versions."""

    def __init__(self, seed):
        self.state = seed * 2654435761 + 12345

    def below(self, bound):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) % 2**64
        return (self.state >> 33) % bound


_POOLS = {
    "int": [0, 1, -1, 42, 2**63, -(2**70)],
    "float": [0.0, -0.0, 1.5, -2.25, 1e100, 1e-07, 0.1, 5e-324, 1.7976931348623157e308],
    "str": ["", " ", "a", "é", "שלום", "\U0001f600", '"', "\\", "line\nbreak", "\x00", " "],
    "bool": [True, False],
}


def _spec(rng, depth):
    """A random type description: the top levels are always containers, the bottom ones always scalars."""
    scalars = ["int", "float", "str", "bool", "any"]
    containers = ["list", "list", "list", "dict", "dict", "dict", "opt"]
    if depth == 0:
        kinds = scalars
    elif depth >= 3:
        kinds = containers
    else:
        kinds = scalars + containers
    kind = kinds[rng.below(len(kinds))]
    if kind in ("list", "dict"):
        return (kind, _spec(rng, depth - 1))
    if kind == "opt":
        inner = _spec(rng, depth - 1)
        return inner if inner[0] == "opt" else ("opt", inner)
    return (kind,)


def _annotation(spec):
    kind = spec[0]
    if kind == "list":
        return list[_annotation(spec[1])]
    if kind == "dict":
        return dict[str, _annotation(spec[1])]
    if kind == "opt":
        return Optional[_annotation(spec[1])]
    return {"int": int, "float": float, "str": str, "bool": bool, "any": Any}[kind]


def _scalar(rng, kind):
    pool = _POOLS[kind]
    return pool[rng.below(len(pool))]


def _any_value(rng, depth):
    choice = rng.below(7 if depth > 0 else 5)
    if choice == 0:
        return None
    if choice < 5:
        return _scalar(rng, ("int", "float", "str", "bool")[choice - 1])
    if choice == 5:
        return [_any_value(rng, depth - 1) for _ in range(rng.below(4))]
    return {_scalar(rng, "str"): _any_value(rng, depth - 1) for _ in range(rng.below(4))}


def _value(spec, rng):
    kind = spec[0]
    if kind in _POOLS:
        return _scalar(rng, kind)
    if kind == "any":
        return _any_value(rng, 3)
    if kind == "opt":
        return None if rng.below(3) == 0 else _value(spec[1], rng)
    if kind == "list":
        return [_value(spec[1], rng) for _ in range(rng.below(4))]
    return {_scalar(rng, "str"): _value(spec[1], rng) for _ in range(rng.below(4))}


def _generated(seed):
    rng = _Lcg(seed)
    spec = ("list", _spec(rng, 4))
    return _value(spec, rng), _annotation(spec)


GENERATED = [pytest.param(*_generated(seed), id=f"seed-{seed}") for seed in range(60)]


@pytest.mark.parametrize(("value", "annotation"), GENERATED)
def test_generated_native_structure_matches_the_stdlib_and_round_trips(value, annotation):
    assert serialize_json(value) == json.dumps(value)
    assert serialize_json(value, sort_keys=True, indent=1) == json.dumps(value, sort_keys=True, indent=1)
    assert serialize_json(value, ensure_ascii=False) == json.dumps(value, ensure_ascii=False)
    assert_strictly_equal(to_json_value(value), value)
    assert json.dumps(to_json_value(value)) == json.dumps(value)

    assert_round_trip(value, annotation)
    parsed = json.loads(json.dumps(value))
    restored = deserialize_json(parsed, annotation)
    assert _container_ids(restored).isdisjoint(_container_ids(parsed))
    assert json.dumps(restored) == json.dumps(value)


def test_generated_structures_are_not_all_trivial():
    # guards the generator itself: a degenerate one would make the tests above vacuous
    sizes = [len(_container_ids(param.values[0])) for param in GENERATED]
    annotations = {repr(param.values[1]) for param in GENERATED}

    assert max(sizes) >= 20
    assert sum(size > 3 for size in sizes) >= 30
    assert len(annotations) >= 30

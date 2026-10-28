"""NumPy values: arrays of many dtypes and shapes, scalars, and both nested inside other objects.

An array serializes as its ``tolist()`` and a scalar as its ``item()``, so the JSON goldens below are written by hand
from ``json.dumps`` rules. Deserialization is driven by the target type and JSON carries no dtype, so an array comes
back with the dtype NumPy infers from the data.

Tests named ``*_limitation`` document current behavior of that kind; they are not promises. Tests marked
``xfail(strict=True)`` assert what a user would expect where the library does something else.
"""

import copy
import dataclasses
import datetime
import decimal
import json
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Optional, Union

import pytest

np = pytest.importorskip("numpy")  # the whole module is skipped without NumPy

from jserpy import deserialize_json, serialize_json, serialize_json_as_dict, serialize_json_as_obj
from jserpy.jsonable import Jsonable
from objects_support import assert_round_trip, assert_strictly_equal, round_trip, to_json_value


def _object_array(*items):
    """A 1-d object array holding exactly ``items`` (``np.array`` would try to build a 2-d array from lists)."""
    array = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        array[index] = item
    return array


def _assert_plain_json(value):
    """No NumPy (or other non-JSON) type is left anywhere in an intermediate JSON tree."""
    pending = [value]
    while pending:
        item = pending.pop()
        assert type(item) in (dict, list, str, int, float, bool, type(None)), type(item)
        if type(item) is dict:
            assert all(type(key) is str for key in item)
            pending.extend(item.values())
        elif type(item) is list:
            pending.extend(item)


def _containers(value):
    """Every dict and list reachable from ``value``, looking into dataclasses and object arrays too."""
    found, seen, pending = [], set(), [value]
    while pending:
        item = pending.pop()
        if isinstance(item, np.ndarray):
            if item.dtype == object:
                pending.extend(item.flat)
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, one.name) for one in dataclasses.fields(item))
        elif isinstance(item, (dict, list, tuple)):
            if id(item) in seen:
                continue
            seen.add(id(item))
            if not isinstance(item, tuple):
                found.append(item)
            pending.extend(item.values() if isinstance(item, dict) else item)
    return found


def _mutate(containers):
    for container in containers:
        if isinstance(container, dict):
            container["mutated"] = True
        else:
            container.append("mutated")


def _state(array):
    """Everything that serializing must leave alone: data, layout and flags."""
    content = json.dumps(array.tolist()) if array.dtype == object else array.tobytes()
    flags = array.flags
    return (
        array.dtype, array.shape, array.strides, flags.writeable, flags.c_contiguous, flags.f_contiguous,
        flags.owndata, content,
    )


# NumPy 2.5 warns when np.datetime64(None) builds the generic-unit NaT that the library returns for null
_GENERIC_NAT_WARNING = pytest.mark.filterwarnings("ignore:The 'generic' unit:DeprecationWarning")
# NumPy 1.x wraps an out-of-range Python int silently (with a deprecation warning at best); 2.x raises OverflowError
_NUMPY_RAISES_ON_OVERFLOW = pytest.mark.skipif(
    int(np.__version__.split(".")[0]) < 2, reason="NumPy 1.x does not raise for out-of-range Python ints"
)

INTEGER_CLASSES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
FLOAT_CLASSES = [np.float16, np.float32, np.float64]
# C-style names: distinct classes (np.longlong is not np.int64 on every platform) that must restore as themselves
C_INTEGER_CLASSES = [
    pytest.param(getattr(np, name), id=name)
    for name in (
        "byte", "ubyte", "short", "ushort", "intc", "uintc", "int_", "uint", "longlong", "ulonglong", "intp", "uintp"
    )
]


class Tagged(np.ndarray):
    """A user subclass of ndarray (created with ``array.view(Tagged)``)."""


class Celsius(np.float64):
    """A user subclass of a NumPy scalar class."""


class Channel(Enum):
    RED = "r"
    GREEN = "g"


def _class_id(cls):
    return cls.__name__


# --- JSON shape of arrays -----------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("make", "text"),
    [
        pytest.param(lambda: np.array(5), "5", id="0d-int"),
        pytest.param(lambda: np.array(1.5), "1.5", id="0d-float"),
        pytest.param(lambda: np.array(True), "true", id="0d-bool"),
        pytest.param(lambda: np.array("hi"), '"hi"', id="0d-str"),
        pytest.param(lambda: np.array([7]), "[7]", id="single-element"),
        pytest.param(lambda: np.array([[7]]), "[[7]]", id="single-element-2d"),
        pytest.param(lambda: np.arange(4), "[0, 1, 2, 3]", id="1d"),
        pytest.param(lambda: np.arange(6).reshape(2, 3), "[[0, 1, 2], [3, 4, 5]]", id="2d"),
        pytest.param(
            lambda: np.arange(24).reshape(2, 3, 4),
            "[[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]",
            id="3d-2x3x4",
        ),
        pytest.param(lambda: np.array([[[1]], [[2]]]), "[[[1]], [[2]]]", id="3d-2x1x1"),
        pytest.param(lambda: np.zeros((1,) * 30), "[" * 30 + "0.0" + "]" * 30, id="30d-ones-shape"),
        pytest.param(lambda: np.array([]), "[]", id="empty"),
        pytest.param(lambda: np.array([], dtype=np.int64), "[]", id="empty-int"),
        pytest.param(lambda: np.zeros((2, 0)), "[[], []]", id="empty-rows-2x0"),
        pytest.param(lambda: np.zeros((3, 0)), "[[], [], []]", id="empty-rows-3x0"),
        pytest.param(lambda: np.zeros((2, 0, 3)), "[[], []]", id="empty-2x0x3"),
        pytest.param(lambda: np.zeros((0, 3)), "[]", id="empty-0x3"),
        pytest.param(lambda: np.zeros((0, 2, 3)), "[]", id="empty-0x2x3"),
        pytest.param(lambda: np.array([True, False, True]), "[true, false, true]", id="bool"),
        pytest.param(lambda: np.array([[True], [False]]), "[[true], [false]]", id="bool-2d"),
        pytest.param(lambda: np.array(["a", "bé"]), '["a", "b\\u00e9"]', id="str-escaped-by-default"),
        pytest.param(lambda: np.array([b"ab", b"c"]), '["YWI=", "Yw=="]', id="bytes-as-base64"),
        pytest.param(lambda: np.array([0.5, -2.25, 1e-07]), "[0.5, -2.25, 1e-07]", id="float64"),
        pytest.param(lambda: np.array([1.0, 2.0]), "[1.0, 2.0]", id="whole-floats-keep-the-point"),
        pytest.param(lambda: np.array([-0.0, 0.0]), "[-0.0, 0.0]", id="negative-zero"),
        pytest.param(
            lambda: np.array([np.nan, np.inf, -np.inf]), "[NaN, Infinity, -Infinity]", id="non-finite-floats"
        ),
        pytest.param(
            lambda: np.array([np.inf, -np.inf, np.nan], dtype=np.float16),
            "[Infinity, -Infinity, NaN]",
            id="non-finite-float16",
        ),
        pytest.param(
            lambda: np.array([2 ** 63 - 1, -2 ** 63]), "[9223372036854775807, -9223372036854775808]", id="int64-limits"
        ),
        pytest.param(lambda: np.array([2 ** 64 - 1], dtype=np.uint64), "[18446744073709551615]", id="uint64-max"),
        pytest.param(lambda: np.eye(3), "[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]", id="eye"),
        pytest.param(lambda: np.linspace(0, 1, 5), "[0.0, 0.25, 0.5, 0.75, 1.0]", id="linspace"),
        pytest.param(
            lambda: np.asfortranarray(np.arange(6).reshape(2, 3)), "[[0, 1, 2], [3, 4, 5]]", id="fortran-ordered"
        ),
        pytest.param(lambda: np.arange(6).reshape(2, 3).T, "[[0, 3], [1, 4], [2, 5]]", id="transposed-view"),
        pytest.param(
            lambda: np.arange(12).reshape(3, 4)[::2, 1::2], "[[1, 3], [9, 11]]", id="strided-slice-view"
        ),
        pytest.param(lambda: np.arange(5)[::-1], "[4, 3, 2, 1, 0]", id="negative-stride-view"),
        pytest.param(
            lambda: np.broadcast_to(np.arange(3), (2, 3)), "[[0, 1, 2], [0, 1, 2]]", id="read-only-broadcast-view"
        ),
        pytest.param(lambda: np.array([1, 2.5], dtype=object), "[1, 2.5]", id="object-numbers"),
        pytest.param(lambda: np.array([None, None], dtype=object), "[null, null]", id="object-none"),
        pytest.param(
            lambda: np.array([[1], [2, 3]], dtype=object), "[[1], [2, 3]]", id="object-ragged-lists"
        ),
        pytest.param(lambda: np.array({"a": [1]}, dtype=object), '{"a": [1]}', id="0d-object-dict"),
        pytest.param(lambda: np.array(["a", None], dtype=object), '["a", null]', id="object-str-and-none"),
        pytest.param(
            lambda: np.array(['a\nb', 'q"uote', "tab\t", "back\\slash"]),
            '["a\\nb", "q\\"uote", "tab\\t", "back\\\\slash"]',
            id="str-json-escapes",
        ),
        pytest.param(
            lambda: np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
            "[[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]",
            id="uint8-image-2x2x3",
        ),
        pytest.param(lambda: np.matrix([[1, 2], [3, 4]]), "[[1, 2], [3, 4]]", id="matrix-subclass"),
        pytest.param(
            lambda: np.arange(4, dtype=np.uint8).reshape(2, 1, 2).view(Tagged),
            "[[[0, 1]], [[2, 3]]]",
            id="user-subclass-view",
        ),
        pytest.param(
            lambda: np.array(["2020-01-01", "2021-02-03"], dtype="datetime64[D]"),
            '["2020-01-01", "2021-02-03"]',
            id="datetime64-days",
        ),
        pytest.param(
            lambda: np.array(["2020-01-01T00:00:01"], dtype="datetime64[s]"),
            '["2020-01-01T00:00:01"]',
            id="datetime64-seconds",
        ),
        pytest.param(
            lambda: np.ma.array([1, 2, 3], mask=[False, True, False]), "[1, null, 3]", id="masked-values-as-null"
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:the matrix subclass:PendingDeprecationWarning")
def test_array_serializes_as_its_nested_lists(make, text):
    assert serialize_json(make()) == text
    # the intermediate tree is the parsed text (NaN does not compare equal to itself: compare dumps)
    expected = json.dumps(json.loads(text))
    assert json.dumps(serialize_json_as_obj(make())) == expected
    assert json.dumps(serialize_json_as_dict(make())) == expected


@pytest.mark.parametrize("cls", INTEGER_CLASSES, ids=_class_id)
def test_integer_array_of_dtype_limits_serializes_exact_integers(cls):
    info = np.iinfo(cls)
    array = np.array([info.min, 0, info.max], dtype=cls)

    assert serialize_json(array) == f"[{info.min}, 0, {info.max}]"
    parsed = to_json_value(array)
    assert parsed == [info.min, 0, info.max]
    assert all(type(item) is int for item in parsed)


@pytest.mark.parametrize("cls", [one for one in INTEGER_CLASSES if one is not np.uint64], ids=_class_id)
def test_integer_array_of_dtype_limits_restores_exact_integers(cls):
    info = np.iinfo(cls)
    array = np.array([info.min, 0, info.max], dtype=cls)

    restored = round_trip(array, np.ndarray)

    assert restored.tolist() == [info.min, 0, info.max]  # exact Python ints, no float rounding on the way
    assert all(type(item) is int for item in restored.tolist())


def test_uint64_array_holding_only_values_beyond_int64_restores_exactly():
    # NumPy infers uint64 when every integer is above the int64 maximum, so nothing is rounded
    array = np.array([2 ** 63, 2 ** 64 - 1], dtype=np.uint64)

    restored = assert_round_trip(array, np.ndarray)

    assert restored.dtype == np.uint64


@pytest.mark.parametrize(
    "values",
    [
        pytest.param([0, 2 ** 64 - 1], id="zero-and-uint64-max"),
        pytest.param([1, 2 ** 63 + 1], id="one-and-int64-max-plus-two"),
        pytest.param([2 ** 63 + 1, 5, 2 ** 64 - 2], id="large-and-small"),
    ],
)
def test_uint64_array_mixing_small_and_beyond_int64_values_loses_precision_limitation(values):
    # Documents current behavior, not a promise: the text is exact, but `np.array` of such a list is float64 (the
    # same inference as for the narrow dtypes), so the low digits are rounded away. An object or uint64 array
    # would be exact.
    array = np.array(values, dtype=np.uint64)

    assert serialize_json(array) == json.dumps(values)
    restored = round_trip(array, np.ndarray)

    assert restored.dtype == np.float64
    assert restored.tolist() == [float(item) for item in values]


@pytest.mark.parametrize("cls", FLOAT_CLASSES, ids=_class_id)
def test_float_array_of_exact_binary_fractions_serializes_the_same_for_every_width(cls):
    array = np.array([0.5, -1.5, 2.0 ** -10, 1024.0], dtype=cls)

    assert serialize_json(array) == "[0.5, -1.5, 0.0009765625, 1024.0]"


def test_narrow_float_values_serialize_as_the_python_float_of_their_exact_value():
    # 0.1 is not representable: the number written is the exact value of the nearest float32 / float16
    assert serialize_json(np.array([0.1], dtype=np.float32)) == "[0.10000000149011612]"
    assert serialize_json(np.array([0.1], dtype=np.float16)) == "[0.0999755859375]"
    assert serialize_json(np.float32(0.1)) == json.dumps(float(np.float32(0.1)))
    assert serialize_json(np.float16(0.1)) == json.dumps(float(np.float16(0.1)))


def test_seeded_random_arrays_serialize_as_their_own_tolist():
    rng = np.random.default_rng(1234)
    arrays = [
        rng.random((50, 40)),
        rng.standard_normal((5, 6, 7)),
        rng.integers(-2 ** 31, 2 ** 31, size=(30, 30), dtype=np.int_),
        rng.integers(0, 2, size=(10, 10)).astype(bool),
    ]

    for array in arrays:
        assert to_json_value(array) == array.tolist()


def test_serializing_an_array_does_not_change_it_or_its_flags():
    base = np.arange(12.0).reshape(3, 4)
    read_only = np.arange(6).reshape(2, 3)
    read_only.flags.writeable = False
    nested_object = _object_array({"a": [1, 2]}, [3, {"b": None}])
    arrays = [
        read_only,
        np.asfortranarray(base),
        base[::2, 1:],
        np.broadcast_to(np.arange(3), (2, 3)),
        nested_object,
        np.array([np.nan, 1.0, -0.0]),
        np.array(7),
    ]

    for array in arrays:
        before = _state(array)
        first = serialize_json(array)
        assert serialize_json(array) == first
        assert _state(array) == before


def test_serializing_an_object_array_does_not_change_the_objects_it_holds():
    inner = {"a": [1, 2], "b": {"c": (3, 4)}}
    array = _object_array(inner, [5, 6])
    before = copy.deepcopy(inner)

    assert serialize_json(array) == '[{"a": [1, 2], "b": {"c": [3, 4]}}, [5, 6]]'
    assert inner == before
    assert type(inner["b"]["c"]) is tuple


# --- python lists versus arrays built from them ------------------------------------------------------------------

@pytest.mark.parametrize(
    "items",
    [
        pytest.param([1, 2, 3], id="ints"),
        pytest.param([[1, 2], [3, 4], [5, 6]], id="nested-ints"),
        pytest.param([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], id="3d-ints"),
        pytest.param([-5, 0, 2 ** 40], id="large-ints"),
        pytest.param([0.5, 2.0, -1e-07, 1e22], id="floats"),
        pytest.param([[0.25, 0.5], [0.75, 1.0]], id="nested-floats"),
        pytest.param([True, False], id="bools"),
        pytest.param(["a", "bé", ""], id="strings"),
        pytest.param([[]], id="one-empty-row"),
        pytest.param([], id="empty"),
    ],
)
def test_array_built_from_a_python_list_serializes_like_the_list(items):
    array = np.array(items)

    assert serialize_json(array) == serialize_json(items)
    assert to_json_value(array) == items
    assert serialize_json(array, indent=2) == serialize_json(items, indent=2)
    assert serialize_json(array, separators=(",", ":")) == serialize_json(items, separators=(",", ":"))


def test_mixed_int_and_float_list_becomes_an_all_float_array():
    # np.array promotes, so the array writes 1.0 where the list wrote 1
    assert serialize_json([1, 2.5]) == "[1, 2.5]"
    assert serialize_json(np.array([1, 2.5])) == "[1.0, 2.5]"


# --- strict round trips of arrays --------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: np.array(5), id="0d-int"),
        pytest.param(lambda: np.array(-2.5), id="0d-float"),
        pytest.param(lambda: np.array(False), id="0d-bool"),
        pytest.param(lambda: np.array("text"), id="0d-str"),
        pytest.param(lambda: np.array([7]), id="single-element"),
        pytest.param(lambda: np.array([[7.5]]), id="single-element-2d"),
        pytest.param(lambda: np.arange(10), id="arange"),
        pytest.param(lambda: np.arange(6).reshape(2, 3), id="2d"),
        pytest.param(lambda: np.arange(24).reshape(2, 3, 4), id="3d-2x3x4"),
        pytest.param(lambda: np.arange(1024).reshape((2,) * 10), id="10d-binary-shape"),
        pytest.param(lambda: np.zeros((1,) * 30), id="30d-ones-shape"),
        pytest.param(lambda: np.arange(100_000), id="large-1d"),
        pytest.param(lambda: np.array([]), id="empty"),
        pytest.param(lambda: np.zeros((2, 0)), id="empty-rows-2x0"),
        pytest.param(lambda: np.zeros((3, 0)), id="empty-rows-3x0"),
        pytest.param(lambda: np.array([True, False, True]), id="bool"),
        pytest.param(lambda: np.array([[True, False], [False, False]]), id="bool-2d"),
        pytest.param(lambda: np.array(["a", "bb", "ccc"]), id="str"),
        pytest.param(lambda: np.array(["bé", "שלום", "\U0001F600", ""]), id="str-unicode"),
        pytest.param(lambda: np.array([2 ** 63 - 1, -2 ** 63]), id="int64-limits"),
        pytest.param(lambda: np.array([2 ** 64 - 1], dtype=np.uint64), id="uint64-max-alone"),
        pytest.param(lambda: np.linspace(0, 1, 7), id="linspace"),
        pytest.param(lambda: np.eye(4), id="eye"),
        pytest.param(lambda: np.array([np.nan, np.inf, -np.inf, 1.5]), id="non-finite"),
        pytest.param(
            lambda: np.array([1.7976931348623157e308, 5e-324, 2.2250738585072014e-308, 0.1 + 0.2]), id="float-extremes"
        ),
        pytest.param(lambda: np.asfortranarray(np.arange(6).reshape(2, 3)), id="fortran-ordered"),
        pytest.param(lambda: np.arange(12).reshape(3, 4).T, id="transposed-view"),
        pytest.param(lambda: np.arange(20).reshape(4, 5)[::2, 1::2], id="strided-slice-view"),
        pytest.param(lambda: np.arange(5)[::-1], id="negative-stride-view"),
        pytest.param(lambda: np.broadcast_to(np.arange(3.0), (4, 3)), id="read-only-broadcast-view"),
        pytest.param(lambda: np.random.default_rng(1234).random((100, 100)), id="seeded-random-floats"),
        pytest.param(
            lambda: np.random.default_rng(1234).integers(-1000, 1000, size=(20, 30), dtype=np.int_),
            id="seeded-random-ints",
        ),
        pytest.param(lambda: np.array([None, None], dtype=object), id="object-none"),
        pytest.param(lambda: np.array([1, None], dtype=object), id="object-int-and-none"),
        pytest.param(lambda: _object_array({"a": 1}, None, "s", 2.5), id="object-mixed-plain-values"),
        pytest.param(lambda: _object_array({"a": [1, {"b": None}]}, {"c": []}), id="object-dicts"),
        pytest.param(lambda: np.array([[1, None], [2, 3]], dtype=object), id="object-2d-with-none"),
    ],
)
def test_array_round_trips_with_dtype_shape_and_values(make):
    value = make()

    restored = assert_round_trip(value, np.ndarray)

    assert restored is not value
    assert restored.shape == value.shape and restored.dtype == value.dtype


def test_negative_zero_survives_an_array_round_trip():
    restored = assert_round_trip(np.array([-0.0, 0.0, -0.0]), np.ndarray)

    assert np.signbit(restored).tolist() == [True, False, True]


def test_non_finite_floats_need_allow_nan():
    for array in (np.array([np.nan]), np.array([1.0, np.inf]), np.array([-np.inf], dtype=np.float32)):
        with pytest.raises(ValueError, match="Out of range float values"):
            serialize_json(array, allow_nan=False)

    finite = np.array([[1.5, -2.5]])
    assert serialize_json(finite, allow_nan=False) == serialize_json(finite) == "[[1.5, -2.5]]"
    assert_round_trip(finite, np.ndarray, allow_nan=False)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.float16("nan"), id="float16-nan"),
        pytest.param(np.float32("inf"), id="float32-inf"),
        pytest.param(np.float64("-inf"), id="float64-negative-inf"),
    ],
)
def test_non_finite_float_scalars_need_allow_nan(value):
    with pytest.raises(ValueError, match="Out of range float values"):
        serialize_json(value, allow_nan=False)
    with pytest.raises(ValueError, match="Out of range float values"):
        serialize_json({"nested": [value]}, allow_nan=False)


def test_output_options_apply_inside_arrays():
    grid = np.arange(4).reshape(2, 2)

    assert serialize_json(grid, indent=2) == "[\n  [\n    0,\n    1\n  ],\n  [\n    2,\n    3\n  ]\n]"
    assert serialize_json(grid, separators=(",", ":")) == "[[0,1],[2,3]]"
    assert serialize_json({"b": np.array([1]), "a": np.array([2])}, sort_keys=True) == '{"a": [2], "b": [1]}'
    assert serialize_json(np.array(["é"])) == '["\\u00e9"]'
    assert serialize_json(np.array(["é"]), ensure_ascii=False) == '["é"]'
    assert serialize_json(np.str_("é"), ensure_ascii=False) == '"é"'
    assert_round_trip(grid, np.ndarray, indent=2, sort_keys=True)


def test_output_options_apply_to_dataclasses_with_arrays():
    value = Frame("f", np.arange(2))

    assert serialize_json(value) == '{"name": "f", "grid": [0, 1]}'
    assert serialize_json(value, sort_keys=True) == '{"grid": [0, 1], "name": "f"}'
    assert serialize_json(value, indent=1) == '{\n "name": "f",\n "grid": [\n  0,\n  1\n ]\n}'
    assert_round_trip(value, Frame, indent=1, sort_keys=True, separators=(",", ":"))


def test_registered_numpy_handlers_take_precedence_over_a_fallback():
    calls = []

    def fallback(value):
        calls.append(type(value))
        return "from the fallback"

    value = [np.arange(2), np.int8(1), np.bool_(True), np.str_("x"), np.float32(1.5), np.datetime64("2020-01-02")]

    assert serialize_json(value, fallback=fallback) == serialize_json(value)
    assert calls == []


def test_intermediate_tree_of_numpy_values_holds_plain_python_types_only():
    value = {
        "array": np.arange(4.0).reshape(2, 2),
        "scalars": [np.int8(1), np.float32(0.5), np.bool_(True), np.str_("s"), np.uint64(2 ** 64 - 1)],
        "when": np.datetime64("2020-01-02"),
        "objects": _object_array(1, None, "x"),
    }

    tree = serialize_json_as_obj(value)

    _assert_plain_json(tree)
    assert tree == {
        "array": [[0.0, 1.0], [2.0, 3.0]],
        "scalars": [1, 0.5, True, "s", 18446744073709551615],
        "when": "2020-01-02",
        "objects": [1, None, "x"],
    }
    tree["array"][0][0] = 99.0  # detached: the next serialization is unaffected
    assert serialize_json_as_obj(value)["array"][0][0] == 0.0


# --- deserializing arrays: the dtype comes from the data ---------------------------------------------------------

@pytest.mark.parametrize(
    ("data", "kind", "shape"),
    [
        pytest.param([1, 2, 3], "i", (3,), id="ints"),
        pytest.param([[1, 2], [3, 4]], "i", (2, 2), id="nested-ints"),
        pytest.param([1.5, 2], "f", (2,), id="float-and-int"),
        pytest.param([True, False], "b", (2,), id="bools"),
        pytest.param(["a", "bb"], "U", (2,), id="strings"),
        pytest.param(5, "i", (), id="scalar-int-as-0d"),
        pytest.param(2.5, "f", (), id="scalar-float-as-0d"),
        pytest.param([], "f", (0,), id="empty-is-float"),
        pytest.param([[], []], "f", (2, 0), id="empty-rows"),
        pytest.param([None, None], "O", (2,), id="nulls-make-an-object-array"),
        pytest.param([1, None], "O", (2,), id="int-and-null-make-an-object-array"),
        pytest.param([[1, 2.5], [3, 4]], "f", (2, 2), id="int-and-float-promote"),
        pytest.param([{"a": 1}, {"b": 2}], "O", (2,), id="dicts-make-an-object-array"),
    ],
)
def test_deserialized_array_takes_kind_and_shape_from_the_data(data, kind, shape):
    restored = deserialize_json(data, np.ndarray)

    assert type(restored) is np.ndarray
    assert restored.dtype.kind == kind
    assert restored.shape == shape
    if kind == "f":
        assert restored.dtype == np.float64


def test_default_dtypes_of_deserialized_arrays():
    assert deserialize_json([1, 2], np.ndarray).dtype == np.dtype(int)
    assert deserialize_json([1.5], np.ndarray).dtype == np.float64
    assert deserialize_json([True], np.ndarray).dtype == np.bool_
    assert deserialize_json(["ab", "c"], np.ndarray).dtype == np.dtype("U2")


@pytest.mark.parametrize(
    ("make", "default_dtype"),
    [
        pytest.param(lambda: np.array([1, -2], dtype=np.int8), np.dtype(int), id="int8"),
        pytest.param(lambda: np.array([0, 255], dtype=np.uint8), np.dtype(int), id="uint8"),
        pytest.param(lambda: np.array([-300, 300], dtype=np.int16), np.dtype(int), id="int16"),
        pytest.param(lambda: np.array([0, 60000], dtype=np.uint16), np.dtype(int), id="uint16"),
        pytest.param(lambda: np.array([0.5, -1.5], dtype=np.float16), np.dtype(float), id="float16"),
        pytest.param(lambda: np.array([0.5, -1.5], dtype=np.float32), np.dtype(float), id="float32"),
        pytest.param(lambda: np.array([0.1], dtype=np.float32), np.dtype(float), id="float32-inexact"),
        pytest.param(lambda: np.array([], dtype=np.int64), np.dtype(float), id="empty-int64"),
        pytest.param(lambda: np.array([1, 2.5], dtype=object), np.dtype(float), id="object-numbers"),
    ],
)
def test_narrow_dtypes_come_back_as_the_default_dtype_limitation(make, default_dtype):
    # Documents current behavior, not a promise: the JSON text has no dtype, so values survive but the dtype is
    # the one NumPy infers from the data. A bare `np.ndarray` target cannot ask for more.
    original = make()

    restored = round_trip(original, np.ndarray)

    assert restored.dtype == default_dtype
    assert restored.shape == original.shape
    assert np.array_equal(restored, original)


@pytest.mark.parametrize(
    ("make", "restored_dtype_kind"),
    [
        pytest.param(lambda: np.array([b"ab", b"c"]), "U", id="bytes"),
        pytest.param(lambda: np.array(["2020-01-01"], dtype="datetime64[D]"), "U", id="datetime64-days"),
    ],
)
def test_bytes_and_datetime_arrays_come_back_as_text_arrays_limitation(make, restored_dtype_kind):
    # Documents current behavior, not a promise: the elements are written as Base64 / ISO text, and a bare
    # `np.ndarray` target restores that text as it is.
    original = make()

    restored = round_trip(original, np.ndarray)

    assert restored.dtype.kind == restored_dtype_kind
    assert restored.tolist() == to_json_value(original)


def test_empty_array_with_a_zero_leading_axis_loses_its_other_axes_limitation():
    # Documents current behavior, not a promise: `[]` is all the JSON text holds of a (0, 3) array.
    assert round_trip(np.zeros((0, 3)), np.ndarray).shape == (0,)
    assert round_trip(np.zeros((0, 2, 3)), np.ndarray).shape == (0,)
    # a zero-width axis behind a non-empty one is still visible in the text
    assert round_trip(np.zeros((3, 0)), np.ndarray).shape == (3, 0)


def test_object_arrays_of_numbers_or_arrays_come_back_as_numeric_arrays_limitation():
    # Documents current behavior, not a promise: the object dtype and the inner arrays are not in the text.
    pair = _object_array(np.array([1, 2]), np.array([3, 4]))
    assert serialize_json(pair) == "[[1, 2], [3, 4]]"

    restored = round_trip(pair, np.ndarray)

    assert restored.dtype == np.dtype(int)
    assert restored.shape == (2, 2)


def test_integers_mixing_negatives_and_values_beyond_int64_become_floats_limitation():
    # Documents current behavior, not a promise: no integer dtype holds both -1 and 2**63 + 1 (only an object array
    # would), so `np.array` picks float64 and the low digit is gone. Such text comes from an object array or by hand.
    restored = deserialize_json([-1, 2 ** 63 + 1], np.ndarray)

    assert restored.dtype == np.float64
    assert restored.tolist() == [-1.0, float(2 ** 63)]
    # beyond uint64 as well, NumPy keeps exact Python ints in an object array
    assert deserialize_json([0, 2 ** 64], np.ndarray).tolist() == [0, 2 ** 64]


@pytest.mark.parametrize("cls", [np.ma.MaskedArray, np.matrix, Tagged], ids=_class_id)
def test_array_subclass_targets_return_plain_arrays_limitation(cls):
    # Documents current behavior, not a promise: the target class is not used to build an array.
    restored = deserialize_json([[1, 2], [3, 4]], cls)

    assert type(restored) is np.ndarray
    assert restored.tolist() == [[1, 2], [3, 4]]


def test_structured_arrays_lose_their_field_names_limitation():
    # Documents current behavior, not a promise: `tolist()` of a structured array is a list of tuples, which are
    # written as arrays, so the JSON text holds the values of each row but not the names or dtypes of the fields.
    rows = np.array([(1, 2.5), (2, 3.5)], dtype=[("a", "i4"), ("b", "f8")])

    assert serialize_json(rows) == "[[1, 2.5], [2, 3.5]]"
    assert serialize_json(rows[0]) == "[1, 2.5]"  # a np.void row
    assert serialize_json(np.rec.array(rows)) == "[[1, 2.5], [2, 3.5]]"
    assert round_trip(rows, np.ndarray).dtype == np.float64


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(5, id="int"),
        pytest.param(2.5, id="float"),
        pytest.param("text", id="str"),
        pytest.param(True, id="bool"),
    ],
)
def test_json_scalar_restores_as_a_0d_array(data):
    # 0-d arrays serialize as bare JSON scalars, so the way back must accept one
    restored = deserialize_json(data, np.ndarray)

    assert restored.shape == ()
    assert restored.tolist() == data
    assert type(restored.tolist()) is type(data)


# --- explicit dtype annotations --------------------------------------------------------------------------------------

def _annotated_ndarray(scalar_class):
    if not hasattr(np.ndarray, "__class_getitem__"):
        pytest.skip("parameterized np.ndarray needs numpy >= 1.22")
    return np.ndarray[Any, np.dtype[scalar_class]]


@pytest.mark.parametrize(
    ("scalar_class", "data", "inferred_dtype"),
    [
        pytest.param(np.float32, [1.5, 2.5], np.dtype(float), id="float32"),
        pytest.param(np.int8, [1, 2], np.dtype(int), id="int8"),
        pytest.param(np.float64, [1, 2], np.dtype(int), id="float64-annotation-int-data"),
    ],
)
def test_dtype_of_a_parameterized_ndarray_annotation_is_ignored_limitation(scalar_class, data, inferred_dtype):
    # Documents current behavior, not a promise: the annotation selects the array handler, and the array is built
    # from the data alone (`np.array(data)`), unlike `list[np.int8]` or `tuple[np.int8, ...]`, which honor the
    # element class. Whether a declared dtype should be applied is a design decision the library has not made.
    restored = deserialize_json(data, _annotated_ndarray(scalar_class))

    assert type(restored) is np.ndarray
    assert restored.dtype == inferred_dtype
    assert restored.tolist() == data


# --- ragged and otherwise unrestorable arrays ------------------------------------------------------------------------

_RAGGED_ERROR = pytest.mark.skipif(
    np.lib.NumpyVersion(np.__version__) < "1.24.0", reason="older NumPy builds an object array from ragged lists"
)
RAGGED = [
    pytest.param([[1], [2, 3]], id="rows-of-different-length"),
    pytest.param([[1, 2], [3]], id="short-last-row"),
    pytest.param([1, [2, 3]], id="scalar-and-list"),
    pytest.param([[[1, 2], [3]], [[4, 5], [6, 7]]], id="deep-ragged"),
]


@pytest.mark.parametrize("data", RAGGED)
@_RAGGED_ERROR
def test_ragged_data_is_rejected_with_type_error(data):
    with pytest.raises(TypeError):
        deserialize_json(data, np.ndarray)


@pytest.mark.parametrize("data", RAGGED)
@_RAGGED_ERROR
def test_union_moves_on_from_ragged_data(data):
    restored = deserialize_json(data, Union[np.ndarray, list])

    assert type(restored) is list and restored == data


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: _object_array(np.array([1, 2]), np.array([3])), id="arrays-of-different-length"),
        pytest.param(lambda: _object_array([1, 2], [3]), id="lists-of-different-length"),
    ],
)
def test_object_array_of_ragged_rows_serializes_as_nested_lists(make):
    assert serialize_json(make()) == "[[1, 2], [3]]"


# --- scalars ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("make", "text"),
    [
        pytest.param(lambda: np.int8(-5), "-5", id="int8"),
        pytest.param(lambda: np.uint8(250), "250", id="uint8"),
        pytest.param(lambda: np.int16(-300), "-300", id="int16"),
        pytest.param(lambda: np.uint16(60000), "60000", id="uint16"),
        pytest.param(lambda: np.int32(-2 ** 31), "-2147483648", id="int32-min"),
        pytest.param(lambda: np.uint32(2 ** 32 - 1), "4294967295", id="uint32-max"),
        pytest.param(lambda: np.int64(2 ** 63 - 1), "9223372036854775807", id="int64-max"),
        pytest.param(lambda: np.uint64(2 ** 64 - 1), "18446744073709551615", id="uint64-max"),
        pytest.param(lambda: np.float16(0.5), "0.5", id="float16"),
        pytest.param(lambda: np.float32(0.25), "0.25", id="float32"),
        pytest.param(lambda: np.float64(1.5), "1.5", id="float64"),
        pytest.param(lambda: np.float64(1e22), "1e+22", id="float64-exponent"),
        pytest.param(lambda: np.float64(0.1) + np.float64(0.2), "0.30000000000000004", id="float64-sum"),
        pytest.param(lambda: np.float32(0.1), "0.10000000149011612", id="float32-inexact"),
        pytest.param(lambda: np.float16(0.1), "0.0999755859375", id="float16-inexact"),
        pytest.param(lambda: np.float64(-0.0), "-0.0", id="negative-zero"),
        pytest.param(lambda: np.float32("nan"), "NaN", id="float32-nan"),
        pytest.param(lambda: np.float64("inf"), "Infinity", id="float64-inf"),
        pytest.param(lambda: np.float16("-inf"), "-Infinity", id="float16-negative-inf"),
        pytest.param(lambda: np.bool_(True), "true", id="bool-true"),
        pytest.param(lambda: np.bool_(False), "false", id="bool-false"),
        pytest.param(lambda: np.str_("x"), '"x"', id="str"),
        pytest.param(lambda: np.str_("é"), '"\\u00e9"', id="str-escaped"),
        pytest.param(lambda: np.bytes_(b"ab"), '"YWI="', id="bytes-as-base64"),
        pytest.param(lambda: np.datetime64("2020-01-02"), '"2020-01-02"', id="datetime64-day"),
        pytest.param(lambda: np.datetime64("2020-01-02T03:04:05"), '"2020-01-02T03:04:05"', id="datetime64-second"),
        pytest.param(
            lambda: np.datetime64("2020-01-02T03:04:05.123456"), '"2020-01-02T03:04:05.123456"', id="datetime64-micro"
        ),
        pytest.param(
            lambda: np.datetime64("2020-01-02T03:04:05.123"),
            '"2020-01-02T03:04:05.123000"',
            id="datetime64-milli-is-padded-to-micro",
        ),
    ],
)
def test_numpy_scalar_serializes_as_its_python_value(make, text):
    assert serialize_json(make()) == text
    assert json.dumps(serialize_json_as_obj(make())) == json.dumps(json.loads(text))


@pytest.mark.parametrize("cls", INTEGER_CLASSES, ids=_class_id)
def test_integer_scalar_limits_round_trip_with_their_exact_class(cls):
    info = np.iinfo(cls)

    for bound in (info.min, info.max):
        value = cls(bound)
        assert serialize_json(value) == str(bound)
        assert_round_trip(value, cls)


@pytest.mark.parametrize("cls", C_INTEGER_CLASSES)
def test_c_named_integer_classes_round_trip_with_their_exact_class(cls):
    info = np.iinfo(cls)

    for bound in (info.min, 0, info.max):
        value = cls(bound)
        assert serialize_json(value) == str(bound)
        restored = assert_round_trip(value, cls)
        assert type(restored) is cls


@pytest.mark.parametrize("cls", FLOAT_CLASSES, ids=_class_id)
def test_float_scalars_round_trip_with_their_exact_class(cls):
    for number in (0.5, -1.5, 0.1, 1e-3, 65504.0):  # 65504 is the largest float16
        assert_round_trip(cls(number), cls)


@pytest.mark.parametrize("cls", FLOAT_CLASSES, ids=_class_id)
def test_negative_zero_scalar_keeps_its_sign_through_a_round_trip(cls):
    # assert_strictly_equal compares float32 / float16 with ==, which cannot tell -0.0 from 0.0
    assert serialize_json(cls(-0.0)) == "-0.0"

    restored = round_trip(cls(-0.0), cls)

    assert type(restored) is cls and restored == 0 and np.signbit(restored)


def test_user_subclass_of_a_numpy_scalar_class_round_trips_as_itself():
    value = Celsius(21.5)

    assert serialize_json(value) == "21.5"
    restored = assert_round_trip(value, Celsius)
    assert type(restored) is Celsius
    assert serialize_json([Celsius(-0.5), Celsius(1e-07)]) == "[-0.5, 1e-07]"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.bool_(True), id="bool-true"),
        pytest.param(np.bool_(False), id="bool-false"),
        pytest.param(np.str_("hello"), id="str"),
        pytest.param(np.str_("שלום \U0001F600"), id="str-unicode"),
        pytest.param(np.str_(""), id="str-empty"),
        pytest.param(np.float64(-0.0), id="negative-zero"),
        pytest.param(np.float64("nan"), id="float64-nan"),
        pytest.param(np.float64("-inf"), id="float64-negative-inf"),
        pytest.param(np.datetime64("2020-01-02"), id="datetime64-day"),
        pytest.param(np.datetime64("2020-01-02T03:04:05"), id="datetime64-second"),
        pytest.param(np.datetime64("2020-01-02T03:04:05.123456"), id="datetime64-micro"),
        pytest.param(np.datetime64("2020-01-02T03:04:05.123"), id="datetime64-milli"),
        pytest.param(np.datetime64("0001-01-01"), id="datetime64-first-day"),
        pytest.param(np.datetime64("9999-12-31T23:59:59"), id="datetime64-last-second"),
    ],
)
def test_other_numpy_scalars_round_trip_with_their_exact_class(value):
    restored = assert_round_trip(value, type(value))

    assert type(restored) is type(value)


def test_float32_and_float16_nan_round_trip_as_nan_of_the_same_class():
    for cls in (np.float32, np.float16):
        restored = round_trip(cls("nan"), cls)
        assert type(restored) is cls and np.isnan(restored)


@pytest.mark.parametrize(
    ("text", "unit", "restored_unit"),
    [
        pytest.param("2020-01-02", "D", "D", id="day"),
        pytest.param("2020-01-02T03:04:05", "s", "s", id="second"),
        pytest.param("2020-01-02T03:04:05.123456", "us", "us", id="microsecond"),
        pytest.param("2020-01-02T03:04:05.123", "ms", "us", id="millisecond-comes-back-as-microsecond"),
        pytest.param("2020-01-02T03", "h", "s", id="hour-comes-back-as-second"),
        pytest.param("2020-03", "M", "D", id="month-comes-back-as-day"),
        pytest.param("2020", "Y", "D", id="year-comes-back-as-day"),
    ],
)
def test_datetime64_value_survives_a_round_trip_but_its_unit_may_change_limitation(text, unit, restored_unit):
    # Documents current behavior, not a promise: the text is the ISO string of the Python date / datetime that
    # `.item()` returns, so only day, second and microsecond precision can be told apart on the way back.
    value = np.datetime64(text, unit)
    assert value.dtype == np.dtype(f"datetime64[{unit}]")

    restored = round_trip(value, np.datetime64)

    assert restored == value
    assert restored.dtype == np.dtype(f"datetime64[{restored_unit}]")


@_NUMPY_RAISES_ON_OVERFLOW
@pytest.mark.parametrize(
    ("cls", "value"),
    [
        pytest.param(np.int8, 128, id="int8-above-max"),
        pytest.param(np.int8, -129, id="int8-below-min"),
        pytest.param(np.uint8, 256, id="uint8-above-max"),
        pytest.param(np.uint8, -1, id="uint8-negative"),
        pytest.param(np.uint16, 65536, id="uint16-above-max"),
        pytest.param(np.int32, 2 ** 31, id="int32-above-max"),
        pytest.param(np.int64, 2 ** 63, id="int64-above-max"),
        pytest.param(np.uint64, 2 ** 64, id="uint64-above-max"),
        pytest.param(np.uint64, -1, id="uint64-negative"),
    ],
)
def test_integer_outside_the_range_of_the_class_raises_a_data_free_type_error(cls, value):
    with pytest.raises(TypeError) as excinfo:
        deserialize_json(value, cls)

    assert str(excinfo.value) == f"Invalid value for {cls}, got int"
    with pytest.raises(TypeError):
        deserialize_json([0, value], list[cls])


@_NUMPY_RAISES_ON_OVERFLOW
def test_out_of_range_integer_lets_a_union_move_on_to_its_next_member():
    assert_strictly_equal(deserialize_json(300, Union[np.uint8, int]), 300)
    assert_strictly_equal(deserialize_json(200, Union[np.uint8, int]), np.uint8(200))
    assert_strictly_equal(deserialize_json([1, 300], list[Union[np.uint8, int]]), [np.uint8(1), 300])


@pytest.mark.parametrize("cls", [np.number, np.integer, np.floating, np.generic], ids=_class_id)
def test_abstract_numpy_scalar_classes_cannot_be_targets(cls):
    with pytest.raises(TypeError, match="Invalid value for"):
        deserialize_json(1, cls)


@_GENERIC_NAT_WARNING
def test_nat_serializes_as_null_limitation():
    # Documents current behavior, not a promise: `.item()` of NaT is None, and None is what the text holds. A null is
    # not a NumPy scalar (see test_null_is_not_a_numpy_scalar), so it cannot be read back as a datetime64.
    assert serialize_json(np.datetime64("NaT", "D")) == "null"
    assert serialize_json({"when": np.datetime64("NaT", "s")}) == '{"when": null}'
    with pytest.raises(TypeError):
        deserialize_json(None, np.datetime64)
    assert deserialize_json(None, Optional[np.datetime64]) is None


def test_nanosecond_datetime64_serializes_as_an_integer_limitation():
    # Documents current behavior, not a promise: a nanosecond value does not fit a `datetime`, so `.item()` is the
    # bare number of nanoseconds since the epoch, which no unit-less datetime64 can be built from.
    value = np.datetime64("2020-01-02T03:04:05.123456789")  # 1577934245 s after the epoch
    assert serialize_json(value) == "1577934245123456789"
    assert serialize_json(np.array(["2020-01-01T00:00:01"], dtype="datetime64[ns]")) == "[1577836801000000000]"

    with pytest.raises(TypeError, match="Invalid value for"):
        deserialize_json(1577934245123456789, np.datetime64)


def test_complex_and_timedelta_values_are_unsupported_limitation():
    # Documents current behavior, not a promise: like Python's own complex and timedelta, they have no JSON form.
    unsupported = [
        np.complex128(1 + 2j),
        np.array([1 + 2j]),
        np.timedelta64(5, "s"),
        np.array([5], dtype="timedelta64[s]"),
    ]

    for value in unsupported:
        with pytest.raises(TypeError, match="not JSON serializable"):
            serialize_json(value)


def test_complex_values_reach_the_fallback_as_python_complex_numbers():
    def as_pair(value):
        number = complex(value)
        return [number.real, number.imag]

    assert serialize_json(np.complex128(1 + 2j), fallback=as_pair) == "[1.0, 2.0]"
    assert serialize_json(np.array([1 + 2j, 3 - 1j]), fallback=as_pair) == "[[1.0, 2.0], [3.0, -1.0]]"
    assert serialize_json({"z": np.array([[0j]])}, fallback=as_pair) == '{"z": [[[0.0, 0.0]]]}'


@pytest.mark.parametrize(
    ("cls", "data", "expected"),
    [
        pytest.param(np.int64, 1.5, 1, id="int64-truncates-a-float"),
        pytest.param(np.int64, "7", 7, id="int64-parses-text"),
        pytest.param(np.int64, True, 1, id="int64-from-bool"),
        pytest.param(np.bool_, "false", True, id="bool-from-nonempty-text"),
        pytest.param(np.float64, "2.5", 2.5, id="float64-parses-text"),
    ],
)
def test_numpy_scalar_targets_coerce_any_primitive_through_numpy_limitation(cls, data, expected):
    # Documents current behavior, not a promise: unlike `int` or `str` targets, which pass a mismatched primitive
    # through, a NumPy scalar target feeds it to the NumPy constructor.
    restored = deserialize_json(data, cls)

    assert type(restored) is cls
    assert restored == expected


@pytest.mark.parametrize("cls", FLOAT_CLASSES, ids=_class_id)
def test_float_scalar_target_accepts_json_integers(cls):
    restored = deserialize_json(3, cls)

    assert type(restored) is cls
    assert restored == 3.0
    assert json.dumps(serialize_json_as_obj(restored)) == "3.0"


@pytest.mark.parametrize(
    ("cls", "data", "expected_type", "expected_text"),
    [
        pytest.param(np.float64, [1.0, 2.0], np.ndarray, "[1.0, 2.0]", id="float64-from-array"),
        pytest.param(np.int8, [1, 2], np.ndarray, "[1, 2]", id="int8-from-array"),
        pytest.param(np.bool_, [1, 0], np.ndarray, "[true, false]", id="bool-from-array"),
        pytest.param(np.str_, [1, 2], np.str_, '"[1, 2]"', id="str-from-array-is-its-repr"),
    ],
)
def test_numpy_scalar_target_given_a_json_array_is_not_rejected_limitation(cls, data, expected_type, expected_text):
    # Documents current behavior, not a promise: `np.float64([1.0, 2.0])` is an array in NumPy, so a scalar class
    # fed a JSON array returns whatever NumPy builds, where the container handlers raise TypeError for a wrong shape.
    restored = deserialize_json(data, cls)

    assert type(restored) is expected_type
    assert serialize_json(restored) == expected_text


def test_union_of_numpy_scalar_classes_takes_the_first_member_numpy_accepts_limitation():
    # Documents current behavior, not a promise: NumPy converts any number, so member order decides, and the first
    # member may truncate (Union[int, float] keeps 2.5 as it is, because `int` passes primitives through).
    assert_strictly_equal(deserialize_json(2.5, Union[np.int64, np.float64]), np.int64(2))
    assert_strictly_equal(deserialize_json(2.5, Union[np.float64, np.int64]), np.float64(2.5))
    assert_strictly_equal(deserialize_json(2, Union[np.float64, np.int64]), np.float64(2.0))
    assert_strictly_equal(deserialize_json(2.5, Union[int, float]), 2.5)


@dataclass
class LabelledReading:
    reading: np.float64
    label: np.str_


def test_null_is_not_a_numpy_scalar():
    # NumPy constructors accept None (np.float64(None) is nan, np.str_(None) 'None', np.bool_(None) False): none of
    # that is a restored null, and an Optional wants the null for None. A dataclass field missing from the data is
    # passed as None, so it is refused too.
    for cls in (np.float64, np.float32, np.str_, np.bool_, np.int8, np.int64, np.uint16):
        with pytest.raises(TypeError):
            deserialize_json(None, cls)

    with pytest.raises(TypeError):
        deserialize_json({}, LabelledReading)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.bytes_(b"ab"), id="ascii"),
        pytest.param(np.bytes_(b"\x00\xff binary"), id="binary"),
    ],
)
@pytest.mark.xfail(strict=True, reason="np.bytes_ serializes as Base64 text, but deserializing does not decode it")
def test_bytes_scalar_round_trips(value):
    assert_round_trip(value, np.bytes_)


_LONG_DOUBLE_LOOPS = pytest.mark.xfail(
    strict=True, reason="item() of a long double is a long double again: 'nested too deeply' after ~200k handler passes"
)


@pytest.mark.parametrize(
    ("make", "number_text"),
    [
        pytest.param(lambda: np.longdouble(1.5), 1.5, marks=_LONG_DOUBLE_LOOPS, id="longdouble-scalar"),
        pytest.param(
            lambda: np.array([1.5], dtype=np.longdouble), [1.5], marks=_LONG_DOUBLE_LOOPS, id="longdouble-array"
        ),
        pytest.param(
            lambda: {"x": [np.longdouble(1.5)]}, {"x": [1.5]}, marks=_LONG_DOUBLE_LOOPS, id="longdouble-nested"
        ),
    ],
)
def test_long_double_serializes_as_a_number_or_is_rejected_as_unsupported(make, number_text):
    # Either answer is acceptable (complex and timedelta64 raise TypeError); the endless loop is not.
    try:
        text = serialize_json(make())
    except TypeError:
        return
    assert json.loads(text) == number_text


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: np.clongdouble(1.5), marks=_LONG_DOUBLE_LOOPS, id="clongdouble-scalar"),
        pytest.param(lambda: np.array([1.5], dtype=np.clongdouble), marks=_LONG_DOUBLE_LOOPS, id="clongdouble-array"),
    ],
)
def test_complex_long_double_serializes_or_is_rejected_as_unsupported(make):
    try:
        text = serialize_json(make())
    except TypeError:
        return
    json.loads(text)  # a pair of numbers would do as well: only the endless loop is a defect


# --- scalars as dataclass fields and inside containers ----------------------------------------------------------------

@dataclass
class Scalars:
    i8: np.int8
    u8: np.uint8
    i16: np.int16
    u16: np.uint16
    i32: np.int32
    u32: np.uint32
    i64: np.int64
    u64: np.uint64
    f16: np.float16
    f32: np.float32
    f64: np.float64
    flag: np.bool_
    text: np.str_
    when: np.datetime64


def _scalars():
    return Scalars(
        i8=np.int8(-128), u8=np.uint8(255), i16=np.int16(-32768), u16=np.uint16(65535),
        i32=np.int32(-2 ** 31), u32=np.uint32(2 ** 32 - 1), i64=np.int64(-2 ** 63), u64=np.uint64(2 ** 64 - 1),
        f16=np.float16(0.1), f32=np.float32(0.1), f64=np.float64(0.1),
        flag=np.bool_(False), text=np.str_("é"), when=np.datetime64("2020-01-02T03:04:05"),
    )


def test_dataclass_of_numpy_scalars_has_the_json_object_of_its_python_values():
    assert to_json_value(_scalars()) == {
        "i8": -128, "u8": 255, "i16": -32768, "u16": 65535, "i32": -2147483648, "u32": 4294967295,
        "i64": -9223372036854775808, "u64": 18446744073709551615,
        "f16": 0.0999755859375, "f32": 0.10000000149011612, "f64": 0.1,
        "flag": False, "text": "é", "when": "2020-01-02T03:04:05",
    }
    assert list(to_json_value(_scalars())) == [one.name for one in dataclasses.fields(Scalars)]


def test_dataclass_of_numpy_scalars_round_trips_with_each_field_class():
    restored = assert_round_trip(_scalars(), Scalars)

    for one in dataclasses.fields(Scalars):
        assert type(getattr(restored, one.name)) is one.type


@dataclass
class Plain:
    x: float
    n: int
    ok: bool
    s: str


def test_numpy_values_in_fields_annotated_with_python_types_come_back_as_python_types():
    value = Plain(x=np.float32(0.5), n=np.int64(7), ok=np.bool_(True), s=np.str_("s"))

    assert serialize_json(value) == '{"x": 0.5, "n": 7, "ok": true, "s": "s"}'

    restored = deserialize_json(to_json_value(value), Plain)
    assert (type(restored.x), type(restored.n), type(restored.ok), type(restored.s)) == (float, int, bool, str)
    assert (restored.x, restored.n, restored.ok, restored.s) == (0.5, 7, True, "s")


@pytest.mark.parametrize(
    ("make", "annotation", "expected"),
    [
        pytest.param(lambda: [np.int8(1), np.int8(-2)], list[np.int8], [1, -2], id="list-of-int8"),
        pytest.param(
            lambda: (np.int8(1), np.float32(0.5), np.bool_(True)),
            tuple[np.int8, np.float32, np.bool_],
            [1, 0.5, True],
            id="fixed-tuple",
        ),
        pytest.param(
            lambda: (np.float16(0.5), np.float16(1.5), np.float16(-2.0)),
            tuple[np.float16, ...],
            [0.5, 1.5, -2.0],
            id="variadic-tuple",
        ),
        pytest.param(
            lambda: {"a": np.float32(0.5), "b": np.float32(-1.5)},
            dict[str, np.float32],
            {"a": 0.5, "b": -1.5},
            id="dict-values",
        ),
        pytest.param(
            lambda: {"k": [(np.uint8(1), np.float16(0.5))], "empty": []},
            dict[str, list[tuple[np.uint8, np.float16]]],
            {"k": [[1, 0.5]], "empty": []},
            id="nested-containers",
        ),
        pytest.param(
            lambda: {np.int64(10), np.int64(2), np.int64(1)},
            set[np.int64],
            [1, 10, 2],  # sets are ordered by the JSON text of their items
            id="set-of-int64",
        ),
        pytest.param(
            lambda: frozenset({np.float64(10.5), np.float64(2.5)}),
            frozenset[np.float64],
            [10.5, 2.5],
            id="frozenset-of-float64",
        ),
        pytest.param(lambda: {np.str_("b"), np.str_("a")}, set[np.str_], ["a", "b"], id="set-of-str"),
        pytest.param(
            lambda: frozenset({np.bool_(True), np.bool_(False)}),
            frozenset[np.bool_],
            [False, True],
            id="frozenset-of-bool",
        ),
        pytest.param(lambda: {np.float32(0.5), np.float32(0.25)}, set[np.float32], [0.25, 0.5], id="set-of-float32"),
    ],
)
def test_containers_of_numpy_scalars_round_trip_with_each_scalar_class(make, annotation, expected):
    value = make()

    assert to_json_value(value) == expected
    assert_round_trip(value, annotation)


def test_set_of_tuples_of_numpy_scalars_round_trips_with_the_scalar_classes():
    value = {(np.int8(1), np.int8(2)), (np.int8(0), np.int8(9)), (np.int8(1), np.int8(-1))}

    assert serialize_json(value) == "[[0, 9], [1, -1], [1, 2]]"  # ordered by the JSON text of each tuple
    restored = assert_round_trip(value, set[tuple[np.int8, np.int8]])
    # set equality cannot tell np.int8 from int: look at the members
    assert {type(item) for member in restored for item in member} == {np.int8}


def test_set_of_numpy_scalars_does_not_depend_on_insertion_order():
    forward = {np.int64(value) for value in (1, 2, 10, 20, 3)}
    backward = {np.int64(value) for value in (3, 20, 10, 2, 1)}

    assert serialize_json(forward) == serialize_json(backward) == "[1, 10, 2, 20, 3]"


@pytest.mark.parametrize(
    ("cls", "data"),
    [
        pytest.param(np.int8, 5, id="int8"),
        pytest.param(np.float32, 0.5, id="float32"),
        pytest.param(np.bool_, True, id="bool"),
        pytest.param(np.str_, "x", id="str"),
        pytest.param(np.datetime64, "2020-01-02", id="datetime64"),
    ],
)
def test_optional_numpy_scalars_restore_present_values_with_their_class(cls, data):
    for annotation in (Optional[cls], cls | None, Union[None, cls]):
        assert_strictly_equal(deserialize_json(data, annotation), cls(data))


@_GENERIC_NAT_WARNING
@pytest.mark.parametrize(
    "cls",
    [
        pytest.param(np.int8, id="int8"),
        pytest.param(np.uint16, id="uint16"),
        pytest.param(np.int64, id="int64"),
        pytest.param(np.float16, id="float16"),
        pytest.param(np.float32, id="float32"),
        pytest.param(np.float64, id="float64"),
        pytest.param(np.bool_, id="bool"),
        pytest.param(np.str_, id="str"),
        pytest.param(np.bytes_, id="bytes"),
        pytest.param(np.datetime64, id="datetime64"),
    ],
)
def test_optional_numpy_scalar_restores_null_as_none(cls):
    assert deserialize_json(None, Optional[cls]) is None


@dataclass
class MaybeScale:
    scale: Optional[np.float64] = None


@pytest.mark.parametrize(
    ("annotation", "data", "expected"),
    [
        pytest.param(
            list[Optional[np.int16]], [1, None, 3], [np.int16(1), None, np.int16(3)], id="int16-in-list"
        ),
        pytest.param(
            dict[str, Optional[np.uint8]], {"a": None, "b": 2}, {"a": None, "b": np.uint8(2)}, id="uint8-in-dict"
        ),
        pytest.param(
            list[Optional[np.float64]],
            [1.5, None, 2.5],
            [np.float64(1.5), None, np.float64(2.5)],
            id="float64-in-list",
        ),
        pytest.param(
            dict[str, Optional[np.str_]],
            {"a": None, "b": "x"},
            {"a": None, "b": np.str_("x")},
            id="str-in-dict",
        ),
    ],
)
def test_nulls_among_optional_numpy_scalars_stay_none(annotation, data, expected):
    # a series with missing readings: the null entries must not turn into nan / 'None'
    assert_strictly_equal(deserialize_json(data, annotation), expected)


def test_optional_numpy_scalar_field_set_to_none_round_trips():
    restored = assert_round_trip(MaybeScale(), MaybeScale)

    assert restored.scale is None


def test_none_listed_first_in_a_union_restores_null_as_none():
    for cls in (np.float64, np.str_, np.ndarray):
        assert deserialize_json(None, Union[None, cls]) is None

    assert type(deserialize_json(0.5, Union[None, np.float64])) is np.float64
    assert deserialize_json([1, 2], Union[None, np.ndarray]).tolist() == [1, 2]


# --- scalars as dict keys ----------------------------------------------------------------------------------------

def test_float64_and_str_keys_follow_their_python_base_class_limitation():
    # Documents current behavior, not a promise: np.float64 is a float and np.str_ a str, so they are written like
    # those; no other NumPy scalar is a subclass of a Python key type.
    assert serialize_json({np.float64(1.5): 1, np.float64(2.0): 2}) == '{"1.5": 1, "2.0": 2}'
    assert serialize_json({np.str_("k"): 1}) == '{"k": 1}'


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(np.int8(3), id="int8"),
        pytest.param(np.int64(3), id="int64"),
        pytest.param(np.uint8(3), id="uint8"),
        pytest.param(np.float32(1.5), id="float32"),
        pytest.param(np.bool_(True), id="bool"),
        pytest.param(np.bytes_(b"k"), id="bytes"),
    ],
)
def test_other_numpy_scalar_keys_are_rejected_limitation(key):
    # Documents current behavior, not a promise: as for json.dumps, they are not Python int / float / bool / str.
    with pytest.raises(TypeError, match=r"Unsupported JSON mapping key type: numpy\."):
        serialize_json({key: 1})


def test_tuple_key_of_numpy_scalars_is_the_json_text_of_the_tuple():
    assert serialize_json({(np.int64(1), np.float32(0.5)): "x"}) == '{"[1, 0.5]": "x"}'


def test_numpy_float_and_str_keys_cannot_be_restored_as_numpy_limitation():
    # Documents current behavior, not a promise: like int keys, a float key cannot be rebuilt from its text, and an
    # np.str_ key comes back as the plain str the text holds.
    with pytest.raises(ValueError, match="Couldn't de-convert JSON mapping key"):
        deserialize_json({"1.5": 1}, dict[np.float64, int])

    restored = deserialize_json({"k": 1}, dict[np.str_, int])
    assert restored == {"k": 1}
    assert [type(key) for key in restored] == [str]


# --- composites: arrays inside dataclasses and containers ------------------------------------------------------------

@dataclass
class Measurement:
    samples: np.ndarray
    label: str
    scale: np.float32
    count: np.int16
    valid: np.bool_
    note: Optional[str] = None


def test_dataclass_with_array_and_scalar_fields_has_the_expected_json_and_round_trips():
    value = Measurement(
        samples=np.array([[1.5, 2.5], [3.5, 4.5]]), label="run-1", scale=np.float32(0.5), count=np.int16(-3),
        valid=np.bool_(True),
    )

    assert serialize_json(value) == (
        '{"samples": [[1.5, 2.5], [3.5, 4.5]], "label": "run-1", "scale": 0.5, "count": -3, "valid": true, '
        '"note": null}'
    )
    restored = assert_round_trip(value, Measurement)
    assert type(restored.samples) is np.ndarray and type(restored.scale) is np.float32


@dataclass
class Frame:
    name: str
    grid: np.ndarray


@dataclass
class Experiment:
    title: str
    runs: dict[str, list[Frame]]


def test_three_by_three_arrays_inside_lists_inside_a_dict_round_trip():
    value = Experiment(
        title="exp",
        runs={
            "a": [Frame("f1", np.arange(9).reshape(3, 3)), Frame("f2", np.eye(3))],
            "b": [],
            "c": [Frame("f3", np.array([["x", "y", "z"]] * 3))],
        },
    )

    assert to_json_value(value) == {
        "title": "exp",
        "runs": {
            "a": [
                {"name": "f1", "grid": [[0, 1, 2], [3, 4, 5], [6, 7, 8]]},
                {"name": "f2", "grid": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]},
            ],
            "b": [],
            "c": [{"name": "f3", "grid": [["x", "y", "z"]] * 3}],
        },
    }
    assert_round_trip(value, Experiment)


@pytest.mark.parametrize(
    ("make", "annotation", "expected"),
    [
        pytest.param(
            lambda: [np.arange(3), np.eye(2), np.array(5), np.array([True, False]), np.array(["x", "y"])],
            list[np.ndarray],
            [[0, 1, 2], [[1.0, 0.0], [0.0, 1.0]], 5, [True, False], ["x", "y"]],
            id="list-of-mixed-arrays",
        ),
        pytest.param(lambda: [], list[np.ndarray], [], id="empty-list-of-arrays"),
        pytest.param(
            lambda: {"ints": np.arange(3), "floats": np.linspace(0, 1, 3), "empty": np.array([])},
            dict[str, np.ndarray],
            {"ints": [0, 1, 2], "floats": [0.0, 0.5, 1.0], "empty": []},
            id="dict-of-arrays",
        ),
        pytest.param(
            lambda: (np.arange(2), np.array([[0.5]])),
            tuple[np.ndarray, ...],
            [[0, 1], [[0.5]]],
            id="variadic-tuple-of-arrays",
        ),
        pytest.param(
            lambda: (np.arange(2), np.float64(1.5), "x"),
            tuple[np.ndarray, np.float64, str],
            [[0, 1], 1.5, "x"],
            id="fixed-tuple-array-scalar-str",
        ),
        pytest.param(lambda: np.arange(3), Optional[np.ndarray], [0, 1, 2], id="optional-array-present"),
        pytest.param(
            lambda: [[np.arange(2)], [], [np.eye(1), np.array(["z"])]],
            list[list[np.ndarray]],
            [[[0, 1]], [], [[[1.0]], ["z"]]],
            id="list-of-lists-of-arrays",
        ),
        pytest.param(
            lambda: {"outer": {"inner": np.arange(4).reshape(2, 2)}},
            dict[str, dict[str, np.ndarray]],
            {"outer": {"inner": [[0, 1], [2, 3]]}},
            id="dict-of-dicts-of-arrays",
        ),
        pytest.param(
            lambda: MappingProxyType({"a": np.arange(2), "b": np.array([[0.5]])}),
            MappingProxyType[str, np.ndarray],
            {"a": [0, 1], "b": [[0.5]]},
            id="mappingproxy-of-arrays",
        ),
        pytest.param(
            lambda: {"k": (np.array([1, 2]), np.int8(3))},
            dict[str, tuple[np.ndarray, np.int8]],
            {"k": [[1, 2], 3]},
            id="dict-of-array-scalar-tuples",
        ),
        pytest.param(
            lambda: [frozenset({np.int64(2), np.int64(1)}), frozenset()],
            list[frozenset[np.int64]],
            [[1, 2], []],
            id="list-of-frozenset-of-scalars",
        ),
    ],
)
def test_arrays_inside_containers_round_trip_with_the_annotation(make, annotation, expected):
    value = make()

    assert to_json_value(value) == expected
    assert_round_trip(value, annotation)


@dataclass
class Dataset:
    ids: np.ndarray
    mask: np.ndarray
    names: np.ndarray
    total: np.ndarray
    scores: np.ndarray
    notes: np.ndarray
    cube: np.ndarray


def test_dataclass_with_arrays_of_every_kind_has_the_expected_json_and_round_trips():
    value = Dataset(
        ids=np.arange(6).reshape(2, 3),
        mask=np.array([True, False, True]),
        names=np.array(["alpha", "beta"]),
        total=np.array(2.5),
        scores=np.zeros((3, 0)),
        notes=np.array([1, None, "x"], dtype=object),
        cube=np.arange(24.0).reshape(2, 3, 4)[:, ::2, 1:],
    )

    assert to_json_value(value) == {
        "ids": [[0, 1, 2], [3, 4, 5]],
        "mask": [True, False, True],
        "names": ["alpha", "beta"],
        "total": 2.5,
        "scores": [[], [], []],
        "notes": [1, None, "x"],
        "cube": [[[1.0, 2.0, 3.0], [9.0, 10.0, 11.0]], [[13.0, 14.0, 15.0], [21.0, 22.0, 23.0]]],
    }
    assert_round_trip(value, Dataset)


@dataclass
class Quoted:
    values: "np.ndarray"
    scale: "np.float32"
    count: "Optional[np.int16]" = None


def test_string_annotations_naming_numpy_classes_are_resolved():
    value = Quoted(values=np.array([[1, 2], [3, 4]]), scale=np.float32(0.25), count=np.int16(-7))

    restored = assert_round_trip(value, Quoted)

    assert type(restored.scale) is np.float32 and type(restored.count) is np.int16
    assert deserialize_json({"values": [1], "scale": 0.5}, Quoted).count is None


def test_dict_keyed_by_an_enum_with_array_values_round_trips():
    value = {Channel.RED: np.arange(3), Channel.GREEN: np.array([[0.5, 1.5]])}

    assert to_json_value(value) == {"r": [0, 1, 2], "g": [[0.5, 1.5]]}
    assert_round_trip(value, dict[Channel, np.ndarray])


def test_many_small_arrays_in_wide_containers_round_trip():
    arrays = [np.arange(index % 5 + 1) * index for index in range(1500)]
    mapping = {f"key-{index}": array for index, array in enumerate(arrays[:500])}

    assert to_json_value(arrays)[1499] == [index * 1499 for index in range(1499 % 5 + 1)]
    assert_round_trip(arrays, list[np.ndarray])
    assert_round_trip(mapping, dict[str, np.ndarray])
    assert_round_trip((arrays[:3], mapping), tuple[list[np.ndarray], dict[str, np.ndarray]])


@dataclass
class Reading:
    value: Union[float, np.ndarray]


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(Union[float, np.ndarray], id="Union"),
        pytest.param(float | np.ndarray, id="pipe"),
        pytest.param(Optional[Union[float, np.ndarray]], id="Optional-Union"),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(2.5, id="float"),
        pytest.param(np.array([1.5, 2.5]), id="1d-array"),
        pytest.param(np.arange(6.0).reshape(2, 3), id="2d-array"),
        pytest.param(np.array(["a", "b"]), id="str-array"),
    ],
)
def test_scalar_or_array_union_picks_the_member_that_fits_the_data(annotation, value):
    assert_round_trip(value, annotation)


def test_scalar_or_array_union_inside_a_dataclass_and_a_list():
    readings = [Reading(0.5), Reading(np.array([[1.0, 2.0]])), Reading(np.array([True, False]))]

    assert to_json_value(readings) == [{"value": 0.5}, {"value": [[1.0, 2.0]]}, {"value": [True, False]}]
    assert_round_trip(readings, list[Reading])


def test_zero_dim_array_in_a_scalar_or_array_union_comes_back_as_a_float_limitation():
    # Documents current behavior, not a promise: a 0-d array is written as a bare number, which the `float` member
    # of the union takes before the array member is tried.
    restored = round_trip(np.array(2.5), Union[float, np.ndarray])

    assert type(restored) is float and restored == 2.5


def test_union_member_order_decides_between_a_scalar_and_a_zero_dim_array_limitation():
    # Documents current behavior, not a promise: the array handler turns any JSON scalar into a 0-d array, so a
    # union that lists it first never reaches its other members.
    assert_strictly_equal(deserialize_json(2.5, Union[np.ndarray, float]), np.array(2.5))
    assert_strictly_equal(deserialize_json("abc", Union[np.ndarray, str]), np.array("abc"))
    assert_strictly_equal(deserialize_json("abc", Union[str, np.ndarray]), "abc")


@dataclass
class MaybeArray:
    values: Optional[np.ndarray] = None


def test_optional_array_field_with_an_array_round_trips():
    restored = assert_round_trip(MaybeArray(values=np.array([1.5, 2.5])), MaybeArray)

    assert type(restored.values) is np.ndarray


def test_optional_array_field_set_to_none_round_trips():
    restored = assert_round_trip(MaybeArray(), MaybeArray)

    assert restored.values is None


def test_optional_array_restores_null_as_none():
    assert deserialize_json(None, Optional[np.ndarray]) is None


def test_null_is_not_an_array():
    # np.array(None) is a 0-d object array: a null (or a dataclass field missing from the data) is refused instead
    with pytest.raises(TypeError):
        deserialize_json(None, np.ndarray)


@dataclass
class TreeNode:
    values: np.ndarray
    weight: np.float64
    children: "list[TreeNode]" = field(default_factory=list)


def _tree(depth, fanout):
    return TreeNode(
        values=np.arange(depth + 1, dtype=float),
        weight=np.float64(depth) / 4,
        children=[_tree(depth - 1, fanout) for _ in range(fanout)] if depth else [],
    )


def _chain(length):
    node = TreeNode(np.array([0.5]), np.float64(0.0))
    for level in range(1, length):
        node = TreeNode(np.arange(level + 1.0), np.float64(level), [node])
    return node


def test_recursive_dataclass_with_arrays_at_every_node_round_trips():
    tree = _tree(depth=4, fanout=2)

    restored = assert_round_trip(tree, TreeNode)

    assert to_json_value(tree)["children"][0]["children"][0]["values"] == [0.0, 1.0, 2.0]
    assert restored.children[1].children[0].values.tolist() == [0.0, 1.0, 2.0]


def test_deep_chain_of_dataclasses_holding_arrays_round_trips():
    chain = _chain(40)

    restored = assert_round_trip(chain, TreeNode)

    depth = 0
    while restored.children:
        restored = restored.children[0]
        depth += 1
    assert depth == 39


def test_array_at_the_bottom_of_deeply_nested_python_containers():
    depth = 100
    value = np.arange(3)
    for _ in range(depth):
        value = {"k": [value]}

    assert serialize_json(value) == '{"k": [' * depth + "[0, 1, 2]" + "]}" * depth

    node = to_json_value(value)
    for _ in range(depth):
        node = node["k"][0]
    assert node == [0, 1, 2]


def test_numpy_values_without_any_annotation_nest_inside_plain_containers():
    value = {
        "b": np.float32(0.5),
        "a": [np.int8(1), (np.uint16(2), np.array([3, 4]))],
        "c": {np.bool_(True)},
        "d": np.array([[1.5]]),
    }

    assert serialize_json(value) == '{"b": 0.5, "a": [1, [2, [3, 4]]], "c": [true], "d": [[1.5]]}'
    assert serialize_json(value, sort_keys=True) == '{"a": [1, [2, [3, 4]]], "b": 0.5, "c": [true], "d": [[1.5]]}'


@dataclass
class Loose:
    payload: dict[str, Any]
    tags: Any


def test_any_annotation_gets_plain_json_back_not_arrays():
    value = Loose(payload={"grid": np.arange(4).reshape(2, 2), "n": np.int16(5)}, tags=np.array(["a", "b"]))

    parsed = to_json_value(value)
    assert parsed == {"payload": {"grid": [[0, 1], [2, 3]], "n": 5}, "tags": ["a", "b"]}

    restored = deserialize_json(parsed, Loose)
    assert type(restored.payload["grid"]) is list and type(restored.tags) is list
    assert restored.payload == {"grid": [[0, 1], [2, 3]], "n": 5}


class Signal(Jsonable):
    """A Jsonable whose JSON form is built from NumPy values and whose `from_json` restores them."""

    def __init__(self, samples, rate):
        self.samples = samples
        self.rate = rate

    def to_json(self):
        return {"samples": self.samples, "rate": self.rate}

    @classmethod
    def from_json(cls, data):
        return cls(deserialize_json(data["samples"], np.ndarray), deserialize_json(data["rate"], np.float32))


@dataclass
class Recording:
    signal: Signal
    label: str


def test_jsonable_returning_numpy_values_is_serialized_recursively_and_restored():
    value = Recording(Signal(np.array([[1, 2], [3, 4]]), np.float32(0.5)), "take-1")

    assert serialize_json(value) == '{"signal": {"samples": [[1, 2], [3, 4]], "rate": 0.5}, "label": "take-1"}'

    restored = deserialize_json(to_json_value(value), Recording)
    assert type(restored.signal) is Signal
    assert_strictly_equal(restored.signal.samples, value.signal.samples)
    assert type(restored.signal.rate) is np.float32 and restored.signal.rate == np.float32(0.5)
    assert restored.label == "take-1"


def test_object_array_of_python_objects_serializes_each_element_with_its_handler():
    array = _object_array(datetime.date(2020, 1, 2), decimal.Decimal("1.50"), (1, 2), {3, 1}, b"ab", np.int8(4))

    assert serialize_json(array) == '["2020-01-02", "1.50", [1, 2], [1, 3], "YWI=", 4]'


def test_object_array_of_numpy_scalars_serializes_their_python_values():
    array = _object_array(np.int64(1), np.float32(2.5), np.bool_(True), np.str_("s"))

    assert serialize_json(array) == '[1, 2.5, true, "s"]'


def test_object_array_holding_itself_is_a_cyclic_reference():
    array = _object_array(1, None)
    array[1] = array

    with pytest.raises(ValueError, match="Cyclic reference"):
        serialize_json(array)


# --- result isolation -------------------------------------------------------------------------------------------------

@dataclass
class Holder:
    arr: np.ndarray
    extra: dict[str, Any]


@pytest.mark.parametrize(
    ("make", "annotation"),
    [
        pytest.param(lambda: [{"a": [1]}, {"b": [2, [3]]}], np.ndarray, id="object-array-of-dicts"),
        pytest.param(
            lambda: [[{"a": 1}, {"a": 2}], [{"a": 3}, {"a": [4]}]], np.ndarray, id="2d-object-array-of-dicts"
        ),
        pytest.param(lambda: {"a": [1, {"b": []}]}, np.ndarray, id="0d-object-array-of-a-dict"),
        pytest.param(lambda: [{"k": [1]}, None], np.ndarray, id="object-array-of-dict-and-none"),
        pytest.param(
            lambda: {"arr": [{"x": [1]}, {"x": [2]}], "extra": {"y": [3, {"z": 4}]}}, Holder, id="dataclass-field"
        ),
        pytest.param(lambda: [[{"a": [1]}], [{"b": [2]}]], list[np.ndarray], id="list-of-object-arrays"),
        pytest.param(lambda: {"k": [{"a": [1]}]}, dict[str, np.ndarray], id="dict-of-object-arrays"),
    ],
)
def test_deserialized_object_arrays_share_no_container_with_the_input(make, annotation):
    source = make()
    source_containers = _containers(source)

    result = deserialize_json(source, annotation)
    result_containers = _containers(result)

    assert source_containers and result_containers
    assert {id(one) for one in result_containers}.isdisjoint({id(one) for one in source_containers})

    serialized_before = serialize_json(result)
    _mutate(source_containers)  # editing the input afterwards must not reach the result
    assert serialize_json(result) == serialized_before

    source_after_its_own_edit = json.dumps(source)
    _mutate(_containers(result))  # and editing the result must not reach the input
    assert json.dumps(source) == source_after_its_own_edit


def test_deserialized_numeric_array_is_independent_of_the_input_list():
    source = [[1, 2], [3, 4]]

    result = deserialize_json(source, np.ndarray)
    result[0, 0] = 99
    source[1][1] = -1

    assert source == [[1, 2], [3, -1]]
    assert result.tolist() == [[99, 2], [3, 4]]


def test_deserialized_numeric_array_is_writable_and_owns_its_data():
    result = deserialize_json([1.5, 2.5, 3.5], np.ndarray)

    assert result.flags.writeable and result.flags.owndata
    result[:] = 0
    assert result.tolist() == [0.0, 0.0, 0.0]

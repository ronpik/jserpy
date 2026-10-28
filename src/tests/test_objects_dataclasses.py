"""Dataclasses from trivial to complex: what goes into the JSON and what comes back out of it.

Every case looks at both sides. The JSON side compares with ``json.dumps`` of a hand-written expectation, so key order,
``1`` versus ``1.0`` versus ``true`` and ``null`` are pinned. The Python side is a strict round trip to the very same
types (``assert_round_trip``), always against the annotation a user would really write.

Shapes the library gets wrong are kept as ``xfail(strict=True)`` cases, one per shape, so that the rest of the module
keeps running and a fix shows up as an unexpected pass. Behavior that is current but not a promise lives in tests whose
names end in ``_limitation``.

The module has no ``from __future__ import annotations`` on purpose: a class names itself (or a class defined after
it) with a quoted name inside a generic (``Optional["Node"]``, ``list["Tree"]``) or with a string for the whole
annotation (``"Optional[Node]"``), and both spellings are tested.
"""

import copy
import dataclasses
import datetime
import itertools
import json
from collections.abc import Mapping
from dataclasses import KW_ONLY, InitVar, dataclass, field, make_dataclass
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import (
    Annotated,
    Any,
    ClassVar,
    Dict,
    FrozenSet,
    Generic,
    List,
    Literal,
    NewType,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import pytest

from jserpy import deserialize_json, serialize_json, serialize_json_as_obj
from objects_support import assert_round_trip, assert_strictly_equal, to_json_value


def assert_text(value, expected):
    """``value`` is written as exactly the text ``json.dumps(expected)``: key order and number spelling included."""
    assert serialize_json(value) == json.dumps(expected)


def assert_json(value, expected):
    """``value`` is written as ``expected`` with the same types and array order, the object keys in any order.

    For values holding sets: the keys of a dataclass inside a set are sorted (see
    ``test_dataclass_keys_inside_a_set_are_sorted_limitation``), which is not what these tests are about.
    """
    actual = to_json_value(value)
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


class Color(Enum):
    RED = "red"
    GREEN = "green"


class Priority(Enum):
    LOW = 1
    HIGH = 2


# --- trivial dataclasses -----------------------------------------------------------------------------------------


@dataclass
class Empty:
    pass


@dataclass
class Wrapper:
    inner: Empty


@dataclass
class Single:
    value: int


@dataclass
class Primitives:
    integer: int
    real: float
    text: str
    flag: bool
    nothing: Optional[str] = None


@dataclass
class Scrambled:
    zeta: int
    alpha: int
    mid: int


@dataclass
class WithDefaults:
    name: str
    count: int = 3
    tags: List[str] = field(default_factory=list)
    ratio: float = 0.5
    note: Optional[str] = None


@dataclass
class IntDefault:
    count: int = 3


@dataclass
class FactoryDefault:
    tags: List[str] = field(default_factory=list)


@dataclass
class OptionalDefault:
    limit: Optional[int] = 10


Wide = make_dataclass("Wide", [(f"field_{index:03d}", int, field(default=index)) for index in range(100)])


@pytest.mark.parametrize(
    ("value", "cls", "text"),
    [
        pytest.param(Empty(), Empty, "{}", id="root"),
        pytest.param(Wrapper(Empty()), Wrapper, '{"inner": {}}', id="field"),
        pytest.param([Empty(), Empty()], List[Empty], "[{}, {}]", id="list"),
        pytest.param((Empty(),), Tuple[Empty, ...], "[{}]", id="tuple"),
        pytest.param({"a": Empty()}, Dict[str, Empty], '{"a": {}}', id="dict"),
        pytest.param(Empty(), Optional[Empty], "{}", id="optional-present"),
    ],
)
def test_empty_dataclass_is_an_empty_object(value, cls, text):
    assert serialize_json(value) == text
    assert_round_trip(value, cls)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param("text", id="string"),
        pytest.param(5, id="number"),
        pytest.param(True, id="boolean"),
        pytest.param([1], id="array"),
        pytest.param(None, id="null"),
    ],
)
def test_data_that_is_not_an_object_is_rejected_for_an_empty_dataclass(data):
    with pytest.raises(TypeError):
        deserialize_json(data, Empty)


@pytest.mark.parametrize(
    ("data", "cls", "expected"),
    [
        pytest.param(None, Optional[Empty], None, id="null-for-optional"),
        pytest.param(5, Union[Empty, int], 5, id="number-after-the-dataclass"),
        pytest.param("text", Union[Empty, str], "text", id="string-after-the-dataclass"),
        pytest.param([1], Union[Empty, List[int]], [1], id="array-after-the-dataclass"),
        pytest.param({}, Union[Empty, int], Empty(), id="object-for-the-dataclass"),
        pytest.param(None, Optional[Single], None, id="null-for-optional-dataclass-with-a-field"),
    ],
)
def test_empty_dataclass_does_not_swallow_the_other_members_of_a_union(data, cls, expected):
    assert_strictly_equal(deserialize_json(data, cls), expected)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(Single(7), '{"value": 7}', id="positive"),
        pytest.param(Single(0), '{"value": 0}', id="zero"),
        pytest.param(Single(-1), '{"value": -1}', id="negative"),
        pytest.param(Single(10**20), '{"value": 1' + "0" * 20 + "}", id="beyond-64-bit"),
    ],
)
def test_single_field_dataclass(value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, Single)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(
            Primitives(-7, 2.5, "plain", True),
            '{"integer": -7, "real": 2.5, "text": "plain", "flag": true, "nothing": null}',
            id="typical",
        ),
        pytest.param(
            Primitives(1, 1.0, "1", True, "1"),
            '{"integer": 1, "real": 1.0, "text": "1", "flag": true, "nothing": "1"}',
            id="look-alikes",
        ),
        pytest.param(
            Primitives(0, 0.0, "", False, ""),
            '{"integer": 0, "real": 0.0, "text": "", "flag": false, "nothing": ""}',
            id="falsy-values-are-kept",
        ),
        pytest.param(
            Primitives(0, -0.0, "x", False),
            '{"integer": 0, "real": -0.0, "text": "x", "flag": false, "nothing": null}',
            id="negative-zero",
        ),
        pytest.param(
            Primitives(2**63, 1e300, "x", True),
            '{"integer": 9223372036854775808, "real": 1e+300, "text": "x", "flag": true, "nothing": null}',
            id="beyond-64-bit-and-huge-float",
        ),
        pytest.param(
            Primitives(-1, 1e-7, "x", False),
            '{"integer": -1, "real": 1e-07, "text": "x", "flag": false, "nothing": null}',
            id="tiny-float",
        ),
    ],
)
def test_flat_dataclass_with_every_primitive_type(value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, Primitives)


@pytest.mark.parametrize(
    ("real", "token"),
    [
        pytest.param(float("nan"), "NaN", id="nan"),
        pytest.param(float("inf"), "Infinity", id="infinity"),
        pytest.param(float("-inf"), "-Infinity", id="negative-infinity"),
    ],
)
def test_non_finite_float_field(real, token):
    value = Primitives(1, real, "x", True)

    assert serialize_json(value) == (
        '{"integer": 1, "real": ' + token + ', "text": "x", "flag": true, "nothing": null}'
    )
    assert_round_trip(value, Primitives)
    with pytest.raises(ValueError, match="not JSON compliant"):
        serialize_json(value, allow_nan=False)


def test_fields_are_written_in_declaration_order():
    value = Scrambled(zeta=1, alpha=2, mid=3)

    assert serialize_json(value) == '{"zeta": 1, "alpha": 2, "mid": 3}'
    assert serialize_json(value, sort_keys=True) == '{"alpha": 2, "mid": 3, "zeta": 1}'


def test_the_order_of_keys_in_the_json_does_not_matter_when_restoring():
    restored = deserialize_json({"mid": 3, "zeta": 1, "alpha": 2}, Scrambled)

    assert_strictly_equal(restored, Scrambled(zeta=1, alpha=2, mid=3))


def test_default_values_are_written_like_any_other_value():
    assert serialize_json(WithDefaults("n")) == '{"name": "n", "count": 3, "tags": [], "ratio": 0.5, "note": null}'
    assert_round_trip(WithDefaults("n"), WithDefaults)
    assert_round_trip(WithDefaults("n", count=0, tags=["a", "b"], ratio=0.0, note=""), WithDefaults)


@pytest.mark.parametrize(
    ("cls", "expected"),
    [
        pytest.param(IntDefault, IntDefault(None), id="scalar"),
        pytest.param(OptionalDefault, OptionalDefault(None), id="optional-with-non-none-default"),
    ],
)
def test_absent_field_ignores_its_dataclass_default_limitation(cls, expected):
    # Documents current behavior, not a promise: a key missing from the JSON becomes None, whatever the field's
    # default is, so JSON written before a defaulted field was added does not pick up the default.
    assert_strictly_equal(deserialize_json({}, cls), expected)
    assert cls() != expected


def test_absent_field_with_a_default_factory_raises_limitation():
    # Documents current behavior, not a promise: the same None lands on an array annotation and is rejected.
    with pytest.raises(TypeError, match="JSON array"):
        deserialize_json({}, FactoryDefault)
    assert_strictly_equal(deserialize_json({"tags": []}, FactoryDefault), FactoryDefault([]))


def test_wide_dataclass_keeps_every_field_in_order():
    value = Wide(*range(100, 200))
    expected = {f"field_{index:03d}": 100 + index for index in range(100)}

    assert_text(value, expected)
    assert_round_trip(value, Wide)
    assert_round_trip(Wide(), Wide)


# --- Optional fields and values of the wrong shape ---------------------------------------------------------------


@dataclass
class Address:
    city: str
    zip_code: Optional[str] = None


@dataclass
class Profile:
    name: str
    nickname: Optional[str] = None
    age: Optional[int] = None
    tags: Optional[List[str]] = None
    address: Optional[Address] = None
    scores: Optional[Dict[str, float]] = None


@dataclass
class ListHolder:
    items: List[int]


@dataclass
class AddressHolder:
    address: Address


@dataclass
class PairHolder:
    pair: Tuple[int, int]


@dataclass
class DictHolder:
    mapping: Dict[str, int]


FULL_PROFILE_JSON = {
    "name": "Ada",
    "nickname": "ada",
    "age": 36,
    "tags": ["math", "code"],
    "address": {"city": "London", "zip_code": "N1"},
    "scores": {"chess": 1.5},
}


def full_profile():
    return Profile(
        name="Ada",
        nickname="ada",
        age=36,
        tags=["math", "code"],
        address=Address("London", "N1"),
        scores={"chess": 1.5},
    )


def test_optional_fields_that_are_present():
    assert_text(full_profile(), FULL_PROFILE_JSON)
    assert_round_trip(full_profile(), Profile)


@dataclass
class OptionalRich:
    when: Optional[datetime.datetime] = None
    day: Optional[datetime.date] = None
    price: Optional[Decimal] = None
    blob: Optional[bytes] = None
    path: Optional[Path] = None
    color: Optional[Color] = None
    flag: Optional[bool] = None
    mode: Optional[Literal["a", "b"]] = None
    pair: Optional[Tuple[int, int]] = None
    unique: Optional[Set[str]] = None


OPTIONAL_RICH_FIELDS = ("when", "day", "price", "blob", "path", "color", "flag", "mode", "pair", "unique")
AB_PATH = Path("a") / "b"


@pytest.mark.parametrize(
    ("field_name", "value", "json_value"),
    [
        pytest.param("when", datetime.datetime(2024, 1, 2, 3, 4, 5), "2024-01-02T03:04:05", id="datetime"),
        pytest.param("day", datetime.date(2024, 1, 2), "2024-01-02", id="date"),
        pytest.param("price", Decimal("1.0"), "1.0", id="decimal-keeps-its-exponent"),
        pytest.param("blob", b"x", "eA==", id="bytes"),
        pytest.param("path", AB_PATH, str(AB_PATH), id="path"),
        pytest.param("color", Color.RED, "red", id="enum"),
        pytest.param("flag", False, False, id="false-is-not-none"),
        pytest.param("mode", "b", "b", id="literal"),
        pytest.param("pair", (1, 2), [1, 2], id="fixed-tuple"),
        pytest.param("unique", {"z", "a"}, ["a", "z"], id="set"),
    ],
)
def test_optional_field_of_a_type_that_needs_converting(field_name, value, json_value):
    instance = OptionalRich(**{field_name: value})
    expected = dict.fromkeys(OPTIONAL_RICH_FIELDS)
    expected[field_name] = json_value

    assert_text(instance, expected)
    assert_round_trip(instance, OptionalRich)


def test_optional_fields_of_types_that_need_converting_are_all_none_or_all_present():
    everything = OptionalRich(
        datetime.datetime(2024, 1, 2, 3, 4, 5), datetime.date(2024, 1, 2), Decimal("1.0"), b"x", AB_PATH,
        Color.RED, False, "b", (1, 2), {"z", "a"},
    )

    assert_text(OptionalRich(), dict.fromkeys(OPTIONAL_RICH_FIELDS))
    assert_round_trip(OptionalRich(), OptionalRich)
    assert_round_trip(everything, OptionalRich)


def test_none_is_written_as_null_and_not_omitted():
    assert serialize_json(Profile("Ada")) == (
        '{"name": "Ada", "nickname": null, "age": null, "tags": null, "address": null, "scores": null}'
    )


def test_explicit_null_is_restored_as_none():
    data = {"name": "Ada", "nickname": None, "age": None, "tags": None, "address": None, "scores": None}

    restored = deserialize_json(data, Profile)

    assert_strictly_equal(restored, Profile("Ada"))
    assert_round_trip(Profile("Ada"), Profile)


def test_optional_fields_absent_from_the_json_are_restored_as_none():
    restored = deserialize_json({"name": "Ada"}, Profile)

    assert_strictly_equal(restored, Profile("Ada"))


def test_empty_values_of_optional_fields_are_not_mistaken_for_none():
    value = Profile("", nickname="", age=0, tags=[], address=None, scores={})

    assert serialize_json(value) == (
        '{"name": "", "nickname": "", "age": 0, "tags": [], "address": null, "scores": {}}'
    )
    assert_round_trip(value, Profile)


def test_unknown_keys_in_the_json_are_ignored_at_every_level_limitation():
    # Documents current behavior, not a promise: only the dataclass fields are read, other keys are neither an error
    # nor kept.
    data = {
        "name": "Ada",
        "unknown": 1,
        "tags": ["a"],
        "address": {"city": "London", "country": "UK", "extra": {"deep": [1, 2]}},
        "other": [{"x": 1}],
    }
    snapshot = copy.deepcopy(data)

    restored = deserialize_json(data, Profile)

    assert_strictly_equal(restored, Profile("Ada", tags=["a"], address=Address("London")))
    assert data == snapshot


def test_whole_numbers_written_by_other_encoders_become_floats_for_float_fields():
    data = {"integer": 1, "real": 2, "text": "t", "flag": True, "nothing": None}

    restored = deserialize_json(data, Primitives)
    scored = deserialize_json({"name": "Ada", "scores": {"chess": 2, "go": 0}}, Profile)

    assert_strictly_equal(restored, Primitives(1, 2.0, "t", True))
    assert_strictly_equal(scored, Profile("Ada", scores={"chess": 2.0, "go": 0.0}))


@pytest.mark.parametrize(
    "data",
    [pytest.param({}, id="absent"), pytest.param({"name": None}, id="explicit-null")],
)
def test_required_scalar_field_without_a_value_is_restored_as_none_limitation(data):
    # Documents current behavior, not a promise: a missing required field is not an error.
    restored = deserialize_json(data, Profile)

    assert_strictly_equal(restored, Profile(None))


@pytest.mark.parametrize(
    ("cls", "message"),
    [
        pytest.param(ListHolder, "JSON array", id="list"),
        pytest.param(PairHolder, "JSON array", id="tuple"),
        pytest.param(AddressHolder, None, id="nested-dataclass"),
    ],
)
def test_required_container_or_dataclass_field_without_a_value_raises_limitation(cls, message):
    # Documents current behavior, not a promise: unlike a scalar, the None stands in for an array or object and fails.
    with pytest.raises(TypeError, match=message):
        deserialize_json({}, cls)


def test_primitive_annotations_are_not_validated_limitation():
    # Documents current behavior, not a promise: a JSON primitive is handed to the field as it is.
    data = {"integer": "abc", "real": "x", "text": 5, "flag": "yes"}

    restored = deserialize_json(data, Primitives)

    assert (restored.integer, restored.real, restored.text, restored.flag) == ("abc", "x", 5, "yes")


@pytest.mark.parametrize(
    "data",
    [
        pytest.param("text", id="string"),
        pytest.param(5, id="number"),
        pytest.param(True, id="boolean"),
        pytest.param([{"name": "Ada"}], id="array"),
        pytest.param(None, id="null"),
    ],
)
def test_data_that_is_not_an_object_is_rejected_for_a_dataclass(data):
    with pytest.raises(TypeError):
        deserialize_json(data, Profile)


@pytest.mark.parametrize(
    "bad_field",
    [
        pytest.param({"tags": "abc"}, id="string-for-array"),
        pytest.param({"tags": {"a": 1}}, id="object-for-array"),
        pytest.param({"tags": 5}, id="number-for-array"),
        pytest.param({"address": "London"}, id="string-for-dataclass"),
        pytest.param({"address": ["London"]}, id="array-for-dataclass"),
        pytest.param({"address": 7}, id="number-for-dataclass"),
        pytest.param({"scores": [1.5]}, id="array-for-object"),
    ],
)
def test_field_of_the_wrong_shape_is_rejected(bad_field):
    with pytest.raises(TypeError):
        deserialize_json({"name": "Ada", **bad_field}, Profile)


@pytest.mark.parametrize(
    ("data", "cls"),
    [
        pytest.param([5], List[Address], id="number-in-list"),
        pytest.param(["London"], List[Address], id="string-in-list"),
        pytest.param({"home": []}, Dict[str, Address], id="array-in-dict"),
        pytest.param([None], Tuple[Address, ...], id="null-in-tuple"),
        pytest.param({"city": "London"}, List[Address], id="object-for-list"),
        pytest.param(5, Union[Address, List[Address]], id="number-for-union"),
    ],
)
def test_item_of_the_wrong_shape_is_rejected(data, cls):
    with pytest.raises(TypeError):
        deserialize_json(data, cls)


@pytest.mark.parametrize("bad", [pytest.param("abc", id="string"), pytest.param(5, id="number"), pytest.param([1], id="array")])
def test_dict_field_given_something_other_than_an_object_is_rejected(bad):
    # dict[str, int] refuses what Mapping[str, int] refuses
    with pytest.raises(TypeError):
        deserialize_json({"mapping": bad}, DictHolder)
    with pytest.raises(TypeError):
        deserialize_json(bad, Mapping[str, int])


def test_dict_field_still_takes_null_and_a_missing_value_as_none():
    # null is what a field absent from the data is passed as
    assert deserialize_json({"mapping": None}, DictHolder).mapping is None
    assert deserialize_json({}, DictHolder).mapping is None


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(Empty(), id="empty"),
        pytest.param(Single(0), id="single"),
        pytest.param(Primitives(1, 0.1, "a", False), id="primitives"),
        pytest.param(
            Primitives(-(10**30), 1e300, 'שלום 🙂 "quoted"\n\t\\', True, "x"), id="extreme-and-awkward-text"
        ),
        pytest.param(Wrapper(Empty()), id="nested-empty"),
        pytest.param(Address("London", "N1"), id="address"),
        pytest.param(Profile("Ada"), id="profile-with-nones"),
        pytest.param(full_profile(), id="profile-with-nested-values"),
        pytest.param(Wide(), id="wide"),
    ],
)
def test_json_of_a_dataclass_without_tuples_and_sets_equals_asdict(value):
    # dataclasses.asdict is an independent oracle for dataclasses made of JSON-native values only. The texts are
    # compared, not the values: 1, 1.0 and True are equal values but different JSON.
    assert json.dumps(to_json_value(value)) == json.dumps(dataclasses.asdict(value))
    assert serialize_json(value) == json.dumps(dataclasses.asdict(value))
    assert_round_trip(value, type(value))


# --- nesting of dataclasses --------------------------------------------------------------------------------------


@dataclass
class Level6:
    sixth: str


@dataclass
class Level5:
    fifth: int
    level6: Level6


@dataclass
class Level4:
    fourth: float
    level5: Level5


@dataclass
class Level3:
    third: bool
    level4: Level4


@dataclass
class Level2:
    second: List[int]
    level3: Level3


@dataclass
class Level1:
    first: str
    level2: Level2


LEVEL6 = Level6("bottom")
LEVEL5 = Level5(5, LEVEL6)
LEVEL4 = Level4(0.5, LEVEL5)
LEVEL3 = Level3(True, LEVEL4)
LEVEL2 = Level2([1, 2], LEVEL3)
LEVEL1 = Level1("top", LEVEL2)

LEVEL6_JSON = {"sixth": "bottom"}
LEVEL5_JSON = {"fifth": 5, "level6": LEVEL6_JSON}
LEVEL4_JSON = {"fourth": 0.5, "level5": LEVEL5_JSON}
LEVEL3_JSON = {"third": True, "level4": LEVEL4_JSON}
LEVEL2_JSON = {"second": [1, 2], "level3": LEVEL3_JSON}
LEVEL1_JSON = {"first": "top", "level2": LEVEL2_JSON}


@dataclass
class LevelHolder:
    level: Level1
    levels: List[Level1]
    by_name: Dict[str, Level1]


@pytest.mark.parametrize(
    ("cls", "value", "expected"),
    [
        pytest.param(Level6, LEVEL6, LEVEL6_JSON, id="depth-1"),
        pytest.param(Level5, LEVEL5, LEVEL5_JSON, id="depth-2"),
        pytest.param(Level4, LEVEL4, LEVEL4_JSON, id="depth-3"),
        pytest.param(Level3, LEVEL3, LEVEL3_JSON, id="depth-4"),
        pytest.param(Level2, LEVEL2, LEVEL2_JSON, id="depth-5"),
        pytest.param(Level1, LEVEL1, LEVEL1_JSON, id="depth-6"),
    ],
)
def test_chain_of_different_dataclasses(cls, value, expected):
    assert_text(value, expected)
    assert_round_trip(value, cls)


def test_deepest_chain_of_dataclasses_golden_text():
    assert serialize_json(LEVEL1) == (
        '{"first": "top", "level2": {"second": [1, 2], "level3": {"third": true, '
        '"level4": {"fourth": 0.5, "level5": {"fifth": 5, "level6": {"sixth": "bottom"}}}}}}'
    )


def test_chain_of_dataclasses_inside_other_containers_and_dataclasses():
    holder = LevelHolder(LEVEL1, [LEVEL1, LEVEL1], {"only": LEVEL1})
    expected = {"level": LEVEL1_JSON, "levels": [LEVEL1_JSON, LEVEL1_JSON], "by_name": {"only": LEVEL1_JSON}}

    assert_text(holder, expected)
    assert_round_trip(holder, LevelHolder)
    assert_round_trip([holder], List[LevelHolder])


ADA = Address("London", "N1")
BOB = Address("Paris")
ADA_JSON = {"city": "London", "zip_code": "N1"}
BOB_JSON = {"city": "Paris", "zip_code": None}


@pytest.mark.parametrize(
    ("value", "cls", "expected"),
    [
        pytest.param([], List[Address], [], id="list-empty"),
        pytest.param([ADA], List[Address], [ADA_JSON], id="list-singleton"),
        pytest.param([ADA, BOB, ADA], List[Address], [ADA_JSON, BOB_JSON, ADA_JSON], id="list-with-repeats"),
        pytest.param({}, Dict[str, Address], {}, id="dict-empty"),
        pytest.param(
            {Color.GREEN: ADA, Color.RED: BOB},
            Dict[Color, Address],
            {"green": ADA_JSON, "red": BOB_JSON},
            id="enum-keys",
        ),
        pytest.param(MappingProxyType({"a": ADA}), MappingProxyType[str, Address], {"a": ADA_JSON}, id="mapping-proxy"),
        pytest.param({"a": ADA}, Mapping[str, Address], {"a": ADA_JSON}, id="mapping"),
        pytest.param({"work": BOB, "home": ADA}, Dict[str, Address], {"work": BOB_JSON, "home": ADA_JSON}, id="dict"),
        pytest.param((), Tuple[Address, ...], [], id="variadic-tuple-empty"),
        pytest.param((ADA, BOB), Tuple[Address, ...], [ADA_JSON, BOB_JSON], id="variadic-tuple"),
        pytest.param((ADA,), Tuple[Address], [ADA_JSON], id="fixed-tuple-of-one"),
        pytest.param(
            (ADA, Single(1)), Tuple[Address, Single], [ADA_JSON, {"value": 1}], id="fixed-heterogeneous-tuple"
        ),
        pytest.param({"a": [ADA], "b": []}, Dict[str, List[Address]], {"a": [ADA_JSON], "b": []}, id="dict-of-lists"),
        pytest.param([{"x": ADA}, {}], List[Dict[str, Address]], [{"x": ADA_JSON}, {}], id="list-of-dicts"),
        pytest.param(
            [[ADA], [], [BOB, ADA]], List[List[Address]], [[ADA_JSON], [], [BOB_JSON, ADA_JSON]], id="list-of-lists"
        ),
        pytest.param(
            {"x": {"y": (ADA, BOB)}},
            Dict[str, Dict[str, Tuple[Address, ...]]],
            {"x": {"y": [ADA_JSON, BOB_JSON]}},
            id="three-containers-deep",
        ),
        pytest.param([ADA, None, BOB], List[Optional[Address]], [ADA_JSON, None, BOB_JSON], id="list-with-nulls"),
        pytest.param(
            {"a": None, "b": ADA}, Dict[str, Optional[Address]], {"a": None, "b": ADA_JSON}, id="dict-with-nulls"
        ),
        pytest.param(None, Optional[List[Address]], None, id="optional-list-null"),
        pytest.param([ADA], Optional[List[Address]], [ADA_JSON], id="optional-list-present"),
        pytest.param(ADA, Optional[Address], ADA_JSON, id="optional-present"),
        pytest.param(None, Optional[Address], None, id="optional-null"),
        pytest.param(ADA, Union[Address, int], ADA_JSON, id="union-dataclass-or-int"),
        pytest.param(5, Union[Address, int], 5, id="union-int-after-dataclass"),
        pytest.param(ADA, Address | str, ADA_JSON, id="pipe-union-dataclass"),
        pytest.param("text", Address | str, "text", id="pipe-union-string-after-dataclass"),
        pytest.param(None, Address | None, None, id="pipe-optional-null"),
        pytest.param([ADA], Union[Address, List[Address]], [ADA_JSON], id="union-dataclass-or-list-with-list"),
        pytest.param(ADA, Union[Address, List[Address]], ADA_JSON, id="union-dataclass-or-list-with-dataclass"),
        pytest.param((ADA, BOB), Optional[Tuple[Address, ...]], [ADA_JSON, BOB_JSON], id="optional-tuple-present"),
        pytest.param(None, Optional[Tuple[Address, ...]], None, id="optional-tuple-null"),
        pytest.param(
            {"a": [None, ADA], "b": None, "c": []},
            Dict[str, Optional[List[Optional[Address]]]],
            {"a": [None, ADA_JSON], "b": None, "c": []},
            id="optional-at-every-level",
        ),
    ],
)
def test_dataclasses_inside_containers(value, cls, expected):
    assert_text(value, expected)
    assert_round_trip(value, cls)


@pytest.mark.parametrize(
    ("value", "cls"),
    [
        pytest.param(["a", "b"], Union[List[Address], List[str]], id="list-of-dataclasses-or-strings-with-strings"),
        pytest.param([ADA, BOB], Union[List[Address], List[str]], id="list-of-dataclasses-or-strings-with-objects"),
        pytest.param({"a": ADA}, Union[List[Address], Dict[str, Address]], id="list-or-dict-with-dict"),
        pytest.param([ADA], Union[Dict[str, List[Address]], List[Address]], id="dict-of-lists-or-list-with-list"),
        pytest.param(
            {"k": [ADA]}, Union[Dict[str, List[Address]], List[Address]], id="dict-of-lists-or-list-with-dict"
        ),
        pytest.param([["a"], ["b", "c"]], Union[List[List[Address]], List[List[str]]], id="second-level-failure"),
        pytest.param(
            (ADA, BOB, ADA),
            Union[Tuple[Address, Address], Tuple[Address, Address, Address]],
            id="tuple-of-three-takes-the-second-member",
        ),
        pytest.param(
            (ADA, BOB),
            Union[Tuple[Address, Address], Tuple[Address, Address, Address]],
            id="tuple-of-two-takes-the-first-member",
        ),
    ],
)
def test_union_of_containers_moves_on_after_a_failure_deep_inside_a_member(value, cls):
    # the first member gets some way into the data (list items, dict values, a tuple's length) before it fails
    assert_round_trip(value, cls)


@pytest.mark.parametrize(
    ("data", "cls"),
    [
        pytest.param([{"city": "x"}, 5], Union[List[Address], Tuple[Address, ...]], id="item-fails-every-member"),
        pytest.param(
            [[{"city": "x"}], ["oops"]],
            Union[List[List[Address]], List[List[Single]]],
            id="second-level-fails-every-member",
        ),
        pytest.param(
            [{"city": "x"}] * 3,
            Union[Tuple[Address, Address], Tuple[Address, Single]],
            id="wrong-length-for-every-member",
        ),
    ],
)
def test_union_of_containers_rejects_data_that_fails_every_member(data, cls):
    with pytest.raises(TypeError, match="Unsupported types for deserialization"):
        deserialize_json(data, cls)


@pytest.mark.parametrize("length", [0, 1, 3])
def test_fixed_tuple_of_dataclasses_rejects_the_wrong_length(length):
    with pytest.raises(TypeError, match="of length 2"):
        deserialize_json([{"city": "x"}] * length, Tuple[Address, Single])


@dataclass
class BareContainers:
    mapping: dict
    items: list
    pair: tuple
    unique: set
    frozen: frozenset


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            BareContainers({"a": [1, {"b": None}]}, [1, [2, "x"]], (1, "two", 3.0), {3, 1, 2}, frozenset({"y", "x"})),
            {
                "mapping": {"a": [1, {"b": None}]},
                "items": [1, [2, "x"]],
                "pair": [1, "two", 3.0],
                "unique": [1, 2, 3],
                "frozen": ["x", "y"],
            },
            id="populated",
        ),
        pytest.param(
            BareContainers({}, [], (), set(), frozenset()),
            {"mapping": {}, "items": [], "pair": [], "unique": [], "frozen": []},
            id="empty",
        ),
    ],
)
def test_fields_annotated_with_unparameterized_containers(value, expected):
    # no type argument: the items are the JSON values as they are, only the container itself is restored
    assert_text(value, expected)
    assert_round_trip(value, BareContainers)


def test_many_dataclasses_in_a_list_and_a_dict():
    addresses = [Address(f"city-{index}", None if index % 2 else f"{index:05d}") for index in range(1000)]
    expected = [{"city": f"city-{index}", "zip_code": None if index % 2 else f"{index:05d}"} for index in range(1000)]

    assert_text(addresses, expected)
    restored = assert_round_trip(addresses, List[Address])
    assert len(restored) == 1000 and restored[999].zip_code is None and restored[998].zip_code == "00998"

    by_city = {f"k{index}": address for index, address in enumerate(addresses[:300])}
    assert_text(by_city, {f"k{index}": expected[index] for index in range(300)})
    assert list(assert_round_trip(by_city, Dict[str, Address])) == [f"k{index}" for index in range(300)]


@dataclass(frozen=True)
class Point:
    x: int
    y: int


POINTS = [Point(10, 0), Point(2, 0), Point(3, 1), Point(1, 1)]
# ordered by the compact text of each item: {"x":1,"y":1} < {"x":10,"y":0} < {"x":2,"y":0} < {"x":3,"y":1}
CANONICAL_POINTS_JSON = [{"x": 1, "y": 1}, {"x": 10, "y": 0}, {"x": 2, "y": 0}, {"x": 3, "y": 1}]


@pytest.mark.parametrize(
    ("make", "cls"),
    [
        pytest.param(set, Set[Point], id="set"),
        pytest.param(frozenset, FrozenSet[Point], id="frozenset"),
    ],
)
def test_set_of_frozen_dataclasses_is_ordered_by_canonical_json_text(make, cls):
    value = make(POINTS)

    assert_json(value, CANONICAL_POINTS_JSON)
    assert {serialize_json(make(order)) for order in itertools.permutations(POINTS)} == {serialize_json(value)}
    assert_round_trip(value, cls)


@pytest.mark.parametrize(
    ("value", "cls", "expected"),
    [
        pytest.param(set(), Set[Point], [], id="empty-set"),
        pytest.param(frozenset(), FrozenSet[Point], [], id="empty-frozenset"),
        pytest.param({Point(1, 2)}, Set[Point], [{"x": 1, "y": 2}], id="singleton-set"),
        pytest.param(frozenset({Point(0, 0)}), FrozenSet[Point], [{"x": 0, "y": 0}], id="singleton-frozenset"),
        pytest.param(
            [frozenset({Point(1, 2)}), frozenset()], List[FrozenSet[Point]], [[{"x": 1, "y": 2}], []], id="list-of-sets"
        ),
        pytest.param({"a": {Point(3, 4)}}, Dict[str, Set[Point]], {"a": [{"x": 3, "y": 4}]}, id="dict-of-sets"),
        pytest.param(
            (Point(5, 6), frozenset({Point(1, 1)})),
            Tuple[Point, FrozenSet[Point]],
            [{"x": 5, "y": 6}, [{"x": 1, "y": 1}]],
            id="tuple-with-a-set",
        ),
    ],
)
def test_sets_of_dataclasses_inside_other_containers(value, cls, expected):
    assert_json(value, expected)
    assert_round_trip(value, cls)


@dataclass(frozen=True)
class Flipped:
    y: int
    x: int


@dataclass(frozen=True)
class Outer:
    z: int
    inner: Flipped


def test_dataclass_keys_inside_a_set_are_sorted_limitation():
    # Documents current behavior, not a promise: a set item is written from its canonical (sorted keys) text, so the
    # fields of a dataclass inside a set are not in declaration order. The same dataclass in a list is.
    assert serialize_json([Flipped(1, 2)]) == '[{"y": 1, "x": 2}]'
    assert serialize_json({Flipped(1, 2)}) == '[{"x": 2, "y": 1}]'
    assert serialize_json(frozenset({Outer(1, Flipped(1, 2))})) == '[{"inner": {"x": 2, "y": 1}, "z": 1}]'
    assert_round_trip({Flipped(1, 2)}, Set[Flipped])


@dataclass(frozen=True)
class Booking:
    kind: Color
    day: datetime.date
    cost: Decimal


@dataclass(unsafe_hash=True)
class Label:
    name: str
    weight: int = 0


@dataclass(frozen=True)
class FrozenTree:
    label: str
    children: FrozenSet["FrozenTree"] = frozenset()


def test_set_of_dataclasses_with_enum_date_and_decimal_fields_is_ordered_by_the_text_of_each_item():
    bookings = {
        Booking(Color.RED, datetime.date(2024, 1, 2), Decimal("1.50")),
        Booking(Color.GREEN, datetime.date(2023, 5, 6), Decimal("20")),
        Booking(Color.RED, datetime.date(2022, 12, 31), Decimal("100")),
    }

    # the compact texts begin {"cost":"1.50",  {"cost":"100",  {"cost":"20",: "." sorts before "0", and "1" before "2".
    # Neither the cost, the date nor the color decides the order.
    assert_json(
        bookings,
        [
            {"kind": "red", "day": "2024-01-02", "cost": "1.50"},
            {"kind": "red", "day": "2022-12-31", "cost": "100"},
            {"kind": "green", "day": "2023-05-06", "cost": "20"},
        ],
    )
    assert_round_trip(bookings, Set[Booking])
    assert_round_trip(frozenset(bookings), FrozenSet[Booking])


def test_set_of_dataclasses_with_a_generated_hash():
    labels = {Label("b", 1), Label("a", 2), Label("a")}

    assert_text(labels, [{"name": "a", "weight": 0}, {"name": "a", "weight": 2}, {"name": "b", "weight": 1}])
    assert_round_trip(labels, Set[Label])


def test_set_of_tuples_of_dataclasses():
    value = frozenset({(Point(1, 2),), (Point(0, 0), Point(5, 5)), ()})

    # the compact texts begin  []  and  [{"x":0  and  [{"x":1
    assert_text(value, [[], [{"x": 0, "y": 0}, {"x": 5, "y": 5}], [{"x": 1, "y": 2}]])
    assert_round_trip(value, FrozenSet[Tuple[Point, ...]])


def test_recursive_dataclass_holding_a_frozenset_of_itself():
    tree = FrozenTree("root", frozenset({FrozenTree("b"), FrozenTree("a", frozenset({FrozenTree("z")}))}))

    # the leaf "b" (children []) is written before "a" (children [{...): "]" sorts before "{" in the compact text
    assert_json(
        tree,
        {
            "label": "root",
            "children": [
                {"label": "b", "children": []},
                {"label": "a", "children": [{"label": "z", "children": []}]},
            ],
        },
    )
    restored = assert_round_trip(tree, FrozenTree)
    assert {child.label for child in restored.children} == {"a", "b"}
    assert_round_trip(FrozenTree("alone"), FrozenTree)


def test_set_of_mutable_dataclasses_cannot_be_restored_but_a_union_moves_on_to_a_list():
    # Address is eq=True and not frozen, so it is unhashable: no set of it exists, and set(...) says so.
    data = [{"city": "Rome", "zip_code": None}]

    with pytest.raises(TypeError, match="unhashable"):
        deserialize_json(data, Set[Address])
    assert_strictly_equal(deserialize_json(data, Union[Set[Address], List[Address]]), [Address("Rome")])


@dataclass(frozen=True)
class Skill:
    name: str
    level: int


@dataclass
class Employee:
    name: str
    skills: FrozenSet[Skill]
    home: Optional[Address] = None
    manager: Optional[str] = None


@dataclass
class Team:
    name: str
    lead: Employee
    members: Tuple[Employee, ...]


@dataclass
class Department:
    title: str
    teams: List[Team]
    budget: Decimal


@dataclass
class Company:
    name: str
    founded: datetime.date
    departments: Dict[str, Department]


def make_company():
    ada = Employee("Ada", frozenset({Skill("python", 5), Skill("go", 3)}), Address("London", "N1"))
    bob = Employee("Bob", frozenset({Skill("sql", 4), Skill("excel", 4)}), None, "Ada")
    cyd = Employee("Cyd", frozenset(), Address("Paris"), "Ada")
    dee = Employee("Dee", frozenset({Skill("ops", 2)}), None, "Cyd")
    return Company(
        "Acme",
        datetime.date(1999, 12, 31),
        {
            "eng": Department(
                "Engineering",
                [Team("core", ada, (bob, cyd)), Team("platform", bob, ())],
                Decimal("1250000.50"),
            ),
            "ops": Department("Operations", [Team("support", dee, (cyd, dee))], Decimal("90000")),
        },
    )


ADA_EMPLOYEE_JSON = {
    "name": "Ada",
    # skills of a set are ordered by their compact JSON text, {"level":3,...} before {"level":5,...}
    "skills": [{"name": "go", "level": 3}, {"name": "python", "level": 5}],
    "home": {"city": "London", "zip_code": "N1"},
    "manager": None,
}
BOB_EMPLOYEE_JSON = {
    "name": "Bob",
    "skills": [{"name": "excel", "level": 4}, {"name": "sql", "level": 4}],
    "home": None,
    "manager": "Ada",
}
CYD_EMPLOYEE_JSON = {"name": "Cyd", "skills": [], "home": {"city": "Paris", "zip_code": None}, "manager": "Ada"}
DEE_EMPLOYEE_JSON = {"name": "Dee", "skills": [{"name": "ops", "level": 2}], "home": None, "manager": "Cyd"}
COMPANY_JSON = {
    "name": "Acme",
    "founded": "1999-12-31",
    "departments": {
        "eng": {
            "title": "Engineering",
            "teams": [
                {"name": "core", "lead": ADA_EMPLOYEE_JSON, "members": [BOB_EMPLOYEE_JSON, CYD_EMPLOYEE_JSON]},
                {"name": "platform", "lead": BOB_EMPLOYEE_JSON, "members": []},
            ],
            "budget": "1250000.50",
        },
        "ops": {
            "title": "Operations",
            "teams": [
                {"name": "support", "lead": DEE_EMPLOYEE_JSON, "members": [CYD_EMPLOYEE_JSON, DEE_EMPLOYEE_JSON]}
            ],
            "budget": "90000",
        },
    },
}


def test_company_with_dataclasses_five_containers_deep():
    company = make_company()

    assert_json(company, COMPANY_JSON)
    restored = assert_round_trip(company, Company)
    assert type(restored.departments["eng"].teams[0].lead.skills) is frozenset
    assert restored.departments["eng"].budget.as_tuple() == Decimal("1250000.50").as_tuple()


def test_company_inside_a_list_and_an_optional():
    company = make_company()

    assert_round_trip([company, company], List[Company])
    assert_round_trip(company, Optional[Company])
    assert_round_trip({"main": company}, Dict[str, Company])


def test_a_dataclass_shared_by_several_fields_is_written_in_full_each_time():
    shared = Address("Rome")
    team = Team("t", Employee("L", frozenset(), shared), (Employee("M", frozenset(), shared),))

    expected_home = {"city": "Rome", "zip_code": None}
    assert to_json_value(team)["lead"]["home"] == expected_home
    assert to_json_value(team)["members"][0]["home"] == expected_home
    restored = assert_round_trip(team, Team)
    assert restored.lead.home is not restored.members[0].home


# --- spellings of the same annotation ----------------------------------------------------------------------------


@dataclass
class TypingSpelling:
    items: List[int]
    mapping: Dict[str, List[int]]
    pair: Tuple[int, str]
    many: Tuple[float, ...]
    unique: Set[str]
    frozen: FrozenSet[int]
    maybe: Optional[int]
    maybe_list: Optional[List[int]]
    either: Union[Address, int]


@dataclass
class BuiltinSpelling:
    items: list[int]
    mapping: dict[str, list[int]]
    pair: tuple[int, str]
    many: tuple[float, ...]
    unique: set[str]
    frozen: frozenset[int]
    maybe: int | None
    maybe_list: list[int] | None
    either: Address | int


@pytest.mark.parametrize(
    "cls",
    [pytest.param(TypingSpelling, id="typing"), pytest.param(BuiltinSpelling, id="builtin")],
)
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        pytest.param(
            ([1, 2], {"k": [3]}, (4, "x"), (1.5, 2.0), {"b", "a"}, frozenset({3, 1}), None, [5], ADA),
            {
                "items": [1, 2],
                "mapping": {"k": [3]},
                "pair": [4, "x"],
                "many": [1.5, 2.0],
                "unique": ["a", "b"],
                "frozen": [1, 3],
                "maybe": None,
                "maybe_list": [5],
                "either": ADA_JSON,
            },
            id="populated",
        ),
        pytest.param(
            ([], {}, (0, ""), (), set(), frozenset(), 0, None, 7),
            {
                "items": [],
                "mapping": {},
                "pair": [0, ""],
                "many": [],
                "unique": [],
                "frozen": [],
                "maybe": 0,
                "maybe_list": None,
                "either": 7,
            },
            id="empty",
        ),
    ],
)
def test_typing_and_builtin_spellings_of_annotations_behave_the_same(cls, arguments, expected):
    value = cls(*arguments)

    assert_text(value, expected)
    assert_round_trip(value, cls)


# --- dataclass features ------------------------------------------------------------------------------------------


@dataclass(slots=True)
class SlottedPoint:
    x: int
    y: int = 0


@dataclass(slots=True)
class SlottedChild(SlottedPoint):
    z: int = 0


@dataclass(kw_only=True)
class KeywordOnly:
    name: str
    size: int = 1


@dataclass
class KeywordMarker:
    name: str
    _: KW_ONLY
    size: int = 1
    flag: bool = False


@dataclass(order=True)
class Ordered:
    rank: int
    label: str


@dataclass(eq=False)
class IdentityEquality:
    name: str
    items: List[int] = field(default_factory=list)


@dataclass(frozen=True, slots=True, order=True, kw_only=True)
class AllOptions:
    key: str
    weight: float = 1.0


@dataclass
class HiddenFields:
    visible: int
    secret: str = field(default="s3", repr=False)
    cache: List[int] = field(default_factory=list, compare=False, repr=False)


@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(Point(1, 2), '{"x": 1, "y": 2}', id="frozen"),
        pytest.param(SlottedPoint(1), '{"x": 1, "y": 0}', id="slots"),
        pytest.param(SlottedChild(1, 2, 3), '{"x": 1, "y": 2, "z": 3}', id="slots-inheritance"),
        pytest.param(KeywordOnly(name="n"), '{"name": "n", "size": 1}', id="kw-only"),
        pytest.param(KeywordMarker("n", flag=True), '{"name": "n", "size": 1, "flag": true}', id="kw-only-marker"),
        pytest.param(Ordered(1, "a"), '{"rank": 1, "label": "a"}', id="order"),
        pytest.param(IdentityEquality("n", [1]), '{"name": "n", "items": [1]}', id="eq-false"),
        pytest.param(AllOptions(key="k"), '{"key": "k", "weight": 1.0}', id="frozen-slots-order-kw-only"),
        pytest.param(
            HiddenFields(1, "classified", [4, 5]),
            '{"visible": 1, "secret": "classified", "cache": [4, 5]}',
            id="repr-false-and-compare-false-fields-are-written",
        ),
    ],
)
def test_dataclass_options_do_not_change_the_json(value, text):
    assert serialize_json(value) == text

    restored = assert_round_trip(value, type(value))

    assert restored is not value


def test_restored_frozen_dataclass_is_frozen_and_hashable():
    restored = deserialize_json({"x": 1, "y": 2}, Point)

    with pytest.raises(dataclasses.FrozenInstanceError):
        restored.x = 5
    assert hash(restored) == hash(Point(1, 2))


def test_restored_slotted_dataclass_has_no_instance_dict():
    restored = deserialize_json({"x": 1, "y": 2, "z": 3}, SlottedChild)

    assert not hasattr(restored, "__dict__")
    assert_strictly_equal(restored, SlottedChild(1, 2, 3))


def test_restored_ordered_dataclasses_can_be_sorted():
    values = [Ordered(3, "c"), Ordered(1, "z"), Ordered(1, "a")]

    restored = assert_round_trip(values, List[Ordered])

    assert [(item.rank, item.label) for item in sorted(restored)] == [(1, "a"), (1, "z"), (3, "c")]


def test_restored_dataclass_without_eq_is_a_new_object():
    original = IdentityEquality("n", [1, 2])

    restored = assert_round_trip(original, IdentityEquality)

    assert restored != original and restored.items is not original.items


@dataclass
class Rectangle:
    width: int
    height: int
    area: int = field(init=False)

    def __post_init__(self):
        self.area = self.width * self.height


@dataclass
class Versioned:
    name: str
    version: int = field(init=False, default=1)


@dataclass(frozen=True)
class FrozenRectangle:
    width: int
    height: int
    area: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "area", self.width * self.height)


@dataclass(slots=True)
class SlottedRectangle:
    width: int
    height: int
    area: int = field(init=False)

    def __post_init__(self):
        self.area = self.width * self.height


@dataclass(frozen=True, slots=True)
class FrozenSlottedVersioned:
    name: str
    version: int = field(init=False, default=1)


@dataclass
class Summary:
    total: int
    cities: List[str]


@dataclass
class Report:
    title: str
    entries: List[Address] = field(default_factory=list)
    summary: Optional[Summary] = field(init=False, default=None)
    history: List[Address] = field(init=False, default_factory=list)
    lookup: Dict[str, Address] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.summary = Summary(len(self.entries), sorted(entry.city for entry in self.entries))


@dataclass
class Plot:
    boxes: List[Rectangle]
    frozen: Dict[str, FrozenRectangle]
    slotted: Optional[SlottedRectangle] = None


@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(Rectangle(3, 4), '{"width": 3, "height": 4, "area": 12}', id="computed-in-post-init"),
        pytest.param(Versioned("v"), '{"name": "v", "version": 1}', id="init-false-with-default"),
        pytest.param(FrozenRectangle(3, 4), '{"width": 3, "height": 4, "area": 12}', id="frozen"),
        pytest.param(SlottedRectangle(3, 4), '{"width": 3, "height": 4, "area": 12}', id="slots"),
        pytest.param(FrozenSlottedVersioned("v"), '{"name": "v", "version": 1}', id="frozen-and-slots"),
    ],
)
def test_init_false_field_is_written(value, text):
    assert serialize_json(value) == text


@pytest.mark.parametrize(
    ("value", "cls"),
    [
        pytest.param(Rectangle(3, 4), Rectangle, id="computed-in-post-init"),
        pytest.param(Versioned("v"), Versioned, id="init-false-with-default"),
        pytest.param(FrozenRectangle(3, 4), FrozenRectangle, id="frozen"),
        pytest.param(SlottedRectangle(3, 4), SlottedRectangle, id="slots"),
        pytest.param(FrozenSlottedVersioned("v"), FrozenSlottedVersioned, id="frozen-and-slots"),
        pytest.param([Rectangle(1, 2), Rectangle(3, 4)], List[Rectangle], id="list"),
        pytest.param({"a": FrozenRectangle(1, 2)}, Dict[str, FrozenRectangle], id="dict-of-frozen"),
        pytest.param(SlottedRectangle(5, 6), Optional[SlottedRectangle], id="optional-slots"),
    ],
)
def test_init_false_field_round_trip(value, cls):
    assert_round_trip(value, cls)


def test_init_false_dataclasses_nested_in_containers_round_trip():
    plot = Plot([Rectangle(1, 2), Rectangle(2, 3)], {"a": FrozenRectangle(3, 4)}, SlottedRectangle(4, 5))

    restored = assert_round_trip(plot, Plot)

    assert [box.area for box in restored.boxes] == [2, 6]
    assert restored.frozen["a"].area == 12 and restored.slotted.area == 20


@pytest.mark.parametrize(
    ("make", "name", "changed"),
    [
        pytest.param(lambda: Rectangle(3, 4), "area", 99, id="computed-in-post-init"),
        pytest.param(lambda: Versioned("v"), "version", 7, id="init-false-with-default"),
        pytest.param(lambda: FrozenRectangle(3, 4), "area", 99, id="frozen"),
        pytest.param(lambda: SlottedRectangle(3, 4), "area", 99, id="slots"),
        pytest.param(lambda: FrozenSlottedVersioned("v"), "version", 7, id="frozen-and-slots"),
    ],
)
def test_init_false_field_changed_after_construction_is_restored_as_changed(make, name, changed):
    value = make()
    assert getattr(value, name) != changed
    object.__setattr__(value, name, changed)  # also for the frozen ones

    restored = assert_round_trip(value, type(value))

    assert getattr(restored, name) == changed  # the value that was written, not the one the constructor makes


def test_init_false_field_holding_nested_values_round_trips():
    report = Report("q", [Address("Oslo"), Address("Bern", "3000")])
    report.history.append(Address("Rome", "00100"))
    report.lookup["home"] = Address("Kyiv")

    assert_text(
        report,
        {
            "title": "q",
            "entries": [{"city": "Oslo", "zip_code": None}, {"city": "Bern", "zip_code": "3000"}],
            "summary": {"total": 2, "cities": ["Bern", "Oslo"]},
            "history": [{"city": "Rome", "zip_code": "00100"}],
            "lookup": {"home": {"city": "Kyiv", "zip_code": None}},
        },
    )
    restored = assert_round_trip(report, Report)

    assert type(restored.summary) is Summary and restored.summary.cities == ["Bern", "Oslo"]
    assert type(restored.history[0]) is Address and type(restored.lookup["home"]) is Address


def test_init_false_field_replaced_after_construction_is_restored_as_replaced():
    report = Report("q", [Address("Oslo")])
    report.summary = Summary(100, ["elsewhere"])  # not what __post_init__ computes
    report.history = [Address("Rome")]

    restored = assert_round_trip(report, Report)

    assert restored.summary == Summary(100, ["elsewhere"])
    assert restored.history == [Address("Rome")]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        pytest.param({"width": 3, "height": 4}, Rectangle(3, 4), id="computed-in-post-init"),
        pytest.param({"name": "v"}, Versioned("v"), id="init-false-with-default"),
        pytest.param({"width": 3, "height": 4}, FrozenRectangle(3, 4), id="frozen"),
        pytest.param({"width": 3, "height": 4}, SlottedRectangle(3, 4), id="slots"),
        pytest.param({"name": "v"}, FrozenSlottedVersioned("v"), id="frozen-and-slots"),
        pytest.param({"title": "q", "entries": [{"city": "Oslo"}]}, Report("q", [Address("Oslo")]), id="nested-values"),
    ],
)
def test_init_false_field_missing_from_the_json_keeps_what_the_constructor_made(data, expected):
    restored = deserialize_json(data, type(expected))

    assert_strictly_equal(restored, expected)


def test_init_false_field_in_the_json_wins_over_what_the_constructor_made():
    restored = deserialize_json({"width": 3, "height": 4, "area": 100}, Rectangle)

    expected = Rectangle(3, 4)
    expected.area = 100  # not what __post_init__ computes
    assert_strictly_equal(restored, expected)


@dataclass
class Offset:
    base: int
    delta: InitVar[int] = 0
    total: int = 0

    def __post_init__(self, delta):
        self.total = self.base + delta


@dataclass
class Scaled:
    base: int
    scale: InitVar[int]
    total: int = 0

    def __post_init__(self, scale):
        self.total = self.base * scale


def test_initvar_is_not_a_field_and_is_not_written():
    assert serialize_json(Offset(1, 10)) == '{"base": 1, "total": 11}'
    assert serialize_json(Scaled(2, 3)) == '{"base": 2, "total": 6}'


def test_initvar_left_at_its_default_round_trips():
    assert_round_trip(Offset(5), Offset)


def test_initvar_value_is_lost_and_post_init_runs_with_the_default_limitation():
    # Documents current behavior, not a promise: an InitVar is neither written nor restored, so post-init recomputes.
    restored = deserialize_json(json.loads(serialize_json(Offset(1, 10))), Offset)

    assert restored.total == 1


def test_initvar_without_a_default_cannot_be_restored_limitation():
    # Documents current behavior, not a promise: the constructor is called with the fields only.
    with pytest.raises(TypeError, match="scale"):
        deserialize_json({"base": 2, "total": 6}, Scaled)


@dataclass
class Registry:
    name: str
    kind: ClassVar[str] = "registry"
    limits: ClassVar[Dict[str, int]] = {"max": 3}


def test_classvar_is_neither_written_nor_restored():
    assert serialize_json(Registry("r")) == '{"name": "r"}'

    restored = deserialize_json({"name": "r", "kind": "forged", "limits": {"max": 99}}, Registry)

    assert_strictly_equal(restored, Registry("r"))
    assert Registry.kind == "registry" and Registry.limits == {"max": 3}


@dataclass
class Normalized:
    name: str
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.name = self.name.strip().lower()
        self.tags = sorted(set(self.tags))


@dataclass
class Positive:
    amount: int

    def __post_init__(self):
        if self.amount is None or self.amount <= 0:
            raise ValueError("amount must be positive")


def test_post_init_normalization_applies_when_restoring():
    normalized = Normalized("  Ada ", ["b", "a", "b"])

    assert serialize_json(normalized) == '{"name": "ada", "tags": ["a", "b"]}'
    assert_round_trip(normalized, Normalized)
    restored = deserialize_json({"name": " ADA ", "tags": ["z", "y", "z"]}, Normalized)
    assert_strictly_equal(restored, Normalized("ada", ["y", "z"]))


@pytest.mark.parametrize(
    "cls",
    [pytest.param(Positive, id="plain"), pytest.param(Optional[Positive], id="inside-a-union")],
)
def test_post_init_validation_error_reaches_the_caller_unchanged(cls):
    assert_round_trip(Positive(2), cls)
    with pytest.raises(ValueError, match="must be positive"):
        deserialize_json({"amount": -1}, cls)


@dataclass
class Temperature:
    celsius: float

    @property
    def fahrenheit(self) -> float:
        return self.celsius * 9 / 5 + 32

    def describe(self) -> str:
        return f"{self.celsius} C"

    @classmethod
    def freezing(cls) -> "Temperature":
        return cls(0.0)

    @staticmethod
    def unit() -> str:
        return "C"


def test_properties_methods_and_ad_hoc_attributes_are_not_written():
    value = Temperature(100.0)
    value.note = "added after construction"

    assert serialize_json(value) == '{"celsius": 100.0}'
    restored = assert_round_trip(value, Temperature)
    assert restored.fahrenheit == 212.0
    assert not hasattr(restored, "note")
    assert_round_trip(Temperature.freezing(), Temperature)


@dataclass
class WithMetadata:
    user_name: str = field(metadata={"json_name": "userName"})


def test_field_metadata_is_not_written_and_does_not_rename_a_key_limitation():
    # Documents current behavior, not a promise: metadata is ignored, the Python field name is the JSON key, and the
    # key the metadata names is not read (so the field is missing, which restores as None).
    assert serialize_json(WithMetadata("ada")) == '{"user_name": "ada"}'
    assert_round_trip(WithMetadata("ada"), WithMetadata)
    assert_strictly_equal(deserialize_json({"userName": "ada"}, WithMetadata), WithMetadata(None))


@dataclass
class Declared:
    items: List[int]
    pair: Tuple[int, int]
    tags: Set[str]


def test_values_are_written_by_their_own_type_and_restored_by_the_annotation_limitation():
    # Documents current behavior, not a promise: a tuple in a List field and a list in a Set field are valid Python
    # that dataclasses never check. They are written as the arrays they are (the list keeps its order, nothing is
    # sorted), and come back as what the annotation says.
    value = Declared((1, 2), [3, 4], ["b", "a"])

    assert serialize_json(value) == '{"items": [1, 2], "pair": [3, 4], "tags": ["b", "a"]}'
    restored = deserialize_json(to_json_value(value), Declared)
    assert_strictly_equal(restored, Declared([1, 2], (3, 4), {"a", "b"}))


@dataclass
class Animal:
    name: str
    legs: int = 4


@dataclass
class Dog(Animal):
    breed: str = "mutt"
    good: bool = True


@dataclass
class Puppy(Dog):
    age_weeks: int = 8
    name: str = "pup"


@dataclass
class Kennel:
    resident: Animal
    guests: List[Animal]


@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(Animal("Rex"), '{"name": "Rex", "legs": 4}', id="base"),
        pytest.param(
            Dog("Rex", 4, "lab", True), '{"name": "Rex", "legs": 4, "breed": "lab", "good": true}', id="child"
        ),
        pytest.param(
            Puppy(),
            '{"name": "pup", "legs": 4, "breed": "mutt", "good": true, "age_weeks": 8}',
            id="grandchild-overriding-a-base-field-keeps-its-position",
        ),
    ],
)
def test_inherited_fields_come_first_and_each_class_round_trips(value, text):
    assert serialize_json(value) == text
    assert_round_trip(value, type(value))


def test_subclass_instances_are_written_with_their_own_fields_whatever_the_annotation():
    kennel = Kennel(Dog("Rex", 3, "lab"), [Animal("Cat", 4), Puppy("Pip"), Dog("Fido")])

    assert_text(
        kennel,
        {
            "resident": {"name": "Rex", "legs": 3, "breed": "lab", "good": True},
            "guests": [
                {"name": "Cat", "legs": 4},
                {"name": "Pip", "legs": 4, "breed": "mutt", "good": True, "age_weeks": 8},
                {"name": "Fido", "legs": 4, "breed": "mutt", "good": True},
            ],
        },
    )


def test_base_class_annotation_restores_only_the_base_fields_limitation():
    # Documents current behavior, not a promise: the annotation decides the class, the subclass fields are dropped.
    kennel = Kennel(Dog("Rex", 3, "lab"), [Animal("Cat", 4), Puppy("Pip"), Dog("Fido")])

    restored = deserialize_json(to_json_value(kennel), Kennel)

    assert_strictly_equal(restored, Kennel(Animal("Rex", 3), [Animal("Cat", 4), Animal("Pip", 4), Animal("Fido", 4)]))


def test_subclass_annotation_restores_a_subclass_and_drops_the_fields_of_a_grandchild_limitation():
    # Documents current behavior, not a promise: Dog does not know age_weeks, so that key is ignored.
    json_value = to_json_value(Puppy("Pip"))

    assert_strictly_equal(deserialize_json(json_value, Puppy), Puppy("Pip"))
    assert_strictly_equal(deserialize_json(json_value, Dog), Dog("Pip"))


class Greeter:
    greeting = "hello"

    def greet(self):
        return f"{self.greeting}, {self.name}"


@dataclass
class Named:
    name: str = ""


@dataclass
class Aged:
    age: int = 0


@dataclass
class Member(Greeter, Named, Aged):
    email: Optional[str] = None


@dataclass
class DiamondBase:
    a: int = 0


@dataclass
class DiamondLeft(DiamondBase):
    left: int = 0


@dataclass
class DiamondRight(DiamondBase):
    right: int = 0


@dataclass
class DiamondBottom(DiamondLeft, DiamondRight):
    bottom: int = 0


def test_multiple_inheritance_with_a_mixin_collects_fields_from_the_last_base_first():
    member = Member(age=30, name="Ada", email="ada@example.com")

    # fields are gathered from the bases in reverse method resolution order: Aged, then Named, then Member's own
    assert serialize_json(member) == '{"age": 30, "name": "Ada", "email": "ada@example.com"}'
    restored = assert_round_trip(member, Member)
    assert restored.greet() == "hello, Ada"


def test_diamond_inheritance_lists_the_shared_base_field_once():
    value = DiamondBottom(a=1, right=2, left=3, bottom=4)

    assert serialize_json(value) == '{"a": 1, "right": 2, "left": 3, "bottom": 4}'
    assert_round_trip(value, DiamondBottom)


T = TypeVar("T")


@dataclass
class Box(Generic[T]):
    item: T
    label: str = ""


@dataclass
class Crate:
    day_box: Box[datetime.date]
    names: List[Box[str]]


@dataclass
class Shelf:
    count_box: Box[int]
    names: List[Box[str]]
    maybe: Optional[Box[int]] = None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(Box(1), {"item": 1, "label": ""}, id="int"),
        pytest.param(Box[int](5, "five"), {"item": 5, "label": "five"}, id="subscripted-constructor"),
        pytest.param(Box(None), {"item": None, "label": ""}, id="none"),
        pytest.param(Box(Decimal("1.50")), {"item": "1.50", "label": ""}, id="decimal"),
        pytest.param(
            Box([Box(Decimal("1.50")), Box({"k": (1, 2)})]),
            {
                "item": [{"item": "1.50", "label": ""}, {"item": {"k": [1, 2]}, "label": ""}],
                "label": "",
            },
            id="nested-boxes",
        ),
        pytest.param(
            Crate(Box(datetime.date(2024, 2, 29)), [Box("a"), Box("b", "second")]),
            {
                "day_box": {"item": "2024-02-29", "label": ""},
                "names": [{"item": "a", "label": ""}, {"item": "b", "label": "second"}],
            },
            id="parameterized-annotations-in-a-dataclass",
        ),
        pytest.param(
            Shelf(Box(7), [Box("a", "first")], Box(8, "eight")),
            {
                "count_box": {"item": 7, "label": ""},
                "names": [{"item": "a", "label": "first"}],
                "maybe": {"item": 8, "label": "eight"},
            },
            id="optional-parameterized-annotation",
        ),
    ],
)
def test_generic_dataclass_is_written_like_a_plain_one(value, expected):
    assert_text(value, expected)


@pytest.mark.parametrize(
    ("value", "cls"),
    [
        pytest.param(Box(5, "five"), Box[int], id="root"),
        pytest.param([Box("a"), Box("b")], List[Box[str]], id="list"),
        pytest.param(Shelf(Box(7), [Box("a", "first")], Box(8, "eight")), Shelf, id="field"),
        pytest.param(Box(datetime.date(2024, 2, 29)), Box[datetime.date], id="date-item"),
        pytest.param(
            Crate(Box(datetime.date(2024, 2, 29)), [Box("a"), Box("b", "second")]), Crate, id="date-item-in-a-field"
        ),
        pytest.param(Box(Box(Decimal("1.50"))), Box[Box[Decimal]], id="nested-decimal"),
    ],
)
def test_parameterized_generic_dataclass_round_trip(value, cls):
    assert_round_trip(value, cls)


def test_unparameterized_generic_dataclass_keeps_typevar_fields_as_raw_json_limitation():
    # Documents current behavior, not a promise: with no type argument the TypeVar field is the JSON value as it is.
    restored = deserialize_json({"item": "2024-02-29", "label": "x"}, Box)

    assert_strictly_equal(restored, Box("2024-02-29", "x"))
    assert_round_trip(Box(5, "five"), Box)


UserId = NewType("UserId", int)


@dataclass
class Account:
    user_id: UserId
    name: str


@dataclass
class OptionalAccount:
    user_id: Optional[UserId] = None


@dataclass
class Measured:
    samples: Annotated[List[int], "unit: ms"]


@dataclass
class SequenceHolder:
    items: Sequence[int]


@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(Account(UserId(5), "x"), '{"user_id": 5, "name": "x"}', id="new-type"),
        pytest.param(OptionalAccount(UserId(5)), '{"user_id": 5}', id="optional-new-type"),
        pytest.param(OptionalAccount(), '{"user_id": null}', id="optional-new-type-null"),
        pytest.param(Measured([1, 2]), '{"samples": [1, 2]}', id="annotated"),
        pytest.param(SequenceHolder([1, 2]), '{"items": [1, 2]}', id="abstract-sequence"),
    ],
)
def test_wrapped_and_abstract_annotations_are_written_from_the_value(value, text):
    assert serialize_json(value) == text


@pytest.mark.parametrize(
    ("value", "cls"),
    [
        pytest.param(Account(UserId(5), "x"), Account, id="new-type"),
        pytest.param(OptionalAccount(UserId(5)), OptionalAccount, id="optional-new-type"),
        pytest.param(Measured([1, 2]), Measured, id="annotated"),
        pytest.param(OptionalAccount(), OptionalAccount, id="optional-new-type-null"),
    ],
)
def test_new_type_and_annotated_fields_round_trip(value, cls):
    assert_round_trip(value, cls)


def test_abstract_sequence_annotation_cannot_be_restored_limitation():
    # Documents current behavior, not a promise: only list, tuple, set and frozenset (and Mapping) annotations have a
    # handler, so a Sequence[int] field is written but not read back.
    with pytest.raises(TypeError, match="Unsupported type"):
        deserialize_json({"items": [1, 2]}, SequenceHolder)


@dataclass
class International:
    שם: str
    año: int
    名前: Optional[str] = None
    ünïcode_value: float = 0.5


@dataclass
class ShadowsBuiltins:
    id: int
    type: str
    list: List[int]
    dict: Dict[str, int]
    str: Optional[str]
    object: Tuple[int, ...]
    input: Optional[Address]


@dataclass
class InternalNames:
    self: int
    cls: str
    data: List[int]
    kwargs: Dict[str, int]
    args: Optional[str] = None
    fields: Tuple[int, ...] = ()
    _private: int = 0
    field2: int = 0


def test_non_ascii_field_names_are_escaped_by_default_and_kept_with_ensure_ascii_false():
    value = International("Ada", 36)

    assert serialize_json(value) == (
        '{"\\u05e9\\u05dd": "Ada", "a\\u00f1o": 36, "\\u540d\\u524d": null, "\\u00fcn\\u00efcode_value": 0.5}'
    )
    assert serialize_json(value, ensure_ascii=False) == (
        '{"שם": "Ada", "año": 36, "名前": null, "ünïcode_value": 0.5}'
    )
    assert_round_trip(value, International)
    assert_round_trip(value, International, ensure_ascii=False)
    assert_round_trip(International("שלום", 1, "名前", -2.5), International)


def test_field_names_that_shadow_builtins():
    value = ShadowsBuiltins(1, "t", [2], {"a": 3}, None, (4, 5), Address("x"))

    assert_text(
        value,
        {
            "id": 1,
            "type": "t",
            "list": [2],
            "dict": {"a": 3},
            "str": None,
            "object": [4, 5],
            "input": {"city": "x", "zip_code": None},
        },
    )
    assert_round_trip(value, ShadowsBuiltins)


def test_field_names_that_are_parameter_names_of_the_constructor_and_the_handlers():
    value = InternalNames(1, "c", [2], {"a": 3}, "x", (4, 5), 6, 7)

    assert_text(
        value,
        {
            "self": 1,
            "cls": "c",
            "data": [2],
            "kwargs": {"a": 3},
            "args": "x",
            "fields": [4, 5],
            "_private": 6,
            "field2": 7,
        },
    )
    assert_round_trip(value, InternalNames)


def test_dataclass_class_object_is_not_serializable():
    # the class is not an instance: only the encoder's own refusal can come out of this
    with pytest.raises(TypeError, match="not JSON serializable"):
        serialize_json(Address)


def test_dataclass_cannot_be_a_mapping_key():
    with pytest.raises(TypeError, match="Unsupported JSON mapping key type"):
        serialize_json({Point(1, 2): "x"})


def test_tuple_of_dataclasses_as_a_mapping_key_is_written_as_its_json_text():
    assert serialize_json({(Point(1, 2),): "x"}) == '{"[{\\"x\\": 1, \\"y\\": 2}]": "x"}'


# --- recursive structures ----------------------------------------------------------------------------------------


@dataclass
class ListNode:
    value: int
    next: Optional["ListNode"] = None


@dataclass
class Tree:
    label: str
    children: List["Tree"] = field(default_factory=list)


@dataclass
class BinNode:
    value: int
    left: Optional["BinNode"] = None
    right: Optional["BinNode"] = None


@dataclass
class AnnotatedNode:
    value: int
    children: list[Annotated[Optional["AnnotatedNode"], "child"]] = field(default_factory=list)
    next: Annotated[Optional["AnnotatedNode"], "link"] = None


@dataclass
class Alpha:
    n: int
    beta: Optional["Beta"] = None


@dataclass
class Beta:
    alphas: List["Alpha"] = field(default_factory=list)


@dataclass
class FileInfo:
    size: int
    mime: Optional[str] = None


@dataclass
class Folder:
    name: str
    files: Dict[str, FileInfo] = field(default_factory=dict)
    folders: Dict[str, "Folder"] = field(default_factory=dict)


@dataclass
class Author:
    name: str
    latest: Optional["Book"] = None


@dataclass
class Book:
    title: str
    author: Optional[Author] = None


def test_quoted_class_name_inside_annotated_is_resolved():
    node = AnnotatedNode(1, [AnnotatedNode(2), None], AnnotatedNode(3, next=AnnotatedNode(4)))

    restored = assert_round_trip(node, AnnotatedNode)

    assert type(restored.children[0]) is AnnotatedNode and type(restored.next.next) is AnnotatedNode


def linked_list(length):
    head = None
    for value in range(length, 0, -1):
        head = ListNode(value, head)
    return head


def linked_json(length):
    data = None
    for value in range(length, 0, -1):
        data = {"value": value, "next": data}
    return data


def chain_tree(depth):
    tree = Tree(f"n{depth - 1}")
    for level in range(depth - 2, -1, -1):
        tree = Tree(f"n{level}", [tree])
    return tree


def chain_json(depth):
    data = {"label": f"n{depth - 1}", "children": []}
    for level in range(depth - 2, -1, -1):
        data = {"label": f"n{level}", "children": [data]}
    return data


def wide_tree(width):
    return Tree("root", [Tree(f"leaf{index}") for index in range(width)])


def wide_json(width):
    return {"label": "root", "children": [{"label": f"leaf{index}", "children": []} for index in range(width)]}


def ternary_tree(depth, label="r"):
    return Tree(label, [ternary_tree(depth - 1, f"{label}{index}") for index in range(3)] if depth else [])


def ternary_json(depth, label="r"):
    children = [ternary_json(depth - 1, f"{label}{index}") for index in range(3)] if depth else []
    return {"label": label, "children": children}


def complete_binary_tree(depth, index=1):
    node = BinNode(index)
    if depth:
        node.left = complete_binary_tree(depth - 1, 2 * index)
        node.right = complete_binary_tree(depth - 1, 2 * index + 1)
    return node


def complete_binary_json(depth, index=1):
    data = {"value": index, "left": None, "right": None}
    if depth:
        data["left"] = complete_binary_json(depth - 1, 2 * index)
        data["right"] = complete_binary_json(depth - 1, 2 * index + 1)
    return data


def tree_labels(root):
    """The labels of a Tree in preorder, found iteratively."""
    labels, pending = [], [root]
    while pending:
        node = pending.pop()
        labels.append(node.label)
        pending.extend(reversed(node.children))
    return labels


def breadth_first_values(root):
    values, queue = [], [root]
    for node in queue:
        values.append(node.value)
        queue.extend(child for child in (node.left, node.right) if child is not None)
    return values


@pytest.mark.parametrize("length", [1, 2, 5, 100, 200])
def test_linked_list(length):
    head = linked_list(length)

    assert_text(head, linked_json(length))
    restored = assert_round_trip(head, ListNode)
    values = []
    while restored is not None:
        values.append(restored.value)
        restored = restored.next
    assert values == list(range(1, length + 1))


def test_linked_list_golden_text_and_optional_root():
    assert serialize_json(linked_list(3)) == (
        '{"value": 1, "next": {"value": 2, "next": {"value": 3, "next": null}}}'
    )
    assert_round_trip(None, Optional[ListNode])
    assert_round_trip(linked_list(3), Optional[ListNode])
    assert_round_trip([linked_list(2), linked_list(1)], List[ListNode])


@pytest.mark.parametrize(
    ("tree", "expected", "labels"),
    [
        pytest.param(Tree("leaf"), {"label": "leaf", "children": []}, ["leaf"], id="leaf"),
        pytest.param(chain_tree(2), chain_json(2), ["n0", "n1"], id="chain-2"),
        pytest.param(chain_tree(100), chain_json(100), [f"n{level}" for level in range(100)], id="chain-100"),
        pytest.param(wide_tree(1), wide_json(1), ["root", "leaf0"], id="wide-1"),
        pytest.param(wide_tree(500), wide_json(500), ["root"] + [f"leaf{i}" for i in range(500)], id="wide-500"),
        pytest.param(ternary_tree(0), ternary_json(0), ["r"], id="ternary-0"),
        pytest.param(
            ternary_tree(1),
            {
                "label": "r",
                "children": [
                    {"label": "r0", "children": []},
                    {"label": "r1", "children": []},
                    {"label": "r2", "children": []},
                ],
            },
            ["r", "r0", "r1", "r2"],
            id="ternary-1",
        ),
    ],
)
def test_tree_with_a_list_of_children(tree, expected, labels):
    assert_text(tree, expected)
    restored = assert_round_trip(tree, Tree)
    assert tree_labels(restored) == labels


def test_ternary_tree_of_depth_four_has_one_node_per_address():
    tree = ternary_tree(4)

    assert_text(tree, ternary_json(4))
    restored = assert_round_trip(tree, Tree)
    labels = tree_labels(restored)
    # the label of a node is its path from the root: "r" followed by a digit 0-2 per level
    assert sorted(labels) == sorted(
        "r" + "".join(path) for size in range(5) for path in itertools.product("012", repeat=size)
    )
    assert len(labels) == 1 + 3 + 9 + 27 + 81


@pytest.mark.parametrize("depth", [0, 1, 3, 8])
def test_complete_binary_tree(depth):
    tree = complete_binary_tree(depth)

    assert_text(tree, complete_binary_json(depth))
    restored = assert_round_trip(tree, BinNode)
    # a complete tree of depth d numbers its nodes 1 .. 2**(d + 1) - 1 level by level
    assert breadth_first_values(restored) == list(range(1, 2 ** (depth + 1)))


def test_binary_tree_of_depth_eight_has_511_nodes_in_a_few_known_places():
    restored = deserialize_json(to_json_value(complete_binary_tree(8)), BinNode)

    node = restored
    for _ in range(8):
        node = node.left
    assert (node.value, node.left, node.right) == (256, None, None)
    node = restored
    for _ in range(8):
        node = node.right
    assert (node.value, node.left, node.right) == (511, None, None)
    assert sum(breadth_first_values(restored)) == 511 * 512 // 2


@pytest.mark.parametrize("side", ["left", "right"])
def test_unbalanced_binary_tree_that_is_a_chain_down_one_side(side):
    root = BinNode(0)
    node = root
    for value in range(1, 100):
        child = BinNode(value)
        setattr(node, side, child)
        node = child

    restored = assert_round_trip(root, BinNode)

    assert breadth_first_values(restored) == list(range(100))
    assert to_json_value(root)["left" if side == "right" else "right"] is None


def test_mutually_recursive_dataclasses_through_optional_and_list():
    beta = Beta([Alpha(1, Beta([Alpha(2, Beta([Alpha(3)]))])), Alpha(4)])

    assert_text(
        beta,
        {
            "alphas": [
                {
                    "n": 1,
                    "beta": {"alphas": [{"n": 2, "beta": {"alphas": [{"n": 3, "beta": None}]}}]},
                },
                {"n": 4, "beta": None},
            ]
        },
    )
    assert_round_trip(beta, Beta)
    assert_round_trip(Alpha(1, beta), Alpha)
    assert_round_trip(Beta(), Beta)


def test_mutually_recursive_dataclasses_alternating_to_depth_six():
    author = None
    for depth in range(3):
        book = Book(f"book{depth}", author)
        author = Author(f"author{depth}", book)

    restored = assert_round_trip(author, Author)

    names = []
    node = restored
    while node is not None:
        names.append(node.name)
        node = node.latest.author if node.latest is not None else None
    assert names == ["author2", "author1", "author0"]


def sample_filesystem():
    return Folder(
        "root",
        files={"README.md": FileInfo(120, "text/markdown"), "LICENSE": FileInfo(1070)},
        folders={
            "src": Folder(
                "src",
                files={"main.py": FileInfo(2048, "text/x-python")},
                folders={"pkg": Folder("pkg", files={"__init__.py": FileInfo(0)})},
            ),
            "empty": Folder("empty"),
        },
    )


SAMPLE_FILESYSTEM_JSON = {
    "name": "root",
    "files": {
        "README.md": {"size": 120, "mime": "text/markdown"},
        "LICENSE": {"size": 1070, "mime": None},
    },
    "folders": {
        "src": {
            "name": "src",
            "files": {"main.py": {"size": 2048, "mime": "text/x-python"}},
            "folders": {
                "pkg": {
                    "name": "pkg",
                    "files": {"__init__.py": {"size": 0, "mime": None}},
                    "folders": {},
                }
            },
        },
        "empty": {"name": "empty", "files": {}, "folders": {}},
    },
}


def test_folder_tree_with_a_self_referential_dict():
    assert_text(sample_filesystem(), SAMPLE_FILESYSTEM_JSON)
    restored = assert_round_trip(sample_filesystem(), Folder)
    assert restored.folders["src"].folders["pkg"].files["__init__.py"].size == 0


def test_folder_tree_nested_one_hundred_folders_deep_and_one_hundred_folders_wide():
    deep = Folder("d99")
    for level in range(98, -1, -1):
        deep = Folder(f"d{level}", folders={f"d{level + 1}": deep})
    wide = Folder("wide", folders={f"f{i}": Folder(f"f{i}", files={"a": FileInfo(i)}) for i in range(100)})

    restored_deep = assert_round_trip(deep, Folder)
    restored = assert_round_trip(wide, Folder)
    depth, folder = 1, restored_deep
    while folder.folders:
        (folder,) = folder.folders.values()
        depth += 1
    assert depth == 100
    assert [folder.files["a"].size for folder in restored.folders.values()] == list(range(100))


def cyclic_list_node():
    node = ListNode(1)
    node.next = ListNode(2, node)
    return node


def cyclic_authors():
    author = Author("a")
    author.latest = Book("b", author)
    return author


def cyclic_tree():
    root = Tree("root")
    root.children.append(Tree("child", [root]))
    return root


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(cyclic_list_node, id="self-loop-through-optional"),
        pytest.param(cyclic_authors, id="mutual"),
        pytest.param(cyclic_tree, id="through-a-list"),
    ],
)
def test_cyclic_dataclasses_are_rejected(make):
    with pytest.raises(ValueError, match="(?i)cyclic"):
        serialize_json(make())


def test_repeated_references_are_not_cycles():
    shared = Tree("shared")
    root = Tree("root", [shared, shared, Tree("other", [shared])])

    assert_text(
        root,
        {
            "label": "root",
            "children": [
                {"label": "shared", "children": []},
                {"label": "shared", "children": []},
                {"label": "other", "children": [{"label": "shared", "children": []}]},
            ],
        },
    )
    restored = assert_round_trip(root, Tree)
    assert restored.children[0] is not restored.children[1]


@dataclass
class RefOptionalNode:
    value: int
    next: Optional["RefOptionalNode"] = None


@dataclass
class RefListTree:
    label: str
    children: List["RefListTree"] = field(default_factory=list)


@dataclass
class RefBuiltinListTree:
    label: str
    children: list["RefBuiltinListTree"] = field(default_factory=list)


@dataclass
class RefDictFolder:
    name: str
    folders: Dict[str, "RefDictFolder"] = field(default_factory=dict)


@dataclass
class RefTupleNode:
    label: str
    children: Tuple["RefTupleNode", ...] = ()


@dataclass
class RefPipeTree:
    label: str
    children: list["RefPipeTree"] | None = None


@dataclass
class RefPipeFolder:
    name: str
    folders: dict[str, "RefPipeFolder"] | None = None


@dataclass
class RefNestedGenerics:
    label: str
    optional_list: Optional[List["RefNestedGenerics"]] = None
    by_name: Dict[str, List["RefNestedGenerics"]] = field(default_factory=dict)
    maybe_by_name: Dict[str, Optional["RefNestedGenerics"]] = field(default_factory=dict)


@dataclass(frozen=True)
class RefFrozenTree:
    label: str
    children: FrozenSet["RefFrozenTree"] = frozenset()


@dataclass
class RefLiteralNode:
    kind: Literal["leaf", "inner"]
    children: List["RefLiteralNode"] = field(default_factory=list)
    mode: Optional[Literal["fast", "slow"]] = None


@dataclass
class RefOwner:
    name: str
    pets: List["RefPet"] = field(default_factory=list)
    favorite: Optional["RefPet"] = None


@dataclass
class RefPet:  # defined after the class that names it, which names this one back
    name: str
    owner: Optional[RefOwner] = None


@dataclass
class RefCountingNode:
    name: str
    children: List["RefCountingNode"] = field(default_factory=list)
    size: int = field(init=False)

    def __post_init__(self):
        self.size = 1 + sum(child.size for child in self.children)


@dataclass
class RefWholeStringNode:
    next: "Optional[RefWholeStringNode]" = None


@dataclass
class RefWholeStringTree:
    label: str
    children: "List[RefWholeStringTree]" = field(default_factory=list)


@dataclass
class RefWholeStringFolder:
    name: str
    folders: "Dict[str, RefWholeStringFolder]" = field(default_factory=dict)


def nested_generics_json(label, optional_list=None, by_name=None, maybe_by_name=None):
    return {
        "label": label,
        "optional_list": optional_list,
        "by_name": by_name or {},
        "maybe_by_name": maybe_by_name or {},
    }


FORWARD_REFERENCE_CASES = [
    ("optional-of-a-name", RefOptionalNode(1, RefOptionalNode(2)), {"value": 1, "next": {"value": 2, "next": None}}),
    (
        "typing-list-of-a-name",
        RefListTree("a", [RefListTree("b")]),
        {"label": "a", "children": [{"label": "b", "children": []}]},
    ),
    (
        "builtin-list-of-a-name",
        RefBuiltinListTree("a", [RefBuiltinListTree("b")]),
        {"label": "a", "children": [{"label": "b", "children": []}]},
    ),
    (
        "dict-of-a-name",
        RefDictFolder("a", {"b": RefDictFolder("b")}),
        {"name": "a", "folders": {"b": {"name": "b", "folders": {}}}},
    ),
    (
        "variadic-tuple-of-a-name",
        RefTupleNode("a", (RefTupleNode("b"),)),
        {"label": "a", "children": [{"label": "b", "children": []}]},
    ),
    (
        "list-of-a-name-or-none",
        RefPipeTree("a", [RefPipeTree("b"), RefPipeTree("c", [])]),
        {"label": "a", "children": [{"label": "b", "children": None}, {"label": "c", "children": []}]},
    ),
    (
        "dict-of-a-name-or-none",
        RefPipeFolder("a", {"b": RefPipeFolder("b", {}), "c": RefPipeFolder("c")}),
        {"name": "a", "folders": {"b": {"name": "b", "folders": {}}, "c": {"name": "c", "folders": None}}},
    ),
    (
        "names-in-generics-in-generics",
        RefNestedGenerics(
            "a",
            [RefNestedGenerics("b")],
            {"k": [RefNestedGenerics("c"), RefNestedGenerics("d")], "none": []},
            {"m": RefNestedGenerics("e"), "n": None},
        ),
        nested_generics_json(
            "a",
            [nested_generics_json("b")],
            {"k": [nested_generics_json("c"), nested_generics_json("d")], "none": []},
            {"m": nested_generics_json("e"), "n": None},
        ),
    ),
    (
        "literals-next-to-a-name",
        RefLiteralNode("inner", [RefLiteralNode("leaf", [], "fast")], "slow"),
        {
            "kind": "inner",
            "children": [{"kind": "leaf", "children": [], "mode": "fast"}],
            "mode": "slow",
        },
    ),
    (
        "name-of-a-class-defined-later",
        RefOwner("o", [RefPet("p")], RefPet("q")),
        {
            "name": "o",
            "pets": [{"name": "p", "owner": None}],
            "favorite": {"name": "q", "owner": None},
        },
    ),
    (
        "two-classes-naming-each-other",
        RefPet("p", RefOwner("o", [RefPet("x", RefOwner("y"))])),
        {
            "name": "p",
            "owner": {
                "name": "o",
                "pets": [{"name": "x", "owner": {"name": "y", "pets": [], "favorite": None}}],
                "favorite": None,
            },
        },
    ),
    (
        "init-false-field-next-to-a-name",
        RefCountingNode("a", [RefCountingNode("b"), RefCountingNode("c", [RefCountingNode("d")])]),
        {
            "name": "a",
            "children": [
                {"name": "b", "children": [], "size": 1},
                {"name": "c", "children": [{"name": "d", "children": [], "size": 1}], "size": 2},
            ],
            "size": 4,
        },
    ),
    (
        "whole-string-optional",
        RefWholeStringNode(RefWholeStringNode()),
        {"next": {"next": None}},
    ),
    (
        "whole-string-list",
        RefWholeStringTree("a", [RefWholeStringTree("b")]),
        {"label": "a", "children": [{"label": "b", "children": []}]},
    ),
    (
        "whole-string-dict",
        RefWholeStringFolder("a", {"b": RefWholeStringFolder("b")}),
        {"name": "a", "folders": {"b": {"name": "b", "folders": {}}}},
    ),
]


@pytest.mark.parametrize(
    ("value", "expected"),
    [pytest.param(value, expected, id=case_id) for case_id, value, expected in FORWARD_REFERENCE_CASES],
)
def test_forward_reference_inside_a_generic_is_written_normally(value, expected):
    assert_text(value, expected)


@pytest.mark.parametrize(
    "value", [pytest.param(value, id=case_id) for case_id, value, _ in FORWARD_REFERENCE_CASES]
)
def test_forward_reference_inside_a_generic_round_trip(value):
    assert_round_trip(value, type(value))


@pytest.mark.parametrize(
    ("value", "cls"),
    [
        pytest.param({"x": RefOptionalNode(1, RefOptionalNode(2))}, Dict[str, RefOptionalNode], id="dict"),
        pytest.param([RefListTree("a", [RefListTree("b")]), RefListTree("c")], List[RefListTree], id="list"),
        pytest.param(RefDictFolder("a", {"b": RefDictFolder("b")}), Optional[RefDictFolder], id="optional"),
        pytest.param(None, Optional[RefPipeTree], id="optional-none"),
        pytest.param((RefPipeTree("a", [RefPipeTree("b")]),), Tuple[RefPipeTree, ...], id="tuple"),
    ],
)
def test_classes_with_forward_references_inside_other_containers_round_trip(value, cls):
    assert_round_trip(value, cls)


def test_forward_references_are_restored_from_hand_written_json():
    assert_strictly_equal(
        deserialize_json({"value": 1, "next": {"value": 2}}, RefOptionalNode),
        RefOptionalNode(1, RefOptionalNode(2)),
    )
    assert_strictly_equal(
        deserialize_json({"label": "a", "children": [{"label": "b"}, {"label": "c", "children": None}]}, RefPipeTree),
        RefPipeTree("a", [RefPipeTree("b"), RefPipeTree("c")]),
    )
    assert_strictly_equal(
        deserialize_json({"name": "a", "folders": {"b": {"name": "b", "folders": {}}}}, RefDictFolder),
        RefDictFolder("a", {"b": RefDictFolder("b")}),
    )


def test_frozenset_of_a_name_round_trips():
    tree = RefFrozenTree("a", frozenset({RefFrozenTree("b", frozenset({RefFrozenTree("c")}))}))

    # the keys of a dataclass inside a set are sorted, hence assert_json
    assert_json(tree, {"label": "a", "children": [{"label": "b", "children": [{"label": "c", "children": []}]}]})
    assert_round_trip(tree, RefFrozenTree)
    assert_round_trip(RefFrozenTree("alone"), RefFrozenTree)


def test_class_defined_inside_a_function_can_name_itself_inside_a_generic():
    @dataclass
    class Local:
        value: int
        next: Optional["Local"] = None
        others: List["Local"] = field(default_factory=list)

    assert_round_trip(Local(1, Local(2), [Local(3, None, [Local(4)])]), Local)


# --- one object holding a bit of everything ----------------------------------------------------------------------


@dataclass
class KitchenSink:
    text: str
    count: int
    ratio: float
    flag: bool
    color: Color
    priority: Priority
    created: datetime.datetime
    day: datetime.date
    at: datetime.time
    price: Decimal
    blob: bytes
    path: Path
    tags: Set[str]
    ids: FrozenSet[int]
    pair: Tuple[int, int]
    scores: Tuple[float, ...]
    mode: Literal["fast", "slow"]
    nickname: Optional[str]
    either: Union[Address, int]
    anything: Any
    by_color: Dict[Color, int]
    frozen_map: MappingProxyType[str, int]
    addresses: List[Address]
    home: Optional[Address]


@dataclass
class SinkHolder:
    sink: KitchenSink
    history: List[KitchenSink]
    by_name: Dict[str, KitchenSink]
    maybe: Optional[KitchenSink]


FULL_PATH = Path("reports") / "2024" / "summary.txt"
PLUS_0530 = datetime.timezone(datetime.timedelta(hours=5, minutes=30))


def full_sink():
    return KitchenSink(
        text='line one\nline "two" — ünï',
        count=2**63,
        ratio=0.1,
        flag=True,
        color=Color.GREEN,
        priority=Priority.HIGH,
        created=datetime.datetime(2024, 2, 29, 23, 59, 59, 123456, tzinfo=PLUS_0530),
        day=datetime.date(2024, 2, 29),
        at=datetime.time(7, 5, 3, 250000),
        price=Decimal("1250.50"),
        blob=b"\x00\xffJSerPy",
        path=FULL_PATH,
        tags={"gamma", "alpha", "beta"},
        ids=frozenset({10, 9, 100}),
        pair=(3, 4),
        scores=(1.5, 2.0, -0.0),
        mode="slow",
        nickname="ada",
        either=Address("Rome", "00100"),
        anything={"k": [1, {"z": None}], "f": 1.0},
        by_color={Color.GREEN: 2, Color.RED: 1},
        frozen_map=MappingProxyType({"b": 2, "a": 1}),
        addresses=[Address("Oslo"), Address("Bern", "3000")],
        home=Address("Kyiv"),
    )


def minimal_sink():
    return KitchenSink(
        text="",
        count=0,
        ratio=0.0,
        flag=False,
        color=Color.RED,
        priority=Priority.LOW,
        created=datetime.datetime(1, 1, 1),
        day=datetime.date(1, 1, 1),
        at=datetime.time(0, 0),
        price=Decimal("0"),
        blob=b"",
        path=Path("."),
        tags=set(),
        ids=frozenset(),
        pair=(0, 0),
        scores=(),
        mode="fast",
        nickname=None,
        either=0,
        anything=None,
        by_color={},
        frozen_map=MappingProxyType({}),
        addresses=[],
        home=None,
    )


FULL_SINK_JSON = {
    "text": 'line one\nline "two" — ünï',
    "count": 9223372036854775808,
    "ratio": 0.1,
    "flag": True,
    "color": "green",
    "priority": 2,
    "created": "2024-02-29T23:59:59.123456+05:30",
    "day": "2024-02-29",
    "at": "07:05:03.250000",
    "price": "1250.50",
    "blob": "AP9KU2VyUHk=",
    "path": str(FULL_PATH),
    "tags": ["alpha", "beta", "gamma"],
    # ordered by the JSON text of each item, not by value: "10" < "100" < "9"
    "ids": [10, 100, 9],
    "pair": [3, 4],
    "scores": [1.5, 2.0, -0.0],
    "mode": "slow",
    "nickname": "ada",
    "either": {"city": "Rome", "zip_code": "00100"},
    "anything": {"k": [1, {"z": None}], "f": 1.0},
    "by_color": {"green": 2, "red": 1},
    "frozen_map": {"b": 2, "a": 1},
    "addresses": [{"city": "Oslo", "zip_code": None}, {"city": "Bern", "zip_code": "3000"}],
    "home": {"city": "Kyiv", "zip_code": None},
}
MINIMAL_SINK_JSON = {
    "text": "",
    "count": 0,
    "ratio": 0.0,
    "flag": False,
    "color": "red",
    "priority": 1,
    "created": "0001-01-01T00:00:00",
    "day": "0001-01-01",
    "at": "00:00:00",
    "price": "0",
    "blob": "",
    "path": ".",
    "tags": [],
    "ids": [],
    "pair": [0, 0],
    "scores": [],
    "mode": "fast",
    "nickname": None,
    "either": 0,
    "anything": None,
    "by_color": {},
    "frozen_map": {},
    "addresses": [],
    "home": None,
}


def test_fully_populated_kitchen_sink():
    assert_text(full_sink(), FULL_SINK_JSON)

    restored = assert_round_trip(full_sink(), KitchenSink)

    assert restored.created.utcoffset() == datetime.timedelta(hours=5, minutes=30)
    assert [key.value for key in restored.by_color] == ["green", "red"]
    assert type(restored.frozen_map) is MappingProxyType


def test_minimal_kitchen_sink():
    assert_text(minimal_sink(), MINIMAL_SINK_JSON)

    assert_round_trip(minimal_sink(), KitchenSink)


@pytest.mark.parametrize(
    ("value", "cls", "expected"),
    [
        pytest.param(
            [minimal_sink(), full_sink()], List[KitchenSink], [MINIMAL_SINK_JSON, FULL_SINK_JSON], id="list"
        ),
        pytest.param(
            {"full": full_sink(), "minimal": minimal_sink()},
            Dict[str, KitchenSink],
            {"full": FULL_SINK_JSON, "minimal": MINIMAL_SINK_JSON},
            id="dict",
        ),
        pytest.param(full_sink(), Optional[KitchenSink], FULL_SINK_JSON, id="optional-present"),
        pytest.param(None, Optional[KitchenSink], None, id="optional-null"),
        pytest.param(
            SinkHolder(full_sink(), [minimal_sink(), full_sink()], {"min": minimal_sink()}, None),
            SinkHolder,
            {
                "sink": FULL_SINK_JSON,
                "history": [MINIMAL_SINK_JSON, FULL_SINK_JSON],
                "by_name": {"min": MINIMAL_SINK_JSON},
                "maybe": None,
            },
            id="field-list-dict-and-null",
        ),
        pytest.param(
            SinkHolder(minimal_sink(), [], {}, full_sink()),
            SinkHolder,
            {"sink": MINIMAL_SINK_JSON, "history": [], "by_name": {}, "maybe": FULL_SINK_JSON},
            id="field-empty-containers-and-optional-present",
        ),
    ],
)
def test_kitchen_sink_inside_other_containers_and_dataclasses(value, cls, expected):
    assert_text(value, expected)
    assert_round_trip(value, cls)


def test_any_field_holding_a_dataclass_comes_back_as_a_dict_limitation():
    # Documents current behavior, not a promise: Any carries no type to rebuild the dataclass from.
    sink = minimal_sink()
    sink.anything = Address("Oslo", "0150")

    restored = deserialize_json(to_json_value(sink), KitchenSink)

    assert restored.anything == {"city": "Oslo", "zip_code": "0150"}
    assert type(restored.anything) is dict


# --- unions of different dataclasses and discriminated unions ----------------------------------------------------


@dataclass
class Circle:
    radius: float


@dataclass
class Square:
    side: float


@dataclass
class Triangle:
    base: float
    height: float


Shape = Union[Circle, Square, Triangle]


@dataclass
class Drawing:
    main: Shape
    extras: List[Shape]


@dataclass
class Cat:
    kind: Literal["cat"] = "cat"
    lives: int = 9


@dataclass
class Fish:
    kind: Literal["fish"] = "fish"
    fins: int = 2


@dataclass
class Bird:
    kind: Literal["bird"] = "bird"
    wingspan: float = 0.5


Pet = Union[Cat, Fish, Bird]


@dataclass
class Household:
    main: Pet
    others: List[Pet]
    by_name: Dict[str, Pet]
    maybe: Optional[Pet] = None


MIXED_SHAPES = [Circle(1.0), Square(2.0), Triangle(3.0, 4.0)]


def test_members_of_a_union_of_dataclasses_are_written_with_their_own_fields():
    assert_text(MIXED_SHAPES, [{"radius": 1.0}, {"side": 2.0}, {"base": 3.0, "height": 4.0}])


def test_union_of_dataclasses_restores_the_first_member():
    assert_round_trip(Circle(1.0), Shape)
    assert_round_trip([Circle(1.0), Circle(2.5)], List[Shape])
    assert_round_trip(Drawing(Circle(1.0), [Circle(2.0)]), Drawing)


@pytest.mark.parametrize(
    ("data", "cls", "expected"),
    [
        pytest.param({"side": 2.0}, Shape, Circle(None), id="square-data-with-circle-first"),
        pytest.param({"base": 3.0, "height": 4.0}, Shape, Circle(None), id="triangle-data-with-circle-first"),
        pytest.param({"radius": 1.0}, Union[Square, Circle], Square(None), id="circle-data-with-square-first"),
        pytest.param({"zzz": 1}, Union[Circle, dict], Circle(None), id="a-dict-member-is-never-reached"),
        pytest.param([{"side": 2.0}], List[Shape], [Circle(None)], id="list"),
        pytest.param({"s": {"side": 2.0}}, Dict[str, Optional[Shape]], {"s": Circle(None)}, id="dict-of-optional"),
    ],
)
def test_union_of_dataclasses_with_scalar_fields_restores_the_first_member_limitation(data, cls, expected):
    # Documents current behavior, not a promise: a missing field becomes None and unknown keys are ignored, so the
    # first dataclass of the union accepts any JSON object and the other members are never tried.
    assert_strictly_equal(deserialize_json(data, cls), expected)


@dataclass
class Polygon:
    sides: List[float]


@dataclass
class Disc:
    radius: float


def test_union_moves_on_when_a_required_array_field_is_missing_limitation():
    # Documents current behavior, not a promise: the first member is passed over only because its absent array field
    # becomes None, which an array annotation rejects. A scalar field (Disc) rejects nothing.
    assert_round_trip(Polygon([1.0, 2.5]), Union[Polygon, Disc])
    assert_round_trip(Disc(1.0), Union[Polygon, Disc])
    assert_round_trip([Disc(1.0), Polygon([]), Disc(2.0)], List[Union[Polygon, Disc]])
    assert_round_trip({"p": Polygon([3.0]), "d": None}, Dict[str, Optional[Union[Polygon, Disc]]])


@dataclass
class Scene:
    circle: Circle
    square: Square
    triangle: Triangle


def test_one_annotation_per_position_restores_different_dataclasses():
    assert_text(
        Scene(*MIXED_SHAPES),
        {"circle": {"radius": 1.0}, "square": {"side": 2.0}, "triangle": {"base": 3.0, "height": 4.0}},
    )
    assert_round_trip(Scene(*MIXED_SHAPES), Scene)
    assert_round_trip(tuple(MIXED_SHAPES), Tuple[Circle, Square, Triangle])


@pytest.mark.parametrize(
    ("value", "cls", "expected"),
    [
        pytest.param(Fish(fins=3), Pet, {"kind": "fish", "fins": 3}, id="root"),
        pytest.param(
            [Fish(), Bird(wingspan=1.5), Cat(), Cat(lives=1)],
            List[Pet],
            [
                {"kind": "fish", "fins": 2},
                {"kind": "bird", "wingspan": 1.5},
                {"kind": "cat", "lives": 9},
                {"kind": "cat", "lives": 1},
            ],
            id="list",
        ),
        pytest.param(
            (Bird(), Fish()),
            Tuple[Pet, ...],
            [{"kind": "bird", "wingspan": 0.5}, {"kind": "fish", "fins": 2}],
            id="tuple",
        ),
        pytest.param(None, Optional[Pet], None, id="optional-null"),
        pytest.param(Bird(), Optional[Pet], {"kind": "bird", "wingspan": 0.5}, id="optional-present"),
        pytest.param(
            Household(Fish(), [Bird(), Cat()], {"tom": Cat(), "nemo": Fish(fins=5)}, Bird(wingspan=2.0)),
            Household,
            {
                "main": {"kind": "fish", "fins": 2},
                "others": [{"kind": "bird", "wingspan": 0.5}, {"kind": "cat", "lives": 9}],
                "by_name": {"tom": {"kind": "cat", "lives": 9}, "nemo": {"kind": "fish", "fins": 5}},
                "maybe": {"kind": "bird", "wingspan": 2.0},
            },
            id="household",
        ),
        pytest.param(
            Household(Cat(), [], {}),
            Household,
            {"main": {"kind": "cat", "lives": 9}, "others": [], "by_name": {}, "maybe": None},
            id="sparse-household",
        ),
    ],
)
def test_union_of_dataclasses_with_a_literal_discriminator_round_trips_every_member(value, cls, expected):
    assert_text(value, expected)
    assert_round_trip(value, cls)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"kind": "dog"}, id="unknown-tag"),
        pytest.param([], id="array"),
        pytest.param("cat", id="string"),
        pytest.param(None, id="null"),
    ],
)
def test_discriminated_union_rejects_data_that_matches_no_member(data):
    with pytest.raises(TypeError):
        deserialize_json(data, Pet)


@pytest.mark.parametrize(
    "cls",
    [
        pytest.param(Union[int, Address], id="int-first"),
        pytest.param(Union[str, Address], id="str-first"),
        pytest.param(Union[float, Address], id="float-first"),
        pytest.param(Union[Address, int], id="dataclass-first"),
        pytest.param(Union[None, Address], id="none-first"),
    ],
)
def test_dataclass_listed_after_a_primitive_member_of_a_union_is_restored(cls):
    assert_round_trip(Address("Rome", "00100"), cls)


# --- nothing is modified, nothing is shared ----------------------------------------------------------------------


@dataclass
class Inventory:
    items: List[str]
    counts: Dict[str, int]
    grid: List[List[int]]
    meta: Dict[str, Any]
    owner: Address
    labels: Set[str]
    pair: Tuple[int, List[int]]
    raw: list
    extra: Any = None


def make_inventory():
    return Inventory(
        items=["a", "b"],
        counts={"a": 1},
        grid=[[1, 2], [3]],
        meta={"deep": [1, {"k": [2]}]},
        owner=Address("Rome", "00100"),
        labels={"x", "y"},
        pair=(1, [2, 3]),
        raw=[{"r": [1]}],
        extra={"e": [0]},
    )


def scribble(value):
    """Add an entry to every dict and list in a JSON value, the nested ones included."""
    if isinstance(value, dict):
        for item in list(value.values()):
            scribble(item)
        value["scribble"] = "scribble"
    elif isinstance(value, list):
        for item in list(value):
            scribble(item)
        value.append("scribble")


def test_serializing_does_not_modify_the_instance():
    inventory = make_inventory()
    held = {name: getattr(inventory, name) for name in ("items", "counts", "grid", "meta", "owner", "labels", "raw")}

    text = serialize_json(inventory)
    serialize_json_as_obj(inventory)

    assert_strictly_equal(inventory, make_inventory())
    assert all(getattr(inventory, name) is member for name, member in held.items())
    assert serialize_json(inventory) == text


def test_changing_the_serialized_tree_does_not_change_the_instance():
    inventory = make_inventory()
    result = serialize_json_as_obj(inventory)

    scribble(result)

    assert_strictly_equal(inventory, make_inventory())


def test_changing_the_instance_does_not_change_a_serialized_tree():
    inventory = make_inventory()
    result = serialize_json_as_obj(inventory)
    snapshot = json.dumps(result)

    inventory.items.append("changed")
    inventory.grid[0].append(99)
    inventory.meta["deep"][1]["k"].append(99)
    inventory.owner.city = "changed"
    inventory.pair[1].append(99)
    inventory.raw[0]["r"].append(99)
    inventory.extra["e"].append(99)

    assert json.dumps(result) == snapshot


def test_restored_dataclass_does_not_share_containers_with_the_input():
    parsed = json.loads(serialize_json(make_inventory()))
    snapshot = copy.deepcopy(parsed)

    restored = deserialize_json(parsed, Inventory)
    assert_strictly_equal(restored, make_inventory())

    restored.items.append("changed")
    restored.counts["changed"] = 1
    restored.grid[0].append(99)
    restored.meta["deep"][1]["k"].append(99)
    restored.owner.city = "changed"
    restored.labels.add("changed")
    restored.pair[1].append(99)
    restored.raw[0]["r"].append(99)
    restored.extra["e"].append(99)

    assert parsed == snapshot


def test_changing_the_input_after_restoring_does_not_change_the_restored_dataclass():
    parsed = json.loads(serialize_json(make_inventory()))

    restored = deserialize_json(parsed, Inventory)
    scribble(parsed)

    assert_strictly_equal(restored, make_inventory())


# --- encoding options on dataclasses -----------------------------------------------------------------------------


@dataclass
class Customer:
    name: str
    address: Address


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        pytest.param(
            {}, '{"name": "Zo\\u00eb", "address": {"city": "Haifa", "zip_code": null}}', id="default"
        ),
        pytest.param(
            {"ensure_ascii": False},
            '{"name": "Zoë", "address": {"city": "Haifa", "zip_code": null}}',
            id="ensure-ascii-false",
        ),
        pytest.param(
            {"separators": (",", ":")},
            '{"name":"Zo\\u00eb","address":{"city":"Haifa","zip_code":null}}',
            id="compact",
        ),
        pytest.param(
            {"sort_keys": True},
            '{"address": {"city": "Haifa", "zip_code": null}, "name": "Zo\\u00eb"}',
            id="sort-keys",
        ),
        pytest.param(
            {"indent": 2},
            '{\n  "name": "Zo\\u00eb",\n  "address": {\n    "city": "Haifa",\n    "zip_code": null\n  }\n}',
            id="indent",
        ),
    ],
)
def test_encoding_options_apply_to_nested_dataclasses(options, expected):
    value = Customer("Zoë", Address("Haifa"))

    assert serialize_json(value, **options) == expected
    assert_round_trip(value, Customer, **options)

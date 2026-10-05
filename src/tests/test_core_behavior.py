import json
from collections import UserDict
from dataclasses import dataclass
from datetime import time
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

from jserpy import deserialize_json, serialize_json, serialize_json_as_obj


@dataclass
class NewCoreTypes:
    amount: Decimal
    at: time
    tags: set[str]
    frozen_numbers: frozenset[int]


def test_time_uses_iso_8601_representation():
    assert serialize_json(time(12, 34, 56, 789000)) == '"12:34:56.789000"'


def test_decimal_is_a_string_that_preserves_precision_and_trailing_zeroes():
    assert serialize_json(Decimal("1.20")) == '"1.20"'


def test_read_only_mapping_is_serialized_as_a_json_object():
    value = MappingProxyType({"b": 2, "a": 1})

    assert serialize_json(value, sort_keys=True) == '{"a": 1, "b": 2}'


def test_user_dict_is_serialized_as_a_json_object():
    value = UserDict({"b": 2, "a": 1})

    assert serialize_json(value, sort_keys=True) == '{"a": 1, "b": 2}'


def test_mapping_intermediate_matches_text_transformations():
    value = UserDict(
        {"read_only": MappingProxyType({"path": Path("report.json")})}
    )

    assert serialize_json_as_obj(value) == json.loads(serialize_json(value)) == {
        "read_only": {"path": "report.json"}
    }


def test_new_core_types_work_in_nested_combinations():
    value = UserDict(
        {
            "mapping": MappingProxyType({"path": Path("report.json")}),
            "values": [
                Decimal("1.20"),
                time(8, 9, 10),
                frozenset({"b", "a"}),
            ],
        }
    )

    assert serialize_json(value, sort_keys=True) == (
        '{"mapping": {"path": "report.json"}, '
        '"values": ["1.20", "08:09:10", ["a", "b"]]}'
    )


def test_set_order_is_canonical_across_different_construction_orders():
    forward = set()
    reverse = set()
    for value in ["zeta", "alpha", "middle"]:
        forward.add(value)
    for value in ["middle", "alpha", "zeta"]:
        reverse.add(value)

    expected = '["alpha", "middle", "zeta"]'
    assert serialize_json(forward) == expected
    assert serialize_json(reverse) == expected


def test_frozenset_uses_lexical_canonical_json_order_not_numeric_order():
    assert serialize_json(frozenset({10, 2, 1})) == '[1, 10, 2]'


def test_set_of_tuples_is_sorted_by_normalized_compact_json():
    value = {("b", 2), ("a", 2), ("a", 10)}

    assert serialize_json(value) == '[["a", 10], ["a", 2], ["b", 2]]'


def test_time_round_trip_uses_requested_type():
    value = time(12, 34, 56, 789000)

    restored = deserialize_json(json.loads(serialize_json(value)), time)

    assert restored == value
    assert type(restored) is time


def test_decimal_round_trip_preserves_exponent():
    value = Decimal("1.20")

    restored = deserialize_json(json.loads(serialize_json(value)), Decimal)

    assert restored.as_tuple() == value.as_tuple()


def test_typed_set_round_trip():
    value = {3, 1, 2}

    restored = deserialize_json(json.loads(serialize_json(value)), set[int])

    assert restored == value
    assert type(restored) is set


def test_typed_frozenset_round_trip():
    value = frozenset({"z", "a"})

    restored = deserialize_json(
        json.loads(serialize_json(value)),
        frozenset[str],
    )

    assert restored == value
    assert type(restored) is frozenset


def test_dataclass_with_new_typed_fields_round_trip():
    value = NewCoreTypes(
        amount=Decimal("1.20"),
        at=time(12, 34, 56),
        tags={"z", "a"},
        frozen_numbers=frozenset({10, 2, 1}),
    )

    restored = deserialize_json(json.loads(serialize_json(value)), NewCoreTypes)

    assert restored == value
    assert restored.amount.as_tuple() == value.amount.as_tuple()

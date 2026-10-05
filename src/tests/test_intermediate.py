import json
from datetime import time
from decimal import Decimal
from enum import Enum
from pathlib import Path

import pytest

from jserpy import serialize_json, serialize_json_as_dict, serialize_json_as_obj


class State(Enum):
    READY = "ready"


class UnsupportedValue:
    pass


def assert_json_compatible_tree(value):
    if value is None or type(value) in {bool, int, float, str}:
        return

    if isinstance(value, list):
        for item in value:
            assert_json_compatible_tree(item)
        return

    if isinstance(value, dict):
        assert all(type(key) is str for key in value)
        for item in value.values():
            assert_json_compatible_tree(item)
        return

    pytest.fail(f"non-JSON intermediate type: {type(value).__name__}")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param({"value": 1}, {"value": 1}, id="dictionary-root"),
        pytest.param([1, "two"], [1, "two"], id="list-root"),
        pytest.param("value", "value", id="scalar-root"),
        pytest.param(None, None, id="null-root"),
    ],
)
def test_intermediate_accepts_every_json_root(value, expected):
    assert serialize_json_as_obj(value) == expected


def test_intermediate_output_is_fully_json_compatible():
    value = {
        "state": State.READY,
        "path": Path("report.json"),
        "time": time(12, 30, 15),
        "amount": Decimal("1.20"),
        "nested": ({"values": {3, 1, 2}},),
        ("tuple", 1): {"nested_key": True},
    }

    result = serialize_json_as_obj(value)

    assert_json_compatible_tree(result)


def test_intermediate_is_detached_from_mutable_source_containers():
    source = {"items": [{"value": 1}]}

    result = serialize_json_as_obj(source)
    result["items"][0]["value"] = 2
    result["items"].append({"value": 3})

    assert source == {"items": [{"value": 1}]}
    source["items"][0]["value"] = 4
    assert result == {"items": [{"value": 2}, {"value": 3}]}


def test_intermediate_matches_text_serializer_transformations():
    value = {
        "state": State.READY,
        "path": Path("report.json"),
        "amount": Decimal("1.20"),
        "values": frozenset({"z", "a"}),
    }

    assert serialize_json_as_obj(value) == json.loads(serialize_json(value))


def test_intermediate_and_text_serializers_have_fallback_parity():
    unsupported = UnsupportedValue()

    def fallback(value):
        assert value is unsupported
        return {"at": time(12, 30), "path": Path("report.json")}

    assert serialize_json_as_obj(unsupported, fallback=fallback) == json.loads(
        serialize_json(unsupported, fallback=fallback)
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param({"value": 1}, {"value": 1}, id="dictionary-root"),
        pytest.param([1, 2], [1, 2], id="list-root"),
        pytest.param("value", "value", id="scalar-root"),
        pytest.param(None, None, id="null-root"),
    ],
)
def test_serialize_json_as_dict_remains_a_compatible_alias(value, expected):
    assert serialize_json_as_dict(value) == expected


def test_compatibility_alias_accepts_fallback():
    unsupported = UnsupportedValue()

    assert serialize_json_as_dict(
        unsupported,
        fallback=lambda value: {"converted": Path("report.json")},
    ) == {"converted": "report.json"}

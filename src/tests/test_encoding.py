from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import pytest

from jserpy import serialize_json


class State(Enum):
    READY = "ready"


@dataclass
class Record:
    path: Path
    state: State


def test_ensure_ascii_false_preserves_unicode():
    assert serialize_json({"message": "שלום"}, ensure_ascii=False) == (
        '{"message": "שלום"}'
    )


def test_sort_keys_orders_nested_dictionary_keys():
    value = {"z": 1, "a": {"d": 4, "b": 2}}

    assert serialize_json(value, sort_keys=True) == (
        '{"a": {"b": 2, "d": 4}, "z": 1}'
    )


def test_indent_two_matches_json_document_format():
    value = {"items": [1, 2], "active": True}

    assert serialize_json(value, indent=2) == (
        "{\n"
        '  "items": [\n'
        "    1,\n"
        "    2\n"
        "  ],\n"
        '  "active": true\n'
        "}"
    )


def test_compact_separators_remove_optional_whitespace():
    value = {"items": [1, 2], "active": True}

    assert serialize_json(value, separators=(",", ":")) == (
        '{"items":[1,2],"active":true}'
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_allow_nan_false_rejects_each_non_finite_float(value):
    with pytest.raises(ValueError):
        serialize_json({"value": value}, allow_nan=False)


def test_document_option_combination_preserves_complex_handlers():
    value = {
        "state": State.READY,
        "message": "שלום",
        "created": datetime(2024, 2, 29, 12, 30),
    }

    assert serialize_json(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) == (
        "{\n"
        '  "created": "2024-02-29T12:30:00",\n'
        '  "message": "שלום",\n'
        '  "state": "ready"\n'
        "}"
    )


def test_compact_jsonl_option_combination_preserves_complex_handlers():
    value = Record(path=Path("report.json"), state=State.READY)

    assert serialize_json(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ) == '{"path":"report.json","state":"ready"}'


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({}, id="defaults"),
        pytest.param({"indent": 2}, id="indented"),
        pytest.param({"separators": (",", ":")}, id="compact"),
    ],
)
def test_serializer_never_adds_a_trailing_newline(options):
    assert not serialize_json({"value": 1}, **options).endswith("\n")

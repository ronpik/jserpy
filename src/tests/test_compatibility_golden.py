from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path

import pytest

from jserpy import serialize_json


@dataclass
class Address:
    city: str
    postal_code: int


@dataclass
class Customer:
    name: str
    address: Address


class Status(Enum):
    READY = "ready"


class Key(Enum):
    ALPHA = "alpha"


def test_ascii_dictionary_zero_option_golden():
    value = {
        "name": "Alice",
        "active": True,
        "score": 12.5,
        "note": None,
    }

    assert serialize_json(value) == (
        '{"name": "Alice", "active": true, "score": 12.5, "note": null}'
    )


def test_unicode_uses_default_ascii_escaping_zero_option_golden():
    assert serialize_json({"message": "שלום"}) == (
        '{"message": "\\u05e9\\u05dc\\u05d5\\u05dd"}'
    )


def test_nested_dataclass_zero_option_golden():
    value = Customer(
        name="Alice",
        address=Address(city="Haifa", postal_code=31000),
    )

    assert serialize_json(value) == (
        '{"name": "Alice", "address": '
        '{"city": "Haifa", "postal_code": 31000}}'
    )


def test_enum_zero_option_golden():
    assert serialize_json(Status.READY) == '"ready"'


def test_datetime_and_date_zero_option_golden():
    value = {
        "created_at": datetime(2024, 2, 29, 12, 34, 56, 789000),
        "day": date(2024, 2, 29),
    }

    assert serialize_json(value) == (
        '{"created_at": "2024-02-29T12:34:56.789000", '
        '"day": "2024-02-29"}'
    )


def test_path_zero_option_golden():
    assert serialize_json(Path("report.json")) == '"report.json"'


def test_tuple_and_list_nesting_zero_option_golden():
    assert serialize_json({"items": (1, ["two", (3, 4)])}) == (
        '{"items": [1, ["two", [3, 4]]]}'
    )


def test_bytes_base64_zero_option_golden():
    assert serialize_json(b"\x00\xffJSerPy") == '"AP9KU2VyUHk="'


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param({"plain": "value"}, '{"plain": "value"}', id="string"),
        pytest.param({7: "value"}, '{"7": "value"}', id="integer"),
        pytest.param({1.5: "value"}, '{"1.5": "value"}', id="float"),
        pytest.param({True: "value"}, '{"true": "value"}', id="boolean"),
        pytest.param({Key.ALPHA: "value"}, '{"alpha": "value"}', id="enum"),
        pytest.param(
            {("alpha", 2): "value"},
            '{"[\\"alpha\\", 2]": "value"}',
            id="tuple",
        ),
        pytest.param(
            {(1, ("alpha", 2)): "value"},
            '{"[1, [\\"alpha\\", 2]]": "value"}',
            id="nested-tuple",
        ),
    ],
)
def test_supported_dictionary_key_zero_option_goldens(value, expected):
    assert serialize_json(value) == expected

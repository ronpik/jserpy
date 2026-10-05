from collections import UserDict
from datetime import datetime, time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from types import MappingProxyType

import pytest

from jserpy import serialize_json, serialize_json_as_obj
from jserpy.jsonable import Jsonable


class State(Enum):
    READY = "ready"


class UnsupportedValue:
    def __repr__(self):
        return "FALLBACK_PAYLOAD_MUST_NOT_LEAK"


class JsonableMapping(UserDict, Jsonable):
    def to_json(self):
        return {"source": "handler"}

    @classmethod
    def from_json(cls, data):
        return cls(data)


@pytest.mark.parametrize(
    ("serializer", "expected"),
    [
        pytest.param(
            serialize_json,
            '{"path": "report.json", "state": "ready", '
            '"unsupported": "converted"}',
            id="text",
        ),
        pytest.param(
            serialize_json_as_obj,
            {
                "path": "report.json",
                "state": "ready",
                "unsupported": "converted",
            },
            id="intermediate",
        ),
    ],
)
def test_fallback_is_called_only_for_unsupported_values(serializer, expected):
    unsupported = UnsupportedValue()
    calls = []

    def fallback(value):
        calls.append(value)
        return "converted"

    value = {
        "path": Path("report.json"),
        "state": State.READY,
        "unsupported": unsupported,
    }

    assert serializer(value, fallback=fallback) == expected
    assert calls == [unsupported]


def test_registered_jsonable_handler_precedes_generic_mapping_support():
    value = JsonableMapping({"source": "mapping"})

    def fallback(unexpected):
        raise AssertionError(f"fallback received supported {type(unexpected).__name__}")

    assert serialize_json(value, fallback=fallback) == '{"source": "handler"}'


@pytest.mark.parametrize(
    ("value", "text_expected", "intermediate_expected"),
    [
        pytest.param(
            Path("report.json"),
            '"report.json"',
            "report.json",
            id="path",
        ),
        pytest.param(State.READY, '"ready"', "ready", id="enum"),
        pytest.param(b"JSerPy", '"SlNlclB5"', "SlNlclB5", id="bytes"),
        pytest.param(
            datetime(2024, 2, 29, 12, 30),
            '"2024-02-29T12:30:00"',
            "2024-02-29T12:30:00",
            id="datetime",
        ),
        pytest.param(time(12, 30), '"12:30:00"', "12:30:00", id="time"),
        pytest.param(Decimal("1.20"), '"1.20"', "1.20", id="decimal"),
        pytest.param(frozenset({2, 1}), "[1, 2]", [1, 2], id="frozenset"),
        pytest.param(
            MappingProxyType({"value": 1}),
            '{"value": 1}',
            {"value": 1},
            id="mapping",
        ),
    ],
)
@pytest.mark.parametrize(
    ("serializer", "expected_index"),
    [
        pytest.param(serialize_json, 0, id="text"),
        pytest.param(serialize_json_as_obj, 1, id="intermediate"),
    ],
)
def test_registered_handlers_take_precedence_over_fallback(
    value,
    text_expected,
    intermediate_expected,
    serializer,
    expected_index,
):
    def fallback(unexpected):
        raise AssertionError(f"fallback received supported {type(unexpected).__name__}")

    expected = (text_expected, intermediate_expected)[expected_index]
    assert serializer(value, fallback=fallback) == expected


@pytest.mark.parametrize(
    ("serializer", "expected"),
    [
        pytest.param(
            serialize_json,
            '{"created": "2024-02-29T12:30:00", "path": "report.json"}',
            id="text",
        ),
        pytest.param(
            serialize_json_as_obj,
            {
                "created": "2024-02-29T12:30:00",
                "path": "report.json",
            },
            id="intermediate",
        ),
    ],
)
def test_fallback_result_is_recursively_normalized(serializer, expected):
    unsupported = UnsupportedValue()

    def fallback(value):
        assert value is unsupported
        return {
            "created": datetime(2024, 2, 29, 12, 30),
            "path": Path("report.json"),
        }

    options = {"sort_keys": True} if serializer is serialize_json else {}
    assert serializer(unsupported, fallback=fallback, **options) == expected


@pytest.mark.parametrize("serializer", [serialize_json, serialize_json_as_obj])
def test_identity_fallback_fails_predictably(serializer):
    unsupported = UnsupportedValue()

    with pytest.raises(ValueError, match="(?i)fallback") as caught:
        serializer(unsupported, fallback=lambda value: value)

    assert "FALLBACK_PAYLOAD_MUST_NOT_LEAK" not in str(caught.value)


@pytest.mark.parametrize("serializer", [serialize_json, serialize_json_as_obj])
def test_fallback_created_cycle_fails_predictably(serializer):
    unsupported = UnsupportedValue()
    cyclic_result = []
    cyclic_result.append(cyclic_result)

    with pytest.raises(ValueError, match="(?i)cyc") as caught:
        serializer(unsupported, fallback=lambda value: cyclic_result)

    assert "FALLBACK_PAYLOAD_MUST_NOT_LEAK" not in str(caught.value)

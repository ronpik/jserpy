"""Conversion handlers (Literal, Enum, Decimal, datetime, bytes, NumPy scalars) reject invalid input with a TypeError.

The message names types only, never the value, and the original error (which often echoes the value) is neither
chained nor kept as `__context__`. Being a TypeError, it lets unions try their next member.
"""

import datetime
import enum
import json
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal, Optional

import pytest

from jserpy import deserialize_json, serialize_json

SENTINEL = "CONV_PAYLOAD_MUST_NOT_LEAK"


class Color(enum.Enum):
    RED = "red"
    GREEN = "green"


class Level(enum.IntEnum):
    LOW = 1
    HIGH = 2


@dataclass
class Record:
    color: Color
    maybe_color: Optional[Color] = None
    kind: Literal["a", "b"] = "a"
    amount: Decimal = Decimal("0")
    when: datetime.datetime = datetime.datetime(2026, 1, 1)
    blob: bytes = b""


def _chained_texts(error: BaseException) -> str:
    texts = []
    current: BaseException | None = error
    while current is not None:
        texts += [str(current), repr(current)]
        current = current.__cause__ or current.__context__
    return "\n".join(texts)


def _rejected(data, annotation) -> TypeError:
    with pytest.raises(TypeError) as caught:
        deserialize_json(data, annotation)
    return caught.value


def _round_trip(value, annotation):
    return deserialize_json(json.loads(serialize_json(value)), annotation)


INVALID_INPUTS = [
    pytest.param(SENTINEL, Literal["a", "b"], id="literal-str"),
    pytest.param([SENTINEL], Literal["a", "b"], id="literal-list"),
    pytest.param(SENTINEL, Color, id="enum-str"),
    pytest.param([SENTINEL], Color, id="enum-list"),
    pytest.param({SENTINEL: SENTINEL}, Color, id="enum-dict"),
    pytest.param(SENTINEL, Level, id="int-enum-str"),
    pytest.param(SENTINEL, Decimal, id="decimal-str"),
    pytest.param([SENTINEL, [1], 0], Decimal, id="decimal-list"),
    pytest.param(SENTINEL, datetime.datetime, id="datetime-str"),
    pytest.param(SENTINEL, datetime.date, id="date-str"),
    pytest.param(SENTINEL, datetime.time, id="time-str"),
    pytest.param([SENTINEL], datetime.datetime, id="datetime-list"),
    # not ASCII: the original ValueError is raised while a UnicodeEncodeError holding the string is handled
    pytest.param("é" + SENTINEL, bytes, id="bytes-non-ascii"),
    pytest.param(SENTINEL, bytes, id="bytes-bad-padding"),
    pytest.param([SENTINEL], bytes, id="bytes-list"),
    pytest.param({SENTINEL: 1}, Path, id="path-dict"),
]


@pytest.mark.parametrize("data, annotation", INVALID_INPUTS)
def test_invalid_value_message_names_types_only(data, annotation):
    error = _rejected(data, annotation)

    assert SENTINEL not in _chained_texts(error)
    assert type(data).__name__ in str(error)


@pytest.mark.parametrize(
    "data, annotation",
    [param for param in INVALID_INPUTS if param.values[1] not in (Literal["a", "b"], Path)],
)
def test_original_conversion_error_is_dropped(data, annotation):
    error = _rejected(data, annotation)

    assert error.__cause__ is None
    assert error.__context__ is None


def test_messages():
    assert str(_rejected("z", Literal["a", "b"])) == "Expected one of the allowed literals ('a', 'b'), got str"
    assert str(_rejected("blue", Color)) == "Invalid value for <enum 'Color'>, got str"
    assert str(_rejected(3, Level)) == "Invalid value for <enum 'Level'>, got int"
    assert str(_rejected("x", Decimal)) == "Invalid value for <class 'decimal.Decimal'>, got str"
    assert str(_rejected("x", datetime.date)) == "Invalid value for <class 'datetime.date'>, got str"
    assert str(_rejected("abc", bytes)) == "Invalid value for <class 'bytes'>, got str"


@pytest.mark.parametrize(
    "annotation",
    [Optional[Color], Color | None, Optional[Level], Optional[Decimal], Optional[datetime.date], Optional[bytes]],
)
def test_optional_accepts_null(annotation):
    assert deserialize_json(None, annotation) is None


@pytest.mark.parametrize(
    "annotation",
    [
        Optional[Color],
        Color | None,
        Literal["a", "b"] | None,
        Optional[Decimal],
        Optional[datetime.datetime],
        Optional[bytes],
    ],
)
def test_optional_rejects_invalid_value_without_leaking(annotation):
    error = _rejected("é" + SENTINEL, annotation)

    assert SENTINEL not in _chained_texts(error)


@pytest.mark.parametrize(
    "annotation",
    [Color | str, Literal["a", "b"] | str, Decimal | str, datetime.datetime | str, datetime.date | str, bytes | str],
)
def test_union_falls_through_to_next_member(annotation):
    assert deserialize_json("é" + SENTINEL, annotation) == "é" + SENTINEL


def test_union_still_prefers_first_matching_member():
    assert deserialize_json("red", Color | str) is Color.RED
    assert deserialize_json("1.20", Decimal | str) == Decimal("1.20")
    assert deserialize_json("2026-10-27", datetime.date | str) == datetime.date(2026, 10, 27)


VALID_RECORD = {
    "color": "red",
    "maybe_color": "green",
    "kind": "b",
    "amount": "2.50",
    "when": "2026-10-27T00:00:00",
    "blob": "AQ==",
}


@pytest.mark.parametrize("field_name", list(VALID_RECORD))
def test_dataclass_field_rejects_invalid_value_without_leaking(field_name):
    error = _rejected({**VALID_RECORD, field_name: "é" + SENTINEL}, Record)

    assert SENTINEL not in _chained_texts(error)


def test_dataclass_optional_enum_field_accepts_null_and_missing():
    assert deserialize_json({**VALID_RECORD, "maybe_color": None}, Record).maybe_color is None
    missing = {key: value for key, value in VALID_RECORD.items() if key != "maybe_color"}
    assert deserialize_json(missing, Record).maybe_color is None


def test_invalid_enum_mapping_key_is_rejected_without_leaking():
    error = _rejected({SENTINEL: 1}, dict[Color, int])

    assert SENTINEL not in _chained_texts(error)
    assert error.__context__ is None


@pytest.mark.parametrize(
    "value, annotation",
    [
        pytest.param(Color.GREEN, Color, id="enum"),
        pytest.param(Level.HIGH, Level, id="int-enum"),
        pytest.param("b", Literal["a", "b"], id="literal"),
        pytest.param(Decimal("1.20"), Decimal, id="decimal"),
        pytest.param(datetime.datetime(2026, 10, 27, 12, 30, tzinfo=datetime.timezone.utc), datetime.datetime,
                     id="datetime"),
        pytest.param(datetime.date(2026, 10, 27), datetime.date, id="date"),
        pytest.param(datetime.time(12, 30, 15), datetime.time, id="time"),
        pytest.param(b"\x00\xffJSerPy", bytes, id="bytes"),
        pytest.param({Color.RED: 1}, dict[Color, int], id="enum-key"),
        pytest.param(Record(Color.GREEN, Color.RED, "b", Decimal("2.50"), datetime.datetime(2026, 10, 27), b"\x01"),
                     Record, id="dataclass"),
    ],
)
def test_valid_values_still_round_trip(value, annotation):
    restored = _round_trip(value, annotation)

    assert restored == value
    assert type(restored) is type(value)


def test_lenient_inputs_are_still_accepted():
    assert deserialize_json(2, Level) is Level.HIGH
    assert deserialize_json(3, Decimal) == Decimal(3)
    assert deserialize_json(1.5, Decimal) == Decimal("1.5")


# --- NumPy scalars ------------------------------------------------------------------------------------------------


def test_numpy_scalar_invalid_value_names_types_only():
    np = pytest.importorskip("numpy")
    big = 98765432109876
    for data, annotation in [
        (SENTINEL, np.int64),
        (SENTINEL, np.float64),
        ([SENTINEL], np.int64),
        ({SENTINEL: 1}, np.int64),
        (SENTINEL, np.datetime64),
        (big, np.int8),
    ]:
        error = _rejected(data, annotation)
        texts = _chained_texts(error)
        assert SENTINEL not in texts and str(big) not in texts, annotation
        assert error.__context__ is None
        assert str(error) == f"Invalid value for {annotation}, got {type(data).__name__}"


def test_numpy_scalar_union_falls_through():
    np = pytest.importorskip("numpy")

    assert deserialize_json(SENTINEL, np.int64 | str) == SENTINEL
    # an OverflowError before
    assert deserialize_json(1000, np.int8 | int) == 1000
    restored = deserialize_json(7, np.int64 | str)
    assert restored == 7 and type(restored) is np.int64

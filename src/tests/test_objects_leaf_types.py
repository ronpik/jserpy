"""Leaf (scalar-like) values, from trivial to awkward: numbers, strings, dates, Decimal, bytes, paths and enums.

Every leaf is checked at the root and nested in the usual places (dataclass field, Optional, list and tuple item,
dict value, next to a null, a dict of lists of dataclasses). Each check is two-sided: the JSON the value produces (exact
text where the format is the contract) and a strict round trip that compares exact types (1 vs 1.0 vs True, Decimal
exponents, -0.0). Further sections cover dict and tuple keys, sets, unions, encoding options and wide records.

Tests named ``*_limitation`` pin what the library does today where it is lenient or lossy; that is current behavior, not
a promise. Strict xfails mark behavior a user would reasonably expect and the library does not deliver yet.
"""

import base64
import decimal
import ipaddress
import json
import math
import os
import re
import uuid
from dataclasses import dataclass, make_dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from decimal import Decimal
from enum import Enum, Flag, IntEnum, IntFlag, auto
from fractions import Fraction
from functools import lru_cache
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any, Optional

import pytest

from jserpy import deserialize_json, serialize_json, serialize_json_as_obj
from objects_support import assert_round_trip, assert_strictly_equal, to_json_value

UTC_TZ = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))
NST = timezone(timedelta(hours=-3, minutes=-30))


# --- fixtures: enums and dataclasses ------------------------------------------------------------------------------


class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class StrColor(str, Enum):
    RED = "red"


class Level(IntEnum):
    LOW = 1
    HIGH = 2


class Signed(IntEnum):
    NEGATIVE = -1
    ZERO = 0


class Ratio(Enum):
    HALF = 0.5
    WHOLE = 1.0


class Switch(Enum):
    ON = True
    OFF = False


class Mixed(Enum):
    """Members of unrelated JSON types; True/1/1.0 are left out because they would be aliases of each other."""

    TEXT = "t"
    EMPTY = ""
    NUMBER = 7
    NEGATIVE = -3
    HUGE = 2**70
    REAL = 2.5
    NOTHING = None


class Maybe(Enum):
    NOTHING = None
    SOMETHING = "something"


class Counter(Enum):
    FIRST = auto()
    SECOND = auto()
    THIRD = auto()


class Alias(Enum):
    PRIMARY = "p"
    DEFAULT = "p"
    SECONDARY = "s"


class Perm(Flag):
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()


class PermInt(IntFlag):
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()


class Greeting(Enum):
    """Values that need escaping in JSON text."""

    HELLO = "\u05e9\u05dc\u05d5\u05dd"
    EMOJI = "\U0001F600"
    QUOTE = 'say "hi"'


class Bits(int, Enum):
    """An int-mixin enum that is not an IntEnum."""

    ONE = 1
    TWO = 2


class Scale(float, Enum):
    HALF = 0.5
    WHOLE = 1.0


class LooseColor(Enum):
    """``_missing_`` widens what the lookup by value accepts (Python's own Enum feature)."""

    RED = "red"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            return cls.__members__.get(value.upper())
        return None


class Grid(Enum):
    SQUARE = (1, 1)
    WIDE = (2, 1)


class Palette(Enum):
    WARM = ["red", "orange"]
    COOL = ["blue"]


class Schedule(Enum):
    LAUNCH = date(2024, 2, 29)
    REVIEW = date(2024, 3, 31)


class Price(Enum):
    LOW = Decimal("1.50")
    HIGH = Decimal("20.00")


class FakeDatetime(datetime):
    """A datetime subclass, like the ones freezegun and pandas provide."""


class FakeDate(date):
    pass


class FakeTime(time):
    pass


class Money(Decimal):
    pass


class Blob(bytes):
    pass


class Label(str):
    pass


class Count(int):
    pass


class Weight(float):
    pass


class ProjectPath(type(Path())):
    """A subclass of the concrete Path of the running platform (Path itself is not subclassable before 3.12)."""


@dataclass
class IntBox:
    n: int


@dataclass
class DateBox:
    day: date


@dataclass
class AnyBox:
    value: Any


@dataclass
class AnyBag:
    a: Any
    b: Any
    c: Any
    d: Any
    e: Any


@dataclass
class ScalarRecord:
    i: int
    f: float
    b: bool
    s: str


@dataclass
class NoneField:
    x: None = None


@dataclass
class DatedText:
    label: str
    day: date
    code: str
    price_text: str


@dataclass
class Prices:
    by_color: dict[Color, list[Decimal]]
    by_name: dict[str, dict[StrColor, date]]


@dataclass
class LeafRecord:
    count: int
    ratio: float
    active: bool
    name: str
    note: Optional[str]
    created: datetime
    day: date
    at: time
    price: Decimal
    blob: bytes
    path: Path
    color: Color
    level: Level
    extra: Any


@dataclass
class Ledger:
    entries: list[LeafRecord]
    by_name: dict[str, LeafRecord]
    latest: Optional[LeafRecord] = None


@lru_cache(maxsize=None)
def _box(annotation):
    # The annotation is a real object, never a string, so building the class at run time is safe for get_type_hints.
    return make_dataclass("Box", [("value", annotation)])


@lru_cache(maxsize=None)
def _optional_box(annotation):
    return make_dataclass("OptionalBox", [("value", Optional[annotation])])


# --- placements: where a leaf value sits -------------------------------------------------------------------------
# Each builder takes the leaf value, its annotation and its JSON form and returns (object, annotation, JSON form).


def _place_root(value, annotation, j):
    return value, annotation, j


def _place_field(value, annotation, j):
    box = _box(annotation)
    return box(value), box, {"value": j}


def _place_optional_root(value, annotation, j):
    return value, Optional[annotation], j


def _place_optional_field(value, annotation, j):
    box = _optional_box(annotation)
    return box(value), box, {"value": j}


def _place_list_items(value, annotation, j):
    return [value, value], list[annotation], [j, j]


def _place_tuple_items(value, annotation, j):
    return (value, value), tuple[annotation, annotation], [j, j]


def _place_dict_value(value, annotation, j):
    return {"k": value}, dict[str, annotation], {"k": j}


def _place_nested(value, annotation, j):
    box = _box(annotation)
    return {"outer": [box(value), box(value)]}, dict[str, list[box]], {"outer": [{"value": j}, {"value": j}]}


def _place_optional_items(value, annotation, j):
    return [value, None, value], list[Optional[annotation]], [j, None, j]


def _place_optional_dict_values(value, annotation, j):
    return {"a": None, "b": value}, dict[str, Optional[annotation]], {"a": None, "b": j}


PLACEMENTS = {
    "root": _place_root,
    "field": _place_field,
    "optional-root": _place_optional_root,
    "optional-field": _place_optional_field,
    "list-items": _place_list_items,
    "tuple-items": _place_tuple_items,
    "dict-value": _place_dict_value,
    "nested": _place_nested,
}

# Placements that put a null next to the value
NULLABLE_PLACEMENTS = {
    "optional-items": _place_optional_items,
    "optional-dict-values": _place_optional_dict_values,
}


def _case(case_id, value, annotation, text):
    return pytest.param(value, annotation, text, id=case_id)


# --- the catalog of leaf values: (value, annotation, exact JSON text under default options) ------------------------

INT_CASES = [
    _case("int-zero", 0, int, "0"),
    _case("int-one", 1, int, "1"),
    _case("int-minus-one", -1, int, "-1"),
    _case("int-2**31-1", 2**31 - 1, int, "2147483647"),
    _case("int-2**31", 2**31, int, "2147483648"),
    _case("int-minus-2**63", -(2**63), int, "-9223372036854775808"),
    _case("int-2**63", 2**63, int, "9223372036854775808"),
    _case("int-2**64-plus-1", 2**64 + 1, int, "18446744073709551617"),
    _case("int-minus-2**100", -(2**100), int, "-1267650600228229401496703205376"),
    _case("int-10**40", 10**40, int, "1" + "0" * 40),
]

FLOAT_CASES = [
    _case("float-zero", 0.0, float, "0.0"),
    _case("float-negative-zero", -0.0, float, "-0.0"),
    _case("float-tenth", 0.1, float, "0.1"),
    _case("float-whole", 1.0, float, "1.0"),
    _case("float-negative", -1.5, float, "-1.5"),
    _case("float-1e15", 1e15, float, "1000000000000000.0"),
    _case("float-1e16", 1e16, float, "1e+16"),
    _case("float-1e-4", 1e-4, float, "0.0001"),
    _case("float-1e-5", 1e-5, float, "1e-05"),
    _case("float-1e-7", 1e-7, float, "1e-07"),
    _case("float-2.5e-8", 2.5e-8, float, "2.5e-08"),
    _case("float-1e-300", 1e-300, float, "1e-300"),
    _case("float-1e22", 1e22, float, "1e+22"),
    _case("float-1e300", 1e300, float, "1e+300"),
    _case("float-hundred", 100.0, float, "100.0"),
    _case("float-123456.789", 123456.789, float, "123456.789"),
    _case("float-2**53", float(2**53), float, "9007199254740992.0"),
    # the shortest text that reads back as the same double, which is not "0.3"
    _case("float-sum-of-tenths", 0.1 + 0.2, float, "0.30000000000000004"),
    _case("float-smallest-subnormal", 5e-324, float, "5e-324"),
    _case("float-negative-smallest-subnormal", -5e-324, float, "-5e-324"),
    _case("float-max", 1.7976931348623157e308, float, "1.7976931348623157e+308"),
    _case("float-nan", float("nan"), float, "NaN"),
    _case("float-inf", float("inf"), float, "Infinity"),
    _case("float-negative-inf", float("-inf"), float, "-Infinity"),
]

BOOL_CASES = [
    _case("bool-true", True, bool, "true"),
    _case("bool-false", False, bool, "false"),
]

# (id, value, text with ensure_ascii=True)
STRING_CASES = [
    ("empty", "", '""'),
    ("plain", "plain", '"plain"'),
    ("looks-like-null", "null", '"null"'),
    ("looks-like-number", "123", '"123"'),
    ("quote", 'say "hi"', '"say \\"hi\\""'),
    ("backslash", "a\\b", '"a\\\\b"'),
    ("newline-and-tab", "a\nb\tc", '"a\\nb\\tc"'),
    ("slash-is-not-escaped", "a/b", '"a/b"'),
    ("single-quote", "it's", '"it\'s"'),
    ("nul", "a\x00b", '"a\\u0000b"'),
    ("delete-character", "\x7f", '"\\u007f"'),
    ("line-separator", "\u2028", '"\\u2028"'),
    ("emoji-outside-the-bmp", "\U0001F600", '"\\ud83d\\ude00"'),
    ("hebrew", "\u05e9\u05dc\u05d5\u05dd", '"\\u05e9\\u05dc\\u05d5\\u05dd"'),
    ("arabic", "\u0645\u0631\u062d\u0628\u0627", '"\\u0645\\u0631\\u062d\\u0628\\u0627"'),
    ("cjk", "\u4f60\u597d", '"\\u4f60\\u597d"'),
    ("combining-mark", "e\u0301", '"e\\u0301"'),
    ("precomposed", "\u00e9", '"\\u00e9"'),
    ("lone-high-surrogate", "\ud800", '"\\ud800"'),
    ("lone-low-surrogate", "\udfff", '"\\udfff"'),
    ("surrogates-in-reverse-order", "\ude00\ud83d", '"\\ude00\\ud83d"'),
    ("whitespace-only", " \t\n ", '" \\t\\n "'),
    ("next-line", "\x85", '"\\u0085"'),
    ("no-break-space", "\xa0", '"\\u00a0"'),
    ("byte-order-mark", "\ufeff", '"\\ufeff"'),
    ("noncharacter", "\uffff", '"\\uffff"'),
    # U+10FFFF is the surrogate pair D800+0x3FF, DC00+0x3FF
    ("highest-code-point", "\U0010ffff", '"\\udbff\\udfff"'),
    ("zero-width-joiner-sequence", "\U0001F468\u200d\U0001F469\u200d\U0001F467",
     '"\\ud83d\\udc68\\u200d\\ud83d\\udc69\\u200d\\ud83d\\udc67"'),
    ("looks-like-iso-date", "2024-02-29", '"2024-02-29"'),
    ("looks-like-base64", "YWJj", '"YWJj"'),
    ("looks-like-decimal", "1.20", '"1.20"'),
    ("100k-ascii", "x" * 100_000, '"' + "x" * 100_000 + '"'),
    ("100k-escapes", "ab\n\u05e9" * 25_000, '"' + "ab\\n\\u05e9" * 25_000 + '"'),
]

STRING_PARAMS = [_case(f"str-{case_id}", value, str, text) for case_id, value, text in STRING_CASES]

# (id, value, text)
DATETIME_CASES = [
    ("naive", datetime(2024, 2, 29, 12, 34, 56), '"2024-02-29T12:34:56"'),
    ("naive-midnight", datetime(2024, 1, 1), '"2024-01-01T00:00:00"'),
    ("naive-one-microsecond", datetime(2024, 2, 29, 12, 34, 56, 1), '"2024-02-29T12:34:56.000001"'),
    ("naive-milliseconds", datetime(2024, 2, 29, 12, 34, 56, 789000), '"2024-02-29T12:34:56.789000"'),
    ("utc", datetime(2024, 2, 29, 12, 34, 56, tzinfo=UTC_TZ), '"2024-02-29T12:34:56+00:00"'),
    ("positive-offset", datetime(2024, 2, 29, 12, 34, 56, tzinfo=IST), '"2024-02-29T12:34:56+05:30"'),
    ("negative-offset", datetime(2024, 2, 29, 12, 34, 56, tzinfo=NST), '"2024-02-29T12:34:56-03:30"'),
    ("offset-with-microseconds", datetime(2024, 2, 29, 12, 34, 56, 123456, tzinfo=IST),
     '"2024-02-29T12:34:56.123456+05:30"'),
    ("offset-with-seconds",
     datetime(2024, 2, 29, 12, 34, 56, tzinfo=timezone(timedelta(hours=5, minutes=30, seconds=15))),
     '"2024-02-29T12:34:56+05:30:15"'),
    ("offset-plus-14", datetime(2024, 2, 29, 12, 34, 56, tzinfo=timezone(timedelta(hours=14))),
     '"2024-02-29T12:34:56+14:00"'),
    ("offset-minus-12", datetime(2024, 2, 29, tzinfo=timezone(timedelta(hours=-12))), '"2024-02-29T00:00:00-12:00"'),
    ("year-1", datetime(1, 1, 1), '"0001-01-01T00:00:00"'),
    ("year-1-with-offset", datetime(1, 1, 1, tzinfo=IST), '"0001-01-01T00:00:00+05:30"'),
    ("year-9999", datetime(9999, 12, 31, 23, 59, 59, 999999), '"9999-12-31T23:59:59.999999"'),
    ("year-9999-utc", datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=UTC_TZ), '"9999-12-31T23:59:59.999999+00:00"'),
]

DATE_CASES = [
    ("date", date(2024, 2, 29), '"2024-02-29"'),
    ("date-year-1", date.min, '"0001-01-01"'),
    ("date-year-9999", date.max, '"9999-12-31"'),
    ("date-last-day-of-a-leap-year", date(2024, 12, 31), '"2024-12-31"'),
]

TIME_CASES = [
    ("time", time(12, 34, 56), '"12:34:56"'),
    ("time-midnight", time(0, 0), '"00:00:00"'),
    ("time-microsecond", time(12, 34, 56, 1), '"12:34:56.000001"'),
    ("time-one-microsecond-after-midnight", time(0, 0, 0, 1), '"00:00:00.000001"'),
    ("time-max", time.max, '"23:59:59.999999"'),
    ("time-utc", time(12, 34, 56, tzinfo=UTC_TZ), '"12:34:56+00:00"'),
    ("time-offset-with-microseconds", time(12, 34, 56, 250000, tzinfo=IST), '"12:34:56.250000+05:30"'),
]

DATETIME_PARAMS = [
    _case(f"datetime-{case_id}", value, datetime, text) for case_id, value, text in DATETIME_CASES
] + [
    _case(case_id, value, date, text) for case_id, value, text in DATE_CASES
] + [
    _case(case_id, value, time, text) for case_id, value, text in TIME_CASES
]

# (id, constructor text, JSON string): the JSON string is Decimal's own canonical formatting
DECIMAL_CASES = [
    ("zero", "0", "0"),
    ("negative-zero", "-0", "-0"),
    ("zero-with-scale", "0.00", "0.00"),
    ("zero-with-exponent", "0E+2", "0E+2"),
    ("trailing-zero", "1.20", "1.20"),
    ("negative-trailing-zero", "-1.50", "-1.50"),
    ("positive-exponent", "1E+3", "1E+3"),
    ("plain-thousand", "1000", "1000"),
    ("exponent-spelled-out-of-range", "10E-1", "1.0"),
    ("six-places", "0.000001", "0.000001"),
    ("seven-places-goes-scientific", "0.0000001", "1E-7"),
    ("tiny-exponent", "1E-10", "1E-10"),
    ("huge-exponent", "1E+1000", "1E+1000"),
    ("minuscule-exponent", "1E-1000", "1E-1000"),
    ("scaled-integer", "12345E-2", "123.45"),
    ("38-digits", "1234567890123456789012345678.9012345678", "1234567890123456789012345678.9012345678"),
    ("40-digit-integer-negative", "-9999999999999999999999999999999999999999",
     "-9999999999999999999999999999999999999999"),
    ("nan", "NaN", "NaN"),
    ("signaling-nan", "sNaN", "sNaN"),
    ("negative-nan", "-NaN", "-NaN"),
    ("nan-with-payload", "NaN123", "NaN123"),
    ("infinity", "Infinity", "Infinity"),
    ("negative-infinity", "-Infinity", "-Infinity"),
]

DECIMAL_PARAMS = [
    _case(f"decimal-{case_id}", Decimal(literal), Decimal, json.dumps(text)) for case_id, literal, text in DECIMAL_CASES
]

ALL_BYTES = bytes(range(256))
HUNDRED_KB = ALL_BYTES * 400


def _base64_text(data):
    return json.dumps(base64.b64encode(data).decode("ascii"))


BYTES_PARAMS = [
    _case("bytes-empty", b"", bytes, '""'),
    _case("bytes-one", b"a", bytes, '"YQ=="'),
    _case("bytes-two", b"ab", bytes, '"YWI="'),
    _case("bytes-three", b"abc", bytes, '"YWJj"'),
    _case("bytes-four", b"abcd", bytes, '"YWJjZA=="'),
    _case("bytes-nul", b"\x00", bytes, '"AA=="'),
    _case("bytes-space", b" ", bytes, '"IA=="'),
    _case("bytes-three-nul", b"\x00\x00\x00", bytes, '"AAAA"'),
    _case("bytes-three-0xff", b"\xff\xff\xff", bytes, '"////"'),
    _case("bytes-utf8-text", "\u05e9\u05dc\u05d5\u05dd".encode("utf-8"), bytes,
          _base64_text("\u05e9\u05dc\u05d5\u05dd".encode("utf-8"))),
    _case("bytes-0xff", b"\xff", bytes, '"/w=="'),
    # index 62 is "+" and 63 is "/": the standard alphabet, not the URL-safe one
    _case("bytes-plus-and-slash", b"\xfb\xef\xbe\xff\xff\xff", bytes, '"++++////"'),
    _case("bytes-all-256-values", ALL_BYTES, bytes, _base64_text(ALL_BYTES)),
    _case("bytes-100kb", HUNDRED_KB, bytes, _base64_text(HUNDRED_KB)),
]


def _path_text(*parts):
    return json.dumps(os.sep.join(parts))


# Path objects of the running platform: only separator-free text is hard-coded, the rest is joined with os.sep
PATH_PARAMS = [
    _case("path-file", Path("report.json"), Path, '"report.json"'),
    _case("path-empty-becomes-dot", Path(""), Path, '"."'),
    _case("path-dot", Path("."), Path, '"."'),
    _case("path-spaces", Path("with space.txt"), Path, '"with space.txt"'),
    _case("path-unicode", Path("\u05d3\u05d5\u05d2\u05de\u05d4"), Path, '"\\u05d3\\u05d5\\u05d2\\u05de\\u05d4"'),
    _case("path-nested", Path("a/b/c.txt"), Path, _path_text("a", "b", "c.txt")),
    _case("path-trailing-slash-dropped", Path("a/b/"), Path, _path_text("a", "b")),
    _case("path-dot-segments-collapse", Path("./a/./b"), Path, _path_text("a", "b")),
    _case("path-parent-segment-kept", Path("../a"), Path, _path_text("..", "a")),
    _case("path-parent-segment-in-the-middle-kept", Path("a/../b"), Path, _path_text("a", "..", "b")),
    _case("path-double-slash-collapses", Path("a//b"), Path, _path_text("a", "b")),
    _case("path-tilde-is-not-expanded", Path("~/x"), Path, _path_text("~", "x")),
    _case("path-hidden-file", Path(".hidden"), Path, '".hidden"'),
    _case("path-trailing-space-kept", Path("a "), Path, '"a "'),
    _case("path-quotes-in-name", Path('say "hi".txt'), Path, json.dumps('say "hi".txt')),
    _case("path-newline-in-name", Path("a\nb"), Path, json.dumps("a\nb")),
    _case("path-hundred-segments", Path(*["seg"] * 100), Path, _path_text(*["seg"] * 100)),
    _case("path-absolute", Path("/usr/local/bin"), Path, _path_text("", "usr", "local", "bin")),
    _case("path-root", Path(os.sep), Path, json.dumps(os.sep)),
]

ENUM_PARAMS = [
    _case("enum-str", Color.GREEN, Color, '"green"'),
    _case("enum-str-mixin", StrColor.RED, StrColor, '"red"'),
    _case("enum-int", Level.HIGH, Level, "2"),
    _case("enum-int-negative", Signed.NEGATIVE, Signed, "-1"),
    _case("enum-int-zero", Signed.ZERO, Signed, "0"),
    _case("enum-float", Ratio.HALF, Ratio, "0.5"),
    _case("enum-float-whole", Ratio.WHOLE, Ratio, "1.0"),
    _case("enum-true", Switch.ON, Switch, "true"),
    _case("enum-false", Switch.OFF, Switch, "false"),
    _case("enum-mixed-str", Mixed.TEXT, Mixed, '"t"'),
    _case("enum-mixed-empty-str", Mixed.EMPTY, Mixed, '""'),
    _case("enum-mixed-int", Mixed.NUMBER, Mixed, "7"),
    _case("enum-mixed-negative-int", Mixed.NEGATIVE, Mixed, "-3"),
    _case("enum-mixed-huge-int", Mixed.HUGE, Mixed, "1180591620717411303424"),
    _case("enum-mixed-float", Mixed.REAL, Mixed, "2.5"),
    _case("enum-unicode-value", Greeting.HELLO, Greeting, '"\\u05e9\\u05dc\\u05d5\\u05dd"'),
    _case("enum-emoji-value", Greeting.EMOJI, Greeting, '"\\ud83d\\ude00"'),
    _case("enum-quote-value", Greeting.QUOTE, Greeting, '"say \\"hi\\""'),
    _case("enum-int-mixin", Bits.TWO, Bits, "2"),
    _case("enum-float-mixin", Scale.HALF, Scale, "0.5"),
    _case("enum-float-mixin-whole", Scale.WHOLE, Scale, "1.0"),
    _case("enum-auto", Counter.THIRD, Counter, "3"),
    _case("enum-alias", Alias.DEFAULT, Alias, '"p"'),
    _case("enum-flag-combination", Perm.READ | Perm.EXECUTE, Perm, "5"),
    _case("enum-flag-empty", Perm(0), Perm, "0"),
    _case("enum-int-flag-combination", PermInt.READ | PermInt.WRITE, PermInt, "3"),
    _case("enum-list-value", Palette.WARM, Palette, '["red", "orange"]'),
]

LEAF_CASES = (
    INT_CASES + FLOAT_CASES + BOOL_CASES + STRING_PARAMS + DATETIME_PARAMS + DECIMAL_PARAMS + BYTES_PARAMS
    + PATH_PARAMS + ENUM_PARAMS
)

# Mixed and Maybe have a None-valued member: next to it a null is ambiguous (see the _limitation test about it)
NON_NULLABLE_LEAF_CASES = [case for case in LEAF_CASES if case.values[1] not in (Mixed, Maybe)]

FINITE_LEAF_CASES = [
    case for case in LEAF_CASES if not (isinstance(case.values[0], float) and not math.isfinite(case.values[0]))
]

# One entry per distinct annotation, for tests that pair every leaf type with None (Mixed and Maybe have a None-valued
# member, which makes null ambiguous: see the _limitation test about it)
LEAF_ANNOTATIONS = [
    pytest.param(annotation, id=annotation.__name__)
    for annotation in (int, float, bool, str, datetime, date, time, Decimal, bytes, Path, Color, Level, Switch)
]


# --- the catalog, in every placement ------------------------------------------------------------------------------


@pytest.mark.parametrize(("value", "annotation", "text"), LEAF_CASES)
def test_leaf_root_text_is_exact(value, annotation, text):
    assert serialize_json(value) == text
    assert json.dumps(serialize_json_as_obj(value)) == json.dumps(json.loads(text))


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize(("value", "annotation", "text"), LEAF_CASES)
def test_leaf_json_shape_in_every_placement(value, annotation, text, placement):
    obj, _, expected = PLACEMENTS[placement](value, annotation, json.loads(text))

    # json.dumps keeps 1, 1.0 and true apart, which == would not
    assert json.dumps(to_json_value(obj)) == json.dumps(expected)


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize(("value", "annotation", "text"), LEAF_CASES)
def test_leaf_round_trip_in_every_placement(value, annotation, text, placement):
    obj, cls, _ = PLACEMENTS[placement](value, annotation, json.loads(text))

    assert_round_trip(obj, cls)


@pytest.mark.parametrize("placement", NULLABLE_PLACEMENTS)
@pytest.mark.parametrize(("value", "annotation", "text"), NON_NULLABLE_LEAF_CASES)
def test_null_next_to_a_leaf_keeps_both(value, annotation, text, placement):
    obj, cls, expected = NULLABLE_PLACEMENTS[placement](value, annotation, json.loads(text))

    assert json.dumps(to_json_value(obj)) == json.dumps(expected)
    assert_round_trip(obj, cls)


# Encoding options never change what a value restores to, only how it is written
OPTION_SETS = [
    pytest.param({"ensure_ascii": False}, id="unicode"),
    pytest.param({"indent": 2, "sort_keys": True}, id="indented-and-sorted"),
    pytest.param({"separators": (",", ":")}, id="compact"),
    pytest.param({"allow_nan": False}, id="no-nan"),
]


@pytest.mark.parametrize("options", OPTION_SETS)
@pytest.mark.parametrize(("value", "annotation", "text"), FINITE_LEAF_CASES)
def test_leaf_round_trip_under_encoding_options(value, annotation, text, options):
    obj, cls, _ = _place_nested(value, annotation, json.loads(text))

    assert_round_trip(obj, cls, **options)


@pytest.mark.parametrize("annotation", LEAF_ANNOTATIONS)
def test_optional_field_holds_none_next_to_every_leaf_type(annotation):
    box = _optional_box(annotation)

    assert serialize_json(box(None)) == '{"value": null}'
    restored = assert_round_trip(box(None), box)
    assert restored.value is None


# --- JSON primitives ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "negative-inf"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_allow_nan_false_rejects_non_finite_floats_in_every_placement(value, placement):
    obj, _, _ = PLACEMENTS[placement](value, float, None)

    with pytest.raises(ValueError, match="not JSON compliant"):
        serialize_json(obj, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "negative-inf"])
@pytest.mark.parametrize("wrap", [lambda key: key, lambda key: ("a", key)], ids=["plain-key", "inside-tuple-key"])
def test_allow_nan_false_rejects_non_finite_float_keys(value, wrap):
    with pytest.raises(ValueError, match="not JSON compliant"):
        serialize_json({wrap(value): 1}, allow_nan=False)


@pytest.mark.parametrize("value", [-0.0, 5e-324, 1e300, 1.7976931348623157e308], ids=str)
def test_allow_nan_false_keeps_extreme_finite_floats(value):
    assert serialize_json([value], allow_nan=False) == serialize_json([value])
    assert_round_trip([value], list[float], allow_nan=False)


def test_non_finite_floats_survive_a_round_trip_with_the_default_allow_nan():
    restored = assert_round_trip([float("nan"), float("inf"), float("-inf")], list[float])

    assert serialize_json(restored) == "[NaN, Infinity, -Infinity]"


@pytest.mark.parametrize(
    ("n", "expected", "text"),
    [
        pytest.param(0, 0.0, "0.0", id="zero"),
        pytest.param(1, 1.0, "1.0", id="one"),
        pytest.param(-3, -3.0, "-3.0", id="negative"),
        pytest.param(2**53, 9007199254740992.0, "9007199254740992.0", id="2**53"),
        pytest.param(2**63, 9223372036854775808.0, "9.223372036854776e+18", id="2**63"),
    ],
)
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_int_valued_json_is_coerced_for_a_float_annotation(placement, n, expected, text):
    # builders take the value the restored object must be and the JSON form it comes from
    expected_obj, cls, json_value = PLACEMENTS[placement](expected, float, n)

    restored = deserialize_json(json_value, cls)

    assert_strictly_equal(restored, expected_obj)
    assert serialize_json(restored) == serialize_json(expected_obj)
    if placement == "root":
        assert serialize_json(restored) == text


def test_float_valued_json_is_not_turned_into_an_int_for_an_int_annotation_limitation():
    # documents current behavior (primitives are passed through, not validated), not a promise
    restored = deserialize_json({"n": 1.0}, IntBox)

    assert type(restored.n) is float
    assert serialize_json(restored) == '{"n": 1.0}'


def test_whole_float_in_an_int_annotated_field_is_written_as_a_float():
    assert serialize_json(IntBox(n=1.0)) == '{"n": 1.0}'
    assert serialize_json(IntBox(n=1)) == '{"n": 1}'
    assert serialize_json(IntBox(n=True)) == '{"n": true}'


def test_bool_is_never_written_as_an_int_and_an_int_never_as_a_bool():
    assert serialize_json([True, 1, False, 0]) == "[true, 1, false, 0]"
    assert serialize_json({"flag": True, "count": 1}) == '{"flag": true, "count": 1}'
    record = ScalarRecord(i=1, f=1.0, b=True, s="1")
    assert serialize_json(record) == '{"i": 1, "f": 1.0, "b": true, "s": "1"}'

    restored = assert_round_trip(record, ScalarRecord)
    assert type(restored.i) is int and type(restored.b) is bool and type(restored.f) is float


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(ScalarRecord(i=1, f=1.0, b=True, s="1"), id="ones"),
        pytest.param(ScalarRecord(i=0, f=0.0, b=False, s=""), id="zeros"),
        pytest.param(ScalarRecord(i=-1, f=-0.0, b=True, s="true"), id="negative-one-and-negative-zero"),
        pytest.param(ScalarRecord(i=2**64 + 1, f=5e-324, b=False, s="0"), id="extremes"),
        pytest.param(ScalarRecord(i=0, f=float("nan"), b=True, s="NaN"), id="nan"),
    ],
)
def test_scalar_record_round_trip_keeps_each_primitive_type(record):
    assert_round_trip(record, ScalarRecord)


@pytest.mark.parametrize(
    ("data", "annotation"),
    [
        pytest.param("x", int, id="str-as-int"),
        pytest.param(1.5, int, id="float-as-int"),
        pytest.param(True, int, id="bool-as-int"),
        pytest.param(1, bool, id="int-as-bool"),
        pytest.param(1, str, id="int-as-str"),
        pytest.param("1", float, id="str-as-float"),
        pytest.param(None, int, id="null-as-int"),
        pytest.param(None, str, id="null-as-str"),
    ],
)
def test_primitives_are_not_validated_against_primitive_annotations_limitation(data, annotation):
    # documents current behavior, not a promise: a JSON primitive is passed through for int, bool, str and float
    restored = deserialize_json(data, annotation)

    assert restored == data
    assert type(restored) is type(data)


def test_float_before_int_in_a_union_coerces_whole_numbers_limitation():
    # documents current behavior, not a promise: the first member that accepts the data wins
    assert type(deserialize_json(3, int | float)) is int
    assert type(deserialize_json(3, float | int)) is float
    assert type(deserialize_json(3.0, int | float)) is float


@pytest.mark.xfail(
    strict=True,
    raises=OverflowError,
    reason="an int too large for a float escapes as OverflowError instead of the TypeError of a data mismatch",
)
@pytest.mark.parametrize(
    "annotation",
    [pytest.param(float, id="float"), pytest.param(float | None, id="optional-float")],
)
def test_huge_int_for_a_float_annotation_is_a_type_error(annotation):
    with pytest.raises(TypeError):
        deserialize_json(10**400, annotation)


@pytest.mark.xfail(
    strict=True,
    raises=OverflowError,
    reason="an int too large for a float escapes as OverflowError, so the union cannot fall through to int",
)
def test_huge_int_falls_through_a_float_union_to_int():
    assert deserialize_json(10**400, float | int) == 10**400


def test_none_is_null_at_every_depth():
    assert serialize_json(None) == "null"
    assert serialize_json([None, {"k": None}]) == '[null, {"k": null}]'
    assert deserialize_json(None, type(None)) is None


def test_none_and_falsy_values_stay_distinct_inside_optional_containers():
    assert serialize_json([None, 0, None, 5]) == "[null, 0, null, 5]"
    assert_round_trip([None, 0, None, 5], list[Optional[int]])
    assert_round_trip({"a": None, "b": ""}, dict[str, Optional[str]])
    assert_round_trip((None, False, 0.0), tuple[Optional[bool], Optional[bool], Optional[float]])
    assert_round_trip({"none": None, "no": False}, dict[str, Optional[bool]])


def test_null_is_rejected_by_the_none_type_only_for_non_null_data():
    with pytest.raises(TypeError, match="Expected null, got int"):
        deserialize_json(0, type(None))


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="an annotation of plain None (a valid spelling of NoneType) reaches issubclass() and fails",
)
@pytest.mark.parametrize(
    ("data", "annotation"),
    [
        pytest.param([None], list[None], id="list-item"),
        pytest.param({"k": None}, dict[str, None], id="dict-value"),
        pytest.param({"x": None}, NoneField, id="dataclass-field"),
    ],
)
def test_plain_none_annotation_matches_null(data, annotation):
    restored = deserialize_json(data, annotation)

    if annotation is NoneField:
        assert restored == NoneField(x=None)
    else:
        assert restored == data


# --- heterogeneous Any: exact types survive ----------------------------------------------------------------------

HETEROGENEOUS = [
    ("true", True),
    ("false", False),
    ("zero", 0),
    ("one", 1),
    ("minus-one", -1),
    ("zero-float", 0.0),
    ("one-float", 1.0),
    ("negative-zero", -0.0),
    ("half", 0.5),
    ("empty-str", ""),
    ("zero-str", "0"),
    ("one-str", "1"),
    ("true-str", "true"),
    ("null-str", "null"),
    ("none", None),
    ("2**63", 2**63),
    ("1e300", 1e300),
    ("nan", float("nan")),
    ("inf", float("inf")),
    ("negative-inf", float("-inf")),
]

HETEROGENEOUS_PARAMS = [pytest.param(value, id=case_id) for case_id, value in HETEROGENEOUS]
HETEROGENEOUS_LIST_TEXT = (
    '[true, false, 0, 1, -1, 0.0, 1.0, -0.0, 0.5, "", "0", "1", "true", "null", null, 9223372036854775808, 1e+300, '
    "NaN, Infinity, -Infinity]"
)


def test_heterogeneous_list_has_exact_text():
    assert serialize_json([value for _, value in HETEROGENEOUS]) == HETEROGENEOUS_LIST_TEXT


def test_heterogeneous_list_keeps_every_type():
    restored = assert_round_trip([value for _, value in HETEROGENEOUS], list[Any])

    assert [type(item) for item in restored] == [type(value) for _, value in HETEROGENEOUS]


def test_heterogeneous_dict_keeps_every_type():
    value = {f"k{index}": item for index, (_, item) in enumerate(HETEROGENEOUS)}

    restored = assert_round_trip(value, dict[str, Any])

    assert {key: type(item) for key, item in restored.items()} == {key: type(item) for key, item in value.items()}


@pytest.mark.parametrize("value", HETEROGENEOUS_PARAMS)
@pytest.mark.parametrize(
    "placement",
    [
        pytest.param(lambda v: (v, Any), id="root"),
        pytest.param(lambda v: (v, Optional[Any]), id="optional"),
        pytest.param(lambda v: ([v], list[Any]), id="list"),
        pytest.param(lambda v: ((v, v), tuple[Any, ...]), id="tuple"),
        pytest.param(lambda v: ({"k": v}, dict[str, Any]), id="dict"),
        pytest.param(lambda v: (AnyBox(v), AnyBox), id="dataclass-field"),
        pytest.param(lambda v: ({"k": [AnyBox(v)]}, dict[str, list[AnyBox]]), id="nested"),
    ],
)
def test_any_keeps_the_exact_primitive_type_in_every_placement(value, placement):
    obj, cls = placement(value)

    assert_round_trip(obj, cls)


@pytest.mark.parametrize(
    "bag",
    [
        pytest.param(AnyBag(a=True, b=1, c=1.0, d="1", e=None), id="ordered"),
        pytest.param(AnyBag(a=1.0, b=True, c=None, d=1, e="1"), id="shuffled"),
        pytest.param(AnyBag(a=0, b=False, c=0.0, d="", e=None), id="falsy"),
        pytest.param(AnyBag(a=[True, 1, 1.0], b={"x": False}, c=[], d={}, e=[[None]]), id="containers"),
    ],
)
def test_dataclass_with_any_fields_keeps_exact_types(bag):
    assert_round_trip(bag, AnyBag)


def test_any_dataclass_exact_text():
    bag = AnyBag(a=True, b=1, c=1.0, d="1", e=None)

    assert serialize_json(bag) == '{"a": true, "b": 1, "c": 1.0, "d": "1", "e": null}'


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(Decimal("1.20"), "1.20", id="decimal"),
        pytest.param(datetime(2024, 2, 29, 12, 34, 56), "2024-02-29T12:34:56", id="datetime"),
        pytest.param(date(2024, 2, 29), "2024-02-29", id="date"),
        pytest.param(time(12, 34, 56), "12:34:56", id="time"),
        pytest.param(b"abc", "YWJj", id="bytes"),
        pytest.param(Path("report.json"), "report.json", id="path"),
        pytest.param(Color.RED, "red", id="enum"),
        pytest.param(Level.HIGH, 2, id="int-enum"),
    ],
)
def test_leaf_stored_under_any_comes_back_as_its_json_form_limitation(value, expected):
    # documents current behavior, not a promise: Any carries no type to restore, so the JSON form is returned
    restored = deserialize_json(to_json_value(AnyBox(value)), AnyBox).value

    assert restored == expected
    assert type(restored) is type(expected)


# --- strings ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ensure_ascii", [True, False], ids=["ascii", "unicode"])
@pytest.mark.parametrize(("value", "ascii_text"), [pytest.param(v, t, id=i) for i, v, t in STRING_CASES])
def test_string_survives_either_ensure_ascii_as_value_key_and_item(value, ascii_text, ensure_ascii):
    text = serialize_json(value, ensure_ascii=ensure_ascii)

    if ensure_ascii:
        assert text == ascii_text
        assert text.isascii()
    assert json.loads(text) == value
    assert_round_trip(value, str, ensure_ascii=ensure_ascii)
    assert_round_trip({value: [value]}, dict[str, list[str]], ensure_ascii=ensure_ascii)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param("\u05e9\u05dc\u05d5\u05dd", '"\u05e9\u05dc\u05d5\u05dd"', id="hebrew"),
        pytest.param("\u4f60\u597d", '"\u4f60\u597d"', id="cjk"),
        pytest.param("\U0001F600", '"\U0001F600"', id="emoji"),
        pytest.param("e\u0301", '"e\u0301"', id="combining-mark"),
        pytest.param("\u2028", '"\u2028"', id="line-separator"),
        pytest.param("\x7f", '"\x7f"', id="delete-character"),
        pytest.param("\ud800", '"\ud800"', id="lone-surrogate"),
        pytest.param("a\x00b", '"a\\u0000b"', id="nul-is-still-escaped"),
        pytest.param("tab\there", '"tab\\there"', id="tab-is-still-escaped"),
        pytest.param('"\\', '"\\"\\\\"', id="quote-and-backslash-are-still-escaped"),
    ],
)
def test_ensure_ascii_false_text(value, expected):
    assert serialize_json(value, ensure_ascii=False) == expected


@pytest.mark.parametrize("code", range(32), ids=lambda code: f"U+{code:04X}")
def test_every_control_character_is_escaped(code):
    short_escapes = {8: "\\b", 9: "\\t", 10: "\\n", 12: "\\f", 13: "\\r"}
    expected = '"' + short_escapes.get(code, f"\\u{code:04x}") + '"'

    for ensure_ascii in (True, False):
        assert serialize_json(chr(code), ensure_ascii=ensure_ascii) == expected
    assert_round_trip(chr(code), str)


def test_decomposed_and_precomposed_strings_stay_different():
    decomposed, precomposed = "e\u0301", "\u00e9"

    assert serialize_json([decomposed, precomposed]) == '["e\\u0301", "\\u00e9"]'
    restored = assert_round_trip([decomposed, precomposed], list[str])
    assert restored[0] != restored[1] and len(restored[0]) == 2 and len(restored[1]) == 1


def test_adjacent_surrogate_code_points_merge_under_ensure_ascii_limitation():
    # documents current behavior (the stdlib json decoder joins escaped surrogate pairs), not a promise
    value = "\ud83d\ude00"  # two lone surrogates, not the emoji

    assert len(value) == 2
    assert len(json.loads(serialize_json(value))) == 1
    assert len(json.loads(serialize_json(value, ensure_ascii=False))) == 2


def test_string_keys_with_special_characters_are_escaped_like_values():
    value = {"": 1, 'q"uote': 2, "back\\slash": 3, "new\nline": 4, "\u05d0": 5, "\U0001F600": 6}

    assert serialize_json(value) == (
        '{"": 1, "q\\"uote": 2, "back\\\\slash": 3, "new\\nline": 4, "\\u05d0": 5, "\\ud83d\\ude00": 6}'
    )
    assert_round_trip(value, dict[str, int])


def test_unicode_equivalent_keys_stay_two_keys():
    value = {"e\u0301": 1, "\u00e9": 2}

    assert serialize_json(value) == '{"e\\u0301": 1, "\\u00e9": 2}'
    assert len(assert_round_trip(value, dict[str, int])) == 2


def test_strings_that_look_like_other_leaf_values_stay_strings_next_to_real_ones():
    record = DatedText(label="2024-02-29", day=date(2024, 2, 29), code="YWJj", price_text="1.20")

    assert serialize_json(record) == (
        '{"label": "2024-02-29", "day": "2024-02-29", "code": "YWJj", "price_text": "1.20"}'
    )
    restored = assert_round_trip(record, DatedText)
    assert type(restored.label) is str and type(restored.day) is date


# --- dict keys of leaf types -------------------------------------------------------------------------------------

# (id, key, the JSON object key it becomes)
KEY_CASES = [
    ("str", "plain", "plain"),
    ("str-empty", "", ""),
    ("str-quote", 'say "hi"', 'say "hi"'),
    ("str-hebrew", "\u05e9\u05dc\u05d5\u05dd", "\u05e9\u05dc\u05d5\u05dd"),
    ("str-nul", "\x00", "\x00"),
    ("str-emoji", "\U0001F600", "\U0001F600"),
    ("str-looks-numeric", "1", "1"),
    ("str-looks-boolean", "true", "true"),
    ("bool-true", True, "true"),
    ("bool-false", False, "false"),
    ("int-zero", 0, "0"),
    ("int-negative", -1, "-1"),
    ("int-2**63", 2**63, "9223372036854775808"),
    ("int-2**64-plus-1", 2**64 + 1, "18446744073709551617"),
    ("int-minus-2**100", -(2**100), "-1267650600228229401496703205376"),
    ("float", 0.1, "0.1"),
    ("float-whole", 1.0, "1.0"),
    ("float-negative-zero", -0.0, "-0.0"),
    ("float-1e300", 1e300, "1e+300"),
    ("float-smallest-subnormal", 5e-324, "5e-324"),
    ("float-1e-5", 1e-5, "1e-05"),
    ("float-inf", float("inf"), "Infinity"),
    ("float-negative-inf", float("-inf"), "-Infinity"),
    ("float-nan", float("nan"), "NaN"),
    ("enum-str", Color.RED, "red"),
    ("enum-str-mixin", StrColor.RED, "red"),
    ("enum-unicode-value", Greeting.HELLO, "\u05e9\u05dc\u05d5\u05dd"),
    ("enum-quote-value", Greeting.QUOTE, 'say "hi"'),
    ("enum-int-mixin", Bits.TWO, "2"),
    ("enum-float-mixin", Scale.HALF, "0.5"),
    ("enum-int", Level.HIGH, "2"),
    ("enum-int-negative", Signed.NEGATIVE, "-1"),
    ("enum-float", Ratio.HALF, "0.5"),
    ("enum-bool", Switch.ON, "true"),
    ("enum-none", Maybe.NOTHING, "null"),
    ("enum-alias", Alias.DEFAULT, "p"),
    ("enum-auto", Counter.SECOND, "2"),
    ("enum-flag", Perm.READ | Perm.WRITE, "3"),
    ("enum-int-flag", PermInt.READ | PermInt.EXECUTE, "5"),
    ("enum-tuple-value", Grid.WIDE, "[2, 1]"),
]

KEY_PARAMS = [pytest.param(key, key_text, id=case_id) for case_id, key, key_text in KEY_CASES]


@pytest.mark.parametrize(("key", "key_text"), KEY_PARAMS)
def test_leaf_dict_key_becomes_its_json_text(key, key_text):
    assert serialize_json({key: 1}) == "{" + json.dumps(key_text) + ": 1}"
    assert to_json_value({key: [key_text]}) == {key_text: [key_text]}


def test_mixed_leaf_keys_in_one_dict():
    value = {2: "int", 2.5: "float", "s": "str", Color.RED: "enum", True: "bool"}

    assert serialize_json(value) == '{"2": "int", "2.5": "float", "s": "str", "red": "enum", "true": "bool"}'


def test_sort_keys_orders_converted_keys_as_text():
    value = {10: "a", 2: "b", "x": "c", True: "d"}

    assert serialize_json(value) == '{"10": "a", "2": "b", "x": "c", "true": "d"}'
    assert serialize_json(value, sort_keys=True) == '{"10": "a", "2": "b", "true": "d", "x": "c"}'


@pytest.mark.parametrize(
    "key",
    [pytest.param(case[1], id=case[0]) for case in KEY_CASES if case[0].startswith("str")],
)
def test_string_keys_are_restored_as_strings(key):
    assert_round_trip({key: 1}, dict[str, int])


@pytest.mark.parametrize(
    ("key", "annotation"),
    [
        pytest.param(Color.BLUE, Color, id="enum-str"),
        pytest.param(StrColor.RED, StrColor, id="enum-str-mixin"),
        pytest.param(Alias.DEFAULT, Alias, id="enum-alias"),
        pytest.param(Greeting.HELLO, Greeting, id="enum-hebrew-value"),
        pytest.param(Greeting.EMOJI, Greeting, id="enum-emoji-value"),
        pytest.param(Greeting.QUOTE, Greeting, id="enum-quote-value"),
    ],
)
def test_enum_keys_with_string_values_are_restored_as_members(key, annotation):
    restored = assert_round_trip({key: [1]}, dict[annotation, list[int]])

    assert next(iter(restored)) is key


@pytest.mark.parametrize(
    ("key", "annotation"),
    [
        pytest.param(Level.HIGH, Level, id="int-enum"),
        pytest.param(Ratio.HALF, Ratio, id="float-enum"),
        pytest.param(Switch.ON, Switch, id="bool-enum"),
        pytest.param(Perm.READ | Perm.WRITE, Perm, id="flag"),
        pytest.param(Bits.ONE, Bits, id="int-mixin-enum"),
        pytest.param(Scale.HALF, Scale, id="float-mixin-enum"),
    ],
)
def test_enum_keys_with_non_string_values_cannot_be_restored_limitation(key, annotation):
    # documents current behavior, not a promise: a JSON key is always a string and is looked up as is
    text = serialize_json({key: 1})

    with pytest.raises(TypeError, match="Invalid value for"):
        deserialize_json(json.loads(text), dict[annotation, int])


@pytest.mark.parametrize(
    ("key", "annotation"),
    [
        pytest.param(7, int, id="int"),
        pytest.param(1.5, float, id="float"),
        pytest.param(True, bool, id="bool"),
    ],
)
def test_int_float_and_bool_keys_cannot_be_restored_limitation(key, annotation):
    # documents current behavior, not a promise: a JSON key is a string, and only str, Enum and tuple key types are
    # restored from it
    text = serialize_json({key: "v"})

    with pytest.raises(ValueError, match="de-convert"):
        deserialize_json(json.loads(text), dict[annotation, str])


def test_non_string_keys_become_strings_without_a_key_annotation_limitation():
    # documents current behavior, not a promise: an unparameterized dict keeps the JSON keys as they are
    text = serialize_json({1: "a", 2.5: "b", False: "c"})

    assert deserialize_json(json.loads(text), dict) == {"1": "a", "2.5": "b", "false": "c"}


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(None, id="none"),
        pytest.param(Decimal("1"), id="decimal"),
        pytest.param(datetime(2024, 1, 1), id="datetime"),
        pytest.param(date(2024, 1, 1), id="date"),
        pytest.param(time(1, 2), id="time"),
        pytest.param(b"k", id="bytes"),
        pytest.param(Path("k"), id="path"),
        pytest.param(PurePosixPath("k"), id="pure-path"),
    ],
)
def test_other_leaf_types_are_not_supported_as_keys_limitation(key):
    # documents current behavior, not a promise: only str, bool, int, float, Enum and tuple keys are converted
    with pytest.raises(TypeError, match="Unsupported JSON mapping key type"):
        serialize_json({key: 1})


# --- datetime, date, time ----------------------------------------------------------------------------------------


def test_aware_datetime_keeps_its_offset_not_just_the_instant():
    in_utc = datetime(2024, 2, 29, 12, 34, 56, tzinfo=UTC_TZ)
    in_ist = in_utc.astimezone(IST)
    assert in_utc == in_ist

    for value, text, offset in [
        (in_utc, '"2024-02-29T12:34:56+00:00"', timedelta(0)),
        (in_ist, '"2024-02-29T18:04:56+05:30"', timedelta(hours=5, minutes=30)),
    ]:
        assert serialize_json(value) == text
        assert assert_round_trip(value, datetime).utcoffset() == offset


def test_naive_datetime_and_time_stay_naive():
    assert assert_round_trip(datetime(2024, 2, 29, 12, 34, 56), datetime).tzinfo is None
    assert assert_round_trip(time(12, 34, 56), time).tzinfo is None


def test_aware_time_keeps_its_offset():
    restored = assert_round_trip(time(12, 34, 56, tzinfo=NST), time)

    assert serialize_json(restored) == '"12:34:56-03:30"'
    assert restored.utcoffset() == timedelta(hours=-3, minutes=-30)


def test_same_wall_clock_with_different_offsets_is_written_differently():
    values = [datetime(2024, 2, 29, 12, tzinfo=tz) for tz in (UTC_TZ, IST, NST)] + [datetime(2024, 2, 29, 12)]

    assert serialize_json(values) == (
        '["2024-02-29T12:00:00+00:00", "2024-02-29T12:00:00+05:30", "2024-02-29T12:00:00-03:30", '
        '"2024-02-29T12:00:00"]'
    )
    assert_round_trip(values, list[datetime])


@pytest.mark.parametrize(
    ("text", "annotation"),
    [
        pytest.param("2024-02-29T12:34:56", date, id="date-from-datetime-text"),
        pytest.param("2024-02-29T12:34:56+00:00", date, id="date-from-aware-datetime-text"),
        pytest.param("12:34:56", date, id="date-from-time-text"),
        pytest.param("12:34:56", datetime, id="datetime-from-time-text"),
        pytest.param("2024-02-29T12:34:56", time, id="time-from-datetime-text"),
    ],
)
def test_iso_text_of_another_kind_is_rejected(text, annotation):
    with pytest.raises(TypeError, match=re.escape(f"Invalid value for {annotation}, got str")):
        deserialize_json(text, annotation)


@pytest.mark.parametrize(
    ("text", "annotation"),
    [
        pytest.param("2023-02-29", date, id="date-not-a-leap-year"),
        pytest.param("2024-02-30", date, id="date-30th-of-february"),
        pytest.param("2024-13-01", date, id="date-month-13"),
        pytest.param("2024-00-10", date, id="date-month-0"),
        pytest.param(" 2024-02-29", date, id="date-leading-space"),
        pytest.param("", date, id="date-empty"),
        pytest.param("2023-02-29T00:00:00", datetime, id="datetime-not-a-leap-year"),
        pytest.param("2024-02-29T25:00:00", datetime, id="datetime-hour-25"),
        pytest.param("2024-02-29T12:60:00", datetime, id="datetime-minute-60"),
        pytest.param("yesterday", datetime, id="datetime-word"),
        pytest.param("", datetime, id="datetime-empty"),
        pytest.param("25:00:00", time, id="time-hour-25"),
        pytest.param("12:60:00", time, id="time-minute-60"),
        pytest.param("noon", time, id="time-word"),
        pytest.param("", time, id="time-empty"),
    ],
)
def test_impossible_or_malformed_iso_text_is_rejected(text, annotation):
    with pytest.raises(TypeError, match=re.escape(f"Invalid value for {annotation}, got str")):
        deserialize_json(text, annotation)


def test_datetime_value_in_a_date_annotated_field_cannot_be_restored():
    text = serialize_json(DateBox(day=datetime(2024, 2, 29, 12, 0)))
    assert text == '{"day": "2024-02-29T12:00:00"}'

    with pytest.raises(TypeError, match=re.escape("Invalid value for <class 'datetime.date'>, got str")):
        deserialize_json(json.loads(text), DateBox)


def test_datetime_annotation_reads_date_text_as_midnight_limitation():
    # documents current behavior (datetime.fromisoformat accepts a bare date), not a promise
    restored = deserialize_json("2024-02-29", datetime)

    assert type(restored) is datetime
    assert restored == datetime(2024, 2, 29, 0, 0)


def test_datetime_before_date_in_a_union_swallows_date_text_limitation():
    # documents current behavior, not a promise: the first member that accepts the text wins
    assert type(deserialize_json("2024-02-29", datetime | date)) is datetime
    assert type(deserialize_json("2024-02-29", date | datetime)) is date
    assert type(deserialize_json("2024-02-29T12:00:00", date | datetime)) is datetime


@pytest.mark.parametrize("annotation", [datetime, date, time, Decimal, bytes, Color], ids=lambda cls: cls.__name__)
@pytest.mark.parametrize(
    "data",
    [
        pytest.param([], id="empty-list"),
        pytest.param({}, id="empty-dict"),
        pytest.param(["a"], id="list"),
        pytest.param({"a": 1}, id="dict"),
        pytest.param(None, id="null"),
    ],
)
def test_conversion_handlers_reject_containers_and_null(annotation, data):
    with pytest.raises(TypeError, match=re.escape(f"Invalid value for {annotation}, got {type(data).__name__}")):
        deserialize_json(data, annotation)


@pytest.mark.parametrize("annotation", [datetime, date, time, bytes, Color], ids=lambda cls: cls.__name__)
@pytest.mark.parametrize(
    "data",
    [
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param(1, id="int"),
        pytest.param(1.5, id="float"),
    ],
)
def test_text_handlers_reject_bool_and_number_json(annotation, data):
    with pytest.raises(TypeError, match=re.escape(f"Invalid value for {annotation}, got {type(data).__name__}")):
        deserialize_json(data, annotation)


def test_decimal_annotation_accepts_json_numbers_and_bools_limitation():
    # documents current behavior (Decimal(data) takes any number), not a promise; a float brings its binary expansion
    assert_strictly_equal(deserialize_json(True, Decimal), Decimal(1))
    assert_strictly_equal(deserialize_json(0.1, Decimal), Decimal(0.1))
    assert deserialize_json(0.1, Decimal) != Decimal("0.1")


def test_datetime_fold_is_not_written_limitation():
    # documents current behavior (ISO 8601 text has no fold), not a promise
    value = datetime(2024, 1, 1, 12, fold=1)

    assert serialize_json(value) == '"2024-01-01T12:00:00"'
    assert deserialize_json(json.loads(serialize_json(value)), datetime).fold == 0


class _PlusThree(tzinfo):
    def utcoffset(self, dt):
        return timedelta(hours=3)

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "PLUS3"


def test_custom_tzinfo_comes_back_as_a_fixed_offset_limitation():
    # documents current behavior, not a promise: only the offset is written
    value = datetime(2024, 1, 1, 8, tzinfo=_PlusThree())

    assert serialize_json(value) == '"2024-01-01T08:00:00+03:00"'
    restored = deserialize_json(json.loads(serialize_json(value)), datetime)
    assert restored == value
    assert type(restored.tzinfo) is timezone


def test_wide_list_of_datetimes_keeps_order():
    start = datetime(2024, 1, 1, tzinfo=UTC_TZ)
    values = [start + timedelta(days=index, microseconds=index) for index in range(400)]

    restored = assert_round_trip(values, list[datetime])

    assert restored[0] == start and restored[-1] == values[-1]


# --- Decimal -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("literal", "text"),
    [pytest.param(literal, text, id=case_id) for case_id, literal, text in DECIMAL_CASES],
)
def test_decimal_is_always_a_json_string_never_a_number(literal, text):
    parsed = to_json_value([Decimal(literal)])[0]

    assert type(parsed) is str
    assert parsed == text


def test_decimal_exponents_stay_apart():
    values = [Decimal("1"), Decimal("1.0"), Decimal("1.00"), Decimal("1E+0"), Decimal("10E-1")]

    assert serialize_json(values) == '["1", "1.0", "1.00", "1", "1.0"]'
    restored = assert_round_trip(values, list[Decimal])
    assert [d.as_tuple().exponent for d in restored] == [0, -1, -2, 0, -1]


@pytest.mark.parametrize("literal", ["NaN", "sNaN", "Infinity", "-Infinity"])
def test_non_finite_decimal_is_fine_with_allow_nan_false(literal):
    # it is a string in the JSON text, so allow_nan does not concern it
    assert serialize_json(Decimal(literal), allow_nan=False) == json.dumps(literal)
    assert_round_trip([Decimal(literal)], list[Decimal], allow_nan=False)


def test_decimal_ignores_the_active_context_precision():
    literal = "12345678901234567890.1234567890123456789"

    with decimal.localcontext() as context:
        context.prec = 3
        assert serialize_json(Decimal(literal)) == json.dumps(literal)
        assert_round_trip(Decimal(literal), Decimal)


def test_decimal_and_float_of_the_same_number_are_written_differently():
    assert serialize_json([Decimal("0.1"), 0.1]) == '["0.1", 0.1]'


def test_malformed_decimal_text_becomes_nan_when_the_context_does_not_trap_limitation():
    # documents current behavior, not a promise: Decimal(text) asks the active decimal context what to do with garbage
    with decimal.localcontext() as context:
        context.traps[decimal.InvalidOperation] = False
        restored = deserialize_json("not a number", Decimal)

        assert type(restored) is Decimal and restored.is_nan()

    with pytest.raises(TypeError, match=re.escape("Invalid value for <class 'decimal.Decimal'>, got str")):
        deserialize_json("not a number", Decimal)


def test_decimal_annotation_accepts_a_sign_digits_exponent_array_limitation():
    # documents current behavior (Decimal reads a (sign, digits, exponent) sequence), not a promise
    assert_strictly_equal(deserialize_json([1, [1, 2, 5], -2], Decimal), Decimal("-1.25"))


# --- bytes -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("length", range(10))
def test_base64_padding_follows_the_length(length):
    data = bytes(range(65, 65 + length))
    text = to_json_value(data)

    assert text == base64.b64encode(data).decode("ascii")
    assert text.count("=") == -length % 3
    assert len(text) == 4 * -(-length // 3)
    assert_round_trip(data, bytes)


def test_bytes_use_the_standard_alphabet_not_the_url_safe_one():
    assert to_json_value(b"\xfb\xef\xbe\xff\xff\xff") == "++++////"
    assert deserialize_json("++++////", bytes) == b"\xfb\xef\xbe\xff\xff\xff"


def test_bytes_and_str_of_the_same_characters_are_written_differently():
    assert serialize_json(["abc", b"abc"]) == '["abc", "YWJj"]'


def test_base64_decoding_drops_characters_outside_the_alphabet_limitation():
    # documents current behavior (base64.b64decode without validation), not a promise
    assert deserialize_json("!!!!", bytes) == b""
    assert deserialize_json("YQ==\n", bytes) == b"a"
    # line-wrapped Base64, as MIME encoders write it
    assert deserialize_json("YWJj\nZGVm", bytes) == b"abcdef"
    assert deserialize_json("-_-_", bytes) == b""


@pytest.mark.parametrize("text", ["YQ", "YQ=", "Y"])
def test_truncated_base64_is_rejected(text):
    with pytest.raises(TypeError, match=re.escape("Invalid value for <class 'bytes'>, got str")):
        deserialize_json(text, bytes)


# --- Path family -------------------------------------------------------------------------------------------------

# (id, value, text): pure paths have a fixed flavour, so their text is the same on every platform
PURE_PATH_CASES = [
    ("posix-absolute", PurePosixPath("/usr/local/bin"), '"/usr/local/bin"'),
    ("posix-relative", PurePosixPath("src/pkg/mod.py"), '"src/pkg/mod.py"'),
    ("posix-trailing-slash-dropped", PurePosixPath("a/b/"), '"a/b"'),
    ("posix-double-slash-collapses", PurePosixPath("a//b"), '"a/b"'),
    ("posix-empty-becomes-dot", PurePosixPath(""), '"."'),
    ("posix-root", PurePosixPath("/"), '"/"'),
    ("posix-spaces", PurePosixPath("my dir/my file.txt"), '"my dir/my file.txt"'),
    ("posix-unicode", PurePosixPath("\u05d3\u05d5/\u0444"), '"\\u05d3\\u05d5/\\u0444"'),
    ("windows-drive", PureWindowsPath("C:/Users/x"), '"C:\\\\Users\\\\x"'),
    ("windows-drive-root", PureWindowsPath("C:/"), '"C:\\\\"'),
    ("windows-relative", PureWindowsPath("a/b"), '"a\\\\b"'),
    ("windows-drive-only", PureWindowsPath("C:"), '"C:"'),
    ("windows-drive-relative", PureWindowsPath("C:a"), '"C:a"'),
    ("windows-mixed-separators", PureWindowsPath("a/b\\c"), '"a\\\\b\\\\c"'),
    ("windows-empty-becomes-dot", PureWindowsPath(""), '"."'),
    ("windows-case-is-kept", PureWindowsPath("C:/Users/X"), '"C:\\\\Users\\\\X"'),
    ("posix-backslash-is-part-of-the-name", PurePosixPath("a\\b"), '"a\\\\b"'),
    ("windows-unc", PureWindowsPath("//server/share/dir"), '"\\\\\\\\server\\\\share\\\\dir"'),
]


@pytest.mark.parametrize(
    ("value", "text"), [pytest.param(value, text, id=case_id) for case_id, value, text in PURE_PATH_CASES]
)
def test_pure_path_text(value, text):
    assert serialize_json(value) == text
    assert to_json_value({"path": [value]}) == {"path": [json.loads(text)]}


def test_unicode_path_with_ensure_ascii_false():
    assert serialize_json(PurePosixPath("\u05d3/\u0444"), ensure_ascii=False) == '"\u05d3/\u0444"'


# The concrete Path of the running platform cannot be a pure path of the other flavour
OTHER_FLAVOUR = PureWindowsPath("C:/Users/x") if os.name == "posix" else PurePosixPath("/usr/local/x")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the Path handler ignores cls and always builds a Path of the running platform, so a pure path of the "
           "other flavour and a Path subclass come back as a different, unequal type",
)
@pytest.mark.parametrize(
    ("value", "annotation"),
    [
        pytest.param(OTHER_FLAVOUR, type(OTHER_FLAVOUR), id="pure-path-of-the-other-flavour"),
        pytest.param(ProjectPath("a/b"), ProjectPath, id="path-subclass"),
    ],
)
def test_path_annotation_restores_an_instance_of_that_annotation(value, annotation):
    restored = deserialize_json(json.loads(serialize_json(value)), annotation)

    assert isinstance(restored, annotation)
    assert restored == value


@pytest.mark.parametrize(
    ("value", "annotation"),
    [
        pytest.param(PurePath("a/b"), PurePath, id="pure-path"),
        pytest.param(type(PurePath())("a/b"), type(PurePath()), id="pure-path-of-the-running-platform"),
    ],
)
def test_pure_path_annotation_restores_the_concrete_platform_path_limitation(value, annotation):
    # documents current behavior, not a promise: a Path is a pure path of its own flavour, so it is an instance of the
    # annotation and equal to the original, but not the exact type that went in
    restored = deserialize_json(json.loads(serialize_json(value)), annotation)

    assert type(restored) is type(Path())
    assert isinstance(restored, annotation)
    assert restored == value


@pytest.mark.parametrize("data", [5, 1.5, True, None, ["a"], {"a": "b"}])
def test_path_annotation_rejects_non_string_json(data):
    with pytest.raises(TypeError):
        deserialize_json(data, Path)


def test_path_union_falls_through_to_the_next_member():
    assert deserialize_json(5, Path | int) == 5
    assert deserialize_json(None, Optional[Path]) is None
    assert deserialize_json("a", Path | int) == Path("a")


def test_empty_and_dot_paths_are_the_same_path_limitation():
    # documents current behavior (pathlib normalizes both to "."), not a promise
    assert serialize_json([Path(""), Path(".")]) == '[".", "."]'
    assert deserialize_json("", Path) == Path(".")


# --- Enum --------------------------------------------------------------------------------------------------------


def test_enum_aliases_are_written_alike_and_restore_the_canonical_member():
    assert Alias.DEFAULT is Alias.PRIMARY
    assert serialize_json([Alias.PRIMARY, Alias.DEFAULT, Alias.SECONDARY]) == '["p", "p", "s"]'
    assert deserialize_json("p", Alias) is Alias.PRIMARY


def test_auto_values_are_written_as_their_numbers():
    members = [Counter.FIRST, Counter.SECOND, Counter.THIRD]

    assert serialize_json(members) == "[1, 2, 3]"
    assert_round_trip(members, list[Counter])


def test_flag_combinations_use_their_combined_value():
    value = [Perm.READ | Perm.WRITE | Perm.EXECUTE, Perm.READ, Perm(0)]

    assert serialize_json(value) == "[7, 1, 0]"
    restored = assert_round_trip(value, list[Perm])
    assert restored[0] == Perm.READ | Perm.WRITE | Perm.EXECUTE


def test_int_enum_is_written_as_a_plain_number_in_every_container():
    assert serialize_json({"level": Level.HIGH, "levels": [Level.LOW, Level.HIGH]}) == (
        '{"level": 2, "levels": [1, 2]}'
    )


def test_str_valued_enum_and_its_string_are_written_alike():
    assert serialize_json([Color.RED, "red", StrColor.RED]) == '["red", "red", "red"]'


@pytest.mark.parametrize(
    "member", [pytest.param(member, id=member.name.lower()) for member in Mixed if member is not Mixed.NOTHING]
)
def test_enum_with_members_of_unrelated_types_keeps_each_member(member):
    restored = assert_round_trip([member, member], list[Mixed])

    assert restored[0] is member


def test_none_valued_enum_member_is_null():
    assert serialize_json(Maybe.NOTHING) == "null"
    assert serialize_json(Mixed.NOTHING) == "null"
    assert assert_round_trip(Maybe.NOTHING, Maybe) is Maybe.NOTHING
    assert assert_round_trip({"v": Maybe.NOTHING, "w": Maybe.SOMETHING}, dict[str, Maybe])["v"] is Maybe.NOTHING
    assert assert_round_trip(Mixed.NOTHING, Mixed) is Mixed.NOTHING


def test_none_valued_enum_member_in_optional_is_ambiguous_with_none_limitation():
    # documents current behavior, not a promise: null is both None and Maybe.NOTHING, the first union member wins
    assert deserialize_json(None, Optional[Maybe]) is Maybe.NOTHING
    assert deserialize_json(None, type(None) | Maybe) is None


def test_list_valued_enum_round_trips_through_its_array():
    assert serialize_json(Palette.COOL) == '["blue"]'
    assert assert_round_trip({"p": Palette.WARM}, dict[str, Palette])["p"] is Palette.WARM


def test_enum_missing_hook_widens_the_accepted_values():
    # the lookup is Enum's own call, so a _missing_ defined by the class applies; writing is still by value
    assert serialize_json(LooseColor.RED) == '"red"'
    assert deserialize_json("red", LooseColor) is LooseColor.RED
    assert deserialize_json("RED", LooseColor) is LooseColor.RED
    assert deserialize_json({"value": "RED"}, _box(LooseColor)).value is LooseColor.RED
    assert deserialize_json({"RED": 1}, dict[LooseColor, int]) == {LooseColor.RED: 1}
    for data in ("purple", 5, None):
        with pytest.raises(TypeError, match="Invalid value for"):
            deserialize_json(data, LooseColor)


def test_enum_keyed_dicts_of_leaf_values_round_trip():
    value = Prices(
        by_color={Color.RED: [Decimal("1.50"), Decimal("2")], Color.BLUE: []},
        by_name={"launch": {StrColor.RED: date(2024, 2, 29)}, "": {}},
    )

    assert serialize_json(value) == (
        '{"by_color": {"red": ["1.50", "2"], "blue": []}, "by_name": {"launch": {"red": "2024-02-29"}, "": {}}}'
    )
    assert_round_trip(value, Prices)


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="a tuple-valued member is written as an array, which Enum lookup by tuple value cannot match",
)
@pytest.mark.parametrize(
    ("obj", "annotation", "text"),
    [
        pytest.param(Grid.SQUARE, Grid, "[1, 1]", id="root"),
        pytest.param([Grid.WIDE, Grid.SQUARE], list[Grid], "[[2, 1], [1, 1]]", id="list-items"),
    ],
)
def test_tuple_valued_enum_round_trips(obj, annotation, text):
    assert serialize_json(obj) == text
    assert_round_trip(obj, annotation)


@pytest.mark.parametrize(
    ("member", "text"),
    [
        pytest.param(Schedule.LAUNCH, '"2024-02-29"', id="date-valued"),
        pytest.param(Price.HIGH, '"20.00"', id="decimal-valued"),
    ],
)
def test_enum_with_non_json_values_is_written_but_cannot_be_restored_limitation(member, text):
    # documents current behavior, not a promise: the value is written like any date or Decimal, but the lookup on the
    # way back compares the JSON string with the date or Decimal and finds no member
    assert serialize_json(member) == text

    with pytest.raises(TypeError, match="Invalid value for"):
        deserialize_json(json.loads(text), type(member))


@pytest.mark.parametrize(
    ("data", "annotation"),
    [
        pytest.param(1.0, Level, id="float-for-int-enum"),
        pytest.param(True, Level, id="bool-for-int-enum"),
        pytest.param(0, Switch, id="int-for-bool-enum"),
    ],
)
def test_enum_lookup_accepts_equal_values_of_another_json_type_limitation(data, annotation):
    # documents current behavior (Enum lookup compares values with ==), not a promise
    assert type(deserialize_json(data, annotation)) is annotation


@pytest.mark.parametrize(
    ("data", "annotation"),
    [
        pytest.param(5, Color, id="int-for-str-enum"),
        pytest.param("2", Level, id="str-for-int-enum"),
        pytest.param("purple", Color, id="unknown-str"),
        pytest.param(3, Level, id="unknown-int"),
        pytest.param(None, Color, id="null-for-enum-without-none"),
    ],
)
def test_enum_rejects_values_that_are_not_members(data, annotation):
    with pytest.raises(TypeError, match=re.escape(f"Invalid value for {annotation}, got {type(data).__name__}")):
        deserialize_json(data, annotation)


# --- unions of leaf types ----------------------------------------------------------------------------------------

UNION_CASES = [
    pytest.param("2024-02-29", date | int, date(2024, 2, 29), id="date-member"),
    pytest.param(5, date | int, 5, id="int-after-date"),
    pytest.param("2024-02-29T12:34:56", date | datetime, datetime(2024, 2, 29, 12, 34, 56), id="datetime-after-date"),
    pytest.param("12:34:56", date | time, time(12, 34, 56), id="time-after-date"),
    pytest.param("red", Color | int, Color.RED, id="enum-member"),
    pytest.param(5, Color | int, 5, id="int-after-enum"),
    pytest.param("purple", Color | str, "purple", id="str-after-enum"),
    pytest.param(2, Level | str, Level.HIGH, id="int-enum-member"),
    pytest.param("2", Level | str, "2", id="str-after-int-enum"),
    pytest.param("1.20", Decimal | None, Decimal("1.20"), id="decimal-member"),
    pytest.param(None, Decimal | None, None, id="none-after-decimal"),
    pytest.param("not a number", Decimal | str, "not a number", id="str-after-decimal"),
    pytest.param("AQID", bytes | None, b"\x01\x02\x03", id="bytes-member"),
    # three characters are not a valid Base64 length
    pytest.param("abc", bytes | str, "abc", id="str-after-bytes"),
]


@pytest.mark.parametrize(("data", "annotation", "expected"), UNION_CASES)
def test_union_of_leaf_types_picks_the_member_the_data_fits(data, annotation, expected):
    assert_strictly_equal(deserialize_json(data, annotation), expected)
    assert_strictly_equal(deserialize_json([data, data], list[annotation]), [expected, expected])
    assert_strictly_equal(deserialize_json({"value": data}, _box(annotation)).value, expected)


def test_bytes_before_str_in_a_union_swallows_base64_looking_text_limitation():
    # documents current behavior, not a promise: the first member that accepts the text wins
    assert deserialize_json("abcd", bytes | str) == b"i\xb7\x1d"
    assert deserialize_json("abcd", str | bytes) == "abcd"


# --- subclasses of leaf types ------------------------------------------------------------------------------------

# Subclasses of str, int, float, Path and Enum are written like their base (the encoder or the handler lookup sees the
# base). The handlers of these five are found by exact type, so a subclass is not recognized at all.
# (value, JSON text, the handled base type)
SUBCLASS_CASES = [
    pytest.param(FakeDatetime(2024, 2, 29, 12, 34, 56), '"2024-02-29T12:34:56"', datetime, id="datetime-subclass"),
    pytest.param(FakeDate(2024, 2, 29), '"2024-02-29"', date, id="date-subclass"),
    pytest.param(FakeTime(12, 34, 56), '"12:34:56"', time, id="time-subclass"),
    pytest.param(Money("1.20"), '"1.20"', Decimal, id="decimal-subclass"),
    pytest.param(Blob(b"abc"), '"YWJj"', bytes, id="bytes-subclass"),
]


@pytest.mark.parametrize(
    ("value", "text"),
    [
        pytest.param(Label("abc"), '"abc"', id="str-subclass"),
        pytest.param(Count(3), "3", id="int-subclass"),
        pytest.param(Weight(3.5), "3.5", id="float-subclass"),
        pytest.param(ProjectPath("report.json"), '"report.json"', id="path-subclass"),
    ],
)
def test_subclasses_of_natively_supported_types_are_written_like_their_base(value, text):
    assert serialize_json(value) == text


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="handlers are found by exact type, so a datetime, date, time, Decimal or bytes subclass is not serializable",
)
@pytest.mark.parametrize(("value", "text", "base"), SUBCLASS_CASES)
def test_subclass_of_a_handled_type_is_written_like_its_base(value, text, base):
    assert serialize_json(value) == text


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="a subclass annotation finds no handler, so the JSON string is passed through unconverted and unvalidated",
)
@pytest.mark.parametrize(("value", "text", "base"), SUBCLASS_CASES)
def test_subclass_annotation_restores_a_value_of_the_handled_base(value, text, base):
    # whether the result is the subclass or the base is left open (the library builds no subclass for list, tuple or
    # dict annotations either); what must not happen is that the raw JSON string comes back
    restored = deserialize_json(json.loads(text), type(value))

    assert isinstance(restored, base)
    assert restored == value


# --- leaf values in sets, tuples and tuple keys ------------------------------------------------------------------

# Set items are ordered by their canonical JSON text (compact, ASCII-escaped), so "10" sorts before "2" and an escaped
# non-ASCII letter sorts before ASCII ones.
SET_CASES = [
    pytest.param({date(2024, 3, 1), date(2024, 2, 29), date(2023, 12, 31)}, set[date],
                 '["2023-12-31", "2024-02-29", "2024-03-01"]', id="dates"),
    pytest.param(frozenset({Decimal("2"), Decimal("10"), Decimal("1.5")}), frozenset[Decimal],
                 '["1.5", "10", "2"]', id="decimals-sort-as-text"),
    pytest.param({2.0, 10.0, 0.5}, set[float], "[0.5, 10.0, 2.0]", id="floats-sort-as-text"),
    pytest.param({b"b", b"a", b""}, set[bytes], '["", "YQ==", "Yg=="]', id="bytes"),
    pytest.param({Color.RED, Color.BLUE, Color.GREEN}, set[Color], '["blue", "green", "red"]', id="str-enum"),
    pytest.param(frozenset({Level.HIGH, Level.LOW}), frozenset[Level], "[1, 2]", id="int-enum"),
    pytest.param({Path("b"), Path("a")}, set[Path], '["a", "b"]', id="paths"),
    pytest.param({datetime(2024, 1, 1, tzinfo=IST), datetime(2024, 1, 1, tzinfo=UTC_TZ)}, set[datetime],
                 '["2024-01-01T00:00:00+00:00", "2024-01-01T00:00:00+05:30"]', id="datetimes-with-offsets"),
    pytest.param({"z", "\u00e9", "e"}, set[str], '["\\u00e9", "e", "z"]', id="escaped-letter-sorts-first"),
    pytest.param({"b", 2, 2.5, False, None}, set, '["b", 2, 2.5, false, null]', id="mixed-json-primitives"),
]


@pytest.mark.parametrize(("value", "annotation", "text"), SET_CASES)
def test_set_of_leaf_values_is_written_in_canonical_text_order(value, annotation, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)


LEAF_TUPLE = (
    datetime(2024, 2, 29, 12, 34, 56, tzinfo=IST),
    date(2024, 2, 29),
    time(8, 9, 10),
    Decimal("1.50"),
    b"\x00\xff",
    Path("a"),
    Color.GREEN,
    Level.LOW,
    1.0,
    True,
    "1",
    None,
)
LEAF_TUPLE_ANNOTATION = tuple[
    datetime, date, time, Decimal, bytes, Path, Color, Level, float, bool, str, Optional[int]
]
LEAF_TUPLE_TEXT = (
    '["2024-02-29T12:34:56+05:30", "2024-02-29", "08:09:10", "1.50", "AP8=", "a", "green", 1, 1.0, true, "1", null]'
)


def test_tuple_with_a_different_leaf_type_at_every_position():
    assert serialize_json(LEAF_TUPLE) == LEAF_TUPLE_TEXT
    assert_round_trip(LEAF_TUPLE, LEAF_TUPLE_ANNOTATION)
    assert_round_trip([LEAF_TUPLE, LEAF_TUPLE], list[LEAF_TUPLE_ANNOTATION])
    assert_round_trip({"t": LEAF_TUPLE}, dict[str, LEAF_TUPLE_ANNOTATION])


def test_tuple_with_a_missing_position_is_rejected():
    with pytest.raises(TypeError, match="Expected a JSON array of length 12"):
        deserialize_json(json.loads(LEAF_TUPLE_TEXT)[:-1], LEAF_TUPLE_ANNOTATION)


# (id, key, the JSON text of the tuple, which is the JSON object key)
TUPLE_KEY_CASES = [
    pytest.param((date(2024, 2, 29), Color.RED), '["2024-02-29", "red"]', id="date-and-enum"),
    pytest.param((Decimal("1.50"), b"ab", None, True, 1.5), '["1.50", "YWI=", null, true, 1.5]',
                 id="decimal-bytes-none-bool-float"),
    pytest.param((datetime(2024, 1, 1, tzinfo=UTC_TZ), time(1, 2), Level.HIGH),
                 '["2024-01-01T00:00:00+00:00", "01:02:00", 2]', id="datetime-time-int-enum"),
    pytest.param((Path("a"), "\u05d0", "\U0001F600"), '["a", "\\u05d0", "\\ud83d\\ude00"]',
                 id="path-and-unicode"),
    pytest.param(("x", ("y", (Color.BLUE,))), '["x", ["y", ["blue"]]]', id="nested"),
    pytest.param((float("nan"), float("-inf"), -0.0), "[NaN, -Infinity, -0.0]", id="non-finite-floats"),
]


@pytest.mark.parametrize(("key", "key_text"), TUPLE_KEY_CASES)
def test_tuple_key_is_the_json_text_of_the_tuple(key, key_text):
    assert serialize_json({key: 1}) == "{" + json.dumps(key_text) + ": 1}"
    assert to_json_value({key: 1}) == {key_text: 1}


def test_tuple_key_text_stays_ascii_even_with_ensure_ascii_false_limitation():
    # documents current behavior, not a promise: the key text is built with ensure_ascii=True, and then escaped again
    key_text = '["\\u05d0"]'  # a backslash, not the letter

    assert serialize_json({("\u05d0",): 1}, ensure_ascii=False) == "{" + json.dumps(key_text) + ": 1}"


def test_bare_tuple_keys_restore_each_json_primitive_exactly():
    key = (1, 1.0, True, None, "x", -0.0)

    restored = assert_round_trip({key: "v"}, dict[tuple, str])

    (restored_key,) = restored
    assert [type(item) for item in restored_key] == [int, float, bool, type(None), str, float]
    assert math.copysign(1.0, restored_key[5]) == -1.0


@pytest.mark.parametrize(
    ("key", "annotation"),
    [
        pytest.param((date(2024, 2, 29), Color.RED), tuple[date, Color], id="date-and-enum"),
        pytest.param((Decimal("1.50"), b"ab", Level.HIGH, True), tuple[Decimal, bytes, Level, bool],
                     id="decimal-bytes-int-enum-bool"),
        pytest.param((datetime(2024, 1, 1, tzinfo=IST), time(1, 2, 3, 4)), tuple[datetime, time],
                     id="datetime-and-time"),
        pytest.param((Path("a"), None), tuple[Path, Optional[int]], id="path-and-none"),
    ],
)
def test_tuple_key_of_leaf_types_round_trips(key, annotation):
    restored = assert_round_trip({key: [key]}, dict[annotation, list[annotation]])

    (restored_key,) = restored
    assert_strictly_equal(restored_key, key)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({Level.HIGH: 1, "2": 2}, id="int-enum-versus-text"),
        pytest.param({Switch.ON: 1, True: 2}, id="bool-enum-versus-bool"),
        pytest.param({Ratio.HALF: 1, 0.5: 2}, id="float-enum-versus-float"),
        pytest.param({Maybe.NOTHING: 1, "null": 2}, id="none-enum-versus-text"),
        pytest.param({Perm.READ | Perm.WRITE: 1, 3: 2}, id="flag-versus-int"),
        pytest.param({Bits.TWO: 1, "2": 2}, id="int-mixin-enum-versus-text"),
        pytest.param({Scale.HALF: 1, "0.5": 2}, id="float-mixin-enum-versus-text"),
        pytest.param({Color.RED: 1, StrColor.RED: 2}, id="two-enums-with-the-same-value"),
        pytest.param({Level.HIGH: 1, Counter.SECOND: 2}, id="int-enum-versus-auto-enum"),
        pytest.param({1.0: 1, "1.0": 2}, id="float-versus-text"),
        pytest.param({10**20: 1, "100000000000000000000": 2}, id="big-int-versus-text"),
    ],
)
def test_leaf_keys_that_write_the_same_json_key_collide(value):
    with pytest.raises(ValueError, match="collision"):
        serialize_json(value)


def test_keys_that_differ_only_in_text_do_not_collide():
    value = {True: "a", "True": "b", 1.5: "c", "1.50": "d", 10: "e", "010": "f"}

    assert serialize_json(value) == '{"true": "a", "True": "b", "1.5": "c", "1.50": "d", "10": "e", "010": "f"}'


# --- unsupported leaf-like types ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "type_name"),
    [
        pytest.param(uuid.UUID(int=1), "UUID", id="uuid"),
        pytest.param(timedelta(days=1), "timedelta", id="timedelta"),
        pytest.param(complex(1, 2), "complex", id="complex"),
        pytest.param(Fraction(1, 3), "Fraction", id="fraction"),
        pytest.param(bytearray(b"a"), "bytearray", id="bytearray"),
        pytest.param(memoryview(b"a"), "memoryview", id="memoryview"),
        pytest.param(range(3), "range", id="range"),
        pytest.param(slice(1), "slice", id="slice"),
        pytest.param(ipaddress.ip_address("127.0.0.1"), "IPv4Address", id="ip-address"),
    ],
)
def test_unregistered_leaf_like_types_raise_type_error_limitation(value, type_name):
    # documents current behavior, not a promise: these types have no handler (a fallback can serialize them)
    for obj in (value, [value], {"k": value}, AnyBox(value)):
        with pytest.raises(TypeError, match=type_name):
            serialize_json(obj)


# --- missing data ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("annotation", [int, float, bool, str], ids=lambda a: a.__name__)
def test_missing_primitive_field_becomes_none_limitation(annotation):
    # documents current behavior (an absent field is passed as None), not a promise
    assert deserialize_json({}, _box(annotation)).value is None


@pytest.mark.parametrize("annotation", [datetime, date, time, Decimal, bytes, Path, Color])
def test_missing_handler_typed_field_raises_type_error_limitation(annotation):
    # documents current behavior, not a promise: None reaches the handler of the field's type, which rejects it
    with pytest.raises(TypeError):
        deserialize_json({}, _box(annotation))


@pytest.mark.parametrize("annotation", [datetime, date, time, Decimal, bytes, Path, Color, Level])
def test_missing_optional_field_is_none(annotation):
    assert deserialize_json({}, _optional_box(annotation)).value is None


# --- encoding options --------------------------------------------------------------------------------------------


def test_leaf_values_honor_sort_keys_and_compact_separators():
    value = {"d": Decimal("1.0"), "b": b"a", "c": Color.RED, "a": date(2024, 2, 29)}

    assert serialize_json(value, sort_keys=True, separators=(",", ":")) == (
        '{"a":"2024-02-29","b":"YQ==","c":"red","d":"1.0"}'
    )


def test_leaf_values_honor_indent():
    assert serialize_json({"level": Level.HIGH, "at": time(1, 2)}, indent=2) == (
        '{\n  "level": 2,\n  "at": "01:02:00"\n}'
    )


def test_leaf_values_honor_ensure_ascii_false():
    value = {"\u05d0": Path("\u05d1"), "kind": StrColor.RED}

    assert serialize_json(value, ensure_ascii=False) == '{"\u05d0": "\u05d1", "kind": "red"}'
    assert serialize_json(value) == '{"\\u05d0": "\\u05d1", "kind": "red"}'


# --- one record of every leaf type -------------------------------------------------------------------------------

TYPICAL_RECORD = LeafRecord(
    count=3,
    ratio=0.5,
    active=True,
    name="Zo\u00eb",
    note=None,
    created=datetime(2024, 2, 29, 12, 34, 56, tzinfo=IST),
    day=date(2024, 2, 29),
    at=time(8, 9, 10, 11),
    price=Decimal("19.90"),
    blob=b"\x00\xffJ",
    path=Path("report.json"),
    color=Color.RED,
    level=Level.HIGH,
    extra=[1, 1.0, True],
)

TYPICAL_RECORD_TEXT = (
    '{"count": 3, "ratio": 0.5, "active": true, "name": "Zo\\u00eb", "note": null, '
    '"created": "2024-02-29T12:34:56+05:30", "day": "2024-02-29", "at": "08:09:10.000011", "price": "19.90", '
    '"blob": "AP9K", "path": "report.json", "color": "red", "level": 2, "extra": [1, 1.0, true]}'
)

# every field at a boundary: falsy, empty, extreme
EDGE_RECORD = LeafRecord(
    count=0,
    ratio=-0.0,
    active=False,
    name="",
    note="",
    created=datetime.min.replace(tzinfo=UTC_TZ),
    day=date.max,
    at=time.max,
    price=Decimal("-0"),
    blob=b"",
    path=Path(""),
    color=Color.BLUE,
    level=Level.LOW,
    extra=None,
)

EDGE_RECORD_JSON = {
    "count": 0,
    "ratio": -0.0,
    "active": False,
    "name": "",
    "note": "",
    "created": "0001-01-01T00:00:00+00:00",
    "day": "9999-12-31",
    "at": "23:59:59.999999",
    "price": "-0",
    "blob": "",
    "path": ".",
    "color": "blue",
    "level": 1,
    "extra": None,
}


def test_typical_record_has_exact_text():
    assert serialize_json(TYPICAL_RECORD) == TYPICAL_RECORD_TEXT


def test_edge_record_has_the_expected_json():
    assert json.dumps(to_json_value(EDGE_RECORD)) == json.dumps(EDGE_RECORD_JSON)


@pytest.mark.parametrize(
    "record", [pytest.param(TYPICAL_RECORD, id="typical"), pytest.param(EDGE_RECORD, id="edge")]
)
def test_record_of_every_leaf_type_round_trips(record):
    assert_round_trip(record, LeafRecord)


def test_ledger_of_records_round_trips():
    ledger = Ledger(
        entries=[TYPICAL_RECORD, EDGE_RECORD, TYPICAL_RECORD],
        by_name={"typical": TYPICAL_RECORD, "edge": EDGE_RECORD},
    )

    assert serialize_json(ledger).startswith('{"entries": [' + TYPICAL_RECORD_TEXT + ", ")
    restored = assert_round_trip(ledger, Ledger)
    assert restored.latest is None
    assert restored.entries[0] is not restored.entries[2]


def make_leaf_record(index):
    return LeafRecord(
        count=index,
        ratio=index / 8,  # exact in binary
        active=index % 2 == 0,
        name=f"n{index}\u05d0",
        note=None if index % 3 == 0 else str(index),
        created=datetime(2000, 1, 1, tzinfo=IST) + timedelta(days=index, microseconds=index),
        day=date(2000, 1, 1) + timedelta(days=index),
        at=time(index % 24, index % 60, index % 60, index),
        price=Decimal(index).scaleb(-2),
        blob=bytes((index + step) % 256 for step in range(index % 5)),
        path=Path(f"dir{index}", "file.txt"),
        color=list(Color)[index % 3],
        level=Level(1 + index % 2),
        extra=[index, float(index), bool(index % 2)],
    )


def test_generated_record_has_the_hand_derived_json():
    expected = {
        "count": 7,
        "ratio": 0.875,
        "active": False,
        "name": "n7\u05d0",
        "note": "7",
        "created": "2000-01-08T00:00:00.000007+05:30",
        "day": "2000-01-08",
        "at": "07:07:07.000007",
        "price": "0.07",
        "blob": "Bwg=",  # b"\x07\x08"
        "path": os.sep.join(["dir7", "file.txt"]),
        "color": "green",
        "level": 2,
        "extra": [7, 7.0, True],
    }

    assert json.dumps(to_json_value(make_leaf_record(7))) == json.dumps(expected)


def test_four_hundred_records_in_a_list_and_in_a_dict_round_trip():
    records = [make_leaf_record(index) for index in range(400)]

    restored = assert_round_trip(records, list[LeafRecord])
    assert [record.count for record in restored] == list(range(400))
    restored_by_name = assert_round_trip({record.name: record for record in records}, dict[str, LeafRecord])
    assert restored_by_name["n399\u05d0"].level is Level.HIGH


# --- depth -------------------------------------------------------------------------------------------------------


def _nest(value, annotation, json_value, depth):
    for _ in range(depth):
        value = {"k": [value]}
        annotation = dict[str, list[annotation]]
        json_value = {"k": [json_value]}
    return value, annotation, json_value


DEEP_LEAVES = [
    pytest.param(datetime(2024, 2, 29, 12, 34, 56, 7, tzinfo=NST), datetime, "2024-02-29T12:34:56.000007-03:30",
                 id="aware-datetime"),
    pytest.param(Decimal("-1.50"), Decimal, "-1.50", id="decimal"),
    pytest.param(b"\x00\xff", bytes, "AP8=", id="bytes"),
    pytest.param(Color.GREEN, Color, "green", id="enum"),
    pytest.param(-0.0, float, -0.0, id="negative-zero"),
    pytest.param("\U0001F600", str, "\U0001F600", id="emoji"),
    pytest.param(2**64 + 1, int, 2**64 + 1, id="big-int"),
    pytest.param(Path("a b"), Path, "a b", id="path"),
]


@pytest.mark.parametrize(("leaf", "annotation", "json_leaf"), DEEP_LEAVES)
def test_leaf_nested_twenty_five_levels_deep(leaf, annotation, json_leaf):
    value, deep_annotation, expected = _nest(leaf, annotation, json_leaf, 25)

    assert json.dumps(to_json_value(value)) == json.dumps(expected)
    assert_round_trip(value, deep_annotation)

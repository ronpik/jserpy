"""Seeded, property-style round trips over generated (annotation, value) pairs.

``Generator`` turns ``random.Random(seed)`` into a type annotation drawn from a recursive grammar (scalars, enums,
Optional / unions, lists, tuples, sets of scalars and frozen dataclasses, string / enum / tuple keyed dicts and
``Mapping`` annotations, dataclasses, ``Any``) and then into a value that conforms to it. Every seed must satisfy the
same universal properties: the JSON text equals an independent reference model, the round trip is exact, encoding
options only change the text, insertion order never matters, the restored value shares no mutable container with the
original or with the parsed JSON, and a corrupted document is rejected with a data-free TypeError or not at all.

A failing seed reproduces with ``generate(seed)``; the seed and the annotation are in the test id and the message.

After the generated properties come self-checks of the generator, tests of the helpers (a check that cannot fail
proves nothing), and hand-written cases for what the grammar leaves out on purpose: the library defects found while
building it are strict xfails, and behavior that is current but not promised is pinned in ``*_limitation`` tests.
"""

import base64
import collections.abc
import contextlib
import copy
import dataclasses
import enum
import functools
import hashlib
import json
import math
import operator
import os
import random
import subprocess
import sys
import types
import typing
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Optional, Union

import pytest

from jserpy import (
    deserialize_json,
    serialize_json,
    serialize_json_as_dict,
    serialize_json_as_obj,
)
from objects_support import assert_round_trip, assert_strictly_equal

SEEDS = range(400)
MAX_DEPTH = 5

UNION_ORIGINS = (typing.Union, types.UnionType)
MUTATION = "mutated"
# Not a number, an enum value, a date, a Decimal or valid Base64 (it is a valid str, Path, bare list element, ...)
SECRET_TEXT = "SECRET-7f3a"
SECRET_NUMBER = 8675309

# A module-level pool: enums created per call would grow every lru_cache keyed by class.
STR_ENUMS = (
    enum.Enum("Color", {"RED": "red", "GREEN": "green", "BLUE": "blue"}, module=__name__),
    enum.Enum(
        "Quirky",
        {"EMPTY": "", "HEBREW": "שלום", "QUOTED": 'say "hi"', "NUMERIC": "42", "PADDED": " padded ", "EMOJI": "😀"},
        module=__name__,
    ),
    enum.Enum("TextMixin", {"UP": "up", "DOWN": "down"}, module=__name__, type=str),
)
INT_ENUMS = (
    enum.Enum(
        "Rank",
        {"FIRST": 1, "SECOND": 2, "ZERO": 0, "NEGATIVE": -1, "HUGE": 2**40},
        module=__name__,
    ),
    enum.IntEnum("Priority", {"LOW": 1, "HIGH": 10}, module=__name__),
)
Color, Quirky, TextMixin = STR_ENUMS
Rank, Priority = INT_ENUMS

FIELD_NAMES = ("a", "b", "c", "d", "value", "items", "name", "type", "id", "data", "_hidden", "שם", "naïve")
DATACLASS_BUCKETS = 12

# Characters that stress JSON escaping and encodings: quotes, backslash, control characters, several scripts, astral
# plane characters, then (as code points, so that the source stays readable) a combining mark, zero-width joiner and
# space, BOM, line and paragraph separators, no-break space and a noncharacter. No lone surrogates: json would pair
# two adjacent ones into a single character.
ALPHABET = (
    "abcxyzABCXYZ0189 _-./\\\"'"
    "\x00\x01\x1f\t\n\r\x7f"
    "éßñü"
    "שלום"
    "مرحبا"
    "日本語한국어"
    "😀🎉𝔘\U0001f1ee\U0001f1f1"
    + "".join(map(chr, (0x0301, 0x200D, 0x200B, 0xFEFF, 0x2028, 0x2029, 0x00A0, 0xFFFF)))  # see the comment above
)
SPECIAL_TEXTS = ("", "null", "true", "123", "2024-01-01", "NaN", "[]", "{}", "\\u00e9", '"quoted"', "a\\nb", " ")
PATH_SEGMENTS = ("a", "b.txt", "with space", "שלום", "日本語", "..", "x.y.z", "UPPER", "~tilde", "naïve", "emoji😀")
DECIMAL_TEXTS = (
    "0", "-0", "0.00", "1.20", "1E+3", "1E-10", "-1.5", "123456789012345678901234567890.123456789", "1E+999",
)
DECIMAL_SPECIAL_TEXTS = ("NaN", "-NaN", "sNaN", "Infinity", "-Infinity")
FLOAT_BOUNDARIES = (
    0.0, -0.0, 1.0, -1.0, 0.1, 0.5, 1e16, 1e-7, 5e-324, 1.7976931348623157e308, -1.7976931348623157e308,
    2.2250738585072014e-308, 123456789.123456789, 1 / 3,
)
INT_BOUNDARIES = (0, 1, -1, 2**31 - 1, -(2**31), 2**53, 2**53 + 1, 2**63 - 1, -(2**63), 2**64, 10**30, -(10**30))
LITERAL_POOLS = (
    (0, 1, -1, 2, 42, 2**40),
    ("a", "b", "", "שלום", "x y", "true"),
    (None, 1, "a"),
    (True, False),
)
LEAF_CLASSES = (int, float, str, bool, type(None), bytes, datetime, date, time, Decimal, Path, dict, list, tuple)
HASHABLE_LEAF_CLASSES = (int, str, bool, float, bytes, datetime, date, time, Decimal, Path)
LEAF_NAMES = {
    int: "int", float: "float", str: "str", bool: "bool", bytes: "bytes", datetime: "datetime", date: "date",
    time: "time", Decimal: "decimal", Path: "path", dict: "bare-dict", list: "bare-list", tuple: "bare-tuple",
}
ALL_CONSTRUCTORS = frozenset(LEAF_NAMES.values()) | {
    "none", "enum", "literal", "any", "optional", "union", "list", "tuple-fixed", "tuple-variable", "dict-str",
    "dict-enum", "dict-tuple", "mapping", "set", "frozenset", "dataclass", "frozen-dataclass",
}

FROZEN_BUCKETS = 4

# Memoized classes of the grammar, keyed by what decides their fields; each class is built from its own seeded Random,
# so the pool does not depend on the order in which tests ask for them.
_DATACLASS_POOL: dict[tuple[int, int], type] = {}
_FROZEN_DATACLASS_POOL: dict[tuple[int, int], type] = {}


class Generator:
    """Builds annotations from the grammar and values that conform to them, all from one seeded ``Random``."""

    LEAF_CHANCE = 0.08

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.hashable = False  # set while building set members and tuple keys: no NaN, no unhashable extremes
        self.nonfinite = False  # whether floats may be NaN / Infinity
        self.scalar_makers = {
            int: self.int_value, float: self.float_value, str: self.text, bool: self.bool_value,
            bytes: self.bytes_value, datetime: self.datetime_value, date: self.date_value, time: self.time_value,
            Decimal: self.decimal_value, Path: self.path_value, dict: self.bare_dict_value,
            list: self.bare_list_value, tuple: self.bare_tuple_value,
        }

    # annotations

    def annotation(self, depth: int) -> Any:
        if depth == 0 or self.rng.random() < self.LEAF_CHANCE:
            return self.leaf()
        branches = [
            self.optional, self.list_of, self.tuple_fixed, self.tuple_variable, self.dict_str, self.dict_enum,
            self.dict_tuple, self.set_of, self.frozenset_of, self.dataclass_of, self.union,
        ]
        return branches[self.rng.randrange(len(branches))](depth)

    def child_depth(self, depth: int) -> int:
        return depth - 1 if self.rng.random() < 0.85 else self.rng.randrange(depth)

    def leaf(self) -> Any:
        kind = self.rng.randrange(len(LEAF_CLASSES) + 3)
        if kind < len(LEAF_CLASSES):
            return LEAF_CLASSES[kind]
        if kind == len(LEAF_CLASSES):
            return self.rng.choice(STR_ENUMS + INT_ENUMS)
        if kind == len(LEAF_CLASSES) + 1:
            return self.literal()
        return Any

    def literal(self) -> Any:
        pool = self.rng.choice(LITERAL_POOLS)
        return Literal[tuple(self.rng.sample(pool, self.rng.randint(1, min(3, len(pool)))))]

    def hashable_leaf(self) -> Any:
        kind = self.rng.randrange(len(HASHABLE_LEAF_CLASSES) + 2)
        if kind < len(HASHABLE_LEAF_CLASSES):
            return HASHABLE_LEAF_CLASSES[kind]
        if kind == len(HASHABLE_LEAF_CLASSES):
            return self.rng.choice(STR_ENUMS + INT_ENUMS)
        return self.literal()

    def hashable_of(self, depth: int) -> Any:
        if depth == 0 or self.rng.random() < 0.5:
            return self.hashable_leaf()
        kind = self.rng.randrange(5)
        if kind == 0:
            count = self.rng.randint(1, 3)
            return tuple[tuple(self.hashable_of(self.child_depth(depth)) for _ in range(count))]
        if kind == 1:
            return tuple[self.hashable_of(self.child_depth(depth)), ...]
        if kind == 2:
            return frozenset[self.hashable_of(self.child_depth(depth))]
        if kind == 3:
            return Optional[self.hashable_leaf()]
        return self.frozen_dataclass_of(depth)

    def frozen_dataclass_of(self, depth: int) -> Any:
        """A hashable dataclass, the only kind of dataclass that can be a set member."""
        key = (depth, self.rng.randrange(FROZEN_BUCKETS))
        if key not in _FROZEN_DATACLASS_POOL:
            inner = Generator(random.Random(f"frozen-dataclass/{key}"))
            names = inner.rng.sample(FIELD_NAMES, inner.rng.choice((1, 2, 2, 3)))
            fields = [(name, inner.hashable_of(inner.child_depth(depth))) for name in names]
            _FROZEN_DATACLASS_POOL[key] = dataclasses.make_dataclass("Frozen%dx%d" % key, fields, frozen=True)
        return _FROZEN_DATACLASS_POOL[key]

    def optional(self, depth: int) -> Any:
        inner = self.annotation(depth)  # Optional adds no nesting layer, so it keeps the depth
        form = self.rng.randrange(3)
        if form == 1:
            with contextlib.suppress(TypeError):  # e.g. ``Any | None`` is not supported on every Python
                return inner | None
        if form == 2:
            return Union[None, inner]  # None first: the member that must reject every non-null value
        return Optional[inner]

    def list_of(self, depth: int) -> Any:
        item = self.annotation(self.child_depth(depth))
        return typing.List[item] if self.typing_spelling() else list[item]

    def typing_spelling(self) -> bool:
        return self.rng.random() < 0.25

    def mapping_of(self, key: Any, item: Any) -> Any:
        """``dict`` in its builtin and ``typing`` spellings, or the ``Mapping`` ABC (which restores to a ``dict``)."""
        spelling = self.rng.randrange(8)
        if spelling < 4:
            return dict[key, item]
        if spelling < 6:
            return typing.Dict[key, item]
        if spelling == 6:
            return typing.Mapping[key, item]
        return collections.abc.Mapping[key, item]

    def tuple_fixed(self, depth: int) -> Any:
        count = self.rng.choice((0, 1, 2, 2, 3, 3))
        items = tuple(self.annotation(self.child_depth(depth)) for _ in range(count))
        # typing.Tuple[()] has the argument () before Python 3.11, so the empty tuple keeps the builtin spelling
        return typing.Tuple[items] if count and self.typing_spelling() else tuple[items]

    def tuple_variable(self, depth: int) -> Any:
        item = self.annotation(self.child_depth(depth))
        return typing.Tuple[item, ...] if self.typing_spelling() else tuple[item, ...]

    def dict_str(self, depth: int) -> Any:
        return self.mapping_of(str, self.annotation(self.child_depth(depth)))

    def dict_enum(self, depth: int) -> Any:
        key, item = self.rng.choice(STR_ENUMS), self.annotation(self.child_depth(depth))
        return self.mapping_of(key, item)

    def dict_tuple(self, depth: int) -> Any:
        key = self.rng.choice((
            tuple[int, str], tuple[str, ...], tuple[int, tuple[str, int]],
            typing.Tuple[int, str], typing.Tuple[str, ...], typing.Tuple[int, typing.Tuple[str, int]],
        ))
        return self.mapping_of(key, self.annotation(self.child_depth(depth)))

    def set_of(self, depth: int) -> Any:
        item = self.hashable_of(self.child_depth(depth))
        return typing.Set[item] if self.typing_spelling() else set[item]

    def frozenset_of(self, depth: int) -> Any:
        item = self.hashable_of(self.child_depth(depth))
        return typing.FrozenSet[item] if self.typing_spelling() else frozenset[item]

    def dataclass_of(self, depth: int) -> Any:
        """A class of the bounded pool, so that the number of classes (and lru_cache entries) stays small."""
        bucket = self.rng.randrange(DATACLASS_BUCKETS)
        key = (depth, bucket)
        if key not in _DATACLASS_POOL:
            inner = Generator(random.Random(f"dataclass/{key}"))
            names = inner.rng.sample(FIELD_NAMES, inner.rng.choice((1, 1, 2, 2, 3, 4)))
            fields = [(name, inner.annotation(inner.child_depth(depth))) for name in names]
            _DATACLASS_POOL[key] = dataclasses.make_dataclass("Gen%dx%d" % key, fields)
        return _DATACLASS_POOL[key]

    def union(self, depth: int) -> Any:
        """Two members of different JSON shapes (array, object), so that no member swallows the other's data."""
        array_builders = [self.list_of, self.tuple_fixed, self.tuple_variable, self.set_of, self.frozenset_of]
        object_builders = [self.dict_str, self.dict_enum, self.dict_tuple, self.dataclass_of]
        array = self.rng.choice(array_builders)(depth)
        obj = self.rng.choice(object_builders)(depth)
        none = type(None)
        members = self.rng.choice(([array, obj], [obj, array], [array, obj, none], [obj, array, none]))
        if self.rng.random() < 0.5:
            return Union[tuple(members)]
        return functools.reduce(operator.or_, members)

    # values

    def value(self, annotation: Any) -> Any:
        rng = self.rng
        origin = typing.get_origin(annotation)
        args = typing.get_args(annotation)
        if annotation is Any:
            return self.json_tree(rng.randrange(4))
        if annotation is type(None):
            return None
        if origin is Literal:
            return rng.choice(args)
        if origin in UNION_ORIGINS:
            members = [arg for arg in args if arg is not type(None)]
            if len(members) < len(args) and rng.random() < 0.2:
                return None
            return self.value(rng.choice(members))
        if origin is list:
            return [self.value(args[0]) for _ in range(self.size(args[0]))]
        if origin is tuple:
            if args and args[-1] is Ellipsis:
                return tuple(self.value(args[0]) for _ in range(self.size(args[0])))
            return tuple(self.value(arg) for arg in args)
        if origin in (dict, collections.abc.Mapping):
            return self.dict_value(*args)
        if origin in (set, frozenset):
            previous, self.hashable = self.hashable, True
            members = [self.value(args[0]) for _ in range(self.size(args[0]))]
            self.hashable = previous
            return origin(members)
        if _is_dataclass_type(annotation):
            return annotation(**{field.name: self.value(field.type) for field in dataclasses.fields(annotation)})
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            return rng.choice(list(annotation))
        return self.scalar_makers[annotation]()

    def size(self, item_annotation: Any) -> int:
        if _is_scalar_annotation(item_annotation):
            return self.rng.choice((0, 0, 1, 1, 2, 3, 4, 6, 12, 40))
        if self.rng.random() < 0.1:
            return 0
        return self.rng.choice((1, 1, 2, 2, 3))

    def dict_value(self, key_annotation: Any, value_annotation: Any) -> dict:
        count = self.size(value_annotation)
        if key_annotation is str:
            keys = [self.text() for _ in range(count)]
        elif isinstance(key_annotation, type) and issubclass(key_annotation, enum.Enum):
            members = list(key_annotation)
            self.rng.shuffle(members)
            keys = members[:count]
        else:
            previous, self.hashable = self.hashable, True
            keys = [self.value(key_annotation) for _ in range(count)]
            self.hashable = previous
        return {key: self.value(value_annotation) for key in keys}

    def json_tree(self, depth: int) -> Any:
        rng = self.rng
        if depth == 0 or rng.random() < 0.3:
            return rng.choice((self.int_value, self.float_value, self.text, self.bool_value, lambda: None))()
        if rng.random() < 0.5:
            return [self.json_tree(depth - 1) for _ in range(rng.choice((0, 1, 2, 3)))]
        return {self.text(): self.json_tree(depth - 1) for _ in range(rng.choice((0, 1, 2, 3)))}

    # scalars

    def text(self) -> str:
        if self.rng.random() < 0.2:
            return self.rng.choice(SPECIAL_TEXTS)
        length = self.rng.choice((0, 1, 2, 3, 5, 8, 30))
        return "".join(self.rng.choice(ALPHABET) for _ in range(length))

    def int_value(self) -> int:
        kind = self.rng.randrange(3)
        if kind == 0:
            return self.rng.choice(INT_BOUNDARIES)
        if kind == 1:
            return self.rng.randrange(-1000, 1000)
        return self.rng.randrange(-(2**40), 2**40)

    def bool_value(self) -> bool:
        return self.rng.choice((True, False))

    def float_value(self) -> float:
        rng = self.rng
        if self.nonfinite and rng.random() < 0.3:
            return rng.choice((math.inf, -math.inf) if self.hashable else (math.nan, math.inf, -math.inf))
        if rng.random() < 0.03:
            return -0.0  # a signed zero compares equal to 0.0, so only the strict comparison notices a loss
        kind = rng.randrange(5)
        if kind == 0:
            return rng.uniform(-1, 1)
        if kind == 1:
            return rng.uniform(-1e6, 1e6)
        if kind == 2:
            return rng.random() * 10 ** rng.randrange(-300, 300)
        if kind == 3:
            return float(rng.randrange(-(2**53), 2**53))
        return rng.choice(FLOAT_BOUNDARIES)

    def bytes_value(self) -> bytes:
        return self.rng.randbytes(self.rng.choice((0, 0, 1, 2, 3, 16, 257)))

    def decimal_value(self) -> Decimal:
        rng = self.rng
        kind = rng.randrange(4)
        if kind == 0:
            return Decimal(rng.choice(DECIMAL_TEXTS))
        if kind == 1 and not self.hashable:
            return Decimal(rng.choice(DECIMAL_SPECIAL_TEXTS))
        digits = tuple(rng.randrange(10) for _ in range(rng.randint(1, 12)))
        return Decimal((rng.randrange(2), digits, rng.randint(-20, 20)))

    def tzinfo(self) -> Optional[timezone]:
        kind = self.rng.randrange(4)
        if kind == 0:
            return None
        if kind == 1:
            return timezone.utc
        hours = self.rng.randint(-12, 14)
        minutes = self.rng.choice((0, 15, 30, 45))
        return timezone(timedelta(hours=hours, minutes=minutes if hours >= 0 else -minutes))

    def datetime_value(self) -> datetime:
        rng = self.rng
        if rng.random() < 0.15:
            boundary = rng.choice((datetime.min, datetime.max))
            return boundary.replace(tzinfo=rng.choice((None, timezone.utc)))
        # the year limits only with no offset: an aware datetime there overflows when compared or hashed
        tz = self.tzinfo()
        year = rng.randint(1900, 2100)
        return datetime(
            year, rng.randint(1, 12), rng.randint(1, 28), rng.randrange(24), rng.randrange(60), rng.randrange(60),
            rng.choice((0, 0, 1, 999999, rng.randrange(1_000_000))), tzinfo=tz,
        )

    def date_value(self) -> date:
        if self.rng.random() < 0.15:
            return self.rng.choice((date.min, date.max))
        return date(self.rng.randint(1, 9999), self.rng.randint(1, 12), self.rng.randint(1, 28))

    def time_value(self) -> time:
        rng = self.rng
        if rng.random() < 0.15:
            return rng.choice((time.min, time.max))
        return time(
            rng.randrange(24), rng.randrange(60), rng.randrange(60), rng.choice((0, 0, 1, 999999)),
            tzinfo=self.tzinfo(),
        )

    def bare_dict_value(self) -> dict:
        return {self.text(): self.json_tree(self.rng.randrange(3)) for _ in range(self.rng.choice((0, 1, 2, 3)))}

    def bare_list_value(self) -> list:
        return [self.json_tree(self.rng.randrange(3)) for _ in range(self.rng.choice((0, 1, 2, 3)))]

    def bare_tuple_value(self) -> tuple:
        return tuple(self.bare_list_value())

    def path_value(self) -> Path:
        segments = [self.rng.choice(PATH_SEGMENTS) for _ in range(self.rng.choice((0, 1, 2, 3, 4)))]
        return Path("/", *segments) if self.rng.random() < 0.3 else Path(*segments)


def generate(seed: int) -> tuple[Any, Any]:
    """The ``(annotation, value)`` of a seed; the depth parameter cycles through 0..MAX_DEPTH with the seed."""
    rng = random.Random(seed)
    generator = Generator(rng)
    generator.nonfinite = rng.random() < 0.4
    depth = seed % (MAX_DEPTH + 1)
    annotation = generator.annotation(depth)
    return annotation, generator.value(annotation)


# walkers: independent of the generator, so that the self-checks measure what was actually produced


def _is_dataclass_type(annotation: Any) -> bool:
    return isinstance(annotation, type) and dataclasses.is_dataclass(annotation)


def _is_scalar_annotation(annotation: Any) -> bool:
    return (
        typing.get_origin(annotation) in (None, Literal)
        and annotation not in (Any, dict, list, tuple)
        and not _is_dataclass_type(annotation)
    )


def _children(item: Any) -> Optional[list]:
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return [getattr(item, field.name) for field in dataclasses.fields(item)]
    if isinstance(item, dict):
        return list(item.values())
    if isinstance(item, (list, tuple, set, frozenset)):
        return list(item)
    return None


def constructors(annotation: Any) -> set[str]:
    """The grammar constructors an annotation is made of; raises KeyError on anything the grammar does not know."""
    found: set[str] = set()
    seen_dataclasses: set[type] = set()
    pending = [annotation]
    while pending:
        current = pending.pop()
        origin = typing.get_origin(current)
        args = typing.get_args(current)
        if current is Any:
            found.add("any")
        elif current is type(None):
            found.add("none")
        elif origin is Literal:
            found.add("literal")
        elif origin in UNION_ORIGINS:
            found.add("optional" if sum(arg is not type(None) for arg in args) == 1 else "union")
            pending.extend(args)
        elif origin is list:
            found.add("list")
            pending.extend(args)
        elif origin is tuple:
            found.add("tuple-variable" if Ellipsis in args else "tuple-fixed")
            pending.extend(arg for arg in args if arg is not Ellipsis)
        elif origin in (dict, collections.abc.Mapping):
            key, item = args
            if origin is collections.abc.Mapping:
                found.add("mapping")
            if typing.get_origin(key) is tuple:
                found.add("dict-tuple")
            elif key is str:
                found.add("dict-str")
            else:
                assert issubclass(key, enum.Enum), key
                found.add("dict-enum")
            pending.append(item)
        elif origin in (set, frozenset):
            found.add(origin.__name__)
            pending.extend(args)
        elif _is_dataclass_type(current):
            found.add("frozen-dataclass" if current.__dataclass_params__.frozen else "dataclass")
            if current not in seen_dataclasses:
                seen_dataclasses.add(current)
                pending.extend(field.type for field in dataclasses.fields(current))
        elif isinstance(current, type) and issubclass(current, enum.Enum):
            found.add("enum")
        else:
            found.add(LEAF_NAMES[current])
    return found


def value_depth(value: Any) -> int:
    """Layers of nested containers and dataclasses: 0 for a scalar, 1 for a flat or empty container."""
    deepest = 0
    pending = [(value, 0)]
    while pending:
        item, level = pending.pop()
        children = _children(item)
        if children is None:
            deepest = max(deepest, level)
        else:
            deepest = max(deepest, level + 1)
            pending.extend((child, level + 1) for child in children)
    return deepest


def value_features(value: Any) -> set[str]:
    features: set[str] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, float):
            if math.isnan(item):
                features.add("nan")
            elif item == math.inf:
                features.add("inf")
            elif item == -math.inf:
                features.add("-inf")
            elif item == 0 and math.copysign(1, item) < 0:
                features.add("-0.0")
        elif isinstance(item, int) and not isinstance(item, bool):
            if abs(item) >= 2**63:
                features.add("big-int")
        elif isinstance(item, str):
            _text_features(item, features)
        elif isinstance(item, bytes):
            if not item:
                features.add("empty-bytes")
        elif isinstance(item, Decimal):
            if not item.is_finite():
                features.add("decimal-special")
            elif item.as_tuple().exponent > 0:
                features.add("decimal-exponent")
        elif isinstance(item, datetime):
            features.add("aware-datetime" if item.tzinfo else "naive-datetime")
        elif isinstance(item, time):
            features.add("aware-time" if item.tzinfo else "naive-time")
        elif isinstance(item, dict):
            if not item:
                features.add("empty-dict")
            for key in item:
                if isinstance(key, enum.Enum):
                    features.add("enum-key")
                elif isinstance(key, tuple):
                    features.add("tuple-key")
                pending.append(key)
        elif isinstance(item, list) and not item:
            features.add("empty-list")
        elif isinstance(item, tuple) and not item:
            features.add("empty-tuple")
        elif isinstance(item, (set, frozenset)) and not item:
            features.add("empty-set")
        pending.extend(_children(item) or ())
    return features


def _text_features(text: str, features: set[str]) -> None:
    if not text:
        features.add("empty-str")
    if any(ord(char) > 0xFFFF for char in text):
        features.add("astral-char")
    if any(ord(char) < 0x20 for char in text):
        features.add("control-char")


def contains_nonfinite(value: Any) -> bool:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, float) and not math.isfinite(item):
            return True
        pending.extend(_children(item) or ())
    return False


def mutable_ids(value: Any) -> set[int]:
    """Identities of every list, dict, set and dataclass instance reachable from ``value``."""
    ids: set[int] = set()
    seen: set[int] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        children = _children(item)
        if children is None or id(item) in seen:
            continue
        seen.add(id(item))
        if not isinstance(item, (tuple, frozenset)):
            ids.add(id(item))
        pending.extend(children)
    return ids


def mutate_deeply(value: Any) -> int:
    """Change every mutable container reachable from ``value`` in place; return how many were changed."""
    containers: list[Any] = []
    seen: set[int] = set()
    pending = [value]
    while pending:
        item = pending.pop()
        children = _children(item)
        if children is None or id(item) in seen:
            continue
        seen.add(id(item))
        containers.append(item)
        pending.extend(children)

    changed = 0
    for container in containers:
        if isinstance(container, list):
            container.append(MUTATION)
        elif isinstance(container, dict):
            container[MUTATION] = MUTATION
        elif isinstance(container, set):
            container.add(MUTATION)
        elif (
            dataclasses.is_dataclass(container)
            and dataclasses.fields(container)
            and not container.__dataclass_params__.frozen
        ):
            setattr(container, dataclasses.fields(container)[0].name, MUTATION)
        else:
            continue  # tuple, frozenset, frozen dataclass: immutable
        changed += 1
    return changed


def rebuild(value: Any, rng: random.Random, *, shuffle_dicts: bool) -> Any:
    """An equal value built in another order: every set grown in a different insertion order (and, optionally, every
    dict's keys inserted in a shuffled order)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return type(value)(**{
            field.name: rebuild(getattr(value, field.name), rng, shuffle_dicts=shuffle_dicts)
            for field in dataclasses.fields(value)
        })
    if isinstance(value, dict):
        items = [(key, rebuild(item, rng, shuffle_dicts=shuffle_dicts)) for key, item in value.items()]
        if shuffle_dicts:
            rng.shuffle(items)
        return dict(items)
    if isinstance(value, (list, tuple)):
        return type(value)(rebuild(item, rng, shuffle_dicts=shuffle_dicts) for item in value)
    if isinstance(value, (set, frozenset)):
        # sorted first: the iteration order of a set of strings changes from process to process, and the seed alone
        # must decide the order the members are inserted in
        ordered = sorted(value, key=lambda item: json.dumps(reference_json(item), sort_keys=True))
        members = [rebuild(item, rng, shuffle_dicts=shuffle_dicts) for item in ordered]
        rng.shuffle(members)
        noise = [object() for _ in range(rng.randrange(40))]  # grow the table, then shrink it back
        scrambled = set(noise)
        scrambled.update(members)
        scrambled.difference_update(noise)
        return type(value)(scrambled)
    return value


def json_paths(tree: Any) -> list[tuple]:
    """The path (keys and indexes) of every node of a JSON value, the root included."""
    paths = []
    pending = [((), tree)]
    while pending:
        path, node = pending.pop()
        paths.append(path)
        if isinstance(node, dict):
            pending.extend(((*path, key), item) for key, item in node.items())
        elif isinstance(node, list):
            pending.extend(((*path, index), item) for index, item in enumerate(node))
    return sorted(paths, key=repr)  # a stable order for the seeded choice


def replace_at(tree: Any, path: tuple, new: Any) -> Any:
    if not path:
        return new
    parent = tree
    for step in path[:-1]:
        parent = parent[step]
    parent[path[-1]] = new
    return tree


def shuffle_json_keys(tree: Any, rng: random.Random) -> Any:
    """The same JSON value with the keys of every object in a shuffled order."""
    if isinstance(tree, dict):
        items = [(key, shuffle_json_keys(item, rng)) for key, item in tree.items()]
        rng.shuffle(items)
        return dict(items)
    if isinstance(tree, list):
        return [shuffle_json_keys(item, rng) for item in tree]
    return tree


def reference_json(value: Any) -> Any:
    """The JSON tree the README promises for ``value``, built without the library.

    Dataclasses are objects of their fields, enums are their values, tuples are arrays, sets are arrays ordered by
    the canonical (compact, sorted keys) JSON text of each member, bytes are Base64, dates and times are ISO 8601,
    decimals and paths are strings; mapping keys are strings, enum keys use their value and tuple keys the JSON text
    of the tuple.
    """
    if isinstance(value, enum.Enum):
        return reference_json(value.value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if dataclasses.is_dataclass(value):
        return {field.name: reference_json(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {_reference_key(key): reference_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [reference_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        # a member is written in its canonical form: the compact text with sorted keys, at every depth (this is only
        # visible for a frozen dataclass member, see test_dataclass_set_members_are_written_with_sorted_keys_limitation)
        canonical = sorted(json.dumps(reference_json(item), sort_keys=True, separators=(",", ":")) for item in value)
        return [json.loads(member) for member in canonical]
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    raise AssertionError(f"the reference model does not know {type(value)!r}")


def _reference_key(key: Any) -> str:
    if isinstance(key, enum.Enum):
        return _reference_key(key.value)
    if isinstance(key, str):
        return key
    if isinstance(key, bool):
        return "true" if key else "false"
    if isinstance(key, int):
        return str(key)
    if isinstance(key, tuple):
        return json.dumps(reference_json(key))
    raise AssertionError(f"the reference model does not know the key type {type(key)!r}")


def describe(annotation: Any) -> str:
    """A readable annotation for messages (a class repr would not show a generated dataclass's fields)."""
    if annotation is Any:
        return "Any"
    if annotation is type(None):
        return "None"
    if _is_dataclass_type(annotation):
        inner = ", ".join(f"{field.name}: {describe(field.type)}" for field in dataclasses.fields(annotation))
        return f"{annotation.__name__}({inner})"
    origin = typing.get_origin(annotation)
    if origin is None:
        return annotation.__name__
    args = typing.get_args(annotation)
    if origin is Literal:
        return "Literal[" + ", ".join(map(repr, args)) + "]"
    if origin in UNION_ORIGINS:
        return " | ".join(describe(arg) for arg in args)
    return f"{origin.__name__}[" + ", ".join("..." if arg is Ellipsis else describe(arg) for arg in args) + "]"


@contextlib.contextmanager
def seed_context(seed: int, annotation: Any):
    """Name the seed and the annotation in whatever fails, so that the failure can be reproduced by hand."""
    try:
        yield
    except Exception as error:
        raise AssertionError(f"seed {seed}, {describe(annotation)}: {type(error).__name__}: {error}") from error


def parse_strictly(text: str, *, allow_constants: bool) -> Any:
    """``json.loads`` that also rejects duplicate object keys and, unless allowed, NaN / Infinity."""

    def reject_constant(name: str) -> None:
        raise AssertionError(f"non-standard JSON constant {name} in a text without non-finite floats")

    def reject_duplicates(pairs: list) -> dict:
        keys = [key for key, _ in pairs]
        assert len(keys) == len(set(keys)), f"duplicate object keys in {keys!r}"
        return dict(pairs)

    return json.loads(
        text, parse_constant=None if allow_constants else reject_constant, object_pairs_hook=reject_duplicates
    )


def _unexpected_fallback(obj: Any) -> Any:
    raise AssertionError(f"the fallback was called for a supported {type(obj).__name__}")


@functools.cache
def wrapper_class(annotation: Any) -> type:
    fields = [("before", int), ("inner", annotation), ("after", Optional[str])]
    return dataclasses.make_dataclass(f"Wrapper{wrapper_class.cache_info().currsize}", fields)


@functools.cache
def pair_class(left: Any, right: Any) -> type:
    return dataclasses.make_dataclass(f"Pair{pair_class.cache_info().currsize}", [("left", left), ("right", right)])


@functools.cache
def _summary(seed: int) -> tuple[frozenset, frozenset, int]:
    annotation, value = generate(seed)
    return frozenset(constructors(annotation)), frozenset(value_features(value)), value_depth(value)

# Options that change the text but not the data. Each must equal json.dumps of the parsed value with the same options.
OPTION_VARIANTS = (
    {"sort_keys": True},
    {"indent": 2},
    {"indent": "\t"},
    {"separators": (",", ":")},
    {"ensure_ascii": False},
    {"ensure_ascii": False, "sort_keys": True, "indent": 1, "separators": (",", ": ")},
)


# properties of every generated seed


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"seed-{seed}")
def test_generated_value_round_trips_exactly(seed):
    annotation, value = generate(seed)
    with seed_context(seed, annotation):
        snapshot = copy.deepcopy(value)

        assert_round_trip(value, annotation)
        assert_strictly_equal(value, snapshot)  # neither serializing nor restoring touched the original

        assert_round_trip(value, annotation, sort_keys=True)
        assert_round_trip(value, annotation, ensure_ascii=False, indent=2, separators=(",", ": "))

        # the same data with every dict and set built in another order restores to the same value
        reordered = rebuild(value, random.Random(seed), shuffle_dicts=True)
        restored = deserialize_json(json.loads(serialize_json(reordered)), annotation)
        assert_strictly_equal(restored, value)

        # the intermediate value restores like the parsed text, and the order of the keys in the JSON is irrelevant
        assert_strictly_equal(deserialize_json(serialize_json_as_obj(value), annotation), value)
        shuffled = shuffle_json_keys(json.loads(serialize_json(value)), random.Random(seed))
        assert_strictly_equal(deserialize_json(shuffled, annotation), value)


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"seed-{seed}")
def test_generated_text_is_strict_json_that_matches_the_reference_model(seed):
    annotation, value = generate(seed)
    with seed_context(seed, annotation):
        snapshot = copy.deepcopy(value)
        finite = not contains_nonfinite(value)

        text = serialize_json(value)
        parsed = parse_strictly(text, allow_constants=not finite)
        expected = reference_json(value)

        assert isinstance(text, str) and text.isascii() and not text.endswith("\n")
        assert text == json.dumps(expected)
        assert_strictly_equal(parsed, expected)
        assert_strictly_equal(serialize_json_as_obj(value), parsed)
        assert_strictly_equal(serialize_json_as_dict(value), parsed)

        if finite:
            assert serialize_json(value, allow_nan=False) == text
        else:
            with pytest.raises(ValueError, match="Out of range float values"):
                serialize_json(value, allow_nan=False)

        # registered handlers take precedence: a fallback is never consulted for supported values
        assert serialize_json(value, fallback=_unexpected_fallback) == text

        # the intermediate value is detached from the object it came from
        detached = serialize_json_as_obj(value)
        mutate_deeply(detached)
        assert_strictly_equal(value, snapshot)


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"seed-{seed}")
def test_encoding_options_only_change_the_text_of_a_generated_value(seed):
    annotation, value = generate(seed)
    with seed_context(seed, annotation):
        parsed = json.loads(serialize_json(value))

        for options in OPTION_VARIANTS:
            text = serialize_json(value, **options)
            assert_strictly_equal(json.loads(text), parsed)
            assert text == json.dumps(parsed, **options), options
            if options.get("ensure_ascii", True):
                assert text.isascii()
            else:
                text.encode("utf-8")  # raises on a lone surrogate: the text must be storable as UTF-8


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"seed-{seed}")
def test_insertion_order_does_not_change_the_text_of_a_generated_value(seed):
    annotation, value = generate(seed)
    with seed_context(seed, annotation):
        text = serialize_json(value)

        for attempt in range(3):
            regrown = rebuild(value, random.Random(f"{seed}/{attempt}"), shuffle_dicts=False)
            assert serialize_json(regrown) == text

        reordered = rebuild(value, random.Random(seed), shuffle_dicts=True)
        assert serialize_json(reordered, sort_keys=True) == serialize_json(value, sort_keys=True)
        assert_strictly_equal(json.loads(serialize_json(reordered)), json.loads(text))


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"seed-{seed}")
def test_wrapping_a_generated_value_keeps_the_round_trip_exact(seed):
    annotation, value = generate(seed)
    with seed_context(seed, annotation):
        assert_round_trip([value], list[annotation])
        assert_round_trip({"key": value}, dict[str, annotation])
        assert_round_trip((value, 1), tuple[annotation, int])

        wrapper = wrapper_class(annotation)
        assert_round_trip(wrapper(seed, value, None), wrapper)

        assert_round_trip(value, Optional[annotation])
        assert_round_trip(None, Optional[annotation])

        # N copies, as one shared object or as N deep copies, are the same N equal but independent arrays
        count = seed % 4
        text = serialize_json([value] * count)
        assert text == serialize_json([copy.deepcopy(value) for _ in range(count)])
        assert text == "[" + ", ".join([serialize_json(value)] * count) + "]"
        restored = deserialize_json(json.loads(text), list[annotation])
        assert len(restored) == count
        for index, item in enumerate(restored):
            assert_strictly_equal(item, value, f"copy[{index}]")
        identities = [mutable_ids(item) for item in restored]
        for index, left in enumerate(identities):
            for right in identities[index + 1:]:
                assert not left & right, "two restored copies share a mutable container"


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"seed-{seed}")
def test_restored_generated_value_shares_nothing_with_the_original_or_the_parsed_json(seed):
    annotation, value = generate(seed)
    with seed_context(seed, annotation):
        text = serialize_json(value)

        # restored versus original
        snapshot = copy.deepcopy(value)
        restored = deserialize_json(json.loads(text), annotation)
        assert not mutable_ids(restored) & mutable_ids(value)
        mutate_deeply(restored)
        assert_strictly_equal(value, snapshot)

        # restored versus the parsed JSON it was built from, in both directions
        parsed = json.loads(text)
        restored = deserialize_json(parsed, annotation)
        assert not mutable_ids(restored) & mutable_ids(parsed)
        restored_snapshot = copy.deepcopy(restored)
        mutate_deeply(parsed)
        assert_strictly_equal(restored, restored_snapshot)

        parsed = json.loads(text)
        restored = deserialize_json(parsed, annotation)
        parsed_text = json.dumps(parsed)
        mutate_deeply(restored)
        assert json.dumps(parsed) == parsed_text


@pytest.mark.parametrize("seed", SEEDS[::2], ids=lambda seed: f"seed-{seed}")
def test_two_generated_values_side_by_side_keep_their_own_types(seed):
    first_annotation, first = generate(seed)
    second_annotation, second = generate(seed + 1)
    with seed_context(seed, first_annotation):
        assert_round_trip((first, second), tuple[first_annotation, second_annotation])

        pair = pair_class(first_annotation, second_annotation)
        assert_round_trip(pair(first, second), pair)
        assert_round_trip(pair(first, second), Optional[pair])
        assert serialize_json(pair(first, second)) == (
            '{"left": ' + serialize_json(first) + ', "right": ' + serialize_json(second) + "}"
        )


WRONG_JSON = (
    SECRET_TEXT, SECRET_NUMBER, 1.5, True, None, [], {}, [None], {"x": None}, [SECRET_TEXT], {SECRET_TEXT: 1},
)


def corrupted_documents(seed: int, count: int = 6) -> list:
    """The JSON of the seed's value with one node replaced by a value of another shape, ``count`` times."""
    annotation, value = generate(seed)
    parsed = json.loads(serialize_json(value))
    rng = random.Random(f"{seed}/corruption")
    paths = json_paths(parsed)
    documents = []
    for _ in range(count):
        path, wrong = rng.choice(paths), copy.deepcopy(rng.choice(WRONG_JSON))
        documents.append(replace_at(copy.deepcopy(parsed), path, wrong))
    return documents


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"seed-{seed}")
def test_corrupted_json_is_accepted_or_rejected_with_a_data_free_type_error(seed):
    # Primitive annotations pass any JSON primitive through, so a corrupted document may restore to something odd: what
    # must hold is that a rejection is always a TypeError (unions depend on it) and never echoes the offending data.
    annotation, _ = generate(seed)
    with seed_context(seed, annotation):
        for document in corrupted_documents(seed):
            snapshot = json.dumps(document)
            try:
                deserialize_json(document, annotation)
            except TypeError as error:
                assert SECRET_TEXT not in str(error) and str(SECRET_NUMBER) not in str(error), str(error)
            assert json.dumps(document) == snapshot, "deserialization mutated the document"


def test_corrupted_json_is_mostly_rejected():
    outcomes = []
    for seed in SEEDS:
        annotation, _ = generate(seed)
        for document in corrupted_documents(seed):
            try:
                deserialize_json(document, annotation)
            except TypeError:
                outcomes.append("rejected")
            else:
                outcomes.append("accepted")

    assert outcomes.count("rejected") > len(outcomes) * 0.4, "the corruptions rarely reach a typed node"
    assert outcomes.count("accepted") > 0  # primitives pass through unchecked, so some corruptions are invisible


COLLIDING_KEYS = (
    pytest.param(((1, "a"), '[1, "a"]'), id="tuple-vs-str"),
    pytest.param((1, "1"), id="int-vs-str"),
    pytest.param((Color.RED, "red"), id="enum-vs-str"),
)


@pytest.mark.parametrize("keys", COLLIDING_KEYS)
@pytest.mark.parametrize("seed", SEEDS[::4], ids=lambda seed: f"seed-{seed}")
def test_key_collisions_next_to_a_generated_value_are_reported(seed, keys):
    annotation, value = generate(seed)
    with seed_context(seed, annotation):
        colliding = dict.fromkeys(keys, value)
        assert len(colliding) == 2  # two different Python keys that spell the same JSON key

        with pytest.raises(ValueError, match="collision"):
            serialize_json([colliding])
        with pytest.raises(ValueError, match="collision"):
            serialize_json_as_obj({"payload": colliding})

        assert_round_trip(value, annotation)  # a failed call leaves nothing behind


@pytest.mark.parametrize("seed", SEEDS[::4], ids=lambda seed: f"seed-{seed}")
def test_cycles_next_to_a_generated_value_are_reported(seed):
    annotation, value = generate(seed)
    with seed_context(seed, annotation):
        cycle = [value]
        cycle.append(cycle)
        with pytest.raises(ValueError, match="(?i)cyclic"):
            serialize_json({"payload": cycle})

        assert_round_trip(value, annotation)


HASH_SEED_PROBE = """
import sys
sys.path.insert(0, sys.argv[1])
import test_objects_generated as module
print(module.texts_digest(range(120)))
"""


def texts_digest(seeds) -> str:
    texts = []
    for seed in seeds:
        _, value = generate(seed)
        texts += [serialize_json(value), serialize_json(value, sort_keys=True, ensure_ascii=False)]
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def test_text_is_the_same_under_every_string_hash_seed():
    # str hashes, and so the iteration order of a set of strings, change from one process to the next
    here = os.path.dirname(os.path.abspath(__file__))
    digests = {texts_digest(range(120))}
    for hash_seed in ("1", "2", "31337"):
        probe = subprocess.run(
            [sys.executable, "-c", HASH_SEED_PROBE, here],
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
            capture_output=True,
            text=True,
        )
        assert probe.returncode == 0, probe.stderr
        digests.add(probe.stdout.strip())

    assert len(digests) == 1, digests


# the generator itself must be reproducible and broad enough for the properties to mean something


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"seed-{seed}")
def test_a_seed_always_generates_the_same_pair(seed):
    first_annotation, first_value = generate(seed)
    second_annotation, second_value = generate(seed)

    assert first_annotation == second_annotation, describe(first_annotation)
    assert_strictly_equal(second_value, first_value)
    assert serialize_json(first_value) == serialize_json(second_value)


def test_the_grammar_reaches_every_constructor():
    seen = set().union(*(_summary(seed)[0] for seed in SEEDS))

    assert seen == ALL_CONSTRUCTORS


def test_every_constructor_is_used_by_several_seeds():
    counts = {name: 0 for name in ALL_CONSTRUCTORS}
    for seed in SEEDS:
        for name in _summary(seed)[0]:
            counts[name] += 1

    assert {name: count for name, count in counts.items() if count < 10} == {}


def test_values_range_from_plain_scalars_to_five_nested_levels():
    depths = [_summary(seed)[2] for seed in SEEDS]

    assert min(depths) == 0
    assert max(depths) >= 5
    assert sum(depth >= 4 for depth in depths) >= 30
    assert sum(depth == 0 for depth in depths) >= 20


def test_values_cover_the_boundary_features():
    expected = {
        "nan", "inf", "-inf", "-0.0", "big-int", "empty-str", "astral-char", "control-char", "empty-bytes",
        "decimal-special", "decimal-exponent", "aware-datetime", "naive-datetime", "aware-time", "naive-time",
        "empty-list", "empty-tuple", "empty-dict", "empty-set", "enum-key", "tuple-key",
    }
    seen = set().union(*(_summary(seed)[1] for seed in SEEDS))

    assert expected - seen == set()


# the checks must be able to fail


@dataclasses.dataclass
class Holder:
    items: list
    table: dict


def test_reference_model_matches_json_derived_by_hand():
    value = {
        "tags": {"b", "ab", "a"},
        "raw": (b"\x00\xff", Decimal("1.50")),
        "when": [datetime(2024, 2, 29, 12, 0, tzinfo=timezone.utc), date(2024, 1, 2), time(3, 4, 5)],
        "color": Color.RED,
        "by_color": {Color.GREEN: 1},
        "by_pair": {(1, "x"): None},
        "path": Path("a/b"),
        "rank": Priority.HIGH,
    }
    # set members are ordered by their JSON text: '"a"' < '"ab"' < '"b"'; b"\x00\xff" is "AP8=" in Base64
    expected_text = (
        '{"tags": ["a", "ab", "b"], "raw": ["AP8=", "1.50"], '
        '"when": ["2024-02-29T12:00:00+00:00", "2024-01-02", "03:04:05"], "color": "red", '
        '"by_color": {"green": 1}, "by_pair": {"[1, \\"x\\"]": null}, "path": "a/b", "rank": 10}'
    )

    assert json.dumps(reference_json(value)) == expected_text
    assert serialize_json(value) == expected_text


def test_mutation_helper_changes_every_mutable_container_but_no_immutable_one():
    inner = [1]
    value = Holder(items=inner, table={"k": (frozenset({1}), [2], {3})})

    assert mutate_deeply(value) == 5  # the dataclass, the list, the dict, the list in the tuple, the set
    assert value.items == MUTATION  # the first field of the dataclass is replaced
    assert inner == [1, MUTATION]
    assert value.table["k"][0] == frozenset({1})
    assert value.table["k"][1:] == ([2, MUTATION], {3, MUTATION})
    assert value.table[MUTATION] == MUTATION


def test_aliasing_is_detected_by_identity_and_by_mutation():
    original = {"inner": [1, 2], "other": [3]}
    snapshot = copy.deepcopy(original)
    aliasing = {"inner": original["inner"], "other": [3]}

    assert mutable_ids(original) & mutable_ids(aliasing) == {id(original["inner"])}
    mutate_deeply(aliasing)
    with pytest.raises(AssertionError):
        assert_strictly_equal(original, snapshot)


def test_rebuild_keeps_the_value_and_may_only_change_the_order():
    value = {"b": {3, 1, 2}, "a": [frozenset({"x", "y"}), (1, {2: "z"})]}

    rebuilt = rebuild(value, random.Random(1), shuffle_dicts=True)

    assert_strictly_equal(rebuilt, value)
    assert rebuilt is not value and rebuilt["b"] is not value["b"]
    assert serialize_json(rebuilt, sort_keys=True) == serialize_json(value, sort_keys=True)


def test_json_paths_and_replace_at_address_every_node():
    tree = {"a": [1, {"b": None}], "c": "x"}

    assert json_paths(tree) == sorted([(), ("a",), ("a", 0), ("a", 1), ("a", 1, "b"), ("c",)], key=repr)
    assert replace_at(copy.deepcopy(tree), ("a", 1, "b"), 5) == {"a": [1, {"b": 5}], "c": "x"}
    assert replace_at(copy.deepcopy(tree), ("a", 0), [2]) == {"a": [[2], {"b": None}], "c": "x"}
    assert replace_at(copy.deepcopy(tree), (), 7) == 7


def test_shuffle_json_keys_changes_only_the_order_of_keys():
    tree = {"a": 1, "b": [{"c": 2, "d": 3, "e": 4}], "f": {"g": 5, "h": 6, "i": 7}, "j": None}

    shuffled = [shuffle_json_keys(tree, random.Random(seed)) for seed in range(10)]

    assert all(item == tree for item in shuffled)
    assert all(json.dumps(item, sort_keys=True) == json.dumps(tree, sort_keys=True) for item in shuffled)
    assert any(json.dumps(item) != json.dumps(tree) for item in shuffled)


def test_mutation_helper_leaves_frozen_dataclasses_alone():
    member = OutOfOrder(zebra=1, apple=2)
    value = {"members": {member}}

    assert mutate_deeply(value) == 2  # the dict and the set
    assert value["members"] == {member, MUTATION}
    assert member == OutOfOrder(zebra=1, apple=2)


def test_set_text_does_not_depend_on_the_iteration_order_of_colliding_members():
    # 8 and 16 share a slot in a small hash table, so the member inserted first is iterated first
    first, second = {8, 16}, {16, 8}

    assert serialize_json(first) == serialize_json(second) == "[16, 8]"  # '16' < '8' as text


@dataclasses.dataclass(frozen=True)
class OutOfOrder:
    zebra: int
    apple: int


def test_dataclass_set_members_are_written_with_sorted_keys_limitation():
    # Documents current behavior, not a promise: a set member is written in the canonical form that decides the order
    # of the set, so a dataclass in a set loses the declaration order of its fields (it is an object either way).
    member = OutOfOrder(zebra=1, apple=2)

    assert serialize_json(member) == '{"zebra": 1, "apple": 2}'
    assert serialize_json({member}) == '[{"apple": 2, "zebra": 1}]'
    assert serialize_json([member]) == '[{"zebra": 1, "apple": 2}]'
    assert_round_trip({member}, set[OutOfOrder])


def test_strict_parsing_rejects_what_the_text_checks_rely_on():
    with pytest.raises(AssertionError, match="NaN"):
        parse_strictly("[NaN]", allow_constants=False)
    with pytest.raises(AssertionError, match="duplicate"):
        parse_strictly('{"a": 1, "a": 2}', allow_constants=True)
    assert math.isnan(parse_strictly("[NaN]", allow_constants=True)[0])


def test_seed_context_names_the_seed_and_the_annotation():
    with pytest.raises(AssertionError, match=r"seed 7, dict\[str, list\[int\]\]: ZeroDivisionError: boom"):
        with seed_context(7, dict[str, list[int]]):
            raise ZeroDivisionError("boom")


# hand-written boundary values the generator avoids on purpose, with the library behavior they expose


@dataclasses.dataclass
class Empty:
    pass


@dataclasses.dataclass
class EmptyHolder:
    e: Optional[Empty]


@dataclasses.dataclass
class HasNoneField:
    flag: None


@dataclasses.dataclass
class Measurement:
    value: float
    error: Optional[float]


@pytest.mark.parametrize(
    ("value", "annotation", "text"),
    [
        pytest.param(
            [math.nan, math.inf, -math.inf, -0.0], list[float], "[NaN, Infinity, -Infinity, -0.0]", id="list"
        ),
        pytest.param({"x": math.nan}, dict[str, float], '{"x": NaN}', id="dict-value"),
        pytest.param(Measurement(-math.inf, None), Measurement, '{"value": -Infinity, "error": null}', id="field"),
        pytest.param(Measurement(1.5, math.nan), Measurement, '{"value": 1.5, "error": NaN}', id="optional-field"),
        pytest.param((math.nan, 1), tuple[float, int], "[NaN, 1]", id="tuple"),
        pytest.param({"a": [math.nan, 2.5]}, Any, '{"a": [NaN, 2.5]}', id="any-tree"),
        # members are ordered by their JSON text: '1.5' < 'Infinity'
        pytest.param({math.inf, 1.5}, set[float], "[1.5, Infinity]", id="set"),
    ],
)
def test_non_finite_floats_round_trip_and_are_refused_when_not_allowed(value, annotation, text):
    assert serialize_json(value) == text
    assert_round_trip(value, annotation)
    with pytest.raises(ValueError, match="Out of range float values"):
        serialize_json(value, allow_nan=False)


@pytest.mark.parametrize(
    ("value", "annotation"),
    [
        pytest.param(Empty(), Empty, id="root"),
        pytest.param([Empty(), Empty()], list[Empty], id="list"),
        pytest.param({"k": Empty()}, dict[str, Empty], id="dict"),
        pytest.param((Empty(), 1), tuple[Empty, int], id="tuple"),
        pytest.param(Empty(), Union[Empty, list[int]], id="union-first"),
        pytest.param(Empty(), Union[list[int], Empty], id="union-second"),
        pytest.param(EmptyHolder(Empty()), EmptyHolder, id="optional-field-with-a-value"),
    ],
)
def test_empty_dataclass_round_trips_as_an_empty_object(value, annotation):
    assert_round_trip(value, annotation)


def test_empty_dataclass_serializes_to_an_empty_object():
    assert serialize_json(Empty()) == "{}"
    assert serialize_json(EmptyHolder(Empty())) == '{"e": {}}'
    assert serialize_json(EmptyHolder(None)) == '{"e": null}'


@pytest.mark.parametrize(
    ("data", "annotation", "expected"),
    [
        pytest.param(None, Optional[Empty], None, id="optional-null"),
        pytest.param(None, Union[Empty, list[int], None], None, id="union-null-last"),
        pytest.param([1, 2], Union[Empty, list[int]], [1, 2], id="union-array"),
        pytest.param({"e": None}, EmptyHolder, EmptyHolder(None), id="optional-field-null"),
    ],
)
def test_null_and_arrays_are_not_swallowed_by_a_dataclass_without_fields(data, annotation, expected):
    assert_strictly_equal(deserialize_json(data, annotation), expected)


@pytest.mark.parametrize(
    "data", [pytest.param("abc", id="str"), pytest.param([1], id="array"), pytest.param(7, id="int")]
)
def test_dataclass_without_fields_rejects_non_objects(data):
    with pytest.raises(TypeError):
        deserialize_json(data, Empty)


@pytest.mark.parametrize(
    "data", [pytest.param("abc", id="str"), pytest.param([1], id="array"), pytest.param(7, id="int")]
)
def test_dataclass_with_a_field_rejects_non_objects(data):
    # the behavior the case above is measured against: unions rely on this TypeError to try their next member
    with pytest.raises(TypeError):
        deserialize_json(data, Measurement)


@pytest.mark.parametrize(
    ("value", "annotation"),
    [
        pytest.param(None, type(None), id="root"),
        pytest.param([None, None], list[type(None)], id="list"),
        pytest.param({"a": None}, dict[str, type(None)], id="dict"),
        pytest.param((None, 1), tuple[type(None), int], id="tuple"),
    ],
)
def test_nonetype_annotation_round_trips(value, annotation):
    assert_round_trip(value, annotation)


@pytest.mark.parametrize(
    ("value", "annotation"),
    [
        pytest.param([None], list[None], id="list"),
        pytest.param({"a": None}, dict[str, None], id="dict"),
        pytest.param((1, None), tuple[int, None], id="tuple"),
        pytest.param(HasNoneField(None), HasNoneField, id="dataclass-field"),
    ],
)
def test_bare_none_annotation_is_accepted(value, annotation):
    assert_round_trip(value, annotation)


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        pytest.param(Rank.SECOND, '{"2": "x"}', id="int-valued-enum"),
        pytest.param(Priority.HIGH, '{"10": "x"}', id="int-enum"),
        pytest.param(TextMixin.UP, '{"up": "x"}', id="str-mixin-enum"),
    ],
)
def test_enum_keys_serialize_by_value(key, expected):
    assert serialize_json({key: "x"}) == expected


@pytest.mark.parametrize(
    ("value", "annotation", "member", "raw"),
    [
        pytest.param({Rank.SECOND: "x"}, dict[Rank, str], Rank.SECOND, 2, id="int-valued-enum"),
        pytest.param({Priority.HIGH: "x", Priority.LOW: "y"}, dict[Priority, str], Priority.HIGH, 10, id="int-enum"),
    ],
)
def test_int_valued_enum_keys_are_restored_as_members(value, annotation, member, raw):
    # a JSON key is always a string: "2" is looked up as the key text of Rank.SECOND, not as Rank("2")
    parsed = json.loads(serialize_json(value))
    assert all(isinstance(key, str) for key in parsed)
    assert deserialize_json(raw, type(member)) is member
    restored = assert_round_trip(value, annotation)
    assert member in restored


@pytest.mark.parametrize(
    ("value", "annotation"),
    [
        pytest.param({TextMixin.UP: 1}, dict[TextMixin, int], id="str-mixin-enum"),
        pytest.param({Quirky.EMPTY: 1, Quirky.NUMERIC: 2, Quirky.HEBREW: 3}, dict[Quirky, int], id="quirky-values"),
        pytest.param({}, dict[Color, int], id="empty"),
    ],
)
def test_str_valued_enum_keys_round_trip(value, annotation):
    assert_round_trip(value, annotation)


@pytest.mark.parametrize(
    ("value", "annotation"),
    [
        pytest.param({(1, "a"): 1}, dict[tuple[int, str], int], id="pair"),
        pytest.param({(): 1, ("x", "שלום"): 2}, dict[tuple[str, ...], int], id="variable-length"),
        pytest.param({(1, ("a", 2)): [1]}, dict[tuple[int, tuple[str, int]], list[int]], id="nested-tuple"),
        pytest.param(
            {(Color.RED, date(2024, 1, 2), Decimal("1.50")): "x"},
            dict[tuple[Color, date, Decimal], str],
            id="enum-date-decimal-elements",
        ),
        pytest.param({(1, "a"): 1}, typing.Dict[typing.Tuple[int, str], int], id="typing-dict"),
        pytest.param({(1, 2, 3): 1}, typing.Mapping[typing.Tuple[int, ...], int], id="typing-mapping-variable-length"),
    ],
)
def test_tuple_dict_keys_round_trip(value, annotation):
    assert_round_trip(value, annotation)


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(dict[tuple[int, str], int], id="dict"),
        pytest.param(typing.Dict[typing.Tuple[int, str], int], id="typing-dict"),
    ],
)
def test_tuple_dict_key_that_is_not_json_text_is_a_type_error(annotation):
    # a TypeError is what lets a union go on to its next member
    with pytest.raises(TypeError) as caught:
        deserialize_json({SECRET_TEXT: 1}, annotation)

    assert SECRET_TEXT not in str(caught.value)


def test_tuple_dict_keys_serialize_to_the_json_text_of_the_tuple():
    value = {(1, "a"): 1, (): 2, ("é", ("x", 2)): 3}

    assert serialize_json(value) == '{"[1, \\"a\\"]": 1, "[]": 2, "[\\"\\\\u00e9\\", [\\"x\\", 2]]": 3}'


def test_tuple_dict_key_text_holds_the_converted_elements():
    # the key text is the JSON of the tuple with every element converted like a value: enum value, ISO date, Decimal str
    value = {(Color.RED, date(2024, 1, 2), Decimal("1.50")): "x"}

    assert serialize_json(value) == '{"[\\"red\\", \\"2024-01-02\\", \\"1.50\\"]": "x"}'


@pytest.mark.parametrize(
    ("data", "annotation", "expected"),
    [
        pytest.param("2024-01-01", Union[int, date], "2024-01-01", id="int-first-keeps-an-iso-date-as-str"),
        pytest.param("2024-01-01", Union[date, int], date(2024, 1, 1), id="date-first-restores-the-date"),
        pytest.param(1, Union[float, int], 1.0, id="float-first-turns-an-int-into-a-float"),
        pytest.param("AAE=", Union[str, bytes], "AAE=", id="str-first-keeps-base64-as-str"),
        pytest.param("AAE=", Union[bytes, str], b"\x00\x01", id="bytes-first-decodes-any-valid-base64"),
    ],
)
def test_scalar_union_members_are_tried_in_order_limitation(data, annotation, expected):
    # Documents current behavior, not a promise: a primitive annotation (int, str, float) passes any JSON primitive
    # through, so the first scalar member decides. This is why the generated unions hold one array and one object.
    assert_strictly_equal(deserialize_json(data, annotation), expected)


@pytest.mark.parametrize(
    ("value", "annotation"),
    [
        pytest.param({1: "x"}, dict[int, str], id="int"),
        pytest.param({1.5: "x"}, dict[float, str], id="float"),
        pytest.param({True: "x"}, dict[bool, str], id="bool"),
    ],
)
def test_scalar_dict_keys_cannot_be_restored_limitation(value, annotation):
    # Documents current behavior, not a promise: JSON keys are strings, and only str, Enum and tuple keys are
    # converted back, so the other key types serialize fine and fail on restore with a ValueError.
    parsed = json.loads(serialize_json(value))
    assert list(parsed.values()) == ["x"]
    with pytest.raises(ValueError, match="de-convert"):
        deserialize_json(parsed, annotation)

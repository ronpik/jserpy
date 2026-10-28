"""Failure behavior on complex objects: where serialization and deserialization stop, and what they leave behind.

Serialization failures sit deep inside realistic structures (dataclass -> list -> dict -> tuple chains, sets of frozen
dataclasses, mapping proxies, ``Jsonable.to_json`` results): unsupported values and keys, key collisions, non-finite
floats with ``allow_nan=False`` and every kind of cycle, next to shared but acyclic references, which must serialize.
Deserialization failures sit deep inside a dataclass -> dict -> list -> union model: wrong container shapes, invalid
enum / date / Base64 / Decimal / Literal values, wrong tuple lengths, mapping annotations, paths, sets that cannot be
built and validation in ``__post_init__`` or ``from_json`` (which runs in a pinned order and stops at the first error).

Each failure is checked for its exact exception type and for messages that name types only: sentinel secrets planted in
the bad value, in valid neighbours of it and in mapping keys must not show up in the exception, its repr, args,
attributes, traceback or cause / context chain. Then for what it leaves behind: the input is unchanged, the same objects
serialize once fixed, later calls work, the deserialization guard is reset and threads do not interfere.

Strict xfails mark library defects, each narrowed to the exception it raises today. ``*_limitation`` tests document
current behavior that is neither promised nor clearly wrong.
"""

import copy
import dataclasses
import datetime
import enum
import gc
import json
import random
import re
import sys
import threading
import traceback
import weakref
from collections import OrderedDict, UserDict, namedtuple
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from decimal import Decimal
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Optional, Union

import pytest

from jserpy import (
    _numpy,
    deserialize_json,
    json_handler,
    serialize_json,
    serialize_json_as_dict,
    serialize_json_as_obj,
)
from jserpy.jsonable import Jsonable
from objects_support import assert_round_trip, assert_strictly_equal, to_json_value

SECRET = "SECRET-9f3a"  # in the bad value itself
SIBLING = "SIBLING-4d1c"  # in valid values next to the bad one: what a half-built result would hold
KEY_SECRET = "KEY-6e72"  # as a mapping key
SENTINELS = (SECRET, SIBLING, KEY_SECRET)

SERIALIZERS = [
    pytest.param(serialize_json, id="text"),
    pytest.param(serialize_json_as_obj, id="as-obj"),
    pytest.param(serialize_json_as_dict, id="as-dict"),
]

# Options that move the encoder to another code path (the pure Python one with ``indent``) or change the text
OPTIONS = [
    pytest.param({}, id="default"),
    pytest.param({"indent": 2}, id="indent"),
    pytest.param({"sort_keys": True}, id="sort-keys"),
    pytest.param({"ensure_ascii": False, "separators": (",", ":")}, id="compact-unicode"),
]


# --- classes used as values -----------------------------------------------------------------------------------------


class Opaque:
    """Supported by no handler. Its repr, str and attributes hold a secret that must never reach an error."""

    def __init__(self):
        self.secret = SECRET

    def __repr__(self):
        return f"Opaque({SECRET})"

    __str__ = __repr__


class Weird(enum.Enum):
    """An enum whose value no handler supports: the failure comes from the enum handler's output."""

    COMPLEX = complex(1, 2)


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class OtherColor(enum.Enum):
    RED = "red"


class LeakyKey(enum.Enum):
    KEY = KEY_SECRET  # as a mapping key it collides with the plain string KEY_SECRET


class Point(enum.Enum):
    ORIGIN = (0, 0)  # as a mapping key it becomes the key of the tuple (0, 0)


class Nothing(enum.Enum):
    NOTHING = None  # as a mapping key it becomes "null"


Duo = namedtuple("Duo", ["first", "second"])


class SubList(list):
    pass


@dataclass
class Cell:
    label: str
    link: Any = None


@dataclass
class Shelf:
    title: str
    items: Any


@dataclass(frozen=True)
class Stamp:
    """Hashable (so it can be a set member) although it carries mutable ``extras``."""

    name: str
    extras: Any = field(default=None, hash=False, compare=False)


class Echo(Jsonable):
    """``to_json`` returns whatever ``result`` holds, so a test can hide any value (also a cycle) behind a Jsonable."""

    def __init__(self, result: Any = None):
        self.result = result

    def to_json(self):
        return self.result

    @classmethod
    def from_json(cls, data):
        return cls(data)


class Pin(Jsonable):
    def __init__(self, label: str):
        self.label = label

    def to_json(self):
        return {"label": self.label}

    @classmethod
    def from_json(cls, data):
        return cls(data["label"])

    def __eq__(self, other):
        return type(other) is Pin and other.label == self.label


class Abort(BaseException):
    """Not an ``Exception``: nothing but ``BaseException`` handlers may catch it."""


# --- the deserialization model: dataclass -> dict -> list -> union -> dataclass -------------------------------------


@dataclass(frozen=True)
class Tag:
    label: str
    color: Color


@dataclass
class Record:
    color: Color
    day: datetime.date
    when: datetime.datetime
    at: datetime.time
    blob: bytes
    amount: Decimal
    kind: Literal["draft", "final"]
    pair: tuple[int, int]
    tags: frozenset[Tag]
    scores: list[float]
    index: dict[str, int]
    note: Optional[str] = None


@dataclass
class Page:
    records: list[Record]
    slots: dict[str, Optional[Record]]


@dataclass
class Book:
    title: str
    pages: list[Page]


@dataclass
class Library:
    name: str
    books: dict[str, Book]
    featured: Optional[Book]
    pinned: Record
    by_color: dict[Color, list[Record]]


@dataclass
class Options:
    color: Optional[Color] = None
    day: Optional[datetime.date] = None
    blob: Optional[bytes] = None
    amount: Optional[Decimal] = None
    kind: Optional[Literal["draft", "final"]] = None
    pair: Optional[tuple[int, int]] = None
    tags: Optional[frozenset[Tag]] = None
    scores: Optional[list[float]] = None
    record: Optional[Record] = None
    book: Optional[Book] = None
    index: Optional[dict[str, int]] = None


@dataclass
class Wrapper:
    items: list[dict[str, Options]]


Choice = Union[list[Union[Color, datetime.date]], dict[str, Union[Color, datetime.date]]]
Deeper = Union[list[Optional[Union[Color, Decimal]]], tuple[Union[datetime.date, bytes], ...]]


@dataclass
class Basket:
    choice: Choice


@dataclass
class Crate:
    baskets: dict[str, list[Basket]]


@dataclass
class Archive:
    entries: dict[str, list[Union[Book, dict[str, Any]]]]


@dataclass
class Grid:
    cells: dict[tuple[int, int], Tag]


@dataclass
class Empty:
    pass


@dataclass
class IndexHolder:
    index: dict[str, int]


@dataclass
class Derived:
    base: int
    doubled: int = field(init=False, default=0)

    def __post_init__(self):
        self.doubled = self.base * 2


@dataclass
class Node:
    label: str
    child: "Optional[Node]" = None


@dataclass
class Fork:
    left: Node
    right: Node


@dataclass
class Lists:
    first: list[int]
    second: list[int]


@dataclass
class Registry:
    by_name: dict[str, Tag]
    members: frozenset[Tag]
    others: dict[str, frozenset[Tag]]


@dataclass
class Bundle:
    pin_a: Pin
    pin_b: Pin
    proxy_a: MappingProxyType[str, int]
    proxy_b: MappingProxyType[str, int]
    pair_a: tuple[int, int]
    pair_b: tuple[int, int]


class Envelope(Jsonable):
    """Delegates its payload to ``deserialize_json``, as Jsonable implementations commonly do."""

    guard_seen: list[bool] = []

    def __init__(self, payload):
        self.payload = payload

    def to_json(self):
        return {"payload": self.payload}

    @classmethod
    def from_json(cls, data):
        cls.guard_seen.append(json_handler._deserializing.get())
        return cls(deserialize_json(data["payload"], Record))


class Outer(Jsonable):
    """Two levels of nested ``deserialize_json`` calls: Outer -> Envelope -> Record."""

    def __init__(self, inner):
        self.inner = inner

    def to_json(self):
        return {"inner": self.inner}

    @classmethod
    def from_json(cls, data):
        return cls(deserialize_json(data["inner"], Envelope))


class Lenient(Jsonable):
    """Swallows the TypeError of its nested call and keeps going."""

    def __init__(self, payload):
        self.payload = payload

    def to_json(self):
        return {"payload": self.payload}

    @classmethod
    def from_json(cls, data):
        try:
            payload = deserialize_json(data["payload"], Record)
        except TypeError:
            payload = None
        return cls(payload)


class Exploding(Jsonable):
    def to_json(self):
        return {}

    @classmethod
    def from_json(cls, data):
        raise data["raise"]


class Tracked(Jsonable):
    """Counts its live instances, to see whether a failed call keeps what it built."""

    instances: "weakref.WeakSet[Tracked]" = weakref.WeakSet()

    def __init__(self, tag):
        self.tag = tag
        Tracked.instances.add(self)

    def to_json(self):
        return {"tag": self.tag}

    @classmethod
    def from_json(cls, data):
        return cls(data["tag"])


@dataclass
class Trays:
    first: Tracked
    second: Record


@dataclass
class Views:
    by_name: Mapping[str, list[Tag]]
    frozen: MappingProxyType[str, Tag]


@dataclass
class Checked:
    """Validates in ``__post_init__`` with a ValueError, as validating dataclasses commonly do."""

    amount: int

    def __post_init__(self):
        if self.amount < 0:
            raise ValueError("negative amount")


@dataclass
class TypeChecked:
    """Validates in ``__post_init__`` with a TypeError, which makes it a member a union can reject."""

    amount: int

    def __post_init__(self):
        if not isinstance(self.amount, int):
            raise TypeError("amount must be an int")


CALLS: list[str] = []


class Probe(Jsonable):
    """Logs every ``from_json`` call and rejects (TypeError) a name that starts with "bad"."""

    def __init__(self, name: str):
        self.name = name

    def to_json(self):
        return {"name": self.name}

    @classmethod
    def from_json(cls, data):
        CALLS.append(data["name"])
        if data["name"].startswith("bad"):
            raise TypeError("rejected")
        return cls(data["name"])


class Spare(Jsonable):
    """Accepts any data, and logs it."""

    def to_json(self):
        return {}

    @classmethod
    def from_json(cls, data):
        CALLS.append("spare")
        return cls()


@dataclass
class Procession:
    first: Probe
    many: list[Probe]
    by_name: dict[str, Probe]
    last: Probe


# --- checking helpers -----------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_shared_state():
    """The probes below log into class and module level containers: every test starts and ends with them empty."""

    def clear():
        CALLS.clear()
        Envelope.guard_seen.clear()
        Tracked.instances.clear()

    clear()
    yield
    clear()


def exception_texts(error: BaseException) -> str:
    """Everything an exception can show: message, repr, args, attributes, notes and traceback, along its whole chain."""
    texts: list[str] = []
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        texts += [str(current), repr(current), repr(current.args), repr(vars(current))]
        texts += getattr(current, "__notes__", [])
        texts.append("".join(traceback.format_exception(type(current), current, current.__traceback__)))
        pending += [link for link in (current.__cause__, current.__context__) if link is not None]
    return "\n".join(texts)


def assert_no_secret(error: BaseException) -> None:
    texts = exception_texts(error)
    for sentinel in SENTINELS:
        assert sentinel not in texts, f"{sentinel} leaked into the {type(error).__name__}"


def assert_guard_clear() -> None:
    assert json_handler._deserializing.get() is False, "deserialize_json left its re-entrancy guard set"


_SCALARS = (str, int, float, bool, bytes, type(None), Decimal, datetime.date, enum.Enum, complex)


def _ref(node: Any) -> tuple:
    if isinstance(node, _SCALARS):
        return "scalar", type(node).__qualname__, repr(node)
    return "node", id(node)


def fingerprint(root: Any) -> list[tuple]:
    """Describe an object graph, cycles included, so that any change to it shows - also a replaced equal object."""
    described: list[tuple] = []
    seen: set[int] = set()
    pending = [root]
    while pending:
        node = pending.pop()
        if isinstance(node, _SCALARS) or id(node) in seen:
            continue
        seen.add(id(node))
        children: list[Any] = []
        parts: list[Any] = []
        if isinstance(node, Mapping):
            for key, value in node.items():
                children += [key, value]
                parts.append((_ref(key), _ref(value)))
        elif isinstance(node, (list, tuple, set, frozenset)):
            children = list(node)
            parts = [_ref(child) for child in children]
        elif dataclasses.is_dataclass(node) and not isinstance(node, type):
            for each in fields(node):
                children.append(getattr(node, each.name))
                parts.append((each.name, _ref(children[-1])))
        elif isinstance(node, Jsonable):
            for name, value in sorted(vars(node).items()):
                children.append(value)
                parts.append((name, _ref(value)))
        described.append((type(node).__qualname__, id(node), parts))
        pending.extend(children)
    return described


def failure_of(call, exception_type: type) -> BaseException:
    with pytest.raises(exception_type) as caught:
        call()
    assert type(caught.value) is exception_type
    return caught.value


# --- the checking helpers can fail ----------------------------------------------------------------------------------


class Quiet(Exception):
    """Hides its arguments from ``str`` and ``repr``."""

    def __str__(self):
        return "quiet"

    def __repr__(self):
        return "Quiet()"


def _caught(make) -> BaseException:
    try:
        make()
    except BaseException as error:
        return error
    raise AssertionError("nothing was raised")


def _raise_clean():
    raise TypeError("clean")


def _raise_from_secret():
    raise TypeError("clean") from ValueError(SECRET)


def _raise_in_handler_of_secret():
    try:
        raise ValueError(SECRET)
    except ValueError:
        raise TypeError("clean")


def _raise_deep_chain():
    try:
        try:
            raise ValueError(SECRET)
        except ValueError as inner:
            raise KeyError("middle") from inner
    except KeyError:
        raise TypeError("clean")


def _raise_with_attribute():
    error = TypeError("clean")
    error.doc = SECRET
    raise error


def _raise_with_hidden_args():
    raise Quiet(SECRET)


def _raise_with_note():
    error = TypeError("clean")
    error.add_note(SECRET)
    raise error


def _raise_with_secret_message():
    raise TypeError(SECRET)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(partial(_caught, _raise_with_secret_message), id="message"),
        pytest.param(partial(_caught, _raise_with_hidden_args), id="args-only"),
        pytest.param(partial(_caught, _raise_from_secret), id="cause"),
        pytest.param(partial(_caught, _raise_in_handler_of_secret), id="context"),
        pytest.param(partial(_caught, _raise_deep_chain), id="chain-of-three"),
        pytest.param(partial(_caught, _raise_with_attribute), id="attribute"),
        pytest.param(
            partial(_caught, _raise_with_note),
            id="note",
            marks=pytest.mark.skipif(not hasattr(BaseException, "add_note"), reason="exception notes need 3.11"),
        ),
    ],
)
def test_leak_detector_sees_a_secret_wherever_an_exception_keeps_it(make):
    with pytest.raises(AssertionError):
        assert_no_secret(make())


def test_leak_detector_accepts_a_clean_exception_and_shows_the_traceback():
    error = _caught(_raise_clean)

    assert_no_secret(error)
    assert "_raise_clean" in exception_texts(error)


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_leak_detector_knows_every_sentinel(sentinel):
    with pytest.raises(AssertionError):
        assert_no_secret(ValueError(f"prefix {sentinel} suffix"))


def _fingerprint_subject():
    inner = [1, {"k": (2, 3)}]
    cell = Cell(SIBLING, inner)
    root = {
        "cell": cell,
        "members": {1, 2},
        "proxy": MappingProxyType({"x": inner}),
        "echo": Echo(inner),
        "user": UserDict({"y": Duo(inner, 1)}),
    }
    return root, inner, cell


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda root, inner, cell: inner.append(4), id="list-append"),
        pytest.param(lambda root, inner, cell: inner[1].update(k=tuple([2, 3])), id="equal-tuple-swapped-in"),
        pytest.param(lambda root, inner, cell: setattr(cell, "link", list(inner)), id="field-replaced-by-equal-list"),
        pytest.param(lambda root, inner, cell: root["members"].add(3), id="set-grows"),
        pytest.param(lambda root, inner, cell: root.__setitem__("cell", root.pop("cell")), id="key-order-changes"),
        pytest.param(lambda root, inner, cell: setattr(root["echo"], "result", [1]), id="jsonable-attribute"),
        pytest.param(lambda root, inner, cell: inner.append(root), id="cycle-added"),
        pytest.param(lambda root, inner, cell: root["user"].update(y=Duo(inner, 1)), id="equal-namedtuple-swapped-in"),
        pytest.param(lambda root, inner, cell: root["user"].update(z=1), id="mapping-subclass-grows"),
    ],
)
def test_fingerprint_notices_changes(mutate):
    root, inner, cell = _fingerprint_subject()
    before = fingerprint(root)
    assert fingerprint(root) == before

    mutate(root, inner, cell)

    assert fingerprint(root) != before


# --- serialization: unsupported values deep inside chains -----------------------------------------------------------


def _generator():
    return (number for number in range(3))


UNSUPPORTED_LEAVES = [
    pytest.param(Opaque, "Opaque", id="custom-class"),
    pytest.param(lambda: complex(1, 2), "complex", id="complex"),
    pytest.param(lambda: (lambda: 1), "function", id="lambda"),
    pytest.param(lambda: json, "module", id="module"),
    pytest.param(_generator, "generator", id="generator"),
    pytest.param(lambda: Opaque, "type", id="class-object"),
    pytest.param(lambda: Weird.COMPLEX, "complex", id="enum-with-unsupported-value"),
]

DEPTHS = [pytest.param(1, id="depth-1"), pytest.param(3, id="depth-3"), pytest.param(6, id="depth-6")]

_LAYERS = ("dataclass", "list", "dict", "tuple")


def build_chain(leaf: Any, depth: int):
    """Wrap ``leaf`` in ``depth`` layers - dataclass, list, dict, tuple, dataclass, ... - each with a sibling secret.

    Return the root and a function that replaces the leaf in place (None if the innermost layer is a tuple).
    """
    value = leaf
    replace = None
    for level in reversed(range(depth)):
        kind = _LAYERS[level % len(_LAYERS)]
        if kind == "dataclass":
            container: Any = Shelf(SIBLING, value)
            setter = partial(setattr, container, "items")
        elif kind == "list":
            container = [SIBLING, value]
            setter = partial(container.__setitem__, 1)
        elif kind == "dict":
            container = {KEY_SECRET: SIBLING, "inner": value}
            setter = partial(container.__setitem__, "inner")
        else:
            container = (SIBLING, value)
            setter = None
        if level == depth - 1:
            replace = setter
        value = container
    return value, replace


def chain_json(leaf: Any, depth: int) -> Any:
    """The JSON value of ``build_chain(leaf, depth)``, written out layer by layer."""
    value = leaf
    for level in reversed(range(depth)):
        kind = _LAYERS[level % len(_LAYERS)]
        if kind == "dataclass":
            value = {"title": SIBLING, "items": value}
        elif kind == "dict":
            value = {KEY_SECRET: SIBLING, "inner": value}
        else:
            value = [SIBLING, value]
    return value


@pytest.mark.parametrize("depth", range(1, 9))
def test_chain_of_supported_values_serializes_to_the_layers_written_out(depth):
    root, _ = build_chain(1, depth)

    assert to_json_value(root) == chain_json(1, depth)
    assert serialize_json(root) == json.dumps(chain_json(1, depth))


@pytest.mark.parametrize("serializer", SERIALIZERS)
@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize(("make_leaf", "type_name"), UNSUPPORTED_LEAVES)
def test_unsupported_value_deep_in_a_chain_raises_type_error_naming_its_type(serializer, depth, make_leaf, type_name):
    root, _ = build_chain(make_leaf(), depth)
    before = fingerprint(root)

    with pytest.raises(TypeError, match=rf"\btype {type_name}\b") as caught:
        serializer(root)

    assert type(caught.value) is TypeError
    assert_no_secret(caught.value)
    assert fingerprint(root) == before


@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize(("make_leaf", "type_name"), UNSUPPORTED_LEAVES)
def test_replacing_the_unsupported_leaf_makes_the_same_objects_serialize(depth, make_leaf, type_name):
    root, replace = build_chain(make_leaf(), depth)
    messages = [str(failure_of(partial(serialize_json, root), TypeError)) for _ in range(2)]
    assert messages[0] == messages[1]

    replace(7)

    assert serialize_json(root) == json.dumps(chain_json(7, depth))
    assert serialize_json_as_obj(root) == chain_json(7, depth)


@pytest.mark.parametrize(
    ("make_value", "options", "exception_type"),
    [
        pytest.param(lambda: {KEY_SECRET: [Opaque()]}, {}, TypeError, id="unsupported-value"),
        pytest.param(
            lambda: Shelf(SIBLING, {KEY_SECRET: {"x": Opaque()}}), {}, TypeError, id="unsupported-in-dataclass"
        ),
        pytest.param(
            lambda: MappingProxyType({KEY_SECRET: Opaque()}), {}, TypeError, id="unsupported-in-mapping-proxy"
        ),
        pytest.param(lambda: Echo({KEY_SECRET: [Opaque()]}), {}, TypeError, id="unsupported-in-jsonable-result"),
        pytest.param(
            lambda: {Stamp("a", extras={KEY_SECRET: Opaque()})}, {}, TypeError, id="unsupported-in-set-member"
        ),
        pytest.param(lambda: {KEY_SECRET: [float("nan")]}, {"allow_nan": False}, ValueError, id="nan"),
        pytest.param(
            lambda: {KEY_SECRET: Stamp("a", extras={1: "left", "1": "right"})}, {}, ValueError, id="key-collision"
        ),
    ],
)
def test_failure_below_a_secret_dict_key_does_not_name_the_key(make_value, options, exception_type):
    error = failure_of(partial(serialize_json, make_value(), **options), exception_type)

    assert_no_secret(error)


def test_unsupported_value_in_the_last_of_many_siblings_leaves_no_partial_text():
    # Everything before the bad value is serializable: none of it may come back as a result or inside the error
    items = [Shelf(SIBLING, [index, {"k": index}]) for index in range(200)] + [Shelf(SIBLING, Opaque())]
    before = fingerprint(items)

    error = failure_of(partial(serialize_json, items), TypeError)

    assert_no_secret(error)
    assert "Shelf" not in str(error)
    assert fingerprint(items) == before


# --- serialization: unsupported and colliding mapping keys inside containers ----------------------------------------


def _unsupported_key(name: str) -> str:
    return rf"Unsupported JSON mapping key type: [\w.]*{name}\b"


BAD_MAPPINGS = [
    pytest.param(lambda: {Opaque(): 1}, TypeError, _unsupported_key("Opaque"), id="custom-class-key"),
    pytest.param(lambda: {b"raw": 1}, TypeError, _unsupported_key("bytes"), id="bytes-key"),
    pytest.param(lambda: {None: 1}, TypeError, _unsupported_key("NoneType"), id="none-key"),
    pytest.param(lambda: {Decimal("1.5"): 1}, TypeError, _unsupported_key("Decimal"), id="decimal-key"),
    pytest.param(lambda: {frozenset({1}): 1}, TypeError, _unsupported_key("frozenset"), id="frozenset-key"),
    pytest.param(lambda: {datetime.date(2026, 10, 27): 1}, TypeError, _unsupported_key("date"), id="date-key"),
    pytest.param(lambda: {Stamp("k"): 1}, TypeError, _unsupported_key("Stamp"), id="frozen-dataclass-key"),
    pytest.param(lambda: {(1, Opaque()): 1}, TypeError, r"\btype Opaque\b", id="tuple-key-with-unsupported-member"),
    pytest.param(
        lambda: {Weird.COMPLEX: 1}, TypeError, _unsupported_key("complex"), id="enum-key-with-unsupported-value"
    ),
    pytest.param(lambda: {1: "left", "1": "right"}, ValueError, "collision", id="int-vs-str"),
    pytest.param(lambda: {True: "left", "true": "right"}, ValueError, "collision", id="bool-vs-str"),
    pytest.param(lambda: {1.5: "left", "1.5": "right"}, ValueError, "collision", id="float-vs-str"),
    pytest.param(lambda: {Color.RED: "left", "red": "right"}, ValueError, "collision", id="enum-vs-str"),
    pytest.param(lambda: {Color.RED: "left", OtherColor.RED: "right"}, ValueError, "collision", id="enum-vs-enum"),
    pytest.param(lambda: {(1, "x"): "left", '[1, "x"]': "right"}, ValueError, "collision", id="tuple-vs-str"),
    pytest.param(
        lambda: {Point.ORIGIN: "left", (0, 0): "right"}, ValueError, "collision", id="enum-with-tuple-value-vs-tuple"
    ),
    pytest.param(
        lambda: {Nothing.NOTHING: "left", "null": "right"}, ValueError, "collision", id="enum-with-null-value-vs-str"
    ),
    pytest.param(lambda: {LeakyKey.KEY: "left", KEY_SECRET: "right"}, ValueError, "collision", id="secret-key-twice"),
]


def valid_mapping() -> dict:
    return {
        "ok": 1,
        2: "two",
        (1, 2): "pair",
        Color.RED: "red",
        True: "yes",
        1.5: "float",
        Point.ORIGIN: "origin",
        Nothing.NOTHING: "nothing",
    }


VALID_MAPPING_JSON = {
    "ok": 1,
    "2": "two",
    "[1, 2]": "pair",
    "red": "red",
    "true": "yes",
    "1.5": "float",
    "[0, 0]": "origin",
    "null": "nothing",
}

# (id, wrap a mapping in a host, the JSON value of the host around a mapping's JSON value)
HOSTS = [
    pytest.param(lambda m: m, lambda j: j, id="root"),
    pytest.param(
        lambda m: {"tags": [{Stamp("a", extras={"deep": [m]}), Stamp("b")}]},
        # set members are ordered by their canonical text: ``"extras":null`` sorts before ``"extras":{``
        lambda j: {"tags": [[{"extras": None, "name": "b"}, {"extras": {"deep": [j]}, "name": "a"}]]},
        id="set-of-frozen-dataclasses",
    ),
    pytest.param(
        lambda m: frozenset({Stamp("a", extras=m)}),
        lambda j: [{"extras": j, "name": "a"}],
        id="frozenset-root",
    ),
    pytest.param(
        lambda m: MappingProxyType({"outer": [MappingProxyType({"inner": m})]}),
        lambda j: {"outer": [{"inner": j}]},
        id="mapping-proxies",
    ),
    pytest.param(lambda m: [Echo({"wrapped": [m]})], lambda j: [{"wrapped": [j]}], id="jsonable-result"),
    pytest.param(lambda m: build_chain(m, 6)[0], lambda j: chain_json(j, 6), id="chain-depth-6"),
    pytest.param(
        lambda m: Shelf(SIBLING, (SIBLING, m)),
        lambda j: {"title": SIBLING, "items": [SIBLING, j]},
        id="dataclass-tuple-field",
    ),
]


HOST_WRAPPERS = {each.id: each.values[0] for each in HOSTS}
HOST_PARAMS = [pytest.param(wrap, id=name) for name, wrap in HOST_WRAPPERS.items()]


@pytest.mark.parametrize(("wrap", "expected"), HOSTS)
def test_hosts_serialize_a_valid_mapping(wrap, expected):
    assert to_json_value(wrap(valid_mapping())) == expected(VALID_MAPPING_JSON)


@pytest.mark.parametrize("wrap", HOST_PARAMS)
@pytest.mark.parametrize(("make_mapping", "exception_type", "pattern"), BAD_MAPPINGS)
def test_bad_mapping_keys_deep_inside_containers(wrap, make_mapping, exception_type, pattern):
    value = wrap(make_mapping())
    before = fingerprint(value)

    with pytest.raises(exception_type, match=pattern) as caught:
        serialize_json(value)

    assert type(caught.value) is exception_type
    assert_no_secret(caught.value)
    assert fingerprint(value) == before


@pytest.mark.parametrize("serializer", SERIALIZERS)
@pytest.mark.parametrize(
    ("make_mapping", "exception_type"),
    [
        pytest.param(lambda: {Opaque(): 1}, TypeError, id="unsupported-key"),
        pytest.param(lambda: {LeakyKey.KEY: 1, KEY_SECRET: 2}, ValueError, id="collision"),
    ],
)
def test_the_public_serializers_fail_alike_on_bad_keys(serializer, make_mapping, exception_type):
    value = [Echo({"x": [Stamp("a", extras={"deep": make_mapping()})]})]

    reference = failure_of(partial(serialize_json, value), exception_type)
    error = failure_of(partial(serializer, value), exception_type)

    assert str(error) == str(reference)
    assert_no_secret(error)


@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize(
    "wrap",
    [
        pytest.param(HOST_WRAPPERS[name], id=name)
        for name in ("set-of-frozen-dataclasses", "mapping-proxies", "jsonable-result")
    ],
)
def test_colliding_keys_fail_under_every_encoder_option(wrap, options):
    value = wrap({LeakyKey.KEY: 1, KEY_SECRET: 2})

    error = failure_of(partial(serialize_json, value, **options), ValueError)

    assert "collision" in str(error)
    assert_no_secret(error)


def test_removing_the_colliding_key_makes_the_same_objects_serialize():
    mapping = {LeakyKey.KEY: 1, KEY_SECRET: 2, "other": 3}
    value = [Echo({"wrapped": MappingProxyType({"m": mapping})}), {Stamp("a", extras=mapping)}]
    failure_of(partial(serialize_json, value), ValueError)

    del mapping[KEY_SECRET]

    assert to_json_value(value) == [
        {"wrapped": {"m": {KEY_SECRET: 1, "other": 3}}},
        [{"extras": {KEY_SECRET: 1, "other": 3}, "name": "a"}],
    ]


# --- serialization: unsupported values inside every kind of host ----------------------------------------------------


@pytest.mark.parametrize("wrap", HOST_PARAMS)
@pytest.mark.parametrize(("make_leaf", "type_name"), UNSUPPORTED_LEAVES)
def test_unsupported_value_inside_every_kind_of_host_raises_type_error_naming_its_type(wrap, make_leaf, type_name):
    value = wrap(make_leaf())
    before = fingerprint(value)

    with pytest.raises(TypeError, match=rf"\btype {type_name}\b") as caught:
        serialize_json(value)

    assert type(caught.value) is TypeError
    assert_no_secret(caught.value)
    assert fingerprint(value) == before


DEEP_CHAIN = 150  # far past the 6 levels above, and below what the encoder of every supported Python handles


@pytest.mark.parametrize(
    ("make_leaf", "type_name"),
    [leaf for leaf in UNSUPPORTED_LEAVES if leaf.id in ("custom-class", "lambda")],
)
def test_unsupported_value_at_the_bottom_of_a_deep_chain_is_still_a_type_error(make_leaf, type_name):
    root, replace = build_chain(make_leaf(), DEEP_CHAIN)
    before = fingerprint(root)

    with pytest.raises(TypeError, match=rf"\btype {type_name}\b") as caught:
        serialize_json(root)

    assert type(caught.value) is TypeError
    assert_no_secret(caught.value)
    assert fingerprint(root) == before
    replace(7)  # the chain itself is fine: the leaf was the only problem
    assert to_json_value(root) == chain_json(7, DEEP_CHAIN)


@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize(
    ("make_value", "extra_options", "exception_type", "pattern"),
    [
        pytest.param(
            lambda: build_chain(Opaque(), 6)[0], {}, TypeError, r"\btype Opaque\b", id="unsupported-value-in-a-chain"
        ),
        pytest.param(
            lambda: [Echo({"v": (Opaque(),)})], {}, TypeError, r"\btype Opaque\b", id="unsupported-value-in-a-jsonable"
        ),
        pytest.param(
            lambda: build_chain({float("nan")}, 6)[0], {"allow_nan": False}, ValueError, "not JSON compliant", id="nan"
        ),
        pytest.param(
            lambda: [Echo({"v": (float("-inf"),)})],
            {"allow_nan": False},
            ValueError,
            "not JSON compliant",
            id="neg-inf-in-a-jsonable",
        ),
    ],
)
def test_unsupported_and_non_finite_values_fail_under_every_encoder_option(
    options, make_value, extra_options, exception_type, pattern
):
    value = make_value()
    before = fingerprint(value)

    with pytest.raises(exception_type, match=pattern) as caught:
        serialize_json(value, **options, **extra_options)

    assert type(caught.value) is exception_type
    assert_no_secret(caught.value)
    assert fingerprint(value) == before


# --- serialization: non-finite floats with allow_nan=False ----------------------------------------------------------

NON_FINITE = [
    pytest.param(float("nan"), "NaN", id="nan"),
    pytest.param(float("inf"), "Infinity", id="inf"),
    pytest.param(float("-inf"), "-Infinity", id="neg-inf"),
]

TOKEN_VALUES = {"NaN": float("nan"), "Infinity": float("inf"), "-Infinity": float("-inf")}


def _chain_text(depth: int):
    # json.dumps writes NaN / Infinity / -Infinity for these floats, and for the finite 1.5 its repr
    return lambda token: json.dumps(chain_json(TOKEN_VALUES[token] if token in TOKEN_VALUES else float(token), depth))


# (host around a float, the text of the host around the float's JSON token): the float is the only thing that can fail
FLOAT_HOSTS = [
    pytest.param(lambda x: build_chain(x, 1)[0], _chain_text(1), id="chain-depth-1"),
    pytest.param(lambda x: build_chain(x, 3)[0], _chain_text(3), id="chain-depth-3"),
    pytest.param(lambda x: build_chain(x, 6)[0], _chain_text(6), id="chain-depth-6"),
    pytest.param(lambda x: {x}, lambda t: f"[{t}]", id="set"),
    pytest.param(lambda x: frozenset({x}), lambda t: f"[{t}]", id="frozenset"),
    pytest.param(lambda x: MappingProxyType({"v": [x]}), lambda t: f'{{"v": [{t}]}}', id="mapping-proxy"),
    pytest.param(lambda x: Echo({"v": (x,)}), lambda t: f'{{"v": [{t}]}}', id="jsonable-result"),
    pytest.param(lambda x: {x: 1}, lambda t: f'{{"{t}": 1}}', id="mapping-key"),
    pytest.param(lambda x: {(1, x): "t"}, lambda t: f'{{"[1, {t}]": "t"}}', id="tuple-key"),
    pytest.param(
        lambda x: [Stamp("a", extras=[x])], lambda t: f'[{{"name": "a", "extras": [{t}]}}]', id="dataclass-in-list"
    ),
]


@pytest.mark.parametrize(("make_host", "text_of"), FLOAT_HOSTS)
@pytest.mark.parametrize(("number", "token"), NON_FINITE)
def test_non_finite_floats_serialize_by_default(make_host, text_of, number, token):
    assert serialize_json(make_host(number)) == text_of(token)


@pytest.mark.parametrize(("make_host", "text_of"), FLOAT_HOSTS)
@pytest.mark.parametrize(("number", "token"), NON_FINITE)
def test_allow_nan_false_rejects_non_finite_floats_anywhere(make_host, text_of, number, token):
    value = make_host(number)
    before = fingerprint(value)

    with pytest.raises(ValueError, match="not JSON compliant") as caught:
        serialize_json(value, allow_nan=False)

    assert type(caught.value) is ValueError
    assert_no_secret(caught.value)
    assert fingerprint(value) == before
    assert serialize_json(value) == text_of(token)  # the same objects serialize once the option is gone


@pytest.mark.parametrize(("make_host", "text_of"), FLOAT_HOSTS)
def test_allow_nan_false_accepts_finite_floats_in_the_same_places(make_host, text_of):
    value = make_host(1.5)

    assert serialize_json(value, allow_nan=False) == text_of("1.5")


@pytest.mark.parametrize(
    ("numbers", "text"),
    [
        pytest.param({1.5, float("nan")}, "[1.5, NaN]", id="nan"),
        pytest.param({1.5, float("inf")}, "[1.5, Infinity]", id="inf"),
        pytest.param({1.5, float("-inf")}, "[-Infinity, 1.5]", id="neg-inf"),
    ],
)
def test_sets_with_non_finite_floats_are_ordered_by_canonical_text(numbers, text):
    assert serialize_json(numbers) == text
    with pytest.raises(ValueError, match="not JSON compliant"):
        serialize_json(numbers, allow_nan=False)


def test_fallback_result_with_nan_is_rejected_by_allow_nan_false():
    opaque = Opaque()

    def fallback(value):
        return [float("nan")]

    assert serialize_json([opaque], fallback=fallback) == "[[NaN]]"
    with pytest.raises(ValueError, match="not JSON compliant") as caught:
        serialize_json([opaque], fallback=fallback, allow_nan=False)
    assert_no_secret(caught.value)


# --- serialization: cycles of every kind ----------------------------------------------------------------------------

# Each factory returns (root, repair, JSON value of root once repaired): repair() breaks the cycle in place.


def _list_contains_itself():
    items = [SIBLING]
    items.append(items)
    return items, items.pop, [SIBLING]


def _dict_contains_itself():
    mapping = {"label": SIBLING}
    mapping[KEY_SECRET] = mapping
    return mapping, partial(mapping.pop, KEY_SECRET), {"label": SIBLING}


def _cyclic_list_in_a_tuple_in_a_chain():
    loop = [SIBLING]
    loop.append(loop)
    return {"a": [{"b": (loop,)}]}, loop.pop, {"a": [{"b": [[SIBLING]]}]}


def _dataclass_points_to_itself():
    cell = Cell(SIBLING)
    cell.link = cell
    return cell, partial(setattr, cell, "link", None), {"label": SIBLING, "link": None}


def _two_dataclasses():
    first = Cell(SIBLING)
    second = Cell(SIBLING, first)
    first.link = second
    return first, partial(setattr, second, "link", None), {
        "label": SIBLING,
        "link": {"label": SIBLING, "link": None},
    }


def _tuple_holding_a_list_holding_the_owner():
    owner = Cell(SIBLING)
    holder = [owner]
    owner.link = (holder,)
    return owner, holder.clear, {"label": SIBLING, "link": [[]]}


def _mapping_proxy_in_its_own_backing_dict():
    backing = {"label": SIBLING}
    proxy = MappingProxyType(backing)
    backing["me"] = proxy
    return proxy, partial(backing.pop, "me"), {"label": SIBLING}


def _dataclass_in_its_own_mapping_proxy():
    cell = Cell(SIBLING)
    cell.link = MappingProxyType({"owner": cell})
    return [cell], partial(setattr, cell, "link", None), [{"label": SIBLING, "link": None}]


def _jsonable_pair():
    first, second = Echo(), Echo()
    first.result = {"peer": second}
    second.result = {"peer": first}
    return first, partial(setattr, second, "result", {"peer": None}), {"peer": {"peer": None}}


def _jsonable_returns_itself():
    echo = Echo()
    echo.result = echo
    return {"x": echo}, partial(setattr, echo, "result", {"ok": SIBLING}), {"x": {"ok": SIBLING}}


def _jsonable_returns_a_list_holding_itself():
    echo = Echo()
    echo.result = [SIBLING, echo]
    return echo, partial(setattr, echo, "result", [SIBLING]), [SIBLING]


def _jsonable_result_reaches_its_owner_through_a_dataclass():
    echo = Echo()
    echo.result = Cell(SIBLING, [echo])
    return [echo], echo.result.link.clear, [{"label": SIBLING, "link": []}]


def _set_member_reaches_its_set():
    member = Stamp("m", extras=[SIBLING])
    members = {member}
    member.extras.append(members)
    return {"set": members}, member.extras.pop, {"set": [{"extras": [SIBLING], "name": "m"}]}


def _frozenset_member_reaches_its_frozenset():
    member = Stamp("m", extras=[])
    members = frozenset([member])
    member.extras.append(members)
    return [members], member.extras.clear, [[{"extras": [], "name": "m"}]]


def _set_member_contains_itself():
    member = Stamp("m", extras=[])
    member.extras.append(member)
    return {member}, member.extras.clear, [{"extras": [], "name": "m"}]


def _set_member_with_a_cycle_of_its_own():
    loop = [SIBLING]
    loop.append(loop)
    return [{Stamp("m", extras=loop)}, "tail"], loop.pop, [[{"extras": [SIBLING], "name": "m"}], "tail"]


def _user_dict_contains_itself():
    # A Mapping that is not a dict
    mapping = UserDict(label=SIBLING)
    mapping[KEY_SECRET] = mapping
    return mapping, partial(mapping.pop, KEY_SECRET), {"label": SIBLING}


def _ordered_dict_contains_itself():
    mapping = OrderedDict(label=SIBLING)
    mapping[KEY_SECRET] = mapping
    return mapping, partial(mapping.pop, KEY_SECRET), {"label": SIBLING}


def _namedtuple_holding_a_list_holding_itself():
    duo = Duo([SIBLING], 1)
    duo.first.append(duo)
    return {"duo": duo}, duo.first.pop, {"duo": [[SIBLING], 1]}


def _list_subclass_contains_itself():
    items = SubList([SIBLING])
    items.append(items)
    return items, items.pop, [SIBLING]


def _list_ring(length):
    ring = [[SIBLING] for _ in range(length)]
    for index, node in enumerate(ring):
        node.append(ring[(index + 1) % length])
    expected: Any = [SIBLING]
    for _ in range(length - 1):
        expected = [SIBLING, expected]
    return ring[0], ring[-1].pop, expected


def _dict_ring(length):
    ring = [{"label": SIBLING} for _ in range(length)]
    for index, node in enumerate(ring):
        node["next"] = ring[(index + 1) % length]
    expected: Any = {"label": SIBLING}
    for _ in range(length - 1):
        expected = {"label": SIBLING, "next": expected}
    return ring[0], partial(ring[-1].pop, "next"), expected


def _cell_levels(levels):
    expected = None
    for _ in range(levels):
        expected = {"label": SIBLING, "link": expected}
    return expected


def _dataclass_ring(length):
    ring = [Cell(SIBLING) for _ in range(length)]
    for index, node in enumerate(ring):
        node.link = ring[(index + 1) % length]
    return ring[0], partial(setattr, ring[-1], "link", None), _cell_levels(length)


def _dataclass_ring_below_a_long_tail(tail_length, ring_length):
    ring = [Cell(SIBLING) for _ in range(ring_length)]
    for index, node in enumerate(ring):
        node.link = ring[(index + 1) % ring_length]
    head = ring[0]
    for _ in range(tail_length):
        head = Cell(SIBLING, head)
    return head, partial(setattr, ring[-1], "link", None), _cell_levels(tail_length + ring_length)


CYCLES = [
    pytest.param(_list_contains_itself, id="list-contains-itself"),
    pytest.param(_dict_contains_itself, id="dict-contains-itself"),
    pytest.param(_cyclic_list_in_a_tuple_in_a_chain, id="cyclic-list-in-a-tuple-in-a-chain"),
    pytest.param(_dataclass_points_to_itself, id="dataclass-points-to-itself"),
    pytest.param(_two_dataclasses, id="dataclass-a-b-a"),
    pytest.param(_tuple_holding_a_list_holding_the_owner, id="tuple-holding-a-list-holding-the-owner"),
    pytest.param(_mapping_proxy_in_its_own_backing_dict, id="mapping-proxy-in-its-backing-dict"),
    pytest.param(_dataclass_in_its_own_mapping_proxy, id="dataclass-in-its-own-mapping-proxy"),
    pytest.param(_jsonable_pair, id="jsonable-a-b-a"),
    pytest.param(_jsonable_returns_itself, id="jsonable-returns-itself"),
    pytest.param(_jsonable_returns_a_list_holding_itself, id="jsonable-returns-a-list-holding-itself"),
    pytest.param(_jsonable_result_reaches_its_owner_through_a_dataclass, id="jsonable-result-through-a-dataclass"),
    pytest.param(_set_member_reaches_its_set, id="set-member-reaches-its-set"),
    pytest.param(_frozenset_member_reaches_its_frozenset, id="frozenset-member-reaches-its-frozenset"),
    pytest.param(_set_member_contains_itself, id="set-member-contains-itself"),
    pytest.param(_set_member_with_a_cycle_of_its_own, id="set-member-with-its-own-cycle"),
    pytest.param(_user_dict_contains_itself, id="user-dict-contains-itself"),
    pytest.param(_ordered_dict_contains_itself, id="ordered-dict-contains-itself"),
    pytest.param(_namedtuple_holding_a_list_holding_itself, id="namedtuple-holding-a-list-holding-itself"),
    pytest.param(_list_subclass_contains_itself, id="list-subclass-contains-itself"),
    pytest.param(partial(_list_ring, 1), id="list-ring-of-1"),
    pytest.param(partial(_list_ring, 50), id="list-ring-of-50"),
    pytest.param(partial(_dict_ring, 1), id="dict-ring-of-1"),
    pytest.param(partial(_dict_ring, 50), id="dict-ring-of-50"),
    pytest.param(partial(_dataclass_ring, 1), id="dataclass-ring-of-1"),
    pytest.param(partial(_dataclass_ring, 50), id="dataclass-ring-of-50"),
    pytest.param(partial(_dataclass_ring_below_a_long_tail, 50, 2), id="dataclass-ring-50-levels-down"),
]


@pytest.mark.parametrize("serializer", SERIALIZERS)
@pytest.mark.parametrize("make_cycle", CYCLES)
def test_every_kind_of_cycle_raises_value_error_without_payload(serializer, make_cycle):
    root, _, _ = make_cycle()
    before = fingerprint(root)

    with pytest.raises(ValueError, match="Cyclic reference") as caught:
        serializer(root)

    assert type(caught.value) is ValueError
    assert_no_secret(caught.value)
    assert fingerprint(root) == before


@pytest.mark.parametrize("make_cycle", CYCLES)
def test_breaking_the_cycle_makes_the_same_objects_serialize(make_cycle):
    root, repair, expected = make_cycle()
    messages = [str(failure_of(partial(serialize_json, root), ValueError)) for _ in range(2)]
    assert messages[0] == messages[1]

    repair()

    assert to_json_value(root) == expected
    assert serialize_json_as_obj(root) == expected
    assert serialize_json(root) == serialize_json(root)


@pytest.mark.parametrize("options", OPTIONS)
@pytest.mark.parametrize(
    "make_cycle",
    [
        pytest.param(_jsonable_pair, id="jsonable-a-b-a"),
        pytest.param(_two_dataclasses, id="dataclass-a-b-a"),
        pytest.param(_set_member_reaches_its_set, id="set-member-reaches-its-set"),
        pytest.param(partial(_dataclass_ring, 50), id="dataclass-ring-of-50"),
        pytest.param(_list_contains_itself, id="list-contains-itself"),
    ],
)
def test_cycles_fail_the_same_way_under_every_encoder_option(make_cycle, options):
    root, repair, expected = make_cycle()

    error = failure_of(partial(serialize_json, root, **options), ValueError)

    assert "Cyclic reference" in str(error)
    assert_no_secret(error)
    repair()
    assert json.loads(serialize_json(root, **options)) == expected


def test_fallback_that_embeds_the_unsupported_value_again_is_a_cycle():
    value = {"x": [Opaque()]}

    error = failure_of(partial(serialize_json, value, fallback=lambda unsupported: {"again": unsupported}), ValueError)

    assert "Cyclic reference" in str(error)
    assert_no_secret(error)


# --- serialization: shared but acyclic references serialize ---------------------------------------------------------


def _diamond():
    bottom = Node("bottom")
    value = Fork(Node("left", bottom), Node("right", bottom))
    leg = {"label": "bottom", "child": None}
    return value, Fork, {"left": {"label": "left", "child": leg}, "right": {"label": "right", "child": leg}}


def _same_list_in_two_fields():
    shared = [1, 2, 3]
    return Lists(shared, shared), Lists, {"first": [1, 2, 3], "second": [1, 2, 3]}


def _same_object_as_dict_value_and_set_member():
    tag = Tag("t", Color.RED)
    value = Registry({"t": tag}, frozenset({tag}), {"a": frozenset({tag}), "b": frozenset({tag})})
    tag_json = {"label": "t", "color": "red"}
    expected = {"by_name": {"t": tag_json}, "members": [tag_json], "others": {"a": [tag_json], "b": [tag_json]}}
    return value, Registry, expected


def _same_jsonable_proxy_and_tuple_twice():
    pin, proxy, pair = Pin("p"), MappingProxyType({"n": 1}), (1, 2)
    value = Bundle(pin, pin, proxy, proxy, pair, pair)
    return value, Bundle, {
        "pin_a": {"label": "p"},
        "pin_b": {"label": "p"},
        "proxy_a": {"n": 1},
        "proxy_b": {"n": 1},
        "pair_a": [1, 2],
        "pair_b": [1, 2],
    }


def _deep_structure_under_two_fields():
    deep = None
    expected = None
    for _ in range(50):
        deep = Node("n", deep)
        expected = {"label": "n", "child": expected}
    return Fork(deep, deep), Fork, {"left": expected, "right": expected}


def _same_frozen_dataclass_in_sets_of_two_fields():
    tag = Tag("t", Color.BLUE)
    value = Registry({}, frozenset({tag}), {"first": frozenset({tag}), "second": frozenset({tag})})
    tag_json = {"label": "t", "color": "blue"}
    expected = {"by_name": {}, "members": [tag_json], "others": {"first": [tag_json], "second": [tag_json]}}
    return value, Registry, expected


SHARED = [
    pytest.param(_diamond, id="diamond"),
    pytest.param(_same_list_in_two_fields, id="same-list-in-two-fields"),
    pytest.param(_same_object_as_dict_value_and_set_member, id="dict-value-and-set-member"),
    pytest.param(_same_jsonable_proxy_and_tuple_twice, id="jsonable-proxy-and-tuple-twice"),
    pytest.param(_deep_structure_under_two_fields, id="deep-structure-under-two-fields"),
    pytest.param(_same_frozen_dataclass_in_sets_of_two_fields, id="frozen-dataclass-in-two-sets"),
]


@pytest.mark.parametrize("make_shared", SHARED)
def test_shared_acyclic_references_serialize_and_round_trip(make_shared):
    value, cls, expected = make_shared()

    assert to_json_value(value) == expected
    assert serialize_json_as_obj(value) == expected
    assert_round_trip(value, cls)


def _same_tuple_as_key_value_and_set_member():
    pair = (1, 2)
    value = {pair: pair, "set": {pair}, "list": [pair, pair]}
    return value, {"[1, 2]": [1, 2], "set": [[1, 2]], "list": [[1, 2], [1, 2]]}


def _same_enum_member_as_key_value_and_set_member():
    return {Color.RED: Color.RED, "set": {Color.RED}}, {"red": "red", "set": ["red"]}


def _same_frozenset_inside_and_outside_a_set():
    inner = frozenset({1, 2})
    # the members of the outer set are ordered by their canonical text: "[1,2]" before "[3]"
    return [inner, {inner, frozenset({3})}, inner], [[1, 2], [[1, 2], [3]], [1, 2]]


def _same_list_as_a_value_and_as_two_jsonable_results():
    shared = [1, 2]
    return [shared, Echo(shared), {"k": Echo(shared)}], [[1, 2], [1, 2], {"k": [1, 2]}]


@pytest.mark.parametrize(
    "make_shared",
    [
        pytest.param(_same_tuple_as_key_value_and_set_member, id="tuple-as-key-value-and-set-member"),
        pytest.param(_same_enum_member_as_key_value_and_set_member, id="enum-member-as-key-value-and-set-member"),
        pytest.param(_same_frozenset_inside_and_outside_a_set, id="frozenset-inside-and-outside-a-set"),
        pytest.param(_same_list_as_a_value_and_as_two_jsonable_results, id="list-as-value-and-jsonable-results"),
    ],
)
def test_shared_acyclic_references_in_unusual_places_serialize(make_shared):
    value, expected = make_shared()

    assert to_json_value(value) == expected
    assert serialize_json_as_obj(value) == expected
    assert serialize_json(value) == serialize_json(value)


def test_shared_references_serialize_after_a_failure_that_involved_them():
    shared = [Cell(SIBLING, [1, 2])]
    failing = {"shared": shared, "cycle": shared}
    shared.append(failing)
    failure_of(partial(serialize_json, failing), ValueError)

    shared.pop()

    assert to_json_value(failing) == {
        "shared": [{"label": SIBLING, "link": [1, 2]}],
        "cycle": [{"label": SIBLING, "link": [1, 2]}],
    }


# --- serialization: random object graphs, cyclic or not -------------------------------------------------------------


class GraphCycle(Exception):
    pass


def reference_json(node: Any, path: frozenset) -> Any:
    """What a graph serializes to, written as a plain recursive walk that remembers the nodes on the current path."""
    if node is None or isinstance(node, (int, str)):
        return node
    if id(node) in path:
        raise GraphCycle
    path = path | {id(node)}
    if isinstance(node, (list, tuple)):
        return [reference_json(item, path) for item in node]
    if isinstance(node, (dict, MappingProxyType)):
        return {key: reference_json(item, path) for key, item in node.items()}
    if isinstance(node, Cell):
        return {"label": node.label, "link": reference_json(node.link, path)}
    if isinstance(node, Echo):
        return reference_json(node.result, path)
    if isinstance(node, Stamp):
        return {"name": node.name, "extras": reference_json(node.extras, path)}
    members = [reference_json(member, path) for member in node]  # a set
    return sorted(members, key=lambda member: json.dumps(member, sort_keys=True, separators=(",", ":")))


def random_graph(rng: random.Random, size: int, back_edges: float) -> Any:
    """Link ``size`` containers of mixed kinds at random: mostly onward (sharing, no cycle), now and then backward."""
    holders: list[Any] = []  # what children go into, per node
    nodes: list[Any] = []
    for index in range(size):
        kind = rng.choice(["list", "tuple-in-list", "dict", "dataclass", "jsonable", "proxy", "set"])
        if kind in ("list", "tuple-in-list"):
            node = holder = []
        elif kind == "dict":
            node = holder = {}
        elif kind == "dataclass":
            node = holder = Cell(f"c{index}")
        elif kind == "jsonable":
            node = holder = Echo()
        elif kind == "proxy":
            holder = {}
            node = MappingProxyType(holder)
        else:
            holder = []
            node = {Stamp(f"s{index}", extras=holder)}
        nodes.append(node)
        holders.append((kind, holder))

    def child(index: int) -> Any:
        if rng.random() < back_edges:
            return nodes[rng.randrange(0, index + 1)]
        if index + 1 < size:
            return nodes[rng.randrange(index + 1, size)]
        return rng.randint(0, 9)

    for index, (kind, holder) in enumerate(holders):
        children = [child(index) for _ in range(rng.randint(0, 2))]
        if kind == "list" or kind == "set":
            holder.extend(children)
        elif kind == "tuple-in-list":
            holder.append(tuple(children))
        elif kind in ("dict", "proxy"):
            holder.update({f"k{position}": item for position, item in enumerate(children)})
        else:  # dataclass, jsonable
            attribute = "link" if kind == "dataclass" else "result"
            setattr(holder, attribute, children[0] if children else None)
    return nodes[0]


@pytest.mark.parametrize("seed", range(30))
def test_random_object_graphs_serialize_exactly_when_they_are_acyclic(seed):
    rng = random.Random(seed)
    cyclic = acyclic = 0

    for _ in range(40):
        root = random_graph(rng, rng.randint(1, 8), rng.choice([0.0, 0.05, 0.15, 0.4]))
        before = fingerprint(root)
        try:
            expected = reference_json(root, frozenset())
        except GraphCycle:
            cyclic += 1
            error = failure_of(partial(serialize_json, root), ValueError)
            assert "Cyclic reference" in str(error)
        else:
            acyclic += 1
            assert to_json_value(root) == expected
        assert fingerprint(root) == before

    assert cyclic > 0 and acyclic > 0, "the seed should produce both kinds of graph"


# --- serialization: errors raised by user code ----------------------------------------------------------------------


class Raiser(Jsonable):
    def __init__(self, error):
        self.error = error

    def to_json(self):
        raise self.error

    @classmethod
    def from_json(cls, data):
        return cls(None)


@pytest.mark.parametrize("error_type", [KeyError, RuntimeError, Abort])
def test_error_of_to_json_deep_inside_propagates_unchanged(error_type):
    error = error_type("to_json failed")  # a new instance per test: raising one object twice piles up its traceback
    value = build_chain([{"k": (Raiser(error),)}], 4)[0]
    before = fingerprint(value)

    with pytest.raises(error_type) as caught:
        serialize_json(value)

    assert caught.value is error
    assert fingerprint(value) == before
    assert serialize_json(Pin("p")) == '{"label": "p"}'


def test_error_of_the_fallback_propagates_unchanged():
    error = RuntimeError("fallback failed")

    def fallback(value):
        raise error

    with pytest.raises(RuntimeError) as caught:
        serialize_json([{"k": Opaque()}], fallback=fallback)

    assert caught.value is error


class Nest(Jsonable):
    """Serializes its child with a nested call, like a Jsonable that renders part of itself separately."""

    def __init__(self, child):
        self.child = child

    def to_json(self):
        return serialize_json_as_obj(self.child)

    @classmethod
    def from_json(cls, data):
        return cls(data)


def test_failure_of_a_nested_serialize_call_surfaces_unchanged():
    ring = Cell(SIBLING)
    ring.link = ring
    direct = failure_of(partial(serialize_json, ring), ValueError)

    nested = failure_of(partial(serialize_json, [{"k": Nest(ring)}]), ValueError)

    assert str(nested) == str(direct)
    assert_no_secret(nested)


def test_mutually_recursive_to_json_through_nested_calls_raises_value_error():
    # Each nested call has a context of its own, so no cycle detector sees the loop: the stack depth ends it
    first, second = Nest(None), Nest(None)
    first.child, second.child = second, first

    with pytest.raises(ValueError, match="nested too deeply") as caught:
        serialize_json(first)

    assert isinstance(caught.value.__cause__, RecursionError)
    assert_no_secret(caught.value)


# --- deserialization: the model, written out twice ------------------------------------------------------------------


def record_json(note: Optional[str]) -> dict:
    return {
        "color": "red",
        "day": "2026-10-27",
        "when": "2026-10-27T12:30:00+00:00",
        "at": "12:30:15",
        "blob": "AQID",  # 01 02 03 -> 000000 010000 001000 000011 -> A Q I D
        "amount": "12.50",
        "kind": "final",
        "pair": [3, 4],
        "tags": [{"label": SIBLING, "color": "blue"}],
        "scores": [0.5, 2.0],
        "index": {"a": 1},
        "note": note,
    }


def library_json() -> dict:
    return {
        "name": SIBLING,
        "books": {
            KEY_SECRET: {
                "title": SIBLING,
                "pages": [{"records": [record_json(SIBLING)], "slots": {"s1": record_json(SIBLING), "s2": None}}],
            }
        },
        "featured": {"title": SIBLING, "pages": [{"records": [record_json(None)], "slots": {}}]},
        "pinned": record_json(SIBLING),
        "by_color": {"blue": [record_json(SIBLING)]},
    }


def make_record(note: Optional[str]) -> Record:
    return Record(
        color=Color.RED,
        day=datetime.date(2026, 10, 27),
        when=datetime.datetime(2026, 10, 27, 12, 30, tzinfo=datetime.timezone.utc),
        at=datetime.time(12, 30, 15),
        blob=b"\x01\x02\x03",
        amount=Decimal("12.50"),
        kind="final",
        pair=(3, 4),
        tags=frozenset({Tag(SIBLING, Color.BLUE)}),
        scores=[0.5, 2.0],
        index={"a": 1},
        note=note,
    )


def make_library() -> Library:
    return Library(
        name=SIBLING,
        books={KEY_SECRET: Book(SIBLING, [Page([make_record(SIBLING)], {"s1": make_record(SIBLING), "s2": None})])},
        featured=Book(SIBLING, [Page([make_record(None)], {})]),
        pinned=make_record(SIBLING),
        by_color={Color.BLUE: [make_record(SIBLING)]},
    )


def with_value(document: Any, path: tuple, bad: Any) -> Any:
    """``document`` with the value at ``path`` replaced by ``bad`` (the path must lead to an existing value)."""
    node = document
    for step in path[:-1]:
        node = node[step]
    if isinstance(node, dict):
        assert path[-1] in node, f"{path} does not lead to an existing value"
    node[path[-1]] = bad
    return document


def corrupted(path: tuple, bad: Any) -> dict:
    """A valid library document with the value at ``path`` replaced by ``bad``."""
    return with_value(library_json(), path, bad)


def test_the_valid_library_document_deserializes_to_the_model_written_out():
    assert to_json_value(make_library()) == library_json()
    assert_strictly_equal(deserialize_json(library_json(), Library), make_library())
    assert_round_trip(make_library(), Library)


# --- deserialization: failures deep inside the model ----------------------------------------------------------------

LOCATIONS = {
    "pinned": ("pinned",),
    "featured": ("featured", "pages", 0, "records", 0),
    "deep": ("books", KEY_SECRET, "pages", 0, "slots", "s1"),
}

DEEP_RECORD = LOCATIONS["deep"]

RECORD_CORRUPTIONS = [
    # (field, label, bad value, fragment the message must contain)
    ("color", "str", SECRET, "enum 'Color'"),
    ("color", "list", [SECRET], "enum 'Color'"),
    ("color", "dict", {KEY_SECRET: SECRET}, "enum 'Color'"),
    ("color", "int", 7, "enum 'Color'"),
    ("color", "null", None, "enum 'Color'"),
    ("day", "str", SECRET, "datetime.date'"),
    ("day", "impossible", "2026-13-45", "datetime.date'"),
    ("day", "int", 20261027, "datetime.date'"),
    ("day", "list", [SECRET], "datetime.date'"),
    ("day", "null", None, "datetime.date'"),
    ("when", "str", SECRET, "datetime.datetime'"),
    ("when", "impossible", "2026-10-27T25:00:00", "datetime.datetime'"),
    ("when", "int", 7, "datetime.datetime'"),
    ("when", "null", None, "datetime.datetime'"),
    ("at", "str", SECRET, "datetime.time'"),
    ("at", "impossible", "25:00:00", "datetime.time'"),
    ("at", "list", [SECRET], "datetime.time'"),
    ("blob", "str", SECRET, "class 'bytes'"),
    ("blob", "bad-padding", "abc", "class 'bytes'"),
    ("blob", "list", [SECRET], "class 'bytes'"),
    ("blob", "int", 5, "class 'bytes'"),
    ("blob", "null", None, "class 'bytes'"),
    ("amount", "str", SECRET, "decimal.Decimal"),
    ("amount", "list", [SECRET], "decimal.Decimal"),
    ("amount", "dict", {KEY_SECRET: SECRET}, "decimal.Decimal"),
    ("amount", "null", None, "decimal.Decimal"),
    ("kind", "str", SECRET, "allowed literals"),
    ("kind", "int", 1, "allowed literals"),
    ("kind", "null", None, "allowed literals"),
    ("kind", "list", ["final"], "allowed literals"),
    ("pair", "too-short", [1], "length 2"),
    ("pair", "too-long", [SECRET, SECRET, SECRET], "length 2"),
    ("pair", "empty", [], "length 2"),
    ("pair", "str", SECRET, "JSON array"),
    ("pair", "dict", {"a": SECRET}, "JSON array"),
    ("pair", "null", None, "JSON array"),
    ("pair", "int", 5, "JSON array"),
    ("tags", "str", SECRET, "JSON array"),
    ("tags", "dict", {"label": SECRET}, "JSON array"),
    ("tags", "null", None, "JSON array"),
    ("tags", "member-str", [SECRET], None),
    ("tags", "member-list", [[SECRET]], None),
    ("tags", "member-null", [None], None),
    ("tags", "member-int", [7], None),
    ("tags", "member-with-bad-color", [{"label": SIBLING, "color": SECRET}], "enum 'Color'"),
    ("scores", "str", SECRET, "JSON array"),
    ("scores", "dict", {"a": 1.5}, "JSON array"),
    ("scores", "null", None, "JSON array"),
    ("scores", "int", 5, "JSON array"),
    ("index", "list", [SECRET], "dict[str, int]"),
    ("index", "list-of-dicts", [{"a": 1}], "dict[str, int]"),
    ("note", "list", [SECRET], "got list"),
]


# An Optional[...] above a record hides why it failed: the union reports its members and the type of the data it got
UNION_ABOVE = {
    "featured": ("Unsupported types for deserialization", "Book", "got dict"),
    "deep": ("Unsupported types for deserialization", "Record", "got dict"),
}


def _record_cases():
    for location_id, location in LOCATIONS.items():
        for field_name, label, bad, fragment in RECORD_CORRUPTIONS:
            yield pytest.param(
                location + (field_name,),
                bad,
                UNION_ABOVE.get(location_id, fragment),
                id=f"{location_id}-{field_name}-{label}",
            )


def _shape_cases(path: tuple, name: str, wrong_values: list, fragment: Optional[str]):
    for index, bad in enumerate(wrong_values):
        yield pytest.param(path, bad, fragment, id=f"{name}-{type(bad).__name__}-{index}")


WRONG_FOR_ARRAY = [{KEY_SECRET: SECRET}, SECRET, 7, None]
WRONG_FOR_DATACLASS = [SECRET, [SECRET], 7, None]
WRONG_FOR_OBJECT = [[SECRET], [{KEY_SECRET: SECRET}]]
WRONG_FOR_OPTIONAL_DATACLASS = [SECRET, [SECRET], 7, {"x": SECRET}]

CONTAINER_CASES = [
    *_shape_cases(("books",), "books", WRONG_FOR_OBJECT, "dict[str,"),
    *_shape_cases(("books", KEY_SECRET), "book", WRONG_FOR_DATACLASS, None),
    *_shape_cases(("books", KEY_SECRET, "pages"), "pages", WRONG_FOR_ARRAY, "JSON array"),
    *_shape_cases(("books", KEY_SECRET, "pages", 0), "page", WRONG_FOR_DATACLASS, None),
    *_shape_cases(("books", KEY_SECRET, "pages", 0, "records"), "records", WRONG_FOR_ARRAY, "JSON array"),
    *_shape_cases(("books", KEY_SECRET, "pages", 0, "records", 0), "record", WRONG_FOR_DATACLASS, None),
    *_shape_cases(("books", KEY_SECRET, "pages", 0, "slots"), "slots", WRONG_FOR_OBJECT, "dict[str,"),
    *_shape_cases(DEEP_RECORD, "slot", WRONG_FOR_OPTIONAL_DATACLASS, "Unsupported types"),
    *_shape_cases(("featured",), "featured", WRONG_FOR_OPTIONAL_DATACLASS, "Unsupported types"),
    *_shape_cases(("featured", "pages"), "featured-pages", WRONG_FOR_ARRAY, "Unsupported types"),
    *_shape_cases(("pinned",), "pinned", WRONG_FOR_DATACLASS, None),
    *_shape_cases(("by_color",), "by-color", WRONG_FOR_OBJECT, "Unsupported type for deserialization"),
    *_shape_cases(("by_color", "blue"), "by-color-records", WRONG_FOR_ARRAY, "JSON array"),
    pytest.param(("by_color",), {SECRET: [record_json(SIBLING)]}, "enum 'Color'", id="by-color-key-not-a-member"),
    pytest.param(
        ("by_color",), {"blue": [record_json(SIBLING)], KEY_SECRET: []}, "enum 'Color'", id="by-color-second-key"
    ),
]


def _check_deep_failure(path, bad, fragment):
    data = corrupted(path, bad)
    snapshot = copy.deepcopy(data)

    with pytest.raises(TypeError) as caught:
        deserialize_json(data, Library)

    assert type(caught.value) is TypeError
    for each in (fragment,) if isinstance(fragment, str) else (fragment or ()):
        assert each in str(caught.value)
    assert_no_secret(caught.value)
    assert data == snapshot, "the parsed input was modified by a failed call"
    assert_guard_clear()
    assert_strictly_equal(deserialize_json(library_json(), Library), make_library())


@pytest.mark.parametrize(("path", "bad", "fragment"), list(_record_cases()))
def test_invalid_record_field_at_any_depth_raises_type_error_without_payload(path, bad, fragment):
    _check_deep_failure(path, bad, fragment)


@pytest.mark.parametrize(("path", "bad", "fragment"), CONTAINER_CASES)
def test_wrong_container_shape_at_any_depth_raises_type_error_without_payload(path, bad, fragment):
    _check_deep_failure(path, bad, fragment)


def test_the_error_names_the_expected_types_not_the_data():
    data = corrupted(("pinned", "pair"), [SECRET, SECRET, SECRET])

    error = failure_of(partial(deserialize_json, data, Library), TypeError)

    assert "length 2" in str(error) and "got length 3" in str(error)
    assert_no_secret(error)


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(dict[str, list[Node]], id="plain"),
        pytest.param(dict[str, list[Union[Node, list[int]]]], id="union-above-the-cycle"),
    ],
)
def test_cyclic_input_deep_inside_a_document_is_a_value_error_not_a_type_error(monkeypatch, annotation):
    # The walk is bounded instead of looping on input that never stops nesting (the bound is lowered here, to stay
    # quick). A ValueError is not a mismatch to a union, and it names no data.
    monkeypatch.setattr(json_handler, "_MAX_DESERIALIZE_DEPTH", 2_000)
    cycle: dict = {"label": SIBLING}
    cycle["child"] = cycle
    data = {KEY_SECRET: [cycle]}

    with pytest.raises(ValueError, match="nested too deeply to deserialize") as caught:
        deserialize_json(data, annotation)

    assert type(caught.value) is ValueError
    assert_no_secret(caught.value)
    assert list(cycle) == ["label", "child"] and cycle["child"] is cycle
    assert_guard_clear()
    restored = deserialize_json({"k": [{"label": "a", "child": {"label": "b"}}]}, annotation)
    assert restored == {"k": [Node("a", Node("b"))]}


# --- deserialization: Optional fields with a bad non-null value -----------------------------------------------------

VALID_OPTIONS_JSON = {
    "color": "blue",
    "day": "2026-10-27",
    "blob": "AQID",
    "amount": "1.50",
    "kind": "draft",
    "pair": [1, 2],
    "tags": [{"label": "t", "color": "red"}],
    "scores": [1.5],
    "record": record_json("n"),
    "book": {"title": "T", "pages": [{"records": [], "slots": {}}]},
    "index": {"a": 1},
}


def make_options() -> Options:
    return Options(
        color=Color.BLUE,
        day=datetime.date(2026, 10, 27),
        blob=b"\x01\x02\x03",
        amount=Decimal("1.50"),
        kind="draft",
        pair=(1, 2),
        tags=frozenset({Tag("t", Color.RED)}),
        scores=[1.5],
        record=make_record("n"),
        book=Book("T", [Page([], {})]),
        index={"a": 1},
    )


OPTION_CORRUPTIONS = [
    ("color", "str", SECRET),
    ("color", "list", [SECRET]),
    ("color", "int", 7),
    ("day", "str", SECRET),
    ("day", "impossible", "2026-13-45"),
    ("day", "int", 20261027),
    ("day", "list", [SECRET]),
    ("blob", "str", SECRET),
    ("blob", "bad-padding", "abc"),
    ("blob", "list", [SECRET]),
    ("blob", "int", 5),
    ("amount", "str", SECRET),
    ("amount", "list", [SECRET]),
    ("amount", "dict", {KEY_SECRET: SECRET}),
    ("kind", "str", SECRET),
    ("kind", "int", 1),
    ("kind", "list", ["draft"]),
    ("pair", "too-short", [1]),
    ("pair", "too-long", [SECRET, SECRET, SECRET]),
    ("pair", "str", SECRET),
    ("pair", "dict", {"a": 1}),
    ("pair", "int", 5),
    ("tags", "str", SECRET),
    ("tags", "dict", {"label": SECRET}),
    ("tags", "member-str", [SECRET]),
    ("tags", "member-with-bad-color", [{"label": SIBLING, "color": SECRET}]),
    ("scores", "str", SECRET),
    ("scores", "dict", {"a": 1.5}),
    ("scores", "int", 5),
    ("record", "list", [SECRET]),
    ("record", "str", SECRET),
    ("record", "int", 7),
    ("record", "with-bad-color", {**record_json(SIBLING), "color": SECRET}),
    ("record", "with-bad-pair", {**record_json(SIBLING), "pair": [SECRET]}),
    ("book", "list", [SECRET]),
    ("book", "str", SECRET),
    ("book", "with-bad-page", {"title": SIBLING, "pages": {KEY_SECRET: SECRET}}),
    ("index", "list", [SECRET]),
]


def test_options_document_is_the_model_written_out_twice():
    assert to_json_value(make_options()) == VALID_OPTIONS_JSON
    assert_round_trip(make_options(), Options)
    restored = deserialize_json({"items": [{"k": VALID_OPTIONS_JSON}]}, Wrapper)
    assert_strictly_equal(restored.items[0]["k"], make_options())


@pytest.mark.parametrize("fill", ["missing", "null"])
def test_optional_fields_accept_null_and_absence_at_depth(fill):
    inner = {} if fill == "missing" else dict.fromkeys(VALID_OPTIONS_JSON)

    restored = deserialize_json({"items": [{"k": inner}]}, Wrapper)

    assert_strictly_equal(restored, Wrapper([{"k": Options()}]))


@pytest.mark.parametrize(
    ("field_name", "bad"),
    [pytest.param(name, bad, id=f"{name}-{label}") for name, label, bad in OPTION_CORRUPTIONS],
)
def test_optional_field_with_a_bad_non_null_value_still_raises(field_name, bad):
    inner = {**VALID_OPTIONS_JSON, field_name: bad}
    data = {"items": [{"first": {}, "k": inner}]}
    snapshot = copy.deepcopy(data)

    with pytest.raises(TypeError) as caught:
        deserialize_json(data, Wrapper)

    assert type(caught.value) is TypeError
    assert re.search(rf"^Unsupported types for deserialization: .*, got {type(bad).__name__}$", str(caught.value))
    assert_no_secret(caught.value)
    assert data == snapshot
    assert_guard_clear()


# --- deserialization: unions whose every branch fails ---------------------------------------------------------------

CHOICE_FAILURES = [
    pytest.param([SECRET], "list", id="list-with-bad-member"),
    pytest.param(["red", SECRET], "list", id="list-with-valid-then-bad-member"),
    pytest.param([["red"]], "list", id="list-in-list"),
    pytest.param({KEY_SECRET: SECRET}, "dict", id="dict-with-bad-value"),
    pytest.param({"a": "red", "b": [SECRET]}, "dict", id="dict-with-valid-then-bad-value"),
]


@pytest.mark.parametrize(("data", "type_name"), CHOICE_FAILURES)
def test_union_of_unions_where_every_branch_fails_names_types_only(data, type_name):
    snapshot = copy.deepcopy(data)

    with pytest.raises(TypeError) as caught:
        deserialize_json(data, Choice)

    message = str(caught.value)
    assert type(caught.value) is TypeError
    assert message.startswith("Unsupported types for deserialization: ")
    assert message.endswith(f", got {type_name}")
    assert "Color" in message and "datetime.date" in message
    assert_no_secret(caught.value)
    assert data == snapshot
    assert_guard_clear()


def test_union_with_a_dict_member_hands_any_primitive_through_limitation():
    # Documents current behavior, not a promise: dict[...] has no handler, and a primitive is handed through as is
    for data in (SECRET, 7, None, True):
        assert deserialize_json(data, Choice) == data


@pytest.mark.parametrize(
    "data",
    [
        pytest.param([SECRET], id="list-with-bad-member"),
        pytest.param(["red", None, "1.5", SECRET], id="list-with-valid-members-then-bad"),
        pytest.param([SECRET, SECRET], id="tuple-of-bad-members"),
        pytest.param(["2026-10-27", "abc"], id="date-then-bad-padding"),
        pytest.param({"a": 1}, id="dict"),
    ],
)
def test_three_levels_of_unions_where_every_branch_fails(data):
    with pytest.raises(TypeError) as caught:
        deserialize_json(data, Deeper)

    assert str(caught.value).startswith("Unsupported types for deserialization: ")
    assert_no_secret(caught.value)


def test_three_levels_of_unions_pick_the_first_branch_that_fits():
    first = deserialize_json(["red", None, "1.5", "blue"], Deeper)
    second = deserialize_json(["2026-10-27", "AQID"], Deeper)

    assert_strictly_equal(first, [Color.RED, None, Decimal("1.5"), Color.BLUE])
    assert_strictly_equal(second, (datetime.date(2026, 10, 27), b"\x01\x02\x03"))
    assert to_json_value(first) == ["red", None, "1.5", "blue"]
    assert to_json_value(second) == ["2026-10-27", "AQID"]


def test_union_of_unions_deep_in_a_document_fails_with_types_only():
    data = {"baskets": {KEY_SECRET: [{"choice": ["red"]}, {"choice": ["blue", SECRET]}]}}

    with pytest.raises(TypeError) as caught:
        deserialize_json(data, Crate)

    assert str(caught.value).endswith(", got list")
    assert_no_secret(caught.value)


def test_union_of_unions_deep_in_a_document_still_accepts_both_branches():
    data = {"baskets": {"a": [{"choice": ["red", "2026-10-27"]}, {"choice": {"x": "blue", "y": "2026-10-28"}}]}}

    restored = deserialize_json(data, Crate)

    assert_strictly_equal(
        restored,
        Crate({"a": [
            Basket([Color.RED, datetime.date(2026, 10, 27)]),
            Basket({"x": Color.BLUE, "y": datetime.date(2026, 10, 28)}),
        ]}),
    )
    assert to_json_value(restored) == data


def test_union_member_failing_deep_inside_hands_over_to_the_next_member_and_keeps_the_order():
    good = library_json()["books"][KEY_SECRET]
    bad = corrupted(DEEP_RECORD + ("color",), SECRET)["books"][KEY_SECRET]
    data = {"entries": {"k": [good, bad, good]}}
    snapshot = copy.deepcopy(data)

    restored = deserialize_json(data, Archive)

    first, second, third = restored.entries["k"]
    expected_book = make_library().books[KEY_SECRET]
    assert_strictly_equal(first, expected_book)
    assert_strictly_equal(third, expected_book)
    assert type(second) is dict and second == bad and second is not bad
    assert data == snapshot
    assert_guard_clear()


def test_union_without_a_fitting_member_deep_inside_names_all_members():
    bad = corrupted(DEEP_RECORD + ("color",), SECRET)["books"][KEY_SECRET]

    with pytest.raises(TypeError, match=r"Unsupported types for deserialization: .*Book.*Record.*, got dict") as caught:
        deserialize_json([bad], list[Union[Book, Record]])

    assert_no_secret(caught.value)


# --- deserialization: mapping annotations, tuples of mixed types, paths and sets that cannot be built --------------


def views_json() -> dict:
    return {
        "by_name": {"a": [{"label": SIBLING, "color": "red"}]},
        "frozen": {KEY_SECRET: {"label": SIBLING, "color": "blue"}},
    }


def make_views() -> Views:
    return Views(
        by_name={"a": [Tag(SIBLING, Color.RED)]},
        frozen=MappingProxyType({KEY_SECRET: Tag(SIBLING, Color.BLUE)}),
    )


def test_mapping_annotations_deep_in_a_document_round_trip():
    assert to_json_value(make_views()) == views_json()

    restored = assert_round_trip(make_views(), Views)
    assert type(restored.frozen) is MappingProxyType

    nested = assert_round_trip({"k": [make_views()]}, dict[str, list[Views]])
    assert type(nested["k"][0].frozen) is MappingProxyType


def _wrong_shapes(field_name: str, kind: str):
    for label, bad in (("list", [SECRET]), ("str", SECRET), ("int", 7), ("bool", True), ("null", None)):
        pattern = rf"^Expected a JSON {kind} for .*, got {type(bad).__name__}$"
        yield pytest.param((field_name,), bad, pattern, id=f"{field_name}-{label}")


VIEW_FAILURES = [
    *_wrong_shapes("by_name", "object"),
    *_wrong_shapes("frozen", "object"),
    pytest.param(
        ("by_name", "a"), {"label": SECRET}, r"^Expected a JSON array for .*, got dict$", id="by-name-value-dict"
    ),
    pytest.param(("by_name", "a"), SECRET, r"^Expected a JSON array for .*, got str$", id="by-name-value-str"),
    pytest.param(("by_name", "a", 0, "color"), SECRET, "enum 'Color'", id="by-name-bad-enum"),
    pytest.param(("by_name", "a", 0), [SECRET], None, id="by-name-member-list"),
    pytest.param(("frozen", KEY_SECRET, "color"), SECRET, "enum 'Color'", id="frozen-bad-enum"),
    pytest.param(("frozen", KEY_SECRET), [SECRET], None, id="frozen-member-list"),
    pytest.param(("frozen", KEY_SECRET), SECRET, None, id="frozen-member-str"),
]


@pytest.mark.parametrize(("path", "bad", "pattern"), VIEW_FAILURES)
def test_mapping_annotations_reject_wrong_shapes_and_bad_values_at_depth(path, bad, pattern):
    data = with_value({"k": [views_json()]}, ("k", 0) + path, bad)
    snapshot = copy.deepcopy(data)

    with pytest.raises(TypeError) as caught:
        deserialize_json(data, dict[str, list[Views]])

    assert type(caught.value) is TypeError
    if pattern is not None:
        assert re.search(pattern, str(caught.value))
    assert_no_secret(caught.value)
    assert data == snapshot
    assert_guard_clear()


Entries = dict[str, list[tuple[str, Color, Optional[datetime.date]]]]


def test_tuples_of_mixed_types_restore_each_position_by_its_own_type():
    data = {"k": [["a", "red", None], ["b", "blue", "2026-10-27"]]}

    restored = deserialize_json(data, Entries)

    assert_strictly_equal(
        restored, {"k": [("a", Color.RED, None), ("b", Color.BLUE, datetime.date(2026, 10, 27))]}
    )


@pytest.mark.parametrize(
    ("entries", "pattern"),
    [
        pytest.param(
            [["a", "red", None], ["b", SECRET, None]], r"^Invalid value for .*Color.*, got str$", id="bad-enum"
        ),
        pytest.param(
            [["a", "red", "2026-13-45"]], r"^Unsupported types for deserialization: .*, got str$", id="bad-date"
        ),
        pytest.param([["a", "red"]], r"length 3 for .*, got length 2$", id="one-too-short"),
        pytest.param([["a", "red", None, SECRET]], r"length 3 for .*, got length 4$", id="one-too-long"),
        pytest.param([[]], r"length 3 for .*, got length 0$", id="empty"),
        pytest.param(["abc"], r"^Expected a JSON array for .*, got str$", id="string-of-three-characters"),
        pytest.param([{"a": SECRET}], r"^Expected a JSON array for .*, got dict$", id="dict"),
        pytest.param([None], r"^Expected a JSON array for .*, got NoneType$", id="null"),
    ],
)
def test_tuples_of_mixed_types_reject_bad_members_and_shapes(entries, pattern):
    data = {"k": [["ok", "blue", None]] + entries}
    snapshot = copy.deepcopy(data)

    with pytest.raises(TypeError) as caught:
        deserialize_json(data, Entries)

    assert type(caught.value) is TypeError
    assert re.search(pattern, str(caught.value))
    assert_no_secret(caught.value)
    assert data == snapshot


UNHASHABLE_MEMBERS = [
    pytest.param(frozenset[Any], [[SECRET]], id="list-member"),
    pytest.param(set[list[int]], [[1]], id="typed-list-member"),
    pytest.param(frozenset[dict[str, Any]], [{"k": SECRET}], id="dict-member"),
    pytest.param(set[Cell], [{"label": SECRET}], id="mutable-dataclass-member"),
]


@pytest.mark.parametrize(("annotation", "members"), UNHASHABLE_MEMBERS)
def test_set_members_that_cannot_be_hashed_are_a_type_error(annotation, members):
    data = {"k": [members]}
    snapshot = copy.deepcopy(data)

    with pytest.raises(TypeError, match="unhashable type") as caught:
        deserialize_json(data, dict[str, list[annotation]])

    assert type(caught.value) is TypeError
    assert_no_secret(caught.value)
    assert data == snapshot
    assert_guard_clear()


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param([SECRET], id="list"),
        pytest.param({KEY_SECRET: SECRET}, id="dict"),
        pytest.param(7, id="int"),
        pytest.param(True, id="bool"),
        pytest.param(None, id="null"),
    ],
)
def test_path_given_something_other_than_a_string_is_a_type_error_naming_types_only(bad):
    data = {"k": ["a/b", bad]}

    with pytest.raises(TypeError) as caught:
        deserialize_json(data, dict[str, list[Path]])

    assert type(caught.value) is TypeError
    assert_no_secret(caught.value)
    assert deserialize_json({"k": ["a/b"]}, dict[str, list[Path]]) == {"k": [Path("a/b")]}


def test_a_union_moves_on_from_a_path_that_cannot_be_built():
    restored = deserialize_json({"k": ["a/b", 7]}, dict[str, list[Union[Path, int]]])

    assert_strictly_equal(restored, {"k": [Path("a/b"), 7]})


def test_a_union_moves_on_from_a_set_that_cannot_be_built():
    restored = deserialize_json({"k": [[[SECRET]]]}, dict[str, list[Union[frozenset[Any], list[Any]]]])

    assert_strictly_equal(restored, {"k": [[[SECRET]]]})


# --- deserialization: validation in __post_init__ and the order of user code ----------------------------------------


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(dict[str, list[Checked]], id="plain"),
        pytest.param(dict[str, list[Optional[Checked]]], id="optional"),
        pytest.param(dict[str, list[Union[Checked, dict[str, int]]]], id="union-with-a-fallback-member"),
    ],
)
def test_value_error_of_post_init_propagates_and_is_not_caught_by_unions(annotation):
    # Unions try the next member on TypeError only: a dataclass that wants fall-through must raise TypeError itself
    data = {"k": [{"amount": 1}, {"amount": -1}]}

    with pytest.raises(ValueError, match="negative amount") as caught:
        deserialize_json(data, annotation)

    assert type(caught.value) is ValueError
    assert_guard_clear()
    assert_strictly_equal(deserialize_json({"k": [{"amount": 1}]}, annotation), {"k": [Checked(1)]})


def test_type_error_of_post_init_makes_a_union_try_the_next_member():
    data = {"k": [{"amount": 1}, {"amount": SECRET}]}

    restored = deserialize_json(data, dict[str, list[Union[TypeChecked, dict[str, Any]]]])

    assert_strictly_equal(restored, {"k": [TypeChecked(1), {"amount": SECRET}]})
    with pytest.raises(TypeError, match="amount must be an int") as caught:
        deserialize_json(data, dict[str, list[TypeChecked]])
    assert_guard_clear()


def test_post_init_rejects_a_missing_field_because_it_arrives_as_none():
    with pytest.raises(TypeError, match="amount must be an int"):
        deserialize_json({"k": [{}]}, dict[str, list[TypeChecked]])


PROCESSION_ORDER = ["first", "m0", "m1", "m2", "kx", "ky", "last"]


def procession_json() -> dict:
    return {
        "first": {"name": "first"},
        "many": [{"name": "m0"}, {"name": "m1"}, {"name": "m2"}],
        "by_name": {"x": {"name": "kx"}, "y": {"name": "ky"}},
        "last": {"name": "last"},
    }


def test_from_json_is_called_for_every_value_in_declaration_order():
    restored = deserialize_json(procession_json(), Procession)

    assert CALLS == PROCESSION_ORDER
    assert [probe.name for probe in restored.many] == ["m0", "m1", "m2"]


@pytest.mark.parametrize(
    ("path", "calls_before_the_failure"),
    [
        pytest.param(("first",), [], id="first-field"),
        pytest.param(("many", 1), ["first", "m0"], id="second-list-item"),
        pytest.param(("by_name", "y"), ["first", "m0", "m1", "m2", "kx"], id="second-dict-value"),
        pytest.param(("last",), PROCESSION_ORDER[:-1], id="last-field"),
    ],
)
def test_from_json_calls_stop_at_the_first_failure(path, calls_before_the_failure):
    data = with_value(procession_json(), path, {"name": "bad"})

    with pytest.raises(TypeError, match="rejected"):
        deserialize_json(data, Procession)

    assert CALLS == calls_before_the_failure + ["bad"]
    assert_guard_clear()


def test_a_failed_union_member_keeps_its_side_effects_and_the_next_member_runs_after_them():
    bad = with_value(procession_json(), ("many", 1), {"name": "bad"})

    restored = deserialize_json([bad, procession_json()], list[Union[Procession, Spare]])

    assert [type(item) for item in restored] == [Spare, Procession]
    assert CALLS == ["first", "m0", "bad", "spare"] + PROCESSION_ORDER


def test_the_first_fitting_union_member_ends_the_search():
    restored = deserialize_json(procession_json(), Union[Spare, Procession])

    assert type(restored) is Spare
    assert CALLS == ["spare"]


def test_failure_at_the_last_of_many_items_names_no_item():
    data = ["red", "blue"] * 5000

    assert len(deserialize_json(data, list[Color])) == 10_000
    with pytest.raises(TypeError, match=r"^Invalid value for .*Color.*, got str$") as caught:
        deserialize_json(data + [SECRET], list[Color])

    assert_no_secret(caught.value)
    assert_guard_clear()


# --- deserialization: user code (Jsonable.from_json) in the middle of the walk --------------------------------------


def _bad_payloads():
    return [
        pytest.param({**record_json(SIBLING), "color": SECRET}, id="bad-enum"),
        pytest.param({**record_json(SIBLING), "pair": [SECRET, SECRET, SECRET]}, id="bad-tuple-length"),
        pytest.param([SECRET], id="list-for-dataclass"),
        pytest.param({**record_json(SIBLING), "tags": [{"label": SIBLING, "color": SECRET}]}, id="bad-set-member"),
    ]


@pytest.mark.parametrize("payload", _bad_payloads())
def test_error_of_a_nested_deserialize_call_is_the_error_of_a_direct_call(payload):
    direct = failure_of(partial(deserialize_json, payload, Record), TypeError)
    Envelope.guard_seen.clear()

    nested = failure_of(partial(deserialize_json, {"payload": payload}, Envelope), TypeError)

    assert str(nested) == str(direct)
    assert_no_secret(nested)
    assert Envelope.guard_seen == [True]
    assert_guard_clear()


@pytest.mark.parametrize("payload", _bad_payloads())
def test_error_two_levels_of_nested_calls_down_is_unchanged(payload):
    direct = failure_of(partial(deserialize_json, payload, Record), TypeError)

    nested = failure_of(partial(deserialize_json, {"inner": {"payload": payload}}, Outer), TypeError)

    assert str(nested) == str(direct)
    assert_no_secret(nested)
    assert_guard_clear()


def test_nested_call_error_passes_through_containers():
    payload = {**record_json(SIBLING), "color": SECRET}
    data = [{"ok": {"payload": record_json(SIBLING)}, "bad": {"payload": payload}}]
    direct = failure_of(partial(deserialize_json, payload, Record), TypeError)

    error = failure_of(partial(deserialize_json, data, list[dict[str, Envelope]]), TypeError)

    assert str(error) == str(direct)
    assert_no_secret(error)
    assert_guard_clear()


def test_nested_call_error_inside_an_optional_is_reported_by_the_union():
    payload = {**record_json(SIBLING), "color": SECRET}
    data = [{"bad": {"payload": payload}}]

    error = failure_of(partial(deserialize_json, data, Optional[list[dict[str, Envelope]]]), TypeError)

    assert str(error).startswith("Unsupported types for deserialization: ")
    assert str(error).endswith(", got list")
    assert_no_secret(error)
    assert_guard_clear()


def test_union_falls_through_when_from_json_raises_type_error_and_fails_when_the_next_member_does_too():
    assert deserialize_json("2026-10-27", Union[Envelope, datetime.date]) == datetime.date(2026, 10, 27)

    data = {"payload": {**record_json(SIBLING), "color": SECRET}}
    pattern = r"Unsupported types for deserialization: .*Envelope.*date.*, got dict"
    with pytest.raises(TypeError, match=pattern) as caught:
        deserialize_json(data, Envelope | datetime.date)

    assert_no_secret(caught.value)
    assert_guard_clear()


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(list[dict[str, Union[Exploding, datetime.date]]], id="member-without-steps"),
        pytest.param(Union[list[dict[str, Exploding]], datetime.date], id="container-member"),
        pytest.param(list[Union[dict[str, Exploding], list[int]]], id="container-member-in-a-list"),
        pytest.param(
            Union[tuple[Color, ...], list[dict[str, Exploding]], datetime.date], id="second-container-member"
        ),
    ],
)
@pytest.mark.parametrize("error_type", [KeyError, ValueError, RuntimeError])
def test_other_exception_types_of_from_json_propagate_unchanged_and_are_not_caught_by_unions(annotation, error_type):
    # Unions try the next member on TypeError only: a Jsonable that wants fall-through must raise TypeError itself
    error = error_type("from_json failed")
    data = [{"k": {"raise": error}}]

    with pytest.raises(error_type) as caught:
        deserialize_json(data, annotation)

    assert caught.value is error
    assert_guard_clear()


def test_a_base_exception_of_from_json_propagates_and_resets_the_guard():
    error = Abort("abort")

    with pytest.raises(Abort) as caught:
        deserialize_json([{"k": {"raise": error}}], list[dict[str, Exploding]])

    assert caught.value is error
    assert_guard_clear()
    assert_strictly_equal(deserialize_json(library_json(), Library), make_library())


def test_from_json_that_swallows_the_nested_error_does_not_disturb_later_calls():
    bad = {"payload": {**record_json(SIBLING), "color": SECRET}}

    restored = deserialize_json([bad, {"payload": record_json(SIBLING)}], list[Lenient])

    assert restored[0].payload is None
    assert_strictly_equal(restored[1].payload, make_record(SIBLING))
    assert_guard_clear()
    failure_of(partial(deserialize_json, bad["payload"], Record), TypeError)


# --- deserialization: limitations and defects around keys -----------------------------------------------------------


def test_parameterized_tuple_keys_deep_in_a_document_round_trip():
    value = {"grids": [{"a": Grid({(0, 1): Tag("t", Color.RED), (2, 3): Tag("u", Color.BLUE)})}]}

    restored = assert_round_trip(value, dict[str, list[dict[str, Grid]]])

    assert list(restored["grids"][0]["a"].cells) == [(0, 1), (2, 3)]


def test_parameterized_tuple_key_of_the_wrong_length_is_a_type_error_naming_the_length():
    data = {"cells": {"[1, 2, 3]": {"label": SIBLING, "color": "red"}}}

    with pytest.raises(TypeError, match="length 2") as caught:
        deserialize_json(data, Grid)

    assert_no_secret(caught.value)


def test_bare_tuple_keys_deep_in_a_document_round_trip():
    value = {"g": {(1, "a"): [Pin("p")], (2, "b"): []}}

    restored = assert_round_trip(value, dict[str, dict[tuple, list[Pin]]])

    assert list(restored["g"]) == [(1, "a"), (2, "b")]


@pytest.mark.parametrize("key", [SECRET, "[1, 2", "", "{'a': 1}"], ids=["words", "truncated", "empty", "single-quotes"])
def test_malformed_tuple_key_is_a_type_error(key):
    with pytest.raises(TypeError) as caught:
        deserialize_json({key: 1}, dict[tuple, int])

    assert_no_secret(caught.value)


def test_union_falls_through_to_plain_string_keys_when_a_key_is_not_a_tuple():
    data = {"not a tuple": 1}

    assert deserialize_json(data, Union[dict[tuple, int], dict[str, int]]) == {"not a tuple": 1}


def test_json_keys_that_restore_to_the_same_tuple_are_merged_limitation():
    # Documents current behavior, not a promise: two spellings of one tuple key end up as a single entry, the later
    # value wins, with no error (the serializer refuses colliding keys, the deserializer does not look). Only
    # hand-written or foreign JSON can hold both spellings: the library writes one per tuple.
    data = {"g": {"[1, 2]": "left", "[1,2]": "right"}}

    assert deserialize_json(data, dict[str, dict[tuple, str]]) == {"g": {(1, 2): "right"}}


def test_tuple_key_that_is_valid_json_of_the_wrong_shape_is_a_type_error():
    for key in ('"text"', "7", "null", '{"a": 1}'):
        with pytest.raises(TypeError) as caught:
            deserialize_json({key: 1}, dict[tuple, int])

        assert "JSON array" in str(caught.value)
        assert_no_secret(caught.value)


def test_dataclass_with_a_computed_field_round_trips():
    assert to_json_value(Derived(3)) == {"base": 3, "doubled": 6}

    assert_round_trip(Derived(3), Derived)


def test_integer_keys_cannot_be_restored_limitation():
    # Documents current behavior, not a promise: only str, Enum and tuple keys are restored. Others raise ValueError
    # (not TypeError, so a union does not try its next member) with a message that names the key type, not the key.
    data = {"items": [{"7": 1}]}

    with pytest.raises(ValueError, match="de-convert JSON mapping key") as caught:
        deserialize_json(data, dict[str, list[dict[int, int]]])
    assert "7" not in str(caught.value)
    with pytest.raises(ValueError):
        deserialize_json({"7": 1}, Union[dict[int, int], dict[str, int]])
    assert_guard_clear()


@pytest.mark.parametrize("key_annotation", [Literal["a", "b"], Optional[str]], ids=["literal", "optional"])
def test_key_annotations_that_are_not_classes_cannot_be_restored_limitation(key_annotation):
    # Documents current behavior, not a promise: the key type goes through issubclass(), so these raise a TypeError
    # about issubclass() for any data, valid or not (which makes a union move on to its next member)
    with pytest.raises(TypeError, match="issubclass"):
        deserialize_json({"a": 1}, dict[key_annotation, int])


def test_primitive_targets_accept_any_json_value_limitation():
    # Documents current behavior, not a promise: a JSON primitive is handed through for annotations without a handler
    # (dict[...], str, int, float, ...), so a wrong primitive is not a failure; a dict is handed through for str
    assert deserialize_json({"index": SECRET}, IndexHolder) == IndexHolder(SECRET)
    assert deserialize_json([SECRET], list[float]) == [SECRET]
    assert deserialize_json({"a": 1}, Optional[str]) == {"a": 1}
    assert deserialize_json(None, dict[str, int]) is None


def test_dataclass_without_fields_takes_an_object_and_ignores_its_keys():
    for data in ({}, {"x": 1}):
        assert deserialize_json(data, Empty) == Empty()

    for data in (SECRET, [SECRET], 7, None):
        with pytest.raises(TypeError) as caught:
            deserialize_json(data, Empty)

        assert_no_secret(caught.value)


def test_missing_required_nested_field_becomes_none_limitation():
    # Documents current behavior, not a promise: an absent field is passed as None to its annotation
    data = corrupted(DEEP_RECORD, {k: v for k, v in record_json(SIBLING).items() if k != "index"})

    restored = deserialize_json(data, Library)

    assert restored.books[KEY_SECRET].pages[0].slots["s1"].index is None


def test_bytes_ignore_characters_outside_the_alphabet_limitation():
    # Documents current behavior, not a promise: base64.b64decode drops them instead of failing
    assert deserialize_json("QUJD-REVG", bytes) == b"ABCDEF"
    assert deserialize_json(["QUJD REVG"], list[bytes]) == [b"ABCDEF"]


# --- state after failures -------------------------------------------------------------------------------------------


def _cache_sizes() -> tuple:
    return json_handler._resolved_field_types.cache_info().currsize, _numpy.get_numpy_kind.cache_info().currsize


def _serialization_failures():
    ring = Cell(SIBLING)
    ring.link = [ring]
    return [
        ("cycle", partial(serialize_json, ring), ValueError),
        ("unsupported", partial(serialize_json, build_chain(Opaque(), 6)[0]), TypeError),
        ("unsupported-key", partial(serialize_json, [{Stamp("a", extras={Opaque(): 1})}]), TypeError),
        ("collision", partial(serialize_json, MappingProxyType({LeakyKey.KEY: 1, KEY_SECRET: 2})), ValueError),
        ("nan", partial(serialize_json, Shelf(SIBLING, {float("nan")}), allow_nan=False), ValueError),
    ]


def test_repeated_serialization_failures_leave_no_state_behind():
    good = Shelf(SIBLING, [Pin("p"), {"k": (1, Color.RED, Decimal("1.50"))}, {Stamp("s", extras=[1])}])
    expected = serialize_json(good)
    assert json.loads(expected) == {
        "title": SIBLING,
        "items": [{"label": "p"}, {"k": [1, "red", "1.50"]}, [{"extras": [1], "name": "s"}]],
    }
    for _, call, exception_type in _serialization_failures():  # warm up the per-class caches
        failure_of(call, exception_type)
    sizes = _cache_sizes()

    for _ in range(100):
        for name, call, exception_type in _serialization_failures():
            failure_of(call, exception_type)
            assert serialize_json(good) == expected, f"a call after the {name} failure"

    assert _cache_sizes() == sizes


def test_serialization_failure_keeps_no_reference_to_its_input():
    leaf, ring = Opaque(), Cell(SIBLING)
    ring.link = [ring]
    probes = [weakref.ref(leaf), weakref.ref(ring)]
    chain = build_chain(leaf, 6)[0]

    for call in (partial(serialize_json, chain), partial(serialize_json, ring)):
        try:
            call()
        except (TypeError, ValueError):
            pass
        else:
            pytest.fail("the serialization should have failed")
    del leaf, ring, chain, call
    gc.collect()

    assert [probe() for probe in probes] == [None, None]


def test_repeated_deserialization_failures_leave_no_state_behind():
    expected = make_library()
    bad = [
        corrupted(DEEP_RECORD + ("color",), SECRET),
        corrupted(("featured", "pages"), SECRET),
        corrupted(("by_color",), {SECRET: []}),
        corrupted(("pinned", "pair"), [SECRET] * 3),
    ]
    snapshots = [copy.deepcopy(data) for data in bad]
    for data in bad:  # warm up the caches of resolved annotations
        failure_of(partial(deserialize_json, data, Library), TypeError)
    sizes = _cache_sizes()

    for _ in range(100):
        for data in bad:
            failure_of(partial(deserialize_json, data, Library), TypeError)
            assert_guard_clear()
            assert_strictly_equal(deserialize_json(library_json(), Library), expected)
        failure_of(partial(deserialize_json, [SECRET], Choice), TypeError)
        assert_guard_clear()

    assert bad == snapshots
    assert _cache_sizes() == sizes


def test_deserialization_failure_keeps_no_reference_to_what_it_built():
    data = {"first": {"tag": "t"}, "second": {**record_json(SIBLING), "color": SECRET}}

    with pytest.raises(TypeError):
        deserialize_json(data, Trays)
    gc.collect()

    assert len(Tracked.instances) == 0
    restored = deserialize_json({"first": {"tag": "t"}, "second": record_json(SIBLING)}, Trays)
    assert restored.first.tag == "t" and len(Tracked.instances) == 1


def test_failures_inside_and_outside_unions_alternate_with_successes():
    good_choice = ["red", "2026-10-27"]
    for _ in range(20):
        failure_of(partial(deserialize_json, [SECRET], Choice), TypeError)
        assert deserialize_json(good_choice, Choice) == [Color.RED, datetime.date(2026, 10, 27)]
        failure_of(partial(deserialize_json, corrupted(("pinned", "kind"), SECRET), Library), TypeError)
        assert_strictly_equal(deserialize_json(library_json(), Library), make_library())
        assert_guard_clear()


# --- threads --------------------------------------------------------------------------------------------------------

SHARED_RING = Cell(SIBLING)
SHARED_RING.link = [SHARED_RING]
SHARED_LIBRARY_BAD = corrupted(DEEP_RECORD + ("color",), SECRET)
SHARED_PAYLOAD_BAD = {**record_json(SIBLING), "pair": [SECRET, SECRET, SECRET]}

# Jobs read objects shared by all threads (never mutating them) or build their own
JOBS = {
    "serialize-shared-cycle": lambda: serialize_json(SHARED_RING),
    "serialize-ok": lambda: serialize_json(build_chain([Pin("p"), {Stamp("s")}], 6)[0]),
    "serialize-unsupported": lambda: serialize_json(build_chain(Opaque(), 6)[0]),
    "serialize-collision": lambda: serialize_json([{LeakyKey.KEY: 1, KEY_SECRET: 2}]),
    "serialize-nan": lambda: serialize_json(Shelf(SIBLING, [float("nan")]), allow_nan=False),
    "deserialize-ok": lambda: serialize_json(deserialize_json(library_json(), Library)),
    "deserialize-shared-bad": lambda: deserialize_json(SHARED_LIBRARY_BAD, Library),
    "deserialize-union-bad": lambda: deserialize_json([SECRET], Choice),
    "deserialize-nested-call-bad": lambda: deserialize_json({"payload": SHARED_PAYLOAD_BAD}, Envelope),
    "deserialize-nested-call-ok": lambda: serialize_json(
        deserialize_json({"payload": record_json(SIBLING)}, Envelope).payload
    ),
}


def outcome(job) -> tuple:
    try:
        return "returned", job()
    except Exception as error:
        return "raised", type(error).__name__, str(error)


def test_calls_from_many_threads_match_a_single_thread_oracle():
    oracle = {name: outcome(job) for name, job in JOBS.items()}
    Envelope.guard_seen.clear()
    assert {value[0] for value in oracle.values()} == {"returned", "raised"}
    assert sum(value[0] == "raised" for value in oracle.values()) == 7
    mismatches: list = []
    crashes: list = []
    names = list(JOBS)
    start = threading.Barrier(8)

    def worker(index: int) -> None:
        try:
            start.wait(timeout=30)
            for iteration in range(50):
                shift = (index + iteration) % len(names)
                for name in names[shift:] + names[:shift]:
                    got = outcome(JOBS[name])
                    if got != oracle[name]:
                        mismatches.append((index, iteration, name, got))
            assert_guard_clear()
        except BaseException as error:  # reported by the test thread
            crashes.append(error)

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    try:
        threads = [threading.Thread(target=worker, args=(index,), daemon=True) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=120)
    finally:
        sys.setswitchinterval(interval)

    assert not any(thread.is_alive() for thread in threads), "a worker did not finish"
    assert crashes == []
    assert mismatches == []
    assert len(Envelope.guard_seen) == 8 * 50 * sum("nested-call" in name for name in JOBS)
    assert set(Envelope.guard_seen) == {True}, "a nested call ran without the guard of its own thread"
    assert_guard_clear()

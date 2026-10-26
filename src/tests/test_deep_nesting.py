import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable

import pytest

from jserpy import deserialize_json, json_handler, serialize_json, serialize_json_as_dict, serialize_json_as_obj
from jserpy.jsonable import Jsonable


NESTED_TOO_DEEPLY = "JSON structure is nested too deeply to serialize"

# The former recursive walk failed between 480 and 600 levels under the default recursion limit (handoff repro).
OLD_DEPTH_LIMIT = 600
# Deepest structure a success case builds; the stdlib may encode much more, but deeper only costs time.
SUCCESS_DEPTH_CAP = 5000
# Indented output grows quadratically with the depth; below the ceiling of the pure-Python encoder used up to 3.12.
INDENTED_DEPTH_CAP = 800
# Search range for a depth the stdlib cannot encode (3.14 encodes about 100k nested lists); beyond the cap, jserpy's
# own bound applies before json.dumps runs.
FAILURE_DEPTH_START = 1000
FAILURE_DEPTH_CAP = json_handler._MAX_PREPARE_DEPTH
# Levels kept below the probed stdlib ceiling: jserpy calls json.dumps a few frames deeper than the probe, and a
# value converted by `CustomEncoder.default` costs the encoder one more level.
DEPTH_MARGIN = 20
# Deeper than json.dumps encodes up to 3.13 (about 10k levels; 3.14 about 60k dicts): a cycle there must be found
# by jserpy's own walk, which runs before json.dumps.
CYCLE_DEPTH = 50_000
# Depth of the other errors jserpy's walk raises before json.dumps runs: far beyond the old limit, and faster to build
WALK_ERROR_DEPTH = 20_000
# Python frames left free above the test function when the recursion limit is lowered.
LOWERED_LIMIT_HEADROOM = 150


@dataclass
class Link:
    k: Any = None


@dataclass
class Holder:
    d: dict[str, Any]


class Unsupported:
    pass


class State(Enum):
    READY = "ready"


class SelfReturningList(list, Jsonable):
    """A list with a registered (Jsonable) handler that returns the value itself."""

    def to_json(self):
        return self

    @classmethod
    def from_json(cls, data):
        return cls(data)


class SelfCopyingMapping(dict, Jsonable):
    """A mapping whose handler returns a new mapping of its own type: a new id each time, so never a cycle."""

    def to_json(self):
        return SelfCopyingMapping(self)

    @classmethod
    def from_json(cls, data):
        return cls(data)


class WrappingView(Mapping):
    """A read-only view that wraps every nested dict in a new view on access."""

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        item = self._data[key]
        return WrappingView(item) if isinstance(item, dict) else item

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


def make_endless_view():
    data: dict[str, Any] = {"a": 1}
    data["self"] = data
    return WrappingView(data)


WRAPPERS: dict[str, Callable[[Any], Any]] = {
    "dict": lambda inner: {"k": inner},
    "list": lambda inner: [inner],
    "tuple": lambda inner: (inner,),
    "mappingproxy": lambda inner: MappingProxyType({"k": inner}),
    "dataclass": lambda inner: Link(inner),
}
# What a container becomes before it reaches json.dumps (tuples stay tuples, which json.dumps encodes as arrays)
PLAIN_KINDS = {"mappingproxy": "dict", "dataclass": "dict"}

SUCCESS_KINDS = [
    pytest.param(("dict",), id="dicts"),
    pytest.param(("list",), id="lists"),
    pytest.param(("tuple",), id="tuples"),
    pytest.param(("mappingproxy",), id="mappingproxies"),
    pytest.param(("dict", "list", "tuple", "mappingproxy"), id="mixed"),
]

ENTRY_POINTS = [
    pytest.param(serialize_json, id="text"),
    pytest.param(serialize_json_as_obj, id="intermediate"),
]

INTERMEDIATE_ENTRY_POINTS = [
    pytest.param(serialize_json_as_obj, id="as-obj"),
    pytest.param(serialize_json_as_dict, id="as-dict"),
]


def plain_kinds(kinds: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(PLAIN_KINDS.get(kind, kind) for kind in kinds)


def nest(depth: int, kinds: tuple[str, ...], *, plain: bool = False, leaf: Any = None) -> Any:
    """Return `depth` nested containers around `leaf`, cycling through `kinds` from the innermost container outwards.

    With `plain`, MappingProxyType and dataclass levels are plain dicts: the same structure, encodable by json.dumps.
    """
    if plain:
        kinds = plain_kinds(kinds)
    value: Any = leaf
    for level in range(depth):
        value = WRAPPERS[kinds[level % len(kinds)]](value)
    return value


def assert_nested(result: Any, depth: int, kinds: tuple[str, ...]) -> None:
    """Check iteratively that `result` is `nest(depth, kinds)` after a JSON round trip (tuples become lists).

    Deep structures are never compared with == or repr(): both recurse.
    """
    kinds = plain_kinds(kinds)
    value = result
    for level in reversed(range(depth)):
        if kinds[level % len(kinds)] == "dict":
            assert type(value) is dict and list(value) == ["k"]
            value = value["k"]
        else:
            assert type(value) is list and len(value) == 1
            value = value[0]
    assert value is None


def _stdlib_round_trips(value: Any, options: dict[str, Any]) -> bool:
    try:
        json.loads(json.dumps(value, **options))
    except RecursionError:
        return False
    return True


def stdlib_max_depth(build: Callable[[int], Any], cap: int, **options: Any) -> int:
    """Return the deepest `build(depth)`, up to `cap`, that json.dumps (with `options`) and json.loads handle here.

    Probed by binary search in the running interpreter under the current recursion limit, because the stdlib ceiling
    depends on the version: 3.10/3.11 count the C encoder against the recursion limit, 3.12/3.13 have a separate C
    recursion limit, 3.13+ also encode `indent` in C, and 3.14 checks the C stack size.
    """
    if _stdlib_round_trips(build(cap), options):
        return cap
    handled, failed = 0, cap
    while failed - handled > 1:
        middle = (handled + failed) // 2
        if _stdlib_round_trips(build(middle), options):
            handled = middle
        else:
            failed = middle
    return handled


def success_depth(build: Callable[[int], Any], **options: Any) -> int:
    depth = stdlib_max_depth(build, SUCCESS_DEPTH_CAP, **options) - DEPTH_MARGIN
    if depth <= OLD_DEPTH_LIMIT:
        pytest.skip(f"the stdlib json module handles only {depth + DEPTH_MARGIN} levels here")
    return depth


def plain_builder(kinds: tuple[str, ...]) -> Callable[[int], Any]:
    return lambda depth: nest(depth, kinds, plain=True)


@lru_cache(maxsize=None)
def stdlib_failure_depth(kinds: tuple[str, ...]) -> int | None:
    """Return a depth at which json.dumps of the plain `nest(depth, kinds)` raises RecursionError (None: none found)."""
    depth = FAILURE_DEPTH_START
    while depth <= FAILURE_DEPTH_CAP:
        try:
            json.dumps(nest(depth, kinds, plain=True))
        except RecursionError:
            return depth
        depth *= 2
    return None


def python_stack_depth() -> int:
    depth = 0
    frame = sys._getframe(1)
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


def run(serializer: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a jserpy entry point; a RecursionError fails the test at once, without a traceback.

    pytest compares the frame locals of a RecursionError traceback to locate the recursion, which takes minutes
    when they hold deeply nested values.
    """
    try:
        return serializer(*args, **kwargs)
    except RecursionError as error:
        message = f"{serializer.__name__} raised RecursionError: {error}"
    pytest.fail(message, pytrace=False)


def assert_nested_too_deeply(error: ValueError) -> None:
    assert str(error) == NESTED_TOO_DEEPLY
    assert isinstance(error.__cause__, RecursionError)


@pytest.mark.parametrize("kinds", SUCCESS_KINDS)
def test_deep_nesting_serializes_like_the_stdlib(kinds):
    depth = success_depth(plain_builder(kinds))

    assert run(serialize_json, nest(depth, kinds)) == json.dumps(nest(depth, kinds, plain=True))


@pytest.mark.parametrize("serializer", INTERMEDIATE_ENTRY_POINTS)
@pytest.mark.parametrize("kinds", SUCCESS_KINDS)
def test_deep_nesting_round_trips_through_the_intermediate_serializer(kinds, serializer):
    depth = success_depth(plain_builder(kinds))

    assert_nested(run(serializer, nest(depth, kinds)), depth, kinds)


@pytest.mark.parametrize(
    ("options", "cap"),
    [
        pytest.param({"indent": 2}, INDENTED_DEPTH_CAP, id="indented"),
        pytest.param(
            {"separators": (",", ":"), "sort_keys": True, "ensure_ascii": False},
            SUCCESS_DEPTH_CAP,
            id="compact-sorted",
        ),
    ],
)
def test_deep_nesting_honors_encoding_options(options, cap):
    # Up to 3.12 `indent` selects the pure-Python encoder, whose recursion counts against the recursion limit.
    kinds = ("dict", "list", "tuple", "mappingproxy")
    depth = stdlib_max_depth(plain_builder(kinds), cap, **options) - DEPTH_MARGIN

    text = run(serialize_json, nest(depth, kinds), **options)

    assert text == json.dumps(nest(depth, kinds, plain=True), **options)


def test_dataclass_holding_a_deep_any_dictionary_serializes():
    # The handoff repro: `D(d: dict[str, Any])` failed with RecursionError at 600 levels.
    depth = success_depth(lambda depth: {"d": nest(depth, ("dict",))})
    value = Holder(nest(depth, ("dict",)))

    assert run(serialize_json, value) == json.dumps({"d": nest(depth, ("dict",))})

    result = run(serialize_json_as_dict, value)

    assert list(result) == ["d"]
    assert_nested(result["d"], depth, ("dict",))

    # `dict[str, Any]` values are passed through on deserialization, so the round trip is not depth-limited either
    decoded = deserialize_json(result, Holder)

    assert type(decoded) is Holder
    assert_nested(decoded.d, depth, ("dict",))


def test_deep_fallback_result_is_prepared_like_any_value():
    kinds = ("dict", "list", "tuple", "mappingproxy")
    depth = success_depth(plain_builder(kinds))

    text = run(serialize_json, Unsupported(), fallback=lambda _: nest(depth, kinds))

    assert text == json.dumps(nest(depth, kinds, plain=True))


def test_mapping_keys_are_converted_at_depth():
    keys = [(1, "1"), (True, "true"), (2.5, "2.5"), ((1, "a"), '[1, "a"]'), (State.READY, "ready")]
    depth = success_depth(plain_builder(("dict",)))
    value: Any = None
    expected: Any = None
    for level in range(depth):
        key, converted_key = keys[level % len(keys)]
        value = {key: value}
        expected = {converted_key: expected}

    assert run(serialize_json, value) == json.dumps(expected)


def test_set_at_depth_is_encoded_in_canonical_order():
    kinds = ("dict", "list", "tuple", "mappingproxy")
    depth = success_depth(lambda depth: nest(depth, kinds, plain=True, leaf=["a", "b", "c"]))

    text = run(serialize_json, nest(depth, kinds, leaf=frozenset({"c", "a", "b"})))

    assert text == json.dumps(nest(depth, kinds, plain=True, leaf=["a", "b", "c"]))


def test_deep_set_item_is_prepared_like_any_value():
    # Every set item is encoded by a walk of its own (canonical order), deep items included
    depth = success_depth(lambda depth: [nest(depth, ("tuple",))])

    text = run(serialize_json, frozenset({nest(depth, ("tuple",))}))

    assert text == json.dumps([nest(depth, ("tuple",))])


def test_mapping_key_collision_at_depth_is_detected():
    value: Any = {1: "LEFT_PAYLOAD", "1": "RIGHT_PAYLOAD"}
    for _ in range(WALK_ERROR_DEPTH):
        value = {"k": value}

    with pytest.raises(ValueError, match="(?i)colli") as caught:
        run(serialize_json, value)

    assert "PAYLOAD" not in str(caught.value)


def nest_with_shared_leaf(depth: int) -> list[Any]:
    shared = {"value": 1}
    value = [shared]
    for _ in range(depth - 1):
        value = [shared, value, shared]
    return value


def test_alias_repeated_at_every_depth_is_not_mistaken_for_a_cycle():
    depth = success_depth(nest_with_shared_leaf)
    value = nest_with_shared_leaf(depth)

    assert run(serialize_json, value) == json.dumps(value)


def build_deep_cycle(kinds: tuple[str, ...], target: str) -> tuple[Any, Any]:
    """Return the root and the deepest of CYCLE_DEPTH nested containers; the deepest one refers back to `target`.

    Containers are dicts, lists or MappingProxyType views (serialized by a registered handler); the deepest one
    returned is a dict or list, to break the cycle with `clear()`.
    """

    def create(kind: str) -> tuple[Any, Any]:
        # (the container its parent refers to, the dict or list that holds its child)
        if kind == "list":
            items: list[Any] = []
            return items, items
        backing: dict[str, Any] = {}
        return (MappingProxyType(backing) if kind == "mappingproxy" else backing), backing

    def attach(container: Any, child: Any) -> None:
        if isinstance(container, dict):
            container["k"] = child
        else:
            container.append(child)

    root, node = create(kinds[0])
    middle = root
    for level in range(1, CYCLE_DEPTH):
        child, holder = create(kinds[level % len(kinds)])
        attach(node, child)
        node = holder
        if level == CYCLE_DEPTH // 2:
            middle = child
    attach(node, root if target == "root" else middle)
    return root, node


@pytest.mark.parametrize(
    ("serializer", "kinds", "target"),
    [
        pytest.param(serialize_json, ("dict",), "root", id="text-dicts-to-root"),
        pytest.param(serialize_json, ("list",), "middle", id="text-lists-to-middle"),
        pytest.param(serialize_json, ("dict", "list"), "middle", id="text-mixed-to-middle"),
        # Even levels are MappingProxyType views: the cycle closes on a value kept tracked by its handler
        pytest.param(serialize_json, ("mappingproxy", "list"), "middle", id="text-mappingproxies-to-middle"),
        pytest.param(serialize_json_as_obj, ("list",), "root", id="intermediate-lists-to-root"),
        pytest.param(serialize_json_as_obj, ("mappingproxy", "dict"), "root", id="intermediate-mappingproxies-to-root"),
    ],
)
def test_cycle_at_depth_is_still_detected(serializer, kinds, target):
    root, deepest = build_deep_cycle(kinds, target)
    try:
        with pytest.raises(ValueError, match="(?i)cyc") as caught:
            run(serializer, root)
    finally:
        deepest.clear()  # break the cycle

    assert NESTED_TOO_DEEPLY not in str(caught.value)
    # Raised by jserpy's walk itself, not mapped from the encoder's circular check (which chains its own error)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("serializer", ENTRY_POINTS)
def test_handler_returning_its_value_at_depth_is_detected(serializer):
    value: Any = SelfReturningList(["LIST_PAYLOAD"])
    for _ in range(WALK_ERROR_DEPTH):
        value = {"k": value}

    with pytest.raises(ValueError, match="Cyclic reference returned by registered JSON handler") as caught:
        run(serializer, value)

    assert "SelfReturningList" in str(caught.value)
    assert "PAYLOAD" not in str(caught.value)


@pytest.mark.parametrize(
    ("serializer", "kinds"),
    [
        # Dicts: the shallowest stdlib ceiling (3.14 encodes about 60k nested dicts, but 100k lists)
        pytest.param(serialize_json, ("dict",), id="text-dicts"),
        pytest.param(serialize_json_as_obj, ("dict",), id="intermediate-dicts"),
        # The encoder fails inside nested `CustomEncoder.default` round trips
        pytest.param(serialize_json, ("dataclass",), id="text-dataclasses"),
    ],
)
def test_nesting_beyond_the_stdlib_encoder_raises_value_error(serializer, kinds):
    depth = stdlib_failure_depth(plain_kinds(kinds))
    if depth is None:
        pytest.skip(f"the stdlib json module encodes {FAILURE_DEPTH_CAP} levels here")
    value = nest(depth, kinds)

    with pytest.raises(ValueError) as caught:
        run(serializer, value)

    assert_nested_too_deeply(caught.value)


@pytest.mark.parametrize(
    "make_value",
    [
        pytest.param(lambda: SelfCopyingMapping(a=1), id="handler-returning-new-container"),
        pytest.param(make_endless_view, id="mapping-creating-new-mappings"),
    ],
)
def test_value_that_never_stops_nesting_fails_instead_of_exhausting_memory(make_value):
    # Every level is a new object, so the cycle check never fires: the bounded walk stops it before json.dumps runs.
    with pytest.raises(ValueError) as caught:
        run(serialize_json, make_value())

    assert str(caught.value) == NESTED_TOO_DEEPLY
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("serializer", ENTRY_POINTS)
def test_frozensets_nested_beyond_the_recursion_limit_raise_value_error(serializer):
    # Every set is encoded on its own (canonical item order), re-entering the encoder: Python frames per level, so
    # more levels than the recursion limit fail on every Python version, independently of the C encoder.
    value: frozenset[Any] = frozenset()
    for _ in range(sys.getrecursionlimit()):
        value = frozenset({value})

    with pytest.raises(ValueError) as caught:
        run(serializer, value)

    assert_nested_too_deeply(caught.value)


def test_shallow_nested_frozensets_still_serialize():
    value: frozenset[Any] = frozenset()
    for _ in range(9):
        value = frozenset({value})

    assert run(serialize_json, value) == "[" * 10 + "]" * 10


def test_intermediate_serializer_reports_a_too_deep_parse_as_value_error(monkeypatch):
    # json.loads nests at least as deep as json.dumps on 3.10-3.14, so its RecursionError is simulated.
    def loads(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded while decoding a JSON array")

    monkeypatch.setattr(json_handler, "json", SimpleNamespace(**{**vars(json), "loads": loads}))

    with pytest.raises(ValueError) as caught:
        run(serialize_json_as_obj, [[1]])

    assert_nested_too_deeply(caught.value)


@pytest.mark.parametrize(
    "kinds",
    [
        pytest.param(("dict",), id="dicts"),
        pytest.param(("list",), id="lists"),
        pytest.param(("dict", "list", "tuple", "mappingproxy"), id="mixed"),
    ],
)
def test_walk_does_not_spend_python_frames_per_level(kinds):
    original_limit = sys.getrecursionlimit()
    lowered_limit = python_stack_depth() + LOWERED_LIMIT_HEADROOM
    sys.setrecursionlimit(lowered_limit)
    try:
        depth = stdlib_max_depth(plain_builder(kinds), SUCCESS_DEPTH_CAP) - DEPTH_MARGIN
        expected = json.dumps(nest(depth, kinds, plain=True))
        text = run(serialize_json, nest(depth, kinds))
        result = run(serialize_json_as_obj, nest(depth, kinds))
    finally:
        sys.setrecursionlimit(original_limit)

    assert text == expected
    assert_nested(result, depth, kinds)
    if sys.version_info >= (3, 12):
        # The C encoder has its own limit there, so the depth exceeds the Python recursion limit: a walk that spends
        # Python frames per level could not have prepared it.
        assert depth > lowered_limit

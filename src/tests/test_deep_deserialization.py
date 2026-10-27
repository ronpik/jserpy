"""`deserialize_json` walks nested values without recursion, and reports what still recurses as ValueError.

The former recursive walk raised a bare RecursionError for recursive typed dataclasses at about 200 levels
(`Optional[Node]`), 250 (`list[Tree]`) and 330 (`dict[str, Node]`). Recursion is left only in user code, such as
`Jsonable.from_json` implementations that call `deserialize_json` themselves: the outermost call reports it as
ValueError, while nested calls let it through unchanged.

Deep values are never compared with == or repr(), and deep dataclasses are not either: both recurse.
"""

from __future__ import annotations

import json
import sys
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field, make_dataclass
from typing import Any, Callable, Optional, Union

import pytest

from jserpy import deserialize_json, json_handler, serialize_json
from jserpy.jsonable import Jsonable


NESTED_TOO_DEEPLY = "JSON structure is nested too deeply to deserialize"

# Deepest level of the former recursive walk under the default recursion limit (`dict[str, Node]`)
OLD_DEPTH_LIMIT = 331
# Depth of the typed structures built by the success cases: far beyond the old limit and the recursion limit, and far
# below the bound of the walk (`json_handler._MAX_DESERIALIZE_DEPTH`)
DEPTH = 5000
# Python frames left free above the test function when the recursion limit is lowered
LOWERED_LIMIT_HEADROOM = 150
# Bound of the walk while testing input that never stops nesting: reached quickly, unlike the real one
SMALL_BOUND = 1000
# Levels kept below the probed ceiling of the round trip through JSON text, for the frames the test spends above it
ROUND_TRIP_DEPTH_MARGIN = 20
SERIALIZED_TOO_DEEPLY = "JSON structure is nested too deeply to serialize"


@dataclass
class Node:
    value: int
    next: Optional[Node] = None


@dataclass
class Tree:
    children: list[Tree] = field(default_factory=list)


@dataclass
class MapNode:
    children: dict[str, MapNode] = field(default_factory=dict)


@dataclass
class TupleNode:
    children: tuple[TupleNode, ...] = ()


@dataclass(frozen=True)
class SetNode:
    children: frozenset[SetNode] = frozenset()


@dataclass
class MappingNode:
    children: Mapping[str, MappingNode] = field(default_factory=dict)


@dataclass
class PipeNode:
    next: PipeNode | int


@dataclass
class CheckedNode:
    value: int
    next: Optional[CheckedNode] = None

    def __post_init__(self):
        if self.value < 0:
            raise ValueError("NEGATIVE_VALUE")


BUILT: list[str] = []


@dataclass
class Logged:
    name: str
    first: Optional[Logged]
    children: list[Logged]
    second: Optional[Logged]

    def __post_init__(self):
        BUILT.append(self.name)


def run(function: Callable[..., Any], *args: Any) -> Any:
    """Call `function`; a RecursionError fails the test at once, without a traceback.

    pytest compares the frame locals of a RecursionError traceback to locate the recursion, which takes minutes
    when they hold deeply nested values.
    """
    try:
        return function(*args)
    except RecursionError as error:
        message = f"{function.__name__} raised RecursionError: {error}"
    pytest.fail(message, pytrace=False)


def wrap(depth: int, outer: Callable[[Any], Any], leaf: Any) -> Any:
    value = leaf
    for _ in range(depth):
        value = outer(value)
    return value


def node_data(depth: int) -> Any:
    # values count the levels from the innermost node (0) outwards
    value = None
    for level in range(depth):
        value = {"value": level, "next": value}
    return value


# Typed shapes: (JSON data `depth` levels deep, annotation, child of a deserialized level or None at the leaf)
SHAPES = [
    pytest.param(node_data, Node, lambda node: node.next, id="optional-dataclass"),
    pytest.param(
        lambda depth: wrap(depth - 1, lambda inner: {"children": [inner]}, {"children": []}),
        Tree,
        lambda tree: tree.children[0] if tree.children else None,
        id="list",
    ),
    pytest.param(
        lambda depth: wrap(depth - 1, lambda inner: {"children": {"k": inner}}, {"children": {}}),
        MapNode,
        lambda node: node.children["k"] if node.children else None,
        id="dict",
    ),
    pytest.param(
        lambda depth: wrap(depth - 1, lambda inner: {"children": [inner]}, {"children": []}),
        TupleNode,
        lambda node: node.children[0] if node.children else None,
        id="tuple",
    ),
    pytest.param(
        lambda depth: wrap(depth - 1, lambda inner: {"children": [inner]}, {"children": []}),
        SetNode,
        lambda node: next(iter(node.children)) if node.children else None,
        id="frozenset",
    ),
    pytest.param(
        lambda depth: wrap(depth - 1, lambda inner: {"children": {"k": inner}}, {"children": {}}),
        MappingNode,
        lambda node: node.children["k"] if node.children else None,
        id="mapping",
    ),
    pytest.param(
        lambda depth: wrap(depth - 1, lambda inner: {"next": inner}, {"next": 7}),
        PipeNode,
        lambda node: node.next if isinstance(node.next, PipeNode) else None,
        id="union-with-container-member",
    ),
]


def count_levels(value: Any, cls: type, child_of: Callable[[Any], Any]) -> int:
    levels = 0
    while value is not None:
        assert type(value) is cls
        levels += 1
        value = child_of(value)
    return levels


def python_stack_depth() -> int:
    depth = 0
    frame = sys._getframe(1)
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


@pytest.mark.parametrize(("make_data", "cls", "child_of"), SHAPES)
def test_deep_typed_structure_deserializes(make_data, cls, child_of):
    result = run(deserialize_json, make_data(DEPTH), cls)

    assert count_levels(result, cls, child_of) == DEPTH


@pytest.mark.parametrize(("make_data", "cls", "child_of"), SHAPES)
def test_walk_does_not_spend_python_frames_per_level(make_data, cls, child_of):
    data = make_data(DEPTH)
    deserialize_json(make_data(2), cls)  # resolve the annotations under the normal limit first
    original_limit = sys.getrecursionlimit()
    lowered_limit = python_stack_depth() + LOWERED_LIMIT_HEADROOM
    sys.setrecursionlimit(lowered_limit)
    try:
        result = run(deserialize_json, data, cls)
    finally:
        sys.setrecursionlimit(original_limit)

    assert DEPTH > lowered_limit
    assert count_levels(result, cls, child_of) == DEPTH


def test_values_and_side_effects_keep_the_recursive_order():
    # Children are built before their parent, fields in their declared order, list items in order
    BUILT.clear()
    data = {
        "name": "root",
        "first": {"name": "first", "children": [], "first": None, "second": None},
        "children": [
            {"name": "a", "children": [{"name": "a1", "children": []}, {"name": "a2", "children": []}]},
            {"name": "b", "children": []},
        ],
        "second": {"name": "second", "children": []},
    }

    deserialize_json(data, Logged)

    assert BUILT == ["first", "a1", "a2", "a", "b", "second", "root"]


def node_chain(depth: int) -> Node | None:
    node = None
    for level in range(depth):
        node = Node(level, node)
    return node


def text_round_trip_depth(cap: int) -> int:
    """Return the deepest node chain, up to `cap`, that `serialize_json` writes and `json.loads` parses here.

    Probed by binary search, because the ceiling depends on the version: 3.10/3.11 count the C encoder against the
    recursion limit, and every dataclass level also costs Python frames in `CustomEncoder.default` (about 500 levels
    there); 3.12/3.13 have a separate C recursion limit, and 3.14 checks the C stack size.
    """

    def handled(depth: int) -> bool:
        try:
            json.loads(serialize_json(node_chain(depth)))
        except RecursionError:
            return False
        except ValueError as error:
            if str(error) != SERIALIZED_TOO_DEEPLY:
                raise
            return False
        return True

    if handled(cap):
        return cap
    low, high = 0, cap
    while high - low > 1:
        middle = (low + high) // 2
        if handled(middle):
            low = middle
        else:
            high = middle
    return low


def test_deep_dataclass_chain_round_trips_through_json_text():
    depth = text_round_trip_depth(DEPTH) - ROUND_TRIP_DEPTH_MARGIN
    if depth <= OLD_DEPTH_LIMIT:
        pytest.skip(f"node chains round trip through JSON text only {depth + ROUND_TRIP_DEPTH_MARGIN} levels deep here")

    result = run(deserialize_json, json.loads(run(serialize_json, node_chain(depth))), Node)

    for level in reversed(range(depth)):
        assert type(result) is Node and result.value == level
        result = result.next
    assert result is None


def test_union_tries_the_next_member_after_a_type_error_at_depth():
    data = node_data(DEPTH)
    innermost = data
    while innermost["next"] is not None:
        innermost = innermost["next"]
    innermost["next"] = ["not", "a", "node"]

    result = run(deserialize_json, data, Union[Node, dict])

    assert type(result) is dict
    assert result is not data


def test_value_error_at_depth_is_not_caught_by_unions():
    data = node_data(DEPTH)
    innermost = data
    while innermost["next"] is not None:
        innermost = innermost["next"]
    innermost["value"] = -1

    with pytest.raises(ValueError, match="NEGATIVE_VALUE"):
        run(deserialize_json, data, Union[CheckedNode, dict])


class Link(Jsonable):
    """A chain of Jsonable values: every `from_json` deserializes the next link through `deserialize_json`."""

    def __init__(self, child: Link | None):
        self.child = child

    def to_json(self):
        return {"child": self.child}

    @classmethod
    def from_json(cls, data):
        return cls(deserialize_json(data["child"], Optional[Link]))


class GuardedLink(Jsonable):
    """A chain whose `from_json` replaces a child that fails with ValueError (e.g. an invalid enum value)."""

    fallbacks = 0

    def __init__(self, child: GuardedLink | None):
        self.child = child

    def to_json(self):
        return {"child": self.child}

    @classmethod
    def from_json(cls, data):
        try:
            child = deserialize_json(data["child"], Optional[GuardedLink])
        except ValueError:
            cls.fallbacks += 1
            child = None
        return cls(child)


def chain_data(depth: int) -> Any:
    return wrap(depth, lambda inner: {"child": inner}, None)


def assert_nested_too_deeply(error: ValueError) -> None:
    assert str(error) == NESTED_TOO_DEEPLY
    assert isinstance(error.__cause__, RecursionError)
    assert json_handler._deserializing.get() is False


def test_jsonable_recursion_beyond_the_recursion_limit_raises_value_error():
    # Every link costs Python frames in `from_json`, so a chain as long as the recursion limit always fails
    with pytest.raises(ValueError) as caught:
        deserialize_json(chain_data(sys.getrecursionlimit()), Link)

    assert_nested_too_deeply(caught.value)


def test_shallow_jsonable_chain_still_deserializes():
    result = deserialize_json(chain_data(50), Optional[Link])

    assert count_levels(result, Link, lambda link: link.child) == 50


def test_nested_calls_let_the_recursion_error_through_to_the_outermost_call():
    # A nested `deserialize_json` (here in `from_json`) does not convert: `except ValueError` in user code would
    # otherwise swallow the error at every level and return a truncated value
    GuardedLink.fallbacks = 0

    with pytest.raises(ValueError) as caught:
        deserialize_json(chain_data(sys.getrecursionlimit()), GuardedLink)

    assert_nested_too_deeply(caught.value)
    assert GuardedLink.fallbacks == 0


def test_jsonable_recursion_inside_a_union_raises_value_error():
    # Unions catch only TypeError: neither the RecursionError nor the ValueError makes them try the next member
    with pytest.raises(ValueError) as caught:
        deserialize_json(chain_data(sys.getrecursionlimit()), Union[Link, dict])

    assert_nested_too_deeply(caught.value)


def test_cyclic_input_raises_value_error(monkeypatch):
    monkeypatch.setattr(json_handler, "_MAX_DESERIALIZE_DEPTH", SMALL_BOUND)
    data: dict[str, Any] = {"value": 1}
    data["next"] = data

    for cls in (Node, Union[Node, dict]):
        with pytest.raises(ValueError) as caught:
            run(deserialize_json, data, cls)

        assert str(caught.value) == NESTED_TOO_DEEPLY
        assert caught.value.__cause__ is None


def test_input_below_the_bound_of_the_walk_deserializes(monkeypatch):
    # An `Optional[Node]` level takes two entries of the walk's stack (the union and the dataclass)
    monkeypatch.setattr(json_handler, "_MAX_DESERIALIZE_DEPTH", SMALL_BOUND)

    result = run(deserialize_json, node_data(SMALL_BOUND // 2), Node)

    assert count_levels(result, Node, lambda node: node.next) == SMALL_BOUND // 2


def make_leaf_class() -> type:
    # A new class each time, so that its annotations are resolved (and cached) by the call under test
    return make_dataclass("Leaf", [("a", "int"), ("b", "Optional[str]")], namespace={"__module__": __name__})


def test_annotations_failing_near_the_recursion_limit_are_not_cached(monkeypatch):
    leaf_cls = make_leaf_class()

    def get_type_hints(*args, **kwargs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(typing, "get_type_hints", get_type_hints)
    with pytest.raises(ValueError) as caught:
        deserialize_json({"a": 1, "b": "x"}, leaf_cls)
    monkeypatch.undo()

    assert_nested_too_deeply(caught.value)
    assert deserialize_json({"a": 1, "b": "x"}, leaf_cls) == leaf_cls(1, "x")


def test_jsonable_chain_ending_in_a_new_dataclass_fails_only_with_value_error():
    # Probes every depth around the recursion limit: at one of them, the leaf's annotations are first resolved with
    # too few frames left (formerly a misleading TypeError, and annotations cached as empty for good)
    failures = 0
    for depth in range(1, sys.getrecursionlimit()):
        leaf_cls = make_leaf_class()

        class LeafLink(Jsonable):
            def to_json(self):
                return {}

            @classmethod
            def from_json(cls, data):
                if "leaf" in data:
                    return deserialize_json(data["leaf"], leaf_cls)
                return deserialize_json(data["child"], LeafLink)

        data = wrap(depth, lambda inner: {"child": inner}, {"leaf": {"a": 1, "b": "x"}})
        try:
            assert deserialize_json(data, LeafLink) == leaf_cls(1, "x")
        except ValueError as error:
            assert_nested_too_deeply(error)
            failures += 1
        assert deserialize_json({"a": 1, "b": "x"}, leaf_cls) == leaf_cls(1, "x")
        if failures == 10:
            break

    assert failures == 10

"""Structures of every size and shape: deep and shallow, narrow and wide, shared and layered.

Each case has an oracle that never touches jserpy: the expected JSON text is built by string repetition or
``json.dumps`` of plain lists and dicts, and the expected object by an iterative builder. Chains are nested up to 3000
levels, so they are built and compared without recursion (``==``, ``repr`` and ``copy.deepcopy`` recurse), and the
cases that need the ``json`` module to encode or parse that depth skip where it cannot (up to 3.11 it stops near
1000 levels). The recursion limits and error bounds themselves are covered by test_deep_nesting and
test_deep_deserialization.

Recursive dataclasses come in two spellings, both walked to the same depths: a field annotated with a whole string
(``"Optional[Node]"``) and a quoted name inside a generic (``Optional["QuotedNode"]``, ``list["QuotedTree"]``).
Deserializing mapping keys other than strings and enums is left to test_objects_containers.
"""

import base64
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field, fields, is_dataclass, make_dataclass
from datetime import date
from decimal import Decimal
from enum import Enum
from functools import lru_cache, reduce
from itertools import product
from types import MappingProxyType
from typing import Any, Callable, Iterator, Optional

import pytest

from jserpy import deserialize_json, serialize_json, serialize_json_as_dict, serialize_json_as_obj
from objects_support import assert_round_trip


DEPTHS = (0, 1, 2, 3, 5, 10, 50, 200, 1000, 3000)
# Depths the shared helpers (recursive ``copy.deepcopy`` and strict comparison) and the pure-Python encoder that
# ``indent`` selects up to 3.12 handle under the default recursion limit
MODERATE_DEPTHS = tuple(depth for depth in DEPTHS if depth <= 200)
HELPER_DEPTHS = tuple(depth for depth in DEPTHS if depth <= 50)
# Levels kept free above a probe of what the json module encodes: the frames of pytest and of jserpy's own calls
STDLIB_MARGIN = 60
LEAF = 42


@dataclass
class Chain:
    child: "Optional[Chain]" = None


@dataclass
class Node:
    value: int
    next: "Optional[Node]" = None


@dataclass
class Rib:
    label: str
    weight: float


@dataclass
class Spine:
    rib: Rib
    next: "Optional[Spine]" = None


@dataclass
class TreeNode:
    value: int
    left: "Optional[TreeNode]" = None
    right: "Optional[TreeNode]" = None


@dataclass
class Branch:
    label: int
    children: "list[Branch]"


@dataclass
class Size:
    width: int
    height: int


@dataclass
class Item:
    sku: int
    name: str
    price: float
    tags: tuple[str, ...]
    size: Size
    note: Optional[str] = None


@dataclass
class Document:
    title: str
    body: str
    parts: list[str]


@dataclass
class Blob:
    name: str
    data: bytes


@dataclass
class Leaf:
    n: int


@dataclass
class Wing:
    leaf: Leaf
    side: str


@dataclass
class Fork:
    left: Wing
    right: Wing


@dataclass
class QuotedNode:
    value: int
    next: Optional["QuotedNode"] = None


@dataclass
class QuotedTree:
    label: int
    children: list["QuotedTree"] = field(default_factory=list)


@dataclass
class QuotedPipeTree:
    label: int
    children: list["QuotedPipeTree"] | None = None


@dataclass(frozen=True)
class Reordered:
    zeta: int
    alpha: int


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass
class PointSets:
    first: frozenset[Point]
    second: frozenset[Point]
    single: Point


@dataclass
class Shelf:
    left: frozenset[int]
    right: frozenset[int]
    rows: list[frozenset[int]]


@dataclass
class AnyBox:
    label: str
    body: Any


class Hue(Enum):
    H0 = "h0"
    H1 = "h1"
    H2 = "h2"
    H3 = "h3"
    H4 = "h4"
    H5 = "h5"
    H6 = "h6"
    H7 = "h7"


# ---------------------------------------------------------------------------------------------------------------
# Helpers that never recurse
# ---------------------------------------------------------------------------------------------------------------


def run(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call `function`; a RecursionError fails the test at once, without a traceback.

    pytest compares the frame locals of a RecursionError traceback to locate the recursion, which takes minutes
    when they hold deeply nested values.
    """
    try:
        return function(*args, **kwargs)
    except RecursionError as error:
        message = f"{function.__name__} raised RecursionError: {error}"
    pytest.fail(message, pytrace=False)


def nest_lists(depth: int) -> list:
    value: Any = []
    for _ in range(depth):
        value = [value]
    return value


def require_stdlib_levels(levels: int, **dump_options: Any) -> None:
    """Skip unless the json module encodes (with `dump_options`) and parses `levels` nested arrays here."""
    probe = nest_lists(levels + STDLIB_MARGIN)
    try:
        json.loads(json.dumps(probe, **dump_options))
    except RecursionError:
        pytest.skip(f"the json module cannot encode and parse {levels} nested levels here")


def _describe(path: Any) -> str:
    labels = []
    while path is not None:
        path, label = path
        labels.append(label)
    labels.reverse()
    if len(labels) > 6:
        return f"value[... {len(labels) - 6} levels ...]" + "".join(labels[-6:])
    return "value" + "".join(labels)


def _short(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _differ(path: Any, what: str) -> None:
    raise AssertionError(f"{_describe(path)}: {what}")


def _type_name(value: Any) -> str:
    return type(value).__qualname__


def assert_same_structure(actual: Any, expected: Any) -> None:
    """Assert that `actual` is `expected`: equal values of the exact same types at every level, without recursion.

    Dicts also have to list their keys in the same order. Deep values are never compared with == or shown with repr.
    Handles the values of this module's chains: dataclasses, dicts, mapping proxies, lists, tuples and scalars (sets and
    decimals are compared with assert_round_trip instead).
    """
    __tracebackhide__ = True
    pending = [(actual, expected, None)]
    while pending:
        actual, expected, path = pending.pop()
        expected_type = type(expected)
        if type(actual) is not expected_type:
            _differ(path, f"type {_type_name(actual)} instead of {_type_name(expected)}")
        if is_dataclass(expected):
            for dataclass_field in fields(expected):
                name = dataclass_field.name
                pending.append((getattr(actual, name), getattr(expected, name), (path, f".{name}")))
        elif expected_type is dict or expected_type is MappingProxyType:
            if list(actual) != list(expected):
                _differ(path, "different keys, or the same keys in a different order")
            for key, item in expected.items():
                pending.append((actual[key], item, (path, f"[{key!r}]")))
        elif expected_type is list or expected_type is tuple:
            if len(actual) != len(expected):
                _differ(path, f"length {len(actual)} instead of {len(expected)}")
            for index, item in enumerate(expected):
                pending.append((actual[index], item, (path, f"[{index}]")))
        elif expected_type is float:
            if repr(actual) != repr(expected):
                _differ(path, f"{actual!r} instead of {expected!r}")
        elif actual != expected:
            _differ(path, f"{_short(actual)} instead of {_short(expected)}")


def assert_same_text(actual: str, expected: str) -> None:
    """Assert equal texts; on a mismatch show where they start to differ instead of a diff of the whole texts."""
    __tracebackhide__ = True
    if actual == expected:
        return
    index = next(
        (position for position, (a, b) in enumerate(zip(actual, expected)) if a != b),
        min(len(actual), len(expected)),
    )
    start = max(0, index - 30)
    raise AssertionError(
        f"texts of {len(actual)} and {len(expected)} characters differ at {index}: "
        f"{actual[start:index + 30]!r} instead of {expected[start:index + 30]!r}"
    )


def iter_json_containers(root: Any) -> Iterator[Any]:
    """Yield every dict and list under `root`, once for each time it is reached (a shared one repeatedly).

    A mapping proxy is looked into but not yielded: the dict it wraps cannot be reached from it.
    """
    pending = [root]
    while pending:
        value = pending.pop()
        value_type = type(value)
        if value_type is dict:
            yield value
            pending.extend(value.values())
        elif value_type is list:
            yield value
            pending.extend(value)
        elif value_type is MappingProxyType:
            pending.extend(value.values())
        elif value_type is tuple or value_type is set or value_type is frozenset:
            pending.extend(value)
        elif is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, dataclass_field.name) for dataclass_field in fields(value))


def assert_independent_containers(result: Any, *sources: Any) -> int:
    """Assert that no dict or list of `result` is reached twice or belongs to any of `sources`; return their count."""
    __tracebackhide__ = True
    foreign = {id(container) for source in sources for container in iter_json_containers(source)}
    seen: set[int] = set()
    for container in iter_json_containers(result):
        if id(container) in foreign:
            raise AssertionError("the result holds a container of the source")
        if id(container) in seen:
            raise AssertionError("the result reaches one container through two paths")
        seen.add(id(container))
    return len(seen)


def build(leaf: Any, depth: int, wrap: Callable[[int, Any], Any]) -> Any:
    """Wrap `leaf` in `depth` layers; `wrap(level, inner)` makes layer `level`, counted from the innermost one."""
    value = leaf
    for level in range(depth):
        value = wrap(level, value)
    return value


# ---------------------------------------------------------------------------------------------------------------
# Chains: one layer per level, from trivial to 3000 levels deep
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainSpec:
    """A family of structures of any depth, each with an oracle written without jserpy.

    `value` is the object, `data` the parsed JSON it serializes to (plain dicts and lists, built separately) and
    `text` its exact compact JSON text, built by string repetition. `annotation` is the type a user restores it to.
    `levels_per_layer` is how many nested levels the json module needs per layer: a dataclass is encoded as the
    object and then as the dict it becomes.
    """

    name: str
    value: Callable[[int], Any]
    data: Callable[[int], Any]
    text: Callable[[int], str]
    annotation: Callable[[int], Any]
    levels_per_layer: int


# nested lists
def lists_data(depth: int) -> Any:
    return build(LEAF, depth, lambda level, inner: [inner])


def lists_text(depth: int) -> str:
    return "[" * depth + str(LEAF) + "]" * depth


def lists_annotation(depth: int) -> Any:
    return reduce(lambda inner, level: list[inner], range(depth), int)


def optional_lists_annotation(depth: int) -> Any:
    """`list[list[...int] | None] | None`: a union at every level, which the deserializer walks one by one.

    Written with `|` (not `Optional[...]`, whose cache hashes the annotation recursively and fails near 1000 levels).
    """
    return reduce(lambda inner, level: list[inner] | None, range(depth), int)


# nested dicts
def dicts_data(depth: int) -> Any:
    return build(LEAF, depth, lambda level, inner: {"k": inner})


def dicts_text(depth: int) -> str:
    return '{"k": ' * depth + str(LEAF) + "}" * depth


def dicts_annotation(depth: int) -> Any:
    return reduce(lambda inner, level: dict[str, inner], range(depth), int)


# dict, list and tuple layers in turn, the dict innermost
ALTERNATING_LAYERS = ("dict", "list", "tuple")


def alternating_value(depth: int) -> Any:
    def wrap(level: int, inner: Any) -> Any:
        kind = ALTERNATING_LAYERS[level % 3]
        if kind == "dict":
            return {"k": inner}
        return [inner] if kind == "list" else (inner,)

    return build(LEAF, depth, wrap)


def alternating_data(depth: int) -> Any:
    return build(
        LEAF,
        depth,
        lambda level, inner: {"k": inner} if ALTERNATING_LAYERS[level % 3] == "dict" else [inner],
    )


def alternating_text(depth: int) -> str:
    opening = "".join(
        '{"k": ' if ALTERNATING_LAYERS[level % 3] == "dict" else "[" for level in reversed(range(depth))
    )
    closing = "".join("}" if ALTERNATING_LAYERS[level % 3] == "dict" else "]" for level in range(depth))
    return opening + str(LEAF) + closing


def alternating_annotation(depth: int) -> Any:
    def wrap(inner: Any, level: int) -> Any:
        kind = ALTERNATING_LAYERS[level % 3]
        if kind == "dict":
            return dict[str, inner]
        return list[inner] if kind == "list" else tuple[inner]

    return reduce(wrap, range(depth), int)


# a single-field dataclass in a dataclass, ending in None
def chain_value(depth: int) -> Optional[Chain]:
    return build(None, depth, lambda level, inner: Chain(inner))


def chain_data(depth: int) -> Any:
    return build(None, depth, lambda level, inner: {"child": inner})


def chain_text(depth: int) -> str:
    return '{"child": ' * depth + "null" + "}" * depth


def chain_annotation(depth: int) -> Any:
    return Chain if depth else Optional[Chain]


# a linked list: the head holds 0, the tail holds depth - 1
def linked_value(depth: int) -> Optional[Node]:
    head = None
    for value in reversed(range(depth)):
        head = Node(value, head)
    return head


def linked_data(depth: int) -> Any:
    head = None
    for value in reversed(range(depth)):
        head = {"value": value, "next": head}
    return head


def linked_text(depth: int) -> str:
    return "".join(f'{{"value": {value}, "next": ' for value in range(depth)) + "null" + "}" * depth


def linked_annotation(depth: int) -> Any:
    return Node if depth else Optional[Node]


# the same linked list and a tree of one child per level, with the references to the class quoted inside a generic:
# `Optional["QuotedNode"]`, `list["QuotedTree"]` and `list["QuotedPipeTree"] | None`
def quoted_linked_annotation(depth: int) -> Any:
    return QuotedNode if depth else Optional[QuotedNode]


def quoted_linked_value(depth: int) -> Optional[QuotedNode]:
    head = None
    for value in reversed(range(depth)):
        head = QuotedNode(value, head)
    return head


def tree_chain_value(cls: type, empty: Callable[[], Any]) -> Callable[[int], Any]:
    """A chain of `depth` objects of `cls`, each the only child of the one above; `empty()` is the leaf's children."""

    def value(depth: int) -> Any:
        node = None
        for index in reversed(range(depth)):
            node = cls(index, empty() if node is None else [node])
        return node

    return value


def tree_chain_data(empty: Callable[[], Any]) -> Callable[[int], Any]:
    def data(depth: int) -> Any:
        node = None
        for index in reversed(range(depth)):
            node = {"label": index, "children": empty() if node is None else [node]}
        return node

    return data


def tree_chain_text(empty_text: str) -> Callable[[int], str]:
    def text(depth: int) -> str:
        if depth == 0:
            return "null"
        opening = "".join(f'{{"label": {index}, "children": [' for index in range(depth - 1))
        return opening + f'{{"label": {depth - 1}, "children": {empty_text}}}' + "]}" * (depth - 1)

    return text


def quoted_tree_annotation(depth: int) -> Any:
    return QuotedTree if depth else Optional[QuotedTree]


def quoted_pipe_tree_annotation(depth: int) -> Any:
    return QuotedPipeTree if depth else Optional[QuotedPipeTree]


# a caterpillar: a spine with a rib hanging at each segment
def caterpillar_value(depth: int) -> Optional[Spine]:
    head = None
    for index in reversed(range(depth)):
        head = Spine(Rib(f"r{index}", index + 0.25), head)
    return head


def caterpillar_data(depth: int) -> Any:
    head = None
    for index in reversed(range(depth)):
        head = {"rib": {"label": f"r{index}", "weight": index + 0.25}, "next": head}
    return head


def caterpillar_text(depth: int) -> str:
    segments = "".join(
        f'{{"rib": {{"label": "r{index}", "weight": {index}.25}}, "next": ' for index in range(depth)
    )
    return segments + "null" + "}" * depth


def caterpillar_annotation(depth: int) -> Any:
    return Spine if depth else Optional[Spine]


# the same payload wrapped in list, dict, tuple and dataclass layers in turn, the list innermost. Every dataclass layer
# is a class of its own whose field is annotated with the exact type of the layers below it.
MIXED_LAYERS = ("list", "dict", "tuple", "dataclass")
_MIXED_CLASSES: dict[int, type] = {}


def mixed_layer_class(level: int, body: Any) -> type:
    if level not in _MIXED_CLASSES:
        _MIXED_CLASSES[level] = make_dataclass(f"MixedLayer{level}", [("body", body)])
    return _MIXED_CLASSES[level]


def mixed_annotation(depth: int) -> Any:
    def wrap(inner: Any, level: int) -> Any:
        kind = MIXED_LAYERS[level % 4]
        if kind == "list":
            return list[inner]
        if kind == "dict":
            return dict[str, inner]
        if kind == "tuple":
            return tuple[inner]
        return mixed_layer_class(level, inner)

    return reduce(wrap, range(depth), Rib)


def mixed_value(depth: int) -> Any:
    mixed_annotation(depth)  # creates the dataclass layers

    def wrap(level: int, inner: Any) -> Any:
        kind = MIXED_LAYERS[level % 4]
        if kind == "list":
            return [inner]
        if kind == "dict":
            return {"k": inner}
        if kind == "tuple":
            return (inner,)
        return _MIXED_CLASSES[level](inner)

    return build(Rib("payload", 1.5), depth, wrap)


def mixed_data(depth: int) -> Any:
    def wrap(level: int, inner: Any) -> Any:
        kind = MIXED_LAYERS[level % 4]
        if kind == "dict":
            return {"k": inner}
        return {"body": inner} if kind == "dataclass" else [inner]

    return build({"label": "payload", "weight": 1.5}, depth, wrap)


def mixed_text(depth: int) -> str:
    opening = {"list": "[", "dict": '{"k": ', "tuple": "[", "dataclass": '{"body": '}
    closing = {"list": "]", "dict": "}", "tuple": "]", "dataclass": "}"}
    return (
        "".join(opening[MIXED_LAYERS[level % 4]] for level in reversed(range(depth)))
        + '{"label": "payload", "weight": 1.5}'
        + "".join(closing[MIXED_LAYERS[level % 4]] for level in range(depth))
    )


# a read-only mapping in a read-only mapping: the same JSON as nested dicts
def proxies_value(depth: int) -> Any:
    return build(LEAF, depth, lambda level, inner: MappingProxyType({"k": inner}))


def proxies_annotation(depth: int) -> Any:
    return reduce(lambda inner, level: MappingProxyType[str, inner], range(depth), int)


# a cons list of pairs of different types, a fixed-length tuple at every level: (0, (1, (2, ... None)))
def cons_value(depth: int) -> Any:
    head = None
    for index in reversed(range(depth)):
        head = (index, head)
    return head


def cons_data(depth: int) -> Any:
    head = None
    for index in reversed(range(depth)):
        head = [index, head]
    return head


def cons_text(depth: int) -> str:
    return "".join(f"[{index}, " for index in range(depth)) + "null" + "]" * depth


def cons_annotation(depth: int) -> Any:
    return reduce(lambda inner, level: tuple[int, inner], range(depth), type(None))


CHAINS = [
    ChainSpec("lists", lambda depth: lists_data(depth), lists_data, lists_text, lists_annotation, 1),
    ChainSpec("optional-lists", lambda depth: lists_data(depth), lists_data, lists_text, optional_lists_annotation, 1),
    ChainSpec("dicts", lambda depth: dicts_data(depth), dicts_data, dicts_text, dicts_annotation, 1),
    ChainSpec("alternating", alternating_value, alternating_data, alternating_text, alternating_annotation, 1),
    ChainSpec("mapping-proxies", proxies_value, dicts_data, dicts_text, proxies_annotation, 1),
    ChainSpec("cons-tuples", cons_value, cons_data, cons_text, cons_annotation, 1),
    ChainSpec("chain", chain_value, chain_data, chain_text, chain_annotation, 2),
    ChainSpec("linked-list", linked_value, linked_data, linked_text, linked_annotation, 2),
    ChainSpec("caterpillar", caterpillar_value, caterpillar_data, caterpillar_text, caterpillar_annotation, 2),
    ChainSpec(
        "quoted-linked-list", quoted_linked_value, linked_data, linked_text, quoted_linked_annotation, 2
    ),
    ChainSpec(
        "quoted-list-tree",
        tree_chain_value(QuotedTree, list),
        tree_chain_data(list),
        tree_chain_text("[]"),
        quoted_tree_annotation,
        3,
    ),
    ChainSpec(
        "quoted-list-or-none-tree",
        tree_chain_value(QuotedPipeTree, lambda: None),
        tree_chain_data(lambda: None),
        tree_chain_text("null"),
        quoted_pipe_tree_annotation,
        3,
    ),
    ChainSpec("mixed", mixed_value, mixed_data, mixed_text, mixed_annotation, 2),
]
CHAIN_PARAMS = [pytest.param(chain, id=chain.name) for chain in CHAINS]
DEPTH_PARAMS = [pytest.param(depth, id=f"depth-{depth}") for depth in DEPTHS]
HELPER_DEPTH_PARAMS = [pytest.param(depth, id=f"depth-{depth}") for depth in HELPER_DEPTHS]


@pytest.mark.parametrize("depth", DEPTH_PARAMS)
@pytest.mark.parametrize("chain", CHAIN_PARAMS)
def test_chain_serializes_to_the_expected_text(chain, depth):
    require_stdlib_levels(chain.levels_per_layer * depth)

    assert_same_text(run(serialize_json, chain.value(depth)), chain.text(depth))


@pytest.mark.parametrize("depth", DEPTH_PARAMS)
@pytest.mark.parametrize("chain", CHAIN_PARAMS)
def test_chain_serializes_to_the_expected_intermediate_value(chain, depth):
    require_stdlib_levels(chain.levels_per_layer * depth)
    value = chain.value(depth)

    result = run(serialize_json_as_obj, value)

    assert_same_structure(result, chain.data(depth))
    assert_independent_containers(result, value)


@pytest.mark.parametrize("depth", [pytest.param(depth, id=f"depth-{depth}") for depth in (0, 1, 3, 50)])
@pytest.mark.parametrize("chain", CHAIN_PARAMS)
def test_the_historical_dict_name_returns_the_same_intermediate_value(chain, depth):
    value = chain.value(depth)

    result = run(serialize_json_as_dict, value)

    assert_same_structure(result, chain.data(depth))
    assert_same_structure(result, run(serialize_json_as_obj, value))


@pytest.mark.parametrize("depth", DEPTH_PARAMS)
@pytest.mark.parametrize("chain", CHAIN_PARAMS)
def test_chain_deserializes_with_the_precise_annotation(chain, depth):
    data = chain.data(depth)

    restored = run(deserialize_json, data, chain.annotation(depth))

    assert_same_structure(restored, chain.value(depth))
    assert_same_structure(data, chain.data(depth))
    assert_independent_containers(restored, data)


@pytest.mark.parametrize("target", ["any", "bare"])
@pytest.mark.parametrize("depth", DEPTH_PARAMS)
@pytest.mark.parametrize("chain", CHAIN_PARAMS)
def test_chain_deserializes_with_a_dynamic_target(chain, depth, target):
    data = chain.data(depth)
    cls = Any if target == "any" else type(data)  # dict or list, or the type of a scalar at depth 0

    restored = run(deserialize_json, data, cls)

    # the JSON as it is: tuples were arrays and dataclasses objects
    assert_same_structure(restored, chain.data(depth))
    assert_same_structure(data, chain.data(depth))
    assert_independent_containers(restored, data)


@pytest.mark.parametrize("depth", DEPTH_PARAMS)
def test_linked_list_that_leaves_out_its_null_tail_is_restored(depth):
    # a producer that omits null fields writes the last node without "next"
    data = None
    for value in reversed(range(depth)):
        data = {"value": value} if data is None else {"value": value, "next": data}

    restored = run(deserialize_json, data, linked_annotation(depth))

    assert_same_structure(restored, linked_value(depth))


@pytest.mark.parametrize(
    ("data_levels", "annotation_levels", "message"),
    [
        pytest.param(0, 1, "Expected a JSON array", id="annotation-one-level-too-deep"),
        pytest.param(1, 0, "Unsupported type for deserialization", id="annotation-one-level-too-shallow"),
    ],
)
@pytest.mark.parametrize("depth", [pytest.param(depth, id=f"depth-{depth}") for depth in (1, 2, 10, 200, 1000, 3000)])
def test_nested_lists_with_an_annotation_of_the_wrong_depth_raise_type_error(
    depth, data_levels, annotation_levels, message
):
    # The error is raised at the bottom of the walk and has to come out of all the levels above it.
    data = lists_data(depth + data_levels)

    with pytest.raises(TypeError, match=message):
        run(deserialize_json, data, lists_annotation(depth + annotation_levels))

    assert_same_structure(data, lists_data(depth + data_levels))


@pytest.mark.parametrize(
    ("data_levels", "annotation_levels"),
    [
        pytest.param(0, 1, id="annotation-one-level-too-deep"),
        pytest.param(1, 0, id="annotation-one-level-too-shallow"),
    ],
)
@pytest.mark.parametrize("depth", [pytest.param(depth, id=f"depth-{depth}") for depth in (1, 5, 50)])
def test_nested_optional_lists_with_an_annotation_of_the_wrong_depth_raise_type_error(
    depth, data_levels, annotation_levels
):
    # Moderate depths only: the message of the error shows the whole annotation, and repr() of one nested 1000 levels
    # raises RecursionError on Python 3.10, which deserialize_json reports as ValueError.
    with pytest.raises(TypeError, match="Unsupported types for deserialization"):
        run(deserialize_json, lists_data(depth + data_levels), optional_lists_annotation(depth + annotation_levels))


@pytest.mark.parametrize("depth", DEPTH_PARAMS)
@pytest.mark.parametrize("chain", CHAIN_PARAMS)
def test_typed_and_dynamic_restores_are_the_same_structure(chain, depth):
    require_stdlib_levels(chain.levels_per_layer * depth)
    data = chain.data(depth)

    typed = run(deserialize_json, data, chain.annotation(depth))
    dynamic = run(deserialize_json, data, Any)

    expected = chain.text(depth)
    assert_same_text(run(serialize_json, typed), expected)
    assert_same_text(run(serialize_json, dynamic), expected)


@pytest.mark.parametrize("depth", DEPTH_PARAMS)
@pytest.mark.parametrize("chain", CHAIN_PARAMS)
def test_chain_round_trips_through_json_text(chain, depth):
    require_stdlib_levels(chain.levels_per_layer * depth)
    value = chain.value(depth)
    text = run(serialize_json, value)
    assert_same_text(run(serialize_json, value), text)

    parsed = json.loads(text)
    restored = run(deserialize_json, parsed, chain.annotation(depth))

    assert_same_structure(restored, value)
    assert_same_structure(parsed, chain.data(depth))
    assert_same_text(run(serialize_json, restored), text)


@pytest.mark.parametrize("depth", HELPER_DEPTH_PARAMS)
@pytest.mark.parametrize("chain", CHAIN_PARAMS)
def test_chain_passes_the_shared_round_trip_checks(chain, depth):
    require_stdlib_levels(chain.levels_per_layer * depth)

    run(assert_round_trip, chain.value(depth), chain.annotation(depth))


# Two spaces at every moderate depth; a tab only at a few (the unit of indentation is pinned by the test below)
INDENT_CASES = [pytest.param(depth, 2, id=f"depth-{depth}-indent-2") for depth in MODERATE_DEPTHS] + [
    pytest.param(depth, "\t", id=f"depth-{depth}-indent-tab") for depth in (1, 3, 10)
]


@pytest.mark.parametrize(("depth", "indent"), INDENT_CASES)
@pytest.mark.parametrize("chain", CHAIN_PARAMS)
def test_indented_output_parses_to_the_same_value(chain, depth, indent):
    # the pure-Python encoder that indent selects up to 3.12 spends up to 3 frames per dataclass level
    require_stdlib_levels(chain.levels_per_layer * 2 * depth, indent=indent)
    value = chain.value(depth)

    indented = run(serialize_json, value, indent=indent)

    parsed = json.loads(indented)
    # a non-empty array or object is spread over lines, a scalar is not
    assert ("\n" in indented) == isinstance(parsed, (dict, list))
    assert_same_structure(parsed, chain.data(depth))
    assert_same_structure(parsed, json.loads(run(serialize_json, value)))
    assert_same_structure(run(deserialize_json, parsed, chain.annotation(depth)), value)


@pytest.mark.parametrize(
    ("indent", "unit"),
    [
        pytest.param(2, "  ", id="indent-2"),
        pytest.param(4, "    ", id="indent-4"),
        pytest.param("\t", "\t", id="indent-tab"),
    ],
)
@pytest.mark.parametrize("depth", [1, 2, 3, 10])
def test_indented_nested_lists_gain_one_level_of_indentation_per_depth(depth, indent, unit):
    lines = (
        [unit * level + "[" for level in range(depth)]
        + [unit * depth + str(LEAF)]
        + [unit * level + "]" for level in reversed(range(depth))]
    )

    assert serialize_json(lists_data(depth), indent=indent) == "\n".join(lines)


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"separators": (",", ":")}, id="compact-separators"),
        pytest.param({"sort_keys": True}, id="sorted-keys"),
    ],
)
@pytest.mark.parametrize("depth", [pytest.param(depth, id=f"depth-{depth}") for depth in (0, 1, 3, 10, 200, 3000)])
@pytest.mark.parametrize("chain", CHAIN_PARAMS)
def test_chain_honors_compact_separators_and_sorted_keys(chain, depth, options):
    require_stdlib_levels(chain.levels_per_layer * depth)
    value = chain.value(depth)

    text = run(serialize_json, value, **options)

    assert_same_text(text, json.dumps(chain.data(depth), **options))
    # a restored dataclass does not depend on the order of the keys of its JSON object
    assert_same_structure(run(deserialize_json, json.loads(text), chain.annotation(depth)), value)


# ---------------------------------------------------------------------------------------------------------------
# Chains nested inside other containers and dataclasses
# ---------------------------------------------------------------------------------------------------------------

ENVELOPES = ("list", "dict", "tuple", "dataclass")


def enveloped(envelope: str, chain: ChainSpec, depth: int) -> tuple[Any, Any, str, Any]:
    """Return (value, parsed JSON, text, annotation) of two copies of the chain in an envelope."""
    annotation = chain.annotation(depth)
    text = chain.text(depth)
    if envelope == "list":
        return (
            [chain.value(depth), chain.value(depth)],
            [chain.data(depth), chain.data(depth)],
            f"[{text}, {text}]",
            list[annotation],
        )
    if envelope == "dict":
        return (
            {"a": chain.value(depth), "b": chain.value(depth)},
            {"a": chain.data(depth), "b": chain.data(depth)},
            f'{{"a": {text}, "b": {text}}}',
            dict[str, annotation],
        )
    if envelope == "tuple":
        return (
            (chain.value(depth), "tail"),
            [chain.data(depth), "tail"],
            f'[{text}, "tail"]',
            tuple[annotation, str],
        )
    cls = make_dataclass("Envelope", [("label", str), ("body", annotation)])
    return (
        cls("env", chain.value(depth)),
        {"label": "env", "body": chain.data(depth)},
        f'{{"label": "env", "body": {text}}}',
        cls,
    )


@pytest.mark.parametrize("envelope", ENVELOPES)
@pytest.mark.parametrize("depth", HELPER_DEPTH_PARAMS)
@pytest.mark.parametrize("chain", CHAIN_PARAMS)
def test_chain_nested_inside_other_containers(chain, depth, envelope):
    value, data, text, annotation = enveloped(envelope, chain, depth)

    assert_same_text(run(serialize_json, value), text)
    run(assert_round_trip, value, annotation)
    assert_same_structure(run(deserialize_json, data, Any), data)


def deep_envelope(envelope: str, chain: ChainSpec, depth: int) -> tuple[Any, str, Any, Any]:
    """Return (value, text, annotation, expected) of two copies of a deep chain in an envelope.

    `expected` is what deserializing the text as `annotation` has to give. A dataclass field annotated with the type of
    the chain is not an envelope here: jserpy resolves the annotations of the fields it restores with
    typing.get_type_hints, which recurses through the annotation, so a type nested 1000 levels is beyond it.
    """
    text = chain.text(depth)
    annotation = chain.annotation(depth)
    if envelope == "list":
        value = [chain.value(depth), chain.value(depth)]
        return value, f"[{text}, {text}]", list[annotation], value
    if envelope == "dict":
        value = {"a": chain.value(depth), "b": chain.value(depth)}
        return value, f'{{"a": {text}, "b": {text}}}', dict[str, annotation], value
    # an Any field takes the JSON as it is: the tuples and dataclasses of the chain come back as lists and dicts
    return (
        AnyBox("box", chain.value(depth)),
        f'{{"label": "box", "body": {text}}}',
        AnyBox,
        AnyBox("box", chain.data(depth)),
    )


# The walks of lists and tuples differ only in the container they return, so the tuple envelope is left to the
# moderate depths of test_chain_nested_inside_other_containers.
DEEP_ENVELOPES = ("list", "dict", "any-field")


@pytest.mark.parametrize("envelope", DEEP_ENVELOPES)
@pytest.mark.parametrize("depth", [pytest.param(depth, id=f"depth-{depth}") for depth in (200, 1000)])
@pytest.mark.parametrize("chain", CHAIN_PARAMS)
def test_deep_chain_nested_inside_a_container_or_an_any_field(chain, depth, envelope):
    # the list and the dict hold two copies side by side: the walk has to carry on after the first one returned from
    # the depths
    require_stdlib_levels(chain.levels_per_layer * depth + 2)
    value, text, annotation, expected = deep_envelope(envelope, chain, depth)

    assert_same_text(run(serialize_json, value), text)

    parsed = json.loads(text)
    restored = run(deserialize_json, parsed, annotation)

    assert_same_structure(restored, expected)
    assert_same_structure(parsed, json.loads(text))
    assert_independent_containers(restored, parsed)


# ---------------------------------------------------------------------------------------------------------------
# Trees: wide, and wide and deep
# ---------------------------------------------------------------------------------------------------------------


def perfect_tree(levels: int) -> tuple[Optional[TreeNode], Any]:
    """Return a perfect binary tree of `levels` levels, 2**levels - 1 nodes numbered 1.. in heap order, and its JSON."""
    if levels == 0:
        return None, None
    count = 2**levels - 1
    nodes: list[Any] = [None] * (count + 1)
    data: list[Any] = [None] * (count + 1)
    for index in range(count, 0, -1):
        left, right = (nodes[2 * index], nodes[2 * index + 1]) if 2 * index <= count else (None, None)
        left_data, right_data = (data[2 * index], data[2 * index + 1]) if 2 * index <= count else (None, None)
        nodes[index] = TreeNode(index, left, right)
        data[index] = {"value": index, "left": left_data, "right": right_data}
    return nodes[1], data[1]


def tree_nodes(root: Optional[TreeNode]) -> list[TreeNode]:
    nodes = []
    pending = [root] if root is not None else []
    while pending:
        node = pending.pop()
        nodes.append(node)
        pending.extend(child for child in (node.right, node.left) if child is not None)
    return nodes


TREE_LEVELS = [pytest.param(levels, id=f"levels-{levels}") for levels in (0, 1, 2, 3, 5, 8, 10)]


@pytest.mark.parametrize("levels", TREE_LEVELS)
def test_perfect_binary_tree_serializes_to_the_oracle_text(levels):
    tree, data = perfect_tree(levels)

    assert serialize_json(tree) == json.dumps(data)
    assert json.dumps(serialize_json_as_obj(tree)) == json.dumps(data)


@pytest.mark.parametrize("levels", TREE_LEVELS)
def test_perfect_binary_tree_round_trips(levels):
    tree, data = perfect_tree(levels)

    restored = assert_round_trip(tree, Optional[TreeNode])

    nodes = tree_nodes(restored)
    count = 2**levels - 1
    assert len(nodes) == count
    assert sorted(node.value for node in nodes) == list(range(1, count + 1))
    assert sum(node.left is None and node.right is None for node in nodes) == (2 ** (levels - 1) if levels else 0)
    assert_same_structure(deserialize_json(data, Any), data)


@pytest.mark.parametrize("levels", TREE_LEVELS)
def test_perfect_binary_tree_restores_into_the_exact_class_at_every_node(levels):
    tree, data = perfect_tree(levels)

    restored = deserialize_json(data, Optional[TreeNode])

    assert all(type(node) is TreeNode for node in tree_nodes(restored))
    assert_same_structure(restored, tree)


def branch_tree(width: int, levels: int, cls: type = Branch) -> tuple[Any, Any]:
    """Return a tree of `cls` whose nodes up to `levels` below the root have `width` children each, and its JSON."""
    label = 0
    root = cls(label, [])
    root_data: Any = {"label": label, "children": []}
    frontier = [(root, root_data)]
    for _ in range(levels):
        next_frontier = []
        for branch, branch_data in frontier:
            for _ in range(width):
                label += 1
                child, child_data = cls(label, []), {"label": label, "children": []}
                branch.children.append(child)
                branch_data["children"].append(child_data)
                next_frontier.append((child, child_data))
        frontier = next_frontier
    return root, root_data


@pytest.mark.parametrize(
    ("width", "levels"),
    [
        pytest.param(1, 0, id="a-lone-root"),
        pytest.param(1, 30, id="narrow-width-1"),
        pytest.param(6, 4, id="width-6-4-levels"),
        pytest.param(100, 1, id="width-100-1-level"),
        pytest.param(12, 3, id="width-12-3-levels"),
    ],
)
@pytest.mark.parametrize(
    "tree_class",
    [pytest.param(Branch, id="whole-string"), pytest.param(QuotedTree, id="quoted-name-in-a-generic")],
)
def test_wide_and_deep_dataclass_tree_round_trips(width, levels, tree_class):
    tree, data = branch_tree(width, levels, tree_class)
    count = sum(width**level for level in range(levels + 1))

    assert serialize_json(tree) == json.dumps(data)
    restored = assert_round_trip(tree, tree_class)

    labels = []
    pending = [restored]
    while pending:
        branch = pending.pop()
        labels.append(branch.label)
        pending.extend(branch.children)
    assert sorted(labels) == list(range(count))
    assert_same_structure(deserialize_json(data, Any), data)


@pytest.fixture(scope="module")
def wide_nested_lists():
    """Lists of 20 children, 4 levels deep: 160_000 integers in 8_421 lists."""
    level: list[Any] = list(range(20**4))
    for _ in range(4):
        level = [level[start:start + 20] for start in range(0, len(level), 20)]
    return level[0]


def test_wide_and_deep_nested_lists(wide_nested_lists):
    annotation = reduce(lambda inner, level: list[inner], range(4), int)

    text = serialize_json(wide_nested_lists)

    assert_same_text(text, json.dumps(wide_nested_lists))
    restored = assert_round_trip(wide_nested_lists, annotation)
    assert (len(restored), len(restored[0]), len(restored[0][0]), len(restored[0][0][0])) == (20, 20, 20, 20)
    assert restored[19][19][19][19] == 20**4 - 1


def test_wide_and_deep_nested_dicts():
    width = 12
    value = {
        f"k1-{outer}": {
            f"k0-{middle}": {f"leaf-{inner}": inner for inner in range(width)} for middle in range(width)
        }
        for outer in range(width)
    }
    annotation = dict[str, dict[str, dict[str, int]]]

    assert serialize_json(value) == json.dumps(value)
    restored = assert_round_trip(value, annotation)
    assert sum(len(leaves) for middle in restored.values() for leaves in middle.values()) == width**3
    assert list(restored["k1-11"]["k0-5"]) == [f"leaf-{inner}" for inner in range(width)]


# ---------------------------------------------------------------------------------------------------------------
# Breadth: many elements, long strings, long bytes
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def many_ints():
    values = list(range(-50_000, 50_000))
    return values, "[" + ", ".join(map(str, values)) + "]"


def test_list_of_100_000_integers(many_ints):
    values, text = many_ints

    assert_same_text(serialize_json(values), text)
    restored = assert_round_trip(values, list[int])
    assert len(restored) == 100_000
    assert sum(restored) == -50_000
    assert (restored[0], restored[-1]) == (-50_000, 49_999)


def test_list_of_100_000_integers_restores_as_plain_json_for_a_dynamic_target(many_ints):
    values, text = many_ints
    parsed = json.loads(text)

    for cls in (Any, list):
        restored = deserialize_json(parsed, cls)

        assert restored is not parsed
        assert_same_structure(restored, values)


@pytest.fixture(scope="module")
def many_strings():
    values = [f"s{index}" if index % 1000 else f"é{index}\n" for index in range(10_000)]
    return values, json.dumps(values)


def test_list_of_10_000_short_strings(many_strings):
    values, text = many_strings

    assert_same_text(serialize_json(values), text)
    restored = assert_round_trip(values, list[str])
    assert len(restored) == 10_000
    assert restored[1000] == "é1000\n"
    assert restored[1] == "s1"


def test_list_of_short_strings_without_ascii_escaping(many_strings):
    values, _ = many_strings

    text = serialize_json(values, ensure_ascii=False)

    assert_same_text(text, json.dumps(values, ensure_ascii=False))
    assert text.startswith('["é0\\n", "s1", "s2"')
    assert_round_trip(values, list[str], ensure_ascii=False)


@pytest.fixture(scope="module")
def many_keys():
    # inserted in descending order, so insertion order and sorted order differ
    return {f"key-{index:05d}": index for index in reversed(range(20_000))}


def test_dict_with_20_000_keys_keeps_its_insertion_order(many_keys):
    assert_same_text(serialize_json(many_keys), json.dumps(many_keys))

    restored = assert_round_trip(many_keys, dict[str, int])

    assert len(restored) == 20_000
    assert list(restored) == list(many_keys)
    assert list(restored)[:2] == ["key-19999", "key-19998"]


def test_dict_with_20_000_keys_sorted(many_keys):
    text = serialize_json(many_keys, sort_keys=True)

    assert text.startswith('{"key-00000": 0, "key-00001": 1, "key-00002": 2')
    assert text.endswith('"key-19999": 19999}')
    assert_same_text(text, json.dumps(many_keys, sort_keys=True))
    restored = assert_round_trip(many_keys, dict[str, int], sort_keys=True)
    assert list(restored) == sorted(many_keys)


def test_dict_with_20_000_keys_restores_as_plain_json_for_a_dynamic_target(many_keys):
    parsed = json.loads(serialize_json(many_keys))

    for cls in (Any, dict):
        restored = deserialize_json(parsed, cls)

        assert restored is not parsed
        assert_same_structure(restored, many_keys)


def make_items(count: int) -> list[Item]:
    return [
        Item(
            sku=index,
            name=f"item-{index}",
            price=index + 0.5,
            tags=("a", f"t{index % 7}"),
            size=Size(index % 100, index % 33),
            note=None if index % 3 else f"note {index}",
        )
        for index in range(count)
    ]


def item_data(count: int) -> list[Any]:
    return [
        {
            "sku": index,
            "name": f"item-{index}",
            "price": index + 0.5,
            "tags": ["a", f"t{index % 7}"],
            "size": {"width": index % 100, "height": index % 33},
            "note": None if index % 3 else f"note {index}",
        }
        for index in range(count)
    ]


def test_list_of_5_000_dataclass_instances():
    items = make_items(5_000)

    assert_same_text(serialize_json(items), json.dumps(item_data(5_000)))
    restored = assert_round_trip(items, list[Item])
    assert len(restored) == 5_000
    assert all(type(item) is Item and type(item.size) is Size and type(item.tags) is tuple for item in restored)
    assert sum(item.sku for item in restored) == 5_000 * 4_999 // 2
    assert sum(item.note is None for item in restored) == 3_333


def test_5_000_dataclass_instances_restore_as_plain_json_for_a_dynamic_target():
    parsed = json.loads(serialize_json(make_items(5_000)))

    restored = deserialize_json(parsed, Any)

    assert_same_structure(restored, item_data(5_000))
    assert assert_independent_containers(restored, parsed) == 5_000 * 3 + 1


@pytest.mark.parametrize(
    "options",
    [
        pytest.param({"indent": 2}, id="indent-2"),
        pytest.param({"indent": "\t", "sort_keys": True}, id="indent-tab-and-sorted-keys"),
        pytest.param({"sort_keys": True}, id="sorted-keys"),
        pytest.param({"separators": (",", ":")}, id="compact-separators"),
    ],
)
def test_1_000_dataclasses_honor_the_encoding_options(options):
    items = make_items(1_000)

    text = serialize_json(items, **options)

    assert_same_text(text, json.dumps(item_data(1_000), **options))
    # whatever the layout and the order of the keys, the same objects come back
    assert_same_structure(deserialize_json(json.loads(text), list[Item]), items)


def test_dict_with_20_000_keys_parses_to_the_same_value_with_indent(many_keys):
    text = serialize_json(many_keys, indent=2)

    assert_same_text(text, json.dumps(many_keys, indent=2))
    assert_same_structure(json.loads(text), many_keys)


def test_list_of_1_500_optional_dataclasses_with_a_gap_every_third():
    items = [None if index % 3 == 0 else item for index, item in enumerate(make_items(1_500))]
    data = [None if index % 3 == 0 else entry for index, entry in enumerate(item_data(1_500))]

    assert_same_text(serialize_json(items), json.dumps(data))
    restored = assert_round_trip(items, list[Optional[Item]])
    assert restored.count(None) == 500
    named = {f"item-{index}": item for index, item in enumerate(items)}
    assert_round_trip(named, dict[str, Item | None])


def test_mapping_proxy_with_20_000_keys(many_keys):
    value = MappingProxyType(many_keys)

    assert_same_text(serialize_json(value), json.dumps(many_keys))
    restored = assert_round_trip(value, MappingProxyType[str, int])
    assert len(restored) == 20_000
    assert list(restored)[:2] == ["key-19999", "key-19998"]


def test_dict_with_2_500_tuple_keys_serializes_each_key_as_its_json_text():
    # Restoring such keys is covered by test_objects_containers; here it is the size of the mapping that counts.
    value = {(x, y): x * y for x in range(50) for y in range(50)}
    by_text = {f"[{x}, {y}]": x * y for x in range(50) for y in range(50)}

    assert_same_text(serialize_json(value), json.dumps(by_text))
    sorted_text = serialize_json(value, sort_keys=True)
    assert_same_text(sorted_text, json.dumps(by_text, sort_keys=True))
    # sorted as text, in which "]" follows "0": [0, 0], [0, 10], [0, 11], ... and only later [0, 1]
    assert sorted_text.startswith('{"[0, 0]": 0, "[0, 10]": 0, "[0, 11]": 0')


@pytest.mark.parametrize("set_type", [pytest.param(set, id="set"), pytest.param(frozenset, id="frozenset")])
def test_set_of_5_000_integers_is_sorted_by_its_json_text(set_type):
    members = list(range(-2_500, 2_500))
    # lexical order of the decimal text, in which "-" sorts before the digits: -1, -10, -100, ..., 0, 1, 10, ...
    expected = "[" + ", ".join(sorted(map(str, members))) + "]"

    forward = set_type(members)
    backward = set_type(reversed(members))

    assert serialize_json(forward) == expected
    assert serialize_json(backward) == expected
    assert expected.startswith("[-1, -10, -100, -1000, -1001")
    restored = assert_round_trip(forward, set_type[int])
    assert len(restored) == 5_000
    assert sum(restored) == -2_500


def test_tuple_of_50_000_integers():
    value = tuple(range(50_000))

    assert_same_text(serialize_json(value), "[" + ", ".join(map(str, range(50_000))) + "]")
    restored = assert_round_trip(value, tuple[int, ...])
    assert len(restored) == 50_000 and restored[-1] == 49_999


def test_list_of_20_000_floats_comes_back_bit_for_bit():
    awkward = [5e-324, 1.7976931348623157e308, -0.0, 0.0, 0.1 + 0.2, 1e22, 1e-7, 123456789.123456789, 1.0, -1.0]
    values = awkward + [(index * 0.1) ** 3 / 7 for index in range(20_000)]

    assert_same_text(serialize_json(values), json.dumps(values))
    restored = assert_round_trip(values, list[float])
    assert [repr(value) for value in restored] == [repr(value) for value in values]
    assert restored[:10] == awkward and repr(restored[2]) == "-0.0"


def test_list_of_50_000_mixed_json_scalars_keeps_the_type_of_each():
    # 1, 1.0, True and "1" are four different JSON values, as are 0, 0.0, False, "0" and None
    cycle = [1, 1.0, True, "1", 0, 0.0, False, "0", None, -0.0, "", [], {}]
    values = [cycle[index % len(cycle)] for index in range(50_000)]
    parsed = json.loads(serialize_json(values))

    assert_same_text(serialize_json(values), json.dumps(values))
    for cls in (Any, list):
        restored = deserialize_json(parsed, cls)

        assert restored is not parsed
        assert_same_structure(restored, values)
        # the list, and an empty list and dict for each complete cycle (the 2 items left over are scalars)
        assert assert_independent_containers(restored, parsed) == 1 + 2 * (50_000 // len(cycle))


def test_fixed_length_tuple_of_1_000_positions_with_a_type_for_each():
    kinds = [
        (int, lambda i: i, lambda i: i),
        (str, lambda i: f"s{i}", lambda i: f"s{i}"),
        (float, lambda i: i + 0.5, lambda i: i + 0.5),
        (bool, lambda i: i % 2 == 0, lambda i: i % 2 == 0),
        (Decimal, lambda i: Decimal(f"{i}.0"), lambda i: f"{i}.0"),
    ]
    positions = [kinds[index % len(kinds)] for index in range(1_000)]
    value = tuple(kind[1](index) for index, kind in enumerate(positions))
    data = [kind[2](index) for index, kind in enumerate(positions)]
    annotation = tuple[tuple(kind[0] for kind in positions)]

    assert serialize_json(value) == json.dumps(data)
    restored = assert_round_trip(value, annotation)
    assert [type(item) for item in restored] == [kind[0] for kind in positions]


# (annotation, value of the field with that index) in turn: the 500 fields of WideRecord. The bool and the optional
# string alternate between their two values from one round of six kinds to the next (`index % 6` is fixed for a kind).
WIDE_FIELD_KINDS = [
    (int, lambda index: index),
    (str, lambda index: f"v{index}"),
    (float, lambda index: index + 0.5),
    (bool, lambda index: (index // 6) % 2 == 0),
    (list[int], lambda index: [index, index + 1]),
    (Optional[str], lambda index: None if (index // 6) % 2 else "x"),
]
WideRecord = make_dataclass(
    "WideRecord", [(f"field_{index:03d}", WIDE_FIELD_KINDS[index % 6][0]) for index in range(500)]
)


def test_dataclass_with_500_fields_keeps_every_field_in_declaration_order():
    data = {f"field_{index:03d}": WIDE_FIELD_KINDS[index % 6][1](index) for index in range(500)}
    value = WideRecord(**data)

    assert serialize_json(value) == json.dumps(data)
    # both values of the bool and of the optional string occur, so that a constant restore would be caught
    assert {data[f"field_{index:03d}"] for index in range(3, 500, 6)} == {True, False}
    assert {data[f"field_{index:03d}"] for index in range(5, 500, 6)} == {None, "x"}
    restored = assert_round_trip(value, WideRecord)
    assert [dataclass_field.name for dataclass_field in fields(restored)] == list(data)
    # the keys of the JSON object may come in any order
    assert_same_structure(deserialize_json(dict(reversed(data.items())), WideRecord), value)


def test_set_of_3_000_tuples_is_sorted_by_its_compact_json_text():
    members = {(x, y) for x in range(-30, 30) for y in range(50)}
    # the sort key is the compact text "[x,y]", the output is written with the default separators
    ordered = sorted(members, key=lambda member: f"[{member[0]},{member[1]}]")
    expected = "[" + ", ".join(f"[{x}, {y}]" for x, y in ordered) + "]"

    assert len(members) == 3_000
    assert_same_text(serialize_json(members), expected)
    assert_same_text(serialize_json(frozenset(reversed(sorted(members)))), expected)
    restored = assert_round_trip(members, set[tuple[int, int]])
    assert len(restored) == 3_000


def test_set_of_2_000_frozen_dataclasses_is_sorted_by_its_compact_json_text():
    members = {Point(x, -x) for x in range(-1_000, 1_000)}
    ordered = sorted(members, key=lambda point: f'{{"x":{point.x},"y":{point.y}}}')
    expected = "[" + ", ".join(f'{{"x": {point.x}, "y": {point.y}}}' for point in ordered) + "]"

    assert len(members) == 2_000
    assert_same_text(serialize_json(members), expected)
    assert_same_text(serialize_json(frozenset(sorted(members, key=lambda point: -point.x))), expected)
    restored = assert_round_trip(members, set[Point])
    assert len(restored) == 2_000


def test_set_of_30_sets_of_30_integers_is_ordered_at_both_levels():
    inner_sets = [frozenset(range(start, start + 30)) for start in range(0, 900, 30)]
    # an inner array is sorted by the text of its integers, the outer one by the compact text of the inner arrays
    inner_lists = [sorted(inner, key=str) for inner in inner_sets]
    outer_list = sorted(inner_lists, key=lambda items: json.dumps(items, separators=(",", ":")))

    value = frozenset(inner_sets)

    assert outer_list[0][:3] == [0, 1, 10]
    assert_same_text(serialize_json(value), json.dumps(outer_list))
    assert_round_trip(value, frozenset[frozenset[int]])


def test_set_order_does_not_depend_on_ensure_ascii_limitation():
    # Documents current behavior, not a promise: a set is ordered by the ASCII-escaped canonical text of its members,
    # whatever ensure_ascii says about the output. A backslash (from é) sorts between "A" and "a", so the
    # non-ASCII members come before the lowercase ones, not after them as by code point.
    members = {"é", "z", "f", "ü", chr(0x1F600), "a\nb", 'q"', "A", ""}
    expected = ["", "A", "é", "ü", chr(0x1F600), "a\nb", "f", 'q"', "z"]

    assert json.loads(serialize_json(members)) == expected
    assert json.loads(serialize_json(members, ensure_ascii=False)) == expected
    assert sorted(members) != expected
    assert_round_trip(members, set[str], ensure_ascii=False)


SET_CHAIN_DEPTHS = [pytest.param(depth, id=f"depth-{depth}") for depth in (0, 1, 2, 3, 5, 10, 20, 50)]


@pytest.mark.parametrize("depth", SET_CHAIN_DEPTHS)
def test_frozensets_nested_in_frozensets_round_trip(depth):
    # the encoder of sets recurses once per level (see test_deep_nesting), so this stays at modest depths
    value = build(LEAF, depth, lambda level, inner: frozenset({inner}))
    annotation = reduce(lambda inner, level: frozenset[inner], range(depth), int)

    assert_same_text(run(serialize_json, value), lists_text(depth))
    restored = run(assert_round_trip, value, annotation)
    for _ in range(depth):
        assert type(restored) is frozenset and len(restored) == 1
        (restored,) = restored
    assert type(restored) is int and restored == LEAF
    assert_same_structure(run(deserialize_json, lists_data(depth), Any), lists_data(depth))


@dataclass
class Empties:
    items: list[int]
    mapping: dict[str, int]
    pair: tuple[int, ...]
    tags: set[str]
    frozen: frozenset[int]
    text: str
    data: bytes


def make_empties() -> Empties:
    return Empties([], {}, (), set(), frozenset(), "", b"")


EMPTIES_TEXT = '{"items": [], "mapping": {}, "pair": [], "tags": [], "frozen": [], "text": "", "data": ""}'

TRIVIAL_STRUCTURES = [
    pytest.param([], list[int], "[]", id="empty-list"),
    pytest.param({}, dict[str, int], "{}", id="empty-dict"),
    pytest.param((), tuple[int, ...], "[]", id="empty-tuple"),
    pytest.param(set(), set[int], "[]", id="empty-set"),
    pytest.param(frozenset(), frozenset[int], "[]", id="empty-frozenset"),
    pytest.param("", str, '""', id="empty-string"),
    pytest.param(b"", bytes, '""', id="empty-bytes"),
    pytest.param(None, Optional[list[int]], "null", id="none-for-an-optional-list"),
    pytest.param([[]], list[list[int]], "[[]]", id="list-of-an-empty-list"),
    pytest.param([[], []], list[list[int]], "[[], []]", id="list-of-two-empty-lists"),
    pytest.param([{}], list[dict[str, int]], "[{}]", id="list-of-an-empty-dict"),
    pytest.param([()], list[tuple[int, ...]], "[[]]", id="list-of-an-empty-tuple"),
    pytest.param({"a": []}, dict[str, list[int]], '{"a": []}', id="dict-of-an-empty-list"),
    pytest.param({"a": {}}, dict[str, dict[str, int]], '{"a": {}}', id="dict-of-an-empty-dict"),
    pytest.param(
        ([], {}, (), set()),
        tuple[list[int], dict[str, int], tuple[int, ...], set[int]],
        "[[], {}, [], []]",
        id="tuple-of-every-empty-container",
    ),
    pytest.param([None], list[Optional[list[int]]], "[null]", id="list-of-none"),
    pytest.param(make_empties(), Empties, EMPTIES_TEXT, id="dataclass-of-every-empty-container"),
    pytest.param([make_empties()], list[Empties], "[" + EMPTIES_TEXT + "]", id="list-of-an-empty-dataclass"),
    pytest.param(
        {"k": make_empties()}, dict[str, Empties], '{"k": ' + EMPTIES_TEXT + "}", id="dict-of-an-empty-dataclass"
    ),
    pytest.param([1], list[int], "[1]", id="one-item-list"),
    pytest.param({"a": 1}, dict[str, int], '{"a": 1}', id="one-item-dict"),
    pytest.param((1,), tuple[int], "[1]", id="one-item-tuple"),
    pytest.param({1}, set[int], "[1]", id="one-item-set"),
    pytest.param(frozenset({"x"}), frozenset[str], '["x"]', id="one-item-frozenset"),
    pytest.param([[1]], list[list[int]], "[[1]]", id="list-of-a-one-item-list"),
    pytest.param(
        {"a": [{"b": (1,)}]},
        dict[str, list[dict[str, tuple[int]]]],
        '{"a": [{"b": [1]}]}',
        id="one-item-each-level",
    ),
]


@pytest.mark.parametrize(("value", "annotation", "text"), TRIVIAL_STRUCTURES)
def test_empty_and_one_item_structures(value, annotation, text):
    assert serialize_json(value) == text

    assert_round_trip(value, annotation)
    # and as the root of a longer structure, and inside another container
    assert serialize_json([value, value]) == f"[{text}, {text}]"
    assert_round_trip([value, value], list[annotation])
    assert serialize_json({"k": value}) == f'{{"k": {text}}}'
    assert_round_trip({"k": value}, dict[str, annotation])


@pytest.mark.parametrize(
    ("make_empty", "annotation", "text"),
    [
        pytest.param(list, list[int], "[]", id="lists"),
        pytest.param(dict, dict[str, int], "{}", id="dicts"),
        pytest.param(tuple, tuple[int, ...], "[]", id="tuples"),
        pytest.param(set, set[int], "[]", id="sets"),
        pytest.param(make_empties, Empties, EMPTIES_TEXT, id="dataclasses"),
    ],
)
def test_list_of_1_000_distinct_empty_containers(make_empty, annotation, text):
    value = [make_empty() for _ in range(1_000)]

    assert_same_text(serialize_json(value), "[" + ", ".join([text] * 1_000) + "]")
    restored = assert_round_trip(value, list[annotation])
    assert len(restored) == 1_000
    if make_empty is not tuple:  # the empty tuple is one object
        assert len({id(item) for item in restored}) == 1_000


AWKWARD_KEYS = [
    'a"b', "back\\slash", chr(0xE9), "line\nbreak", "tab\t", "", "null", "1", " ", chr(0x1F600), "{}", "[0]",
]


def test_nested_dicts_keyed_by_awkward_strings():
    value: Any = LEAF
    for key in AWKWARD_KEYS:
        value = {key: value}
    annotation = reduce(lambda inner, level: dict[str, inner], range(len(AWKWARD_KEYS)), int)

    text = serialize_json(value)
    unescaped_text = serialize_json(value, ensure_ascii=False)

    assert_same_text(text, json.dumps(value))
    assert_same_text(unescaped_text, json.dumps(value, ensure_ascii=False))
    assert '{"[0]": {"{}": {"' + chr(0x1F600) + '": ' in unescaped_text
    assert '"a\\"b": 42' in text and '"line\\nbreak": {' in text and '"back\\\\slash": {' in text
    assert_round_trip(value, annotation)
    assert_round_trip(value, annotation, ensure_ascii=False)


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def long_strings():
    plain = "0123456789abcdef" * 65_536
    unit = 'a"b\\c\n\té☃\U0001F600'
    return plain, unit * 100_000


def test_one_megabyte_ascii_string(long_strings):
    plain, _ = long_strings

    assert len(plain) == 1_048_576
    assert_same_text(serialize_json(plain), '"' + plain + '"')
    restored = assert_round_trip(plain, str)
    assert len(restored) == 1_048_576
    assert sha256_of(restored) == sha256_of(plain)


def test_one_million_characters_that_need_escaping(long_strings):
    _, escaped = long_strings
    # a, \", b, \\, c, \n, \t, é, ☃ and a surrogate pair: 35 characters of text for 10 of string
    prefix = 'a\\"b\\\\c\\n\\t'  # the text of a, ", b, \, c, newline and tab
    u = "\\u"
    escaped_unit = prefix + u + "00e9" + u + "2603" + u + "d83d" + u + "de00"
    unescaped_unit = prefix + chr(0xE9) + chr(0x2603) + chr(0x1F600)

    ascii_text = serialize_json(escaped)
    unicode_text = serialize_json(escaped, ensure_ascii=False)

    assert len(escaped) == 1_000_000
    assert len(ascii_text) == 100_000 * 35 + 2
    assert_same_text(ascii_text, '"' + escaped_unit * 100_000 + '"')
    assert_same_text(unicode_text, '"' + unescaped_unit * 100_000 + '"')
    for options in ({}, {"ensure_ascii": False}):
        restored = assert_round_trip(escaped, str, **options)
        assert len(restored) == 1_000_000
        assert sha256_of(restored) == sha256_of(escaped)


def test_long_strings_inside_a_dataclass_and_a_list(long_strings):
    plain, escaped = long_strings
    document = Document("long", plain, [escaped, "", plain[:3]])

    text = serialize_json(document)

    expected = (
        '{"title": "long", "body": ' + json.dumps(plain) + ', "parts": ['
        + json.dumps(escaped) + ', "", ' + json.dumps(plain[:3]) + "]}"
    )
    assert_same_text(text, expected)
    restored = assert_round_trip(document, Document)
    assert restored.body == plain and restored.parts[0] == escaped


def test_long_strings_as_dict_keys_and_set_members(long_strings):
    plain, escaped = long_strings
    mapping = {plain: 1, escaped: 2, "short": 3}

    assert_same_text(serialize_json(mapping), json.dumps(mapping))
    restored = assert_round_trip(mapping, dict[str, int])
    assert [sha256_of(key) for key in restored] == [sha256_of(key) for key in mapping]

    # each member is 300_000 characters or so; the first two differ only in their last character
    members = {plain[:300_000] + "z", plain[:300_000], escaped[:300_000], "b"}
    ordered = sorted(members, key=json.dumps)
    assert len(members) == 4
    assert_same_text(serialize_json(members), json.dumps(ordered))
    assert_round_trip(members, set[str])


@pytest.mark.parametrize(
    ("size", "padding"),
    [
        pytest.param(1_048_575, "", id="multiple-of-3"),
        pytest.param(1_048_576, "==", id="one-over-a-multiple-of-3"),
        pytest.param(1_048_577, "=", id="two-over-a-multiple-of-3"),
    ],
)
def test_one_megabyte_of_bytes_is_base64(size, padding):
    data = (bytes(range(256)) * 4_097)[:size]

    text = serialize_json(data)

    assert len(text) == 2 + 4 * -(-size // 3)
    assert text.startswith('"AAECAwQF')  # bytes 0, 1, 2 and 3, 4, 5
    assert text.endswith(padding + '"') and not text.endswith("=" + padding + '"')
    assert_same_text(text, '"' + base64.b64encode(data).decode("ascii") + '"')
    restored = assert_round_trip(data, bytes)
    assert len(restored) == size
    assert hashlib.sha256(restored).hexdigest() == hashlib.sha256(data).hexdigest()


def test_long_bytes_inside_a_dataclass_and_a_list():
    data = bytes(range(256)) * 4_096
    blobs = [Blob("big", data), Blob("empty", b""), Blob("one", b"\x00")]

    text = serialize_json(blobs)

    assert text.startswith('[{"name": "big", "data": "AAECAwQF')
    assert text.endswith('{"name": "empty", "data": ""}, {"name": "one", "data": "AA=="}]')
    restored = assert_round_trip(blobs, list[Blob])
    assert restored[0].data == data


# ---------------------------------------------------------------------------------------------------------------
# Shape matrix: outer container x inner container x leaf kind, at depth 1, 2 and 3
# ---------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LeafKind:
    name: str
    sample: Callable[[int], Any]  # the i-th value
    json: Callable[[int], Any]  # what the i-th value serializes to
    annotation: Any
    distinct: bool = True  # whether there are 8 different values, which a set of 8 leaves needs


LEAF_KINDS = [
    LeafKind("int", lambda i: i + 1, lambda i: i + 1, int),
    LeafKind("float", lambda i: i + 0.5, lambda i: i + 0.5, float),
    LeafKind("str", lambda i: f"s{i}", lambda i: f"s{i}", str),
    LeafKind("bool", lambda i: i % 2 == 0, lambda i: i % 2 == 0, bool, distinct=False),
    LeafKind("none", lambda i: None, lambda i: None, Optional[int], distinct=False),
    LeafKind("decimal", lambda i: Decimal(f"{i + 1}.50"), lambda i: f"{i + 1}.50", Decimal),
    LeafKind("date", lambda i: date(2024, 1, i + 1), lambda i: f"2024-01-{i + 1:02d}", date),
    LeafKind(
        "bytes",
        lambda i: bytes([65 + i]),
        lambda i: base64.b64encode(bytes([65 + i])).decode("ascii"),
        bytes,
    ),
    LeafKind("enum", lambda i: list(Hue)[i], lambda i: f"h{i}", Hue),
]
LAYERS = ("list", "tuple", "dict", "frozenset", "dataclass")
# Layers that can be members of a set. A frozenset itself is hashable, a tuple or dataclass if its members are.
HASHABLE_LAYERS = ("tuple", "frozenset", "dataclass")


@lru_cache(maxsize=None)
def pair_class(inner: Any) -> type:
    return make_dataclass("Pair", [("left", inner), ("right", inner)], frozen=True)


def canonical_text(data: Any) -> str:
    return json.dumps(data, separators=(",", ":"), sort_keys=True)


def layer_value(kind: str, left: Any, right: Any, inner_annotation: Any) -> Any:
    if kind == "list":
        return [left, right]
    if kind == "tuple":
        return (left, right)
    if kind == "dict":
        return {"left": left, "right": right}
    if kind == "frozenset":
        return frozenset({left, right})
    return pair_class(inner_annotation)(left, right)


def layer_data(kind: str, left: Any, right: Any) -> Any:
    if kind in ("list", "tuple"):
        return [left, right]
    if kind in ("dict", "dataclass"):
        return {"left": left, "right": right}
    return sorted([left, right], key=canonical_text)


def layer_annotation(kind: str, inner: Any) -> Any:
    if kind == "list":
        return list[inner]
    if kind == "tuple":
        return tuple[inner, inner]
    if kind == "dict":
        return dict[str, inner]
    if kind == "frozenset":
        return frozenset[inner]
    return pair_class(inner)


def build_shape(kinds: tuple[str, ...], leaf: LeafKind) -> tuple[Any, Any, Any]:
    """Return (value, parsed JSON, annotation) of the layers `kinds` (innermost first) over 2**len(kinds) leaves."""
    count = 2 ** len(kinds)
    values = [leaf.sample(index) for index in range(count)]
    datas = [leaf.json(index) for index in range(count)]
    annotation = leaf.annotation
    for kind in kinds:
        values = [layer_value(kind, values[i], values[i + 1], annotation) for i in range(0, len(values), 2)]
        datas = [layer_data(kind, datas[i], datas[i + 1]) for i in range(0, len(datas), 2)]
        annotation = layer_annotation(kind, annotation)
    return values[0], datas[0], annotation


def is_valid_shape(kinds: tuple[str, ...], leaf: LeafKind) -> bool:
    for index, kind in enumerate(kinds):
        if kind == "frozenset" and not (leaf.distinct and all(below in HASHABLE_LAYERS for below in kinds[:index])):
            return False
    return True


# Leaf kinds under the three-layer combinations that wrap a different layer around the middle one (outer + inner + outer
# has them all): one plain kind, and two that a handler converts
MIXED_TRIPLE_LEAVES = ("int", "decimal", "enum")


def matrix_cases() -> list[Any]:
    """The layers outermost first: 1 layer, every pair of layers, and every triple of layers."""
    layer_orders = [(layer,) for layer in LAYERS]
    layer_orders += list(product(LAYERS, repeat=2))
    layer_orders += list(product(LAYERS, repeat=3))
    cases = []
    for outermost_first in layer_orders:
        depth = len(outermost_first)
        kinds = tuple(reversed(outermost_first))
        for leaf in LEAF_KINDS:
            if depth == 3 and outermost_first[0] != outermost_first[2] and leaf.name not in MIXED_TRIPLE_LEAVES:
                continue
            if is_valid_shape(kinds, leaf):
                cases.append(pytest.param(kinds, leaf, id=f"d{depth}-{'+'.join(outermost_first)}-{leaf.name}"))
    return cases


MATRIX = matrix_cases()


def test_the_matrix_holds_the_combinations_that_can_exist_and_no_others():
    ids = {param.id for param in MATRIX}

    assert len(ids) == len(MATRIX)
    # the outermost layer first: every container around every kind of leaf, one to three levels deep
    assert {"d1-list-int", "d1-dataclass-enum", "d2-dict+tuple-bytes", "d3-list+dict+list-decimal"} <= ids
    # three different layers, for the leaf kinds of MIXED_TRIPLE_LEAVES
    assert {"d3-dict+list+tuple-int", "d3-list+dataclass+frozenset-enum", "d3-tuple+dict+dataclass-decimal"} <= ids
    assert not ids & {"d3-dict+list+tuple-bytes", "d3-tuple+dict+dataclass-date"}
    # a set needs hashable members and 8 different leaves: of a tuple, a frozenset or a dataclass, never of bools
    assert {"d2-frozenset+tuple-int", "d2-frozenset+dataclass-date", "d3-frozenset+frozenset+frozenset-enum"} <= ids
    assert not ids & {"d1-frozenset-bool", "d1-frozenset-none", "d2-frozenset+list-int", "d2-frozenset+dict-str"}
    assert not ids & {"d3-frozenset+tuple+list-int", "d3-tuple+frozenset+tuple-bool", "d3-tuple+frozenset+dict-int"}
    assert not any(case.startswith("d1-") and "+" in case for case in ids)


@pytest.mark.parametrize(("kinds", "leaf"), MATRIX)
def test_matrix_shape_serializes_to_the_oracle_text(kinds, leaf):
    value, data, _ = build_shape(kinds, leaf)

    assert serialize_json(value) == json.dumps(data)
    assert json.dumps(serialize_json_as_obj(value)) == json.dumps(data)


@pytest.mark.parametrize(("kinds", "leaf"), MATRIX)
def test_matrix_shape_round_trips_with_the_precise_annotation_and_as_plain_json(kinds, leaf):
    value, data, annotation = build_shape(kinds, leaf)

    assert_round_trip(value, annotation)
    # without a type to go by, the same JSON comes back as plain dicts, lists and strings
    assert_same_structure(deserialize_json(json.loads(serialize_json(value)), Any), data)


def test_dataclass_in_a_set_has_its_keys_sorted_limitation():
    # Documents current behavior, not a promise: the members of a set are re-read from their sorted-keys canonical
    # text, so a dataclass keeps its field order in a list but not in a set. The JSON value is the same.
    member = Reordered(zeta=1, alpha=2)

    assert serialize_json([member]) == '[{"zeta": 1, "alpha": 2}]'
    assert serialize_json(frozenset({member})) == '[{"alpha": 2, "zeta": 1}]'
    assert_round_trip(frozenset({member}), frozenset[Reordered])


# ---------------------------------------------------------------------------------------------------------------
# Sharing is not a cycle
# ---------------------------------------------------------------------------------------------------------------


def test_diamond_of_plain_containers_is_serialized_in_full_on_every_path():
    shared = [1, 2, 3]
    value = {"left": {"x": shared}, "right": {"x": shared, "y": shared}}

    assert serialize_json(value) == (
        '{"left": {"x": [1, 2, 3]}, "right": {"x": [1, 2, 3], "y": [1, 2, 3]}}'
    )
    result = serialize_json_as_obj(value)
    assert result == {"left": {"x": [1, 2, 3]}, "right": {"x": [1, 2, 3], "y": [1, 2, 3]}}
    assert assert_independent_containers(result, value) == 6


def test_diamond_of_tuples_round_trips_with_every_tuple_restored():
    pair = (1, 2)
    value = ([pair, pair], {"p": pair})

    assert serialize_json(value) == '[[[1, 2], [1, 2]], {"p": [1, 2]}]'
    assert_round_trip(value, tuple[list[tuple[int, int]], dict[str, tuple[int, int]]])


def test_diamond_of_dataclass_instances_restores_independent_instances():
    leaf = Leaf(1)
    value = Fork(Wing(leaf, "l"), Wing(leaf, "r"))

    assert serialize_json(value) == (
        '{"left": {"leaf": {"n": 1}, "side": "l"}, "right": {"leaf": {"n": 1}, "side": "r"}}'
    )
    restored = assert_round_trip(value, Fork)
    assert restored.left.leaf is not restored.right.leaf
    assert restored.left.leaf is not leaf
    restored.left.leaf.n = 99
    assert restored.right.leaf.n == 1
    assert leaf.n == 1


def test_one_frozenset_in_several_fields_is_serialized_every_time():
    members = frozenset({1, 2})
    shelf = Shelf(members, members, [members, members, members])

    assert serialize_json(shelf) == '{"left": [1, 2], "right": [1, 2], "rows": [[1, 2], [1, 2], [1, 2]]}'
    assert_round_trip(shelf, Shelf)


def test_one_frozen_dataclass_in_two_sets_and_a_field_is_serialized_every_time():
    point, other = Point(1, 2), Point(3, 4)
    value = PointSets(frozenset({point}), frozenset({point, other}), point)

    assert serialize_json(value) == (
        '{"first": [{"x": 1, "y": 2}], "second": [{"x": 1, "y": 2}, {"x": 3, "y": 4}], "single": {"x": 1, "y": 2}}'
    )
    assert_round_trip(value, PointSets)


def test_a_tuple_that_is_a_mapping_key_and_a_value_at_once_is_not_a_cycle():
    # Restoring tuple keys is covered by test_objects_containers; this is about serializing shared objects.
    pair = (1, 2)
    point = Point(1, 2)

    assert serialize_json({pair: pair}) == '{"[1, 2]": [1, 2]}'
    assert serialize_json([{pair: 1}, {pair: 2}, pair]) == '[{"[1, 2]": 1}, {"[1, 2]": 2}, [1, 2]]'
    assert serialize_json({(point, 1): point}) == json.dumps({json.dumps([{"x": 1, "y": 2}, 1]): {"x": 1, "y": 2}})


@pytest.mark.parametrize("count", [2, 10, 1000])
def test_one_list_referenced_many_times_is_serialized_every_time(count):
    shared = [1, 2, 3]

    assert serialize_json([shared] * count) == "[" + ", ".join(["[1, 2, 3]"] * count) + "]"
    assert serialize_json({f"k{index}": shared for index in range(count)}) == (
        "{" + ", ".join(f'"k{index}": [1, 2, 3]' for index in range(count)) + "}"
    )
    assert serialize_json([Leaf(7)] * count) == "[" + ", ".join(['{"n": 7}'] * count) + "]"
    assert assert_independent_containers(serialize_json_as_obj([shared] * count), shared) == count + 1


@pytest.mark.parametrize("count", [2, 10, 1000])
def test_references_to_one_list_are_restored_as_independent_lists(count):
    shared = [1, 2, 3]
    value = [shared] * count

    restored = assert_round_trip(value, list[list[int]])

    assert len({id(item) for item in restored}) == count
    assert all(item is not shared for item in restored)
    restored[0].append(4)
    assert all(item == [1, 2, 3] for item in restored[1:])
    assert shared == [1, 2, 3]
    restored_dicts = assert_round_trip({f"k{index}": shared for index in range(count)}, dict[str, list[int]])
    assert len({id(item) for item in restored_dicts.values()}) == count
    restored_leaves = assert_round_trip([Leaf(7)] * count, list[Leaf])
    assert len({id(item) for item in restored_leaves}) == count


@pytest.mark.parametrize("count", [2, 1000])
def test_shared_input_is_restored_as_independent_lists_for_a_typed_target(count):
    shared = [1, 2, 3]

    restored = deserialize_json([shared] * count, list[list[int]])

    assert len({id(item) for item in restored}) == count
    assert all(item is not shared for item in restored)


def test_shared_input_keeps_its_sharing_for_an_untyped_target_limitation():
    # Documents current behavior, not a promise (test_input_isolation pins it for other shapes): an untyped target
    # copies the input, preserving which containers were one object, so references stay references to one copy.
    shared = [1, 2, 3]

    restored = deserialize_json([shared] * 1000, Any)

    assert len({id(item) for item in restored}) == 1
    assert restored[0] is not shared
    assert restored[0] == [1, 2, 3]


@pytest.mark.parametrize("levels", [1, 2, 5, 10, 12])
def test_doubling_diamond_is_expanded_on_every_path(levels):
    value: Any = 1
    text = "1"
    for _ in range(levels):
        value = [value, value]  # `levels` lists stand for 2**levels - 1 lists in the expanded text
        text = f"[{text}, {text}]"
    annotation = reduce(lambda inner, level: list[inner], range(levels), int)

    assert_same_text(serialize_json(value), text)
    restored = assert_round_trip(value, annotation)
    assert assert_independent_containers(restored, value) == 2**levels - 1


@pytest.mark.parametrize("levels", [1, 2, 5, 8, 10])
def test_doubling_diamond_of_dataclass_nodes_restores_every_node_as_its_own_object(levels):
    node = None
    data = None
    for height in range(1, levels + 1):
        node = TreeNode(height, node, node)
        data = {"value": height, "left": data, "right": data}

    assert serialize_json(node) == json.dumps(data)
    restored = assert_round_trip(node, TreeNode)

    restored_nodes = tree_nodes(restored)
    assert len(restored_nodes) == 2**levels - 1
    assert len({id(restored_node) for restored_node in restored_nodes}) == 2**levels - 1
    assert Counter(restored_node.value for restored_node in restored_nodes) == {
        height: 2 ** (levels - height) for height in range(1, levels + 1)
    }


@pytest.mark.parametrize(
    ("serializer", "diamond"),
    [
        pytest.param(serialize_json, '{"left": {"x": [1]}, "right": {"x": [1]}}', id="text"),
        pytest.param(serialize_json_as_obj, {"left": {"x": [1]}, "right": {"x": [1]}}, id="intermediate-value"),
    ],
)
def test_the_diamond_with_a_back_edge_is_a_cycle(serializer, diamond):
    shared = [1]
    value = {"left": {"x": shared}, "right": {"x": shared}}

    assert serializer(value) == diamond  # as it is, the diamond is fine
    shared.append(value)

    with pytest.raises(ValueError, match="Cyclic reference detected"):
        serializer(value)


# ---------------------------------------------------------------------------------------------------------------
# Many structures of different depths in one value
# ---------------------------------------------------------------------------------------------------------------


def test_forest_of_linked_lists_of_every_length():
    forest = {f"list-{length:03d}": linked_value(length) for length in range(100)}
    text = "{" + ", ".join(f'"list-{length:03d}": {linked_text(length)}' for length in range(100)) + "}"

    assert_same_text(serialize_json(forest), text)
    restored = assert_round_trip(forest, dict[str, Optional[Node]])
    lengths = []
    for head in restored.values():
        length = 0
        while head is not None:
            length, head = length + 1, head.next
        lengths.append(length)
    assert lengths == list(range(100))


def test_forest_of_chains_of_every_kind_next_to_each_other():
    depth = 40
    parts = [(chain.value(depth), chain.data(depth), chain.text(depth), chain.annotation(depth)) for chain in CHAINS]
    value = [part[0] for part in parts]

    text = serialize_json(value)

    assert_same_text(text, "[" + ", ".join(part[2] for part in parts) + "]")
    restored = deserialize_json(json.loads(text), Any)
    assert_same_structure(restored, [part[1] for part in parts])
    for restored_part, part in zip(json.loads(text), parts):
        assert_same_structure(deserialize_json(restored_part, part[3]), part[0])


# ---------------------------------------------------------------------------------------------------------------
# Quoted forward references inside generic annotations
# ---------------------------------------------------------------------------------------------------------------


def test_dataclasses_with_quoted_forward_references_serialize():
    node = QuotedNode(1, QuotedNode(2))
    tree = QuotedTree(1, [QuotedTree(2), QuotedTree(3, [QuotedTree(4)])])

    assert serialize_json(node) == '{"value": 1, "next": {"value": 2, "next": null}}'
    assert serialize_json(tree) == (
        '{"label": 1, "children": [{"label": 2, "children": []},'
        ' {"label": 3, "children": [{"label": 4, "children": []}]}]}'
    )


# Three spellings: Optional['X'] and list['X'] (a ForwardRef and a plain str inside a typing / builtin generic) and
# list['X'] | None (a str inside a generic inside a union).
@pytest.mark.parametrize(
    ("value", "cls"),
    [
        pytest.param(QuotedNode(1, QuotedNode(2, QuotedNode(3))), QuotedNode, id="optional-quoted-class"),
        pytest.param(
            QuotedTree(1, [QuotedTree(2), QuotedTree(3, [QuotedTree(4)])]), QuotedTree, id="list-of-quoted-class"
        ),
        pytest.param(
            QuotedPipeTree(1, [QuotedPipeTree(2), QuotedPipeTree(3, [QuotedPipeTree(4)])]),
            QuotedPipeTree,
            id="list-of-quoted-class-or-none",
        ),
    ],
)
def test_dataclasses_with_quoted_forward_references_round_trip(value, cls):
    assert_round_trip(value, cls)


def test_quoted_forward_references_are_restored_from_hand_written_json():
    restored = deserialize_json(
        {"label": 1, "children": [{"label": 2, "children": None}, {"label": 3}]}, QuotedPipeTree
    )

    assert_same_structure(restored, QuotedPipeTree(1, [QuotedPipeTree(2), QuotedPipeTree(3)]))
    assert_same_structure(
        deserialize_json({"value": 1, "next": {"value": 2}}, QuotedNode), QuotedNode(1, QuotedNode(2))
    )

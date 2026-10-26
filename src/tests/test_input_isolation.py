"""Deserialized values must not alias the caller's input, and serialized trees must not alias the object.

`deserialize_json` passes some values through as is (`Any`, unparameterized containers, ...). Every JSON container
(dict / list) in the result must still be a new object, at every depth: mutating the input after the call must not
change the result, and vice versa. Values that are not JSON containers (scalars, tuples or arbitrary objects in
hand-built input) may be kept by reference, so these tests only use JSON input (as produced by `json.loads`).
"""

import gc
import json
import sys
import threading
import typing
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from types import MappingProxyType
from typing import Any, Optional

import pytest

from jserpy import deserialize_json, serialize_json_as_dict, serialize_json_as_obj

MUTATION = "MUTATED"


def reachable_containers(value):
    """Every dict / list reachable from `value`, found iteratively (deep and cyclic values are fine).

    Descends into dicts, lists, tuples, sets, frozensets, mapping proxies and dataclass fields. For a mapping proxy
    the wrapped mapping itself is included: it is not reachable through the public API, but it is shared state all
    the same (mutating it changes the proxy).
    """
    containers = []
    seen = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, (dict, list, tuple, set, frozenset, MappingProxyType)) or (
            is_dataclass(current) and not isinstance(current, type)
        ):
            if id(current) in seen:
                continue
            seen.add(id(current))
        else:
            continue

        if isinstance(current, dict):
            containers.append(current)
            pending.extend(current.values())
        elif isinstance(current, list):
            containers.append(current)
            pending.extend(current)
        elif isinstance(current, (tuple, set, frozenset)):
            pending.extend(current)
        elif isinstance(current, MappingProxyType):
            pending.extend(gc.get_referents(current))
        else:
            pending.extend(getattr(current, field.name) for field in fields(current))
    return containers


def container_ids(value):
    return {id(container) for container in reachable_containers(value)}


def mutate_deeply(value):
    """Add an entry to every dict / list reachable from `value`."""
    containers = reachable_containers(value)
    assert containers, "nothing to mutate: the test would be vacuous"
    for container in containers:
        if isinstance(container, dict):
            container[MUTATION] = MUTATION
        else:
            container.append(MUTATION)


# Catalogue of JSON payloads. Each constant is parsed with `json.loads` whenever a fresh, independent copy is needed.

OBJECT_TEXT = json.dumps(
    {
        "list": [1, {"inner": [2, 3]}, [4, [5, {}]]],
        "object": {"deeper": {"deepest": [], "empty": {}}},
        "text": "value",
        "number": 1.5,
        "flag": True,
        "null": None,
    }
)

ARRAY_TEXT = json.dumps([{"inner": [1, {"deeper": []}]}, [2, [3, {}]], "text", 4, None])

RECORDS_TEXT = json.dumps([{"a": [1, {"b": []}]}, {"c": {"d": [2]}}, {}])

LISTS_BY_KEY_TEXT = json.dumps({"first": [1, [2, {"inner": []}]], "second": [{"deeper": [3]}], "empty": []})


@dataclass
class Leaf:
    any_value: Any
    any_dict: dict[str, Any]
    any_list: list[Any]
    bare_dict: dict
    bare_list: list
    bare_tuple: tuple
    any_tuple: tuple[Any, ...]
    any_mapping: Mapping[str, Any]
    bare_mapping: Mapping
    any_proxy: MappingProxyType[str, Any]
    bare_proxy: MappingProxyType
    optional_any: Optional[Any]
    optional_dict: dict[str, Any] | None
    typing_dict: typing.Dict[str, Any]
    lists_by_key: dict[str, list]


@dataclass
class Branch:
    leaf: Leaf
    leaves: list[Leaf]
    leaves_by_name: dict[str, Leaf]
    optional_leaf: Optional[Leaf]
    proxied_leaves: MappingProxyType[str, Leaf]
    leaf_tuple: tuple[Leaf, ...]


def leaf_json():
    return {
        "any_value": json.loads(OBJECT_TEXT),
        "any_dict": json.loads(OBJECT_TEXT),
        "any_list": json.loads(ARRAY_TEXT),
        "bare_dict": json.loads(OBJECT_TEXT),
        "bare_list": json.loads(ARRAY_TEXT),
        "bare_tuple": json.loads(ARRAY_TEXT),
        "any_tuple": json.loads(ARRAY_TEXT),
        "any_mapping": json.loads(OBJECT_TEXT),
        "bare_mapping": json.loads(OBJECT_TEXT),
        "any_proxy": json.loads(OBJECT_TEXT),
        "bare_proxy": json.loads(OBJECT_TEXT),
        "optional_any": json.loads(ARRAY_TEXT),
        "optional_dict": json.loads(OBJECT_TEXT),
        "typing_dict": json.loads(OBJECT_TEXT),
        "lists_by_key": json.loads(LISTS_BY_KEY_TEXT),
    }


BRANCH_TEXT = json.dumps(
    {
        "leaf": leaf_json(),
        "leaves": [leaf_json(), leaf_json()],
        "leaves_by_name": {"first": leaf_json(), "second": leaf_json()},
        "optional_leaf": leaf_json(),
        "proxied_leaves": {"first": leaf_json()},
        "leaf_tuple": [leaf_json()],
    }
)


def expected_leaf(data):
    return Leaf(
        any_value=data["any_value"],
        any_dict=data["any_dict"],
        any_list=data["any_list"],
        bare_dict=data["bare_dict"],
        bare_list=data["bare_list"],
        bare_tuple=tuple(data["bare_tuple"]),
        any_tuple=tuple(data["any_tuple"]),
        any_mapping=data["any_mapping"],
        bare_mapping=data["bare_mapping"],
        any_proxy=MappingProxyType(data["any_proxy"]),
        bare_proxy=MappingProxyType(data["bare_proxy"]),
        optional_any=data["optional_any"],
        optional_dict=data["optional_dict"],
        typing_dict=data["typing_dict"],
        lists_by_key=data["lists_by_key"],
    )


def expected_branch(data):
    return Branch(
        leaf=expected_leaf(data["leaf"]),
        leaves=[expected_leaf(item) for item in data["leaves"]],
        leaves_by_name={key: expected_leaf(item) for key, item in data["leaves_by_name"].items()},
        optional_leaf=expected_leaf(data["optional_leaf"]),
        proxied_leaves=MappingProxyType(
            {key: expected_leaf(item) for key, item in data["proxied_leaves"].items()}
        ),
        leaf_tuple=tuple(expected_leaf(item) for item in data["leaf_tuple"]),
    )


def same(data):
    return data


SHAPES = [
    pytest.param(Any, OBJECT_TEXT, same, id="any-object"),
    pytest.param(Any, ARRAY_TEXT, same, id="any-array"),
    pytest.param(dict[str, Any], OBJECT_TEXT, same, id="dict-str-any"),
    pytest.param(typing.Dict[str, Any], OBJECT_TEXT, same, id="typing-dict-str-any"),
    pytest.param(dict, OBJECT_TEXT, same, id="bare-dict"),
    pytest.param(dict[str, list], LISTS_BY_KEY_TEXT, same, id="dict-str-bare-list"),
    pytest.param(list[Any], ARRAY_TEXT, same, id="list-any"),
    pytest.param(typing.List[Any], ARRAY_TEXT, same, id="typing-list-any"),
    pytest.param(list, ARRAY_TEXT, same, id="bare-list"),
    pytest.param(list[dict[str, Any]], RECORDS_TEXT, same, id="list-dict-str-any"),
    pytest.param(tuple, ARRAY_TEXT, tuple, id="bare-tuple"),
    pytest.param(tuple[Any, ...], ARRAY_TEXT, tuple, id="tuple-any-ellipsis"),
    pytest.param(Mapping[str, Any], OBJECT_TEXT, same, id="mapping-str-any"),
    pytest.param(typing.Mapping[str, Any], OBJECT_TEXT, same, id="typing-mapping-str-any"),
    pytest.param(Mapping, OBJECT_TEXT, same, id="bare-mapping"),
    pytest.param(MappingProxyType[str, Any], OBJECT_TEXT, MappingProxyType, id="mappingproxy-str-any"),
    pytest.param(MappingProxyType, OBJECT_TEXT, MappingProxyType, id="bare-mappingproxy"),
    pytest.param(Optional[Any], OBJECT_TEXT, same, id="optional-any"),
    pytest.param(dict[str, Any] | None, OBJECT_TEXT, same, id="dict-str-any-or-none"),
    pytest.param(Optional[list[Any]], ARRAY_TEXT, same, id="optional-list-any"),
    pytest.param(Leaf, json.dumps(leaf_json()), expected_leaf, id="dataclass"),
    pytest.param(Branch, BRANCH_TEXT, expected_branch, id="nested-dataclasses"),
]


def test_identity_walk_detects_aliasing_at_every_depth():
    # Self-check of the helper: a shared container is found through lists, tuples, proxies and dataclass fields.
    shared = [1]
    source = {"outer": [{"inner": shared}]}
    source_ids = container_ids(source)

    assert len(source_ids) == 4
    assert not container_ids([({"value": shared},)]).isdisjoint(source_ids)
    assert not container_ids(MappingProxyType({"value": [shared]})).isdisjoint(source_ids)
    assert not container_ids(MappingProxyType(source)).isdisjoint(source_ids)
    assert not container_ids(Settings(options={}, values=[(shared,)], untyped={})).isdisjoint(source_ids)
    assert container_ids(json.loads(json.dumps(source))).isdisjoint(source_ids)


@pytest.mark.parametrize(("annotation", "text", "expected_from"), SHAPES)
def test_deserialized_value_shares_no_container_with_input(annotation, text, expected_from):
    source = json.loads(text)

    result = deserialize_json(source, annotation)

    assert container_ids(result).isdisjoint(container_ids(source))


@pytest.mark.parametrize(("annotation", "text", "expected_from"), SHAPES)
def test_deserialized_value_round_trips_equal(annotation, text, expected_from):
    result = deserialize_json(json.loads(text), annotation)

    expected = expected_from(json.loads(text))
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(("annotation", "text", "expected_from"), SHAPES)
def test_mutating_input_after_deserializing_leaves_result_unchanged(annotation, text, expected_from):
    source = json.loads(text)
    result = deserialize_json(source, annotation)

    mutate_deeply(source)

    assert source != json.loads(text)
    assert result == expected_from(json.loads(text))


@pytest.mark.parametrize(("annotation", "text", "expected_from"), SHAPES)
def test_mutating_result_leaves_input_unchanged(annotation, text, expected_from):
    source = json.loads(text)
    result = deserialize_json(source, annotation)

    mutate_deeply(result)

    assert result != expected_from(json.loads(text))
    assert source == json.loads(text)


def test_mappingproxy_results_stay_mappingproxies():
    source = json.loads(BRANCH_TEXT)

    result = deserialize_json(source, Branch)
    mutate_deeply(source)

    assert type(result.proxied_leaves) is MappingProxyType
    leaves = [result.leaf, *result.leaves, *result.leaves_by_name.values(), result.optional_leaf]
    leaves += [*result.proxied_leaves.values(), *result.leaf_tuple]
    for leaf in leaves:
        assert type(leaf.any_proxy) is MappingProxyType
        assert type(leaf.bare_proxy) is MappingProxyType
        assert leaf.any_proxy == json.loads(OBJECT_TEXT)
        with pytest.raises(TypeError):
            leaf.any_proxy["new"] = 1  # type: ignore[index]


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param(MappingProxyType[str, Any], id="mappingproxy-str-any"),
        pytest.param(MappingProxyType, id="bare-mappingproxy"),
    ],
)
def test_mappingproxy_result_does_not_wrap_the_input(annotation):
    source = json.loads(OBJECT_TEXT)

    result = deserialize_json(source, annotation)
    source["added"] = 1

    assert type(result) is MappingProxyType
    assert "added" not in result
    assert all(wrapped is not source for wrapped in gc.get_referents(result))


def call_with_timeout(function, *args, timeout=10.0):
    """Run `function` in a daemon thread, so that an endless loop fails the test instead of hanging the run."""
    outcome = {}

    def target():
        try:
            outcome["value"] = function(*args)
        except BaseException as error:  # re-raised in the calling thread
            outcome["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        pytest.fail(f"call did not return within {timeout} seconds")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def make_self_containing_list():
    value = [1]
    value.append(value)
    return value


def make_self_containing_dictionary():
    value = {"number": 1}
    value["self"] = value
    return value


def make_mutually_containing_lists():
    first = []
    second = [first]
    first.append(second)
    return first


ANY_OR_BARE_LIST = [
    pytest.param(Any, id="any"),
    pytest.param(list, id="bare-list"),
    pytest.param(Optional[Any], id="optional-any"),
]

ANY_OR_BARE_DICT = [
    pytest.param(Any, id="any"),
    pytest.param(dict, id="bare-dict"),
    pytest.param(Optional[Any], id="optional-any"),
]


@pytest.mark.parametrize("annotation", ANY_OR_BARE_LIST)
def test_self_containing_list_is_copied_with_its_cycle(annotation):
    source = make_self_containing_list()

    result = call_with_timeout(deserialize_json, source, annotation)

    assert result is not source
    assert type(result) is list
    assert len(result) == 2
    assert result[0] == 1
    assert result[1] is result


@pytest.mark.parametrize("annotation", ANY_OR_BARE_DICT)
def test_self_containing_dictionary_is_copied_with_its_cycle(annotation):
    source = make_self_containing_dictionary()

    result = call_with_timeout(deserialize_json, source, annotation)

    assert result is not source
    assert type(result) is dict
    assert result.keys() == {"number", "self"}
    assert result["number"] == 1
    assert result["self"] is result


def test_mutually_containing_lists_are_copied_with_their_cycle():
    source = make_mutually_containing_lists()

    result = call_with_timeout(deserialize_json, source, Any)

    assert result is not source
    assert result[0] is not source[0]
    assert result[0][0] is result


@pytest.mark.parametrize(
    ("make_source", "annotation"),
    [
        pytest.param(make_self_containing_list, list[Any], id="list-any"),
        pytest.param(make_self_containing_list, tuple, id="bare-tuple"),
        pytest.param(make_self_containing_list, tuple[Any, ...], id="tuple-any-ellipsis"),
        pytest.param(make_self_containing_dictionary, dict[str, Any], id="dict-str-any"),
        pytest.param(make_self_containing_dictionary, Mapping, id="bare-mapping"),
        pytest.param(make_self_containing_dictionary, Mapping[str, Any], id="mapping-str-any"),
        pytest.param(make_self_containing_dictionary, MappingProxyType, id="bare-mappingproxy"),
        pytest.param(make_self_containing_dictionary, MappingProxyType[str, Any], id="mappingproxy-str-any"),
    ],
)
def test_cyclic_input_under_containers_returns_detached_value(make_source, annotation):
    source = make_source()

    result = call_with_timeout(deserialize_json, source, annotation)

    assert container_ids(result).isdisjoint(container_ids(source))


@pytest.mark.parametrize("annotation", ANY_OR_BARE_DICT)
def test_shared_reference_stays_shared_within_one_copied_value(annotation):
    inner = {"values": [1, 2]}
    source = {"first": inner, "second": inner, "nested": [inner]}

    result = deserialize_json(source, annotation)

    assert result["first"] is result["second"]
    assert result["nested"][0] is result["first"]
    assert result["first"] is not inner
    assert result["first"]["values"] is not inner["values"]
    result["first"]["values"].append(3)
    assert result["second"]["values"] == [1, 2, 3]
    assert inner == {"values": [1, 2]}


@pytest.mark.parametrize(
    ("annotation", "in_object"),
    [
        pytest.param(Any, False, id="any"),
        pytest.param(list, False, id="bare-list"),
        pytest.param(list[Any], False, id="list-any"),
        pytest.param(tuple, False, id="bare-tuple"),
        pytest.param(dict[str, Any], True, id="dict-str-any"),
        pytest.param(Mapping[str, Any], True, id="mapping-str-any"),
        pytest.param(MappingProxyType[str, Any], True, id="mappingproxy-str-any"),
    ],
)
def test_shared_reference_inside_one_value_stays_shared(annotation, in_object):
    # Each item of a parameterized container is copied on its own: sharing is kept within one item.
    inner = [1]
    pair = [inner, inner]
    source = {"pair": pair} if in_object else [pair]

    result = deserialize_json(source, annotation)

    copied_pair = result["pair"] if in_object else result[0]
    assert copied_pair[0] is copied_pair[1]
    assert copied_pair[0] is not inner
    assert container_ids(result).isdisjoint(container_ids(source))


DEPTH = 50_000


def nested_dictionaries(depth):
    root = {}
    node = root
    for _ in range(depth):
        child = {}
        node["child"] = child
        node = child
    node["bottom"] = True
    return root


def nested_lists(depth):
    root = []
    node = root
    for _ in range(depth):
        child = []
        node.append(child)
        node = child
    node.append("bottom")
    return root


def child_of(node):
    if isinstance(node, (dict, MappingProxyType)):
        return node.get("child")
    if isinstance(node, (list, tuple)) and node and isinstance(node[0], (list, tuple)):
        return node[0]
    return None


def compare_chains(result, source):
    """Walk two nested chains side by side, iteratively (`==` on them would recurse).

    Returns (number of levels in the result, number of levels in the source, number of result levels that are
    the source's own object, number of result levels whose inner node is not a plain JSON container).
    """
    result_levels = source_levels = aliased = not_plain = 0
    result_node, source_node = result, source
    while result_node is not None or source_node is not None:
        if result_node is not None:
            result_levels += 1
            if result_levels > 1 and type(result_node) not in (dict, list):
                not_plain += 1
        if source_node is not None:
            source_levels += 1
        if result_node is not None and result_node is source_node:
            aliased += 1
        result_node = child_of(result_node) if result_node is not None else None
        source_node = child_of(source_node) if source_node is not None else None
    return result_levels, source_levels, aliased, not_plain


def frame_depth():
    depth = 0
    frame = sys._getframe()
    while frame is not None:
        depth += 1
        frame = frame.f_back
    return depth


@contextmanager
def recursion_limit_near_current_depth(headroom=100):
    original = sys.getrecursionlimit()
    sys.setrecursionlimit(frame_depth() + headroom)
    try:
        yield
    finally:
        sys.setrecursionlimit(original)


def test_lowered_recursion_limit_rejects_shallow_recursion():
    # Self-check of the helper: under it, even a few hundred nested calls fail, so the deep tests prove iteration.
    def recurse(levels):
        return 0 if levels == 0 else 1 + recurse(levels - 1)

    original = sys.getrecursionlimit()
    with recursion_limit_near_current_depth():
        with pytest.raises(RecursionError):
            recurse(300)

    assert sys.getrecursionlimit() == original


DEEP_SHAPES = [
    pytest.param(nested_dictionaries, Any, id="dictionaries-any"),
    pytest.param(nested_dictionaries, dict, id="dictionaries-bare-dict"),
    pytest.param(nested_dictionaries, dict[str, Any], id="dictionaries-dict-str-any"),
    pytest.param(nested_dictionaries, typing.Dict[str, Any], id="dictionaries-typing-dict-str-any"),
    pytest.param(nested_dictionaries, Mapping, id="dictionaries-bare-mapping"),
    pytest.param(nested_dictionaries, Mapping[str, Any], id="dictionaries-mapping-str-any"),
    pytest.param(nested_dictionaries, MappingProxyType, id="dictionaries-bare-mappingproxy"),
    pytest.param(nested_dictionaries, MappingProxyType[str, Any], id="dictionaries-mappingproxy-str-any"),
    pytest.param(nested_dictionaries, Optional[Any], id="dictionaries-optional-any"),
    pytest.param(nested_dictionaries, dict[str, Any] | None, id="dictionaries-dict-str-any-or-none"),
    pytest.param(nested_lists, Any, id="lists-any"),
    pytest.param(nested_lists, list, id="lists-bare-list"),
    pytest.param(nested_lists, list[Any], id="lists-list-any"),
    pytest.param(nested_lists, tuple, id="lists-bare-tuple"),
    pytest.param(nested_lists, tuple[Any, ...], id="lists-tuple-any-ellipsis"),
    pytest.param(nested_lists, Optional[Any], id="lists-optional-any"),
]


@pytest.mark.parametrize(("make_source", "annotation"), DEEP_SHAPES)
def test_very_deep_input_is_copied_without_recursion_error(make_source, annotation):
    source = make_source(DEPTH)

    result = deserialize_json(source, annotation)

    result_levels, source_levels, aliased, not_plain = compare_chains(result, source)
    assert (result_levels, source_levels) == (DEPTH + 1, DEPTH + 1)
    assert aliased == 0
    assert not_plain == 0
    disjoint = container_ids(result).isdisjoint(container_ids(source))
    assert disjoint


@pytest.mark.parametrize(("make_source", "annotation"), DEEP_SHAPES)
def test_very_deep_input_is_copied_under_a_low_recursion_limit(make_source, annotation):
    source = make_source(DEPTH)
    # warm up lazy imports and caches outside of the lowered limit
    deserialize_json(make_source(2), annotation)

    with recursion_limit_near_current_depth():
        result = deserialize_json(source, annotation)

    result_levels, source_levels, aliased, not_plain = compare_chains(result, source)
    assert (result_levels, source_levels) == (DEPTH + 1, DEPTH + 1)
    assert aliased == 0
    assert not_plain == 0


def test_very_deep_input_under_a_dataclass_field_is_copied():
    @dataclass
    class Holder:
        value: Any
        mapping: dict[str, Any]

    source = {"value": nested_lists(DEPTH), "mapping": {"root": nested_dictionaries(DEPTH)}}
    deserialize_json({"value": [], "mapping": {}}, Holder)

    with recursion_limit_near_current_depth():
        result = deserialize_json(source, Holder)

    assert compare_chains(result.value, source["value"]) == (DEPTH + 1, DEPTH + 1, 0, 0)
    assert compare_chains(result.mapping["root"], source["mapping"]["root"]) == (DEPTH + 1, DEPTH + 1, 0, 0)
    disjoint = container_ids(result).isdisjoint(container_ids(source))
    assert disjoint


# Serialization: the intermediate tree is built by `json.loads`, so it never shares a container with the object.


@dataclass
class Settings:
    options: dict[str, Any]
    values: list[Any]
    untyped: dict


@dataclass
class Document:
    settings: Settings
    tags: list[str]
    sections: dict[str, list[dict[str, Any]]]
    history: list[Settings]
    shared: list[Any]


def make_document():
    shared = [{"marker": [1]}]
    return Document(
        settings=Settings(options={"a": [1, {"b": [2]}]}, values=[[3], {"c": {}}], untyped={"d": [4]}),
        tags=["x", "y"],
        sections={"intro": [{"lines": ["hello"], "meta": {"order": [1]}}]},
        history=[Settings(options={"e": {"f": []}}, values=[], untyped={})],
        shared=[shared, shared],
    )


EXPECTED_DOCUMENT_TREE = {
    "settings": {"options": {"a": [1, {"b": [2]}]}, "values": [[3], {"c": {}}], "untyped": {"d": [4]}},
    "tags": ["x", "y"],
    "sections": {"intro": [{"lines": ["hello"], "meta": {"order": [1]}}]},
    "history": [{"options": {"e": {"f": []}}, "values": [], "untyped": {}}],
    "shared": [[{"marker": [1]}], [{"marker": [1]}]],
}


@pytest.mark.parametrize(
    "serializer",
    [
        pytest.param(serialize_json_as_dict, id="as-dict"),
        pytest.param(serialize_json_as_obj, id="as-obj"),
    ],
)
def test_serialized_tree_shares_no_container_with_object(serializer):
    document = make_document()

    tree = serializer(document)

    assert tree == EXPECTED_DOCUMENT_TREE
    assert container_ids(tree).isdisjoint(container_ids(document))


@pytest.mark.parametrize(
    "serializer",
    [
        pytest.param(serialize_json_as_dict, id="as-dict"),
        pytest.param(serialize_json_as_obj, id="as-obj"),
    ],
)
def test_mutating_object_after_serializing_leaves_tree_unchanged(serializer):
    document = make_document()
    tree = serializer(document)

    mutate_deeply(document)

    assert document != make_document()
    assert tree == EXPECTED_DOCUMENT_TREE


@pytest.mark.parametrize(
    "serializer",
    [
        pytest.param(serialize_json_as_dict, id="as-dict"),
        pytest.param(serialize_json_as_obj, id="as-obj"),
    ],
)
def test_mutating_serialized_tree_leaves_object_unchanged(serializer):
    document = make_document()
    tree = serializer(document)

    mutate_deeply(tree)

    assert tree != EXPECTED_DOCUMENT_TREE
    assert document == make_document()


def test_round_trip_through_intermediate_tree_is_detached_from_both_ends():
    document = make_document()
    tree = serialize_json_as_obj(document)

    restored = deserialize_json(tree, Document)

    assert restored == document
    assert container_ids(restored).isdisjoint(container_ids(tree))
    assert container_ids(restored).isdisjoint(container_ids(document))
    mutate_deeply(tree)
    assert restored == make_document()


def test_numpy_array_field_is_detached_from_input_list():
    np = pytest.importorskip("numpy")

    @dataclass
    class Measurement:
        samples: np.ndarray
        extra: dict[str, Any]

    source = {"samples": [1, 2, 3], "extra": {"unit": ["cm"]}}

    result = deserialize_json(source, Measurement)
    source["samples"].append(4)
    source["extra"]["unit"].append("mm")

    assert np.array_equal(result.samples, np.array([1, 2, 3]))
    assert result.extra == {"unit": ["cm"]}
    assert container_ids(result).isdisjoint(container_ids(source))


def test_numpy_object_array_does_not_hold_input_containers():
    np = pytest.importorskip("numpy")
    source = json.loads('[{"a": [1]}, {"b": [2]}]')

    result = deserialize_json(source, np.ndarray)

    assert result.dtype == object
    assert all(item is not original for item, original in zip(result, source))
    source[0]["a"].append(99)
    assert result[0] == {"a": [1]}


def test_numpy_object_scalar_does_not_return_the_input():
    np = pytest.importorskip("numpy")
    source = {"a": [1]}

    result = deserialize_json(source, np.object_)

    assert result == {"a": [1]}
    assert container_ids(result).isdisjoint(container_ids(source))

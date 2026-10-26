"""Sequence annotations (tuple, list, set, frozenset) accept only a JSON array.

A str, bytes, object or scalar is rejected with a data-free TypeError instead of being iterated into
characters, byte values or keys. The string annotations of this module's dataclasses (resolved via
`typing.get_type_hints`) are part of what is tested.
"""

from __future__ import annotations

import json
import re
import typing
from collections.abc import Mapping
from dataclasses import dataclass, fields, make_dataclass
from typing import Optional

import pytest

from jserpy import deserialize_json, serialize_json

SENTINEL = "SEQ_PAYLOAD_MUST_NOT_LEAK"


class Tags(list):
    pass


class Pair(tuple):
    pass


class TagSet(set):
    pass


@dataclass(frozen=True)
class Element:
    n: int
    label: str


@dataclass
class Model:
    ids: tuple[str, ...]
    name: str


@dataclass
class Bag:
    ids: tuple[str, ...]
    names: list[str]
    tags: set[str]
    frozen: frozenset[str]
    pair: tuple[int, int]
    elements: tuple[Element, ...]
    maybe_ids: Optional[tuple[str, ...]] = None


@dataclass
class Sparse:
    maybe_ids: Optional[tuple[str, ...]]
    maybe_names: list[int] | None
    bare: list


RealAnnotations = make_dataclass(
    "RealAnnotations",
    [
        ("ids", tuple[str, ...]),
        ("names", list[str]),
        ("tags", set[str]),
        ("frozen", frozenset[str]),
        ("pair", tuple[int, int]),
        ("maybe_ids", Optional[tuple[str, ...]], None),
    ],
)


def _annotation_id(annotation: object) -> str:
    if isinstance(annotation, type) and not typing.get_args(annotation):
        return annotation.__name__
    return repr(annotation)


def _round_trip(value, annotation):
    return deserialize_json(json.loads(serialize_json(value)), annotation)


def _chained_messages(error: BaseException) -> str:
    messages = []
    current: BaseException | None = error
    while current is not None:
        messages.append(str(current))
        current = current.__cause__ or current.__context__
    return "\n".join(messages)


SEQUENCE_ANNOTATIONS = [
    tuple,
    list,
    set,
    frozenset,
    tuple[str, ...],
    tuple[int, int],
    tuple[str],
    list[str],
    set[str],
    frozenset[str],
    typing.List[str],
    typing.Tuple[str, ...],
    typing.Set[str],
    typing.FrozenSet[str],
    typing.List,
    typing.Tuple,
    typing.Set,
    typing.FrozenSet,
    Tags,
    Pair,
    TagSet,
]

NON_ARRAYS = [
    pytest.param("abc", id="str"),
    pytest.param("", id="empty-str"),
    pytest.param("AP9KU2VyUHk=", id="base64-str"),
    pytest.param(b"\x00\xffJSerPy", id="bytes"),
    pytest.param({"a": 1}, id="dict"),
    pytest.param({}, id="empty-dict"),
    pytest.param(7, id="int"),
    pytest.param(1.5, id="float"),
    pytest.param(True, id="bool"),
]


# --- non-arrays are rejected ---------------------------------------------------------------------------------


@pytest.mark.parametrize("data", NON_ARRAYS)
@pytest.mark.parametrize("annotation", SEQUENCE_ANNOTATIONS, ids=_annotation_id)
def test_sequence_annotation_rejects_non_array(annotation, data):
    with pytest.raises(TypeError):
        deserialize_json(data, annotation)


@pytest.mark.parametrize("data", NON_ARRAYS)
@pytest.mark.parametrize("annotation", [tuple, list, set, frozenset], ids=_annotation_id)
def test_rejection_names_target_and_offending_types(annotation, data):
    with pytest.raises(TypeError) as caught:
        deserialize_json(data, annotation)

    message = str(caught.value)
    assert annotation.__name__ in message
    assert re.search(rf"\b{type(data).__name__}\b", message)


@pytest.mark.parametrize(
    "data",
    [
        pytest.param(SENTINEL, id="str"),
        pytest.param(SENTINEL.encode(), id="bytes"),
        pytest.param({SENTINEL: SENTINEL}, id="dict-key-and-value"),
    ],
)
@pytest.mark.parametrize(
    "annotation",
    [
        tuple,
        list,
        set,
        frozenset,
        tuple[str, ...],
        list[str],
        set[str],
        frozenset[str],
        typing.List,
        tuple[str, ...] | None,
        Optional[list[str]],
    ],
    ids=_annotation_id,
)
def test_rejection_message_does_not_leak_payload(annotation, data):
    with pytest.raises(TypeError) as caught:
        deserialize_json(data, annotation)

    assert SENTINEL not in _chained_messages(caught.value)


def test_dataclass_field_rejection_does_not_leak_payload():
    with pytest.raises(TypeError) as caught:
        deserialize_json({"ids": SENTINEL, "name": SENTINEL}, Model)

    assert SENTINEL not in _chained_messages(caught.value)


@pytest.mark.parametrize(
    ("data", "annotation"),
    [
        pytest.param(["abc"], list[tuple[str, ...]], id="list-of-tuples"),
        pytest.param([["a"], "bc"], tuple[list[str], ...], id="tuple-of-lists"),
        pytest.param({"k": "abc"}, Mapping[str, tuple[str, ...]], id="Mapping-of-tuples"),
        pytest.param({"k": "abc"}, dict[str, list[str]], id="dict-of-lists"),
    ],
)
def test_nested_sequence_annotation_rejects_string(data, annotation):
    with pytest.raises(TypeError):
        deserialize_json(data, annotation)


# --- fixed-length tuples -------------------------------------------------------------------------------------


@pytest.mark.parametrize("data", [[], [1], [1, 2, 3]], ids=["empty", "short", "long"])
def test_fixed_length_tuple_rejects_wrong_length(data):
    with pytest.raises(TypeError) as caught:
        deserialize_json(data, tuple[int, int])

    assert re.search(rf"\b{len(data)}\b", str(caught.value))


def test_fixed_length_tuple_length_error_does_not_leak_items():
    with pytest.raises(TypeError) as caught:
        deserialize_json([SENTINEL] * 3, tuple[str, str])

    assert SENTINEL not in _chained_messages(caught.value)


def test_optional_fixed_length_tuple_rejects_wrong_length_instead_of_returning_none():
    with pytest.raises(TypeError):
        deserialize_json([1, 2, 3], tuple[int, int] | None)


@pytest.mark.parametrize(("data", "expected"), [([1, 2], (1, 2)), ([1, 2, 3], (1, 2, 3))], ids=["pair", "triple"])
def test_fixed_length_tuple_union_selects_member_by_arity(data, expected):
    assert deserialize_json(data, tuple[int, int] | tuple[int, int, int]) == expected


def test_single_argument_tuple_keeps_homogeneous_behavior():
    assert deserialize_json([1, 2, 3], tuple[int]) == (1, 2, 3)


# --- valid arrays still round-trip ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "annotation"),
    [
        pytest.param(("a", "b"), tuple[str, ...], id="tuple[str, ...]"),
        pytest.param((1, "a"), tuple[int, str], id="tuple[int, str]"),
        pytest.param((1, "a"), tuple, id="tuple"),
        pytest.param(["a", "b"], list[str], id="list[str]"),
        pytest.param([1, "a"], list, id="list"),
        pytest.param({1, 2}, set[int], id="set[int]"),
        pytest.param({1, "a"}, set, id="set"),
        pytest.param(frozenset({"a", "b"}), frozenset[str], id="frozenset[str]"),
        pytest.param(("a", "b"), typing.Tuple[str, ...], id="typing.Tuple[str, ...]"),
        pytest.param(["a", "b"], typing.List[str], id="typing.List[str]"),
        pytest.param({"a", "b"}, typing.Set[str], id="typing.Set[str]"),
        pytest.param(frozenset({"a"}), typing.FrozenSet[str], id="typing.FrozenSet[str]"),
        pytest.param([("a", 1), ("b", 2)], list[tuple[str, int]], id="list-of-tuples"),
        pytest.param((Element(1, "x"), Element(2, "y")), tuple[Element, ...], id="tuple-of-dataclasses"),
        pytest.param([Element(1, "x")], list[Element], id="list-of-dataclasses"),
        pytest.param(frozenset({Element(1, "x"), Element(2, "y")}), frozenset[Element], id="frozenset-of-dataclasses"),
        pytest.param((), tuple[str, ...], id="empty-tuple[str, ...]"),
        pytest.param((), tuple, id="empty-tuple"),
        pytest.param([], list[int], id="empty-list[int]"),
        pytest.param([], list, id="empty-list"),
        pytest.param(set(), set[int], id="empty-set"),
        pytest.param(frozenset(), frozenset[str], id="empty-frozenset"),
    ],
)
def test_valid_arrays_round_trip(value, annotation):
    restored = _round_trip(value, annotation)

    assert restored == value
    assert type(restored) is type(value)


def test_nested_dataclass_elements_are_restored():
    restored = _round_trip([(Element(1, "x"),)], list[tuple[Element, ...]])

    assert restored == [(Element(1, "x"),)]
    assert type(restored[0]) is tuple
    assert type(restored[0][0]) is Element


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        pytest.param(tuple[str, ...], ("a", "b"), id="tuple[str, ...]"),
        pytest.param(tuple[str, str], ("a", "b"), id="tuple[str, str]"),
        pytest.param(tuple, ("a", "b"), id="tuple"),
        pytest.param(list[str], ["a", "b"], id="list[str]"),
        pytest.param(set[str], {"a", "b"}, id="set[str]"),
        pytest.param(frozenset[str], frozenset({"a", "b"}), id="frozenset[str]"),
    ],
)
def test_python_tuple_is_accepted_as_an_array(annotation, expected):
    restored = deserialize_json(("a", "b"), annotation)

    assert restored == expected
    assert type(restored) is type(expected)


@pytest.mark.parametrize("annotation", [list, typing.List, Tags], ids=_annotation_id)
def test_unparameterized_list_converts_python_tuple_to_list(annotation):
    restored = deserialize_json(("a", "b"), annotation)

    assert isinstance(restored, list)
    assert list(restored) == ["a", "b"]


# --- unions --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "annotation",
    [
        tuple[str, ...] | None,
        Optional[tuple[str, ...]],
        list[int] | None,
        Optional[list[int]],
        set[str] | None,
        Optional[frozenset[str]],
        Optional[list],
        Optional[typing.List[str]],
    ],
    ids=_annotation_id,
)
def test_optional_sequence_rejects_string_and_accepts_null(annotation):
    with pytest.raises(TypeError):
        deserialize_json("abc", annotation)

    assert deserialize_json(None, annotation) is None


def test_union_rejection_names_the_offending_type():
    with pytest.raises(TypeError, match=r"\bgot str\b"):
        deserialize_json("abc", tuple[str, ...] | None)


def test_union_falls_through_to_string_member():
    annotation = list[int] | str

    assert deserialize_json("abc", annotation) == "abc"
    assert deserialize_json([1, 2], annotation) == [1, 2]


def test_union_selects_mapping_member_for_an_object():
    annotation = list[str] | Mapping[str, int]

    assert deserialize_json({"a": 1}, annotation) == {"a": 1}
    assert deserialize_json(["a"], annotation) == ["a"]


# --- dataclass fields ----------------------------------------------------------------------------------------


def _bag() -> Bag:
    return Bag(
        ids=("a", "b"),
        names=["c"],
        tags={"d", "e"},
        frozen=frozenset({"f"}),
        pair=(1, 2),
        elements=(Element(1, "x"),),
        maybe_ids=("g",),
    )


def test_handoff_repro_string_for_tuple_field_is_rejected():
    with pytest.raises(TypeError):
        deserialize_json({"ids": "abc"}, Model)


def test_string_annotated_dataclass_round_trips():
    assert all(isinstance(field.type, str) for field in fields(Bag))
    original = _bag()

    restored = _round_trip(original, Bag)

    assert restored == original
    assert type(restored.ids) is tuple
    assert type(restored.frozen) is frozenset
    assert type(restored.elements[0]) is Element


@pytest.mark.parametrize("field_name", [field.name for field in fields(Bag)])
def test_string_annotated_dataclass_field_rejects_string(field_name):
    data = json.loads(serialize_json(_bag()))
    data[field_name] = "abc"

    with pytest.raises(TypeError):
        deserialize_json(data, Bag)


def test_real_annotated_dataclass_round_trips():
    assert not any(isinstance(field.type, str) for field in fields(RealAnnotations))
    original = RealAnnotations(
        ids=("a",), names=["b"], tags={"c"}, frozen=frozenset({"d"}), pair=(1, 2), maybe_ids=("e",)
    )

    assert _round_trip(original, RealAnnotations) == original


@pytest.mark.parametrize("field_name", [field.name for field in fields(RealAnnotations)])
def test_real_annotated_dataclass_field_rejects_string(field_name):
    data = {"ids": ["a"], "names": ["b"], "tags": ["c"], "frozen": ["d"], "pair": [1, 2], "maybe_ids": None}
    data[field_name] = "abc"

    with pytest.raises(TypeError):
        deserialize_json(data, RealAnnotations)


def test_missing_optional_and_bare_list_fields_stay_none():
    assert deserialize_json({}, Sparse) == Sparse(maybe_ids=None, maybe_names=None, bare=None)


def test_missing_required_parameterized_sequence_field_still_raises_type_error():
    with pytest.raises(TypeError):
        deserialize_json({"name": "n"}, Model)


# --- null handling (unchanged) -------------------------------------------------------------------------------


@pytest.mark.parametrize("annotation", [list, typing.List, Tags], ids=_annotation_id)
def test_null_for_unparameterized_list_passes_through(annotation):
    assert deserialize_json(None, annotation) is None


@pytest.mark.parametrize(
    "annotation",
    [
        tuple,
        set,
        frozenset,
        tuple[str, ...],
        tuple[int, int],
        list[str],
        set[str],
        frozenset[str],
        typing.List[str],
        typing.Tuple,
        typing.Set,
    ],
    ids=_annotation_id,
)
def test_null_for_other_sequence_annotations_still_raises_type_error(annotation):
    with pytest.raises(TypeError):
        deserialize_json(None, annotation)


# --- bytes as a target type is unaffected --------------------------------------------------------------------


def test_bytes_target_still_decodes_base64():
    assert deserialize_json("AP9KU2VyUHk=", bytes) == b"\x00\xffJSerPy"


def test_bytes_elements_round_trip_inside_a_sequence():
    original = (b"\x00\xff", b"JSerPy")

    assert _round_trip(original, tuple[bytes, ...]) == original


def test_sequence_of_bytes_rejects_a_single_base64_string():
    with pytest.raises(TypeError):
        deserialize_json("AP9KU2VyUHk=", list[bytes])

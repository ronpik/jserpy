"""Regression tests for deserialization paths that must keep working next to the new serialization features."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, make_dataclass
from typing import Any, Optional

import pytest

from jserpy import deserialize_json, serialize_json


@dataclass
class OptionalMapping:
    m: Optional[Mapping[str, int]] = None


def test_optional_mapping_field_round_trips_none():
    original = OptionalMapping()

    restored = deserialize_json(json.loads(serialize_json(original)), OptionalMapping)

    assert restored == OptionalMapping(m=None)


def test_optional_mapping_field_missing_from_data_is_none():
    assert deserialize_json({}, OptionalMapping) == OptionalMapping(m=None)


def test_optional_mapping_field_round_trips_value():
    original = OptionalMapping(m={"a": 1, "b": 2})

    restored = deserialize_json(json.loads(serialize_json(original)), OptionalMapping)

    assert restored == original


@pytest.mark.parametrize(
    "annotation",
    [Optional[Mapping[str, int]], Mapping[str, int] | None],
    ids=["Optional", "PEP604"],
)
def test_optional_mapping_annotation_accepts_null_and_object(annotation):
    assert deserialize_json(None, annotation) is None
    assert deserialize_json({"a": 1}, annotation) == {"a": 1}


def test_mapping_union_selects_the_member_matching_the_data():
    annotation = Mapping[str, int] | list[int]

    assert deserialize_json({"a": 1}, annotation) == {"a": 1}
    assert deserialize_json([1, 2], annotation) == [1, 2]


def test_mapping_annotation_rejects_non_object_data():
    with pytest.raises(TypeError):
        deserialize_json([1, 2], Mapping[str, int])


def _ndarray_annotation(kind):
    np = pytest.importorskip("numpy")
    if not hasattr(np.ndarray, "__class_getitem__"):
        pytest.skip("parameterized np.ndarray needs numpy >= 1.22")
    if kind == "ndarray":
        return np, np.ndarray[Any, np.dtype[np.float64]]

    npt = pytest.importorskip("numpy.typing")
    if type(npt.NDArray).__name__ == "TypeAliasType":
        pytest.skip("numpy >= 2.5 defines NDArray as a TypeAliasType, which jserpy has never supported")
    return np, npt.NDArray[np.float64]


ndarray_kinds = pytest.mark.parametrize("kind", ["ndarray", "NDArray"])


@ndarray_kinds
def test_parameterized_ndarray_annotation_deserializes(kind):
    np, annotation = _ndarray_annotation(kind)

    restored = deserialize_json([1.0, 2.0], annotation)

    assert isinstance(restored, np.ndarray)
    assert restored.tolist() == [1.0, 2.0]


@ndarray_kinds
def test_parameterized_ndarray_dataclass_field_round_trips(kind):
    np, annotation = _ndarray_annotation(kind)
    sample_cls = make_dataclass("Sample", [("values", annotation), ("maybe", Optional[annotation], None)])
    original = sample_cls(values=np.array([1.5, 2.5]), maybe=np.array([3.5]))

    restored = deserialize_json(json.loads(serialize_json(original)), sample_cls)

    assert isinstance(restored.values, np.ndarray)
    assert restored.values.tolist() == [1.5, 2.5]
    assert isinstance(restored.maybe, np.ndarray)
    assert restored.maybe.tolist() == [3.5]


@ndarray_kinds
def test_optional_parameterized_ndarray_annotation_deserializes_array(kind):
    np, annotation = _ndarray_annotation(kind)

    restored = deserialize_json([1.0], Optional[annotation])

    assert isinstance(restored, np.ndarray)
    assert restored.tolist() == [1.0]

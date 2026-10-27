"""Helpers shared by the ``test_objects_*`` modules: round trips and a type-strict comparison.

``==`` is too forgiving for serialization tests: ``1 == 1.0 == True``, ``(1, 2) != [1, 2]`` only by luck of the
container type, a ``frozenset`` equals nothing but a ``frozenset`` yet a ``set`` equals neither, and ``-0.0 == 0.0``.
``assert_strictly_equal`` compares the exact type at every level, so a restored value is the value that went in, not
just something that compares equal to it.
"""

import copy
import dataclasses
import json
import math
from decimal import Decimal
from types import MappingProxyType
from typing import Any

from jserpy import deserialize_json, serialize_json, serialize_json_as_obj

try:
    import numpy as np
except ModuleNotFoundError:  # numpy is an optional dependency
    np = None


def to_json_value(value: Any, **options: Any) -> Any:
    """What a consumer of the JSON text sees: ``json.loads(serialize_json(value))``."""
    return json.loads(serialize_json(value, **options))


def round_trip(value: Any, cls: Any, **options: Any) -> Any:
    """Serialize ``value`` to text, parse it back, and deserialize it as ``cls``."""
    return deserialize_json(to_json_value(value, **options), cls)


def assert_round_trip(value: Any, cls: Any = None, **options: Any) -> Any:
    """Assert that ``value`` survives text -> JSON value -> ``cls`` exactly, and return the restored value.

    Besides the strict comparison (``cls`` defaults to ``type(value)``), it pins invariants that hold for any value:
    the text is valid JSON, ``serialize_json_as_obj`` agrees with it, serializing is deterministic, deserializing
    leaves the parsed JSON value untouched, and the restored value serializes to the very same text.
    """
    if cls is None:
        cls = type(value)

    text = serialize_json(value, **options)
    assert serialize_json(value, **options) == text, "serialization is not deterministic"

    parsed = json.loads(text)
    snapshot = copy.deepcopy(parsed)
    restored = deserialize_json(parsed, cls)
    assert json.dumps(parsed) == json.dumps(snapshot), "deserialization mutated the JSON value it was given"

    assert_strictly_equal(restored, value)
    assert serialize_json(restored, **options) == text, "the restored value serializes differently"
    if not options:
        assert json.dumps(serialize_json_as_obj(value)) == json.dumps(parsed)
    return restored


def assert_strictly_equal(actual: Any, expected: Any, path: str = "value") -> None:
    """Assert equality of values *and* exact types, recursively; ``path`` locates the first difference."""
    assert type(actual) is type(expected), f"{path}: type {type(actual)!r} != {type(expected)!r}"

    if isinstance(expected, float):
        if math.isnan(expected):
            assert math.isnan(actual), f"{path}: {actual!r} != nan"
        else:
            assert actual == expected and math.copysign(1.0, actual) == math.copysign(1.0, expected), (
                f"{path}: {actual!r} != {expected!r}"
            )
    elif isinstance(expected, Decimal):
        # Decimal("1.20") == Decimal("1.2"), but the exponent is part of what a round trip must keep
        assert actual.as_tuple() == expected.as_tuple(), f"{path}: {actual!r} != {expected!r}"
    elif np is not None and isinstance(expected, np.floating):
        # float32 / float16 scalars are not Python floats
        if np.isnan(expected):
            assert np.isnan(actual), f"{path}: {actual!r} != nan"
        else:
            assert actual == expected and np.signbit(actual) == np.signbit(expected), (
                f"{path}: {actual!r} != {expected!r}"
            )
    elif np is not None and isinstance(expected, np.ndarray):
        assert actual.dtype == expected.dtype, f"{path}: dtype {actual.dtype} != {expected.dtype}"
        assert actual.shape == expected.shape, f"{path}: shape {actual.shape} != {expected.shape}"
        if expected.dtype == object:
            for index in np.ndindex(expected.shape):
                assert_strictly_equal(actual[index], expected[index], f"{path}{list(index)}")
        else:
            assert np.array_equal(actual, expected, equal_nan=expected.dtype.kind in "fc"), (
                f"{path}: {actual!r} != {expected!r}"
            )
    elif dataclasses.is_dataclass(expected):
        for field in dataclasses.fields(expected):
            assert_strictly_equal(
                getattr(actual, field.name), getattr(expected, field.name), f"{path}.{field.name}"
            )
    elif isinstance(expected, (dict, MappingProxyType)):
        _assert_mappings_equal(actual, expected, path)
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), f"{path}: length {len(actual)} != {len(expected)}"
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            assert_strictly_equal(actual_item, expected_item, f"{path}[{index}]")
    elif isinstance(expected, (set, frozenset)):
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
        # members that are equal may still differ in type, also deeper inside (e.g. (1, 2) vs (1.0, 2))
        actual_members = {member: member for member in actual}
        for member in expected:
            assert_strictly_equal(actual_members[member], member, f"{path}{{{member!r}}}")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


def _assert_mappings_equal(actual: Any, expected: Any, path: str) -> None:
    assert len(actual) == len(expected), f"{path}: {len(actual)} keys != {len(expected)} keys"
    actual_keys = {key: key for key in actual}
    for key, expected_value in expected.items():
        assert key in actual_keys, f"{path}: missing key {key!r}"
        assert_strictly_equal(actual_keys[key], key, f"{path} key {key!r}")
        assert_strictly_equal(actual[key], expected_value, f"{path}[{key!r}]")

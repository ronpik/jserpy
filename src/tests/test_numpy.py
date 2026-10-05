import json

import pytest

from jserpy import deserialize_json, serialize_json

np = pytest.importorskip("numpy")


def test_numpy_array_round_trip():
    value = np.array([1, 2, 3, 4, 5])

    restored = deserialize_json(json.loads(serialize_json(value)), np.ndarray)

    assert np.array_equal(restored, value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(np.int64(7), id="integer"),
        pytest.param(np.float64(1.25), id="float"),
        pytest.param(np.bool_(True), id="boolean"),
    ],
)
def test_numpy_scalar_round_trip_preserves_scalar_type(value):
    restored = deserialize_json(json.loads(serialize_json(value)), type(value))

    assert restored == value
    assert type(restored) is type(value)


def test_numpy_values_work_when_nested():
    value = {
        "array": np.array([1, 2]),
        "integer": np.int64(7),
        "float": np.float64(1.25),
        "boolean": np.bool_(True),
    }

    assert json.loads(serialize_json(value)) == {
        "array": [1, 2],
        "integer": 7,
        "float": 1.25,
        "boolean": True,
    }

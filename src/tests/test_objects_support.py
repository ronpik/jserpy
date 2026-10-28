"""The strict comparison of ``objects_support`` must reject what ``==`` lets through."""

from dataclasses import dataclass
from decimal import Decimal

import pytest

from objects_support import assert_round_trip, assert_strictly_equal, round_trip


@dataclass
class Pair:
    left: object
    right: object


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        pytest.param(1, 1.0, id="int-vs-float"),
        pytest.param(1, True, id="int-vs-bool"),
        pytest.param([1, 2], (1, 2), id="list-vs-tuple"),
        pytest.param({1, 2}, frozenset({1, 2}), id="set-vs-frozenset"),
        pytest.param(0.0, -0.0, id="signed-zero"),
        pytest.param(Decimal("1.2"), Decimal("1.20"), id="decimal-exponent"),
        pytest.param([1, [2, (3,)]], [1, [2, [3]]], id="nested-tuple-vs-list"),
        pytest.param({"a": 1}, {"a": 1.0}, id="dict-value-type"),
        pytest.param({1: "a"}, {True: "a"}, id="dict-key-type"),
        pytest.param({1, 2}, {1.0, 2}, id="set-member-type"),
        pytest.param({(1, 2)}, {(1.0, 2)}, id="set-member-inner-type"),
        pytest.param({(1, 2): "a"}, {(1.0, 2): "a"}, id="dict-key-inner-type"),
        pytest.param(Pair(1, 2), Pair(1, 2.0), id="dataclass-field-type"),
        pytest.param([1, 2], [1, 2, 3], id="length"),
        pytest.param({"a": 1}, {"b": 1}, id="key"),
    ],
)
def test_strict_comparison_rejects_values_that_only_compare_equal(actual, expected):
    with pytest.raises(AssertionError):
        assert_strictly_equal(actual, expected)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(-0.0, id="negative-zero"),
        pytest.param(Decimal("1.20"), id="decimal"),
        pytest.param({"a": [1, (2, 3), {4, 5}]}, id="nested"),
        pytest.param(Pair(frozenset({1}), {(1, 2): "x"}), id="dataclass"),
    ],
)
def test_strict_comparison_accepts_identical_values(value):
    assert_strictly_equal(value, value)


def test_round_trip_helper_returns_the_restored_value():
    restored = round_trip({"a": (1, 2)}, dict[str, tuple[int, int]])

    assert restored == {"a": (1, 2)}
    assert type(restored["a"]) is tuple


def test_assert_round_trip_detects_a_lossy_round_trip():
    # a tuple deserialized as an unparameterized list comes back as a list: the round trip is not exact
    with pytest.raises(AssertionError):
        assert_round_trip((1, 2), list)


def test_strict_comparison_distinguishes_the_sign_of_zero_in_narrow_floats():
    np = pytest.importorskip("numpy")

    with pytest.raises(AssertionError):
        assert_strictly_equal(np.float32(0.0), np.float32(-0.0))
    assert_strictly_equal(np.float32("nan"), np.float32("nan"))

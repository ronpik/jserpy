from dataclasses import dataclass

from jserpy import deserialize_json


@dataclass
class Prices:
    value: float
    maybe: float | None
    many: tuple[float, ...]


def test_int_is_coerced_to_float_for_float_annotations():
    decoded = deserialize_json({"value": 480, "maybe": 3, "many": [1, 2.5]}, Prices)
    assert type(decoded.value) is float
    assert type(decoded.maybe) is float
    assert [type(v) for v in decoded.many] == [float, float]


def test_bool_is_not_coerced_and_none_stays_none():
    assert deserialize_json(True, float) is True
    assert deserialize_json({"value": 1, "maybe": None, "many": []}, Prices).maybe is None

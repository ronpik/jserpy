from enum import Enum

import pytest

from jserpy import serialize_json


class CollisionKey(Enum):
    ALPHA = "alpha"


class NullKey(Enum):
    EMPTY = None


class OddInt(int):
    def __str__(self):
        return "overridden"


class OddString(str):
    def __str__(self):
        return "overridden"


class SensitiveKey:
    def __hash__(self):
        return id(self)

    def __repr__(self):
        return "KEY_PAYLOAD_MUST_NOT_LEAK"


class SensitiveValue:
    def __repr__(self):
        return "VALUE_PAYLOAD_MUST_NOT_LEAK"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(
            {1: "LEFT_PAYLOAD", "1": "RIGHT_PAYLOAD"},
            id="integer-versus-string",
        ),
        pytest.param(
            {1.5: "LEFT_PAYLOAD", "1.5": "RIGHT_PAYLOAD"},
            id="float-versus-string",
        ),
        pytest.param(
            {True: "LEFT_PAYLOAD", "true": "RIGHT_PAYLOAD"},
            id="boolean-versus-string",
        ),
        pytest.param(
            {CollisionKey.ALPHA: "LEFT_PAYLOAD", "alpha": "RIGHT_PAYLOAD"},
            id="enum-versus-string",
        ),
        pytest.param(
            {
                ("alpha", 2): "LEFT_PAYLOAD",
                '["alpha", 2]': "RIGHT_PAYLOAD",
            },
            id="tuple-versus-string",
        ),
        pytest.param(
            {
                (1, ("alpha", 2)): "LEFT_PAYLOAD",
                '[1, ["alpha", 2]]': "RIGHT_PAYLOAD",
            },
            id="nested-tuple-versus-string",
        ),
    ],
)
def test_logical_key_collisions_fail_without_payload_values(value):
    with pytest.raises(ValueError, match="(?i)colli") as caught:
        serialize_json(value)

    message = str(caught.value)
    assert "LEFT_PAYLOAD" not in message
    assert "RIGHT_PAYLOAD" not in message


def test_unsupported_key_error_names_type_without_key_payload():
    with pytest.raises((TypeError, ValueError)) as caught:
        serialize_json({SensitiveKey(): "value"})

    message = str(caught.value)
    assert "SensitiveKey" in message
    assert "KEY_PAYLOAD_MUST_NOT_LEAK" not in message


def test_unsupported_value_error_names_type_without_value_payload():
    with pytest.raises(TypeError) as caught:
        serialize_json(SensitiveValue())

    message = str(caught.value)
    assert "SensitiveValue" in message
    assert "VALUE_PAYLOAD_MUST_NOT_LEAK" not in message


def test_none_valued_enum_key_preserves_legacy_null_member_name():
    assert serialize_json({NullKey.EMPTY: "value"}) == '{"null": "value"}'


def test_integer_subclass_key_ignores_overridden_string_conversion():
    assert serialize_json({OddInt(7): "value"}) == '{"7": "value"}'


def test_string_subclass_key_ignores_overridden_string_conversion():
    assert serialize_json({OddString("actual"): "value"}) == (
        '{"actual": "value"}'
    )

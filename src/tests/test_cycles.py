from dataclasses import dataclass
from typing import Any

import pytest

from jserpy import serialize_json, serialize_json_as_obj


@dataclass
class Node:
    child: Any = None


def make_cyclic_list():
    value = []
    value.append(value)
    return value


def make_cyclic_dictionary():
    value = {"CYCLE_PAYLOAD_MUST_NOT_LEAK": None}
    value["CYCLE_PAYLOAD_MUST_NOT_LEAK"] = value
    return value


def make_cyclic_dataclass():
    value = Node()
    value.child = value
    return value


@pytest.mark.parametrize(
    "make_value",
    [
        pytest.param(make_cyclic_list, id="list"),
        pytest.param(make_cyclic_dictionary, id="dictionary"),
        pytest.param(make_cyclic_dataclass, id="dataclass"),
    ],
)
@pytest.mark.parametrize(
    "serializer",
    [
        pytest.param(serialize_json, id="text"),
        pytest.param(serialize_json_as_obj, id="intermediate"),
    ],
)
def test_real_cycles_fail_predictably_without_payload(make_value, serializer):
    with pytest.raises(ValueError, match="(?i)cyc") as caught:
        serializer(make_value())

    assert "CYCLE_PAYLOAD_MUST_NOT_LEAK" not in str(caught.value)


def test_repeated_alias_is_not_mistaken_for_a_cycle_by_text_serializer():
    child = {"value": 1}

    assert serialize_json([child, child]) == '[{"value": 1}, {"value": 1}]'


def test_repeated_alias_is_not_mistaken_for_a_cycle_by_intermediate_serializer():
    child = {"value": 1}

    result = serialize_json_as_obj([child, child])

    assert result == [{"value": 1}, {"value": 1}]
    assert result[0] is not result[1]

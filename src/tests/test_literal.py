from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import pytest

from jserpy import deserialize_json, serialize_json_as_dict

Kind = Literal["x", "y"]


@dataclass(frozen=True)
class WithLiterals:
    kind: Kind
    maybe: Literal["a", "b"] | None
    mapping: Mapping[str, int] | None = None
    proxy: MappingProxyType[str, int] | None = None


def test_bare_literal():
    assert deserialize_json("x", Literal["x", "y"]) == "x"
    with pytest.raises(TypeError):
        deserialize_json("z", Literal["x", "y"])


def test_literal_fields_round_trip_through_alias():
    obj = WithLiterals("x", "a")
    assert deserialize_json(serialize_json_as_dict(obj), WithLiterals) == obj
    assert deserialize_json({"kind": "y", "maybe": None}, WithLiterals).maybe is None


@pytest.mark.parametrize(
    "data",
    [{"kind": "zzz", "maybe": None}, {"kind": "x", "maybe": "zzz"}],
)
def test_invalid_literal_rejected(data):
    with pytest.raises(TypeError):
        deserialize_json(data, WithLiterals)


def test_optional_mapping_accepts_null_and_decodes():
    decoded = deserialize_json(
        {"kind": "x", "maybe": None, "mapping": None, "proxy": {"k": 1}}, WithLiterals
    )
    assert decoded.mapping is None
    assert type(decoded.proxy) is MappingProxyType

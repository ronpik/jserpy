from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Literal

from jserpy import deserialize_json, serialize_json, serialize_json_as_dict


@dataclass(frozen=True, slots=True)
class Holder:
    later: Later
    laters: tuple[Later, ...]


@dataclass(frozen=True, slots=True)
class Later:
    n: int


@dataclass(frozen=True, slots=True)
class Row:
    code: int
    treater_id: str | None
    price: float
    is_available: bool


@dataclass(frozen=True, slots=True)
class Service:
    item_code: str
    duration_minutes: int | None
    duration_source: Literal["override", "default"] | None
    price: float | None
    own_price_row: Row | None


@dataclass(frozen=True, slots=True)
class UserServices:
    user_id: str
    user_name: str | None
    services: tuple[Service, ...]


@dataclass(frozen=True, slots=True)
class Item:
    code: str
    staff_user_ids: tuple[str, ...]
    raw_data: dict[str, Any] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Catalog:
    department_id: int
    items: Mapping[str, Item]
    by_user: Mapping[str, UserServices]
    staff_durations: Mapping[str, tuple[Row, ...]]
    fetched_at: datetime

    def __post_init__(self) -> None:
        if self.fetched_at.tzinfo is None:
            raise ValueError("fetched_at must be timezone-aware")
        object.__setattr__(self, "items", MappingProxyType(dict(self.items)))
        object.__setattr__(self, "by_user", MappingProxyType(dict(self.by_user)))
        object.__setattr__(
            self,
            "staff_durations",
            MappingProxyType({k: tuple(v) for k, v in self.staff_durations.items()}),
        )


def _catalog() -> Catalog:
    row = Row(code=9001, treater_id=None, price=987.6500000001, is_available=True)
    return Catalog(
        department_id=9,
        items={"A": Item("A", ("U1",), {"k": [1, {"n": "סינתטי"}]})},
        by_user={
            "U1": UserServices(
                "U1",
                None,
                (
                    Service("A", 0, "default", 0.0, row),
                    Service("B", None, None, None, None),
                ),
            )
        },
        staff_durations={"A": (row,)},
        fetched_at=datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc),
    )


def test_catalog_round_trip():
    catalog = _catalog()
    data = serialize_json_as_dict(catalog)
    assert isinstance(data["items"], dict)
    assert data["fetched_at"] == "2026-10-05T12:00:00+00:00"
    assert json.dumps(data, ensure_ascii=False)

    decoded = deserialize_json(data, Catalog)
    assert decoded == catalog
    assert type(decoded.items) is MappingProxyType
    assert type(decoded.staff_durations["A"]) is tuple
    assert type(decoded.by_user["U1"].services) is tuple
    assert decoded.fetched_at.tzinfo is not None
    assert decoded.staff_durations["A"][0].price == 987.6500000001
    assert decoded.items["A"].raw_data == catalog.items["A"].raw_data


def test_explicit_mappingproxy_annotation():
    data = {"a": 1}
    result = deserialize_json(data, MappingProxyType[str, int])
    assert type(result) is MappingProxyType
    assert dict(result) == data
    assert json.loads(serialize_json(result)) == data


def test_mapping_annotation_with_typed_values():
    result = deserialize_json({"a": [1, 2]}, Mapping[str, tuple[int, ...]])
    assert result == {"a": (1, 2)}


def test_forward_reference_string_annotations():
    holder = Holder(Later(1), (Later(2), Later(3)))
    decoded = deserialize_json(serialize_json_as_dict(holder), Holder)
    assert decoded == holder
    assert type(decoded.laters) is tuple

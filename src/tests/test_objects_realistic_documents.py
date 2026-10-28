"""Realistic composite documents: what users really serialize, nested several levels deep with mixed types.

Every document has (1) the expected JSON written out by hand, (2) a strict typed round trip, (3) agreement of
``serialize_json_as_obj`` with the parsed text, and (4) corrupted variants of its JSON that must be rejected with a
TypeError. Where the library is deliberately lenient (absent fields, unchecked primitives, ...) a ``*_limitation``
test documents the current behavior; it is not a promise. Strict xfails mark behavior a user would not expect.
The tests at the end of the module run the same contract over every document (DOCUMENTS): golden JSON, key order,
round trips under several output options, no change to the document or to the shared goldens, no aliasing of the input.

Paths: documents that hold ``Path`` values are serialized with ``PurePosixPath`` (platform independent goldens) and
round-tripped with ``Path`` (POSIX only), because ``Path`` annotations restore concrete platform paths.
"""

from __future__ import annotations

import base64
import copy
import json
import math
import os
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum, IntEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Literal, Optional, Union

import pytest

from jserpy import deserialize_json, serialize_json, serialize_json_as_obj
from jserpy.jsonable import Jsonable
from objects_support import assert_round_trip, assert_strictly_equal, to_json_value

try:
    import numpy as np
except ModuleNotFoundError:  # numpy is an optional dependency; its tests below skip without it
    np = None

UTC = timezone.utc
PLUS_TWO = timezone(timedelta(hours=2))
NAN = float("nan")
INF = float("inf")

posix_only = pytest.mark.skipif(os.name != "posix", reason="round-tripped Path values are POSIX paths")

REMOVED = object()  # marks a corruption that deletes a key instead of replacing its value


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------


def corrupted(document: Any, path: tuple, replacement: Any) -> Any:
    """A deep copy of the hand-written JSON ``document`` with the value at ``path`` replaced, or its key removed."""
    broken = copy.deepcopy(document)
    parent = broken
    for step in path[:-1]:
        parent = parent[step]
    if replacement is REMOVED:
        del parent[path[-1]]
    else:
        parent[path[-1]] = replacement
    return broken


def assert_rejected(document: Any, cls: Any, path: tuple, replacement: Any) -> None:
    """Corrupt one spot of ``document``; the intact document must load and the corrupted one must raise TypeError."""
    deserialize_json(copy.deepcopy(document), cls)  # the corruption, not the document, is what raises below
    broken = corrupted(document, path, replacement)
    with pytest.raises(TypeError):
        deserialize_json(broken, cls)


def assert_same_json(actual: Any, expected: Any) -> None:
    """Equality of JSON values that also tells 1 from 1.0 from true, and treats NaN as equal to NaN."""
    if json.dumps(actual, sort_keys=True) != json.dumps(expected, sort_keys=True):
        assert actual == expected
        raise AssertionError(f"{actual!r} differs from {expected!r} only in the type of a value")


def key_orders(value: Any, path: str = "$"):
    """Yield (path, keys) of every object in a JSON value, to compare the order in which keys are written."""
    if isinstance(value, dict):
        yield path, list(value)
        for key, item in value.items():
            yield from key_orders(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from key_orders(item, f"{path}[{index}]")


# ---------------------------------------------------------------------------------------------------------------------
# 1. e-commerce order
# ---------------------------------------------------------------------------------------------------------------------


class OrderStatus(Enum):
    PENDING = "pending"
    PAID = "paid"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class Sku(Enum):
    MUG = "SKU-MUG-001"
    TSHIRT = "SKU-TSH-042"
    COFFEE = "SKU-COF-150"


@dataclass
class Address:
    street: str
    city: str
    postal_code: str
    country: str
    apartment: Optional[str] = None


@dataclass
class Contact:
    email: str
    phone: Optional[str] = None


@dataclass
class Customer:
    customer_id: int
    name: str
    shipping_address: Address
    billing_address: Optional[Address] = None
    contact: Optional[Contact] = None


@dataclass
class LineItem:
    sku: Sku
    description: str
    quantity: Decimal
    unit_price: Decimal
    discount: Optional[Decimal] = None


@dataclass
class ShippingEvent:
    at: datetime
    location: str
    detail: Optional[str] = None


@dataclass
class Order:
    order_id: str
    customer: Customer
    status: OrderStatus
    items: list[LineItem]
    total: Decimal
    placed_at: datetime
    paid_at: Optional[datetime]
    invoice_pdf: Optional[bytes]
    shipping_events: dict[str, list[ShippingEvent]]
    tags: set[str]
    metadata: dict[str, Any]


def make_full_order() -> Order:
    return Order(
        order_id="ORD-2026-000123",
        customer=Customer(
            customer_id=48151,
            name="Dana Levi",
            shipping_address=Address("12 Herzl St", "Haifa", "3303310", "IL", apartment="4B"),
            billing_address=Address("7 Ben Yehuda St", "Tel Aviv", "6380101", "IL"),
            contact=Contact(email="dana.levi@example.com"),
        ),
        status=OrderStatus.SHIPPED,
        items=[
            LineItem(Sku.MUG, "Ceramic mug 350ml", Decimal("2"), Decimal("24.90")),
            LineItem(Sku.TSHIRT, "T-shirt, size M", Decimal("1"), Decimal("79.00"), discount=Decimal("7.90")),
            LineItem(Sku.COFFEE, "Coffee beans, loose (kg)", Decimal("0.750"), Decimal("98.00")),
        ],
        total=Decimal("194.40"),
        placed_at=datetime(2026, 3, 1, 9, 30, tzinfo=PLUS_TWO),
        paid_at=datetime(2026, 3, 1, 7, 31, 15, 250000, tzinfo=UTC),
        invoice_pdf=b"%PDF-1.4\n%\xff\x00",
        shipping_events={
            "PARCEL-1": [
                ShippingEvent(datetime(2026, 3, 2, 8, 0, tzinfo=UTC), "Haifa depot", "Picked up"),
                ShippingEvent(datetime(2026, 3, 2, 14, 45, tzinfo=PLUS_TWO), "Tel Aviv hub"),
                ShippingEvent(datetime(2026, 3, 3, 9, 15, 30, tzinfo=PLUS_TWO), "Haifa", "Out for delivery"),
            ],
            "PARCEL-2": [],
        },
        tags={"gift", "priority", "fragile"},
        metadata={
            "source": "web",
            "coupon": None,
            "gift_wrap": True,
            "item_weights_kg": [0.35, 0.2, 0.75],
            "utm": {"campaign": "spring-sale", "medium": "email", "term": None},
            "history": [{"rev": 1, "by": "web"}, {"rev": 2, "by": "support", "note": "address fixed"}],
        },
    )


FULL_ORDER_JSON = {
    "order_id": "ORD-2026-000123",
    "customer": {
        "customer_id": 48151,
        "name": "Dana Levi",
        "shipping_address": {
            "street": "12 Herzl St",
            "city": "Haifa",
            "postal_code": "3303310",
            "country": "IL",
            "apartment": "4B",
        },
        "billing_address": {
            "street": "7 Ben Yehuda St",
            "city": "Tel Aviv",
            "postal_code": "6380101",
            "country": "IL",
            "apartment": None,
        },
        "contact": {"email": "dana.levi@example.com", "phone": None},
    },
    "status": "shipped",
    "items": [
        {"sku": "SKU-MUG-001", "description": "Ceramic mug 350ml", "quantity": "2", "unit_price": "24.90",
         "discount": None},
        {"sku": "SKU-TSH-042", "description": "T-shirt, size M", "quantity": "1", "unit_price": "79.00",
         "discount": "7.90"},
        {"sku": "SKU-COF-150", "description": "Coffee beans, loose (kg)", "quantity": "0.750",
         "unit_price": "98.00", "discount": None},
    ],
    "total": "194.40",
    "placed_at": "2026-03-01T09:30:00+02:00",
    "paid_at": "2026-03-01T07:31:15.250000+00:00",
    "invoice_pdf": "JVBERi0xLjQKJf8A",
    "shipping_events": {
        "PARCEL-1": [
            {"at": "2026-03-02T08:00:00+00:00", "location": "Haifa depot", "detail": "Picked up"},
            {"at": "2026-03-02T14:45:00+02:00", "location": "Tel Aviv hub", "detail": None},
            {"at": "2026-03-03T09:15:30+02:00", "location": "Haifa", "detail": "Out for delivery"},
        ],
        "PARCEL-2": [],
    },
    "tags": ["fragile", "gift", "priority"],
    "metadata": {
        "source": "web",
        "coupon": None,
        "gift_wrap": True,
        "item_weights_kg": [0.35, 0.2, 0.75],
        "utm": {"campaign": "spring-sale", "medium": "email", "term": None},
        "history": [{"rev": 1, "by": "web"}, {"rev": 2, "by": "support", "note": "address fixed"}],
    },
}


def make_guest_order() -> Order:
    """The smallest order: every optional part absent, every container empty."""
    return Order(
        order_id="ORD-2026-000124",
        customer=Customer(99, "Guest", Address("1 Main St", "Eilat", "8800000", "IL")),
        status=OrderStatus.PENDING,
        items=[LineItem(Sku.MUG, "Ceramic mug 350ml", Decimal("1"), Decimal("24.90"))],
        total=Decimal("24.90"),
        placed_at=datetime(2026, 3, 5, 18, 0),
        paid_at=None,
        invoice_pdf=None,
        shipping_events={},
        tags=set(),
        metadata={},
    )


GUEST_ORDER_TEXT = (
    '{"order_id": "ORD-2026-000124", '
    '"customer": {"customer_id": 99, "name": "Guest", '
    '"shipping_address": {"street": "1 Main St", "city": "Eilat", "postal_code": "8800000", "country": "IL", '
    '"apartment": null}, "billing_address": null, "contact": null}, '
    '"status": "pending", '
    '"items": [{"sku": "SKU-MUG-001", "description": "Ceramic mug 350ml", "quantity": "1", "unit_price": "24.90", '
    '"discount": null}], '
    '"total": "24.90", "placed_at": "2026-03-05T18:00:00", "paid_at": null, "invoice_pdf": null, '
    '"shipping_events": {}, "tags": [], "metadata": {}}'
)


def test_guest_order_exact_text_pins_field_order_and_nulls():
    assert serialize_json(make_guest_order()) == GUEST_ORDER_TEXT


def test_guest_order_text_restores_the_guest_order():
    restored = deserialize_json(json.loads(GUEST_ORDER_TEXT), Order)

    assert_strictly_equal(restored, make_guest_order())
    assert restored.customer.billing_address is None
    assert restored.paid_at is None and restored.invoice_pdf is None


def test_full_order_keeps_money_scale_and_utc_offsets():
    text = serialize_json(make_full_order())
    restored = deserialize_json(json.loads(text), Order)

    assert '"total": "194.40"' in text and '"quantity": "0.750"' in text
    assert restored.items[2].quantity.as_tuple().exponent == -3
    assert restored.items[0].quantity.as_tuple().exponent == 0
    assert restored.placed_at.utcoffset() == timedelta(hours=2)
    assert restored.paid_at.utcoffset() == timedelta(0)
    assert restored.shipping_events["PARCEL-1"][1].at.utcoffset() == timedelta(hours=2)
    assert restored.paid_at.microsecond == 250000


ALL_BYTE_VALUES = bytes(range(256))

ATTACHMENTS = [
    pytest.param(None, None, id="no-attachment"),
    pytest.param(b"", "", id="empty-attachment"),
    pytest.param(b"\x00", "AA==", id="single-null-byte"),
    pytest.param(b"%PDF-1.4\n%\xff\x00", "JVBERi0xLjQKJf8A", id="pdf-header-with-non-utf8-bytes"),
    # the expected text comes from the stdlib encoder, not from jserpy
    pytest.param(ALL_BYTE_VALUES, base64.b64encode(ALL_BYTE_VALUES).decode("ascii"), id="all-256-byte-values"),
]


@pytest.mark.parametrize(("attachment", "text"), ATTACHMENTS)
def test_order_attachment_is_base64_text_and_none_stays_distinct_from_empty(attachment, text):
    order = replace(make_guest_order(), invoice_pdf=attachment)

    assert to_json_value(order)["invoice_pdf"] == text
    assert_round_trip(order, Order)


def test_quantities_written_as_json_integers_by_another_producer_restore_as_exact_decimals():
    # Decimal(2) is exact, unlike Decimal(0.75) of a float; whole quantities are what spreadsheets export as numbers
    other_producer = copy.deepcopy(FULL_ORDER_JSON)
    other_producer["items"][0]["quantity"] = 2
    other_producer["items"][1]["quantity"] = 1

    assert_strictly_equal(deserialize_json(other_producer, Order), make_full_order())


def test_prices_written_as_json_numbers_are_exact_when_the_text_is_parsed_with_decimal_floats():
    # the documented way to read a price list that has plain numbers: json.loads(..., parse_float=Decimal) keeps
    # "24.90" as written, where parsing it to a float first would leave only the nearest binary fraction
    text = (
        '[{"sku": "SKU-MUG-001", "description": "Ceramic mug", "quantity": 2, "unit_price": 24.90, "discount": null},'
        ' {"sku": "SKU-COF-150", "description": "Coffee", "quantity": 0.750, "unit_price": 98.00, "discount": 9.80}]'
    )

    items = deserialize_json(json.loads(text, parse_float=Decimal), list[LineItem])

    assert_strictly_equal(items, [
        LineItem(Sku.MUG, "Ceramic mug", Decimal("2"), Decimal("24.90")),
        LineItem(Sku.COFFEE, "Coffee", Decimal("0.750"), Decimal("98.00"), Decimal("9.80")),
    ])


def test_base64_junk_characters_are_discarded_when_reading_an_attachment_limitation():
    # Documents current behavior, not a promise: b64decode() skips characters outside the alphabet, so a damaged
    # attachment can load as different bytes instead of being rejected.
    damaged = corrupted(FULL_ORDER_JSON, ("invoice_pdf",), "AAE*C")

    assert deserialize_json(damaged, Order).invoice_pdf == b"\x00\x01\x02"


def test_large_order_with_two_hundred_line_items_round_trips():
    skus = list(Sku)
    items = [
        LineItem(
            skus[index % 3],
            f"Item {index}",
            Decimal(index % 7 + 1),
            Decimal(f"{index}.{index % 100:02d}"),
            Decimal("0.50") if index % 10 == 0 else None,
        )
        for index in range(200)
    ]
    order = replace(make_full_order(), items=items)

    parsed = to_json_value(order)
    assert len(parsed["items"]) == 200
    # 137 = 19 * 7 + 4 = 45 * 3 + 2, so the quantity is 5 and the SKU is the third one
    assert parsed["items"][137] == {
        "sku": "SKU-COF-150",
        "description": "Item 137",
        "quantity": "5",
        "unit_price": "137.37",
        "discount": None,
    }
    assert parsed["items"][10]["discount"] == "0.50" and parsed["items"][11]["discount"] is None
    assert_round_trip(order, Order)


ORDER_CORRUPTIONS = [
    pytest.param(("status",), "lost", id="unknown-status"),
    pytest.param(("status",), "SHIPPED", id="status-written-as-the-member-name"),
    pytest.param(("status",), REMOVED, id="missing-status"),
    pytest.param(("items", 1, "sku"), "SKU-NOPE-000", id="unknown-sku-in-a-line-item"),
    pytest.param(("items", 0, "unit_price"), "twenty-four ninety", id="unparseable-decimal"),
    pytest.param(("items", 2, "quantity"), [0.75], id="array-where-decimal-expected"),
    pytest.param(("items",), REMOVED, id="missing-items"),
    pytest.param(("total",), REMOVED, id="missing-total"),
    pytest.param(("placed_at",), "2026-13-45T25:61:00", id="impossible-timestamp"),
    pytest.param(("placed_at",), REMOVED, id="missing-timestamp"),
    pytest.param(("shipping_events", "PARCEL-1", 2, "at"), "yesterday", id="bad-timestamp-in-shipping-event"),
    pytest.param(("shipping_events", "PARCEL-1"), "in transit", id="string-where-event-list-expected"),
    pytest.param(("shipping_events", "PARCEL-2"), {"at": "2026-03-02T08:00:00"}, id="object-where-event-list-expected"),
    pytest.param(("shipping_events",), [["PARCEL-2", []]], id="array-of-pairs-where-event-table-expected"),
    pytest.param(("invoice_pdf",), "abc", id="base64-with-bad-padding"),
    pytest.param(("tags",), {"gift": True}, id="object-where-tag-set-expected"),
    pytest.param(("tags",), "gift", id="string-where-tag-set-expected"),
    pytest.param(("customer",), "Dana Levi", id="string-where-customer-expected"),
    pytest.param(("customer",), REMOVED, id="missing-customer"),
    pytest.param(("customer", "shipping_address"), ["12 Herzl St", "Haifa"], id="array-where-address-expected"),
    pytest.param(("customer", "shipping_address"), REMOVED, id="missing-shipping-address"),
]


@pytest.mark.parametrize(("path", "replacement"), ORDER_CORRUPTIONS)
def test_corrupted_order_is_rejected(path, replacement):
    assert_rejected(FULL_ORDER_JSON, Order, path, replacement)


@pytest.mark.parametrize(
    "bad",
    [pytest.param("in transit", id="string"), pytest.param(7, id="number"), pytest.param(True, id="bool")],
)
def test_primitive_where_a_dict_field_is_expected_is_passed_through_limitation(bad):
    # Documents current behavior, not a promise: a dict[...] annotation has no handler of its own and JSON primitives
    # are returned unchecked for such annotations, so the field holds the primitive. Mapping[...], list, tuple and set
    # annotations raise TypeError for the same input (see the corruptions of the config and of this order).
    broken = corrupted(FULL_ORDER_JSON, ("shipping_events",), bad)

    restored = deserialize_json(broken, Order)

    assert type(restored.shipping_events) is type(bad) and restored.shipping_events == bad


def test_order_whose_metadata_contains_itself_is_rejected():
    order = make_full_order()
    order.metadata["self"] = order.metadata

    with pytest.raises(ValueError, match="(?i)cycl"):
        serialize_json(order)


# ---------------------------------------------------------------------------------------------------------------------
# 2. organisation chart: a recursive dataclass
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Employee:
    employee_id: int
    name: str
    title: str
    hire_date: date
    manager_id: Optional[int]
    skills: frozenset[str]
    rating: Optional[float]
    reports: list[Employee]


def make_org_chart() -> Employee:
    lead = Employee(4, "Lior Bar", "Engineering Lead", date(2020, 9, 1), 2,
                    frozenset({"python", "mentoring", "code review"}), None, [])
    vp_engineering = Employee(2, "Amit Cohen", "VP Engineering", date(2017, 6, 15), 1,
                              frozenset({"python", "architecture"}), 4.0, [lead])
    vp_sales = Employee(3, "Yael Mizrahi", "VP Sales", date(2019, 1, 10), 1, frozenset(), 3.0, [])
    return Employee(1, "Noa Katz", "CEO", date(2015, 2, 1), None,
                    frozenset({"strategy", "fundraising"}), 4.5, [vp_engineering, vp_sales])


ORG_PRETTY_TEXT = """\
{
  "employee_id": 1,
  "name": "Noa Katz",
  "title": "CEO",
  "hire_date": "2015-02-01",
  "manager_id": null,
  "skills": [
    "fundraising",
    "strategy"
  ],
  "rating": 4.5,
  "reports": [
    {
      "employee_id": 2,
      "name": "Amit Cohen",
      "title": "VP Engineering",
      "hire_date": "2017-06-15",
      "manager_id": 1,
      "skills": [
        "architecture",
        "python"
      ],
      "rating": 4.0,
      "reports": [
        {
          "employee_id": 4,
          "name": "Lior Bar",
          "title": "Engineering Lead",
          "hire_date": "2020-09-01",
          "manager_id": 2,
          "skills": [
            "code review",
            "mentoring",
            "python"
          ],
          "rating": null,
          "reports": []
        }
      ]
    },
    {
      "employee_id": 3,
      "name": "Yael Mizrahi",
      "title": "VP Sales",
      "hire_date": "2019-01-10",
      "manager_id": 1,
      "skills": [],
      "rating": 3.0,
      "reports": []
    }
  ]
}"""

ORG_JSON = json.loads(ORG_PRETTY_TEXT)


def test_org_chart_pretty_printed_golden():
    assert serialize_json(make_org_chart(), indent=2) == ORG_PRETTY_TEXT


def test_org_chart_pretty_text_round_trips_with_the_same_options():
    assert_round_trip(make_org_chart(), Employee, indent=2)


def test_org_chart_restores_the_tree_not_just_the_root():
    restored = deserialize_json(json.loads(ORG_PRETTY_TEXT), Employee)

    assert [report.name for report in restored.reports] == ["Amit Cohen", "Yael Mizrahi"]
    lead = restored.reports[0].reports[0]
    assert type(lead) is Employee and lead.skills == frozenset({"code review", "mentoring", "python"})
    assert lead.hire_date == date(2020, 9, 1) and lead.rating is None and lead.reports == []
    assert restored.manager_id is None and lead.manager_id == 2


ORG_CORRUPTIONS = [
    pytest.param(("reports", 0, "reports", 0, "hire_date"), "2020-13-01", id="month-13-three-levels-down"),
    pytest.param(("reports", 0, "reports", 0, "hire_date"), 20200901, id="number-where-date-expected"),
    pytest.param(("hire_date",), "2015-02-01T00:00:00", id="timestamp-where-date-expected"),
    pytest.param(("reports", 0, "reports", 0, "skills"), "python", id="string-where-skill-set-expected"),
    pytest.param(("reports", 0, "reports", 0, "reports"), REMOVED, id="leaf-without-report-list"),
    pytest.param(("reports", 1), "Yael Mizrahi", id="string-where-report-expected"),
    pytest.param(("reports",), {"0": {}}, id="object-where-report-list-expected"),
    pytest.param(("reports", 0, "reports"), None, id="null-where-report-list-expected"),
    pytest.param(("skills",), REMOVED, id="missing-skills"),
    pytest.param(("hire_date",), REMOVED, id="missing-hire-date"),
]


@pytest.mark.parametrize(("path", "replacement"), ORG_CORRUPTIONS)
def test_corrupted_org_chart_is_rejected(path, replacement):
    assert_rejected(ORG_JSON, Employee, path, replacement)


def test_management_chain_fifty_levels_deep_round_trips():
    node = None
    for level in range(50, 0, -1):
        node = Employee(level, f"Manager {level}", "Manager", date(2021, 9, 1), level - 1 if level > 1 else None,
                        frozenset({f"skill-{level}"}), None, [] if node is None else [node])
    chain = node

    parsed = to_json_value(chain)
    cursor = parsed
    for level in range(1, 51):
        assert cursor["employee_id"] == level
        assert cursor["skills"] == [f"skill-{level}"]
        cursor = cursor["reports"][0] if cursor["reports"] else None
    assert cursor is None

    restored = assert_round_trip(chain, Employee)
    assert restored.reports[0].reports[0].employee_id == 3


def test_manager_with_two_hundred_direct_reports_round_trips():
    team = [Employee(number, f"Dev {number}", "Developer", date(2021, 9, 1), 1, frozenset({"python"}), None, [])
            for number in range(2, 202)]
    manager = Employee(1, "Manager", "Director", date(2018, 1, 1), None, frozenset(), 4.5, team)

    parsed = to_json_value(manager)
    assert [report["employee_id"] for report in parsed["reports"]] == list(range(2, 202))
    restored = assert_round_trip(manager, Employee)
    assert len(restored.reports) == 200 and all(type(report) is Employee for report in restored.reports)


def test_employee_shared_by_two_managers_is_not_a_cycle():
    shared = Employee(9, "Gal Sharon", "Designer", date(2022, 2, 2), None, frozenset({"figma"}), None, [])
    left = Employee(2, "Left", "Manager", date(2020, 1, 1), 1, frozenset(), None, [shared])
    right = Employee(3, "Right", "Manager", date(2020, 1, 1), 1, frozenset(), None, [shared])
    ceo = Employee(1, "Boss", "CEO", date(2010, 1, 1), None, frozenset(), None, [left, right])

    restored = assert_round_trip(ceo, Employee)

    assert restored.reports[0].reports[0] == restored.reports[1].reports[0]
    assert restored.reports[0].reports[0] is not restored.reports[1].reports[0]


def test_org_chart_with_a_report_loop_is_rejected():
    ceo = make_org_chart()
    lead = ceo.reports[0].reports[0]
    lead.reports.append(ceo)

    with pytest.raises(ValueError, match="(?i)cycl"):
        serialize_json(ceo)


# ---------------------------------------------------------------------------------------------------------------------
# 3. application configuration tree
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class PostgresSettings:
    kind: Literal["postgres"]
    host: str
    port: int
    database: str
    ssl_cert: Optional[Path] = None
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class MySqlSettings:
    # the same fields as PostgresSettings: only the Literal tells them apart
    kind: Literal["mysql"]
    host: str
    port: int
    database: str
    ssl_cert: Optional[Path] = None
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class SqliteSettings:
    kind: Literal["sqlite"]
    file: Path
    read_only: bool = False


DatabaseSettings = Union[PostgresSettings, MySqlSettings, SqliteSettings]


@dataclass
class RedisCacheSettings:
    host: str
    port: int
    ttl_seconds: int
    key_prefix: Optional[str] = None


@dataclass
class LoggingSettings:
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    file: Optional[Path]
    rotate_mb: Optional[float]


@dataclass
class AppConfig:
    app_name: str
    env: Literal["dev", "staging", "prod"]
    data_dir: Path
    database: DatabaseSettings
    replicas: list[DatabaseSettings]
    logging: LoggingSettings
    cache: Optional[RedisCacheSettings]
    plugin_dirs: tuple[Path, ...]
    allowed_hosts: frozenset[str]
    feature_flags: dict[str, bool]
    overrides: MappingProxyType[str, Any]


def make_prod_config(path: Any = Path) -> AppConfig:
    return AppConfig(
        app_name="billing-service",
        env="prod",
        data_dir=path("/srv/billing/data"),
        database=PostgresSettings(
            "postgres", "db.internal", 5432, "billing", ssl_cert=path("/etc/ssl/db.pem"),
            options={"sslmode": "verify-full", "application_name": "billing"},
        ),
        replicas=[
            PostgresSettings("postgres", "replica-1.internal", 5432, "billing"),
            MySqlSettings("mysql", "reports.internal", 3306, "billing_reports"),
            SqliteSettings("sqlite", path("/srv/billing/fallback.db"), read_only=True),
        ],
        logging=LoggingSettings("INFO", path("/var/log/billing/app.log"), 50.0),
        cache=RedisCacheSettings("cache.internal", 6379, 300, key_prefix="billing:"),
        plugin_dirs=(path("/opt/billing/plugins"), path("/opt/shared/plugins")),
        allowed_hosts=frozenset({"billing.example.com", "api.example.com"}),
        feature_flags={"new_checkout": True, "legacy_export": False},
        overrides=MappingProxyType({
            "http": {"timeout": 2.5, "retries": [1, 2, 4]},
            "banner": None,
            "workers": 8,
        }),
    )


def make_dev_config(path: Any = Path) -> AppConfig:
    return AppConfig(
        app_name="billing-service",
        env="dev",
        data_dir=path("/home/dev/billing/data"),
        database=SqliteSettings("sqlite", path("/home/dev/billing.db")),
        replicas=[],
        logging=LoggingSettings("DEBUG", None, None),
        cache=None,
        plugin_dirs=(),
        allowed_hosts=frozenset({"localhost"}),
        feature_flags={"new_checkout": True},
        overrides=MappingProxyType({}),
    )


PROD_CONFIG_JSON = {
    "app_name": "billing-service",
    "env": "prod",
    "data_dir": "/srv/billing/data",
    "database": {
        "kind": "postgres",
        "host": "db.internal",
        "port": 5432,
        "database": "billing",
        "ssl_cert": "/etc/ssl/db.pem",
        "options": {"sslmode": "verify-full", "application_name": "billing"},
    },
    "replicas": [
        {"kind": "postgres", "host": "replica-1.internal", "port": 5432, "database": "billing",
         "ssl_cert": None, "options": {}},
        {"kind": "mysql", "host": "reports.internal", "port": 3306, "database": "billing_reports",
         "ssl_cert": None, "options": {}},
        {"kind": "sqlite", "file": "/srv/billing/fallback.db", "read_only": True},
    ],
    "logging": {"level": "INFO", "file": "/var/log/billing/app.log", "rotate_mb": 50.0},
    "cache": {"host": "cache.internal", "port": 6379, "ttl_seconds": 300, "key_prefix": "billing:"},
    "plugin_dirs": ["/opt/billing/plugins", "/opt/shared/plugins"],
    "allowed_hosts": ["api.example.com", "billing.example.com"],
    "feature_flags": {"new_checkout": True, "legacy_export": False},
    "overrides": {"http": {"timeout": 2.5, "retries": [1, 2, 4]}, "banner": None, "workers": 8},
}

DEV_CONFIG_JSON = {
    "app_name": "billing-service",
    "env": "dev",
    "data_dir": "/home/dev/billing/data",
    "database": {"kind": "sqlite", "file": "/home/dev/billing.db", "read_only": False},
    "replicas": [],
    "logging": {"level": "DEBUG", "file": None, "rotate_mb": None},
    "cache": None,
    "plugin_dirs": [],
    "allowed_hosts": ["localhost"],
    "feature_flags": {"new_checkout": True},
    "overrides": {},
}

DEV_CONFIG_SORTED_COMPACT_TEXT = (
    '{"allowed_hosts":["localhost"],"app_name":"billing-service","cache":null,'
    '"data_dir":"/home/dev/billing/data",'
    '"database":{"file":"/home/dev/billing.db","kind":"sqlite","read_only":false},'
    '"env":"dev","feature_flags":{"new_checkout":true},'
    '"logging":{"file":null,"level":"DEBUG","rotate_mb":null},'
    '"overrides":{},"plugin_dirs":[],"replicas":[]}'
)


def test_dev_config_sorted_compact_golden():
    text = serialize_json(make_dev_config(PurePosixPath), sort_keys=True, separators=(",", ":"))

    assert text == DEV_CONFIG_SORTED_COMPACT_TEXT


@posix_only
def test_dev_config_round_trips_with_sorted_compact_output():
    assert_round_trip(make_dev_config(), AppConfig, sort_keys=True, separators=(",", ":"))


@posix_only
def test_sorted_compact_text_restores_the_dev_config():
    restored = deserialize_json(json.loads(DEV_CONFIG_SORTED_COMPACT_TEXT), AppConfig)

    assert_strictly_equal(restored, make_dev_config())


@posix_only
def test_config_sections_that_are_absent_restore_as_none():
    without_cache = {key: value for key, value in DEV_CONFIG_JSON.items() if key != "cache"}
    without_log_file = corrupted(DEV_CONFIG_JSON, ("logging",), {"level": "DEBUG"})

    assert deserialize_json(without_cache, AppConfig).cache is None
    assert_strictly_equal(
        deserialize_json(without_log_file, AppConfig).logging, LoggingSettings("DEBUG", None, None)
    )


@posix_only
@pytest.mark.parametrize(
    ("database", "expected_type"),
    [
        pytest.param({"kind": "postgres", "host": "h", "port": 1, "database": "d"}, PostgresSettings, id="postgres"),
        pytest.param({"kind": "mysql", "host": "h", "port": 1, "database": "d"}, MySqlSettings, id="mysql"),
        pytest.param({"kind": "sqlite", "file": "/x.db"}, SqliteSettings, id="sqlite"),
    ],
)
def test_the_literal_kind_selects_the_union_member(database, expected_type):
    document = {**DEV_CONFIG_JSON, "database": database, "replicas": [database, database]}

    restored = deserialize_json(document, AppConfig)

    assert type(restored.database) is expected_type
    assert [type(replica) for replica in restored.replicas] == [expected_type, expected_type]
    assert restored.database.kind == database["kind"]


@posix_only
def test_config_restores_read_only_overrides_detached_from_the_input():
    parsed = to_json_value(make_prod_config())
    restored = deserialize_json(parsed, AppConfig)

    parsed["overrides"]["http"]["retries"].append(8)
    parsed["overrides"]["workers"] = 99

    assert type(restored.overrides) is MappingProxyType
    assert restored.overrides["http"]["retries"] == [1, 2, 4] and restored.overrides["workers"] == 8
    with pytest.raises(TypeError, match="mappingproxy"):
        restored.overrides["workers"] = 1


CONFIG_CORRUPTIONS = [
    pytest.param(("env",), "qa", id="unknown-environment"),
    pytest.param(("database", "kind"), "oracle", id="unknown-database-kind"),
    pytest.param(("database",), {"host": "db.internal", "port": 5432, "database": "billing"},
                 id="database-without-kind"),
    pytest.param(("replicas", 2, "file"), REMOVED, id="sqlite-replica-without-file"),
    pytest.param(("replicas", 1, "kind"), "mongo", id="unknown-kind-in-the-second-replica"),
    pytest.param(("replicas",), {"kind": "sqlite"}, id="object-where-replica-list-expected"),
    pytest.param(("logging", "level"), "TRACE", id="unknown-log-level"),
    pytest.param(("logging",), REMOVED, id="missing-logging-section"),
    pytest.param(("cache",), [], id="array-where-cache-section-expected"),
    pytest.param(("plugin_dirs",), "/opt/billing/plugins", id="string-where-plugin-dirs-expected"),
    pytest.param(("allowed_hosts",), {"billing.example.com": True}, id="object-where-allowed-hosts-expected"),
    pytest.param(("overrides",), [["workers", 8]], id="array-where-overrides-expected"),
    pytest.param(("overrides",), "workers=8", id="string-where-overrides-expected"),
    pytest.param(("data_dir",), 42, id="number-where-path-expected"),
    pytest.param(("data_dir",), REMOVED, id="missing-data-dir"),
]


@posix_only
@pytest.mark.parametrize(("path", "replacement"), CONFIG_CORRUPTIONS)
def test_corrupted_config_is_rejected(path, replacement):
    assert_rejected(PROD_CONFIG_JSON, AppConfig, path, replacement)


@dataclass
class RemoteHost:
    name: str
    install_dir: PureWindowsPath


def test_windows_path_field_is_serialized_with_its_backslashes():
    host = RemoteHost("build-01", PureWindowsPath("C:/Program Files/App"))

    assert to_json_value(host) == {"name": "build-01", "install_dir": "C:\\Program Files\\App"}


def test_windows_path_field_restores_as_a_pure_windows_path():
    restored = deserialize_json({"name": "build-01", "install_dir": "C:\\Program Files\\App"}, RemoteHost)

    assert isinstance(restored.install_dir, PureWindowsPath)
    assert restored.install_dir.parts == ("C:\\", "Program Files", "App")


@posix_only
def test_pure_posix_path_is_written_unnormalized_and_restored_as_an_equal_posix_path():
    parsed = to_json_value({"root": PurePosixPath("/srv/app/../data")})
    restored = deserialize_json(parsed["root"], PurePosixPath)

    assert parsed == {"root": "/srv/app/../data"}
    assert isinstance(restored, PurePosixPath) and restored == PurePosixPath("/srv/app/../data")
    assert restored.parts == ("/", "srv", "app", "..", "data")


# ---------------------------------------------------------------------------------------------------------------------
# 4. audit log: a list whose items are one of three dataclasses, told apart by a Literal field
# ---------------------------------------------------------------------------------------------------------------------


class AccessMode(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"


@dataclass
class LoginEvent:
    kind: Literal["login"]
    at: datetime
    user: str
    ip: str
    success: bool
    mfa: Optional[Literal["totp", "sms"]] = None


@dataclass
class FileAccessEvent:
    kind: Literal["file_access"]
    at: datetime
    user: str
    path: str
    mode: AccessMode
    size_bytes: Optional[int] = None
    digest: Optional[bytes] = None


@dataclass
class PermissionChangeEvent:
    kind: Literal["permission_change"]
    at: datetime
    actor: str
    target: str
    granted: frozenset[str]
    revoked: frozenset[str]
    role_changes: dict[str, tuple[str, str]]


AuditEvent = Union[LoginEvent, FileAccessEvent, PermissionChangeEvent]


@dataclass
class AuditLog:
    tenant: str
    generated_at: datetime
    events: list[AuditEvent]
    checksum: bytes


@dataclass
class AuditLogPep604:
    tenant: str
    generated_at: datetime
    events: list[LoginEvent | FileAccessEvent | PermissionChangeEvent]
    checksum: bytes


def _audit_events() -> list:
    return [
        LoginEvent("login", datetime(2026, 4, 1, 8, 0, tzinfo=UTC), "dana", "10.0.0.7", True, mfa="totp"),
        LoginEvent("login", datetime(2026, 4, 1, 8, 5, 12, tzinfo=UTC), "mallory", "203.0.113.9", False),
        FileAccessEvent("file_access", datetime(2026, 4, 1, 8, 10, tzinfo=UTC), "dana", "/finance/q1.xlsx",
                        AccessMode.READ, size_bytes=48213, digest=b"\xde\xad\xbe\xef"),
        FileAccessEvent("file_access", datetime(2026, 4, 1, 8, 11, tzinfo=UTC), "dana", "/finance/q1.xlsx",
                        AccessMode.DELETE),
        PermissionChangeEvent(
            "permission_change", datetime(2026, 4, 1, 9, 0, tzinfo=UTC), "admin", "dana",
            frozenset({"finance:write", "finance:read"}), frozenset(),
            {"finance": ("viewer", "editor"), "hr": ("none", "viewer")},
        ),
    ]


def make_audit_log() -> AuditLog:
    return AuditLog("acme", datetime(2026, 4, 1, 10, 0, tzinfo=UTC), _audit_events(), b"\x00\x01\x02")


def make_audit_log_pep604() -> AuditLogPep604:
    return AuditLogPep604("acme", datetime(2026, 4, 1, 10, 0, tzinfo=UTC), _audit_events(), b"\x00\x01\x02")


AUDIT_LOG_JSON = {
    "tenant": "acme",
    "generated_at": "2026-04-01T10:00:00+00:00",
    "events": [
        {"kind": "login", "at": "2026-04-01T08:00:00+00:00", "user": "dana", "ip": "10.0.0.7", "success": True,
         "mfa": "totp"},
        {"kind": "login", "at": "2026-04-01T08:05:12+00:00", "user": "mallory", "ip": "203.0.113.9",
         "success": False, "mfa": None},
        {"kind": "file_access", "at": "2026-04-01T08:10:00+00:00", "user": "dana", "path": "/finance/q1.xlsx",
         "mode": "read", "size_bytes": 48213, "digest": "3q2+7w=="},
        {"kind": "file_access", "at": "2026-04-01T08:11:00+00:00", "user": "dana", "path": "/finance/q1.xlsx",
         "mode": "delete", "size_bytes": None, "digest": None},
        {"kind": "permission_change", "at": "2026-04-01T09:00:00+00:00", "actor": "admin", "target": "dana",
         "granted": ["finance:read", "finance:write"], "revoked": [],
         "role_changes": {"finance": ["viewer", "editor"], "hr": ["none", "viewer"]}},
    ],
    "checksum": "AAEC",
}


def test_audit_event_exact_text_has_sorted_sets_and_arrays_for_tuples():
    event = _audit_events()[4]

    assert serialize_json(event, separators=(",", ":")) == (
        '{"kind":"permission_change","at":"2026-04-01T09:00:00+00:00","actor":"admin","target":"dana",'
        '"granted":["finance:read","finance:write"],"revoked":[],'
        '"role_changes":{"finance":["viewer","editor"],"hr":["none","viewer"]}}'
    )


@pytest.mark.parametrize("log_cls", [AuditLog, AuditLogPep604])
def test_each_audit_event_restores_as_its_own_class_in_order(log_cls):
    restored = deserialize_json(copy.deepcopy(AUDIT_LOG_JSON), log_cls)

    assert [type(event) for event in restored.events] == [
        LoginEvent, LoginEvent, FileAccessEvent, FileAccessEvent, PermissionChangeEvent,
    ]
    assert restored.events[2].digest == b"\xde\xad\xbe\xef" and restored.events[2].mode is AccessMode.READ
    assert restored.events[4].role_changes["finance"] == ("viewer", "editor")
    assert restored.events[4].granted == frozenset({"finance:read", "finance:write"})


def test_audit_log_ignores_fields_added_by_a_newer_producer():
    newer = copy.deepcopy(AUDIT_LOG_JSON)
    newer["schema"] = 7
    newer["events"][0]["user_agent"] = "curl/8.0"
    newer["events"][4]["approvals"] = [{"by": "cto", "at": "2026-04-01T08:59:00+00:00"}]

    assert_strictly_equal(deserialize_json(newer, AuditLog), make_audit_log())


def test_audit_log_of_three_hundred_mixed_events_round_trips():
    start = datetime(2026, 4, 1, tzinfo=UTC)
    events = []
    for index in range(300):
        at = start + timedelta(minutes=index)
        if index % 3 == 0:
            events.append(LoginEvent("login", at, f"user{index}", f"10.0.{index // 256}.{index % 256}",
                                     index % 2 == 0))
        elif index % 3 == 1:
            events.append(FileAccessEvent("file_access", at, f"user{index}", f"/files/{index}.txt",
                                          list(AccessMode)[index % 3], size_bytes=index * 1024))
        else:
            events.append(PermissionChangeEvent("permission_change", at, "admin", f"user{index}",
                                                frozenset({f"role-{index}", "base"}), frozenset(), {}))
    log = AuditLog("acme", start, events, b"")

    parsed = to_json_value(log)
    assert [event["kind"] for event in parsed["events"]] == ["login", "file_access", "permission_change"] * 100
    assert parsed["events"][299]["granted"] == ["base", "role-299"]
    assert parsed["events"][298]["mode"] == "write" and parsed["events"][298]["size_bytes"] == 298 * 1024
    restored = assert_round_trip(log, AuditLog)
    assert [type(event) for event in restored.events[:3]] == [LoginEvent, FileAccessEvent, PermissionChangeEvent]


AUDIT_CORRUPTIONS = [
    pytest.param(("events", 0, "kind"), "logout", id="unknown-event-kind"),
    pytest.param(("events", 0, "kind"), REMOVED, id="event-without-kind"),
    pytest.param(("events", 0, "mfa"), "fido", id="unknown-mfa-literal"),
    pytest.param(("events", 1, "at"), "last tuesday", id="bad-timestamp-in-login"),
    pytest.param(("events", 2, "mode"), "execute", id="unknown-access-mode"),
    pytest.param(("events", 2, "digest"), "abc", id="base64-with-bad-padding"),
    pytest.param(("events", 4, "role_changes", "finance"), ["viewer"], id="one-element-role-change-pair"),
    pytest.param(("events", 4, "role_changes", "hr"), ["none", "viewer", "editor"],
                 id="three-element-role-change-pair"),
    pytest.param(("events", 4, "granted"), "finance:read", id="string-where-granted-set-expected"),
    pytest.param(("events", 2), "file_access", id="string-where-event-expected"),
    pytest.param(("events",), {"0": {}}, id="object-where-event-list-expected"),
    pytest.param(("events",), REMOVED, id="missing-events"),
    pytest.param(("generated_at",), "2026-04-31T10:00:00+00:00", id="april-31st"),
]


@pytest.mark.parametrize("log_cls", [AuditLog, AuditLogPep604])
@pytest.mark.parametrize(("path", "replacement"), AUDIT_CORRUPTIONS)
def test_corrupted_audit_log_is_rejected(log_cls, path, replacement):
    assert_rejected(AUDIT_LOG_JSON, log_cls, path, replacement)


def test_event_relabelled_with_another_kind_is_accepted_with_empty_fields_limitation():
    # Documents current behavior, not a promise: the file access below is read as a LoginEvent, because absent
    # fields become None and primitives are not validated; only the Literal discriminator is checked.
    relabelled = corrupted(AUDIT_LOG_JSON, ("events", 3, "kind"), "login")

    restored = deserialize_json(relabelled, AuditLog)

    event = restored.events[3]
    assert type(event) is LoginEvent
    assert event.user == "dana" and event.ip is None and event.success is None


# ---------------------------------------------------------------------------------------------------------------------
# 5. GeoJSON-like feature collection: coordinates nested up to four lists deep
# ---------------------------------------------------------------------------------------------------------------------

Position = Union[tuple[float, float], tuple[float, float, float]]


@dataclass
class Point:
    type: Literal["Point"]
    coordinates: Position


@dataclass
class LineString:
    type: Literal["LineString"]
    coordinates: list[Position]


@dataclass
class Polygon:
    type: Literal["Polygon"]
    coordinates: list[list[Position]]


@dataclass
class MultiPolygon:
    type: Literal["MultiPolygon"]
    coordinates: list[list[list[Position]]]


Geometry = Union[Point, LineString, Polygon, MultiPolygon]


@dataclass
class Feature:
    type: Literal["Feature"]
    geometry: Optional[Geometry]
    properties: Optional[dict[str, Any]]
    id: Optional[Union[int, str]] = None
    bbox: Optional[tuple[float, float, float, float]] = None


@dataclass
class FeatureCollection:
    type: Literal["FeatureCollection"]
    features: list[Feature]
    bbox: Optional[tuple[float, float, float, float]] = None


def make_feature_collection() -> FeatureCollection:
    return FeatureCollection(
        "FeatureCollection",
        [
            Feature("Feature", Point("Point", (34.9896, 32.794)),
                    {"name": "Haifa", "population": 285316, "capital": False}, id=1),
            Feature("Feature",
                    LineString("LineString", [(34.99, 32.79), (35.0, 32.8), (35.01, 32.81, 120.5)]),
                    {"name": "Carmel trail", "length_km": 4.2, "surface": ["dirt", "gravel"]},
                    id="trail-7", bbox=(34.99, 32.79, 35.01, 32.81)),
            Feature("Feature",
                    Polygon("Polygon", [
                        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)],
                        [(2.0, 2.0), (2.0, 4.0), (4.0, 4.0), (4.0, 2.0), (2.0, 2.0)],
                    ]),
                    {"name": "Carmel Park", "area_km2": 12.5, "visits": {"2025": 1200, "2026": 1450},
                     "closed": None, "tags": ["park", "forest"]}),
            Feature("Feature",
                    MultiPolygon("MultiPolygon", [
                        [[(20.0, 20.0), (30.0, 20.0), (30.0, 30.0), (20.0, 20.0)]],
                        [
                            [(40.0, 40.0), (50.0, 40.0), (50.0, 50.0), (40.0, 40.0)],
                            [(42.0, 42.0), (44.0, 42.0), (44.0, 44.0), (42.0, 42.0)],
                        ],
                    ]),
                    {}, id=4),
            Feature("Feature", None, None),
        ],
        bbox=(0.0, 0.0, 50.0, 50.0),
    )


FEATURE_COLLECTION_JSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [34.9896, 32.794]},
            "properties": {"name": "Haifa", "population": 285316, "capital": False},
            "id": 1,
            "bbox": None,
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[34.99, 32.79], [35.0, 32.8], [35.01, 32.81, 120.5]],
            },
            "properties": {"name": "Carmel trail", "length_km": 4.2, "surface": ["dirt", "gravel"]},
            "id": "trail-7",
            "bbox": [34.99, 32.79, 35.01, 32.81],
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]],
                    [[2.0, 2.0], [2.0, 4.0], [4.0, 4.0], [4.0, 2.0], [2.0, 2.0]],
                ],
            },
            "properties": {
                "name": "Carmel Park",
                "area_km2": 12.5,
                "visits": {"2025": 1200, "2026": 1450},
                "closed": None,
                "tags": ["park", "forest"],
            },
            "id": None,
            "bbox": None,
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [
                    [[[20.0, 20.0], [30.0, 20.0], [30.0, 30.0], [20.0, 20.0]]],
                    [
                        [[40.0, 40.0], [50.0, 40.0], [50.0, 50.0], [40.0, 40.0]],
                        [[42.0, 42.0], [44.0, 42.0], [44.0, 44.0], [42.0, 42.0]],
                    ],
                ],
            },
            "properties": {},
            "id": 4,
            "bbox": None,
        },
        {"type": "Feature", "geometry": None, "properties": None, "id": None, "bbox": None},
    ],
    "bbox": [0.0, 0.0, 50.0, 50.0],
}


def test_small_feature_collection_compact_golden():
    collection = FeatureCollection(
        "FeatureCollection",
        [Feature("Feature", Point("Point", (35.0, 32.0)), {"name": "A"}, id=1), Feature("Feature", None, None)],
    )

    assert serialize_json(collection, separators=(",", ":")) == (
        '{"type":"FeatureCollection","features":['
        '{"type":"Feature","geometry":{"type":"Point","coordinates":[35.0,32.0]},"properties":{"name":"A"},'
        '"id":1,"bbox":null},'
        '{"type":"Feature","geometry":null,"properties":null,"id":null,"bbox":null}],"bbox":null}'
    )


def test_geometry_union_restores_every_geometry_class_and_position_arity():
    restored = deserialize_json(copy.deepcopy(FEATURE_COLLECTION_JSON), FeatureCollection)

    assert [type(feature.geometry) for feature in restored.features] == [
        Point, LineString, Polygon, MultiPolygon, type(None),
    ]
    assert [len(position) for position in restored.features[1].geometry.coordinates] == [2, 2, 3]
    assert restored.features[3].geometry.coordinates[1][1][2] == (44.0, 44.0)
    assert restored.features[1].id == "trail-7" and restored.features[0].id == 1
    assert restored.features[4].properties is None and restored.features[3].properties == {}


def test_whole_number_coordinates_written_as_integers_are_restored_as_floats():
    # other encoders write 35.0 as 35
    document = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 3], [0, 0]]]},
            "properties": {"n": 1},
        }],
        "bbox": [0, 0, 4, 3],
    }

    restored = deserialize_json(document, FeatureCollection)

    expected = FeatureCollection(
        "FeatureCollection",
        [Feature("Feature", Polygon("Polygon", [[(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 0.0)]]), {"n": 1})],
        bbox=(0.0, 0.0, 4.0, 3.0),
    )
    assert_strictly_equal(restored, expected)


def test_polygon_with_five_hundred_vertices_keeps_every_coordinate():
    ring = [(float(index), float(index * index % 97)) for index in range(499)] + [(0.0, 0.0)]
    fine = [(34.0 + index * 0.0001, 32.0 + index * 0.0003) for index in range(500)]
    collection = FeatureCollection("FeatureCollection", [
        Feature("Feature", Polygon("Polygon", [ring]), {"vertices": 500}),
        Feature("Feature", LineString("LineString", fine), None),
    ])

    parsed = to_json_value(collection)
    polygon_ring = parsed["features"][0]["geometry"]["coordinates"][0]
    assert len(polygon_ring) == 500
    assert polygon_ring[3] == [3.0, 9.0] and polygon_ring[498] == [498.0, 72.0] and polygon_ring[499] == [0.0, 0.0]
    assert_round_trip(collection, FeatureCollection)


def test_multipolygon_of_twenty_islands_with_lakes_round_trips():
    islands = []
    for number in range(20):
        shore = [(number * 10.0, 0.0), (number * 10.0 + 5.0, 0.0), (number * 10.0 + 5.0, 5.0), (number * 10.0, 0.0)]
        lake = [(number * 10.0 + 1.0, 1.0), (number * 10.0 + 2.0, 1.0), (number * 10.0 + 2.0, 2.0),
                (number * 10.0 + 1.0, 1.0)]
        islands.append([shore, lake] if number % 2 else [shore])
    collection = FeatureCollection(
        "FeatureCollection", [Feature("Feature", MultiPolygon("MultiPolygon", islands), {"islands": 20})],
    )

    parsed = to_json_value(collection)
    assert [len(island) for island in parsed["features"][0]["geometry"]["coordinates"]] == [1, 2] * 10
    assert parsed["features"][0]["geometry"]["coordinates"][19][1][0] == [191.0, 1.0]
    assert_round_trip(collection, FeatureCollection)


def test_feature_properties_nested_fifteen_levels_deep_are_kept_as_they_are():
    properties: dict = {"leaf": [1, 2.5, "three", None, True]}
    for level in range(15):
        properties = {"level": 15 - level, "child": properties, "siblings": [level, {"x": level}]}
    collection = FeatureCollection("FeatureCollection", [Feature("Feature", None, properties)])

    restored = assert_round_trip(collection, FeatureCollection)

    cursor = restored.features[0].properties
    for level in range(1, 16):
        assert cursor["level"] == level
        cursor = cursor["child"]
    assert cursor == {"leaf": [1, 2.5, "three", None, True]}


def test_empty_feature_collection_round_trips():
    empty = FeatureCollection("FeatureCollection", [])

    assert to_json_value(empty) == {"type": "FeatureCollection", "features": [], "bbox": None}
    assert_round_trip(empty, FeatureCollection)


GEO_CORRUPTIONS = [
    pytest.param(("features", 0, "geometry", "type"), "Circle", id="unknown-geometry-type"),
    pytest.param(("features", 0, "geometry", "coordinates"), [34.9, 32.7, 1.0, 2.0], id="four-numbers-in-a-position"),
    pytest.param(("features", 0, "geometry", "coordinates"), "34.9,32.7", id="string-where-position-expected"),
    pytest.param(("features", 0, "geometry", "coordinates"), [34.9], id="one-number-in-a-position"),
    pytest.param(("features", 1, "geometry", "coordinates", 1), 35.0, id="number-where-position-expected"),
    pytest.param(("features", 2, "geometry", "coordinates", 0), [0.0, 1.0], id="position-where-ring-expected"),
    pytest.param(("features", 2, "geometry", "coordinates"), {"outer": []}, id="object-where-rings-expected"),
    pytest.param(("features", 3, "geometry", "coordinates", 1, 1, 0), [42.0], id="short-position-four-levels-down"),
    pytest.param(("features", 3, "geometry", "coordinates", 0, 0, 0), [[20.0, 20.0]],
                 id="ring-nested-one-level-too-deep"),
    pytest.param(("features", 1, "bbox"), [34.99, 32.79, 35.01], id="three-number-bbox"),
    pytest.param(("bbox",), "0,0,50,50", id="string-bbox"),
    pytest.param(("features",), {"type": "Feature"}, id="object-where-feature-list-expected"),
    pytest.param(("features", 0), "Feature", id="string-where-feature-expected"),
    pytest.param(("features", 0, "type"), "Layer", id="unknown-feature-type"),
    pytest.param(("type",), "Feature", id="wrong-collection-type"),
    pytest.param(("features",), REMOVED, id="missing-features"),
]


@pytest.mark.parametrize(("path", "replacement"), GEO_CORRUPTIONS)
def test_corrupted_feature_collection_is_rejected(path, replacement):
    assert_rejected(FEATURE_COLLECTION_JSON, FeatureCollection, path, replacement)


# ---------------------------------------------------------------------------------------------------------------------
# 6. graph and matrices: tuple keys, enum keys, sets, infinities
# ---------------------------------------------------------------------------------------------------------------------


class NodeKind(Enum):
    SERVICE = "service"
    DATABASE = "database"
    QUEUE = "queue"


class Relation(Enum):
    DEPENDS_ON = "depends_on"
    PUBLISHES_TO = "publishes_to"
    CONFLICTS_WITH = "conflicts_with"


@dataclass
class Topology:
    name: str
    nodes: list[str]
    kinds: dict[str, NodeKind]
    adjacency: dict[str, set[str]]
    adjacency_matrix: list[list[int]]
    distances: list[list[float]]
    allowed_edges: dict[Relation, list[NodeKind]]
    capacity: dict[NodeKind, dict[Relation, tuple[int, ...]]]


@dataclass
class EdgeWeights:
    by_index: dict[tuple[int, int], float]
    by_name: dict[tuple[str, str], float]


@dataclass
class GraphDocument:
    topology: Topology
    weights: EdgeWeights


def make_topology() -> Topology:
    return Topology(
        name="checkout",
        nodes=["api", "auth", "db", "queue"],
        kinds={"api": NodeKind.SERVICE, "auth": NodeKind.SERVICE, "db": NodeKind.DATABASE, "queue": NodeKind.QUEUE},
        adjacency={"api": {"queue", "auth", "db"}, "auth": {"db"}, "db": set(), "queue": {"db"}},
        adjacency_matrix=[[0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0]],
        distances=[[0.0, 1.0, 1.0, 1.0], [INF, 0.0, 1.0, INF], [INF, INF, 0.0, INF], [INF, INF, 1.0, 0.0]],
        allowed_edges={
            Relation.DEPENDS_ON: [NodeKind.SERVICE, NodeKind.DATABASE],
            Relation.PUBLISHES_TO: [NodeKind.QUEUE],
            Relation.CONFLICTS_WITH: [],
        },
        capacity={
            NodeKind.SERVICE: {Relation.DEPENDS_ON: (10, 20, 30), Relation.PUBLISHES_TO: ()},
            NodeKind.QUEUE: {Relation.PUBLISHES_TO: (1000,)},
        },
    )


def make_edge_weights() -> EdgeWeights:
    return EdgeWeights(
        by_index={(0, 1): 1.5, (0, 2): 0.25, (1, 2): 3.0, (3, 2): 7.0},
        by_name={("api", "auth"): 0.5, ("auth", "db"): 2.0},
    )


def make_graph_document() -> GraphDocument:
    return GraphDocument(make_topology(), make_edge_weights())


TOPOLOGY_JSON = {
    "name": "checkout",
    "nodes": ["api", "auth", "db", "queue"],
    "kinds": {"api": "service", "auth": "service", "db": "database", "queue": "queue"},
    "adjacency": {"api": ["auth", "db", "queue"], "auth": ["db"], "db": [], "queue": ["db"]},
    "adjacency_matrix": [[0, 1, 1, 1], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0]],
    "distances": [[0.0, 1.0, 1.0, 1.0], [INF, 0.0, 1.0, INF], [INF, INF, 0.0, INF], [INF, INF, 1.0, 0.0]],
    "allowed_edges": {"depends_on": ["service", "database"], "publishes_to": ["queue"], "conflicts_with": []},
    "capacity": {"service": {"depends_on": [10, 20, 30], "publishes_to": []}, "queue": {"publishes_to": [1000]}},
}

EDGE_WEIGHTS_JSON = {
    "by_index": {"[0, 1]": 1.5, "[0, 2]": 0.25, "[1, 2]": 3.0, "[3, 2]": 7.0},
    "by_name": {'["api", "auth"]': 0.5, '["auth", "db"]': 2.0},
}

GRAPH_JSON = {"topology": TOPOLOGY_JSON, "weights": EDGE_WEIGHTS_JSON}


def test_edge_weights_exact_text_has_json_texts_of_tuples_as_keys():
    assert serialize_json(make_edge_weights()) == (
        '{"by_index": {"[0, 1]": 1.5, "[0, 2]": 0.25, "[1, 2]": 3.0, "[3, 2]": 7.0}, '
        '"by_name": {"[\\"api\\", \\"auth\\"]": 0.5, "[\\"auth\\", \\"db\\"]": 2.0}}'
    )


def test_distance_matrix_text_spells_unreachable_as_infinity():
    topology = make_topology()
    text = serialize_json(topology.distances)

    assert text == (
        "[[0.0, 1.0, 1.0, 1.0], [Infinity, 0.0, 1.0, Infinity], "
        "[Infinity, Infinity, 0.0, Infinity], [Infinity, Infinity, 1.0, 0.0]]"
    )
    with pytest.raises(ValueError):
        serialize_json(topology.distances, allow_nan=False)


def test_topology_round_trips_with_enum_keys_sets_and_infinities():
    restored = assert_round_trip(make_topology(), Topology)

    assert restored.adjacency["api"] == {"auth", "db", "queue"} and restored.adjacency["db"] == set()
    assert restored.allowed_edges[Relation.CONFLICTS_WITH] == []
    assert restored.capacity[NodeKind.QUEUE][Relation.PUBLISHES_TO] == (1000,)
    assert restored.distances[3][0] == INF and restored.distances[3][3] == 0.0


def test_edge_weights_round_trip_restores_tuple_keys():
    restored = assert_round_trip(make_edge_weights(), EdgeWeights)

    assert restored.by_index[(3, 2)] == 7.0 and restored.by_name[("auth", "db")] == 2.0


def test_unparameterized_tuple_key_annotation_restores_tuple_keys():
    # `tuple` without arguments: the elements just are not typed
    weights = {(0, 1): 1.5, (3, 2): 7.0}

    restored = assert_round_trip(weights, dict[tuple, float])

    assert list(restored) == [(0, 1), (3, 2)]


def test_integer_node_ids_as_dict_keys_cannot_be_restored_limitation():
    # Documents current behavior, not a promise: "0" cannot be turned back into the int 0.
    adjacency = {0: {1, 2}, 1: {2}, 2: set()}
    parsed = to_json_value(adjacency)

    assert parsed == {"0": [1, 2], "1": [2], "2": []}
    with pytest.raises(ValueError, match="(?i)mapping key"):
        deserialize_json(parsed, dict[int, set[int]])


def test_graph_whose_int_and_string_node_ids_collide_is_rejected():
    # node 1 from one source and "1" from another would be one JSON key: refuse rather than merge the neighbours
    adjacency = {1: {2}, "1": {3}, 2: set(), 3: set()}

    with pytest.raises(ValueError, match="(?i)colli"):
        serialize_json(adjacency)


def test_grid_graph_of_one_hundred_nodes_round_trips_with_symmetric_neighbours():
    adjacency: dict[str, set[str]] = {}
    for row in range(10):
        for col in range(10):
            neighbours = set()
            for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= row + d_row < 10 and 0 <= col + d_col < 10:
                    neighbours.add(f"r{row + d_row}c{col + d_col}")
            adjacency[f"r{row}c{col}"] = neighbours

    parsed = to_json_value(adjacency)
    assert parsed["r0c0"] == ["r0c1", "r1c0"]
    assert parsed["r5c5"] == ["r4c5", "r5c4", "r5c6", "r6c5"]
    assert parsed["r9c9"] == ["r8c9", "r9c8"]
    assert sum(len(neighbours) for neighbours in parsed.values()) == 2 * 180
    assert all(node in parsed[other] for node, others in parsed.items() for other in others)
    assert_round_trip(adjacency, dict[str, set[str]])


TOPOLOGY_CORRUPTIONS = [
    pytest.param(("kinds", "db"), "cache", id="unknown-node-kind"),
    pytest.param(("adjacency", "api"), "auth", id="string-where-neighbour-set-expected"),
    pytest.param(("adjacency", "api"), {"auth": 1}, id="object-where-neighbour-set-expected"),
    pytest.param(("adjacency",), [["api", "auth"]], id="array-where-adjacency-expected"),
    pytest.param(("allowed_edges", "depends_on"), ["service", "cache"], id="unknown-kind-in-lookup-table"),
    pytest.param(("capacity", "queue", "publishes_to"), "1000", id="string-where-capacity-tuple-expected"),
    pytest.param(("capacity", "queue"), {"consumes": [1]}, id="unknown-relation-as-lookup-key"),
    pytest.param(("capacity", "disk"), {"depends_on": []}, id="unknown-kind-as-lookup-key"),
    pytest.param(("adjacency_matrix", 1), 0, id="number-where-matrix-row-expected"),
    pytest.param(("distances",), REMOVED, id="missing-distance-matrix"),
    pytest.param(("nodes",), "api,auth,db,queue", id="string-where-node-list-expected"),
]


@pytest.mark.parametrize(("path", "replacement"), TOPOLOGY_CORRUPTIONS)
def test_corrupted_topology_is_rejected(path, replacement):
    assert_rejected(TOPOLOGY_JSON, Topology, path, replacement)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        pytest.param(("by_index",), [[0, 1]], id="array-where-index-weights-expected"),
        pytest.param(("by_name",), [["api", "auth", 0.5]], id="array-where-named-weights-expected"),
    ],
)
def test_weight_table_given_an_array_is_rejected(path, replacement):
    broken = corrupted(EDGE_WEIGHTS_JSON, path, replacement)

    with pytest.raises(TypeError):
        deserialize_json(broken, EdgeWeights)


# ---------------------------------------------------------------------------------------------------------------------
# 7. machine-learning experiment record: nan and infinity in metrics
# ---------------------------------------------------------------------------------------------------------------------


class RunStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    DIVERGED = "diverged"


@dataclass
class HyperParameters:
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: Literal["sgd", "adam", "adamw"]
    betas: tuple[float, float]
    weight_decay: Optional[float]
    layer_sizes: list[int]
    augmentations: frozenset[str]
    seed: int


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    grad_norm: float
    accuracy: Optional[float]


@dataclass
class ExperimentRun:
    run_id: str
    status: RunStatus
    started_at: datetime
    finished_at: Optional[datetime]
    params: HyperParameters
    metrics: list[EpochMetrics]
    best_epoch: Optional[int]
    artifact_dir: Path
    checkpoints: dict[str, Path]
    tags: dict[str, str]


def make_diverged_run(path: Any = Path) -> ExperimentRun:
    return ExperimentRun(
        run_id="exp-0042-r3",
        status=RunStatus.DIVERGED,
        started_at=datetime(2026, 5, 4, 21, 15, tzinfo=UTC),
        finished_at=datetime(2026, 5, 4, 21, 47, 30, 500000, tzinfo=UTC),
        params=HyperParameters(
            learning_rate=0.001, batch_size=64, epochs=3, optimizer="adamw", betas=(0.9, 0.999),
            weight_decay=0.01, layer_sizes=[784, 256, 64, 10], augmentations=frozenset({"flip", "crop"}), seed=42,
        ),
        metrics=[
            EpochMetrics(1, 2.302, 2.298, 1.75, 0.113),
            EpochMetrics(2, 1.4, NAN, INF, None),
            EpochMetrics(3, NAN, -INF, NAN, None),
        ],
        best_epoch=1,
        artifact_dir=path("/data/experiments/exp-0042/r3"),
        checkpoints={"best": path("/data/experiments/exp-0042/r3/ckpt-001.pt"),
                     "last": path("/data/experiments/exp-0042/r3/ckpt-003.pt")},
        tags={"dataset": "mnist", "owner": "ron"},
    )


def make_running_run(path: Any = Path) -> ExperimentRun:
    return ExperimentRun(
        run_id="exp-0043-r1",
        status=RunStatus.RUNNING,
        started_at=datetime(2026, 5, 5, 9, 0, tzinfo=UTC),
        finished_at=None,
        params=HyperParameters(1e-05, 8, 100, "sgd", (0.9, 0.99), None, [], frozenset(), 0),
        metrics=[],
        best_epoch=None,
        artifact_dir=path("/data/experiments/exp-0043/r1"),
        checkpoints={},
        tags={},
    )


DIVERGED_RUN_JSON = {
    "run_id": "exp-0042-r3",
    "status": "diverged",
    "started_at": "2026-05-04T21:15:00+00:00",
    "finished_at": "2026-05-04T21:47:30.500000+00:00",
    "params": {
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 3,
        "optimizer": "adamw",
        "betas": [0.9, 0.999],
        "weight_decay": 0.01,
        "layer_sizes": [784, 256, 64, 10],
        "augmentations": ["crop", "flip"],
        "seed": 42,
    },
    "metrics": [
        {"epoch": 1, "train_loss": 2.302, "val_loss": 2.298, "grad_norm": 1.75, "accuracy": 0.113},
        {"epoch": 2, "train_loss": 1.4, "val_loss": NAN, "grad_norm": INF, "accuracy": None},
        {"epoch": 3, "train_loss": NAN, "val_loss": -INF, "grad_norm": NAN, "accuracy": None},
    ],
    "best_epoch": 1,
    "artifact_dir": "/data/experiments/exp-0042/r3",
    "checkpoints": {
        "best": "/data/experiments/exp-0042/r3/ckpt-001.pt",
        "last": "/data/experiments/exp-0042/r3/ckpt-003.pt",
    },
    "tags": {"dataset": "mnist", "owner": "ron"},
}

RUNNING_RUN_JSON = {
    "run_id": "exp-0043-r1",
    "status": "running",
    "started_at": "2026-05-05T09:00:00+00:00",
    "finished_at": None,
    "params": {
        "learning_rate": 1e-05,
        "batch_size": 8,
        "epochs": 100,
        "optimizer": "sgd",
        "betas": [0.9, 0.99],
        "weight_decay": None,
        "layer_sizes": [],
        "augmentations": [],
        "seed": 0,
    },
    "metrics": [],
    "best_epoch": None,
    "artifact_dir": "/data/experiments/exp-0043/r1",
    "checkpoints": {},
    "tags": {},
}

DIVERGED_METRICS_TEXT = (
    '[{"epoch": 1, "train_loss": 2.302, "val_loss": 2.298, "grad_norm": 1.75, "accuracy": 0.113}, '
    '{"epoch": 2, "train_loss": 1.4, "val_loss": NaN, "grad_norm": Infinity, "accuracy": null}, '
    '{"epoch": 3, "train_loss": NaN, "val_loss": -Infinity, "grad_norm": NaN, "accuracy": null}]'
)


def test_diverged_run_metrics_text_spells_nan_and_infinity_like_python_json():
    assert serialize_json(make_diverged_run(PurePosixPath).metrics) == DIVERGED_METRICS_TEXT


def test_diverged_run_cannot_be_written_when_nan_is_forbidden_but_a_healthy_one_can():
    run = make_diverged_run(PurePosixPath)

    with pytest.raises(ValueError):
        serialize_json(run, allow_nan=False)
    healthy = replace(run, metrics=run.metrics[:1])
    assert json.loads(serialize_json(healthy, allow_nan=False))["metrics"] == DIVERGED_RUN_JSON["metrics"][:1]


def test_nan_and_infinity_in_metrics_come_back_as_such():
    restored = deserialize_json(json.loads(DIVERGED_METRICS_TEXT), list[EpochMetrics])

    assert math.isnan(restored[1].val_loss) and restored[1].grad_norm == INF
    assert restored[2].val_loss == -INF and math.isnan(restored[2].train_loss) and math.isnan(restored[2].grad_norm)
    assert restored[1].accuracy is None and restored[0].accuracy == 0.113
    assert [type(metrics.grad_norm) for metrics in restored] == [float, float, float]


def test_hyperparameters_written_as_whole_numbers_are_restored_as_floats():
    params = {"learning_rate": 1, "batch_size": 8, "epochs": 1, "optimizer": "sgd", "betas": [1, 1],
              "weight_decay": 0, "layer_sizes": [4], "augmentations": [], "seed": 1}

    restored = deserialize_json(params, HyperParameters)

    assert_strictly_equal(restored, HyperParameters(1.0, 8, 1, "sgd", (1.0, 1.0), 0.0, [4], frozenset(), 1))


RUN_CORRUPTIONS = [
    pytest.param(("status",), "crashed", id="unknown-status"),
    pytest.param(("started_at",), "05/04/2026 21:15", id="locale-formatted-timestamp"),
    pytest.param(("finished_at",), "never", id="unparseable-optional-timestamp"),
    pytest.param(("params", "optimizer"), "rmsprop", id="unknown-optimizer-literal"),
    pytest.param(("params", "betas"), [0.9], id="one-beta"),
    pytest.param(("params", "betas"), [0.9, 0.999, 0.5], id="three-betas"),
    pytest.param(("params", "layer_sizes"), "784,256,64,10", id="string-where-layer-list-expected"),
    pytest.param(("params", "augmentations"), {"crop": 1}, id="object-where-augmentation-set-expected"),
    pytest.param(("params",), REMOVED, id="missing-hyperparameters"),
    pytest.param(("metrics", 1), [2, 1.4], id="array-where-epoch-metrics-expected"),
    pytest.param(("metrics",), "none", id="string-where-metric-list-expected"),
    pytest.param(("checkpoints",), ["ckpt-001.pt"], id="array-where-checkpoint-table-expected"),
    pytest.param(("artifact_dir",), {"path": "/data"}, id="object-where-path-expected"),
    pytest.param(("tags",), [["dataset", "mnist"]], id="array-where-tag-table-expected"),
]


@posix_only
@pytest.mark.parametrize(("path", "replacement"), RUN_CORRUPTIONS)
def test_corrupted_experiment_run_is_rejected(path, replacement):
    assert_rejected(DIVERGED_RUN_JSON, ExperimentRun, path, replacement)


@dataclass
class TrainingArrays:
    run_id: str
    loss_curve: np.ndarray
    confusion: np.ndarray
    embeddings: np.ndarray
    best_loss: np.float32
    step: np.int64
    converged: np.bool_
    holdout: Optional[np.ndarray] = None


def make_training_arrays() -> TrainingArrays:
    return TrainingArrays(
        run_id="run-17",
        loss_curve=np.array([2.302, 1.4, np.nan, np.inf]),
        confusion=np.array([[50, 2], [3, 45]]),
        embeddings=np.array([[0.5, -0.25, 0.125], [1.0, 0.0, -1.0]]),
        best_loss=np.float32(0.5),
        step=np.int64(1200),
        converged=np.bool_(False),
        holdout=np.array([0.25, 0.75]),
    )


TRAINING_ARRAYS_JSON = {
    "run_id": "run-17",
    "loss_curve": [2.302, 1.4, NAN, INF],
    "confusion": [[50, 2], [3, 45]],
    "embeddings": [[0.5, -0.25, 0.125], [1.0, 0.0, -1.0]],
    "best_loss": 0.5,
    "step": 1200,
    "converged": False,
    "holdout": [0.25, 0.75],
}


def test_numpy_training_arrays_serialize_as_plain_json_values():
    pytest.importorskip("numpy")

    assert_same_json(to_json_value(make_training_arrays()), TRAINING_ARRAYS_JSON)
    assert "NaN" in serialize_json(make_training_arrays()) and "Infinity" in serialize_json(make_training_arrays())


def test_numpy_training_arrays_round_trip_with_dtypes_and_scalar_types():
    pytest.importorskip("numpy")

    restored = assert_round_trip(make_training_arrays(), TrainingArrays)

    assert restored.loss_curve.dtype == np.float64 and restored.confusion.shape == (2, 2)
    assert type(restored.best_loss) is np.float32 and type(restored.step) is np.int64
    assert type(restored.converged) is np.bool_


def test_numpy_hand_written_json_restores_the_training_arrays():
    pytest.importorskip("numpy")

    restored = deserialize_json(copy.deepcopy(TRAINING_ARRAYS_JSON), TrainingArrays)

    assert_strictly_equal(restored, make_training_arrays())


def test_numpy_empty_holdout_array_is_not_none():
    pytest.importorskip("numpy")
    arrays = replace(make_training_arrays(), holdout=np.array([]))

    assert to_json_value(arrays)["holdout"] == []
    restored = assert_round_trip(arrays, TrainingArrays)
    assert restored.holdout is not None and restored.holdout.shape == (0,)


@pytest.mark.parametrize(
    "annotation",
    [
        pytest.param("ndarray", id="ndarray"),
        pytest.param("float64", id="float64"),
        pytest.param("bool_", id="bool_"),
        pytest.param("int64", id="int64"),
    ],
)
def test_unset_optional_numpy_field_restores_as_none(annotation):
    pytest.importorskip("numpy")

    restored = deserialize_json(None, Optional[getattr(np, annotation)])

    assert restored is None


def test_training_arrays_without_a_holdout_set_are_written_with_null():
    pytest.importorskip("numpy")

    assert to_json_value(replace(make_training_arrays(), holdout=None))["holdout"] is None


def test_training_arrays_without_a_holdout_set_round_trip():
    pytest.importorskip("numpy")

    assert_round_trip(replace(make_training_arrays(), holdout=None), TrainingArrays)


def test_ragged_nested_list_for_an_array_field_is_rejected_with_a_type_error():
    np = pytest.importorskip("numpy")
    if np.lib.NumpyVersion(np.__version__) < "1.24.0":
        pytest.skip("older numpy builds an object array from ragged lists")
    broken = corrupted(TRAINING_ARRAYS_JSON, ("confusion",), [[50, 2], [3]])

    with pytest.raises(TypeError):
        deserialize_json(broken, TrainingArrays)


def test_array_field_given_a_string_or_an_object_is_not_split_into_an_array_limitation():
    # Documents current behavior, not a promise: np.array() accepts anything, so scalars and strings become 0-d arrays.
    pytest.importorskip("numpy")

    restored = deserialize_json(corrupted(TRAINING_ARRAYS_JSON, ("loss_curve",), "diverged"), TrainingArrays)

    assert restored.loss_curve.shape == () and restored.loss_curve.dtype.kind == "U"


def test_float32_array_comes_back_as_float64_limitation():
    # Documents current behavior, not a promise: the dtype is not part of the JSON.
    pytest.importorskip("numpy")
    arrays = replace(make_training_arrays(), holdout=np.array([0.5, 1.5], dtype=np.float32))

    restored = deserialize_json(to_json_value(arrays), TrainingArrays)

    assert restored.holdout.dtype == np.float64 and restored.holdout.tolist() == [0.5, 1.5]


# ---------------------------------------------------------------------------------------------------------------------
# 8. international content
# ---------------------------------------------------------------------------------------------------------------------


class Greeting(Enum):
    HEBREW = "שלום"
    ARABIC = "مرحبا"
    EMOJI = "👋"
    PLAIN = "hello"


@dataclass
class Review:
    author: str
    rating: int
    text: str


@dataclass
class LocalizedProduct:
    sku: str
    names: dict[str, str]
    attributes: dict[str, str]
    keywords: frozenset[str]
    greetings: dict[Greeting, int]
    reviews: list[Review]


def make_localized_product() -> LocalizedProduct:
    return LocalizedProduct(
        sku="OIL-001",
        names={"en": "Olive oil", "he": "שמן זית", "ar": "زيت الزيتون", "ja": "オリーブオイル", "ru": "Оливковое масло"},
        attributes={"צבע": "ירוק", "origin": "Galilee \u2764", "caf\u00e9": "composed", "cafe\u0301": "decomposed"},
        keywords=frozenset({"שמן", "زيت", "オリーブ"}),
        greetings={Greeting.HEBREW: 120, Greeting.ARABIC: 80, Greeting.EMOJI: 7, Greeting.PLAIN: 300},
        reviews=[
            Review("דנה", 5, "מצוין \U0001F389"),
            Review("Ahmad أحمد", 4, "جيد جدا"),
            Review("田中", 3, "普通です\nちょっと高い"),
            Review("", 1, ""),
        ],
    )


LOCALIZED_PRODUCT_JSON = {
    "sku": "OIL-001",
    "names": {"en": "Olive oil", "he": "שמן זית", "ar": "زيت الزيتون", "ja": "オリーブオイル", "ru": "Оливковое масло"},
    "attributes": {"צבע": "ירוק", "origin": "Galilee \u2764", "caf\u00e9": "composed", "cafe\u0301": "decomposed"},
    "keywords": ["שמן", "زيت", "オリーブ"],
    "greetings": {"שלום": 120, "مرحبا": 80, "👋": 7, "hello": 300},
    "reviews": [
        {"author": "דנה", "rating": 5, "text": "מצוין \U0001F389"},
        {"author": "Ahmad أحمد", "rating": 4, "text": "جيد جدا"},
        {"author": "田中", "rating": 3, "text": "普通です\nちょっと高い"},
        {"author": "", "rating": 1, "text": ""},
    ],
}

SMALL_PRODUCT = LocalizedProduct(
    sku="OIL-001",
    names={"en": "Olive oil", "he": "שמן"},
    attributes={"צבע": "ירוק"},
    keywords=frozenset({"زيت", "שמן"}),
    greetings={},
    reviews=[Review("דנה", 5, "מצוין \U0001F389")],
)

SMALL_PRODUCT_ASCII_TEXT = (
    r'{"sku": "OIL-001", "names": {"en": "Olive oil", "he": "\u05e9\u05de\u05df"}, '
    r'"attributes": {"\u05e6\u05d1\u05e2": "\u05d9\u05e8\u05d5\u05e7"}, '
    r'"keywords": ["\u05e9\u05de\u05df", "\u0632\u064a\u062a"], "greetings": {}, '
    r'"reviews": [{"author": "\u05d3\u05e0\u05d4", "rating": 5, '
    r'"text": "\u05de\u05e6\u05d5\u05d9\u05df \ud83c\udf89"}]}'
)

SMALL_PRODUCT_UNICODE_TEXT = (
    '{"sku": "OIL-001", "names": {"en": "Olive oil", "he": "שמן"}, '
    '"attributes": {"צבע": "ירוק"}, '
    '"keywords": ["שמן", "زيت"], "greetings": {}, '
    '"reviews": [{"author": "דנה", "rating": 5, "text": "מצוין \U0001F389"}]}'
)


def test_hebrew_arabic_and_emoji_ascii_escaped_golden():
    assert serialize_json(SMALL_PRODUCT) == SMALL_PRODUCT_ASCII_TEXT


def test_hebrew_arabic_and_emoji_unicode_golden():
    assert serialize_json(SMALL_PRODUCT, ensure_ascii=False) == SMALL_PRODUCT_UNICODE_TEXT


def test_ascii_escaped_text_is_pure_ascii_and_unicode_text_is_not():
    assert serialize_json(SMALL_PRODUCT).isascii()
    unicode_text = serialize_json(SMALL_PRODUCT, ensure_ascii=False)
    assert not unicode_text.isascii() and "שמן" in unicode_text and "\U0001F389" in unicode_text


@pytest.mark.parametrize("ensure_ascii", [True, False], ids=["ascii-escaped", "unicode"])
def test_both_encodings_round_trip_the_localized_product(ensure_ascii):
    restored = assert_round_trip(make_localized_product(), LocalizedProduct, ensure_ascii=ensure_ascii)

    assert restored.attributes["caf\u00e9"] == "composed" and restored.attributes["cafe\u0301"] == "decomposed"
    assert len(restored.attributes) == 4
    assert restored.greetings[Greeting.EMOJI] == 7 and restored.reviews[3] == Review("", 1, "")


def test_localized_product_survives_a_utf8_file():
    text = serialize_json(make_localized_product(), ensure_ascii=False)

    assert "\\u" not in text  # no control characters in this document, so nothing needs a \u escape
    data = json.loads(text.encode("utf-8"))
    assert_strictly_equal(deserialize_json(data, LocalizedProduct), make_localized_product())


ESCAPING_CASES = [
    pytest.param("caf\u00e9", r'"caf\u00e9"', '"caf\u00e9"', id="composed-accent"),
    pytest.param("cafe\u0301", r'"cafe\u0301"', '"cafe\u0301"', id="decomposed-accent"),
    pytest.param("\U0001F468\u200d\U0001F469\u200d\U0001F467", r'"\ud83d\udc68\u200d\ud83d\udc69\u200d\ud83d\udc67"',
                 '"\U0001F468\u200d\U0001F469\u200d\U0001F467"', id="family-emoji-with-joiners"),
    pytest.param("\u200fשלום\u200f", r'"\u200f\u05e9\u05dc\u05d5\u05dd\u200f"',
                 '"\u200f\u05e9\u05dc\u05d5\u05dd\u200f"', id="right-to-left-marks"),
    pytest.param("a\x00b\x1fc", r'"a\u0000b\u001fc"', r'"a\u0000b\u001fc"', id="control-characters"),
    pytest.param('tab\there "quoted" back\\slash\nnew', r'"tab\there \"quoted\" back\\slash\nnew"',
                 r'"tab\there \"quoted\" back\\slash\nnew"', id="json-escapes"),
    pytest.param("", '""', '""', id="empty"),
    pytest.param("\u20aa12.50", r'"\u20aa12.50"', '"\u20aa12.50"', id="shekel-sign"),
    pytest.param("\u2028\u2029", r'"\u2028\u2029"', '"\u2028\u2029"', id="line-and-paragraph-separators"),
]


@pytest.mark.parametrize(("value", "ascii_text", "unicode_text"), ESCAPING_CASES)
def test_awkward_strings_in_nested_positions_have_the_expected_escapes(value, ascii_text, unicode_text):
    assert serialize_json(value) == ascii_text
    assert serialize_json(value, ensure_ascii=False) == unicode_text
    assert serialize_json({value: [value]}) == "{" + ascii_text + ": [" + ascii_text + "]}"

    document = {value: Review(value, 1, value)}
    for options in ({}, {"ensure_ascii": False}):
        assert_round_trip(document, dict[str, Review], **options)


def test_large_multilingual_review_list_round_trips_in_both_encodings():
    samples = ["שלום עולם", "مرحبا بالعالم", "Привет, мир", "こんにちは世界", "hello \U0001F600", "", "plain ascii"]
    reviews = [Review(f"{samples[index % 7]} #{index}", index % 6, samples[(index * 3) % 7] * (index % 4 + 1))
               for index in range(400)]
    product = replace(SMALL_PRODUCT, reviews=reviews)

    for ensure_ascii in (True, False):
        assert_round_trip(product, LocalizedProduct, ensure_ascii=ensure_ascii)
    # index 8 picks the second sample (8 % 7 == 1)
    assert to_json_value(product)["reviews"][8]["author"] == "مرحبا بالعالم #8"


def test_set_order_of_mixed_scripts_is_independent_of_insertion_order_and_of_ensure_ascii():
    words = ["zeta", "ß", "שלום", "\U0001F600", "alpha", "مرحبا", "e\u0301"]

    forward = frozenset(words)
    backward = frozenset(reversed(words))
    ascii_text = serialize_json(forward)

    assert serialize_json(backward) == ascii_text
    assert json.loads(serialize_json(forward, ensure_ascii=False)) == json.loads(ascii_text)
    assert sorted(json.loads(ascii_text)) == sorted(words) and len(json.loads(ascii_text)) == 7


def test_set_of_one_script_is_ordered_by_code_point():
    assert to_json_value(frozenset({"בגד", "אבג", "אא", "א"})) == ["א", "אא", "אבג", "בגד"]
    assert to_json_value({"Charlie", "alpha", "Bravo"}) == ["Bravo", "Charlie", "alpha"]


I18N_CORRUPTIONS = [
    pytest.param(("names",), ["en", "he"], id="array-where-name-table-expected"),
    pytest.param(("keywords",), "שמן", id="string-where-keyword-set-expected"),
    pytest.param(("reviews", 0), "דנה", id="string-where-review-expected"),
    pytest.param(("reviews",), REMOVED, id="missing-reviews"),
    pytest.param(("greetings", "bonjour"), 1, id="unknown-enum-key"),
    pytest.param(("greetings",), [["שלום", 1]], id="array-where-greeting-table-expected"),
]


@pytest.mark.parametrize(("path", "replacement"), I18N_CORRUPTIONS)
def test_corrupted_localized_product_is_rejected(path, replacement):
    assert_rejected(LOCALIZED_PRODUCT_JSON, LocalizedProduct, path, replacement)


# ---------------------------------------------------------------------------------------------------------------------
# 9. workflow definition / state machine
# ---------------------------------------------------------------------------------------------------------------------


class DocState(Enum):
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class Action(Enum):
    SUBMIT = "submit"
    APPROVE = "approve"
    REJECT = "reject"
    PUBLISH = "publish"
    REOPEN = "reopen"
    ARCHIVE = "archive"
    EXPIRE = "expire"


class Priority(IntEnum):
    LOW = 1
    NORMAL = 2
    URGENT = 3


GuardName = Literal["always", "has_reviewer", "all_checks_passed", "manager_approved"]


@dataclass
class Transition:
    action: Action
    target: DocState
    guard: GuardName
    timeout_seconds: Optional[int] = None
    on_timeout: Optional[DocState] = None


@dataclass
class WorkflowDefinition:
    name: str
    version: int
    initial: DocState
    terminal: frozenset[DocState]
    reachable: dict[DocState, tuple[DocState, ...]]
    rules: dict[DocState, list[Transition]]
    sla_hours: dict[DocState, int]
    escalation: dict[DocState, Priority]
    history: list[tuple[DocState, Action, DocState]]


def make_workflow() -> WorkflowDefinition:
    S, A = DocState, Action
    return WorkflowDefinition(
        name="document-approval",
        version=3,
        initial=S.DRAFT,
        terminal=frozenset({S.ARCHIVED, S.REJECTED}),
        reachable={
            S.DRAFT: (S.IN_REVIEW,),
            S.IN_REVIEW: (S.APPROVED, S.REJECTED, S.DRAFT),
            S.APPROVED: (S.PUBLISHED, S.DRAFT),
            S.PUBLISHED: (S.ARCHIVED,),
            S.REJECTED: (S.DRAFT, S.ARCHIVED),
            S.ARCHIVED: (),
        },
        rules={
            S.DRAFT: [Transition(A.SUBMIT, S.IN_REVIEW, "has_reviewer")],
            S.IN_REVIEW: [
                Transition(A.APPROVE, S.APPROVED, "all_checks_passed"),
                Transition(A.REJECT, S.REJECTED, "always"),
                Transition(A.EXPIRE, S.REJECTED, "always", timeout_seconds=172800, on_timeout=S.REJECTED),
            ],
            S.APPROVED: [Transition(A.PUBLISH, S.PUBLISHED, "manager_approved", 604800, S.DRAFT)],
            S.PUBLISHED: [Transition(A.ARCHIVE, S.ARCHIVED, "always", 31536000, S.ARCHIVED)],
            S.REJECTED: [Transition(A.REOPEN, S.DRAFT, "always"), Transition(A.ARCHIVE, S.ARCHIVED, "always")],
            S.ARCHIVED: [],
        },
        sla_hours={S.IN_REVIEW: 48, S.APPROVED: 168},
        escalation={S.IN_REVIEW: Priority.URGENT, S.DRAFT: Priority.LOW},
        history=[
            (S.DRAFT, A.SUBMIT, S.IN_REVIEW),
            (S.IN_REVIEW, A.REJECT, S.REJECTED),
            (S.REJECTED, A.REOPEN, S.DRAFT),
        ],
    )


WORKFLOW_JSON = {
    "name": "document-approval",
    "version": 3,
    "initial": "draft",
    "terminal": ["archived", "rejected"],
    "reachable": {
        "draft": ["in_review"],
        "in_review": ["approved", "rejected", "draft"],
        "approved": ["published", "draft"],
        "published": ["archived"],
        "rejected": ["draft", "archived"],
        "archived": [],
    },
    "rules": {
        "draft": [
            {"action": "submit", "target": "in_review", "guard": "has_reviewer", "timeout_seconds": None,
             "on_timeout": None},
        ],
        "in_review": [
            {"action": "approve", "target": "approved", "guard": "all_checks_passed", "timeout_seconds": None,
             "on_timeout": None},
            {"action": "reject", "target": "rejected", "guard": "always", "timeout_seconds": None,
             "on_timeout": None},
            {"action": "expire", "target": "rejected", "guard": "always", "timeout_seconds": 172800,
             "on_timeout": "rejected"},
        ],
        "approved": [
            {"action": "publish", "target": "published", "guard": "manager_approved", "timeout_seconds": 604800,
             "on_timeout": "draft"},
        ],
        "published": [
            {"action": "archive", "target": "archived", "guard": "always", "timeout_seconds": 31536000,
             "on_timeout": "archived"},
        ],
        "rejected": [
            {"action": "reopen", "target": "draft", "guard": "always", "timeout_seconds": None, "on_timeout": None},
            {"action": "archive", "target": "archived", "guard": "always", "timeout_seconds": None,
             "on_timeout": None},
        ],
        "archived": [],
    },
    "sla_hours": {"in_review": 48, "approved": 168},
    "escalation": {"in_review": 3, "draft": 1},
    "history": [["draft", "submit", "in_review"], ["in_review", "reject", "rejected"], ["rejected", "reopen", "draft"]],
}


def test_workflow_restores_enum_keys_tuples_and_int_enums():
    restored = deserialize_json(copy.deepcopy(WORKFLOW_JSON), WorkflowDefinition)

    assert restored.reachable[DocState.IN_REVIEW] == (DocState.APPROVED, DocState.REJECTED, DocState.DRAFT)
    assert restored.reachable[DocState.ARCHIVED] == ()
    assert restored.escalation[DocState.IN_REVIEW] is Priority.URGENT
    assert type(restored.history[0]) is tuple
    assert restored.history[2] == (DocState.REJECTED, Action.REOPEN, DocState.DRAFT)
    assert restored.rules[DocState.IN_REVIEW][2].on_timeout is DocState.REJECTED
    assert restored.terminal == frozenset({DocState.ARCHIVED, DocState.REJECTED})


def test_workflow_text_lists_enum_keys_in_insertion_order_and_int_enums_as_numbers():
    text = serialize_json(make_workflow().escalation)

    assert text == '{"in_review": 3, "draft": 1}'


WORKFLOW_CORRUPTIONS = [
    pytest.param(("initial",), "started", id="unknown-initial-state"),
    pytest.param(("terminal",), ["archived", "deleted"], id="unknown-terminal-state"),
    pytest.param(("terminal",), "archived", id="string-where-terminal-set-expected"),
    pytest.param(("reachable", "published"), "archived", id="string-where-target-tuple-expected"),
    pytest.param(("reachable", "published", 0), "gone", id="unknown-target-state"),
    pytest.param(("reachable", "limbo"), ["draft"], id="unknown-state-as-key"),
    pytest.param(("rules", "draft", 0, "guard"), "sometimes", id="unknown-guard-literal"),
    pytest.param(("rules", "in_review", 1, "action"), "veto", id="unknown-action"),
    pytest.param(("rules", "in_review", 2, "on_timeout"), "limbo", id="unknown-timeout-state"),
    pytest.param(("rules", "approved"), {"action": "publish"}, id="object-where-rule-list-expected"),
    pytest.param(("rules", "approved", 0), "publish", id="string-where-rule-expected"),
    pytest.param(("escalation", "draft"), 9, id="unknown-priority-number"),
    pytest.param(("history", 1), ["in_review", "reject"], id="two-element-history-entry"),
    pytest.param(("history", 2), ["rejected", "reopen", "draft", "extra"], id="four-element-history-entry"),
    pytest.param(("history",), REMOVED, id="missing-history"),
]


@pytest.mark.parametrize(("path", "replacement"), WORKFLOW_CORRUPTIONS)
def test_corrupted_workflow_is_rejected(path, replacement):
    assert_rejected(WORKFLOW_JSON, WorkflowDefinition, path, replacement)


Step = Enum("Step", {f"S{index:02d}": f"step-{index:02d}" for index in range(30)})


@dataclass
class Pipeline:
    name: str
    first: Step
    next_step: dict[Step, Optional[Step]]
    retries: dict[Step, Literal[0, 1, 3]]


def make_pipeline() -> Pipeline:
    steps = list(Step)
    return Pipeline(
        name="nightly-etl",
        first=steps[0],
        next_step=dict(zip(steps, steps[1:] + [None])),
        retries={step: 3 if index % 10 == 0 else 1 if index % 2 == 0 else 0 for index, step in enumerate(steps)},
    )


def test_thirty_step_pipeline_serializes_in_step_order_with_a_null_tail():
    parsed = to_json_value(make_pipeline())

    assert parsed["first"] == "step-00"
    assert list(parsed["next_step"])[:3] == ["step-00", "step-01", "step-02"] and len(parsed["next_step"]) == 30
    assert parsed["next_step"]["step-07"] == "step-08" and parsed["next_step"]["step-29"] is None
    assert [parsed["retries"][f"step-{index:02d}"] for index in (0, 10, 11, 12, 20, 29)] == [3, 3, 0, 1, 3, 0]


def test_thirty_step_pipeline_round_trips_with_functional_enum_keys():
    restored = assert_round_trip(make_pipeline(), Pipeline)

    assert restored.next_step[Step.S29] is None and restored.next_step[Step.S07] is Step.S08
    assert type(restored.retries[Step.S10]) is int


PIPELINE_CORRUPTIONS = [
    pytest.param(("retries", "step-04"), 2, id="retry-count-not-in-the-literal"),
    pytest.param(("retries", "step-04"), True, id="bool-for-the-int-literal-1"),
    pytest.param(("retries", "step-04"), 1.0, id="float-for-the-int-literal-1"),
    pytest.param(("retries", "step-04"), "1", id="string-for-the-int-literal-1"),
    pytest.param(("retries", "step-04"), None, id="null-retry-count"),
    pytest.param(("next_step", "step-03"), "step-30", id="next-step-that-does-not-exist"),
    pytest.param(("next_step", "step-30"), "step-01", id="step-key-that-does-not-exist"),
    pytest.param(("first",), "step-99", id="first-step-that-does-not-exist"),
]


@pytest.mark.parametrize(("path", "replacement"), PIPELINE_CORRUPTIONS)
def test_corrupted_pipeline_is_rejected(path, replacement):
    # `retries` is a Literal[0, 1, 3] of ints: the type of the value counts, not only its numeric value
    assert_rejected(to_json_value(make_pipeline()), Pipeline, path, replacement)


@dataclass
class SlaTable:
    hours: dict[Priority, int]


def make_sla_table() -> SlaTable:
    return SlaTable({Priority.LOW: 72, Priority.NORMAL: 24, Priority.URGENT: 4})


def test_dict_keyed_by_an_int_enum_is_written_with_the_member_values_as_keys():
    assert to_json_value(make_sla_table()) == {"hours": {"1": 72, "2": 24, "3": 4}}


def test_dict_keyed_by_an_int_enum_round_trips():
    assert_round_trip(make_sla_table(), SlaTable)


class Role(str, Enum):
    """The enum style many code bases use: a member is a str, equal to its value."""

    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"


@dataclass
class AccessPolicy:
    default_role: Role
    grants: dict[Role, frozenset[str]]
    escalation: list[tuple[Role, Role]]
    allowed: frozenset[Role]
    users: dict[str, Role]


def make_access_policy() -> AccessPolicy:
    return AccessPolicy(
        default_role=Role.VIEWER,
        grants={Role.ADMIN: frozenset({"write", "read", "delete"}), Role.VIEWER: frozenset({"read"}),
                Role.EDITOR: frozenset()},
        escalation=[(Role.VIEWER, Role.EDITOR), (Role.EDITOR, Role.ADMIN)],
        allowed=frozenset({Role.EDITOR, Role.ADMIN}),
        users={"dana": Role.ADMIN, "omer": Role.VIEWER},
    )


ACCESS_POLICY_JSON = {
    "default_role": "viewer",
    "grants": {"admin": ["delete", "read", "write"], "viewer": ["read"], "editor": []},
    "escalation": [["viewer", "editor"], ["editor", "admin"]],
    "allowed": ["admin", "editor"],
    "users": {"dana": "admin", "omer": "viewer"},
}


def test_str_enum_members_are_written_as_their_values_as_keys_items_and_values():
    assert serialize_json(make_access_policy(), separators=(",", ":")) == (
        '{"default_role":"viewer","grants":{"admin":["delete","read","write"],"viewer":["read"],"editor":[]},'
        '"escalation":[["viewer","editor"],["editor","admin"]],"allowed":["admin","editor"],'
        '"users":{"dana":"admin","omer":"viewer"}}'
    )


def test_str_enum_members_come_back_as_members_not_as_plain_strings():
    restored = assert_round_trip(make_access_policy(), AccessPolicy, sort_keys=True)

    assert all(type(role) is Role for role in restored.grants) and all(type(role) is Role for role in restored.allowed)
    assert restored.escalation[0] == (Role.VIEWER, Role.EDITOR) and type(restored.escalation[0][1]) is Role
    assert restored.users["dana"] is Role.ADMIN


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        pytest.param(("default_role",), "superuser", id="unknown-role"),
        pytest.param(("default_role",), "VIEWER", id="role-written-as-the-member-name"),
        pytest.param(("grants", "root"), ["read"], id="unknown-role-as-key"),
        pytest.param(("escalation", 1), ["editor"], id="one-element-escalation-pair"),
        pytest.param(("allowed",), ["admin", "owner"], id="unknown-role-in-the-allowed-set"),
        pytest.param(("users", "omer"), "ADMIN", id="user-role-written-as-the-member-name"),
        pytest.param(("grants", "viewer"), "read", id="string-where-permission-set-expected"),
    ],
)
def test_corrupted_access_policy_is_rejected(path, replacement):
    assert_rejected(ACCESS_POLICY_JSON, AccessPolicy, path, replacement)


# ---------------------------------------------------------------------------------------------------------------------
# 10. versioned API payloads: older producers omit fields, newer ones add them
# ---------------------------------------------------------------------------------------------------------------------


class Plan(Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class Profile:
    bio: Optional[str] = None
    website: Optional[str] = None
    interests: Optional[frozenset[str]] = None


@dataclass
class UserRecord:
    id: int
    name: str
    email: Optional[str] = None
    plan: Optional[Plan] = None
    created_at: Optional[datetime] = None
    address: Optional[Address] = None
    roles: Optional[list[str]] = None
    profile: Optional[Profile] = None


@dataclass
class ListUsersResponse:
    api_version: str
    users: list[UserRecord]
    next_cursor: Optional[str] = None
    meta: Optional[dict[str, Any]] = None


def make_api_response() -> ListUsersResponse:
    return ListUsersResponse(
        api_version="3",
        users=[
            UserRecord(1, "Dana", "dana@example.com", Plan.PRO, datetime(2025, 11, 2, 10, 0, tzinfo=UTC),
                       Address("12 Herzl St", "Haifa", "3303310", "IL", "4B"), ["admin", "billing"],
                       Profile("Cat person", "https://dana.example.com", frozenset({"hiking", "cats"}))),
            UserRecord(2, "Omer", None, Plan.FREE, datetime(2026, 1, 15, 8, 30, tzinfo=UTC), None, [], None),
            UserRecord(3, "Noa", "noa@example.com", Plan.ENTERPRISE, None, None, None, Profile()),
        ],
        next_cursor="c4",
        meta={"total": 3, "took_ms": 12.5, "filters": {"plan": ["pro", "free"], "active": True}},
    )


API_V3_JSON = {
    "api_version": "3",
    "users": [
        {
            "id": 1,
            "name": "Dana",
            "email": "dana@example.com",
            "plan": "pro",
            "created_at": "2025-11-02T10:00:00+00:00",
            "address": {"street": "12 Herzl St", "city": "Haifa", "postal_code": "3303310", "country": "IL",
                        "apartment": "4B"},
            "roles": ["admin", "billing"],
            "profile": {"bio": "Cat person", "website": "https://dana.example.com",
                        "interests": ["cats", "hiking"]},
        },
        {
            "id": 2,
            "name": "Omer",
            "email": None,
            "plan": "free",
            "created_at": "2026-01-15T08:30:00+00:00",
            "address": None,
            "roles": [],
            "profile": None,
        },
        {
            "id": 3,
            "name": "Noa",
            "email": "noa@example.com",
            "plan": "enterprise",
            "created_at": None,
            "address": None,
            "roles": None,
            "profile": {"bio": None, "website": None, "interests": None},
        },
    ],
    "next_cursor": "c4",
    "meta": {"total": 3, "took_ms": 12.5, "filters": {"plan": ["pro", "free"], "active": True}},
}

# what the first API version sent: no optional field at all
API_V1_JSON = {"api_version": "1", "users": [{"id": 1, "name": "Dana"}, {"id": 2, "name": "Omer"}]}

# the second version added a few user fields and paging
API_V2_JSON = {
    "api_version": "2",
    "users": [
        {"id": 1, "name": "Dana", "email": "dana@example.com", "plan": "pro",
         "created_at": "2025-11-02T10:00:00+00:00"},
        {"id": 2, "name": "Omer", "email": None, "plan": "free", "created_at": "2026-01-15T08:30:00+00:00"},
    ],
    "next_cursor": "c2",
}


def test_v1_payload_restores_every_missing_optional_field_as_none():
    restored = deserialize_json(copy.deepcopy(API_V1_JSON), ListUsersResponse)

    expected = ListUsersResponse("1", [UserRecord(1, "Dana"), UserRecord(2, "Omer")])
    assert_strictly_equal(restored, expected)
    assert restored.next_cursor is None and restored.meta is None
    assert restored.users[0].roles is None and restored.users[0].profile is None


def test_v2_payload_restores_the_fields_it_has_and_none_for_the_rest():
    restored = deserialize_json(copy.deepcopy(API_V2_JSON), ListUsersResponse)

    expected = ListUsersResponse(
        "2",
        [UserRecord(1, "Dana", "dana@example.com", Plan.PRO, datetime(2025, 11, 2, 10, 0, tzinfo=UTC)),
         UserRecord(2, "Omer", None, Plan.FREE, datetime(2026, 1, 15, 8, 30, tzinfo=UTC))],
        next_cursor="c2",
    )
    assert_strictly_equal(restored, expected)


def test_a_restored_old_payload_is_written_with_explicit_nulls_for_the_new_fields():
    restored = deserialize_json(copy.deepcopy(API_V1_JSON), ListUsersResponse)

    assert serialize_json(restored.users[0]) == (
        '{"id": 1, "name": "Dana", "email": null, "plan": null, "created_at": null, '
        '"address": null, "roles": null, "profile": null}'
    )
    assert_round_trip(restored, ListUsersResponse)


def test_payload_from_a_newer_producer_loses_the_fields_this_version_does_not_know():
    newer = copy.deepcopy(API_V3_JSON)
    newer["rate_limit"] = {"remaining": 99, "reset": "2026-03-01T00:00:00+00:00"}
    newer["users"][0]["avatar_url"] = "https://cdn.example.com/dana.png"
    newer["users"][0]["address"]["geo"] = {"lat": 32.79, "lon": 34.99}
    newer["users"][0]["profile"]["pronouns"] = "she/her"
    newer["users"][1]["flags"] = ["beta"]
    newer["users"][2]["profile"]["theme"] = {"dark": True}

    restored = deserialize_json(newer, ListUsersResponse)

    assert_strictly_equal(restored, make_api_response())
    assert_same_json(to_json_value(restored), API_V3_JSON)


def test_unknown_values_of_a_known_enum_from_a_newer_producer_are_rejected():
    newer = corrupted(API_V3_JSON, ("users", 2, "plan"), "enterprise_plus")

    with pytest.raises(TypeError):
        deserialize_json(newer, ListUsersResponse)


def test_free_form_meta_keeps_whatever_the_server_sent():
    meta = {"total": 3, "cursor": None, "nested": {"list": [1, [2, {"three": 3.5}]], "flag": False}, "empty": {}}
    document = {**API_V1_JSON, "meta": meta}

    restored = deserialize_json(document, ListUsersResponse)

    assert_same_json(restored.meta, meta)
    assert restored.meta is not meta and restored.meta["nested"] is not meta["nested"]


def test_64_bit_and_larger_integer_ids_are_written_and_restored_exactly():
    users = [UserRecord(2**63 - 1, "max"), UserRecord(-(2**63), "min"), UserRecord(10**30, "huge")]

    text = serialize_json(users, separators=(",", ":"))

    assert '"id":9223372036854775807,' in text and '"id":-9223372036854775808,' in text
    assert f'"id":{10**30},' in text
    assert_round_trip(users, list[UserRecord])


API_CORRUPTIONS = [
    pytest.param(("users", 1, "plan"), "platinum", id="unknown-plan"),
    pytest.param(("users", 0, "created_at"), "2026-02-30T00:00:00+00:00", id="february-30th"),
    pytest.param(("users", 0, "roles"), "admin", id="string-where-role-list-expected"),
    pytest.param(("users", 0, "address"), "12 Herzl St, Haifa", id="string-where-address-expected"),
    pytest.param(("users", 0, "profile", "interests"), "cats", id="string-where-interest-set-expected"),
    pytest.param(("users", 2, "profile"), [], id="array-where-profile-expected"),
    pytest.param(("users", 1), "Omer", id="string-where-user-expected"),
    pytest.param(("users",), {"1": "Dana"}, id="object-where-user-list-expected"),
    pytest.param(("users",), REMOVED, id="missing-users"),
    pytest.param(("users",), None, id="null-users"),
    pytest.param(("meta",), ["total", 3], id="array-where-meta-object-expected"),
]


@pytest.mark.parametrize(("path", "replacement"), API_CORRUPTIONS)
def test_corrupted_api_response_is_rejected(path, replacement):
    assert_rejected(API_V3_JSON, ListUsersResponse, path, replacement)


def test_primitive_fields_are_not_validated_limitation():
    # Documents current behavior, not a promise: `id: int` happily holds a string, and `name: str` a number.
    payload = {"api_version": "1", "users": [{"id": "seven", "name": 8}]}

    restored = deserialize_json(payload, ListUsersResponse)

    assert restored.users[0].id == "seven" and restored.users[0].name == 8


def test_absent_required_field_is_restored_as_none_limitation():
    # Documents current behavior, not a promise: a missing key is passed on as None, even for `name: str`.
    restored = deserialize_json({"api_version": "1", "users": [{"id": 1}]}, ListUsersResponse)

    assert restored.users[0].name is None


def test_absent_mapping_field_becomes_none_while_an_absent_list_is_rejected_limitation():
    # Documents current behavior, not a promise: `dict[...]` takes None as is, `list[...]` raises.
    assert deserialize_json({"api_version": "1", "users": []}, ListUsersResponse).meta is None
    with pytest.raises(TypeError):
        deserialize_json({"api_version": "1"}, ListUsersResponse)


@dataclass
class PageSize:
    page_size: int = 25


@dataclass
class PageOrder:
    order: Literal["asc", "desc"] = "asc"


@dataclass
class PageLabels:
    labels: list[str] = field(default_factory=list)


def test_absent_field_ignores_its_dataclass_default_limitation():
    # Documents current behavior, not a promise: the default is not applied, None is passed instead. For `int` that
    # leaves None in the field; for a Literal or a list None is not a valid value, so the whole object is rejected.
    assert deserialize_json({}, PageSize).page_size is None
    with pytest.raises(TypeError):
        deserialize_json({}, PageOrder)
    with pytest.raises(TypeError):
        deserialize_json({}, PageLabels)
    assert deserialize_json({"page_size": 50}, PageSize) == PageSize(50)


def test_all_optional_dataclass_swallows_any_object_in_an_undiscriminated_union_limitation():
    # Documents current behavior, not a promise: members are tried in order and the first that does not raise wins.
    # Profile has only optional fields, so an address object is "a profile" whose fields are all None.
    address = {"street": "12 Herzl St", "city": "Haifa", "postal_code": "3303310", "country": "IL"}

    assert deserialize_json(address, Union[Profile, Address]) == Profile()
    assert deserialize_json(address, Union[Address, Profile]) == Address("12 Herzl St", "Haifa", "3303310", "IL")


# ---------------------------------------------------------------------------------------------------------------------
# 11. price list with a user-defined value object (Jsonable)
# ---------------------------------------------------------------------------------------------------------------------


class Money(Jsonable):
    """A value object with its own text form, "ILS 12.50"; from_json rejects what it cannot read with a TypeError."""

    def __init__(self, amount: Decimal, currency: str):
        self.amount = amount
        self.currency = currency

    def to_json(self) -> str:
        return f"{self.currency} {self.amount}"

    @classmethod
    def from_json(cls, data: Any) -> Money:
        if not isinstance(data, str):
            raise TypeError(f"Expected a string, got {type(data).__name__}")
        currency, _, amount = data.partition(" ")
        try:
            return cls(Decimal(amount), currency)
        except ArithmeticError:
            raise TypeError("Not an amount") from None

    def __eq__(self, other: object) -> bool:
        return (type(other) is Money and self.currency == other.currency
                and self.amount.as_tuple() == other.amount.as_tuple())

    def __hash__(self) -> int:
        return hash((self.currency, self.amount))

    def __repr__(self) -> str:
        return f"Money({self.amount!r}, {self.currency!r})"


@dataclass
class PriceList:
    default_currency: str
    prices: dict[str, Money]
    shipping: Optional[Money]
    bundles: list[tuple[str, Money, Money]]


def make_price_list() -> PriceList:
    return PriceList(
        default_currency="ILS",
        prices={"mug": Money(Decimal("24.90"), "ILS"), "shirt": Money(Decimal("79.00"), "ILS")},
        shipping=None,
        bundles=[("starter", Money(Decimal("103.90"), "ILS"), Money(Decimal("95.00"), "ILS"))],
    )


def make_price_list_with_shipping() -> PriceList:
    return replace(make_price_list(), shipping=Money(Decimal("3"), "USD"), bundles=[])


PRICE_LIST_JSON = {
    "default_currency": "ILS",
    "prices": {"mug": "ILS 24.90", "shirt": "ILS 79.00"},
    "shipping": None,
    "bundles": [["starter", "ILS 103.90", "ILS 95.00"]],
}

PRICE_LIST_WITH_SHIPPING_JSON = {**PRICE_LIST_JSON, "shipping": "USD 3", "bundles": []}


def test_price_list_exact_text_uses_the_value_objects_own_format():
    assert serialize_json(make_price_list(), separators=(",", ":")) == (
        '{"default_currency":"ILS","prices":{"mug":"ILS 24.90","shirt":"ILS 79.00"},"shipping":null,'
        '"bundles":[["starter","ILS 103.90","ILS 95.00"]]}'
    )


def test_price_list_restores_value_objects_inside_a_dict_an_optional_and_tuples():
    restored = deserialize_json(copy.deepcopy(PRICE_LIST_WITH_SHIPPING_JSON), PriceList)

    assert type(restored.shipping) is Money and restored.shipping == Money(Decimal("3"), "USD")
    assert restored.prices["mug"].amount.as_tuple().exponent == -2
    restored = deserialize_json(copy.deepcopy(PRICE_LIST_JSON), PriceList)
    assert restored.shipping is None
    assert restored.bundles[0] == ("starter", Money(Decimal("103.90"), "ILS"), Money(Decimal("95.00"), "ILS"))


PRICE_LIST_CORRUPTIONS = [
    pytest.param(("prices", "mug"), 24.9, id="number-where-money-expected"),
    pytest.param(("prices", "mug"), "ILS twenty", id="unparseable-amount"),
    pytest.param(("prices", "mug"), ["ILS", "24.90"], id="array-where-money-expected"),
    pytest.param(("shipping",), ["USD", "3"], id="array-where-optional-money-expected"),
    pytest.param(("bundles", 0), ["starter", "ILS 103.90"], id="bundle-with-two-elements"),
    pytest.param(("bundles", 0, 2), 95, id="number-in-the-bundle-price"),
    pytest.param(("bundles",), "starter", id="string-where-bundle-list-expected"),
]


@pytest.mark.parametrize(("path", "replacement"), PRICE_LIST_CORRUPTIONS)
def test_corrupted_price_list_is_rejected(path, replacement):
    assert_rejected(PRICE_LIST_JSON, PriceList, path, replacement)


class NaiveMoney(Jsonable):
    """The from_json most users write first: it assumes it is given a string."""

    def __init__(self, text: str):
        self.text = text

    def to_json(self) -> str:
        return self.text

    @classmethod
    def from_json(cls, data: Any) -> NaiveMoney:
        currency, _, amount = data.partition(" ")
        return cls(f"{currency} {amount}")


@dataclass
class NaivePriceList:
    shipping: Optional[NaiveMoney] = None


def test_optional_value_object_is_given_null_to_read_limitation():
    # Documents current behavior, not a promise: Optional[X] tries X before NoneType, so X.from_json receives the
    # null of an unset (or absent) field. Unions only move on after a TypeError: Money.from_json raises one for what
    # it cannot read, the naive version above leaks its own AttributeError instead.
    assert deserialize_json({"shipping": "USD 3"}, NaivePriceList).shipping.text == "USD 3"

    for payload in ({"shipping": None}, {}):
        with pytest.raises(AttributeError, match="partition"):
            deserialize_json(payload, NaivePriceList)
    careful = deserialize_json({"default_currency": "ILS", "prices": {}, "shipping": None, "bundles": []}, PriceList)
    assert careful.shipping is None


# ---------------------------------------------------------------------------------------------------------------------
# 12. file-system snapshot: a recursive union (file | symlink | directory) inside a dict
# ---------------------------------------------------------------------------------------------------------------------


class Permission(Enum):
    READ = "r"
    WRITE = "w"
    EXECUTE = "x"


@dataclass
class FileNode:
    kind: Literal["file"]
    size_bytes: int
    modified: datetime
    permissions: frozenset[Permission]
    content: Optional[bytes] = None


@dataclass
class LinkNode:
    kind: Literal["symlink"]
    target: str


@dataclass
class DirNode:
    kind: Literal["dir"]
    modified: datetime
    children: dict[str, Union[FileNode, LinkNode, DirNode]]


FsNode = Union[FileNode, LinkNode, DirNode]

SNAPSHOT_TIME = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
READ_ONLY = frozenset({Permission.READ})
READ_WRITE = frozenset({Permission.READ, Permission.WRITE})


def make_file_tree() -> DirNode:
    stamp = SNAPSHOT_TIME
    return DirNode("dir", stamp, {
        "README.md": FileNode("file", 1204, datetime(2026, 6, 30, 18, 45, 10, tzinfo=UTC), READ_WRITE),
        ".env": FileNode("file", 10, datetime(2026, 6, 1, 8, 0, tzinfo=UTC), READ_ONLY, content=b"KEY=value\n"),
        "bin": DirNode("dir", stamp, {
            "run.sh": FileNode("file", 311, datetime(2026, 5, 20, 9, 30, tzinfo=PLUS_TWO), frozenset(Permission)),
        }),
        "docs": DirNode("dir", stamp, {
            "api": DirNode("dir", stamp, {
                "v1": DirNode("dir", stamp, {
                    "index.html": FileNode("file", 5120, datetime(2026, 6, 2, 10, 0, 0, 500000, tzinfo=UTC),
                                           READ_WRITE),
                }),
                "latest": LinkNode("symlink", "v1"),
            }),
            "résumé.pdf": FileNode("file", 52000, datetime(2025, 12, 24, 23, 59, 59, tzinfo=UTC), READ_ONLY),
            "שלום.txt": FileNode("file", 0, datetime(2026, 1, 1, tzinfo=UTC), READ_WRITE, content=b""),
        }),
        "empty": DirNode("dir", stamp, {}),
    })


FILE_TREE_JSON = {
    "kind": "dir",
    "modified": "2026-07-01T12:00:00+00:00",
    "children": {
        "README.md": {"kind": "file", "size_bytes": 1204, "modified": "2026-06-30T18:45:10+00:00",
                      "permissions": ["r", "w"], "content": None},
        ".env": {"kind": "file", "size_bytes": 10, "modified": "2026-06-01T08:00:00+00:00",
                 "permissions": ["r"], "content": "S0VZPXZhbHVlCg=="},
        "bin": {
            "kind": "dir",
            "modified": "2026-07-01T12:00:00+00:00",
            "children": {
                "run.sh": {"kind": "file", "size_bytes": 311, "modified": "2026-05-20T09:30:00+02:00",
                           "permissions": ["r", "w", "x"], "content": None},
            },
        },
        "docs": {
            "kind": "dir",
            "modified": "2026-07-01T12:00:00+00:00",
            "children": {
                "api": {
                    "kind": "dir",
                    "modified": "2026-07-01T12:00:00+00:00",
                    "children": {
                        "v1": {
                            "kind": "dir",
                            "modified": "2026-07-01T12:00:00+00:00",
                            "children": {
                                "index.html": {"kind": "file", "size_bytes": 5120,
                                               "modified": "2026-06-02T10:00:00.500000+00:00",
                                               "permissions": ["r", "w"], "content": None},
                            },
                        },
                        "latest": {"kind": "symlink", "target": "v1"},
                    },
                },
                "résumé.pdf": {"kind": "file", "size_bytes": 52000, "modified": "2025-12-24T23:59:59+00:00",
                               "permissions": ["r"], "content": None},
                "שלום.txt": {"kind": "file", "size_bytes": 0, "modified": "2026-01-01T00:00:00+00:00",
                             "permissions": ["r", "w"], "content": ""},
            },
        },
        "empty": {"kind": "dir", "modified": "2026-07-01T12:00:00+00:00", "children": {}},
    },
}

SMALL_TREE_SORTED_COMPACT_TEXT = (
    '{"children":{"a":{"kind":"symlink","target":"b.txt"},'
    '"b.txt":{"content":"aGkK","kind":"file","modified":"2026-07-01T12:00:00+00:00",'
    '"permissions":["r","w"],"size_bytes":3}},"kind":"dir","modified":"2026-07-01T12:00:00+00:00"}'
)

# the path to the "latest" symlink, and to its parent directory's children
API_CHILDREN = ("children", "docs", "children", "api", "children")


def test_small_tree_sorted_compact_golden():
    tree = DirNode("dir", SNAPSHOT_TIME, {
        "b.txt": FileNode("file", 3, SNAPSHOT_TIME, frozenset({Permission.WRITE, Permission.READ}), b"hi\n"),
        "a": LinkNode("symlink", "b.txt"),
    })

    assert serialize_json(tree, sort_keys=True, separators=(",", ":")) == SMALL_TREE_SORTED_COMPACT_TEXT
    assert_round_trip(tree, DirNode, sort_keys=True, separators=(",", ":"))


def test_file_tree_restores_every_node_class_at_every_level():
    restored = deserialize_json(copy.deepcopy(FILE_TREE_JSON), DirNode)

    api = restored.children["docs"].children["api"]
    index = api.children["v1"].children["index.html"]
    assert type(restored) is DirNode and type(restored.children["README.md"]) is FileNode
    assert type(api) is DirNode and type(api.children["latest"]) is LinkNode and api.children["latest"].target == "v1"
    assert type(index) is FileNode and index.modified.microsecond == 500000
    assert restored.children["bin"].children["run.sh"].permissions == frozenset(Permission)
    assert restored.children["bin"].children["run.sh"].modified.utcoffset() == timedelta(hours=2)
    assert restored.children["empty"].children == {} and type(restored.children["empty"]) is DirNode
    assert list(restored.children["docs"].children) == ["api", "résumé.pdf", "שלום.txt"]
    # an empty file is not a file without content
    assert restored.children["docs"].children["שלום.txt"].content == b""
    assert restored.children["README.md"].content is None and restored.children[".env"].content == b"KEY=value\n"


def test_file_tree_children_and_nodes_restore_through_the_union_alone():
    tree = make_file_tree()

    assert_round_trip(tree.children, dict[str, FsNode])
    assert_round_trip(tree, FsNode)
    assert_round_trip(tree.children["docs"].children["api"].children["latest"], FsNode)
    assert_round_trip([tree.children["README.md"], tree.children["empty"]], list[FsNode])


def make_directory_chain(depth: int) -> DirNode:
    """d1 holds d2 holds ... d<depth>, which holds leaf.txt."""
    leaf = FileNode("file", 1, SNAPSHOT_TIME, READ_ONLY, b"\x00")
    node = DirNode("dir", SNAPSHOT_TIME, {"leaf.txt": leaf})
    for level in range(depth - 1, 0, -1):
        node = DirNode("dir", SNAPSHOT_TIME, {f"d{level + 1}": node})
    return DirNode("dir", SNAPSHOT_TIME, {"d1": node})


def chain_path_to_leaf(depth: int) -> tuple:
    path: tuple = ()
    for level in range(1, depth + 1):
        path += ("children", f"d{level}")
    return path + ("children", "leaf.txt")


def test_directory_chain_forty_levels_deep_round_trips_through_the_recursive_union():
    depth = 40
    chain = make_directory_chain(depth)

    parsed = to_json_value(chain)
    cursor = parsed
    for level in range(1, depth + 1):
        assert list(cursor["children"]) == [f"d{level}"]
        cursor = cursor["children"][f"d{level}"]
    assert cursor["children"]["leaf.txt"]["content"] == "AA=="
    assert_round_trip(chain, DirNode)


def test_corruption_at_the_bottom_of_a_forty_level_chain_is_rejected():
    document = to_json_value(make_directory_chain(40))

    assert_rejected(document, DirNode, chain_path_to_leaf(40) + ("kind",), "pipe")
    assert_rejected(document, DirNode, chain_path_to_leaf(40) + ("modified",), "noon")


def test_directory_of_five_hundred_mixed_entries_round_trips():
    children: dict = {}
    for index in range(500):
        name = f"entry-{index:03d}"
        if index % 3 == 0:
            permissions = READ_ONLY if index % 2 else READ_WRITE
            content = bytes([index % 256]) if index % 5 == 0 else None
            children[name] = FileNode("file", index, SNAPSHOT_TIME, permissions, content)
        elif index % 3 == 1:
            children[name] = LinkNode("symlink", f"entry-{index - 1:03d}")
        else:
            children[name] = DirNode("dir", SNAPSHOT_TIME, {})
    big = DirNode("dir", SNAPSHOT_TIME, children)

    parsed = to_json_value(big)["children"]
    assert len(parsed) == 500
    assert parsed["entry-499"] == {"kind": "symlink", "target": "entry-498"}
    assert parsed["entry-005"] == {"kind": "dir", "modified": "2026-07-01T12:00:00+00:00", "children": {}}
    assert parsed["entry-006"]["permissions"] == ["r", "w"] and parsed["entry-003"]["permissions"] == ["r"]
    assert parsed["entry-015"]["content"] == "Dw==" and parsed["entry-003"]["content"] is None
    assert [parsed[f"entry-{index:03d}"]["kind"] for index in range(6)] == [
        "file", "symlink", "dir", "file", "symlink", "dir",
    ]
    assert_round_trip(big, DirNode)


def test_directory_linked_into_its_own_subtree_is_a_cycle():
    tree = make_file_tree()
    tree.children["docs"].children["api"].children["root"] = tree

    with pytest.raises(ValueError, match="(?i)cycl"):
        serialize_json(tree)


def test_directory_linked_from_two_parents_is_not_a_cycle_and_is_restored_as_two_copies():
    shared = DirNode("dir", SNAPSHOT_TIME, {"x.txt": FileNode("file", 1, SNAPSHOT_TIME, READ_ONLY)})
    tree = DirNode("dir", SNAPSHOT_TIME, {
        "a": DirNode("dir", SNAPSHOT_TIME, {"shared": shared}),
        "b": DirNode("dir", SNAPSHOT_TIME, {"shared": shared}),
    })

    restored = assert_round_trip(tree, DirNode)

    assert restored.children["a"].children["shared"] is not restored.children["b"].children["shared"]


FILE_TREE_CORRUPTIONS = [
    pytest.param(("children", "README.md", "kind"), "pipe", id="unknown-node-kind"),
    pytest.param(("children", "README.md", "permissions"), "rw", id="string-where-permission-set-expected"),
    pytest.param(("children", "bin", "children", "run.sh", "permissions"), ["r", "z"],
                 id="unknown-permission-in-a-subdirectory"),
    pytest.param(("children", ".env", "content"), "abc", id="base64-with-bad-padding"),
    pytest.param(API_CHILDREN + ("v1", "children", "index.html", "modified"), "2026-07-32T00:00:00+00:00",
                 id="impossible-date-five-levels-down"),
    pytest.param(API_CHILDREN + ("latest",), "v1", id="string-where-node-expected"),
    pytest.param(("children", "docs", "children"), [], id="array-where-children-expected"),
    pytest.param(("children", "bin", "modified"), REMOVED, id="directory-without-timestamp"),
    pytest.param(("modified",), REMOVED, id="root-without-timestamp"),
    # a node of another kind fits a wrong member only if it has the fields of that member
    pytest.param(("children", "empty", "kind"), "file", id="directory-relabelled-as-file"),
    pytest.param(API_CHILDREN + ("latest", "kind"), "dir", id="symlink-relabelled-as-directory"),
    pytest.param(API_CHILDREN + ("latest", "kind"), "file", id="symlink-relabelled-as-file"),
]


@pytest.mark.parametrize(("path", "replacement"), FILE_TREE_CORRUPTIONS)
def test_corrupted_file_tree_is_rejected(path, replacement):
    assert_rejected(FILE_TREE_JSON, DirNode, path, replacement)


# ---------------------------------------------------------------------------------------------------------------------
# 13. survey submission: skipped questions are null items of a list of discriminated unions
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class ChoiceAnswer:
    kind: Literal["choice"]
    selected: str


@dataclass
class MultiChoiceAnswer:
    kind: Literal["multi"]
    selected: frozenset[str]


@dataclass
class RatingAnswer:
    kind: Literal["rating"]
    stars: Literal[1, 2, 3, 4, 5]
    comment: Optional[str] = None


@dataclass
class TextAnswer:
    kind: Literal["text"]
    text: str
    language: Optional[Literal["en", "he", "ar"]] = None


SurveyAnswer = Union[ChoiceAnswer, MultiChoiceAnswer, RatingAnswer, TextAnswer]


@dataclass
class Respondent:
    respondent_id: str
    age_band: Optional[Literal["18-24", "25-34", "35-44", "45+"]]
    country: Optional[str]


@dataclass
class SurveySubmission:
    survey_id: str
    submitted_at: datetime
    time_spent: tuple[int, int]
    respondent: Optional[Respondent]
    answers: list[Optional[SurveyAnswer]]
    by_question: dict[str, Optional[SurveyAnswer]]


def make_survey() -> SurveySubmission:
    answers = [
        ChoiceAnswer("choice", "monthly"),
        None,
        MultiChoiceAnswer("multi", frozenset({"sms", "email", "push"})),
        RatingAnswer("rating", 4, comment="fast delivery"),
        None,
        TextAnswer("text", "שירות מצוין", language="he"),
        RatingAnswer("rating", 1),
        TextAnswer("text", "no comment"),
    ]
    return SurveySubmission(
        survey_id="nps-2026-q2",
        submitted_at=datetime(2026, 6, 15, 7, 45, 30, tzinfo=PLUS_TWO),
        time_spent=(3, 25),
        respondent=Respondent("r-8841", "25-34", "IL"),
        answers=answers,
        by_question={"q1": answers[0], "q2": None, "q6": answers[5]},
    )


SURVEY_JSON = {
    "survey_id": "nps-2026-q2",
    "submitted_at": "2026-06-15T07:45:30+02:00",
    "time_spent": [3, 25],
    "respondent": {"respondent_id": "r-8841", "age_band": "25-34", "country": "IL"},
    "answers": [
        {"kind": "choice", "selected": "monthly"},
        None,
        {"kind": "multi", "selected": ["email", "push", "sms"]},
        {"kind": "rating", "stars": 4, "comment": "fast delivery"},
        None,
        {"kind": "text", "text": "שירות מצוין", "language": "he"},
        {"kind": "rating", "stars": 1, "comment": None},
        {"kind": "text", "text": "no comment", "language": None},
    ],
    "by_question": {
        "q1": {"kind": "choice", "selected": "monthly"},
        "q2": None,
        "q6": {"kind": "text", "text": "שירות מצוין", "language": "he"},
    },
}

TINY_SURVEY_TEXT = (
    '{"survey_id": "s1", "submitted_at": "2026-06-15T07:45:30+02:00", "time_spent": [0, 45], "respondent": null, '
    '"answers": [null, {"kind": "rating", "stars": 5, "comment": null}], "by_question": {"q1": null}}'
)


def test_tiny_survey_exact_text_keeps_null_items_in_place():
    tiny = SurveySubmission(
        "s1", datetime(2026, 6, 15, 7, 45, 30, tzinfo=PLUS_TWO), (0, 45), None,
        [None, RatingAnswer("rating", 5)], {"q1": None},
    )

    assert serialize_json(tiny) == TINY_SURVEY_TEXT
    assert_strictly_equal(deserialize_json(json.loads(TINY_SURVEY_TEXT), SurveySubmission), tiny)


def test_skipped_questions_are_none_and_every_answer_keeps_its_own_class():
    restored = deserialize_json(copy.deepcopy(SURVEY_JSON), SurveySubmission)

    assert [type(answer) for answer in restored.answers] == [
        ChoiceAnswer, type(None), MultiChoiceAnswer, RatingAnswer, type(None), TextAnswer, RatingAnswer, TextAnswer,
    ]
    assert restored.answers[1] is None and restored.by_question["q2"] is None
    assert restored.answers[2].selected == frozenset({"email", "push", "sms"})
    assert restored.answers[3].stars == 4 and type(restored.answers[3].stars) is int
    assert restored.answers[5].language == "he" and restored.answers[7].language is None
    assert restored.by_question["q6"] == restored.answers[5] and restored.by_question["q6"] is not restored.answers[5]


def test_survey_without_a_respondent_and_with_every_question_skipped():
    skipped = replace(make_survey(), respondent=None, answers=[None, None, None], by_question={"q1": None, "q2": None})

    parsed = to_json_value(skipped)
    assert parsed["respondent"] is None and parsed["answers"] == [None, None, None]
    restored = assert_round_trip(skipped, SurveySubmission)
    assert restored.answers == [None, None, None]


def test_survey_without_any_questions_round_trips():
    empty = replace(make_survey(), answers=[], by_question={})

    assert to_json_value(empty)["answers"] == [] and to_json_value(empty)["by_question"] == {}
    assert_round_trip(empty, SurveySubmission)


def test_survey_of_two_hundred_and_fifty_answers_round_trips():
    languages = [None, "en", "he", "ar"]
    answers: list = []
    for index in range(250):
        if index % 6 == 0:
            answers.append(ChoiceAnswer("choice", f"option-{index}"))
        elif index % 6 == 2:
            answers.append(MultiChoiceAnswer("multi", frozenset({f"opt-{index % 7}", f"opt-{index % 4}"})))
        elif index % 6 == 3:
            answers.append(RatingAnswer("rating", index % 5 + 1))
        elif index % 6 == 4:
            answers.append(TextAnswer("text", f"answer {index}", languages[(index // 6) % 4]))
        else:
            answers.append(None)
    by_question = {f"q{index}": answer for index, answer in enumerate(answers)}
    survey = replace(make_survey(), answers=answers, by_question=by_question)

    parsed = to_json_value(survey)
    assert parsed["answers"][1] is None and parsed["answers"][5] is None
    assert parsed["answers"][9] == {"kind": "rating", "stars": 5, "comment": None}
    assert parsed["answers"][22] == {"kind": "text", "text": "answer 22", "language": "ar"}
    assert parsed["answers"][248] == {"kind": "multi", "selected": ["opt-0", "opt-3"]}
    assert parsed["by_question"]["q248"] == parsed["answers"][248] and len(parsed["by_question"]) == 250
    assert_round_trip(survey, SurveySubmission)


SURVEY_CORRUPTIONS = [
    pytest.param(("answers", 0, "kind"), "slider", id="unknown-answer-kind"),
    pytest.param(("answers", 0, "kind"), REMOVED, id="answer-without-kind"),
    pytest.param(("answers", 3, "stars"), 6, id="six-stars"),
    pytest.param(("answers", 3, "stars"), 0, id="zero-stars"),
    pytest.param(("answers", 3, "stars"), "4", id="stars-as-text"),
    pytest.param(("answers", 3, "stars"), 4.0, id="stars-as-float"),
    pytest.param(("answers", 3, "stars"), True, id="stars-as-bool"),
    pytest.param(("answers", 5, "language"), "fr", id="unsupported-language"),
    pytest.param(("answers", 2, "selected"), "email", id="string-where-selection-set-expected"),
    pytest.param(("answers", 1), "skipped", id="string-where-answer-or-null-expected"),
    pytest.param(("answers", 4), [], id="array-where-answer-or-null-expected"),
    pytest.param(("answers",), {"0": None}, id="object-where-answer-list-expected"),
    pytest.param(("by_question", "q6", "kind"), "essay", id="unknown-kind-in-the-question-table"),
    pytest.param(("by_question",), [["q1", None]], id="array-of-pairs-where-question-table-expected"),
    pytest.param(("respondent", "age_band"), "65+", id="unknown-age-band"),
    pytest.param(("respondent",), "r-8841", id="string-where-respondent-expected"),
    pytest.param(("time_spent",), [3], id="one-element-duration"),
    pytest.param(("time_spent",), [3, 25, 1], id="three-element-duration"),
    pytest.param(("submitted_at",), "yesterday morning", id="unparseable-submission-time"),
    pytest.param(("answers",), REMOVED, id="missing-answers"),
]


@pytest.mark.parametrize(("path", "replacement"), SURVEY_CORRUPTIONS)
def test_corrupted_survey_is_rejected(path, replacement):
    assert_rejected(SURVEY_JSON, SurveySubmission, path, replacement)


# ---------------------------------------------------------------------------------------------------------------------
# 14. quarterly store report: nested enum-keyed tables, decimals with different exponents, dates and times of day
# ---------------------------------------------------------------------------------------------------------------------


class Region(Enum):
    NORTH = "north"
    CENTER = "center"
    SOUTH = "south"


class Channel(Enum):
    ONLINE = "online"
    STORE = "store"
    PHONE = "phone"


@dataclass
class Refund:
    booked: date
    order_id: str
    amount: Decimal
    reason: Optional[str]


@dataclass
class QuarterReport:
    store_id: str
    period_start: date
    period_end: date
    currency: Literal["ILS", "USD", "EUR"]
    totals: dict[Region, dict[Channel, Decimal]]
    monthly_revenue: dict[Region, dict[str, list[Decimal]]]
    ledger: list[tuple[date, str, Decimal]]
    refunds: tuple[Refund, ...]
    holidays: frozenset[date]
    opening_hours: dict[str, list[tuple[time, time]]]
    last_audit: Optional[tuple[date, time]]


def make_quarter_report() -> QuarterReport:
    return QuarterReport(
        store_id="TLV-07",
        period_start=date(2026, 4, 1),
        period_end=date(2026, 6, 30),
        currency="ILS",
        totals={
            Region.NORTH: {Channel.ONLINE: Decimal("125000.50"), Channel.STORE: Decimal("98000")},
            Region.CENTER: {
                Channel.ONLINE: Decimal("301250.00"),
                Channel.STORE: Decimal("0.00"),
                Channel.PHONE: Decimal("-1250.75"),
            },
            Region.SOUTH: {},
        },
        monthly_revenue={
            # 1E+4 is what Decimal("10000").normalize() gives: scientific notation is part of the value
            Region.NORTH: {"2026-04": [Decimal("30000.10"), Decimal("1E+4")], "2026-05": [],
                           "2026-06": [Decimal("9999.99")]},
            Region.CENTER: {},
        },
        ledger=[
            (date(2026, 4, 1), "opening-balance", Decimal("1000.00")),
            (date(2026, 4, 2), "refund #17", Decimal("-75.50")),
            (date(2026, 6, 30), "closing", Decimal("-0.00")),
        ],
        refunds=(
            Refund(date(2026, 4, 2), "ORD-17", Decimal("75.50"), "damaged"),
            Refund(date(2026, 5, 9), "ORD-93", Decimal("12"), None),
        ),
        holidays=frozenset({date(2026, 6, 2), date(2026, 4, 2), date(2026, 5, 1)}),
        opening_hours={
            "sun": [(time(9, 0), time(13, 0)), (time(16, 0), time(20, 30))],
            "fri": [(time(8, 0, tzinfo=PLUS_TWO), time(13, 0, tzinfo=PLUS_TWO))],
            "sat": [],
        },
        last_audit=(date(2026, 3, 31), time(23, 59, 59, 999999)),
    )


QUARTER_REPORT_JSON = {
    "store_id": "TLV-07",
    "period_start": "2026-04-01",
    "period_end": "2026-06-30",
    "currency": "ILS",
    "totals": {
        "north": {"online": "125000.50", "store": "98000"},
        "center": {"online": "301250.00", "store": "0.00", "phone": "-1250.75"},
        "south": {},
    },
    "monthly_revenue": {
        "north": {"2026-04": ["30000.10", "1E+4"], "2026-05": [], "2026-06": ["9999.99"]},
        "center": {},
    },
    "ledger": [
        ["2026-04-01", "opening-balance", "1000.00"],
        ["2026-04-02", "refund #17", "-75.50"],
        ["2026-06-30", "closing", "-0.00"],
    ],
    "refunds": [
        {"booked": "2026-04-02", "order_id": "ORD-17", "amount": "75.50", "reason": "damaged"},
        {"booked": "2026-05-09", "order_id": "ORD-93", "amount": "12", "reason": None},
    ],
    "holidays": ["2026-04-02", "2026-05-01", "2026-06-02"],
    "opening_hours": {
        "sun": [["09:00:00", "13:00:00"], ["16:00:00", "20:30:00"]],
        "fri": [["08:00:00+02:00", "13:00:00+02:00"]],
        "sat": [],
    },
    "last_audit": ["2026-03-31", "23:59:59.999999"],
}


def test_quarter_report_ledger_and_opening_hours_exact_text():
    report = make_quarter_report()

    assert serialize_json(report.ledger) == (
        '[["2026-04-01", "opening-balance", "1000.00"], ["2026-04-02", "refund #17", "-75.50"], '
        '["2026-06-30", "closing", "-0.00"]]'
    )
    assert serialize_json(report.opening_hours, separators=(",", ":")) == (
        '{"sun":[["09:00:00","13:00:00"],["16:00:00","20:30:00"]],"fri":[["08:00:00+02:00","13:00:00+02:00"]],"sat":[]}'
    )


def test_quarter_report_restores_nested_enum_keyed_tables_with_their_decimal_scales():
    restored = deserialize_json(copy.deepcopy(QUARTER_REPORT_JSON), QuarterReport)

    center = restored.totals[Region.CENTER]
    assert list(restored.totals) == [Region.NORTH, Region.CENTER, Region.SOUTH]
    assert restored.totals[Region.SOUTH] == {} and list(center) == [Channel.ONLINE, Channel.STORE, Channel.PHONE]
    assert center[Channel.STORE].as_tuple() == Decimal("0.00").as_tuple()
    assert restored.totals[Region.NORTH][Channel.STORE].as_tuple().exponent == 0
    assert restored.monthly_revenue[Region.NORTH]["2026-04"][1].as_tuple() == Decimal("1E+4").as_tuple()
    assert restored.monthly_revenue[Region.NORTH]["2026-05"] == [] and restored.monthly_revenue[Region.CENTER] == {}
    negative_zero = restored.ledger[2][2]
    assert negative_zero.is_signed() and negative_zero.is_zero() and negative_zero.as_tuple().exponent == -2


def test_quarter_report_restores_dates_times_and_tuples_of_mixed_types():
    restored = deserialize_json(copy.deepcopy(QUARTER_REPORT_JSON), QuarterReport)

    assert restored.ledger[1] == (date(2026, 4, 2), "refund #17", Decimal("-75.50"))
    assert type(restored.ledger[1]) is tuple and type(restored.refunds) is tuple
    assert restored.opening_hours["fri"][0][0].utcoffset() == timedelta(hours=2)
    assert restored.opening_hours["sun"][0][0].utcoffset() is None
    assert restored.last_audit == (date(2026, 3, 31), time(23, 59, 59, 999999))
    assert restored.refunds[1].reason is None and restored.refunds[1].amount.as_tuple().exponent == 0


@pytest.mark.parametrize("insertion_order", [(0, 1, 2), (2, 1, 0), (1, 2, 0)])
def test_holidays_are_written_chronologically_whatever_the_insertion_order(insertion_order):
    days = [date(2026, 4, 2), date(2026, 5, 1), date(2026, 6, 2)]
    holidays = frozenset(days[index] for index in insertion_order)

    assert serialize_json(replace(make_quarter_report(), holidays=holidays).holidays) == (
        '["2026-04-02", "2026-05-01", "2026-06-02"]'
    )


def test_report_with_every_table_empty_round_trips():
    empty = QuarterReport("TLV-08", date(2026, 4, 1), date(2026, 6, 30), "USD", {}, {}, [], (), frozenset(), {}, None)

    parsed = to_json_value(empty)
    assert parsed["totals"] == {} and parsed["refunds"] == [] and parsed["holidays"] == []
    assert parsed["last_audit"] is None
    assert_round_trip(empty, QuarterReport)


QUARTER_CORRUPTIONS = [
    pytest.param(("totals", "east"), {"online": "1"}, id="unknown-region-as-key"),
    pytest.param(("totals", "north", "kiosk"), "5", id="unknown-channel-as-key"),
    pytest.param(("totals", "north", "online"), "n/a", id="unparseable-amount"),
    pytest.param(("totals",), [["north", {}]], id="array-of-pairs-where-totals-expected"),
    pytest.param(("monthly_revenue", "north", "2026-04"), "30000.10", id="string-where-revenue-list-expected"),
    pytest.param(("monthly_revenue", "north", "2026-06", 0), [9999.99], id="array-where-amount-expected"),
    pytest.param(("ledger", 1), ["2026-04-02", "refund #17"], id="ledger-row-with-two-cells"),
    pytest.param(("ledger", 2, 0), "2026-06-31", id="june-31st-in-the-ledger"),
    pytest.param(("ledger", 0, 2), None, id="null-amount-in-the-ledger"),
    pytest.param(("refunds",), {"booked": "2026-04-02"}, id="object-where-refund-tuple-expected"),
    pytest.param(("refunds", 1, "booked"), "May 9th", id="unparseable-refund-date"),
    pytest.param(("holidays",), "2026-05-01", id="string-where-holiday-set-expected"),
    pytest.param(("holidays",), ["2026-05-01", "labour day"], id="holiday-that-is-not-a-date"),
    pytest.param(("opening_hours", "sun", 0), ["09:00:00"], id="opening-range-with-one-time"),
    pytest.param(("opening_hours", "sun", 1), ["16:00:00", "20:30:00", "22:00:00"],
                 id="opening-range-with-three-times"),
    pytest.param(("opening_hours", "fri", 0, 1), "25:00:00", id="hour-25"),
    pytest.param(("opening_hours", "sat"), {}, id="object-where-range-list-expected"),
    pytest.param(("last_audit",), ["2026-03-31"], id="audit-stamp-without-time"),
    pytest.param(("last_audit",), "2026-03-31T23:59:59", id="audit-stamp-as-one-string"),
    pytest.param(("currency",), "GBP", id="currency-not-in-the-literal"),
    pytest.param(("period_start",), "Q2", id="quarter-name-as-date"),
    pytest.param(("period_end",), REMOVED, id="missing-period-end"),
]


@pytest.mark.parametrize(("path", "replacement"), QUARTER_CORRUPTIONS)
def test_corrupted_quarter_report_is_rejected(path, replacement):
    assert_rejected(QUARTER_REPORT_JSON, QuarterReport, path, replacement)


@pytest.mark.parametrize("level", ["region", "channel"])
def test_enum_keys_written_as_member_names_instead_of_values_are_rejected(level):
    # `Region.NORTH` is written as "north", its value; the member name is a different key and must not be guessed
    by_name = copy.deepcopy(QUARTER_REPORT_JSON)
    if level == "region":
        by_name["totals"] = {name.upper(): table for name, table in by_name["totals"].items()}
    else:
        by_name["totals"]["north"] = {name.upper(): amount for name, amount in by_name["totals"]["north"].items()}
    deserialize_json(copy.deepcopy(QUARTER_REPORT_JSON), QuarterReport)  # the intact report loads

    with pytest.raises(TypeError):
        deserialize_json(by_name, QuarterReport)


# ---------------------------------------------------------------------------------------------------------------------
# 15. documents inside other containers, and all of them in one bundle
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class ExportBundle:
    exported_at: datetime
    orders: list[Order]
    org: Optional[Employee]
    audit: AuditLog
    geo: Optional[FeatureCollection]
    workflows: dict[str, WorkflowDefinition]
    api: tuple[ListUsersResponse, ...]
    catalog: list[LocalizedProduct]
    prices: PriceList
    files: Optional[DirNode]
    surveys: list[SurveySubmission]
    quarter: Optional[QuarterReport]


def make_export_bundle() -> ExportBundle:
    return ExportBundle(
        exported_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
        orders=[make_full_order(), make_guest_order()],
        org=make_org_chart(),
        audit=make_audit_log(),
        geo=make_feature_collection(),
        workflows={"approval": make_workflow()},
        api=(make_api_response(),),
        catalog=[make_localized_product()],
        prices=make_price_list(),
        files=make_file_tree(),
        surveys=[make_survey()],
        quarter=make_quarter_report(),
    )


EXPORT_BUNDLE_JSON = {
    "exported_at": "2026-06-01T12:00:00+00:00",
    "orders": [FULL_ORDER_JSON, json.loads(GUEST_ORDER_TEXT)],
    "org": ORG_JSON,
    "audit": AUDIT_LOG_JSON,
    "geo": FEATURE_COLLECTION_JSON,
    "workflows": {"approval": WORKFLOW_JSON},
    "api": [API_V3_JSON],
    "catalog": [LOCALIZED_PRODUCT_JSON],
    "prices": PRICE_LIST_JSON,
    "files": FILE_TREE_JSON,
    "surveys": [SURVEY_JSON],
    "quarter": QUARTER_REPORT_JSON,
}


def test_bundle_without_its_optional_parts_keeps_the_rest():
    bundle = replace(
        make_export_bundle(),
        org=None, geo=None, files=None, quarter=None, orders=[], workflows={}, api=(), catalog=[], surveys=[],
    )

    parsed = to_json_value(bundle)

    assert [parsed[name] for name in ("org", "geo", "files", "quarter")] == [None, None, None, None]
    assert [parsed[name] for name in ("orders", "workflows", "api", "catalog", "surveys")] == [[], {}, [], [], []]
    assert_round_trip(bundle, ExportBundle)


BUNDLE_CORRUPTIONS = [
    pytest.param(("orders", 1, "items", 0, "sku"), "SKU-X", id="unknown-sku-in-the-second-order"),
    pytest.param(("org", "reports", 0, "reports", 0, "hire_date"), "2020-02-30", id="february-30th-in-the-org-chart"),
    pytest.param(("audit", "events", 2, "mode"), "execute", id="unknown-access-mode-in-the-audit-log"),
    pytest.param(("geo", "features", 3, "geometry", "coordinates", 1, 1, 0), [1.0], id="short-position-in-the-map"),
    pytest.param(("workflows", "approval", "rules", "draft", 0, "guard"), "never", id="unknown-guard-in-a-workflow"),
    pytest.param(("api", 0, "users", 2, "plan"), "gold", id="unknown-plan-in-the-api-response"),
    pytest.param(("catalog", 0, "reviews", 3), "none", id="string-where-review-expected"),
    pytest.param(("prices", "bundles", 0, 1), 103.9, id="number-where-money-expected"),
    pytest.param(("files", "children", "docs", "children", "api", "children", "latest", "kind"), "dir",
                 id="symlink-relabelled-as-directory-in-the-file-tree"),
    pytest.param(("surveys", 0, "answers", 3, "stars"), 7, id="seven-stars-in-the-survey"),
    pytest.param(("quarter", "totals", "south", "mail"), "1", id="unknown-channel-in-the-quarter-report"),
    pytest.param(("exported_at",), "June 1st", id="unparseable-export-time"),
    pytest.param(("api",), {"0": {}}, id="object-where-response-tuple-expected"),
]


@pytest.mark.parametrize(("path", "replacement"), BUNDLE_CORRUPTIONS)
def test_corruption_deep_inside_the_bundle_is_rejected(path, replacement):
    assert_rejected(EXPORT_BUNDLE_JSON, ExportBundle, path, replacement)


def test_documents_as_elements_of_root_containers_round_trip():
    full, guest = make_full_order(), make_guest_order()

    assert_round_trip([full, guest], list[Order])
    assert_round_trip({"a": full, "b": guest}, dict[str, Order])
    assert_round_trip((full, make_api_response()), tuple[Order, ListUsersResponse])
    assert_round_trip([[full], [], [guest, guest]], list[list[Order]])
    assert_round_trip({"org": [make_org_chart()], "none": []}, dict[str, list[Employee]])
    assert_round_trip([], list[Order])
    assert_round_trip({}, dict[str, Order])
    assert_round_trip((), tuple[Order, ...])


@pytest.mark.parametrize("annotation", [Optional[Order], Order | None], ids=["Optional", "PEP604"])
def test_optional_document_at_the_root(annotation):
    assert_round_trip(make_full_order(), annotation)
    assert_round_trip(None, annotation)


def test_union_of_unrelated_documents_picks_the_one_the_data_fits():
    for annotation in (Union[Order, Employee], Union[Employee, Order]):
        order = assert_round_trip(make_guest_order(), annotation)
        employee = assert_round_trip(make_org_chart(), annotation)
        assert type(order) is Order and type(employee) is Employee


def test_duplicate_documents_in_one_list_are_independent_after_a_round_trip():
    order = make_guest_order()

    restored = assert_round_trip([order, order, order], list[Order])

    assert len({id(item) for item in restored}) == 3
    restored[0].tags.add("changed")
    assert restored[1].tags == set()


# ---------------------------------------------------------------------------------------------------------------------
# generic contract, run over every document
# ---------------------------------------------------------------------------------------------------------------------

# (id, builder, builder of the value to round-trip (None: the same), class, expected JSON, marks of the round trips)
DOCUMENTS = [
    ("order-full", make_full_order, None, Order, FULL_ORDER_JSON, ()),
    ("order-guest", make_guest_order, None, Order, json.loads(GUEST_ORDER_TEXT), ()),
    ("org-chart", make_org_chart, None, Employee, ORG_JSON, ()),
    ("config-prod", lambda: make_prod_config(PurePosixPath), make_prod_config, AppConfig, PROD_CONFIG_JSON, ()),
    ("config-dev", lambda: make_dev_config(PurePosixPath), make_dev_config, AppConfig, DEV_CONFIG_JSON, ()),
    ("audit-log", make_audit_log, None, AuditLog, AUDIT_LOG_JSON, ()),
    ("audit-log-pep604", make_audit_log_pep604, None, AuditLogPep604, AUDIT_LOG_JSON, ()),
    ("feature-collection", make_feature_collection, None, FeatureCollection, FEATURE_COLLECTION_JSON, ()),
    ("graph", make_graph_document, None, GraphDocument, GRAPH_JSON, ()),
    ("run-diverged", lambda: make_diverged_run(PurePosixPath), make_diverged_run, ExperimentRun, DIVERGED_RUN_JSON, ()),
    ("run-running", lambda: make_running_run(PurePosixPath), make_running_run, ExperimentRun, RUNNING_RUN_JSON, ()),
    ("localized-product", make_localized_product, None, LocalizedProduct, LOCALIZED_PRODUCT_JSON, ()),
    ("workflow", make_workflow, None, WorkflowDefinition, WORKFLOW_JSON, ()),
    ("api-response", make_api_response, None, ListUsersResponse, API_V3_JSON, ()),
    ("price-list", make_price_list, None, PriceList, PRICE_LIST_JSON, ()),
    ("access-policy", make_access_policy, None, AccessPolicy, ACCESS_POLICY_JSON, ()),
    ("price-list-with-shipping", make_price_list_with_shipping, None, PriceList, PRICE_LIST_WITH_SHIPPING_JSON, ()),
    ("file-tree", make_file_tree, None, DirNode, FILE_TREE_JSON, ()),
    ("survey", make_survey, None, SurveySubmission, SURVEY_JSON, ()),
    ("quarter-report", make_quarter_report, None, QuarterReport, QUARTER_REPORT_JSON, ()),
    ("export-bundle", make_export_bundle, None, ExportBundle, EXPORT_BUNDLE_JSON, ()),
]


def _document_params(*, round_trip: bool) -> list:
    params = []
    for doc_id, build, build_exact, cls, expected, marks in DOCUMENTS:
        if round_trip:
            marks = [*marks, *([posix_only] if build_exact else [])]
            params.append(pytest.param(build_exact or build, cls, expected, id=doc_id, marks=marks))
        else:
            params.append(pytest.param(build, cls, expected, id=doc_id))
    return params


@pytest.mark.parametrize(("build", "cls", "expected"), _document_params(round_trip=False))
def test_document_serializes_to_the_hand_written_json(build, cls, expected):
    assert_same_json(to_json_value(build()), expected)


@pytest.mark.parametrize(("build", "cls", "expected"), _document_params(round_trip=False))
def test_document_keys_are_written_in_field_declaration_order(build, cls, expected):
    assert list(key_orders(to_json_value(build()))) == list(key_orders(expected))


@pytest.mark.parametrize(("build", "cls", "expected"), _document_params(round_trip=False))
def test_serialize_json_as_obj_agrees_with_the_parsed_text(build, cls, expected):
    value = build()

    as_obj = serialize_json_as_obj(value)

    assert_same_json(as_obj, json.loads(serialize_json(value)))
    assert_same_json(as_obj, expected)


@pytest.mark.parametrize(("build", "cls", "expected"), _document_params(round_trip=True))
def test_document_round_trips_exactly(build, cls, expected):
    assert_round_trip(build(), cls)


@pytest.mark.parametrize(("build", "cls", "expected"), _document_params(round_trip=True))
def test_hand_written_json_restores_the_document(build, cls, expected):
    assert_strictly_equal(deserialize_json(copy.deepcopy(expected), cls), build())


@pytest.mark.parametrize(("build", "cls", "expected"), _document_params(round_trip=False))
def test_document_json_does_not_depend_on_the_formatting_options(build, cls, expected):
    value = build()

    for options in ({"sort_keys": True}, {"indent": 2}, {"indent": "\t", "sort_keys": True, "ensure_ascii": False}):
        assert_same_json(json.loads(serialize_json(value, **options)), expected)


@pytest.mark.parametrize(("build", "cls", "expected"), _document_params(round_trip=True))
def test_document_round_trips_through_sorted_indented_unicode_text(build, cls, expected):
    assert_round_trip(build(), cls, sort_keys=True, indent=2, ensure_ascii=False)


@pytest.mark.parametrize(("build", "cls", "expected"), _document_params(round_trip=False))
def test_default_text_is_what_json_dumps_makes_of_its_own_parsed_form(build, cls, expected):
    # also proves that no object of the text has a repeated key: json.loads would keep only the last one
    text = serialize_json(build())

    assert json.dumps(json.loads(text)) == text


@pytest.mark.parametrize(("build", "cls", "expected"), _document_params(round_trip=False))
def test_serializing_a_document_leaves_it_unchanged(build, cls, expected):
    value = build()

    serialize_json(value)
    serialize_json(value, sort_keys=True, indent=2, ensure_ascii=False)

    assert_strictly_equal(value, build())


def _mutable_container_ids(value: Any) -> set:
    """The ids of every dict, list and set reachable from a JSON value or from a restored document."""
    found: set = set()
    pending = [value]
    while pending:
        item = pending.pop()
        if is_dataclass(item) and not isinstance(item, type):
            pending.extend(getattr(item, one.name) for one in fields(item))
        elif isinstance(item, (dict, list, set)):
            if id(item) in found:
                continue
            found.add(id(item))
            pending.extend(item.items() if isinstance(item, dict) else item)
        elif isinstance(item, MappingProxyType):
            pending.extend(item.items())
        elif isinstance(item, (tuple, frozenset)):
            pending.extend(item)
    return found


@pytest.mark.parametrize(("build", "cls", "expected"), _document_params(round_trip=True))
def test_restored_document_shares_no_mutable_container_with_the_json_it_came_from(build, cls, expected):
    parsed = copy.deepcopy(expected)

    restored = deserialize_json(parsed, cls)

    source_ids, restored_ids = _mutable_container_ids(parsed), _mutable_container_ids(restored)
    assert source_ids and restored_ids
    assert source_ids.isdisjoint(restored_ids)


# The JSON goldens are module-level and shared by all tests: no test may leave one of them changed (by itself or
# through the library), or later tests would depend on the order in which tests run
SHARED_GOLDENS = [expected for *_, expected, _ in DOCUMENTS] + [
    API_V1_JSON, API_V2_JSON, EDGE_WEIGHTS_JSON, TRAINING_ARRAYS_JSON, EXPORT_BUNDLE_JSON,
]
GOLDEN_SNAPSHOTS = [json.dumps(golden, sort_keys=True) for golden in SHARED_GOLDENS]


@pytest.fixture(autouse=True)
def _goldens_stay_untouched():
    yield
    for golden, snapshot in zip(SHARED_GOLDENS, GOLDEN_SNAPSHOTS):
        assert json.dumps(golden, sort_keys=True) == snapshot, "a test changed a shared JSON golden"

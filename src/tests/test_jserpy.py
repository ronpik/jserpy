import json

import pytest
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple, Optional
from datetime import datetime

from jserpy import serialize_json, deserialize_json, serialize_json_as_dict


# Test data classes
@dataclass
class Person:
    name: str
    age: int


@dataclass
class ComplexPerson:
    name: str
    age: int
    scores: List[float]
    metadata: Dict[str, str]
    birth_date: datetime
    optional_field: Optional[str] = None


class Color(Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"


# Test cases
def test_simple_dataclass():
    person = Person(name="John", age=30)
    json_str = serialize_json(person)
    restored = deserialize_json(json.loads(json_str), Person)
    assert person == restored


def test_complex_dataclass():
    person = ComplexPerson(
        name="Alice",
        age=25,
        scores=[95.5, 87.0, 92.5],
        metadata={"city": "New York", "occupation": "Engineer"},
        birth_date=datetime(1998, 5, 15)
    )
    json_str = serialize_json(person)
    restored = deserialize_json(json.loads(json_str), ComplexPerson)
    assert person == restored


def test_numpy_array():
    arr = np.array([1, 2, 3, 4, 5])
    json_str = serialize_json(arr)
    restored = deserialize_json(json.loads(json_str), np.ndarray)
    assert np.array_equal(arr, restored)


def test_enum():
    color = Color.RED
    json_str = serialize_json(color)
    restored = deserialize_json(json.loads(json_str), Color)
    assert color == restored


def test_nested_types():
    data = {
        "tuple": (1, "two", 3.0),
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2},
        "enum": Color.BLUE
    }
    json_str = serialize_json(data)
    restored = deserialize_json(json.loads(json_str), dict)
    assert restored["tuple"] == list(data["tuple"])
    assert restored["list"] == data["list"]
    assert restored["dict"] == data["dict"]
    assert restored["enum"] == data["enum"].value


def test_serialize_as_dict():
    person = Person(name="John", age=30)
    dict_data = serialize_json_as_dict(person)
    assert isinstance(dict_data, dict)
    assert dict_data["name"] == "John"
    assert dict_data["age"] == 30


def test_optional_fields():
    person1 = ComplexPerson(
        name="Bob",
        age=40,
        scores=[85.0],
        metadata={},
        birth_date=datetime(1984, 3, 12),
        optional_field="present"
    )
    person2 = ComplexPerson(
        name="Charlie",
        age=35,
        scores=[90.0],
        metadata={},
        birth_date=datetime(1989, 7, 23)
    )

    json_str1 = serialize_json(person1)
    json_str2 = serialize_json(person2)

    json1 = json.loads(json_str1)
    json2 = json.loads(json_str2)

    restored1 = deserialize_json(json1, ComplexPerson)
    restored2 = deserialize_json(json2, ComplexPerson)

    assert person1 == restored1
    assert person2 == restored2
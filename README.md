# JSerPy

JSerPy serializes and deserializes complex Python objects while preserving the
standard JSON data model.

## Installation

Install the core package without NumPy:

```bash
pip install jserpy
```

Install the NumPy integration when array and scalar support is needed:

```bash
pip install "jserpy[numpy]"
```

If NumPy is already installed, JSerPy detects it automatically; the extra only
controls installation.

## Usage

```python
import json
from dataclasses import dataclass

from jserpy import deserialize_json, serialize_json


@dataclass
class Person:
    name: str
    age: int


person = Person(name="John", age=30)
json_str = serialize_json(person)
restored_person = deserialize_json(json.loads(json_str), Person)

assert person == restored_person
```

Calling `serialize_json(value)` without options retains the default JSON output
format. Keyword-only encoding controls can customize the text representation:

```python
document = serialize_json(
    person,
    ensure_ascii=False,
    allow_nan=False,
    indent=2,
    separators=None,
    sort_keys=True,
)
```

The available controls are:

- `ensure_ascii: bool = True`
- `allow_nan: bool = True`
- `indent: int | str | None = None`
- `separators: tuple[str, str] | None = None`
- `sort_keys: bool = False`
- `fallback: Callable[[Any], Any] | None = None`

A fallback handles values for which JSerPy has no registered handler. Registered
handlers always take precedence, and a fallback result is processed recursively:

```python
json_str = serialize_json(
    {"value": object()},
    fallback=lambda value: {"unsupported_type": type(value).__name__},
)
```

`serialize_json()` returns a string and does not append a trailing newline.

## Intermediate JSON values

Use `serialize_json_as_obj()` when a detached JSON-compatible value is needed
before text encoding. Returned containers are mutable:

```python
from jserpy import serialize_json_as_obj

json_value = serialize_json_as_obj(person)
json_value["name"] = "Jane"
```

The result may be a dictionary, list, string, number, boolean, or `None`.
`serialize_json_as_dict()` remains available as a compatibility name and accepts
the same arguments and root values despite its historical name.

## Supported values

JSerPy supports JSON primitives and containers together with:

- dataclasses and `Jsonable` implementations
- enums
- lists and tuples
- `bytes` encoded as Base64
- `datetime.datetime`, `datetime.date`, and `datetime.time` as ISO-8601 strings
- `Decimal` as a string, preserving precision and trailing zeroes
- `Path` and `PurePath`
- general `Mapping` implementations
- sets and frozensets as deterministically ordered arrays
- NumPy arrays and scalars when NumPy is installed

Enum and tuple dictionary keys are supported. Serialization raises an error
rather than silently losing data when distinct keys produce the same JSON key.

## Contributing

Contributions are welcome. Run the test suite with:

```bash
python -m pytest src/tests
```

## License

This project is licensed under the MIT License. See `LICENSE` for details.

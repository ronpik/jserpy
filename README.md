# JSerPy

JSerPy is a Python library for serializing and deserializing complex objects to and from JSON. It provides robust support for handling various Python types including dataclasses, enums, numpy arrays, and custom objects.

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/ronpik/jserpy.git
```

## Usage

Here's a simple example of how to use JSerPy:

```python
from dataclasses import dataclass
from jserpy import serialize_json, deserialize_json

@dataclass
class Person:
    name: str
    age: int

# Create an instance
person = Person(name="John", age=30)

# Serialize to JSON
json_str = serialize_json(person)

# Deserialize back to object
restored_person = deserialize_json(json_str, Person)

assert person == restored_person
```

## Features

- Supports serialization of complex Python objects
- Handles dataclasses, enums, numpy arrays, and custom objects
- Type-safe deserialization
- Customizable serialization behavior

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
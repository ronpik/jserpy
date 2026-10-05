from inspect import Parameter, signature
from typing import get_type_hints

import jserpy
from jserpy import (
    deserialize_json,
    serialize_json,
    serialize_json_as_dict,
    serialize_json_as_obj,
)
from jserpy.json_handler import (
    deserialize_json as handler_deserialize_json,
    serialize_json as handler_serialize_json,
    serialize_json_as_dict as handler_serialize_json_as_dict,
    serialize_json_as_obj as handler_serialize_json_as_obj,
)
from jserpy.json_typing import JSON


def test_required_functions_are_deliberate_root_exports():
    assert jserpy.deserialize_json is handler_deserialize_json
    assert jserpy.serialize_json is handler_serialize_json
    assert jserpy.serialize_json_as_dict is handler_serialize_json_as_dict
    assert jserpy.serialize_json_as_obj is handler_serialize_json_as_obj


def test_serialize_json_public_signature():
    parameters = signature(serialize_json).parameters

    assert tuple(parameters) == (
        "obj",
        "ensure_ascii",
        "allow_nan",
        "indent",
        "separators",
        "sort_keys",
        "fallback",
    )
    assert parameters["obj"].kind is Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["obj"].default is Parameter.empty
    for name in tuple(parameters)[1:]:
        assert parameters[name].kind is Parameter.KEYWORD_ONLY
    assert parameters["ensure_ascii"].default is True
    assert parameters["allow_nan"].default is True
    assert parameters["indent"].default is None
    assert parameters["separators"].default is None
    assert parameters["sort_keys"].default is False
    assert parameters["fallback"].default is None
    assert get_type_hints(serialize_json)["return"] is str


def test_intermediate_public_signatures_and_return_types():
    for serializer in (serialize_json_as_obj, serialize_json_as_dict):
        parameters = signature(serializer).parameters
        assert tuple(parameters) == ("obj", "fallback")
        assert parameters["obj"].kind is Parameter.POSITIONAL_OR_KEYWORD
        assert parameters["obj"].default is Parameter.empty
        assert parameters["fallback"].kind is Parameter.KEYWORD_ONLY
        assert parameters["fallback"].default is None
        assert get_type_hints(serializer)["return"] == JSON


def test_deserialize_json_keeps_existing_public_signature():
    parameters = signature(deserialize_json).parameters

    assert tuple(parameters) == ("data", "cls")
    assert all(
        parameter.kind is Parameter.POSITIONAL_OR_KEYWORD
        for parameter in parameters.values()
    )
    assert all(parameter.default is Parameter.empty for parameter in parameters.values())

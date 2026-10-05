"""Lazy NumPy integration helpers.

This module deliberately avoids importing NumPy at module import time. NumPy is
loaded only after dispatch encounters a class defined by NumPy (or one of its
subclasses), so importing and using JSerPy's core remains NumPy-independent.
"""

from functools import lru_cache
from typing import Any, Literal


NumpyKind = Literal["array", "scalar"]


def _is_numpy_candidate(cls: type[Any]) -> bool:
    for candidate in getattr(cls, "__mro__", (cls,)):
        module_name = getattr(candidate, "__module__", "")
        if module_name == "numpy" or module_name.startswith("numpy."):
            return True
    return False


@lru_cache(maxsize=None)
def get_numpy_kind(cls: type[Any]) -> NumpyKind | None:
    if not _is_numpy_candidate(cls):
        return None

    try:
        import numpy as np
    except ModuleNotFoundError:
        return None

    if issubclass(cls, np.ndarray):
        return "array"
    if issubclass(cls, np.generic):
        return "scalar"
    return None


def serialize_numpy(obj: Any, kind: NumpyKind) -> Any:
    if kind == "array":
        return obj.tolist()
    return obj.item()


def deserialize_numpy(data: Any, cls: type[Any], kind: NumpyKind) -> Any:
    if kind == "array":
        import numpy as np

        return np.array(data)
    return cls(data)

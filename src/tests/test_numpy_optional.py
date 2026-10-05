import subprocess
import sys
import textwrap

import pytest

from jserpy import deserialize_json, serialize_json


def test_import_does_not_load_numpy():
    code = textwrap.dedent(
        """
        import sys
        import jserpy
        jserpy.serialize_json({"a": [1, 2]})
        assert "numpy" not in sys.modules
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_numpy_round_trip_when_available():
    np = pytest.importorskip("numpy")
    assert serialize_json(np.int64(3)) == "3"
    assert serialize_json(np.array([1, 2])) == "[1, 2]"
    assert deserialize_json([1, 2], np.ndarray).tolist() == [1, 2]
    assert deserialize_json(2, np.int64) == np.int64(2)

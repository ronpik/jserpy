"""The suite must collect under every pytest import mode a workflow uses.

CI runs the default (prepend) mode; the release validation in scripts/bump_version.py runs
--import-mode=importlib against the built wheel, where the shared test helpers are not importable
unless the pytest configuration puts them on sys.path.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("mode", ["prepend", "importlib"])
def test_the_suite_collects_in_every_import_mode(mode):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "src/tests",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--import-mode={mode}",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

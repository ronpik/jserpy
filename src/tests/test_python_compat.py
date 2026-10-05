"""The package declares Python >= 3.10: guard sources and tests against newer-only stdlib names."""

import ast
from pathlib import Path

SOURCES_ROOT = Path(__file__).resolve().parents[1]


def _python_311_only_names(tree: ast.AST) -> list[int]:
    offending_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "datetime":
            if any(alias.name == "UTC" for alias in node.names):
                offending_lines.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == "UTC":
            if isinstance(node.value, ast.Name) and node.value.id == "datetime":
                offending_lines.append(node.lineno)
    return offending_lines


def test_sources_and_tests_avoid_datetime_utc():
    # datetime.UTC exists only on Python 3.11+; importing it made the whole suite uncollectable on 3.10.
    offenders = {}
    for path in sorted(SOURCES_ROOT.rglob("*.py")):
        lines = _python_311_only_names(ast.parse(path.read_text(encoding="utf-8")))
        if lines:
            offenders[str(path.relative_to(SOURCES_ROOT))] = lines

    assert offenders == {}, "use datetime.timezone.utc instead of datetime.UTC"

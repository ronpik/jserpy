# Repository Guidelines

## Project Structure & Module Organization

JSerPy is a Python 3.10+ library using a setuptools `src` layout. Package code lives in `src/jserpy/`: `json_handler.py` contains serialization handlers and dispatch logic, while `json_handler_utils.py`, `json_typing.py`, and `jsonable.py` provide supporting protocols and types. Keep intentional public exports in `src/jserpy/__init__.py`. Tests live in `src/tests/` as behavior-specific `test_*.py` modules (compatibility goldens, encoding options, fallback, cycles, mapping keys, NumPy, public API, ...). NumPy support is optional and imported lazily through `src/jserpy/_numpy.py`. Packaging metadata is in `pyproject.toml`; `.github/workflows/python-tests.yml` runs the suite on every supported Python with and without NumPy, and `.github/workflows/python-publish.yml` owns release builds. There is no application entry point or asset directory.

## Build, Test, and Development Commands

Create an isolated environment and install the package in editable mode:

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install -e ".[numpy]" pytest build
```

- `python -m pytest src/tests` runs the complete test suite; NumPy tests are skipped when NumPy is not installed, so also run it once in an environment without the `numpy` extra.
- `python -m build` creates source and wheel distributions in `dist/`, matching the release workflow.

The project does not define a development dependency group, so install `pytest` and `build` explicitly. Use Python 3.10 or newer; implementation code imports `types.UnionType`. Keep sources and tests 3.10-compatible (for example `datetime.timezone.utc`, not `datetime.UTC`).

## Coding Style & Naming Conventions

Follow standard PEP 8 formatting with four-space indentation. Use `snake_case` for functions and variables, `PascalCase` for classes and handlers, and a leading underscore for internal helpers. Preserve type annotations on public APIs and handler methods. Group imports as standard library, third-party, then local modules. No formatter or linter is configured, so keep edits focused and review whitespace and imports manually.

## Testing Guidelines

Pytest discovers `test_*.py` files and `test_<behavior>` functions. Add regression tests for both serialization and deserialization, preferably as round trips. Cover relevant type behavior such as dataclasses, enums, NumPy values, paths, optional fields, and nested containers. There is no enforced coverage threshold; every behavior change should include a focused test and pass on Python 3.10+.

## Commit & Pull Request Guidelines

History uses short descriptive subjects without Conventional Commit prefixes. Prefer a concise imperative subject such as `Handle pathlib objects`, and keep each commit to one logical change. Pull requests should summarize the change, note compatibility or JSON-shape impact, list local test/build results, and link an issue when applicable. Include a minimal before/after JSON example for serialization changes; screenshots are unnecessary for this library. Publishing is triggered by a GitHub Release, not by routine pull requests.

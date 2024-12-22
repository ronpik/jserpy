from typing import Dict, List, Union

# https://github.com/python/typing/issues/182#issuecomment-1320974824
# All major type checkers now support recursive type aliases by default, so this should largely work:
# JSON: TypeAlias = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]
JSON = Union[Dict[str, "JSON"], List["JSON"], str, int, float, bool, None]

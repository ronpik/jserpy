from abc import abstractmethod, ABC

from jserpy.json_typing import JSON


class Jsonable(ABC):
    @abstractmethod
    def to_json(self) -> JSON:
        pass

    @classmethod
    @abstractmethod
    def from_json(cls, data: JSON):
        pass

from abc import ABC, abstractmethod
from typing import Tuple
from typing import Union, Callable


class VectorStoreInterface(ABC):
    @abstractmethod
    def add(self, embedding: list,  **kwargs) -> str:
        pass

    @abstractmethod
    def search(self, embedding: list, top_n: int = 1, include_distances: bool = True, **kwargs) -> Tuple[list, list]:
        pass
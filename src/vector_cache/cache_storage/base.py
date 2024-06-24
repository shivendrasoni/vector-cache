from abc import ABC, abstractmethod


class CacheStorageInterface(ABC):
    @abstractmethod
    def set_response(self, query_index: int, response: str):
        pass

    @abstractmethod
    def get_response(self, query_index: int) -> str:
        pass
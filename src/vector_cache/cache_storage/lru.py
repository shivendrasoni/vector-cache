from collections import OrderedDict
from vector_cache.cache_storage.base import CacheStorageInterface


class LRUCache(CacheStorageInterface):
    def __init__(self, capacity: int=500):
        self.cache = OrderedDict()
        self.capacity = capacity

    def set_response(self, query_index: int, response: str):
        if query_index in self.cache:
            self.cache.move_to_end(query_index)
        self.cache[query_index] = response
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def get_response(self, query_index: int) -> str:
        if query_index not in self.cache:
            return None
        self.cache.move_to_end(query_index)
        return self.cache[query_index]
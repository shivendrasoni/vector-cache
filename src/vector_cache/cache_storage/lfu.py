from collections import defaultdict, OrderedDict
from vector_cache.cache_storage.base import CacheStorageInterface


class LFUCache(CacheStorageInterface):
    def __init__(self, capacity: int=500):
        self.capacity = capacity
        self.cache = {}  # stores the query_index: (response, frequency)
        self.freq = defaultdict(OrderedDict)  # stores frequency: {query_index: response}
        self.min_freq = 0

    def set_response(self, query_index: int, response: str):
        if self.capacity == 0:
            return

        if query_index in self.cache:
            # Update the existing response
            _, freq = self.cache[query_index]
            self.freq[freq].pop(query_index)
            if not self.freq[freq]:
                del self.freq[freq]
                if self.min_freq == freq:
                    self.min_freq += 1
            freq += 1
            self.freq[freq][query_index] = response
            self.cache[query_index] = (response, freq)
        else:
            if len(self.cache) >= self.capacity:
                # Evict the least frequently used item
                evict_query_index, _ = self.freq[self.min_freq].popitem(last=False)
                if not self.freq[self.min_freq]:
                    del self.freq[self.min_freq]
                del self.cache[evict_query_index]

            # Insert the new response
            self.cache[query_index] = (response, 1)
            self.freq[1][query_index] = response
            self.min_freq = 1

    def get_response(self, query_index: int) -> str:
        if query_index not in self.cache:
            return None

        response, freq = self.cache[query_index]
        self.freq[freq].pop(query_index)
        if not self.freq[freq]:
            del self.freq[freq]
            if self.min_freq == freq:
                self.min_freq += 1

        freq += 1
        self.freq[freq][query_index] = response
        self.cache[query_index] = (response, freq)
        return response
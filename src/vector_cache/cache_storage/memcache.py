from vector_cache.cache_storage.base import CacheStorageInterface
from pymemcache.client.base import Client

class MemcacheCached(CacheStorageInterface):
    def __init__(self, host: str = 'localhost', port: int = 11211):
        self.client = Client((host, port))

    def set_response(self, query_index: int, response: str):
        self.client.set(str(query_index), response)

    def get_response(self, query_index: int) -> str:
        response = self.client.get(str(query_index))
        if response is not None:
            return response.decode('utf-8')
        return None
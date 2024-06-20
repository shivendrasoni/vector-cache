from src.cache_storage.base import CacheStorageInterface
import redis


class RedisStorage(CacheStorageInterface):
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.Redis(host=host, port=port, db=db)

    def set_response(self, query_index: int, response: str):
        self.r.set(f"response:{query_index}", response)

    def get_response(self, query_index: int) -> str:
        response = self.r.get(f"response:{query_index}")
        return response.decode() if response else None

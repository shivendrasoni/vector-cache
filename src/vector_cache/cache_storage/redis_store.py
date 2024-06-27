from vector_cache.cache_storage.base import CacheStorageInterface
import redis


class RedisStorage(CacheStorageInterface):
    def __init__(self, host='localhost', port=6379, db=0, eviction_policy='volatile-lru', ttl=None):
        self.r = redis.Redis(host=host, port=port, db=db)
        self.eviction_policy = eviction_policy
        self.ttl = ttl
        self._configure_eviction_policy()

    def _configure_eviction_policy(self):
        self.r.config_set('maxmemory-policy', self.eviction_policy)

    def set_response(self, query_index: int, response: str):
        key = f"response:{query_index}"
        self.r.set(key, response)
        if self.ttl is not None:
            self.r.expire(key, self.ttl)

    def get_response(self, query_index: int) -> str:
        response = self.r.get(f"response:{query_index}")
        return response.decode() if response else None

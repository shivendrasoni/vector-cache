from .lfu import LFUCache
from .lru import LRUCache


try:
    from .redis_store import RedisStorage
except ImportError:
    RedisStorage = None

try:
    from .memcache import MemcacheCache
except ImportError:
    MemcacheCache = None

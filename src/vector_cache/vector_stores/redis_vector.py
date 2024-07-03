from typing import Tuple, Union
import numpy as np
from redis import Redis
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from vector_cache.vector_stores.base import VectorStoreInterface
from vector_cache.utils.key_util import get_query_index
from typing import Union, Callable

class RedisVectorStore(VectorStoreInterface):
    def __init__(self, index_name: str, redis_url: str = "redis://localhost:6379", vector_dim: int = 1536, identifier: Union[str, Callable, None] = None):
        self.redis_client = Redis.from_url(redis_url)
        self.index_name = index_name
        self.vector_dim = vector_dim
        self.create_index()
        self.identifier = identifier

    def create_index(self):
        try:
            self.redis_client.ft(self.index_name).info()
        except:
            schema = (
                VectorField("vector", "HNSW", {"TYPE": "FLOAT32", "DIM": self.vector_dim, "DISTANCE_METRIC": "COSINE"}),
            )
            self.redis_client.ft(self.index_name).create_index(
                schema,
                definition=IndexDefinition(prefix=[f"{self.index_name}:"], index_type=IndexType.HASH)
            )

    def add(self, embedding: Union[list, np.ndarray], **kwargs) -> str:
        vector_id = get_query_index(self.identifier)

        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        elif not isinstance(embedding, list):
            raise ValueError("Embedding must be a list or numpy array.")

        # Normalize the vector to unit length
        embedding_np = np.array(embedding, dtype=np.float32)
        embedding_normalized = embedding_np / np.linalg.norm(embedding_np)

        key = f"{self.index_name}:{vector_id}"

        try:
            self.redis_client.hset(key, mapping={
                "vector": embedding_normalized.tobytes()
            })
        except Exception as e:
            raise RuntimeError(f"Failed to add embedding to Redis: {str(e)}")

        return vector_id

    def search(self, embedding: Union[list, np.ndarray], top_n: int = 1, include_distances: bool = True, **kwargs) -> Tuple[list, list]:
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        elif not isinstance(embedding, list):
            raise ValueError("Embedding must be a list or numpy array.")

        # Normalize the query vector
        embedding_np = np.array(embedding, dtype=np.float32)
        embedding_normalized = embedding_np / np.linalg.norm(embedding_np)
        query_vector = embedding_normalized.tobytes()

        try:
            query = (
                Query(f"*=>[KNN {top_n} @vector $vector AS distance]")
                .sort_by("distance")
                .paging(0, top_n)
                .dialect(2)
            )
            results = self.redis_client.ft(self.index_name).search(query, query_params={"vector": query_vector})

            ids = [doc.id.split(":")[-1] for doc in results.docs]
            distances = [1 - float(doc.distance) for doc in results.docs] if include_distances else []

            return ids, distances
        except Exception as e:
            raise RuntimeError(f"Failed to search Redis index: {str(e)}")

    def close(self):
        self.redis_client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
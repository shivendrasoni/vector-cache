from vector_cache.utils.time_utils import time_measurement
from vector_cache.cache_storage.base import CacheStorageInterface
from vector_cache.vector_stores.base import VectorStoreInterface
from vector_cache.embedding.base_embedding import BaseEmbedding


class VectorCache:
    def __init__(self, embedding_model: BaseEmbedding, db: CacheStorageInterface, vector_store: VectorStoreInterface,
                 cosine_threshold, verbose=False):
        self.embedding_model = embedding_model
        self.db = db
        self.vector_store = vector_store
        self.cosine_threshold = cosine_threshold
        self.verbose = verbose

    @time_measurement
    def add_query_to_index(self, query: str, response: str):
        embedding = self.embedding_model.get_embeddings(query)
        cache_key = self.vector_store.add(embedding)
        self.db.set_response(cache_key, response)

    @time_measurement
    async def add_query_to_index_async(self, query: str, response: str):
        embedding = self.embedding_model.get_embeddings(query)
        query_index = self.vector_store.index.get_n_items()
        self.vector_store.add(embedding, query_index)
        self.db.set_response(query_index, response)

    @time_measurement
    def find_similar_queries(self, query: str, search_k: int = 1, include_distances=True):
        embedding = self.embedding_model.get_embeddings(query)
        nearest_indices, distances = self.vector_store.search(embedding, search_k, include_distances)
        if nearest_indices:
            nearest_index = nearest_indices[0]
            distance = distances[0]
            if distance < self.cosine_threshold:  # similarity threshold
                cached_response = self.db.get_response(nearest_index)
                return cached_response, distance
        return None, None


def semantic_cache_decorator(semantic_cache: VectorCache):
    def print_log(log):
        if semantic_cache.verbose:
            print(log)

    def decorator(func):
        def wrapper(query, *args, **kwargs):
            # Try to find a cached response
            cached_response, distance = semantic_cache.find_similar_queries(query)
            if cached_response is not None:
                # If a cached response exists, return it
                print_log(f"Cache Hit: Query: {query}, response: {cached_response} (distance: {distance})")
                return cached_response
            print_log(f"Cache Miss: {query}")
            # If there is no cached response, call the actual function
            response = func(query, *args, **kwargs)

            # Add the query-response pair to the cache
            semantic_cache.add_query_to_index(query, response)

            # Return the actual function's response
            return response
        return wrapper
    return decorator



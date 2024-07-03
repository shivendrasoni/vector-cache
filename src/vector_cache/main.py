from vector_cache.utils.time_utils import time_measurement
from vector_cache.cache_storage.base import CacheStorageInterface
from vector_cache.vector_stores.base import VectorStoreInterface
from vector_cache.embedding.base_embedding import BaseEmbedding
import time
from functools import wraps
from typing import Tuple, Optional
import logging

class VectorCache:
    def __init__(self, embedding_model: BaseEmbedding, db: CacheStorageInterface, vector_store: VectorStoreInterface,
                 initial_similarity_threshold: float = 0.8, target_hit_rate: float = 0.8,
                 min_threshold: float = 0.7, max_threshold: float = 1.99,
                 adjustment_rate: float = 0.01, verbose=False, use_adjustable_threshold=False):
        self.embedding_model = embedding_model
        self.db = db
        self.vector_store = vector_store
        self.similarity_threshold = initial_similarity_threshold
        self.target_hit_rate = target_hit_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.adjustment_rate = adjustment_rate
        self.verbose = verbose
        self.use_adjustable_threshold =  use_adjustable_threshold

        # Metrics for adaptive thresholding
        self.total_queries = 0
        self.cache_hits = 0

    @time_measurement
    def add_query_to_index(self, query: str, response: str, context:str='',):
        embedding = self.get_context_aware_embedding(query, context)
        cache_key = self.vector_store.add(embedding)
        self.db.set_response(cache_key, response)

    def get_context_aware_embedding(self, query: str, context: str):
        augmented_query = f"{context}: {query}"
        return self.embedding_model.get_embeddings(augmented_query)

    def find_similar_queries(self, query: str, context:str = '', search_k: int = 1, include_distances=True) -> Tuple[Optional[str], Optional[float]]:
        self.total_queries += 1
        embedding = self.get_context_aware_embedding(query, context)
        nearest_indices, similarities = self.vector_store.search(embedding, search_k, include_distances)

        result = None
        similarity = None

        if nearest_indices:
            nearest_index = nearest_indices[0]
            similarity = similarities[0]
            if similarity > self.similarity_threshold:  # similarity threshold
                cached_response = self.db.get_response(nearest_index)
                self.cache_hits += 1
                result = cached_response

        if self.use_adjustable_threshold:
            self._adjust_threshold()
        return result, similarity

    def _adjust_threshold(self):
        if self.total_queries == 0:
            return

        current_hit_rate = self.cache_hits / self.total_queries

        if current_hit_rate < self.target_hit_rate:
            # Lower the threshold to increase hits
            self.similarity_threshold = max(self.similarity_threshold - self.adjustment_rate, self.min_threshold)
        else:
            # Raise the threshold to decrease hits
            self.similarity_threshold = min(self.similarity_threshold + self.adjustment_rate, self.max_threshold)

        if self.verbose:
            print(f"Current hit rate: {current_hit_rate:.2f}, Adjusted threshold: {self.similarity_threshold:.4f}")

    def get_stats(self) -> dict:
        hit_rate = self.cache_hits / self.total_queries if self.total_queries > 0 else 0
        stats = {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "hit_rate": hit_rate,
            "current_threshold": self.cosine_threshold
        }
        self.logger.info(f"Current stats: {stats}")
        return stats

def semantic_cache_decorator(semantic_cache: VectorCache):
    def print_log(log):
        if semantic_cache.verbose:
            print(log)

    def decorator(func):
        @wraps(func)
        def wrapper(query, context="", *args, **kwargs):
            # Try to find a cached response
            cached_response, distance = semantic_cache.find_similar_queries(query)

            if cached_response is not None:
                # If a cached response exists, return it
                print(f"Cache Hit: Query: {query}, Context: {context}, Distance: {distance:.4f}")
                return cached_response

            print(f"Cache Miss: Query: {query}, Context: {context}")

            # If there is no cached response, call the actual function
            response = func(query, context, *args, **kwargs)

            # Add the query-response pair to the cache
            semantic_cache.add_query_to_index(query, response, context)

            print_log(f"Function call: Query: {query}, response: {response}")

            # Return the actual function's response
            return response

        return wrapper
    return decorator



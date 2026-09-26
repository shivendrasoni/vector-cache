from vector_cache.utils.time_utils import time_measurement
from vector_cache.cache_storage.base import CacheStorageInterface
from vector_cache.vector_stores.base import VectorStoreInterface
from vector_cache.embedding.base_embedding import BaseEmbedding
from vector_cache.judges.base import BaseJudge, Candidate
from vector_cache.utils.guards import negations_match, numbers_match
import json
import time
from functools import wraps
from typing import Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

ENTRY_MARKER = "__vector_cache__"


def encode_entry(query: str, context: str, response: Any):
    """Store the original query next to the response so a judge can compare against it later.

    Responses that are not JSON serialisable are stored as-is (and cannot be judged).
    """
    try:
        return json.dumps({ENTRY_MARKER: 1, "query": query, "context": context, "response": response})
    except (TypeError, ValueError):
        return response


def decode_entry(raw) -> Tuple[Optional[str], Optional[str], Any]:
    """Returns (query, context, response). Entries written before this format yield (None, None, raw)."""
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except ValueError:
            data = None
        if isinstance(data, dict) and ENTRY_MARKER in data:
            return data["query"], data["context"], data["response"]
    return None, None, raw

class VectorCache:
    def __init__(self, embedding_model: BaseEmbedding, db: CacheStorageInterface, vector_store: VectorStoreInterface,
                 initial_similarity_threshold: float = 0.8, target_hit_rate: float = 0.8,
                 min_threshold: float = 0.7, max_threshold: float = 1.99,
                 adjustment_rate: float = 0.01, verbose=False, use_adjustable_threshold=False,
                 judge: Optional[BaseJudge] = None, judge_band: Tuple[float, float] = (0.80, 0.95),
                 judge_top_k: int = 3, check_cacheable: bool = False):
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

        # Optional verifier for the grey zone between judge_band[0] (below: miss) and judge_band[1] (above: hit).
        # When a judge is set, judge_band replaces similarity_threshold for hit decisions.
        self.judge = judge
        self.judge_band = judge_band
        self.judge_top_k = judge_top_k
        self.check_cacheable = check_cacheable
        self.judge_calls = 0
        self.judge_rejects = 0

        # Metrics for adaptive thresholding
        self.total_queries = 0
        self.cache_hits = 0

    @time_measurement
    def add_query_to_index(self, query: str, response: str, context:str='',):
        if self.check_cacheable and self.judge is not None and not self.judge.is_cacheable(query, context, response):
            if self.verbose:
                print(f"Not cacheable, skipping insert: {query}")
            return
        embedding = self.get_context_aware_embedding(query, context)
        cache_key = self.vector_store.add(embedding)
        self.db.set_response(cache_key, encode_entry(query, context, response))

    def get_context_aware_embedding(self, query: str, context: str):
        augmented_query = f"{context}: {query}"
        return self.embedding_model.get_embeddings(augmented_query)

    def find_similar_queries(self, query: str, context:str = '', search_k: int = 1, include_distances=True) -> Tuple[Optional[str], Optional[float]]:
        self.total_queries += 1
        embedding = self.get_context_aware_embedding(query, context)

        if self.judge is not None:
            nearest_indices, similarities = self.vector_store.search(embedding, max(search_k, self.judge_top_k), True)
            similarity = similarities[0] if nearest_indices else None
            result = self._judged_lookup(query, context, nearest_indices, similarities) if nearest_indices else None
            if result is not None:
                self.cache_hits += 1
            return result, similarity

        nearest_indices, similarities = self.vector_store.search(embedding, search_k, include_distances)

        result = None
        similarity = None

        if nearest_indices:
            nearest_index = nearest_indices[0]
            similarity = similarities[0]
            if similarity > self.similarity_threshold:  # similarity threshold
                _, _, cached_response = decode_entry(self.db.get_response(nearest_index))
                self.cache_hits += 1
                result = cached_response

        if self.use_adjustable_threshold:
            self._adjust_threshold()
        return result, similarity

    def _judged_lookup(self, query: str, context: str, indices: list, similarities: list):
        floor, ceiling = self.judge_band
        candidates = []
        for index, similarity in zip(indices, similarities):
            if similarity < floor:
                break  # results are sorted, the rest are lower
            cached_query, cached_context, response = decode_entry(self.db.get_response(index))
            if cached_query is None or response is None:
                continue  # evicted, or legacy entry without the original query: cannot be verified
            if not numbers_match(query, cached_query):
                continue
            # Near-exact matches skip the judge unless one side negates something the other does not.
            if similarity >= ceiling and negations_match(query, cached_query):
                return response
            candidates.append(Candidate(cached_query, cached_context, response))

        if not candidates:
            return None

        self.judge_calls += 1
        verdict = self.judge.judge(query, context, candidates)
        if verdict.index is None:
            self.judge_rejects += 1
            if verdict.uncertain:
                # Worth reviewing: these pairs are the ones to calibrate judge_band / accept on.
                logger.info("Uncertain judge verdict for %r: %s", query, verdict.probabilities)
            return None
        return candidates[verdict.index].response

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



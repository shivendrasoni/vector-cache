from utils.time_utils import time_measurement
from cache_storage.base import CacheStorageInterface
from vector_stores.base import VectorStoreInterface
from embedding.base_embedding import BaseEmbedding

import asyncio


class SemanticCache:
    def __init__(self, embedding_model: BaseEmbedding, db: CacheStorageInterface, vector_store: VectorStoreInterface,
                 cosine_threshold):
        self.embedding_model = embedding_model
        self.db = db
        self.vector_store = vector_store
        self.cosine_threshold = cosine_threshold

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


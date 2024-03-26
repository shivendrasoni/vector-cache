from sentence_transformers import SentenceTransformer
from utils.time_utils import time_measurement
from cache_storage.base import CacheStorageInterface
from cache_storage.redis_store import RedisStorage
from vector_stores.base import VectorStoreInterface
from vector_stores.annoy_store import AnnoyStore
from vector_stores.chroma_db import ChromeDB
from embedding.base_embedding import BaseEmbedding
from embedding.sentence_bert import SentenceBertEmbeddings
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

model = SentenceBertEmbeddings()
db = RedisStorage()
embedding_size = model.dimension
# Initialize components

# vector_store = AnnoyStore(embedding_size)
vector_store = ChromeDB()
semantic_cache = SemanticCache(model, db, vector_store, cosine_threshold=0.8)


# Usage
query_response_pairs = [
    {'query': "What is the capital of France?", 'response': "The capital of France is Paris."},
    {'query': "What is the largest ocean on Earth?", 'response': "The largest ocean on Earth is the Pacific Ocean."},
    {'query': "Who wrote 'Hamlet'?", 'response': "William Shakespeare wrote 'Hamlet'."},
    {'query': "What is the speed of light?",
     'response': "The speed of light is approximately 299,792 kilometers per second."},
    {'query': "Who is known as the father of computers?",
     'response': "Charles Babbage is known as the father of computers."},
    {'query': "What is the chemical symbol for water?", 'response': "The chemical symbol for water is H2O."},
    {'query': "When did the first man land on the moon?",
     'response': "The first man landed on the moon on July 20, 1969."},
    {'query': "What is the tallest mountain in the world?",
     'response': "Mount Everest is the tallest mountain in the world."},
    {'query': "Who painted the Mona Lisa?", 'response': "Leonardo da Vinci painted the Mona Lisa."},
    {'query': "What is the capital of Japan?", 'response': "The capital of Japan is Tokyo."}
]

for pair in query_response_pairs:
    query = pair['query']
    response = pair['response']
    semantic_cache.add_query_to_index(query, response)

# vector_store.build(num_trees=10)

similar_queries = [
    "What's the main city of France?",
    "Which ocean is the biggest on the planet?",
    "Who is the author of 'Hamlet'?",
    "How fast does sound travel?",
    "Who is considered the pioneer of computing?",
    "What's the formula for water?",
    "When did humans first step on the moon?",
    "What is the highest peak on Earth?",
    "Who created the Starry Night?",
    "What's the capital city of Japan?"
]

for query in similar_queries:
    cached_response, distance = semantic_cache.find_similar_queries(query)
    if cached_response:
        print(f"Cache Hit: Query: {query}, response: {cached_response} (distance: {distance})")
    else:
        print("Cache Miss: ", query)

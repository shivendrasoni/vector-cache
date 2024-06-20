from src.vector_stores.chroma_db import ChromeDB
from src.embedding.openai import OpenAIEmbeddings
from src.cache_storage.redis_store import RedisStorage
from src.main import SemanticCache

model = OpenAIEmbeddings()
db = RedisStorage()
embedding_size = model.dimension
# Initialize components

vector_store = ChromeDB()
semantic_cache = SemanticCache(model, db, vector_store, cosine_threshold=0.2)


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
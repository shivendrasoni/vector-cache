from vector_cache import VectorCache, semantic_cache_decorator
from vector_cache.vector_stores import ChromeDB
from vector_cache.embedding import SentenceBertEmbeddings
from vector_cache.cache_storage import RedisStorage

model = SentenceBertEmbeddings()
db = RedisStorage()
embedding_size = model.dimension
# Initialize components

vector_store = ChromeDB()
semantic_cache = VectorCache(model, db, vector_store, cosine_threshold=0.8)
semantic_cache.verbose = False


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

@semantic_cache_decorator(semantic_cache)
def call_llm(query, **kwargs):
    if 'response' in kwargs:
        response = kwargs['response']
    else:
        response = 'Garbage sample response'
    return query, response


# Add the query-response pairs to the cache
for pair in query_response_pairs:
    call_llm(query = pair['query'], response = pair['response'])

# vector_store.build(num_trees=10)

# Get similar queries to the queries with few guaranteed miss cases?"
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

for q in similar_queries:
    call_llm(query=q)
from vector_cache import VectorCache, semantic_cache_decorator
from vector_cache.cache_storage import LRUCache
from vector_cache.embedding.openai import OpenAIEmbeddings
from vector_cache.judges import JevJudge
from vector_cache.vector_stores import ChromaDB
import openai
import os

try:
    from dotenv import load_dotenv  # optional: pip install python-dotenv
    load_dotenv()
except ImportError:
    pass

# Needs OPENAI_API_KEY and TYPESAFE_API_KEY in the environment or .env (see .env.example).
# pip install "vector-cache[typesafe]"
# Similarity >= 0.95 is served directly, < 0.80 is a miss, anything in between is verified by Jev.
semantic_cache = VectorCache(
    embedding_model=OpenAIEmbeddings(),
    db=LRUCache(capacity=1000),
    vector_store=ChromaDB(),
    judge=JevJudge(mode="query", accept=0.9),
    judge_band=(0.80, 0.95),
    judge_top_k=3,
    check_cacheable=True,
    verbose=True,
)

openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@semantic_cache_decorator(semantic_cache)
def chat_completion(query, **kwargs):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": query}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(chat_completion("How do I make a vegan chocolate cake?"))
    print(chat_completion("What's a recipe for chocolate cake without animal products?"))  # paraphrase: judge hit
    print(chat_completion("How do I make a chocolate cake with eggs and butter?"))  # near-miss: judge rejects
    print(chat_completion("Best laptops under $500"))
    print(chat_completion("Best laptops under $300"))  # numeric guard: miss without calling Jev
    print(f"judge calls: {semantic_cache.judge_calls}, rejects: {semantic_cache.judge_rejects}")

from vector_cache import VectorCache, semantic_cache_decorator
from vector_cache.vector_stores import ChromaDB
from vector_cache.embedding import OpenAIEmbeddings
from vector_cache.cache_storage import RedisStorage
from openai import OpenAI
import os
embedding_model = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = RedisStorage()
import time
# Initialize components
key_generator = lambda: f"inventory_{str(int(round(time.time() * 1000)))}"
vector_store = ChromaDB(persistent=True, identifier=key_generator)
#lambda to generate key using time in milliseconds



semantic_cache = VectorCache(embedding_model, db, vector_store, cosine_threshold=0.9, verbose=True)


@semantic_cache_decorator(semantic_cache)
def chat_completion(query, **kwargs):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{query}"},
        ]
    )

    return response.choices[0].message.content


print(chat_completion("What is the speed of light?"))

print(chat_completion("How fast is light in vaccum?"))
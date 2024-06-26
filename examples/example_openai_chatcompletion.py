from vector_cache import VectorCache, semantic_cache_decorator
from vector_cache.vector_stores import ChromeDB
from vector_cache.embedding import OpenAIEmbeddings
from vector_cache.cache_storage import RedisStorage
from openai import OpenAI
import os
model = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = RedisStorage()
embedding_size = model.dimension
# Initialize components

vector_store = ChromeDB(persistent=True)
semantic_cache = VectorCache(model, db, vector_store, cosine_threshold=0.9, verbose=True)


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
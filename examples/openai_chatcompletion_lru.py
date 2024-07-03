from vector_cache import VectorCache, semantic_cache_decorator
from vector_cache.vector_stores import ChromaDB
from vector_cache.embedding import OpenAIEmbeddings
from vector_cache.cache_storage import LRUCache
from openai import OpenAI
import os
embedding_model = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = LRUCache()
# Initialize components

vector_store = ChromaDB(persistent=True, collection="chroma_example")
semantic_cache = VectorCache(embedding_model= embedding_model, db=db, vector_store=vector_store,  verbose=True, initial_similarity_threshold=0.86)


@semantic_cache_decorator(semantic_cache)
def chat_completion(query, **kwargs):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{query}"},
        ]
    )

    return response.choices[0].message.content


print("------- Section 1 ----------")
print(chat_completion("When did India gain freedom?"))
print(chat_completion("Who is known as the father of modern physics?"))
print(chat_completion("What is the capital of France?"))



print(' ----------- section 2-------------')
print(chat_completion("In which year did India become independent?"))
print(chat_completion("Which scientist developed the theory of relativity?"))


print(' ----------- section 3-------------')
print(chat_completion("What happened to India in 1947?"))
print(chat_completion("Who formulated the equation E=mc^2?"))




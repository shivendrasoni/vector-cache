from vector_cache import VectorCache, semantic_cache_decorator
from vector_cache.vector_stores import ChromaDB
from vector_cache.embedding import OpenAIEmbeddings
from vector_cache.cache_storage import LRUCache
from openai import OpenAI
import os
# Initialize components
embedding_model = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = LRUCache()
vector_store = ChromaDB(persistent=True, collection="animal_facts")
semantic_cache = VectorCache(embedding_model=embedding_model, db=db, vector_store=vector_store, verbose=True, initial_similarity_threshold=0.9)

@semantic_cache_decorator(semantic_cache)
def chat_completion(query, context="", **kwargs):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant with knowledge about animals."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {query}"},
        ]
    )

    return response.choices[0].message.content

def ask_question(question, animal):
    print(f"\nQuestion about {animal}: {question}")
    response = chat_completion(question, context=animal)
    print(f"Answer: {response}\n")

print("------- Section 1: Initial Questions ----------")
ask_question("What do they eat?", "Elephants")
ask_question("Where do they live?", "Tigers")
ask_question("How long do they live?", "Parrots")

print("------- Section 2: Similar Questions, Different Animals ----------")
ask_question("What is their diet?", "Tigers")
ask_question("What is their habitat?", "Elephants")
ask_question("What is their lifespan?", "Tigers")

print("------- Section 3: Rephrased Questions ----------")
ask_question("What kind of food do they consume?", "Elephants")
ask_question("In what regions can they be found?", "Tigers")
ask_question("How many years can they survive?", "Parrots")
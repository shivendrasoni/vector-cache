from vector_cache import VectorCache, semantic_cache_decorator
from vector_cache.vector_stores import ChromaDB
from vector_cache.embedding import OpenAIEmbeddings
from vector_cache.cache_storage import RedisStorage
from openai import OpenAI
import os
embedding_model = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = RedisStorage()
# Initialize components

vector_store = ChromaDB(persistent=True, collection="chroma_example")
semantic_cache = VectorCache(embedding_model= embedding_model, db=db, vector_store=vector_store,  verbose=True)


@semantic_cache_decorator(semantic_cache)
def chat_completion(query, **kwargs):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4",
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
print(chat_completion("What is the chemical symbol for water?"))
print(chat_completion("Who wrote 'Romeo and Juliet'?"))
print(chat_completion("What is the speed of light?"))
print(chat_completion("What is the tallest mountain in the world?"))
print(chat_completion("Who was the first president of the United States?"))
print(chat_completion("What is the boiling point of water?"))
print(chat_completion("What is the largest planet in our solar system?"))

print(' ----------- section 2-------------')
print(chat_completion("In which year did India become independent?"))
print(chat_completion("Which scientist developed the theory of relativity?"))
print(chat_completion("Which city is known as the City of Light?"))
print(chat_completion("How is H2O commonly known?"))
print(chat_completion("Which playwright created the character Romeo?"))
print(chat_completion("How fast does light travel in a vacuum?"))
print(chat_completion("Which peak is the highest on Earth?"))
print(chat_completion("Who was the inaugural president of the USA?"))
print(chat_completion("At what temperature does water boil?"))
print(chat_completion("Which planet is the biggest in our solar system?"))

print(' ----------- section 3-------------')
print(chat_completion("What happened to India in 1947?"))
print(chat_completion("Who formulated the equation E=mc^2?"))
print(chat_completion("Where is the Eiffel Tower located?"))
print(chat_completion("What is the molecular formula of water?"))
print(chat_completion("Who is the author of the play 'Romeo and Juliet'?"))
print(chat_completion("What is the velocity of light?"))
print(chat_completion("Where is Mount Everest located?"))
print(chat_completion("Which leader was the first president of America?"))
print(chat_completion("When does water reach its boiling point?"))
print(chat_completion("Which is the largest planet orbiting the Sun?"))



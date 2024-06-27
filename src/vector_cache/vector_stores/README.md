# Vector Store Adapters for Vector Cache

This module provides adapters for various vector stores to be used with the Vector Cache library. These adapters allow you to easily integrate different vector databases into your semantic caching system for LLM applications.

## Available Vector Store Adapters

1. ChromaDB
2. PG Vector (PostgreSQL with pgvector)
3. Qdrant
4. Pinecone
5. Redis

## Installation and Usage

The Vector Cache library uses optional dependencies for each vector store. Below are instructions for installing and using each vector store adapter.

### ChromaDB

```bash
pip install vector-cache[chromadb]
```

```python
from vector_cache import VectorCache
from vector_cache.vector_stores import ChromaDB
from vector_cache.embedding import OpenAIEmbeddings
from vector_cache.cache_storage import RedisStorage
import os

embedding_model = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = RedisStorage()

embedding_model = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = RedisStorage()

vector_store = ChromaDB(persistent=True)
semantic_cache = VectorCache(embedding_model, db, vector_store, cosine_threshold=0.9, verbose=True)
```

### PG Vector

```bash
pip install vector-cache[pgvector]
```

```python
from vector_cache import VectorCache
from vector_cache.vector_stores import PGVector
from vector_cache.embedding import OpenAIEmbeddings
from vector_cache.cache_storage import RedisStorage
import os

embedding_model = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = RedisStorage()

connection_string = "postgresql://user:password@localhost/dbname"
vector_store = PGVector(connection_string, table_name="vector_store")
semantic_cache = VectorCache(embedding_model, db, vector_store, cosine_threshold=0.9, verbose=True)
```

### Qdrant

```bash
pip install vector-cache[qdrant]
```

```python
from vector_cache import VectorCache
from vector_cache.vector_stores import QdrantStore
from vector_cache.embedding import OpenAIEmbeddings
from vector_cache.cache_storage import RedisStorage
import os

embedding_model = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = RedisStorage()

vector_store = QdrantStore(collection_name="my_collection", host="localhost", port=6333)
semantic_cache = VectorCache(embedding_model, db, vector_store, cosine_threshold=0.9, verbose=True)
```

### Pinecone

```bash
pip install vector-cache[pinecone]
```

```python
from vector_cache import VectorCache
from vector_cache.vector_stores import PineconeVectorStore
from vector_cache.embedding import OpenAIEmbeddings
from vector_cache.cache_storage import RedisStorage
import os

embedding_model = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = RedisStorage()

vector_store = PineconeVectorStore(index_name="my_index", api_key="your_api_key", environment="us-west1-gcp")
semantic_cache = VectorCache(embedding_model, db, vector_store, cosine_threshold=0.9, verbose=True)
```

### Redis

```bash
pip install vector-cache[redis]
```

```python
from vector_cache import VectorCache
from vector_cache.vector_stores import RedisVectorStore
from vector_cache.embedding import OpenAIEmbeddings
from vector_cache.cache_storage import RedisStorage
import os

embedding_model = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = RedisStorage()

vector_store = RedisVectorStore(index_name="my_index", redis_url="redis://localhost:6379", vector_dim=1536)
semantic_cache = VectorCache(embedding_model, db, vector_store, cosine_threshold=0.9, verbose=True)
```
To install with support for more than one vectore stores:

```bash
pip install vector-cache[chromadb,pgvector,cohere]
````

To install with support for all vector stores:

pip install vector-cache[all]
```

## Common Usage Pattern

Regardless of the chosen vector store, the usage pattern remains consistent:

1. Install the Vector Cache library with the desired vector store support.
2. Import the desired vector store adapter.
3. Initialize the vector store adapter with appropriate parameters.
4. Create a `VectorCache` instance using the adapter.
5. Use the `semantic_cache` in your LLM application.

Example:

```python
from vector_cache import VectorCache, semantic_cache_decorator
from vector_cache.vector_stores import ChromaDB  # Or any other vector store
from openai import OpenAI
from vector_cache.embedding import OpenAIEmbeddings
from vector_cache.cache_storage import RedisStorage
import os

embedding_model = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
db = RedisStorage()

# Initialize the vector store
vector_store = ChromaDB(persistent=True)

# Create the semantic cache
semantic_cache = VectorCache(embedding_model, db, vector_store, cosine_threshold=0.9, verbose=True)

# Use the semantic cache in your application
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
```

## Notes

- Ensure that you have the necessary services running (e.g., PostgreSQL, Qdrant server, Redis) when using the respective vector stores.
- For Pinecone, you need to sign up for an account and obtain an API key.
- The `model` and `db` parameters in the `VectorCache` constructor should be provided based on your specific LLM and database setup.

## Implementing New Vector Store Adapters

If you want to implement a new vector store adapter, create a new class that inherits from `VectorStoreInterface` and implements the following methods:

- `__init__`: Initialize the vector store connection.
- `add`: Add a new embedding to the vector store.
- `search`: Search for similar embeddings in the vector store.
- `close`: Close the vector store connection (if applicable).

Refer to the existing implementations for guidance on how to structure your new adapter.

For more detailed information on each vector store and advanced usage, please refer to the individual documentation for each adapter.
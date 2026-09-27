# VectorCache: A Python Library for Efficient LLM Query Caching

A streamlined Python library that enhances LLM query performance through semantic caching, making responses faster and more cost-effective.

![Pypi Build](https://github.com/shivendrasoni/vector-cache/actions/workflows/publish.yml/badge.svg)

## What is VectorCache?

As AI applications gain traction, the costs and latency of using large language models (LLMs) can escalate. VectorCache addresses these issues by caching LLM responses based on semantic similarity, thereby reducing both costs and response times.

### 🤠 Overview

Semantic Vector Cache leverages the power of LLMs to provide two main advantages:

1. **Faster Responses**: By caching, it significantly reduces latency, offering quicker feedback to user queries.
2. **Lower Costs**: It minimizes the number of direct LLM requests, thereby saving on usage costs.

### 🤔 Benefits

VectorCache, akin to a more nuanced Redis, enables efficient caching by recognizing not just exact matches but also semantically similar queries. This efficiency is particularly useful in domains where queries within a specific topic or field are frequent.

### 😊 Getting Started

**Prerequisites**:
- Ensure Python version is **3.9 or higher** (`python --version`).
- `pip install vector-cache`

NOTE: vector-cache comes with support for multiple vector stores and storage databases, which are optional installs. [Read here for more information](./src/vector_cache/vector_stores/README.md)

- Install from source
  - For library installation issues, upgrade pip: `python -m pip install -r requirements.txt`.

- ### Initialise vector cache:
**Initialization Parameters for Vector Cache**

The `VectorCache` class can be initialized with the following parameters:

- `embedding_model` (`BaseEmbedding`): The embedding model used to convert text or other data into vector embeddings. This must be an instance of a class that inherits from `BaseEmbedding`.

  - `db` (`CacheStorageInterface`): The storage interface for caching and retrieving vectors. This must be an instance of a class that inherits from `CacheStorageInterface`.

  - `vector_store` (`VectorStoreInterface`): The vector store interface for storing and searching vectors. This must be an instance of a class that inherits from `VectorStoreInterface`.

  - `initial_similarity_threshold` (`float`, optional, default=0.99): The initial similarity threshold used to determine if a cached vector is considered a hit.

  - `target_hit_rate` (`float`, optional, default=0.8): The target hit rate for cache queries. This is used if `use_adjustable_threshold` is set to `True` to dynamically adjust the similarity threshold.

  - `min_threshold` (`float`, optional, default=0.7): The minimum allowed similarity threshold for adaptive thresholding.

  - `max_threshold` (`float`, optional, default=1.99): The maximum allowed similarity threshold for adaptive thresholding.

  - `adjustment_rate` (`float`, optional, default=0.01): The rate at which the similarity threshold is adjusted to achieve the target hit rate.

  - `verbose` (`bool`, optional, default=False): If set to `True`, enables verbose logging for debugging purposes.

  - `use_adjustable_threshold` (`bool`, optional, default=False): If set to `True`, enables dynamic adjustment of the similarity threshold based on the target hit rate.

### Metrics for Adaptive Thresholding

- `total_queries` (`int`): The total number of queries made to the vector cache. This is used internally to track performance metrics.
  - `cache_hits` (`int`): The number of cache hits. This is used internally to track performance metrics and adjust the similarity threshold if `use_adjustable_threshold` is enabled.

  ### Example
  
  ```python
  vector_cache = VectorCache(
      embedding_model=my_embedding_model,
      db=my_cache_storage,
      vector_store=my_vector_store,
      initial_similarity_threshold=0.95,
      target_hit_rate=0.85,
      min_threshold=0.75,
      max_threshold=1.0,
      adjustment_rate=0.02,
      verbose=True,
      use_adjustable_threshold=True
  )


- Refer to the `examples` folder for sample usage.

VectorCache is designed to work with any LLM provider. It includes modules for:
- **Embedding Models**: Facilitates similarity searches through various vector_cache.embedding APIs.
- **Cache Storage**: Stores LLM responses for future retrieval based on semantic matches.
- **Vector Store**: Identifies similar requests using the input request's embeddings.
- **Cache Manager**: Manages cache storage and vector store operations, including eviction policies.
- **Similarity Evaluator**: Determines the similarity between requests to ascertain cache matches.

** Features: ** 
- Adaptive Threshold (Alpha) : The vector cache can now incrementally adjust the similarity threshold till a desired hit rate is met. This should be used with caution as the high threshold can cause non-relevant matches to be returned.  

### ⚖️ Verifying hits with a System One judge (optional)

Embedding similarity is a fuzzy key: "cake with gluten" and "cake without gluten" can score above 0.95.
A plain threshold then serves the wrong answer. A judge adds the equality check a normal cache gets for free:
the vector store shortlists candidates, and a fast System One decision model confirms the match before a hit is served.

The default judge is [TypeSafe Jev](https://docs.typesafe.ai/introduction) (vendor-quoted ~100 ms per call).
Any other System One model (for example Laaya) can be used: point `TYPESAFE_API_URL` / `TYPESAFE_MODEL` at it if it
is served behind the same `/v1/systemone` request shape, otherwise subclass `vector_cache.judges.BaseJudge`.

```bash
pip install "vector-cache[typesafe]"
cp .env.example .env   # fill in TYPESAFE_API_KEY (and optionally TYPESAFE_MODEL / TYPESAFE_API_URL)
```

```python
from vector_cache.judges import JevJudge

vector_cache = VectorCache(
    embedding_model=my_embedding_model,
    db=my_cache_storage,
    vector_store=my_vector_store,
    judge=JevJudge(mode="query", accept=0.9),  # mode="answer" judges the cached response instead
    judge_band=(0.80, 0.95),                   # < 0.80 miss, >= 0.95 served directly, in between -> judge
    judge_top_k=3,                             # candidates sent to the judge in one call
    check_cacheable=True,                      # skip caching time-sensitive, personal or failed responses
)
```

How a lookup is decided when a judge is set:
1. Candidates below `judge_band[0]` are a miss.
2. Candidates whose numbers differ from the new query (`$300` vs `$500`, `2024` vs `2025`) are dropped, with no judge call.
3. Candidates at or above `judge_band[1]` are served directly, unless one query has a negation the other lacks; those go to the judge.
4. Everything else goes to the judge in one request. Judge errors and timeouts are treated as misses.

`judge_band` replaces `similarity_threshold` (and adaptive thresholding) when a judge is set.
`judge_calls` and `judge_rejects` on the cache track how often the judge ran and said no.
Entries cached before this feature (no stored query) are never served through the judge path.
See `examples/openai_chatcompletion_jev_judge.py`.

### 🔍 Vector Stores

VectorCache supports multiple vector store options for efficient similarity search. For detailed information on how to use different vector stores, please refer to our [Vector Stores README](src/vector_cache/vector_stores/README.md).

Supported vector stores include:

- [X] Redis
- [X] Qdrant
- [ ] Deeplake
- [X] ChromaDB
- [X] pgvector
- [x] Pinecone
- [ ] Milvus

We're continuously working on expanding our support for other popular vector stores. If you don't see your preferred vector store listed, check our documentation for the most up-to-date information or consider contributing to add support for it!

### 😆 Contributing

Interested in contributing to VectorCache? Check our [contribution guidelines](docs/contributing.md).
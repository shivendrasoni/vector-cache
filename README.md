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

- Install from source
  - For library installation issues, upgrade pip: `python -m pip install -r requirements.txt`.

- Refer to the `examples` folder for sample usage.

VectorCache is designed to work with any LLM provider. It includes modules for:
- **Embedding Models**: Facilitates similarity searches through various vector_cache.embedding APIs.
- **Cache Storage**: Stores LLM responses for future retrieval based on semantic matches.
- **Vector Store**: Identifies similar requests using the input request's embeddings.
- **Cache Manager**: Manages cache storage and vector store operations, including eviction policies.
- **Similarity Evaluator**: Determines the similarity between requests to ascertain cache matches.

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
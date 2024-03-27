# SemanticCache: A Python Library for Efficient LLM Query Caching

A streamlined Python library that enhances LLM query performance through semantic caching, making responses faster and more cost-effective.

## What is SemanticCache?

As AI applications gain traction, the costs and latency of using large language models (LLMs) can escalate. SemanticCache addresses these issues by caching LLM responses based on semantic similarity, thereby reducing both costs and response times.


### 🤠 Overview

Semantic Cache leverages the power of LLMs to provide two main advantages:

1. **Faster Responses**: By caching, it significantly reduces latency, offering quicker feedback to user queries.
2. **Lower Costs**: It minimizes the number of direct LLM requests, thereby saving on usage costs.

### 🤔 Benefits

SemanticCache, akin to a more nuanced Redis, enables efficient caching by recognizing not just exact matches but also semantically similar queries. This efficiency is particularly useful in domains where queries within a specific topic or field are frequent.

### 😊 Getting Started

**Prerequisites**:
- Ensure Python version is **3.8.1 or higher** (`python --version`).
- For library installation issues, upgrade pip: `python -m pip install -r requirements.txt`.

- For now, refer to the `examples` folder for sample usage. Pip package coming soon.

SemanticCache is designed to work with any LLM provider. It includes modules for:
- **Embedding Models**: Facilitates similarity searches through various embedding APIs.
- **Cache Storage**: Stores LLM responses for future retrieval based on semantic matches.
- **Vector Store**: Identifies similar requests using the input request's embeddings.
- **Cache Manager**: Manages cache storage and vector store operations, including eviction policies.
- **Similarity Evaluator**: Determines the similarity between requests to ascertain cache matches.

### 😆 Contributing

Interested in contributing to SemanticCache? Check our [contribution guidelines](docs/contributing.md).

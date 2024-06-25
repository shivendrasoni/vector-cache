from typing import Tuple
import pinecone  # Make sure to install Pinecone client
import numpy as np
import uuid
from vector_cache.vector_stores.base import VectorStoreInterface


class PineconeVectorStore(VectorStoreInterface):
    def __init__(self, index_name: str, api_key: str, environment: str = 'us-west1-gcp'):
        """
        Initialize the Pinecone vector store client.

        Parameters:
        - index_name: The name of the Pinecone index to use.
        - api_key: Your Pinecone API key.
        - environment: The Pinecone environment to connect to.
        """
        pinecone.init(api_key=api_key, environment=environment)
        self.index_name = index_name
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name)
        self.index = pinecone.Index(index_name)

    def add(self, embedding: list, **kwargs):
        """
        Add an vector_cache.embedding to the Pinecone index.

        Parameters:
        - vector_cache.embedding: The vector_cache.embedding to add, as a numpy array.

        Returns:
        - A reference to the index where it's stored (in Pinecone, this is the 'id').
        """
        # Generate a new UUID if an 'id' is not provided in kwargs.
        vector_id = kwargs.get("id", str(uuid.uuid4()))

        # Convert list to numpy array if not already
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        self.index.upsert(vectors=[(vector_id, embedding.tolist())])
        return vector_id

    def search(self, embedding: list, top_n: int = 1, include_distances: bool = True, **kwargs) -> Tuple[list, list]:
        """
        Search for similar embeddings in the Pinecone index.

        Parameters:
        - vector_cache.embedding: The query vector_cache.embedding, as a numpy array.
        - top_n: The number of top similar results to return.
        - include_distances: Whether to include distances in the results.

        Returns:
        - A tuple of two lists: indices of the closest embeddings, and their respective distances.
        """
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        query_result = self.index.query(queries=[embedding.tolist()], top_k=top_n, include_scores=include_distances)
        matches = query_result["matches"]
        ids = [match["id"] for match in matches]
        distances = [match["score"] for match in matches] if include_distances else []
        return ids, distances

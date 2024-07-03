from typing import Tuple, Union
import pinecone
import numpy as np
import uuid
from vector_cache.vector_stores.base import VectorStoreInterface
from typing import Union, Callable
from vector_cache.utils.key_util import get_query_index

class PineconeVectorStore(VectorStoreInterface):
    def __init__(self, index_name: str, api_key: str, environment: str = 'us-west1-gcp', identifier: Union[str, Callable, None] = None):
        """
        Initialize the Pinecone vector store client.

        Parameters:
        - index_name: The name of the Pinecone index to use.
        - api_key: Your Pinecone API key.
        - environment: The Pinecone environment to connect to.
        """
        pinecone.init(api_key=api_key, environment=environment)
        self.index_name = index_name
        self.create_index()
        self.index = pinecone.Index(index_name)
        self.identifier = identifier

    def create_index(self):
        """Create the Pinecone index if it doesn't exist."""
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(self.index_name, dimension=1536, metric="cosine")

    def add(self, embedding: list, **kwargs) -> str:

        """
        Add an embedding to the Pinecone index.

        Parameters:
        - embedding: The embedding to add, as a list or numpy array.
        - identifier: How to identify different_keys, can be prefix string, a function or None (default)
        - **kwargs: Additional keyword arguments.

        Returns:
        - A reference to the index where it's stored (in Pinecone, this is the 'id').
        """
        vector_id = get_query_index(self.identifier)

        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        elif not isinstance(embedding, list):
            raise ValueError("Embedding must be a list or numpy array.")

        try:
            self.index.upsert(vectors=[(vector_id, embedding)])
        except Exception as e:
            raise RuntimeError(f"Failed to add embedding to Pinecone: {str(e)}")

        return vector_id

    def search(self, embedding: Union[list, np.ndarray], top_n: int = 1, include_distances: bool = True, **kwargs) -> Tuple[list, list]:
        """
        Search for similar embeddings in the Pinecone index.

        Parameters:
        - embedding: The query embedding, as a list or numpy array.
        - top_n: The number of top similar results to return.
        - include_distances: Whether to include distances in the results.

        Returns:
        - A tuple of two lists: indices of the closest embeddings, and their respective distances.
        """
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        elif not isinstance(embedding, list):
            raise ValueError("Embedding must be a list or numpy array.")

        try:
            query_result = self.index.query(vector=embedding, top_k=top_n, include_values=False)
            matches = query_result.matches
            ids = [match.id for match in matches]
            distances = [1 - match.score for match in matches] if include_distances else []
            return ids, distances
        except Exception as e:
            raise RuntimeError(f"Failed to search Pinecone index: {str(e)}")

    def close(self):
        """Close the Pinecone client (not strictly necessary, but good practice)."""
        pinecone.deinit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
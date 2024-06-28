from typing import Tuple
import deeplake  # Hypothetical, replace with actual DeepLake client import
import numpy as np
import uuid
from vector_cache.vector_stores.base import VectorStoreInterface
from typing import Union, Callable
from vector_cache.utils.key_util import get_query_index
#WIP : DO NOT USE YET
class DeepLakeVectorStore(VectorStoreInterface):
    def __init__(self, index_name: str, api_key: str, identifier: Union[str, Callable, None] = None):
        """
        Initialize the DeepLake vector store client.

        Parameters:
        - index_name: The name of the DeepLake index to use.
        - api_key: Your DeepLake API key.
        """
        # Dummy initializing the DeepLake client. Replace with actual initialization.
        deeplake.init(api_key=api_key)
        self.index_name = index_name
        self.index = deeplake.Index(index_name)  # Hypothetical way to access an index
        self.identifier = identifier

    def add(self, embedding: list, **kwargs) -> str:
        """
        Add an vector_cache.embedding to the DeepLake index.

        Parameters:
        - vector_cache.embedding: The vector_cache.embedding to add, as a numpy array.

        Returns:
        - A reference to the index where it's stored (in DeepLake, this might be an 'id').
        """
        # Generate a new UUID if an 'id' is not provided in kwargs.
        vector_id = get_query_index(self.identifier)

        # Convert list to numpy array if not already
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Hypothetical method to add a vector to DeepLake. Adapt based on actual API.
        self.index.add_vectors(vectors=[(vector_id, embedding.tolist())])
        return vector_id

    def search(self, embedding: list, top_n: int = 1, include_distances: bool = True, **kwargs) -> Tuple[list, list]:
        """
        Search for similar embeddings in the DeepLake index.

        Parameters:
        - vector_cache.embedding: The query vector_cache.embedding, as a numpy array.
        - top_n: The number of top similar results to return.
        - include_distances: Whether to include distances in the results.

        Returns:
        - A tuple of two lists: indices of the closest embeddings, and their respective distances.
        """
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding)

        # Hypothetical search function. Replace with the actual method DeepLake uses.
        search_results = self.index.search(query=embedding.tolist(), top_k=top_n)

        # Extracting IDs and distances based on hypothetical search result structure.
        ids = [result["id"] for result in search_results["matches"]]
        distances = [result["distance"] for result in search_results["matches"]] if include_distances else []

        return ids, distances

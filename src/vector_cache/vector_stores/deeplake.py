from typing import Tuple, List, Union, Callable, Optional, Dict, Any
import deeplake
import numpy as np
from pathlib import Path
from vector_cache.vector_stores.base import VectorStoreInterface
from vector_cache.utils.key_util import get_query_index

class DeepLakeVectorStore(VectorStoreInterface):
    def __init__(self,
                 path: Optional[Union[str, Path]] = None,
                 embedding_function: Optional[Any] = None,
                 read_only: bool = False,
                 token: Optional[str] = None,
                 index_params: Optional[Dict[str, Union[int, str]]] = None,
                 exec_option: str = 'auto',
                 overwrite: bool = False,
                 runtime: Optional[Dict] = None,
                 creds: Optional[Union[Dict, str]] = None,
                 identifier: Union[str, Callable, None] = None):
        """
        Initialize the DeepLake vector store client.

        Parameters:
        - path: The full path for storing to the Deep Lake Vector Store.
        - embedding_function: Function to convert data into embeddings.
        - read_only: Opens dataset in read-only mode if True.
        - token: Activeloop token for authentication.
        - index_params: Dictionary containing information about vector index.
        - exec_option: Default method for search execution.
        - overwrite: If True, overwrites the Vector Store if it already exists.
        - runtime: Parameters for creating the Vector Store in Deep Lake's Managed Tensor Database.
        - creds: Credentials for accessing the dataset.
        - identifier: Identifier for the vector store.
        """
        self.embedding_function = embedding_function
        self.identifier = identifier

        # Set up index parameters
        if index_params is None:
            index_params = {
                'distance_metric': 'COS',  # Use cosine similarity
                'threshold': 0,  # Always create an index
            }

        # Initialize the DeepLake vector store
        self.vector_store = deeplake.VectorStore(
            path=path,
            embedding_function=embedding_function,
            read_only=read_only,
            index_params=index_params,
            exec_option=exec_option,
            token=token,
            overwrite=overwrite,
            runtime=runtime,
            creds=creds
        )

    def add(self, embedding: List[float], text: str = "", metadata: Dict = None, **kwargs) -> str:
        """
        Add an embedding to the DeepLake vector store.

        Parameters:
        - embedding: The embedding to add, as a list of floats.
        - text: The text associated with the embedding.
        - metadata: Additional metadata to store with the embedding.

        Returns:
        - A reference to the index where it's stored.
        """
        vector_id = get_query_index(self.identifier)

        # Convert list to numpy array if not already
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)

        # Normalize the embedding for cosine similarity
        normalized_embedding = embedding / np.linalg.norm(embedding)

        # Add the embedding to the vector store
        self.vector_store.add(
            embedding=normalized_embedding,
            text=text,
            metadata=metadata or {},
            id=vector_id
        )

        return vector_id

    def search(self, embedding: List[float], top_n: int = 1, include_similarities: bool = True, **kwargs) -> Tuple[List[str], List[float]]:
        """
        Search for similar embeddings in the DeepLake vector store.

        Parameters:
        - embedding: The query embedding, as a list of floats.
        - top_n: The number of top similar results to return.
        - include_similarities: Whether to include similarities in the results.

        Returns:
        - A tuple of two lists: indices of the closest embeddings, and their respective cosine similarities.
        """
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)

        # Normalize the query embedding for cosine similarity
        normalized_embedding = embedding / np.linalg.norm(embedding)

        # Perform the search
        results = self.vector_store.search(
            embedding=normalized_embedding,
            k=top_n,
            exec_option=self.vector_store.exec_option
        )

        ids = [result['id'] for result in results]

        if include_similarities:
            # DeepLake returns cosine similarities directly, no need for conversion
            similarities = [result['score'] for result in results]
        else:
            similarities = []

        return ids, similarities
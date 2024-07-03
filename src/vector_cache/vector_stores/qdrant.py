import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from vector_cache.vector_stores.base import VectorStoreInterface
from typing import Tuple
from typing import Union, Callable
from vector_cache.utils.key_util import get_query_index

class QdrantVectorStore(VectorStoreInterface):
    def __init__(self, collection_name: str = "default_collection", host: str = "localhost", port: int = 6333, identifier: Union[str, Callable, None] = None):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name
        self.create_collection()
        self.identifier = identifier

    def create_collection(self):
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
            )
        except Exception as e:
            if "already exists" not in str(e):
                raise

    def add(self, embedding: list,  **kwargs) -> str:
        vector_id = get_query_index(self.identifier)

        self.client.upsert(
            collection_name=self.collection_name,
            points=[models.PointStruct(id=id, vector=embedding)]
        )
        return id

    def search(self, embedding: list, top_n: int = 1, include_distances=True, **kwargs) -> Tuple[list, list]:
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_n
        )

        ids = [str(point.id) for point in search_result]

        if include_distances:
            distances = [point.score for point in search_result]
            return ids, distances
        else:
            return (ids,)

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
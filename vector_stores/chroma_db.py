import uuid

from chromadb import QueryResult

from vector_stores.base import VectorStoreInterface
from typing import Tuple, List
import chromadb
class ChromeDB(VectorStoreInterface):

    default_collection = 'default_collection'

    def __init__(self, collection: str = default_collection) -> None:
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name=collection)

    def add(self, embedding: list, **kwargs) -> None:
        id = str(uuid.uuid4())
        embedding_list = embedding.tolist()
        self.collection.add(ids=[id], embeddings=[embedding_list])
        return id

    def get(self, id: str) -> Tuple[str, float]:
        pass

    def search(self, embedding: list, top_k: int, include_distances:bool = True, **kwargs) -> Tuple[list, list]:
        embedding_list = embedding.tolist()

        if include_distances:
            query_result: QueryResult = self.collection.query(query_embeddings=[embedding_list], n_results=top_k, include=["distances"])
            return query_result['ids'][0], query_result['distances'][0]
        else:
            query_result: QueryResult = self.collection.query(query_embeddings=[embedding_list], n_results=top_k, include=["distances"])
            return query_result['ids'][0]


    def __enter__(self) -> "ChromeDB":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
from vector_stores.base import VectorStoreInterface
from annoy import AnnoyIndex
from typing import Tuple
class AnnoyStore(VectorStoreInterface):
    def __init__(self, embedding_size: int, metric='angular', index_file='query_index.ann'):
        self.index = AnnoyIndex(embedding_size, metric)
        self.index_file = index_file

    def add(self, embedding: list, **kwargs):
        query_index = self.index.get_n_items()
        self.index.add_item(query_index, embedding)
        return query_index

    def build(self, num_trees: int = 10):
        self.index.build(num_trees)
        self.index.save(self.index_file)

    def search(self, embedding: list, top_n: int = 1, include_distances=True) -> Tuple[list, list]:
        self.index.load(self.index_file)
        return self.index.get_nns_by_vector(vector=embedding, n=top_n, include_distances=include_distances)


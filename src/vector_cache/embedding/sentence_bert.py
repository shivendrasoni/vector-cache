from vector_cache.embedding.base_embedding import BaseEmbedding
from sentence_transformers import SentenceTransformer


class SentenceBertEmbeddings(BaseEmbedding):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self._dimension = None
        self.model_to_dimension_mapping = {"sentence-transformers/all-MiniLM-L6-v2": 384}
    
    def get_embeddings(self, text):
        return self.model.encode([text])[0]

    @property
    def dimension(self):
        """
        The vector_cache.embedding dimension based on the model.

        :return: An integer representing the size of the embeddings.
        """
        if self._dimension is None:
            # Check if the model's dimension is known
            if self.model_name in self.model_to_dimension_mapping:
                self._dimension = self.model_to_dimension_mapping[self.model_name]
            else:
                # Calculate dimension dynamically for unknown models
                sample_embedding = self.get_embeddings("sample")
                self._dimension = len(sample_embedding)
                # Optionally, update the mapping for future reference
                self.model_to_dimension_mapping[self.model_name] = self._dimension
        return self._dimension


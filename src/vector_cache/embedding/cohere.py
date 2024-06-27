from vector_cache.embedding.base_embedding import BaseEmbedding
import cohere

class CohereEmbeddings(BaseEmbedding):
    def __init__(self, api_key, model_name="embed-english-v3.0"):
        """
        Initialize the Cohere Embeddings with the desired model.

        :param api_key: Your Cohere API key.
        :param model_name: The model to use for generating embeddings.
        """
        self.client = cohere.Client(api_key)
        self.model_name = model_name
        self._dimension = None  # Lazy-loaded embedding dimension

        # Model to dimension mapping
        self.model_to_dimension_mapping = {
            "embed-english-v3.0": 1024,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-v3.0": 1024,
            "embed-english-v2.0": 4096,
            "embed-english-light-v2.0": 1024,
            "embed-multilingual-v2.0": 768
        }

    def get_embeddings(self, text, **kwargs):
        """
        Generate embeddings for a given text using Cohere's API.

        :param text: A string or a list of strings for which to generate embeddings.
        :return: The generated embeddings.
        """
        if isinstance(text, str):
            text = [text]

        response = self.client.embed(
            texts=text,
            model=self.model_name,
            input_type="search_document"
        )

        embeddings = response.embeddings
        return embeddings[0] if len(embeddings) == 1 else embeddings

    @property
    def dimension(self):
        """
        The embedding dimension based on the model.

        :return: An integer representing the size of the embeddings.
        """
        if self._dimension is None:
            if self.model_name in self.model_to_dimension_mapping:
                self._dimension = self.model_to_dimension_mapping[self.model_name]
            else:
                # Calculate dimension dynamically for unknown models
                sample_embedding = self.get_embeddings("sample")
                self._dimension = len(sample_embedding)
                # Update the mapping for future reference
                self.model_to_dimension_mapping[self.model_name] = self._dimension
        return self._dimension

    def set_model(self, model_name):
        """
        Change the model used for generating embeddings.

        :param model_name: The new model name to use.
        """
        self.model_name = model_name
        # Invalidate the cached embedding dimension
        self._dimension = None
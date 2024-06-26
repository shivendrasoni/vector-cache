from vector_cache.embedding.base_embedding import BaseEmbedding
import openai
from openai.types import CreateEmbeddingResponse, Embedding
from openai import OpenAI

class OpenAIEmbeddings(BaseEmbedding):
    def __init__(self, api_key, model_name="text-embedding-ada-002"):
        """
        Initialize the OpenAI Embeddings with the desired model.

        :param api_key: Your OpenAI API key.
        :param model_name: The model to use for generating embeddings.
        """
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self._dimension = None  # Lazy-loaded vector_cache.embedding dimension

        # TODO: Move this to config or constants
        self.model_to_dimension_mapping = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }

    def get_embeddings(self, text, **kwargs):
        """
        Generate embeddings for a given text using OpenAI's API.

        :param text: A string or a list of strings for which to generate embeddings.
        :return: The generated embeddings.
        """
        text = text.replace("\n", " ")
        response = self.client.embeddings.create(input = [text], model=self.model_name)
        if isinstance(text, list):
            return [item.embedding for item in response.data]
        else:
            return response.data[0].embedding


    def get_embedding_dimension(self):
        """
        Dynamically determine and return the vector_cache.embedding dimension.

        :return: An integer representing the size of the embeddings.
        """
        if self._dimension is None:
            # Use a sample text to determine the vector_cache.embedding dimension
            sample_embedding = self.get_embeddings("sample")
            self._dimension = len(sample_embedding)
        return self._dimension

    # Additional methods as required, for example, to change the model dynamically
    def set_model(self, model_name):
        """
        Change the model used for generating embeddings.

        :param model_name: The new model name to use.
        """
        self.model_name = model_name
        # Invalidate the cached vector_cache.embedding dimension
        self._dimension = None


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


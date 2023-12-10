from typing import Any, Callable, Dict, List, Union

from abc import ABC, abstractmethod

import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings

from app.agent.consts import DOCUMENT_SEPERATOR
from app.core.config import settings


class Retriever(ABC):
    """Abstract base class for retrievers."""

    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 5, **kwargs: Any):
        """Return docs most similar to query."""
        raise NotImplementedError


class PineconeRetriever(Retriever):
    """
    A retriever class that uses Pinecone for similarity search.

    Args:
        index_name (str): The name of the Pinecone index.
        embedding_model (Union[Embeddings, Callable]): The embedding model used for generating embeddings.
            Defaults to OpenAIEmbeddings with the provided OpenAI API key.

    Attributes:
        _index (pinecone.Index): The Pinecone index object.
        _embedding_model (Union[Embeddings, Callable]): The embedding model used for generating embeddings.

    Methods:
        get_pinecone_index(index_name: str) -> pinecone.Index:
            Retrieves the Pinecone index with the given name.

        _convert_response_to_string(item: Dict) -> str:
            Converts a response item to a formatted string.

        _combine_documents(responses: List[Dict]) -> str:
            Combines multiple response documents into a single string.

        similarity_search(query: str, top_k: int = 5, **kwargs: Any) -> List[Dict]:
            Performs a similarity search using the given query and returns the top-k matching documents.

        get_relevant_doc_string(query: str, top_k: int = 5) -> str:
            Retrieves the relevant document string for the given query.

    """

    def __init__(
        self,
        index_name: str,
        embedding_model: Union[Embeddings, Callable] = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY),
    ):
        pinecone.init(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENVIRONMENT_REGION)
        self._index = self.get_pinecone_index(index_name=index_name)
        self._embedding_model = embedding_model

    def get_pinecone_index(self, index_name: str):
        """
        Retrieves the Pinecone index with the given name.

        Args:
            index_name (str): The name of the Pinecone index.

        Returns:
            pinecone.Index: The Pinecone index object.

        Raises:
            ValueError: If the index with the given name is not found in the Pinecone project.
        """
        indexes = pinecone.list_indexes()

        if index_name in indexes:
            index = pinecone.Index(index_name)
        elif len(indexes) == 0:
            raise ValueError("No active indexes found in your Pinecone project.")
        else:
            raise ValueError(f"Index '{index_name}' not found in your Pinecone project.")
        return index

    def _convert_response_to_string(self, item: Dict) -> str:
        """
        Converts a response item to a formatted string.

        Args:
            item (Dict): The response item containing metadata.

        Returns:
            str: The formatted string representation of the response item.
        """
        doc = f"""
        page_url: {item['metadata']['page_url']}
        document: {item['metadata']['text']}
        """
        return doc

    def _combine_documents(self, responses: List[Dict]) -> str:
        """
        Combines multiple response documents into a single string.

        Args:
            responses (List[Dict]): The list of response items.

        Returns:
            str: The combined document string.
        """
        docs = [self._convert_response_to_string(response) for response in responses]
        doc_string = DOCUMENT_SEPERATOR.join(docs)
        return doc_string

    def similarity_search(self, query: str, top_k: int = 5, **kwargs: Any) -> List[Dict]:
        """
        Performs a similarity search using the given query and returns the top-k matching documents.

        Args:
            query (str): The query string.
            top_k (int): The number of top matching documents to retrieve. Defaults to 5.
            **kwargs (Any): Additional keyword arguments to be passed to the similarity search.

        Returns:
            List[Dict]: The list of top-k matching documents with metadata.
        """
        embeddings = self._embedding_model.embed_query(query)
        responses = self._index.query([embeddings], top_k=top_k, include_metadata=True)
        return responses

    def get_relevant_doc_string(self, query: str, top_k: int = 5) -> str:
        """
        Retrieves the relevant document string for the given query.

        Args:
            query (str): The query string.
            top_k (int): The number of top matching documents to retrieve. Defaults to 5.

        Returns:
            str: The combined document string of the top-k matching documents.
        """
        responses = self.similarity_search(query=query, top_k=top_k)
        doc_string = self._combine_documents(responses=responses["matches"])
        return doc_string

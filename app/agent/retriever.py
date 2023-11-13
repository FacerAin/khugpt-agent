from typing import Any, Callable, Dict, List, Union

from abc import ABC, abstractmethod

import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings

from app.agent.consts import DOCUMENT_SEPERATOR
from app.core.config import settings


class Retriever(ABC):
    @abstractmethod
    def similarity_search(self, query: str, top_k: int = 10, **kwargs: Any):
        """Return docs most similar to query."""
        raise NotImplementedError


class PineconeRetriever(Retriever):
    pinecone.init(api_key=settings.PINECONE_API_KEY, environment=settings.PINECONE_ENVIRONMENT_REGION)

    def __init__(
        self,
        index_name: str,
        embedding_model: Union[Embeddings, Callable] = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY),
    ):
        self._index = self.get_pinecone_index(index_name=index_name)
        self._embedding_model = embedding_model

    def get_pinecone_index(self, index_name: str):
        indexes = pinecone.list_indexes()

        if index_name in indexes:
            index = pinecone.Index(index_name)
        elif len(indexes) == 0:
            raise ValueError("No active indexes found in your Pinecone project.")
        else:
            raise ValueError(f"Index '{index_name}' not found in your Pinecone project.")
        return index

    def _convert_response_to_string(self, item: Dict) -> str:
        doc = f"""
        page_url: {item['metadata']['page_url']}
        document: {item['metadata']['text']}
        """
        return doc

    def _combine_documents(self, responses: List[Dict]) -> List[str]:
        docs = [self._convert_response_to_string(response) for response in responses]
        doc_string = DOCUMENT_SEPERATOR.join(docs)
        return doc_string

    def similarity_search(self, query: str, top_k: int = 10, **kwargs: Any):
        embeddings = self._embedding_model.embed_query(query)
        responses = self._index.query([embeddings], top_k=top_k, include_metadata=True)
        return responses

    def get_relevant_doc_string(self, query: str, top_k: int = 10):
        responses = self.similarity_search(query=query, top_k=top_k)
        doc_string = self._combine_documents(responses=responses["matches"])
        return doc_string

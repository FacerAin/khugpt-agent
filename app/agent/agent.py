import datetime
import os

from langchain.chat_models import ChatOpenAI

from app.agent.prompts import system_prompt_template
from app.agent.retriever import PineconeRetriever
from app.core.config import settings


class ChatAgent:
    def __init__(self, model_name: str = "gpt-4-1106-preview", index_name: str = "khugpt") -> None:
        self.llm = ChatOpenAI(model_name=model_name, temperature=0, openai_api_key=settings.OPENAI_API_KEY)
        self.retreiver = PineconeRetriever(index_name=index_name)

    def run(self, query: str):
        context = self.retreiver.get_relevant_doc_string(query, top_k=20)
        system_prompt = system_prompt_template.format(
            question=query, context=context, current_date=datetime.datetime.now()
        )
        answer = self.llm.predict(system_prompt)
        return answer

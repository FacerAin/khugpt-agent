import datetime
import os

from langchain.chat_models import ChatOpenAI

from app.agent.context import SAMPLE_CONTEXT
from app.agent.prompts import system_prompt_template
from app.agent.retriever import PineconeRetriever
from app.core.config import settings


class ChatAgent:
    def __init__(self, index_name: str = "khugpt") -> None:
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, openai_api_key=settings.OPENAI_API_KEY)
        self.retreiver = PineconeRetriever(index_name=index_name)

    def run(self, query: str):
        context = self.retreiver.get_relevant_doc_string(query)
        system_prompt = system_prompt_template.format(
            question=query, context=context, current_date=datetime.datetime.now()
        )
        answer = self.llm.predict(system_prompt)
        return answer

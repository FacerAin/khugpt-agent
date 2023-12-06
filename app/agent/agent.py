import datetime

from langchain.agents import AgentExecutor, LLMSingleActionAgent, OpenAIFunctionsAgent, Tool
from langchain.cache import InMemoryCache
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_core.messages import SystemMessage

from app.agent.parser import CustomAgentOutputParser
from app.agent.prompts import retriever_prompt_template, system_message
from app.agent.retriever import PineconeRetriever
from app.agent.tools import get_meal_info, get_today_date
from app.core.config import settings

set_llm_cache(InMemoryCache())


class ExecutorAgent:
    def __init__(self):
        self.retriever = PineconeRetriever(index_name="khugpt")

        self.llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=settings.OPENAI_API_KEY)
        self.tools = [
            Tool(
                name="retreiver",
                func=self.retriever.get_relevant_doc_string,
                description="""useful for when you need to answer questions about campus internal information.
                Please input the information in the form of a question that is easy to search for.
                Since the database is stored in Korean, please ask in Korean.""",
            ),
            Tool(
                name="cafeterial_menu",
                func=get_meal_info,
                description="If a user is looking for campus cafeterial menu information, use this information.",
            ),
            Tool(
                name="today_date",
                func=get_today_date,
                description="If you’re curious about today’s date, you can use this. It returns today’s date as a string in the format of YYYY-mm-dd.",
            ),
        ]

        self.agent_prompt = OpenAIFunctionsAgent.create_prompt(system_message=SystemMessage(content=system_message))
        self.output_parser = CustomAgentOutputParser()
        self.agent = OpenAIFunctionsAgent(llm=self.llm, tools=self.tools, prompt=self.agent_prompt)

        self.executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=self.tools, verbose=True, max_iterations=2
        )

    def run(self, query):
        response = self.executor.run(query)
        return response


class RetrieverAgent:
    def __init__(self, index_name: str = "khugpt") -> None:
        self.llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0, openai_api_key=settings.OPENAI_API_KEY)
        self.retreiver = PineconeRetriever(index_name=index_name)

    def run(self, query: str):
        context = self.retreiver.get_relevant_doc_string(query)
        system_prompt = retriever_prompt_template.format(
            question=query, context=context, current_date=datetime.datetime.now()
        )
        answer = self.llm.predict(system_prompt)
        return answer

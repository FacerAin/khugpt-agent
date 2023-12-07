import datetime

from langchain.agents import AgentExecutor, LLMSingleActionAgent, OpenAIFunctionsAgent, Tool
from langchain.cache import InMemoryCache
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_core.messages import SystemMessage

from app.agent.parser import CustomAgentOutputParser
from app.agent.prompts import AgentPromptTemplate, agent_prompt_template, retriever_prompt_template
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

        self.agent_prompt = AgentPromptTemplate(
            template=agent_prompt_template,
            tools=self.tools,
            input_variables=["input", "intermediate_steps"],
        )
        self.output_parser = CustomAgentOutputParser()
        llm_chain = LLMChain(llm=self.llm, prompt=self.agent_prompt)
        tool_names = [tool.name for tool in self.tools]
        self.agent = LLMSingleActionAgent(
            llm_chain=llm_chain, output_parser=self.output_parser, stop=["\nObservation:"], allowed_tools=tool_names
        )

        self.executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=self.tools, verbose=True, max_iterations=5
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

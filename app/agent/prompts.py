from typing import List

from langchain.agents import Tool
from langchain.prompts import PromptTemplate, StringPromptTemplate

agent_prompt_template = """You are a helpful assistant for Kyung Hee University students.
Answer the following questions as best you can. If a page_url is provided in the document, please also provide a link to the related page.
You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Remember to speak in a korean when giving your final answer.

Question: {input}
{agent_scratchpad}"""

retriever_prompt_template = PromptTemplate(
    input_variables=["current_date", "context", "question"],
    template="""

Use the following pieces of context to answer the query at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Users tend to want up-to-date information, so please refer to the current date to answer. Unless otherwise instructed, use this year’s information whenever possible.
Each context is separated by [SEP].
The provided documents may be duplicated or contain typos. In this case, please edit and use appropriately according to the context.
If you think there are multiple appropriate answers, it’s okay to paste a reference link and a brief explanation at the bottom of the answer.
When providing a reference link, provide it neatly using Markdown syntax.
In situations where the provided context is insufficient, it’s possible to list multiple candidate links to inform the user.
When the context cannot be found, instead of saying that the context was not found, advise the user on suitable alternative actions.
Attach the page link of the corresponding context at the bottom of the answer.

Current date: {current_date}

Contexts: {context}

Query: {question}
Helpful answer:
""",
)


class AgentPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

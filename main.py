import os
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
load_dotenv()

# This is a different way to specify the model. This is a dictionary
# in the past I have used model="gpt4o-mini"
llm = init_chat_model("openai:gpt-4o-mini")


tool = TavilySearch(max_results=2)
tools=[tool]
tool.invoke("What's a node in Langgraph?")


# incorporate it into a StateGraph:

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Modification: tell the LLM which tools it can call
# highlight-next-line
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)










def main():
    print("Hello from lg-add-tools!")

if __name__ == "__main__":
    main()

import json
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
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


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""
    
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}


    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
    
# incorporate it into a StateGraph:
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Modification: tell the LLM which tools it can call
# highlight-next-line
llm_with_tools = llm.bind_tools(tools)
tool_node = BasicToolNode(tools)

def chatbot(state: State,):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def route_tools(state: State,):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END




graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)










def main():
    print("Hello from lg-add-tools!")

if __name__ == "__main__":
    main()

import json
import io
from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from IPython.display import Image, display
from PIL import Image as PILImage

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# This is a different way to specify the model. This is a dictionary
# in the past I have used model="gpt4o-mini"
llm = init_chat_model("openai:gpt-4o-mini")

def draw_graph(graph):
    try:
        img_data = graph.get_graph().draw_mermaid_png()
        image = PILImage.open(io.BytesIO(img_data))
        image.show()
    except Exception:
        # This requires some extra dependencies and is optional
            pass


tool = TavilySearch(max_results=2)
tools=[tool]


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


# Modification: tell the LLM which tools it can call
# highlight-next-line
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State,):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def route_tools(state: State,):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    # if isinstance(state, list):
    #     ai_message = state[-1]
    # elif messages := state.get("messages", []):
    #     ai_message = messages[-1]
    # else:
    #     raise ValueError(f"No messages found in input state to tool_edge: {state}")
    messages = state.get("messages", [])
    # if there are no messages, we con't have tool calls
    if not messages:
        print("No messages found in input")
        return END
    ai_message = messages[-1]
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

graph_builder = StateGraph(State)
tool_node = BasicToolNode(tools)

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

def main():
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() in ["draw", "graph", "show"]:
                draw_graph(graph)
                continue
                
            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break



if __name__ == "__main__":
    main()

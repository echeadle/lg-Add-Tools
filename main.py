import json
import io
from typing import Annotated, List
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from PIL import Image as PILImage

from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Initialize the language model
llm = init_chat_model("openai:gpt-4o-mini")

def draw_graph(graph):
    """
    Generates and displays a diagram of the graph.
    """
    try:
        # Get the graph as a PNG image
        img_data = graph.get_graph().draw_mermaid_png()
        # Open the image and display it
        image = PILImage.open(io.BytesIO(img_data))
        image.show()
    except Exception as e:
        # This requires some extra dependencies and is optional.
        # Added error printing for better feedback.
        print(f"Error drawing graph: {e}")


# Initialize the search tool
tool = TavilySearch(max_results=2)
tools = [tool]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        """
        Initializes the node with a list of tools.
        """
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        """
        Invokes the tools based on the tool calls in the last message.
        """
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            # Invoke the tool and get the result
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            # Format the result as a ToolMessage
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


class State(TypedDict):
    """
    Represents the state of our graph.
    """
    messages: Annotated[list, add_messages]


# Bind the tools to the language model
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    """
    A node that invokes the chatbot to respond to the user's message.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def route_tools(state: State) -> str:
    """
    Routes to the ToolNode if the last message has tool calls, otherwise ends.
    """
    messages = state.get("messages", [])
    if not messages:
        # If there are no messages, we can't have tool calls
        return END

    ai_message = messages[-1]
    # Check if the AI message has tool calls
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# Build the graph
graph_builder = StateGraph(State)
tool_node = BasicToolNode(tools)

# Add the nodes to the graph
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

# Set the entry point
graph_builder.add_edge(START, "chatbot")

# Add a conditional edge to route to the tool node or end
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)

# Any time a tool is called, we return to the chatbot
graph_builder.add_edge("tools", "chatbot")

# Compile the graph
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    """
    Streams the graph's response to the user's input.
    """
    # Stream the events from the graph
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            # Print the content of the last message
            if value['messages'][-1].content:
                print("Assistant:", value['messages'][-1].content)


def main():
    """
    Main loop to run the chatbot from the command line.
    """
    print("Chatbot is ready. Type 'quit', 'exit', or 'q' to end.")
    print("Type 'draw' or 'graph' to see a diagram of the graph.")

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            elif user_input.lower() in ["draw", "graph", "show"]:
                print("Attempting to draw the graph...")
                draw_graph(graph)
                continue

            # Stream the graph updates for the user's input
            stream_graph_updates(user_input)
        except (KeyboardInterrupt, EOFError):
            # Handle Ctrl+C or Ctrl+D
            print("\nGoodbye!")
            break
        except Exception as e:
            # Fallback for other errors
            print(f"\nAn unexpected error occurred: {e}")
            print("Switching to a non-interactive example.")
            user_input = "What is the weather in SF?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break


if __name__ == "__main__":
    main()

import io
from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from PIL import Image as PILImage

from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Initialize the language model
llm = init_chat_model("openai:gpt-4o-mini")

def draw_graph(graph):
    """
    Generates and displays a diagram of the graph.
    """`
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


class State(TypedDict):
    """
    Represents the state of our graph.
    """
    messages: Annotated[list, add_messages]


# Initialize the graph builder
graph_builder = StateGraph(State)

# Initialize the search tool and bind it to the LLM
tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    """
    A node that invokes the chatbot to respond to the user's message.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


# Add the chatbot node
graph_builder.add_node("chatbot", chatbot)

# Use the prebuilt ToolNode, which is a convenient way to run tools
# without having to write a custom node function.
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Use the prebuilt tools_condition to decide whether to call tools or not.
# This function checks the last message in the state and returns "tools"
# if it has tool calls, and "__end__" otherwise.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
# Set the entry point of the graph
graph_builder.add_edge(START, "chatbot")

# Compile the graph
graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    """
    Streams the graph's response to the user's input.
    """
    # Stream the events from the graph
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            # Print the content of the last message, if it exists
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

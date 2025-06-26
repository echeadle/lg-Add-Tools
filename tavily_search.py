from langchain_tavily import TavilySearch
# This is an quick example of using tavily search with langchain
# It is a built in search tool.

#LangGraph examples all have the api_keys in the code.
# I don't do that so I add dotenv to all the code.
# Apparently this does not work alone it is just an example
#of how to create the TavilySearch tool.
from dotenv import load_dotenv
load_dotenv()

tool = TavilySearch(max_results=2)
tools=[tool]

def main():
    print("Hello from lg-add-tools!")
    tool.invoke("What's a node in Langgraph?")


if __name__ == "__main__":
    main()


from typing import List, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class Source(BaseModel):
    """Schema for the source used by the agent"""
    url: str = Field(description="The url of the source")

class AgentResponse(BaseModel):
    """Schema for the response used by the agent"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


@tool
def search(query: str) -> str:
    """
    Tool that searches the web for information
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Searching for {query}")
    return tavily.search(query)

llm = ChatOpenAI()
tools = [search]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    print("Hello from LangChain!")

    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")]})
    print(result)
    pass

if __name__ == "__main__":
    main()
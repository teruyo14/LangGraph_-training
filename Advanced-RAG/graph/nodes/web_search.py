import json
import os
from collections import defaultdict

from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.prebuilt import ToolInvocation
from langchain.schema import Document

from graph.state import GraphState

from chains import parser

load_dotenv()
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

from serpapi import GoogleSearch

def serpapi_search(query: str) -> dict:
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "location": "United States",
        "num": 5 
    }
    search = GoogleSearch(params)
    result = search.get_dict()
    return result

def web_execute_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    tool_invocation: AIMessage = state[-1]
    parsed_tool_calls = parser.invoke(tool_invocation)
    ids = []
    tool_invocations = []

    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            tool_invocations.append(
                {
                    "tool": "serpapi_search",
                    "tool_input": query,
                    "id": parsed_call["id"]
                }
            )
            ids.append(parsed_call["id"])

    outputs_map = defaultdict(dict)
    tool_messages = []

    for invocation in tool_invocations:
        tool_input = invocation["tool_input"]
        id_ = invocation["id"]
        try:
            output = serpapi_search(tool_input)
            outputs_map[id_][tool_input] = output
        except Exception as e:
            outputs_map[id_][tool_input] = {"error": str(e)}

    for id_, query_outputs in outputs_map.items():
        tool_messages.append(
            ToolMessage(content=json.dumps(query_outputs), tool_call_id=id_)
        )

    return tool_messages

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    docs = web_execute_tools.invoke({"query": question})
    docs = web_execute_tools.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}
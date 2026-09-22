from langgraph.graph import StateGraph,START,END

from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()


GROK_API_KEY = os.getenv("GROK_API_KEY")

llm = ChatOpenAI(
    model="qwen/qwen3.8-27b",
    temperature=0,
    api_key=GROK_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    # Groq free tier caps OUTPUT at 1000 tokens/minute (OTPM). Keep every
    # single response under that so a call isn't rejected before it runs.
    max_tokens=900,
)



from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]


def chat_node(state:ChatState):
    messages=state['messages']

    response=llm.invoke(messages)
    return {'messages':[response]}





checkpoint=MemorySaver()
graph=StateGraph(ChatState)

#add nodes
graph.add_node('chat_node',chat_node)

#add edges
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpoint)
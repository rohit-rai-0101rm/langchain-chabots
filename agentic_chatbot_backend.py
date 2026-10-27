from langgraph.graph import StateGraph,START ,END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

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


class ChatState(TypedDict):

    messages:Annotated[list[BaseMessage],add_messages]


def chat_node(state:ChatState):
    messages=state['messages']
    response=llm.invoke(messages)
    return {'messages':[response]}


checkpoint=MemorySaver()

graph=StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot=graph.compile(checkpointer=checkpoint)


if __name__ == '__main__':
    thread_id="1"


    initial_state={
        'messages':[HumanMessage(content='What is Object oriented programming')]
    }
    config={'configurable':{'thread_id':thread_id}}

    response=chatbot.invoke(initial_state,config=config)

    print(response['messages'][-1].content)

    print(chatbot.get_state(config=config))

    # Type "/thread <id>" to switch conversations
    while True:
        user_message=input(f'[thread {thread_id}] Type here')
        print('User',user_message)

        if user_message.strip().lower() in ['exit','quit','bye']:
            break
        if user_message.strip().startswith('/thread'):
            thread_id=user_message.split(maxsplit=1)[1].strip()
            print('Switched to thread',thread_id)
            continue
        config={'configurable':{'thread_id':thread_id}}
        response=chatbot.invoke({'messages':[HumanMessage(content=user_message)]},config=config)
        print('Ai:',response['messages'][-1].content)

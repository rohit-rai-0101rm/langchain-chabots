from agentic_chatbot import chatbot
from langchain_core.messages import BaseMessage,HumanMessage
import streamlit as st
config = {'configurable': {'thread_id': 'thread1'}}
st.title('Agentic chatbot with langgraph')

if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]

for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
# response = chatbot.invoke(
#     {'messages': [HumanMessage(content="your message here")]},
#     config=config
# )

# print(response['messages'][-1].content)

user_input=st.chat_input('type here')

# st.write('user',user_input)

def stream_response(user_input):
    for message_chunk, metadata in chatbot.stream(
        {'messages': [HumanMessage(content=user_input)]},
        config=config,
        stream_mode="messages"
    ):
        if message_chunk.content:
            yield message_chunk.content

if user_input:

    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        ai_message = st.write_stream(stream_response(user_input))
    st.session_state['message_history'].append({'role':'assistant','content':ai_message})

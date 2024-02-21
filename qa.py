## Conversational Q&A Chatbot
import streamlit as st
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_community.llms import Ollama


st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Hey, Let's Chat")

from dotenv import load_dotenv
load_dotenv()
import os

chat=Ollama(model="mistral",temperature=0.5)

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages']=[
        SystemMessage(content="Yor are a comedian AI assitant")
    ]


def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    last_human_message = next(filter(lambda x: isinstance(x, HumanMessage), reversed(st.session_state['flowmessages'])), None)
    if last_human_message is not None:
        answer = chat(last_human_message.content)
        st.session_state['flowmessages'].append(AIMessage(content=answer))
        return answer
    else:
        return "Please provide a valid question."


input=st.text_input("Input: ",key="input")
response=get_chatmodel_response(input)

submit=st.button("Ask the question")


if submit:
    st.subheader("The Response is")
    st.write(response)
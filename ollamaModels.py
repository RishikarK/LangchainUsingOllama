from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
from datetime import datetime


st.header("Summarization using Ollama-Models")
selected_model=st.sidebar.selectbox("Choose the Ollama model " , ('llama2', 'mistral', 'llava' , 'dolphin-mixtral' , 'openchat' , 'falcon' ,'llama2-uncensored:latest',
                                                          'vicuna', 'phi'), index=0)

if(selected_model):
    st.spinner("Model Selected is "+ selected_model)

llm = Ollama(model=selected_model)
print(llm)

## Summary Convertor
template = '''Provide a detailed summary of the following speech, breaking down each subject into bullet points and explaining each point thoroughly: TEXT: {speech}
'''
summary_prompt = PromptTemplate(
    input_variables=["speech"],
    template=template,
)   
target = st.text_area("Enter the text :")
chain = LLMChain(llm=llm, prompt=summary_prompt , output_parser=StrOutputParser())

summarize=st.button("Summarize")

if(summarize):
    translation_result = chain({'speech': target})
    st.write(translation_result["text"])
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.write(f"Translation Result (Time: {current_time}):")
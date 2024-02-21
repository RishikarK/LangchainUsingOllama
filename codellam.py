from langchain_community.llms import Ollama
from langchain import PromptTemplate, LLMChain
import streamlit as st

llm = Ollama(model='codellama')

template = '''Provide a code in {language} of the following requirement {speech} '''
summary_prompt = PromptTemplate(
    input_variables=["speech", "language"],
    template=template,
)

lang_options = ["java", "python", "nodejs", "R-Lang", "C", "C++", "C#", "php", "go-lang", "Oracle sql", "mongodb" ,"Html" ,"Css"]
lang = st.selectbox("Choose The language to write the code ", lang_options, index=0)
target = st.text_area("Enter the text:")

chain = LLMChain(llm=llm, prompt=summary_prompt)

submit = st.button("Generate Code")

if submit:
    translation_result = chain({'speech': target, 'language': lang})
    st.code(translation_result["text"])

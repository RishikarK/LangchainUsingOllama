from langchain_community.llms import Ollama
from langchain import PromptTemplate, LLMChain

llm = Ollama(model='codellama')

template = '''Provide a code in {language} of the following requirement {speech} '''
summary_prompt = PromptTemplate(
    input_variables=["speech", "language"],
    template=template,
)

# lang_options = ["java", "python", "nodejs", "R-Lang", "C", "C++", "C#", "php", "go-lang", "Oracle sql", "mongodb"]
# lang = st.selectbox("Choose The language to write the code ", lang_options, index=0)
lang=input("Enter the Lang :")
target = input("Enter the text:")

chain = LLMChain(llm=llm, prompt=summary_prompt)


translation_result = chain({'speech': target, 'language': lang})
print(translation_result["text"])

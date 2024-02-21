# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.llms import Ollama
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.chains.question_answering import load_qa_chain
# from langchain.vectorstores import FAISS

# # Initialize Ollama
# llm = Ollama(model="llama2")

# # PDF Processing
# st.title("PDF Question Answering")

# pdf_path = st.file_uploader("Upload PDF file", type="pdf")
# if pdf_path is not None:
#     pdfreader = PdfReader(pdf_path)

#     raw_text = ''
#     for i, page in enumerate(pdfreader.pages):
#         content = page.extract_text()
#         if content:
#             raw_text += content

#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=800,
#         chunk_overlap=200,
#         length_function=len,
#     )
#     texts = text_splitter.split_text(raw_text)

#     # Question Answering
#     embeddings = llm.embed_documents(texts)  # Embed documents using Ollama
#     document_search = FAISS.from_texts(texts, embeddings=embeddings)

#     chain = load_qa_chain(llm, chain_type="stuff")

#     query = st.text_input("Ask The Question")

#     if st.button("Search"):
#         if query:
#             docs = document_search.similarity_search(query)
#             result = chain.run(input_documents=docs, question=query)
#             st.write(result)
#         else:
#             st.warning("Please enter a question.")

import streamlit as st
from PyPDF2 import PdfReader
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
import os

# Set Ollama model path
OLLAMA_MODEL_PATH = "/usr/local/bin/ollama/usr/share/ollama/mistral:latest"  # Update this with your Ollama model path

# PDF Processing
st.title("PDF Question Answering")

pdf_path = st.file_uploader("Upload PDF file", type="pdf")
if pdf_path is not None:
    pdfreader = PdfReader(pdf_path)

    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Embedding
    ollama_embeddings = OllamaEmbeddings(model="mistral")
    embeddings = ollama_embeddings.embed_documents(texts)

    print(embeddings)
    # FAISS Indexing
    document_search = FAISS.from_embeddings(texts,embeddings)

    # Question Answering
    chain = load_qa_chain(Ollama(OLLAMA_MODEL_PATH), chain_type="stuff")

    query = st.text_input("Ask The Question")

    if st.button("Search"):
        if query:
            docs = document_search.similarity_search(query)
            result = chain.run(input_documents=docs, question=query)
            st.write(result)
        else:
            st.warning("Please enter a question.")

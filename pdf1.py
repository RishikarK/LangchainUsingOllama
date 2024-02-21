from fastapi import FastAPI, Form, Request, Response, File, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
import os 
import json
import time
import uvicorn
import aiofiles
from PyPDF2 import PdfReader
import csv

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# Load LLM model for answer generation
llm_answer_gen = CTransformers(
    model="mistral",
    max_new_tokens=1048,
    temperature=0.3
)

# Load embeddings for answer generation
embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Load FAISS vector store for answer retrieval
vector_store = FAISS.load("your_vector_store_path")  # Provide your vector store path here

answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, chain_type="stuff", retriever=vector_store.as_retriever())

def file_processing(file_path):
    # Load data from PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content
        
    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 30
    )

    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen

@app.post("/upload")
async def upload_file(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    base_folder = 'static/docs/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    pdf_filename = os.path.join(base_folder, filename)

    async with aiofiles.open(pdf_filename, 'wb') as f:
        await f.write(pdf_file)
    response_data = jsonable_encoder(json.dumps({"msg": 'success',"pdf_filename": pdf_filename}))
    res = Response(response_data)
    return res

@app.post("/analyze")
async def analyze_pdf(request: Request, pdf_filename: str = Form(...), question: str = Form(...)):
    # Process PDF file and generate question and answer chains
    document_ques_gen, document_answer_gen = file_processing(pdf_filename)

    # Get answer for the entered question
    answer = answer_generation_chain.run(question)

    response_data = jsonable_encoder(json.dumps({"question": question, "answer": answer}))
    res = Response(response_data)
    return res

if __name__ == "__main__":
    uvicorn.run("app:app", host='0.0.0.0', port=8000, reload=True)

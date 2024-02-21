import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
# from langchain.vectorstores import FAISS

# loader = TextLoader("RohitSharma.pdf")
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# embeddings = OllamaEmbeddings()
# db = FAISS.from_documents(docs, embeddings)
# # print(db)
# query = "What is hugguingface?"
# docs = db.similarity_search(query)
# print(docs[0].page_content)


import PyPDF2
from transformers import pipeline

# Extract text from PDF
def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

file_path = "budget_speech.pdf"
pdf_text = read_pdf(file_path)

# Question Answering Pipeline using transformers library
question_answerer = pipeline("question-answering")

# Ask questions and get answers
while True:
    question = input("Ask a question (or type 'quit' to exit): ")
    if question.lower() == 'quit':
        break
    else:
        result = question_answerer(question=question, context=pdf_text)
        print("Answer:", result['answer'])


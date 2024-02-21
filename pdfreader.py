from langchain_community.document_loaders import PyPDFLoader
from transformers import pipeline

def load_pdf_data(file_path):
    # Create a PyPDFLoader object with the file path
    loader = PyPDFLoader(file_path=file_path)
    # Load the PDF file
    docs = loader.load()
    # Return the loaded document
    return docs

def get_ollama_answer(question, context):
    # Initialize the Ollama question-answering pipeline
    ollama_qa = pipeline("question-answering", model="ollama/orca-mini-3b")

    # Extract the answer using Ollama
    answer = ollama_qa(question=question, context=context)["answer"]
    return answer

# Load the PDF data
docs = load_pdf_data('RohitSharma.pdf')

# Assuming 'docs' contains the loaded PDF data
question = "When was Rohit Sharma Born?"
# Get the answer using Ollama
answer = get_ollama_answer(question, context=docs[0].page_content)
print("Answer:", answer)

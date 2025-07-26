from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import os
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

import uvicorn
import time

# Load environment variables
load_dotenv()
os.environ['NVIDIA_API_KEY'] = os.getenv("NVIDIA_API_KEY")

# FastAPI instance
app = FastAPI(title="Nvidia NIM RAG API")

# Request schema
class QueryRequest(BaseModel):
    question: str

# Global state
vector_store = None
retriever = None
llm = ChatNVIDIA(model="meta/llama3-70b-instruct")
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Initialization of vector store
def initialize_vector_store():
    global vector_store, retriever
    embeddings = NVIDIAEmbeddings()
    loader = PyPDFDirectoryLoader("./us_census")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs[:30])
    vector_store = FAISS.from_documents(split_docs, embeddings)
    retriever = vector_store.as_retriever()

# Health check
@app.get("/")
def read_root():
    return {"message": "Nvidia RAG API is running"}

# POST endpoint to handle RAG question
@app.post("/ask")
def ask_question(request: QueryRequest):
    global vector_store, retriever

    if vector_store is None:
        initialize_vector_store()

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start_time = time.process_time()
    result = retrieval_chain.invoke({"input": request.question})
    duration = time.process_time() - start_time

    return {
        "question": request.question,
        "answer": result["answer"],
        "response_time": duration,
        "context": [doc.page_content for doc in result["context"]]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

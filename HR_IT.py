import os

import chromadb
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# --------------------------------------------------
# Environment Variables
# --------------------------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMADB_API_KEY = os.getenv("CHROMADB_API_KEY")
CHROMADB_TENANT = os.getenv("CHROMADB_TENANT")
CHROMADB_DB = os.getenv("CHROMADB_DB")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found.")

if not CHROMADB_API_KEY:
    raise RuntimeError("CHROMADB_API_KEY not found.")

if not CHROMADB_TENANT:
    raise RuntimeError("CHROMADB_TENANT not found.")

if not CHROMADB_DB:
    raise RuntimeError("CHROMADB_DB not found.")

# --------------------------------------------------
# OpenAI Models
# --------------------------------------------------

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

# --------------------------------------------------
# Chroma Cloud
# --------------------------------------------------

client = chromadb.CloudClient(
    api_key=CHROMADB_API_KEY,
    tenant=CHROMADB_TENANT,
    database=CHROMADB_DB,
)

vectorstore = Chroma(
    client=client,
    collection_name="medical_hr",
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

# --------------------------------------------------
# Prompt
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template(
    """
You are an HR assistant for a medical organization.

Use ONLY the supplied HR documentation to answer questions.

If the answer cannot be found in the provided context, reply:

"I don't know based on the available HR documentation."

Keep answers professional and concise.

At the end of your response include the document names you used if available.

Context:
{context}

Question:
{input}
"""
)

# --------------------------------------------------
# RAG Chain
# --------------------------------------------------

document_chain = create_stuff_documents_chain(
    llm,
    prompt,
)

rag_chain = create_retrieval_chain(
    retriever,
    document_chain,
)

# --------------------------------------------------
# Chat
# --------------------------------------------------

def chat(question: str) -> str:
    """Ask the Medical HR knowledge base."""

    if not question.strip():
        return "Please enter a question."

    response = rag_chain.invoke(
        {
            "input": question
        }
    )

    return response["answer"]


# --------------------------------------------------
# CLI Demo
# --------------------------------------------------

if __name__ == "__main__":

    print("Medical HR Assistant")
    print("Type 'quit' to exit.\n")

    while True:

        question = input("Question: ")

        if question.lower() in {"quit", "exit"}:
            break

        answer = chat(question)

        print("\nAnswer:\n")
        print(answer)
        print()
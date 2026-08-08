"""
Medical HR RAG Backend
Stack:
- LangChain
- OpenRouter - LLM
- OpenAI-compatible API
- Chroma Cloud
- RAG
"""

import os
import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHROMADB_API_KEY = os.getenv("CHROMADB_API_KEY")
CHROMADB_TENANT = os.getenv("CHROMADB_TENANT")
CHROMADB_DB = os.getenv("CHROMADB_DB")

# can remove once API keys all work
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is missing from your .env file."
    )

if not CHROMADB_API_KEY:
    raise RuntimeError(
        "CHROMADB_API_KEY is missing from your .env file."
    )

if not CHROMADB_TENANT:
    raise RuntimeError(
        "CHROMADB_TENANT is missing from your .env file."
    )

if not CHROMADB_DB:
    raise RuntimeError(
        "CHROMADB_DB is missing from your .env file."
    )

# OpenRouter LLM

llm = ChatOpenAI(
    model="google/gemma-4-26b-a4b-it:free",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=512,
)

# OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
)


# Chroma Cloud
chroma_client = chromadb.CloudClient(
    api_key=CHROMADB_API_KEY,
    tenant=CHROMADB_TENANT,
    database=CHROMADB_DB,
)


vectorstore = Chroma(
    client=chroma_client,
    collection_name="medical_hr_documents",
    embedding_function=embeddings,
)

# Retriever

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
    }
)

# Medical HR Prompt

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Medical HR Assistant for a company called Daisy_Health.

Your job is to answer questions using ONLY
the information contained in the retrieved
HR documentation.

The documentation may contain information
about:

- Employee policies
- PTO
- Benefits
- Leave
- HIPAA training
- Compliance
- Employee handbooks
- Workplace policies
- Credentialing
- Medical staff policies
- Scheduling
- HR procedures

Do not invent policies.

Do not use outside knowledge.

If the retrieved documentation does not
contain enough information to answer the
question, say:

"I don't know based on the available HR
documentation."

When possible, mention the source document
used to answer the question.
""",
        ),
        (
            "human",
            """
Retrieved HR documentation:

{context}

Employee question:

{input}
""",
        ),
    ]
)

# Document Chain

document_chain = create_stuff_documents_chain(
    llm,
    prompt,
)

# RAG Chain

rag_chain = create_retrieval_chain(
    retriever,
    document_chain,
)

# Chat Function -for testing purposes


def chat(question: str) -> str:
    """
    Send a question through the Medical HR RAG pipeline.
    """

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    question = question.strip()

    if not question:
        return "Please enter a question."

    response = rag_chain.invoke(
        {
            "input": question
        }
    )

    return response["answer"]

# Test


if __name__ == "__main__":

    print("Medical HR Assistant")
    print("--------------------")
    print("Using OpenRouter + Chroma Cloud")
    print("Type 'quit' to exit.\n")

    while True:

        question = input("Question: ")

        if question.lower() in {
            "quit",
            "exit",
        }:
            break

        try:

            answer = chat(question)

            print("\nAnswer:")
            print(answer)

        except Exception as error:

            print("\nError:")
            print(error)

        print("\n" + "-" * 60 + "\n")
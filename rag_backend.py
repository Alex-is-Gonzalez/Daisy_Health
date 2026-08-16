import os
import chromadb

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

load_dotenv()


# ─────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────

def get_config():
    required_vars = {
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "CHROMADB_API_KEY": os.getenv("CHROMADB_API_KEY"),
        "CHROMADB_TENANT": os.getenv("CHROMADB_TENANT"),
        "CHROMADB_DB": os.getenv("CHROMADB_DB"),
    }

    missing = [
        name for name, value in required_vars.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}"
        )

    return required_vars


# ─────────────────────────────────────────────
# LAZY RAG INITIALIZATION
# ─────────────────────────────────────────────

_rag_components = None


def get_rag_components():
    """
    Initialize the RAG stack only when it is actually needed.
    Components are cached for the lifetime of the Python process.
    """

    global _rag_components

    if _rag_components is not None:
        return _rag_components

    config = get_config()

    # ─────────────────────────────────────────
    # LLM
    # ─────────────────────────────────────────

    llm = ChatOpenAI(
        model="google/gemma-4-26b-a4b-it:free",
        api_key=config["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=512,
    )

    # ─────────────────────────────────────────
    # EMBEDDINGS
    # ─────────────────────────────────────────

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ─────────────────────────────────────────
    # CHROMA CLOUD
    # ─────────────────────────────────────────

    chroma_client = chromadb.CloudClient(
        api_key=config["CHROMADB_API_KEY"],
        tenant=config["CHROMADB_TENANT"],
        database=config["CHROMADB_DB"],
    )

    vectorstore = Chroma(
        client=chroma_client,
        collection_name="medical_hr_documents_hf",
        embedding_function=embeddings,
    )

    # ─────────────────────────────────────────
    # RETRIEVER
    # ─────────────────────────────────────────

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    # ─────────────────────────────────────────
    # PROMPT
    # ─────────────────────────────────────────

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a Medical HR Assistant for a company called Daisy Health.

Your job is to answer questions using ONLY
the information contained in the retrieved
HR documentation.

The documentation may contain information about:

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

If the retrieved documentation does not contain
enough information to answer the question, say:

"I don't know based on the available HR
documentation. Please e-mail HR at
people@daisyhealth.com"

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

    # ─────────────────────────────────────────
    # CHAINS
    # ─────────────────────────────────────────

    document_chain = create_stuff_documents_chain(
        llm,
        prompt,
    )

    rag_chain = create_retrieval_chain(
        retriever,
        document_chain,
    )

    _rag_components = {
        "llm": llm,
        "embeddings": embeddings,
        "chroma_client": chroma_client,
        "vectorstore": vectorstore,
        "retriever": retriever,
        "document_chain": document_chain,
        "rag_chain": rag_chain,
    }

    return _rag_components


# ─────────────────────────────────────────────
# RAG FUNCTION
# ─────────────────────────────────────────────

def chat(question: str):
    """
    Run a question through the RAG pipeline.
    """

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "documents": [],
        }

    rag = get_rag_components()

    documents = rag["retriever"].invoke(question)

    response = rag["rag_chain"].invoke(
        {
            "input": question
        }
    )

    return {
        "answer": response["answer"],
        "documents": documents,
    }


def get_document_count():
    """Return the number of documents/chunks in Chroma."""

    rag = get_rag_components()

    return rag["vectorstore"]._collection.count()
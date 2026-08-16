import os
import chromadb

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

def get_config():

    required_vars = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "CHROMADB_API_KEY": os.getenv("CHROMADB_API_KEY"),
        "CHROMADB_TENANT": os.getenv("CHROMADB_TENANT"),
        "CHROMADB_DB": os.getenv("CHROMADB_DB"),
    }

    missing = [
        name
        for name, value in required_vars.items()
        if not value
    ]

    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}"
        )

    return required_vars


# ============================================================
# LAZY INITIALIZATION
# ============================================================

_rag_components = None


def get_rag_components():

    global _rag_components

    if _rag_components is not None:
        return _rag_components

    config = get_config()


    # ========================================================
    # OPENAI LLM
    # ========================================================

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=config["OPENAI_API_KEY"],
        temperature=0,
        max_tokens=512,
    )


    # ========================================================
    # OPENAI EMBEDDINGS
    # ========================================================

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=config["OPENAI_API_KEY"],
    )


    # ========================================================
    # CHROMA CLOUD
    # ========================================================

    chroma_client = chromadb.CloudClient(
        api_key=config["CHROMADB_API_KEY"],
        tenant=config["CHROMADB_TENANT"],
        database=config["CHROMADB_DB"],
    )

    vectorstore = Chroma(
        client=chroma_client,
        collection_name="medical_hr_documents_openai",
        embedding_function=embeddings,
    )


    # ========================================================
    # RETRIEVER
    # ========================================================

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )


    # ========================================================
    # PROMPT
    # ========================================================

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a Medical HR Assistant for Daisy Health.

Answer questions ONLY using the retrieved HR
documentation.

Do not invent policies.

Do not use outside knowledge.

If the retrieved documentation does not contain
enough information, say:

"I don't know based on the available HR
documentation. Please e-mail HR at
people@daisyhealth.com"

When possible, mention the source document.
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


    # ========================================================
    # CHAINS
    # ========================================================

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


# ============================================================
# CHAT
# ============================================================

def chat(question: str):

    if not isinstance(question, str):
        raise TypeError("Question must be a string.")

    question = question.strip()

    if not question:
        return {
            "answer": "",
            "documents": [],
        }

    rag = get_rag_components()

    documents = rag["retriever"].invoke(question)

    return {
        "answer": "",
        "documents": documents,
    }


# ============================================================
# DOCUMENT COUNT
# ============================================================

def get_document_count():

    rag = get_rag_components()

    return rag["vectorstore"]._collection.count()
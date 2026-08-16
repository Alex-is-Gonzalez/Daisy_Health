import os
import chromadb

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
)
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# Environment
# ============================================================

load_dotenv()

CHROMADB_API_KEY = os.getenv("CHROMADB_API_KEY")
CHROMADB_TENANT = os.getenv("CHROMADB_TENANT")
CHROMADB_DB = os.getenv("CHROMADB_DB")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not CHROMADB_API_KEY:
    raise RuntimeError("CHROMADB_API_KEY is missing.")

if not CHROMADB_TENANT:
    raise RuntimeError("CHROMADB_TENANT is missing.")

if not CHROMADB_DB:
    raise RuntimeError("CHROMADB_DB is missing.")


# ============================================================
# Hugging Face Embeddings
# ============================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)


# ============================================================
# Chroma Cloud
# ============================================================

chroma_client = chromadb.CloudClient(
    api_key=CHROMADB_API_KEY,
    tenant=CHROMADB_TENANT,
    database=CHROMADB_DB,
)


vectorstore = Chroma(
    client=chroma_client,
    collection_name="medical_hr_documents_openai",
    embedding_function=embeddings,
)


# ============================================================
# Ingest Documents
# ============================================================

def ingest_documents():
    """
    Load HR PDFs, split them into chunks,
    create Hugging Face embeddings,
    and store them in Chroma Cloud.
    """

    print("\nLoading HR documents...")

    loader = DirectoryLoader(
        "data/handbooks",
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    documents = loader.load()

    if not documents:
        raise RuntimeError(
            "No PDF documents were found in data/handbooks."
        )

    print(f"Loaded {len(documents)} pages.")

    # ========================================================
    # Split documents
    # ========================================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # ========================================================
    # Add metadata
    # ========================================================

    for chunk in chunks:

        source = chunk.metadata.get("source", "")

        chunk.metadata["document_type"] = "medical_hr"

        chunk.metadata["source_file"] = os.path.basename(
            source
        )

    # ========================================================
    # Add to Chroma Cloud
    # ========================================================

    print("Adding documents to Chroma Cloud...")

    vectorstore.add_documents(chunks)

    print("Documents successfully added to Chroma Cloud.")

    print(
        "Chroma collection count:",
        vectorstore._collection.count()
    )


# ============================================================
# Run ingestion
# ============================================================

if __name__ == "__main__":

    print("Daisy Health - HR Document Ingestion")
    print("-------------------------------------")
    print(f"Chroma database: {CHROMADB_DB}")
    print("Collection: medical_hr_documents_hf")
    print("Embedding model: all-MiniLM-L6-v2")

    ingest_documents()
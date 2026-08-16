import os
import re
import chromadb

from dotenv import load_dotenv
from pathlib import Path
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not CHROMADB_API_KEY:
    raise RuntimeError("CHROMADB_API_KEY is missing.")

if not CHROMADB_TENANT:
    raise RuntimeError("CHROMADB_TENANT is missing.")

if not CHROMADB_DB:
    raise RuntimeError("CHROMADB_DB is missing.")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing.")


# ============================================================
# OpenAI Embeddings
# ============================================================

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY,
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
    create OpenAI embeddings,
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

        source_file = os.path.basename(source)
        stem = Path(source_file).stem
        title = stem.replace("_", " ").title()
        policy_id_match = re.search(r"HR-[A-Z]{2}-\d{3}", chunk.page_content)

        chunk.metadata["source_file"] = source_file
        chunk.metadata["document_title"] = title
        chunk.metadata["policy_id"] = (
            policy_id_match.group(0) if policy_id_match else source_file
        )

    # ========================================================
    # Add to Chroma Cloud
    # ========================================================

    print("Adding documents to Chroma Cloud...")

    # Stable ids make ingestion repeatable instead of silently duplicating
    # chunks each time the corpus is rebuilt.
    ids = [
        f"{chunk.metadata['source_file']}:p{chunk.metadata.get('page', 0)}:c{i}"
        for i, chunk in enumerate(chunks)
    ]
    vectorstore.add_documents(chunks, ids=ids)

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
    print("Collection: medical_hr_documents_openai")
    print("Embedding model: text-embedding-3-small")

    ingest_documents()

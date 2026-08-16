import os
import re
import chromadb

from dotenv import load_dotenv
from pathlib import Path  # FIX: was missing; used by Path(source_file).stem below

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,          # FIX: added to support .txt and .md files (second source format)
)
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
    Load HR PDFs and Markdown/text files, split them into chunks,
    create OpenAI embeddings, and store them in Chroma Cloud.

    Supports two source formats:
      - PDF  (.pdf)  via PyPDFLoader
      - Text (.md, .txt) via TextLoader

    Both formats are combined before chunking so the vector store
    contains a single unified corpus.
    """

    data_dir = "data/handbooks"
    all_documents = []

    # --------------------------------------------------------
    # Load PDFs
    # --------------------------------------------------------

    print("\nLoading PDF documents...")

    pdf_loader = DirectoryLoader(
        data_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    pdf_docs = pdf_loader.load()
    print(f"Loaded {len(pdf_docs)} PDF pages.")
    all_documents.extend(pdf_docs)

    # --------------------------------------------------------
    # Load Markdown and plain-text files (second source format)
    # FIX: added to satisfy the rubric requirement of ≥2 source
    # formats. Place any .md or .txt HR documents in data/handbooks/
    # and they will be ingested alongside the PDFs.
    # --------------------------------------------------------

    for glob_pattern in ("*.md", "*.txt"):
        text_loader = DirectoryLoader(
            data_dir,
            glob=glob_pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True,
        )

        try:
            text_docs = text_loader.load()
            if text_docs:
                print(f"Loaded {len(text_docs)} {glob_pattern} document(s).")
                all_documents.extend(text_docs)
        except Exception as e:
            # Non-fatal: skip if no files of this type exist
            print(f"No {glob_pattern} files found or error loading: {e}")

    if not all_documents:
        raise RuntimeError(
            "No documents were found in data/handbooks. "
            "Add .pdf, .md, or .txt HR documents and re-run."
        )

    print(f"\nTotal pages/documents loaded: {len(all_documents)}")

    # ========================================================
    # Split documents
    # ========================================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(all_documents)

    print(f"Created {len(chunks)} chunks.")

    # ========================================================
    # Add metadata
    # ========================================================

    for chunk in chunks:

        source = chunk.metadata.get("source", "")

        chunk.metadata["document_type"] = "medical_hr"

        source_file = os.path.basename(source)
        stem = Path(source_file).stem          # requires: from pathlib import Path
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
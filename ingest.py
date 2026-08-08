import os
import chromadb

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

load_dotenv()

# ingest
def ingest_documents():
    """
    Load HR PDFs, split them into chunks,
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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    # Add useful metadata
    for chunk in chunks:

        source = chunk.metadata.get("source", "")

        chunk.metadata["document_type"] = "medical_hr"
        chunk.metadata["source_file"] = os.path.basename(source)

    vectorstore.add_documents(chunks)

    print("Documents successfully added to Chroma Cloud.")

    print(
        "Chroma document count:",
        vectorstore._collection.count()
    )
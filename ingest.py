# ingest.py
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import os

# Load the open-source embedding model (runs locally)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Set up ChromaDB (persistent so data survives restarts)
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("study_notes")

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def ingest_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = chunk_text(full_text)
    embeddings = embedder.encode(chunks).tolist()

    # Store chunks with unique IDs and source metadata
    ids = [f"{os.path.basename(pdf_path)}_chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=[{"source": pdf_path} for _ in chunks]
    )
    print(f"Ingested {len(chunks)} chunks from {pdf_path}")

# Run this for each PDF
ingest_pdf("thermodynamics.pdf")
ingest_pdf("organic_chemistry.pdf")
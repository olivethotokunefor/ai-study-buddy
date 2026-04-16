# rag.py
import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection("study_notes")

def query_rag(question: str, top_k: int = 3) -> str:
    # Embed the user's question with the SAME model used during ingestion
    query_embedding = embedder.encode([question]).tolist()

    # Retrieve the top-K most similar chunks
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    context_chunks = results["documents"][0]
    context = "\n\n---\n\n".join(context_chunks)

    # Build the prompt (using OpenAI or Anthropic here)
    prompt = f"""You are a helpful study assistant. Answer the student's question using ONLY the context below.
If the answer isn't in the context, say "I don't have notes on that topic."

CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:"""

    # Call your LLM — swap in whatever you prefer
    import openai
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400
    )
    return response.choices[0].message.content.strip()
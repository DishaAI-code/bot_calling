import os
import tempfile
import uuid
from typing import Optional
import requests
from openai import OpenAI
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter

from pinecone import Pinecone
from dotenv import load_dotenv

# ------------------------
# Load environment
# ------------------------
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "dishaai")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set in environment")

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Create/validate Pinecone index
if pc.has_index(PINECONE_INDEX):
    idx_desc = pc.describe_index(PINECONE_INDEX)
    if idx_desc['dimension'] != EMBED_DIM:
        pc.delete_index(PINECONE_INDEX)
        pc.create_index_for_model(
            name=PINECONE_INDEX,
            cloud=PINECONE_ENV,
            region=PINECONE_REGION,
            embed={"model": EMBED_MODEL, "field_map": {"text": "chunk_text"}}
        )
else:
    pc.create_index_for_model(
        name=PINECONE_INDEX,
        cloud=PINECONE_ENV,
        region=PINECONE_REGION,
        embed={"model": EMBED_MODEL, "field_map": {"text": "chunk_text"}}
    )

index = pc.Index(PINECONE_INDEX)

# ------------------------
# RAG Logic
# ------------------------
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K = 4
SIMILARITY_SCORE_THRESHOLD = 0.2


# def check_if_pdf_exists(file_id: str) -> bool:
#     """Check if PDF chunks already embedded by file_id"""
#     try:
#         results = index.query(
#             vector=[0.0] * EMBED_DIM,
#             top_k=1,
#             filter={"file_id": {"$eq": file_id}}
#         )
#         return len(results.get("matches", [])) > 0
#     except:
#         return False


# def chunk_pdf_to_text(pdf_path: str) -> list:
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
#     )
#     chunks = []
#     for doc in docs:
#         for chunk in splitter.split_text(doc.page_content):
#             chunks.append({
#                 "text": chunk,
#                 "source": getattr(doc, "metadata", {}).get("source", "unknown")
#             })
#     return chunks


# def upsert_chunks_to_pinecone(chunks: list, file_id: str):
#     texts = [c["text"] for c in chunks]
#     embeddings_response = client.embeddings.create(
#         model=EMBED_MODEL,
#         input=texts
#     )
#     vectors = embeddings_response.data
#     to_upsert = []
#     for i, v in enumerate(vectors):
#         to_upsert.append({
#             "id": f"{file_id}_{i}",
#             "values": v.embedding,
#             "metadata": {
#                 "text": chunks[i]["text"],
#                 "source": chunks[i]["source"],
#                 "file_id": file_id
#             }
#         })
#     index.upsert(vectors=to_upsert)


# def query_pinecone(text: str, top_k: int = TOP_K):
#     embedding = client.embeddings.create(
#         model=EMBED_MODEL,
#         input=text
#     ).data[0].embedding

#     response = index.query(
#         vector=embedding,
#         top_k=top_k,
#         include_metadata=True
#     )
#     return response.get("matches", [])


# def generate_rag_response(question: str, hits):
#     context = "\n\n".join([h["metadata"]["text"] for h in hits])
#     prompt = f"""
#     Use ONLY this context to answer:

#     {context}

#     Question: {question}
#     """
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0
#     )
#     return response.choices[0].message.content


# kgnokfnga 

# def generate_rag_response(question: str, hits=None):
#     """
#     Full RAG pipeline:
#     - If `hits` is an integer → treat as top_k (query Pinecone automatically)
#     - If `hits` is None → use default top_k from query_pinecone()
#     - If hits list is provided → use directly
#     - Always return LLM answer based ONLY on the context
#     """

#     #  If user passed an INTEGER → treat as top_k (dynamic)
#     if isinstance(hits, int):
#         hits = query_pinecone(question, top_k=hits)
#         print(hits)
#     #  If user passed NOTHING → query Pinecone normally
#     if hits is None:
#         hits = query_pinecone(question)
#     #  No results?
#     if not hits:
#         return "I couldn't find any relevant information about this question in the stored documents."

#     #  Filter by similarity threshold
#     good_hits = [h for h in hits if h["score"] >= SIMILARITY_SCORE_THRESHOLD]
#     if not good_hits:
#         return "I couldn't find any relevant information about this question in the stored documents."

#     #  Build answer context for LLM
#     context = "\n\n---\n\n".join(
#         f"(score={h['score']:.3f})\n{h['metadata']['text']}"
#         for h in good_hits
#     )

#     prompt = f"""
# You are a retrieval QA assistant.

# You MUST follow these rules:
# - Use ONLY the information in the CONTEXT.
# - If the answer is not clearly in the CONTEXT, reply:
#   "I couldn't find this information in the stored documents."
# - Do NOT use any outside knowledge.
# - Do NOT guess or hallucinate.

# CONTEXT:
# {context}

# Question: {question}

# Answer strictly based on the CONTEXT:
# """

#     #  CALL LLM ONLY AFTER getting the relevant Pinecone chunks
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0,
#     )

#     return response.choices[0].message.content.strip()

def generate_rag_response(question: str, hits=10):
    """
    PURE RETRIEVAL MODE:
    Only retrieve and return relevant document chunks from Pinecone.
    NO LLM CALL - LiveKit will handle LLM processing after context injection.
    
    Returns:
        str: Formatted context from top-k relevant chunks, or empty string if no matches.
    """
    
    # If hits is an integer, treat it as top_k
    if isinstance(hits, int):
        hits = query_pinecone(question, top_k=hits)

    # If no hits provided, fetch automatically
    if hits is None:
        hits = query_pinecone(question)

    # If still no hits, return empty (no context available)
    if not hits:
        return ""

    # Filter by similarity threshold
    good_hits = [h for h in hits if h.get("score", 0) >= SIMILARITY_SCORE_THRESHOLD]

    # If no good hits, return empty (no relevant context)
    if not good_hits:
        return ""

    # Build a concise context from the top hits (RETRIEVAL ONLY - NO LLM)
    context = "\n\n---\n\n".join(
        f"[Document Chunk - Score: {h.get('score', 0):.3f}]\n{h['metadata'].get('text', '')}"
        for h in good_hits
    )

    return context


# ------------------------------
# 🔥 MAIN FUNCTION CALLED FROM LIVEKIT
# ------------------------------
def process_pdf_and_ask(uploaded_pdf, question: str) -> str:
    # ====== CASE 1: PDF just uploaded ======
    if uploaded_pdf:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name

        with open(pdf_path, "rb") as f:
            file_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f.read().hex()))

        if not check_if_pdf_exists(file_id):
            chunks = chunk_pdf_to_text(pdf_path)
            upsert_chunks_to_pinecone(chunks, file_id)

    # ====== CASE 2: Query only (speech / no PDF) ======
    hits = query_pinecone(question)
    good_hits = [h for h in hits if h["score"] >= SIMILARITY_SCORE_THRESHOLD]

    if good_hits:
        return generate_rag_response(question, good_hits)

    return generate_general_response(question)


def generate_general_response(query: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": query}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


def query_pinecone(text: str, top_k: int = TOP_K):
    """Query Pinecone and return RAW chunks (no LLM)."""
    embedding = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    ).data[0].embedding

    response = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True
    )
    return response.get("matches", [])
import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL:
    raise RuntimeError("Missing SUPABASE_URL in .env")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_SERVICE_ROLE_KEY in .env")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="RAG Agent API", version="1.2.0")


# ---------------------------
# Request model
# ---------------------------
class AskRequest(BaseModel):
    question: str
    show_sources: Optional[bool] = True
    show_images: Optional[bool] = True


# ---------------------------
# Controlled abbreviation expansion
# (ONLY used for retrieval)
# ---------------------------
def expand_query(question: str) -> str:
    q = question

    replacements = {
        "GPR": "GPR (Ground Penetrating Radar)",
        "NDT": "NDT (Non-Destructive Testing)",
        "CNDT": "Construction Non-Destructive Testing",
        "UPV": "UPV (Ultrasonic Pulse Velocity)",
        "UPE": "UPE (Ultrasonic Pulse Echo)",
        "IE": "Impact-Echo",
        "MCGPR": "Multi-Channel Ground Penetrating Radar",
    }

    for short, expanded in replacements.items():
        if short in q:
            q = q.replace(short, expanded)

    return q


# ---------------------------
# Common Supabase headers
# ---------------------------
def supabase_headers() -> dict:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Accept-Profile": "agents",
        "Content-Profile": "agents",
    }


# ---------------------------
# Embedding
# ---------------------------
def get_query_embedding(question: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    return response.data[0].embedding


# ---------------------------
# Retrieval from Supabase
# ---------------------------
def retrieve_chunks(question: str, match_count: int = 8):
    expanded_question = expand_query(question)
    query_embedding = get_query_embedding(expanded_question)

    rpc_url = f"{SUPABASE_URL}/rest/v1/rpc/match_document_chunks"

    payload = {
        "query_embedding": query_embedding,
        "match_count": match_count
    }

    response = requests.post(
        rpc_url,
        headers=supabase_headers(),
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"Retrieval failed: {response.status_code} {response.text}")

    return response.json()


# ---------------------------
# Image retrieval
# ---------------------------
def get_images_for_documents(document_ids, limit: int = 6):
    if not document_ids:
        return []

    cleaned_ids = [doc_id for doc_id in document_ids if doc_id]
    if not cleaned_ids:
        return []

    ids_csv = ",".join(cleaned_ids)

    url = f"{SUPABASE_URL}/rest/v1/document_images"
    params = {
        "select": "document_id,filename,public_url,page_number,ocr_text,vision_description",
        "document_id": f"in.({ids_csv})",
        "limit": str(limit),
        "order": "page_number.asc",
    }

    response = requests.get(
        url,
        headers=supabase_headers(),
        params=params,
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Image retrieval failed: {response.status_code} {response.text}")

    return response.json()


# ---------------------------
# Build context
# ---------------------------
def build_context(chunks):
    parts = []

    for i, row in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}]\n"
            f"text:\n{row['text_content']}\n"
        )

    return "\n\n".join(parts)


# ---------------------------
# Answer generation (STRICT)
# ---------------------------
def answer_question(question: str, chunks):
    if not chunks:
        return "I don’t know based on the retrieved documents."

    context = build_context(chunks)

    system_prompt = (
        "You are a strict retrieval-grounded assistant.\n\n"
        "Rules:\n"
        "- Answer ONLY using the provided sources\n"
        "- You MAY combine multiple sources if they support the answer\n"
        "- DO NOT use outside knowledge\n"
        "- DO NOT provide alternative meanings or general explanations\n"
        "- DO NOT expand abbreviations unless they appear in the sources\n"
        "- If the answer is not clearly supported, say exactly:\n"
        "  'I don’t know based on the retrieved documents.'\n"
    )

    user_prompt = f"""Question:
{question}

Sources:
{context}

Answer using ONLY the sources.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.output_text.strip()


# ---------------------------
# API endpoint
# ---------------------------
@app.post("/ask")
def ask(request: AskRequest):
    chunks = retrieve_chunks(request.question, match_count=8)
    answer = answer_question(request.question, chunks)

    result = {
        "answer": answer
    }

    if request.show_sources:
        result["sources"] = chunks

    if request.show_images:
        document_ids = list({
            row.get("document_id")
            for row in chunks
            if row.get("document_id")
        })
        result["images"] = get_images_for_documents(document_ids)

    return result

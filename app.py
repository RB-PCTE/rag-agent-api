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

app = FastAPI(title="RAG Agent API", version="1.0.0")


class AskRequest(BaseModel):
    question: str
    show_sources: bool = False


def get_query_embedding(question: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )
    return response.data[0].embedding


def retrieve_chunks(question: str, match_count: int = 5):
    query_embedding = get_query_embedding(question)

    rpc_url = f"{SUPABASE_URL}/rest/v1/rpc/match_document_chunks"

    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "application/json",
        "Content-Profile": "agents",
    }

    payload = {
        "query_embedding": query_embedding,
        "match_count": match_count
    }

    response = requests.post(
        rpc_url,
        headers=headers,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(f"Retrieval failed: {response.status_code} {response.text}")

    return response.json()


def build_context(chunks):
    parts = []

    for i, row in enumerate(chunks, start=1):
        parts.append(
            f"[Source {i}]\n"
            f"document_id: {row['document_id']}\n"
            f"similarity: {row['similarity']}\n"
            f"text:\n{row['text_content']}\n"
        )

    return "\n\n".join(parts)


def answer_question(question: str, chunks):
    if not chunks:
        return "I could not find relevant information in the knowledge base."

    context = build_context(chunks)

system_prompt = (
    "You are a strict retrieval-grounded assistant. "
    "Answer only from the provided sources. "
    "Do not use outside knowledge. "
    "Do not provide alternate meanings, background context, or broader explanations "
    "unless they are explicitly supported by the sources. "
    "If the answer is not supported by the sources, say exactly: "
    "'I don’t know based on the retrieved documents.'"
)

    user_prompt = f"""Question:
{question}

Retrieved sources:
{context}

Write a grounded answer using only the retrieved sources.
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.output_text.strip()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(request: AskRequest):
    chunks = retrieve_chunks(request.question, match_count=5)
    answer = answer_question(request.question, chunks)

    result = {
        "answer": answer
    }

    if request.show_sources:
        result["sources"] = chunks

    return result

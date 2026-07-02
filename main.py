import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse, HealthResponse
from app.retrieval import AssessmentRetriever
from app.agent import handle_chat

CATALOG_PATH = os.getenv("CATALOG_PATH", "data/catalog.json")

app = FastAPI(title="SHL Assessment Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_index: Optional[AssessmentRetriever] = None


@app.on_event("startup")
def load_index():
    global _index

    _index = AssessmentRetriever(CATALOG_PATH)

    print(
        f"Loaded catalog with {len(_index.assessments)} assessments "
        f"from {CATALOG_PATH}"
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if _index is None:
        raise HTTPException(
            status_code=503,
            detail="Catalog not loaded yet.",
        )

    return handle_chat(req.messages, _index)
    
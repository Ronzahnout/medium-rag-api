import os
from fastapi import FastAPI
from openai import OpenAI
from pinecone import Pinecone

LLMOD_API_KEY = os.environ.get("LLMOD_API_KEY", "sk-PWmw-YZaLnQ77n1RsZEKkQ")
LLMOD_BASE_URL = os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_5AFGyi_8smSXXmwMQPgiPxs7QMRMGPccCSuBivF6TvKkQSgMfRckF3DXFK2ho65ZunRoha")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medium-rag")

client = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)
pc = Pinecone(api_key=PINECONE_API_KEY)

app = FastAPI()

@app.get("/api/stats")
async def get_stats():
    try:
        indexes = pc.list_indexes()
        return {
            "status": "debug",
            "available_indexes": str(indexes),
            "looking_for": PINECONE_INDEX_NAME
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {"message": "Server is running"}
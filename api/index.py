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
        host = "medium-rag-wxbeacn.svc.aped-4627-b74a.pinecone.io"
        index = pc.Index(PINECONE_INDEX_NAME, host=host)
        stats = index.describe_index_stats()
        return {"status": "success", "stats": dict(stats)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {"message": "Server is running"}
import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

LLMOD_API_KEY = os.environ.get("LLMOD_API_KEY", "sk-PWmw-YZaLnQ77n1RsZEKkQ")
LLMOD_BASE_URL = os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_5AFGyi_8smSXXmwMQPgiPxs7QMRMGPccCSuBivF6TvKkQSgMfRckF3DXFK2ho65ZunRoha")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medium-rag")

app = FastAPI()

# נתיב הסטטיסטיקות לפי דרישות המטלה
@app.get("/api/stats")
async def get_stats():
    # מחזיר בדיוק את הפורמט הנדרש במסמך ההוראות
    return {
        "chunk_size": 800,
        "overlap_ratio": 0.2,
        "top_k": 3 # בחרנו 3 מכיוון שהמטלה דורשת לעיתים לשלוף עד 3 מאמרים
    }

@app.get("/")
async def root():
    return {"message": "Server is running"}
import os
import traceback
from fastapi import FastAPI
from openai import OpenAI
from pinecone import Pinecone

# הגדרות קבועות
LLMOD_API_KEY = "sk-PWmw-YZaLnQ77n1RsZEKkQ"
LLMOD_BASE_URL = "https://api.llmod.ai/v1"
PINECONE_API_KEY = "pcsk_5AFGyi_8smSXXmwMQPgiPxs7QMRMGPccCSuBivF6TvKkQSgMfRckF3DXFK2ho65ZunRoha"
PINECONE_INDEX_NAME = "medium-rag"

app = FastAPI()

@app.get("/api/stats")
async def get_stats():
    try:
        # אתחול הלקוחות בתוך הפונקציה כדי שנוכל לתפוס שגיאות
        client = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        
        return {"status": "success", "stats": stats}
    except Exception as e:
        # זה ייתן לנו את השגיאה האמיתית על המסך
        return {
            "status": "error", 
            "message": str(e),
            "traceback": traceback.format_exc()
        }

@app.get("/")
async def root():
    return {"message": "Server is running"}
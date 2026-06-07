from fastapi import FastAPI
from openai import OpenAI
from pinecone import Pinecone

# הגדרות המערכת
# הנה המפתחות שלך מוזנים ישירות כדי ש-Vercel לא יפספס אותם
LLMOD_API_KEY = "sk-PWmw-YZaLnQ77n1RsZEKkQ"
LLMOD_BASE_URL = "https://api.llmod.ai/v1"
PINECONE_API_KEY = "pcsk_5AFGyi_8smSXXmwMQPgiPxs7QMRMGPccCSuBivF6TvKkQSgMfRckF3DXFK2ho65ZunRoha"
PINECONE_INDEX_NAME = "medium-rag"

# אתחול הלקוחות
client = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)
pc = Pinecone(api_key=PINECONE_API_KEY)

app = FastAPI()

@app.get("/api/stats")
async def get_stats():
    try:
        # בדיקת חיבור ל-Pinecone
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        
        # מחזירים תשובה שהשרת עובד
        return {
            "status": "success",
            "index_name": PINECONE_INDEX_NAME,
            "stats": stats
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
async def root():
    return {"message": "Server is running! Try /api/stats"}
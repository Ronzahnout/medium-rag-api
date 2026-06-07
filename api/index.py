import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

# הגדרת המבנה של בקשת המשתמש
class PromptRequest(BaseModel):
    question: str

LLMOD_API_KEY = os.environ.get("LLMOD_API_KEY", "sk-PWmw-YZaLnQ77n1RsZEKkQ")
LLMOD_BASE_URL = os.environ.get("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "pcsk_5AFGyi_8smSXXmwMQPgiPxs7QMRMGPccCSuBivF6TvKkQSgMfRckF3DXFK2ho65ZunRoha")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "medium-rag")

app = FastAPI()
client = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)
pc = Pinecone(api_key=PINECONE_API_KEY)

# ==========================================
# 1. נתיב הסטטיסטיקות (עובד מושלם)
# ==========================================
@app.get("/api/stats")
async def get_stats():
    return {
        "chunk_size": 800,
        "overlap_ratio": 0.2,
        "top_k": 3
    }

# ==========================================
# 2. מערכת ה-RAG המלאה!
# ==========================================
@app.post("/api/prompt")
async def process_prompt(req: PromptRequest):
    try:
        # א. הפיכת השאלה לוקטור
        emb_res = client.embeddings.create(
            input=req.question,
            model="4UHRUIN-text-embedding-3-small"
        )
        query_vector = emb_res.data[0].embedding

        # ב. חיפוש במאגר Pinecone
        index = pc.Index(PINECONE_INDEX_NAME)
        search_results = index.query(
            vector=query_vector,
            top_k=3,
            include_metadata=True
        )

        # ג. ארגון התוצאות לפורמט הנדרש
        context_chunks = []
        context_text = ""
        
        for match in search_results.matches:
            meta = match.metadata
            chunk_str = meta.get("text", "")
            
            # הוספה למערך ה-context לפי המבנה במטלה
            context_chunks.append({
                "article_id": str(meta.get("article_id", "")),
                "title": str(meta.get("title", "")),
                "chunk": chunk_str,
                "score": float(match.score)
            })
            # שרשור הטקסט עבור ה-Prompt של המודל
            context_text += f"\n--- Article: {meta.get('title')} ---\n{chunk_str}\n"

        # ד. הגדרת הפרומפט בדיוק כפי שהוגדר במטלה
        system_prompt = (
            "You are a Medium-article assistant that answers questions strictly and only "
            "based on the Medium articles dataset context provided to you (metadata "
            "and article passages). You must not use any external knowledge, the open "
            "internet, or information that is not explicitly contained in the retrieved "
            "context. If the answer cannot be determined from the provided context, "
            "respond: \"I don't know based on the provided Medium articles data.\" "
            "Always explain your answer using the given context, quoting or "
            "paraphrasing the relevant article passage or metadata when helpful."
        )
        
        user_prompt = f"Context:\n{context_text}\n\nQuestion:\n{req.question}"

        # ה. פנייה למודל צ'אט
        chat_res = client.chat.completions.create(
            model="4UHRUIN-gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = chat_res.choices[0].message.content

        # ו. החזרת התשובה המלאה
        return {
            "response": answer,
            "context": context_chunks,
            "Augmented_prompt": {
                "System": system_prompt,
                "User": user_prompt
            }
        }

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "Server is running"} 

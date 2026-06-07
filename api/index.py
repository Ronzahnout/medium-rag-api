from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

app = FastAPI()

# ==========================================
# 1. הגדרות המערכת והמפתחות שלך
# ==========================================
# חשוב: לפני ההעלאה ל-Vercel נהוג להעביר את אלו למשתני סביבה, אבל כרגע נשאיר קשיח לבדיקות.
LLMOD_API_KEY = "sk-PWmw-YZaLnQ77n1RsZEKkQ"
LLMOD_BASE_URL = "https://api.llmod.ai/v1"
PINECONE_API_KEY = "pcsk_5AFGyi_8smSXXmwMQPgiPxs7QMRMGPccCSuBivF6TvKkQSgMfRckF3DXFK2ho65ZunRoha"
PINECONE_INDEX_NAME = "medium-rag"

client = OpenAI(api_key=LLMOD_API_KEY, base_url=LLMOD_BASE_URL)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ההגדרות שבחרנו עבור ה-RAG
CHUNK_SIZE = 800
OVERLAP_RATIO = 0.2
TOP_K = 5  # שולפים 5 תוצאות מהאינדקס כדי שלמודל יהיה מספיק מידע לבסס עליו תשובה

# הפרומפט המערכתי בדיוק כפי שהוגדר במטלה
SYSTEM_PROMPT = """You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don't know based on the provided Medium articles data."
Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful."""

# מודל הקלט הנדרש
class PromptRequest(BaseModel):
    question: str

# ==========================================
# 2. נתיב ה-POST הראשי לקבלת שאלות
# ==========================================
@app.post("/api/prompt")
async def process_prompt(request: PromptRequest):
    question = request.question
    
    # א. המרת השאלה לוקטור (Embedding)
    res_emb = client.embeddings.create(
        input=question,
        model="4UHRUIN-text-embedding-3-small"
    )
    query_vector = res_emb.data[0].embedding
    
    # ב. חיפוש התוצאות הרלוונטיות ב-Pinecone
    search_results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True
    )
    
    # ג. הכנת המידע למודל ולהחזרה ללקוח
    context_for_output = []
    context_text_for_model = ""
    
    for match in search_results.matches:
        meta = match.metadata
        chunk_text = meta.get("text", "")
        title = meta.get("title", "Unknown Title")
        article_id = meta.get("article_id", "Unknown ID")
        score = float(match.score)
        
        # מבנה האובייקט שיוחזר ללקוח על פי הגדרת המטלה
        context_for_output.append({
            "article_id": str(article_id),
            "title": str(title),
            "chunk": chunk_text,
            "score": score
        })
        
        # בניית הטקסט שהמודל יקרא
        context_text_for_model += f"Title: {title}\nArticle ID: {article_id}\nContent: {chunk_text}\n\n"
        
    # ד. בניית בקשת ה-User המלאה
    user_prompt = f"Context:\n{context_text_for_model}\n\nQuestion: {question}"
    
    # ה. קריאה למודל השפה לקבלת תשובה מבוססת טקסט בלבד
    chat_res = client.chat.completions.create(
        model="4UHRUIN-gpt-5-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0 # חשוב! טמפרטורה 0 מונעת מהמודל "להמציא" דברים (Hallucinations)
    )
    
    answer = chat_res.choices[0].message.content
    
    # ו. החזרת התשובה בפורמט ה-JSON המחמיר
    return {
        "response": answer,
        "context": context_for_output,
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt
        }
    }

# ==========================================
# 3. נתיב ה-GET לקבלת הגדרות ה-RAG
# ==========================================
@app.get("/api/stats")
async def get_stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }
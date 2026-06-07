import pandas as pd
import tiktoken
from openai import OpenAI
from pinecone import Pinecone

# ==========================================
# 1. הגדרות המערכת והמפתחות שלך
# ==========================================
# אלו הפרטים שמצאת במערכת של הקורס:
LLMOD_API_KEY = "sk-PWmw-YZaLnQ77n1RsZEKkQ"
LLMOD_BASE_URL = "https://api.llmod.ai/v1"

# ===> כאן את צריכה להדביק את מפתח ה-Pinecone שלך <===
PINECONE_API_KEY = "pcsk_5AFGyi_8smSXXmwMQPgiPxs7QMRMGPccCSuBivF6TvKkQSgMfRckF3DXFK2ho65ZunRoha" 
PINECONE_INDEX_NAME = "medium-rag"

# ==========================================
# 2. התחברות לשרתים
# ==========================================
# חיבור לשרת הפרוקסי של הקורס
client = OpenAI(
    api_key=LLMOD_API_KEY,
    base_url=LLMOD_BASE_URL
)

# חיבור למסד הנתונים Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# ==========================================
# 3. פונקציית חיתוך הטקסט
# ==========================================
def create_chunks(text, max_tokens=800, overlap_ratio=0.2):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(str(text))
    chunks = []
    overlap_tokens = int(max_tokens * overlap_ratio)
    step_size = max_tokens - overlap_tokens
    
    if len(tokens) <= max_tokens:
        return [str(text)]
        
    for i in range(0, len(tokens), step_size):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(encoding.decode(chunk_tokens))
    return chunks

# ==========================================
# 4. עיבוד, המרה לווקטורים ואחסון
# ==========================================
def main():
    print("Loading data...")
    # קוראים רק 50 מאמרים כדי לבדוק את התהליך ולשמור על התקציב
    df = pd.read_csv("medium-english-50mb.csv", nrows=50)
    
    vectors_to_upsert = []
    
    print("Starting embedding process...")
    for i, row in df.iterrows():
        chunks = create_chunks(row['text'])
        
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_id = f"article_{i}_chunk_{chunk_idx}"
            
            try:
                # שליחה למודל ההטמעה של הקורס
                response = client.embeddings.create(
                    input=chunk_text,
                    model="4UHRUIN-text-embedding-3-small"
                )
                embedding = response.data[0].embedding
                
                # בניית הווקטור עם המטא-דאטה שיישמר ב-Pinecone
                vectors_to_upsert.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        "article_id": str(i),
                        "title": str(row['title']),
                        "url": str(row['url']),
                        "text": chunk_text
                    }
                })
                print(f"Successfully embedded: {chunk_id}")
                
            except Exception as e:
                print(f"Error embedding {chunk_id}: {e}")
                
    # מעלים הכל ל-Pinecone בקבוצות של 100 כדי למנוע עומס
    print(f"\nUploading {len(vectors_to_upsert)} vectors to Pinecone...")
    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)
        
    print("\nQueen Ronza, the database is ready! Data successfully stored in Pinecone.")

if __name__ == "__main__":
    main()
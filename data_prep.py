import pandas as pd
import tiktoken

def create_chunks(text, max_tokens=1000, overlap_ratio=0.2):
    """
    פונקציה המקבלת טקסט וחותכת אותו למקטעים לפי כמות טוקנים וחפיפה.
    """
    # אתחול הטוקנייזר (cl100k_base מתאים לרוב המודלים החדשים)
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # המרת הטקסט לטוקנים
    tokens = encoding.encode(text)
    
    chunks = []
    overlap_tokens = int(max_tokens * overlap_ratio)
    step_size = max_tokens - overlap_tokens
    
    # אם הטקסט קצר מהמקסימום, נחזיר אותו כפי שהוא
    if len(tokens) <= max_tokens:
        return [text]
    
    # חיתוך הטוקנים למקטעים עם חפיפה
    for i in range(0, len(tokens), step_size):
        chunk_tokens = tokens[i:i + max_tokens]
        # המרה חזרה מטוקנים לטקסט קריא
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        
    return chunks

def process_medium_dataset(file_path, num_articles=50):
    """
    טעינת חלק מהדאטה ועיבוד שלו למקטעים מוכנים להטמעה
    """
    print(f"Loading first {num_articles} articles from {file_path}...")
    # טעינת תת-קבוצה כדי לחסוך בעלויות פיתוח
    df = pd.read_csv(file_path, nrows=num_articles)
    
    processed_data = []
    
    for index, row in df.iterrows():
        # שימוש ב-800 טוקנים וחפיפה של 20% כברירת מחדל בטוחה
        article_chunks = create_chunks(str(row['text']), max_tokens=800, overlap_ratio=0.2)
        
        for chunk_id, chunk_text in enumerate(article_chunks):
            processed_data.append({
                "article_id": str(index),
                "title": row['title'],
                "authors": row['authors'],
                "url": row['url'],
                "chunk_id": f"{index}_{chunk_id}",
                "text": chunk_text
            })
            
    result_df = pd.DataFrame(processed_data)
    print(f"Generated {len(result_df)} chunks from {num_articles} articles.")
    return result_df

# הרצת התהליך
if __name__ == "__main__":
    file_name = "medium-english-50mb.csv"
    chunked_df = process_medium_dataset(file_name, num_articles=50)
    print(chunked_df.head())
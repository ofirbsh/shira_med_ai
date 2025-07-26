import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from openai import OpenAI
import numpy as np
import pickle

# טעינת משתני סביבה
load_dotenv()
print("DEBUG KEY (pdf_loader):", os.getenv("SHIRA_MED_AI_KEY"))

# הפעלת לקוח OpenAI
client = OpenAI(api_key=os.getenv("SHIRA_MED_AI_KEY"))

# קובץ Embeddings
EMBEDDINGS_FILE = "pdf_embeddings.pkl"


def extract_text_from_pdf(pdf_path: str) -> str:
    """קריאת PDF והחזרת כל הטקסט כ-String."""
    reader = PdfReader(pdf_path)
    text = []
    for i, page in enumerate(reader.pages, start=1):
        print(f"[INFO] Extracting text from page {i}/{len(reader.pages)}...")
        text.append(page.extract_text())
    return "\n".join(text)


def chunk_text(text: str, chunk_size=500) -> list:
    """פיצול הטקסט לקטעים קטנים יותר (500 תווים)."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def build_embeddings(pdf_path: str):
    """בניית embeddings ושמירה לקובץ."""
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"[INFO] Embeddings file '{EMBEDDINGS_FILE}' already exists. Skipping...")
        return

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    embeddings = []

    print(f"[INFO] Starting embeddings creation: {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks, start=1):
        print(f"[INFO] Processing chunk {i}/{len(chunks)}...")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        vector = response.data[0].embedding
        embeddings.append({"chunk": chunk, "embedding": vector})

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"[INFO] Embeddings saved to {EMBEDDINGS_FILE}")


def load_embeddings():
    """טעינת embeddings מהקובץ."""
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError("לא נמצא קובץ embeddings, צריך להריץ build_embeddings קודם.")
    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)


def semantic_search(query: str, top_k=3):
    """חיפוש סמנטי לפי השאלה."""
    embeddings = load_embeddings()

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    scores = []
    for entry in embeddings:
        vector = np.array(entry["embedding"])
        score = np.dot(vector, query_embedding) / (np.linalg.norm(vector) * np.linalg.norm(query_embedding))
        scores.append((score, entry["chunk"]))

    scores.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scores[:top_k]]

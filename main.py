from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import pickle
from datetime import datetime
from openai import OpenAI
from pdf_loader import semantic_search

# טעינת משתני הסביבה (כולל ה-API KEY)
load_dotenv()
print("DEBUG KEY (main.py):", os.getenv("SHIRA_MED_AI_KEY"))
client = OpenAI(api_key=os.getenv("SHIRA_MED_AI_KEY"))

app = FastAPI()

# ---- כאן אנחנו מוסיפים את ה-CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
        "https://clever-ganache-0ccf87.netlify.app/"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------------------

@app.get("/")
def root():
    print("DEBUG KEY:", os.getenv("SHIRA_MED_AI_KEY"))

@app.post("/ask")
async def ask_question(question: str = Body(..., embed=True)):
    """
    API בסיסי שמחזיר תשובה משירות ChatGPT
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}]
    )
    return {"answer": response.choices[0].message.content}

@app.post("/ask_pdf")
async def ask_pdf(question: str = Body(..., embed=True)):
    context_chunks = semantic_search(question, top_k=2)
    context_text = "\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"ענה על השאלה תוך שימוש בטקסט הבא:\n{context_text}"},
            {"role": "user", "content": question}
        ]
    )
    return {"answer": response.choices[0].message.content}

@app.post("/ask_pdf_debug")
async def ask_pdf_debug(question: str = Body(..., embed=True)):
    chunks = semantic_search(question, top_k=2)
    return {"chunks_found": chunks}


@app.get("/pdf_status")
def pdf_status():
    if not os.path.exists("pdf_embeddings.pkl"):
        return {"status": "no_embeddings", "message": "אין קובץ embeddings. צריך להריץ prepare_pdf."}

    # בדיקת כמות קטעים
    with open("pdf_embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
        chunks_count = len(embeddings)

    modified_time = datetime.fromtimestamp(os.path.getmtime("pdf_embeddings.pkl")).strftime("%Y-%m-%d %H:%M:%S")

    return {
        "status": "ready",
        "chunks_count": chunks_count,
        "last_modified": modified_time
    }

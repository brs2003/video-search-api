from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI()

# ✅ ADD THIS CORS SECTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/ask")
def ask_question(q: Question):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(q.question)
        return {"answer": response.text}
    except Exception as e:
        return {"error": str(e)}

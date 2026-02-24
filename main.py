from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import yt_dlp
import os
import time
import uuid

app = FastAPI()

# CORS (required for validator)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Request model (must match validator)
class AskRequest(BaseModel):
    video_url: str
    topic: str


# Download AUDIO ONLY (very important)
def download_audio(video_url: str, output_filename: str):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_filename,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "extractaudio": True,
        "audioformat": "mp3",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


@app.post("/ask")
def ask(request: AskRequest):
    temp_filename = f"{uuid.uuid4()}.mp3"

    try:
        # STEP 1 — Download audio
        download_audio(request.video_url, temp_filename)

        if not os.path.exists(temp_filename):
            raise HTTPException(status_code=500, detail="Audio download failed")

        # STEP 2 — Upload to Gemini Files API
        uploaded_file = client.files.upload(file=temp_filename)

        # STEP 3 — Wait until ACTIVE
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            raise HTTPException(status_code=500, detail="File processing failed")

        # STEP 4 — Structured schema
        response_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "timestamp": types.Schema(
                    type=types.Type.STRING,
                    pattern=r"^\d{2}:\d{2}:\d{2}$"
                )
            },
            required=["timestamp"],
        )

        # STEP 5 — Ask Gemini
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                uploaded_file,
                f"Find the FIRST time the topic '{request.topic}' is spoken. "
                "Return ONLY the timestamp in HH:MM:SS format."
            ],
            config=types.GenerateContentConfig(
                response_schema=response_schema,
                response_mime_type="application/json",
            ),
        )

        # Safer parsing
        if not response.parsed:
            raise HTTPException(status_code=500, detail="Gemini parsing failed")

        timestamp = response.parsed["timestamp"]

        return {
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # STEP 6 — Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

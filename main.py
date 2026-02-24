from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
import yt_dlp
import os
import re
import time
import uuid

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Request model (MUST match assignment)
class AskRequest(BaseModel):
    video_url: str
    topic: str


# Extract YouTube video ID
def extract_video_id(url: str):
    pattern = r"(?:v=|youtu\.be/)([^&]+)"
    match = re.search(pattern, url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    return match.group(1)


# Download audio only using yt-dlp
def download_audio(video_url: str, output_path: str):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


@app.post("/ask")
def ask(request: AskRequest):
    temp_filename = f"{uuid.uuid4()}.mp3"

    try:
        # Step 1: Download audio
        download_audio(request.video_url, temp_filename)

        # Step 2: Upload to Gemini Files API
        uploaded_file = client.files.upload(file=temp_filename)

        # Step 3: Poll until file becomes ACTIVE
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            raise HTTPException(status_code=500, detail="File processing failed")

        # Step 4: Structured output schema
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

        # Step 5: Ask Gemini to find timestamp
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                uploaded_file,
                f"Find the FIRST time the topic '{request.topic}' is spoken in this audio. "
                "Return ONLY the timestamp in HH:MM:SS format."
            ],
            config=types.GenerateContentConfig(
                response_schema=response_schema,
                response_mime_type="application/json",
            ),
        )

        result = response.parsed

        return {
            "timestamp": result["timestamp"],
            "video_url": request.video_url,
            "topic": request.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Step 6: Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai
import os
import re

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class AskRequest(BaseModel):
    video_url: str
    topic: str


def extract_video_id(url: str):
    pattern = r"(?:v=|youtu\.be/)([^&]+)"
    match = re.search(pattern, url)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    return match.group(1)


def seconds_to_hhmmss(seconds: float):
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02}:{m:02}:{s:02}"


@app.post("/ask")
def ask(req: AskRequest):
    try:
        video_id = extract_video_id(req.video_url)

        transcript = YouTubeTranscriptApi.get_transcript(video_id)

        # Combine transcript text
        full_text = " ".join([t["text"] for t in transcript])

        # Ask Gemini to locate phrase
        model = genai.GenerativeModel("gemini-1.5-pro")

        prompt = f"""
        Here is a YouTube transcript:

        {full_text}

        Find when the topic '{req.topic}' is FIRST mentioned.
        Respond ONLY with the exact phrase from transcript that best matches.
        """

        response = model.generate_content(prompt)
        matched_text = response.text.strip()

        # Find matched segment timestamp
        for segment in transcript:
            if matched_text.lower() in segment["text"].lower():
                timestamp = seconds_to_hhmmss(segment["start"])
                return {
                    "timestamp": timestamp,
                    "video_url": req.video_url,
                    "topic": req.topic
                }

        raise HTTPException(status_code=404, detail="Topic not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import os
import time
import tempfile
import yt_dlp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

app = FastAPI()

# Request model
class AskRequest(BaseModel):
    video_url: str
    topic: str


# Download audio only
def download_audio(video_url: str, output_path: str):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


@app.post("/ask")
def ask(data: AskRequest):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_file.close()

    try:
        # Step 1: Download audio
        download_audio(data.video_url, temp_file.name)

        # Step 2: Upload to Gemini Files API
        file = genai.upload_file(temp_file.name)

        # Step 3: Poll until ACTIVE
        while file.state.name != "ACTIVE":
            time.sleep(2)
            file = genai.get_file(file.name)

        # Step 4: Call Gemini with structured output
        model = genai.GenerativeModel("gemini-2.0-flash")

        response = model.generate_content(
            [
                file,
                f"Find the FIRST timestamp (HH:MM:SS) when this topic is spoken: '{data.topic}'. Return only JSON."
            ],
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "timestamp": {
                            "type": "string",
                            "pattern": "^\d{2}:\d{2}:\d{2}$"
                        }
                    },
                    "required": ["timestamp"]
                }
            }
        )

        result = response.text

        return {
            "timestamp": eval(result)["timestamp"],
            "video_url": data.video_url,
            "topic": data.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Step 5: Cleanup
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
import os
import time
import tempfile
import json
import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Configure Gemini API Key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = FastAPI()

# ✅ CORS FIX (VERY IMPORTANT for validator)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Model
class AskRequest(BaseModel):
    video_url: str
    topic: str


# Download audio only (NO ffmpeg required)
def download_audio(video_url: str, output_path: str):
    ydl_opts = {
        "format": "bestaudio",
        "outtmpl": output_path,
        "quiet": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


@app.post("/ask")
def ask(data: AskRequest):

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()

    try:
        # 1️⃣ Download audio
        download_audio(data.video_url, temp_file.name)

        # 2️⃣ Upload to Gemini Files API
        file = genai.upload_file(temp_file.name)

        # 3️⃣ Wait until file becomes ACTIVE
        while file.state.name != "ACTIVE":
            time.sleep(2)
            file = genai.get_file(file.name)

        # 4️⃣ Use Gemini to find timestamp
        model = genai.GenerativeModel("gemini-2.0-flash")

        response = model.generate_content(
            [
                file,
                f"""
Find the FIRST timestamp when the topic "{data.topic}" is spoken.
Return ONLY JSON in this format:
{{"timestamp": "HH:MM:SS"}}
Do not return anything else.
"""
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

        parsed = json.loads(response.text)

        return {
            "timestamp": parsed["timestamp"],
            "video_url": data.video_url,
            "topic": data.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 5️⃣ Cleanup temp file
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


# Required for Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

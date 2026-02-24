import os
import time
import tempfile
import json
import yt_dlp
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

app = FastAPI()

class AskRequest(BaseModel):
    video_url: str
    topic: str


def download_audio(video_url: str, output_path: str):
    ydl_opts = {
        "format": "bestaudio",
        "outtmpl": output_path,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


@app.post("/ask")
def ask(data: AskRequest):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()

    try:
        download_audio(data.video_url, temp_file.name)

        file = genai.upload_file(temp_file.name)

        while file.state.name != "ACTIVE":
            time.sleep(2)
            file = genai.get_file(file.name)

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

        parsed = json.loads(response.text)

        return {
            "timestamp": parsed["timestamp"],
            "video_url": data.video_url,
            "topic": data.topic
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

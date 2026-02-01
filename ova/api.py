import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .pipeline import OVAPipeline, OVAProfile

OVA_PROFILE = os.getenv("OVA_PROFILE", "default")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPELINE = OVAPipeline.from_profile(OVAProfile.from_str(OVA_PROFILE))


@app.post("/chat", response_class=Response)
async def chat_request_handler(request: Request):
    audio_in = await request.body()

    transcribed_text = PIPELINE.transcribe(audio_in)

    if not transcribed_text:
        # return "empty" bytes if no transcription
        return Response(content=bytes(), media_type="audio/wav")

    chat_response = PIPELINE.chat(transcribed_text)

    audio_out = PIPELINE.tts(chat_response)

    return Response(content=audio_out, media_type="audio/wav")


@app.post("/tts", response_class=Response)
async def tts_request_handler(request: Request):
    """Text-to-speech endpoint that accepts text and returns audio."""
    return Response(content=bytes(), media_type="audio/wav", status_code=200)

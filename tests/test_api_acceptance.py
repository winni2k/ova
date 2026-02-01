"""
Acceptance tests for the OVA API endpoints.
"""

from ova.api import PIPELINE
from ova.pipeline import OVAPipeline


def test_tts_endpoint_returns_200(client):
    """
    ACCEPTANCE TEST: The /tty endpoint should return a 200 status code.

    This is a basic health check to ensure the tty endpoint is accessible
    and responds successfully.
    """
    response = client.post("/tts", json={"text": "Hi world!"})
    assert response.status_code == 200


def test_tts_endpoint_returns_wav_of_speech(client):
    # given
    pipeline = PIPELINE
    input_text = "hi world"

    # when
    response = client.post("/tts", json={"text": input_text})

    # then
    response_text = pipeline.transcribe(response.content).lower()

    assert "hi" in response_text
    assert "world" in response_text


def test_chat_endpoint_returns_200(client, wav_blob):
    """
    ACCEPTANCE TEST: The /chat endpoint should return a 200 status code.

    This is a basic health check to ensure the chat endpoint is accessible
    and responds successfully with valid audio input.
    """
    response = client.post(
        "/chat", content=wav_blob, headers={"Content-Type": "audio/wav"}
    )
    assert response.status_code == 200

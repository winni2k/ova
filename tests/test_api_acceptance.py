"""
Acceptance tests for the OVA API endpoints.
"""


def test_tts_endpoint_returns_200(client, wav_blob):
    """
    ACCEPTANCE TEST: The /tty endpoint should return a 200 status code.

    This is a basic health check to ensure the tty endpoint is accessible
    and responds successfully.
    """
    response = client.post(
        "/tts", content=wav_blob, headers={"Content-Type": "audio/wav"}
    )
    assert response.status_code == 200


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

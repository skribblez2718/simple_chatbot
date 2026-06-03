"""Tests for the simple chatbot backend API."""

from __future__ import annotations

from httpx import AsyncClient
from pytest import mark


class TestRootEndpoint:
    """Tests for GET /."""

    async def test_root_returns_ok(self, client: AsyncClient) -> None:
        """GET / should return a 200 status and info about the API."""
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "Simple Chatbot API"


class TestChatEndpoint:
    """Tests for POST /chat."""

    async def test_chat_with_valid_message(self, client: AsyncClient) -> None:
        """POST /chat with a valid message should return a response."""
        response = await client.post(
            "/chat",
            json={"message": "Hello, how are you?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0

    async def test_chat_with_empty_message(self, client: AsyncClient) -> None:
        """POST /chat with an empty message should return 422."""
        response = await client.post(
            "/chat",
            json={"message": ""},
        )
        assert response.status_code == 422

    async def test_chat_with_none_message(self, client: AsyncClient) -> None:
        """POST /chat with None message should return 422."""
        response = await client.post(
            "/chat",
            json={"message": None},
        )
        assert response.status_code == 422

    async def test_chat_with_missing_message(self, client: AsyncClient) -> None:
        """POST /chat without message field should return 422."""
        response = await client.post("/chat", json={})
        assert response.status_code == 422

    async def test_chat_with_non_string_message(self, client: AsyncClient) -> None:
        """POST /chat with a non-string message should return 422."""
        response = await client.post(
            "/chat",
            json={"message": 123},
        )
        assert response.status_code == 422

    @mark.skip("Integration test: requires model to be loaded")
    async def test_chat_with_long_message(self, client: AsyncClient) -> None:
        """POST /chat with a very long message should still work."""
        long_msg = "Hello " * 500
        response = await client.post(
            "/chat",
            json={"message": long_msg},
        )
        assert response.status_code == 200
        assert "response" in response.json()

    async def test_chat_returns_generic_error_on_failure(self, client: AsyncClient) -> None:
        """POST /chat should return a generic error, not a traceback."""
        response = await client.post(
            "/chat",
            json={"message": "Hi"},
        )
        # Even on error, never expose internal details
        if response.status_code == 500:
            data = response.json()
            assert "traceback" not in str(data).lower()
            assert "stack" not in str(data).lower()
            assert "error" in data or "detail" in data


class TestConversationsEndpoint:
    """Tests for conversation management."""

    async def test_list_conversations_returns_list(self, client: AsyncClient) -> None:
        """GET /conversations should return a list."""
        response = await client.get("/conversations")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    async def test_create_conversation(self, client: AsyncClient) -> None:
        """POST /conversations should create a new conversation."""
        response = await client.post(
            "/conversations",
            json={"title": "Test Chat"},
        )
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["title"] == "Test Chat"
        assert "messages" in data
        assert "created_at" in data

    async def test_create_conversation_default_title(self, client: AsyncClient) -> None:
        """POST /conversations without title should use a default."""
        response = await client.post("/conversations", json={})
        assert response.status_code == 201
        data = response.json()
        assert "title" in data
        assert len(data["title"]) > 0

    async def test_get_conversation_by_id(self, client: AsyncClient) -> None:
        """GET /conversations/{id} should return the conversation."""
        create_resp = await client.post("/conversations", json={"title": "Get Test"})
        conv_id = create_resp.json()["id"]

        response = await client.get(f"/conversations/{conv_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == conv_id
        assert data["title"] == "Get Test"

    async def test_get_conversation_not_found(self, client: AsyncClient) -> None:
        """GET /conversations/{id} with invalid id should return 404."""
        response = await client.get("/conversations/nonexistent-id")
        assert response.status_code == 404

    async def test_add_message_to_conversation(self, client: AsyncClient) -> None:
        """POST /conversations/{id}/messages should add a message and return AI response."""
        create_resp = await client.post("/conversations", json={"title": "Message Test"})
        conv_id = create_resp.json()["id"]

        response = await client.post(
            f"/conversations/{conv_id}/messages",
            json={"content": "What is AI?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "user_message" in data
        assert data["user_message"]["content"] == "What is AI?"
        assert data["user_message"]["role"] == "user"
        assert "assistant_message" in data
        assert data["assistant_message"]["role"] == "assistant"
        assert len(data["assistant_message"]["content"]) > 0

    async def test_add_message_to_nonexistent_conversation(self, client: AsyncClient) -> None:
        """POST /conversations/{id}/messages with invalid id should return 404."""
        response = await client.post(
            "/conversations/nonexistent-id/messages",
            json={"content": "Hello"},
        )
        assert response.status_code == 404

    async def test_add_message_empty_content(self, client: AsyncClient) -> None:
        """POST /conversations/{id}/messages with empty content should return 422."""
        create_resp = await client.post("/conversations", json={"title": "Empty Msg"})
        conv_id = create_resp.json()["id"]

        response = await client.post(
            f"/conversations/{conv_id}/messages",
            json={"content": ""},
        )
        assert response.status_code == 422

    async def test_delete_conversation(self, client: AsyncClient) -> None:
        """DELETE /conversations/{id} should remove the conversation."""
        create_resp = await client.post("/conversations", json={"title": "To Delete"})
        conv_id = create_resp.json()["id"]

        response = await client.delete(f"/conversations/{conv_id}")
        assert response.status_code == 204

        # Verify it's gone
        get_resp = await client.get(f"/conversations/{conv_id}")
        assert get_resp.status_code == 404

    async def test_delete_nonexistent_conversation(self, client: AsyncClient) -> None:
        """DELETE /conversations/{id} with invalid id should return 404."""
        response = await client.delete("/conversations/nonexistent-id")
        assert response.status_code == 404


class TestCORSEndpoint:
    """Tests for CORS configuration."""

    async def test_cors_headers_present(self, client: AsyncClient) -> None:
        """CORS headers should be present for cross-origin requests."""
        response = await client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "http://localhost:8501"

    async def test_cors_rejects_unknown_origin(self, client: AsyncClient) -> None:
        """CORS should reject origins that are not Streamlit."""
        response = await client.options(
            "/chat",
            headers={
                "Origin": "https://evil-site.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        allow_origin = response.headers.get("access-control-allow-origin", "")
        assert "evil-site.com" not in allow_origin

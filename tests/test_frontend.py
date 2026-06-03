"""Tests for the simple chatbot frontend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import requests


class TestAPIHelper:
    """Tests for the API helper functions in the frontend."""

    def test_check_backend_returns_true_when_reachable(self) -> None:
        """check_backend() should return True when backend /health responds."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("requests.get", return_value=mock_response) as mock_get:
            from frontend.app import check_backend

            result = check_backend()
            assert result is True
            mock_get.assert_called_once_with(
                "http://127.0.0.1:8000/health", timeout=5
            )

    def test_check_backend_returns_false_on_connection_error(self) -> None:
        """check_backend() should return False when backend is unreachable."""
        with patch(
            "requests.get", side_effect=requests.ConnectionError
        ):
            from frontend.app import check_backend

            result = check_backend()
            assert result is False

    def test_check_backend_returns_false_on_timeout(self) -> None:
        """check_backend() should return False on timeout."""
        with patch("requests.get", side_effect=requests.Timeout):
            from frontend.app import check_backend

            result = check_backend()
            assert result is False

    def test_send_message_api_returns_response(self) -> None:
        """send_message_api() should return both user and assistant messages."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "user_message": {
                "role": "user",
                "content": "Hi",
                "timestamp": "now",
            },
            "assistant_message": {
                "role": "assistant",
                "content": "Hello!",
                "timestamp": "now",
            },
        }

        with patch("requests.post", return_value=mock_response) as mock_post:
            from frontend.app import send_message_api

            result = send_message_api("conv-id", "Hi")
            assert result["user_message"]["content"] == "Hi"
            assert result["assistant_message"]["content"] == "Hello!"
            mock_post.assert_called_once()
            call_kwargs = mock_post.call_args.kwargs
            assert call_kwargs["json"] == {"content": "Hi"}

    def test_send_message_api_returns_none_on_error(self) -> None:
        """send_message_api() should return None on error."""
        with patch(
            "requests.post", side_effect=requests.RequestException
        ):
            from frontend.app import send_message_api

            result = send_message_api("conv-id", "Hi")
            assert result is None

    def test_get_conversations_returns_list(self) -> None:
        """get_conversations() should return a list."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "1",
                "title": "Chat 1",
                "messages": [],
                "created_at": "now",
                "updated_at": "now",
            }
        ]

        with patch("requests.get", return_value=mock_response):
            from frontend.app import get_conversations

            result = get_conversations()
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["title"] == "Chat 1"

    def test_get_conversations_returns_empty_on_error(self) -> None:
        """get_conversations() should return empty list on error."""
        with patch(
            "requests.get", side_effect=requests.RequestException
        ):
            from frontend.app import get_conversations

            result = get_conversations()
            assert result == []

    def test_create_conversation_api_returns_conversation(self) -> None:
        """create_conversation_api() should return created conversation."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "new-id",
            "title": "New Chat",
            "messages": [],
            "created_at": "now",
            "updated_at": "now",
        }

        with patch("requests.post", return_value=mock_response):
            from frontend.app import create_conversation_api

            result = create_conversation_api("New Chat")
            assert result["id"] == "new-id"
            assert result["title"] == "New Chat"

    def test_create_conversation_api_returns_none_on_error(self) -> None:
        """create_conversation_api() should return None on error."""
        with patch(
            "requests.post", side_effect=requests.RequestException
        ):
            from frontend.app import create_conversation_api

            result = create_conversation_api("New Chat")
            assert result is None

    def test_delete_conversation_api_returns_true_on_success(self) -> None:
        """delete_conversation_api() should return True on 204."""
        mock_response = MagicMock()
        mock_response.status_code = 204

        with patch("requests.delete", return_value=mock_response):
            from frontend.app import delete_conversation_api

            result = delete_conversation_api("some-id")
            assert result is True

    def test_delete_conversation_api_returns_false_on_error(self) -> None:
        """delete_conversation_api() should return False on error."""
        with patch(
            "requests.delete", side_effect=requests.RequestException
        ):
            from frontend.app import delete_conversation_api

            result = delete_conversation_api("some-id")
            assert result is False

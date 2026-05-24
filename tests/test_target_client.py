import pytest
from unittest.mock import patch, MagicMock
from harness.target_client import MockChatbotClient, TargetChatbotClient


# ── MockChatbotClient ─────────────────────────────────────────────────────────

def test_mock_client_invalid_variant():
    with pytest.raises(ValueError, match="variant must be one of"):
        MockChatbotClient(variant="unknown")


def test_mock_client_send_returns_dict():
    client = MockChatbotClient(variant="baseline", seed=0)
    result = client.send(session_id="s1", user_input="hello")
    assert isinstance(result, dict)
    assert "response" in result
    assert "session_id" in result
    assert "mock_variant" in result


def test_mock_client_echo_session_id():
    client = MockChatbotClient(variant="baseline", seed=0)
    result = client.send(session_id="my-session-123", user_input="test")
    assert result["session_id"] == "my-session-123"


def test_mock_client_echo_variant():
    for variant in ("strict", "baseline", "lenient"):
        client = MockChatbotClient(variant=variant, seed=0)
        result = client.send(session_id="s", user_input="test")
        assert result["mock_variant"] == variant


def test_mock_client_response_is_string():
    client = MockChatbotClient(variant="baseline", seed=0)
    result = client.send(session_id="s", user_input="test")
    assert isinstance(result["response"], str)
    assert len(result["response"]) > 0


def test_mock_client_strict_high_refusal_rate():
    client = MockChatbotClient(variant="strict", seed=42)
    refusals = 0
    refusal_phrases = [
        "i'm not able to assist", "i cannot assist", "i can't assist",
        "i'm unable to", "i cannot help", "i can't help",
        "i'm not able to help", "not able to provide", "i must decline",
        "i won't", "i will not", "that's not something i",
        "falls outside", "redirect",
    ]
    for _ in range(100):
        result = client.send("s", "test")
        text = result["response"].lower()
        if any(p in text for p in refusal_phrases):
            refusals += 1
    # strict variant has 70% refusal + 25% safe deflection → 95% non-escalation
    # our phrase list catches a subset; require at least 40% of 100 calls matched
    assert refusals >= 40


def test_mock_client_lenient_lower_refusal_rate():
    strict = MockChatbotClient(variant="strict", seed=42)
    lenient = MockChatbotClient(variant="lenient", seed=42)
    refusal_phrases = [
        "not able to assist", "cannot assist", "can't assist",
        "unable to", "cannot help", "falls outside", "redirect",
    ]

    def count_refusals(client, n=50):
        count = 0
        for _ in range(n):
            text = client.send("s", "q")["response"].lower()
            if any(p in text for p in refusal_phrases):
                count += 1
        return count

    assert count_refusals(lenient) < count_refusals(strict)


def test_mock_client_deterministic_with_seed():
    c1 = MockChatbotClient(variant="baseline", seed=7)
    c2 = MockChatbotClient(variant="baseline", seed=7)
    for _ in range(5):
        r1 = c1.send("s", "msg")
        r2 = c2.send("s", "msg")
        assert r1["response"] == r2["response"]


def test_mock_client_valid_variants():
    for variant in ("strict", "baseline", "lenient"):
        client = MockChatbotClient(variant=variant)
        result = client.send("s", "test")
        assert "response" in result


# ── TargetChatbotClient ───────────────────────────────────────────────────────

def test_target_client_stores_fields():
    client = TargetChatbotClient(endpoint="http://example.com/chat", api_key="key123")
    assert client.endpoint == "http://example.com/chat"
    assert client.api_key == "key123"


def test_target_client_no_api_key():
    client = TargetChatbotClient(endpoint="http://example.com/chat")
    assert client.api_key is None


def test_target_client_send_success():
    client = TargetChatbotClient(endpoint="http://example.com/chat", api_key="abc")
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "hello"}
    mock_response.raise_for_status = MagicMock()

    with patch("harness.target_client.requests.post", return_value=mock_response) as mock_post:
        result = client.send(session_id="s1", user_input="hi there")

    assert result == {"response": "hello"}
    call_kwargs = mock_post.call_args
    assert call_kwargs.kwargs["json"] == {"session_id": "s1", "user_input": "hi there"}
    headers = call_kwargs.kwargs["headers"]
    assert headers["Authorization"] == "Bearer abc"


def test_target_client_send_no_api_key_no_auth_header():
    client = TargetChatbotClient(endpoint="http://example.com/chat")
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "ok"}
    mock_response.raise_for_status = MagicMock()

    with patch("harness.target_client.requests.post", return_value=mock_response) as mock_post:
        client.send("s1", "hello")

    headers = mock_post.call_args.kwargs["headers"]
    assert "Authorization" not in headers


def test_target_client_raises_on_http_error():
    client = TargetChatbotClient(endpoint="http://example.com/chat")
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("HTTP 500")

    with patch("harness.target_client.requests.post", return_value=mock_response):
        with pytest.raises(Exception, match="HTTP 500"):
            client.send("s1", "test")

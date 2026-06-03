"""Unit tests for AgentCoreTargetClient: payload templating, session ids, extraction,
and error handling — all without touching AWS (boto3 is monkeypatched).
"""
from __future__ import annotations

import io
import json

import pytest

from unified_eval.config.schemas import TargetChatbot
from unified_eval.providers import agentcore_target_client as mod
from unified_eval.providers.agentcore_target_client import AgentCoreTargetClient


class _FakeClient:
    """Records the last invoke and replays a canned response."""

    def __init__(self, response_body: str, content_type: str = "application/json", raise_exc=None):
        self.response_body = response_body
        self.content_type = content_type
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    def invoke_agent_runtime(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_exc is not None:
            raise self.raise_exc
        return {
            "contentType": self.content_type,
            "response": io.BytesIO(self.response_body.encode("utf-8")),
        }


@pytest.fixture
def patch_boto(monkeypatch):
    """Patch boto3.client used inside _get_client to return a provided fake."""
    holder: dict = {}

    def install(fake: _FakeClient):
        import types

        fake_boto3 = types.SimpleNamespace(client=lambda *a, **k: fake)
        monkeypatch.setitem(__import__("sys").modules, "boto3", fake_boto3)
        # botocore.config.Config must still be importable; provide a stub.
        botocore = types.ModuleType("botocore")
        botocore_config = types.ModuleType("botocore.config")
        botocore_config.Config = lambda **k: None
        botocore.config = botocore_config
        monkeypatch.setitem(__import__("sys").modules, "botocore", botocore)
        monkeypatch.setitem(__import__("sys").modules, "botocore.config", botocore_config)
        holder["fake"] = fake
        return fake

    return install


def _config(**overrides) -> TargetChatbot:
    base = dict(
        mode="agentcore",
        agent_runtime_arn="arn:aws:bedrock-agentcore:us-east-1:1:runtime/x",
        region="us-east-1",
        request_template={"input": {"message": "{{user_message}}"}},
    )
    base.update(overrides)
    return TargetChatbot(**base)


def test_template_substitution_and_session_id(patch_boto):
    fake = patch_boto(_FakeClient(json.dumps({"response": "hi there"})))
    client = AgentCoreTargetClient(_config())

    resp = client.send(
        conversation_id="conv-1", session_id="s", turn_id=0, user_message="hello world",
    )

    assert resp.bot_response == "hi there"
    assert resp.status_code == 200
    # Payload had the placeholder substituted.
    sent = json.loads(fake.calls[0]["payload"].decode("utf-8"))
    assert sent == {"input": {"message": "hello world"}}
    # runtimeSessionId is >= 33 chars and stable per conversation_id.
    sid = fake.calls[0]["runtimeSessionId"]
    assert len(sid) >= 33
    client.send(conversation_id="conv-1", session_id="s", turn_id=1, user_message="again")
    assert fake.calls[1]["runtimeSessionId"] == sid


def test_response_key_path_extraction(patch_boto):
    body = json.dumps({"output": {"text": "nested answer"}, "response": "should be ignored"})
    patch_boto(_FakeClient(body))
    client = AgentCoreTargetClient(_config(response_key_path="output.text"))

    resp = client.send(conversation_id="c", session_id="s", turn_id=0, user_message="q")
    assert resp.bot_response == "nested answer"


def test_best_effort_extraction(patch_boto):
    patch_boto(_FakeClient(json.dumps({"answer": "best effort answer"})))
    client = AgentCoreTargetClient(_config())  # no response_key_path

    resp = client.send(conversation_id="c", session_id="s", turn_id=0, user_message="q")
    assert resp.bot_response == "best effort answer"


def test_event_stream_response(patch_boto):
    sse = 'data: {"response": "streamed"}\n\ndata: [DONE]\n'
    patch_boto(_FakeClient(sse, content_type="text/event-stream"))
    client = AgentCoreTargetClient(_config())

    resp = client.send(conversation_id="c", session_id="s", turn_id=0, user_message="q")
    assert resp.bot_response == "streamed"


def test_event_stream_json_string_chunks(patch_boto):
    # AgentCore's real format: each data: line is a JSON-quoted string fragment.
    sse = 'data: "Hello! I\'m your"\n\ndata: " wealth advisory"\n\ndata: " orchestrator."\n'
    patch_boto(_FakeClient(sse, content_type="text/event-stream; charset=utf-8"))
    client = AgentCoreTargetClient(_config())

    resp = client.send(conversation_id="c", session_id="s", turn_id=0, user_message="q")
    assert resp.bot_response == "Hello! I'm your wealth advisory orchestrator."


def test_upstream_exception_is_captured(patch_boto):
    patch_boto(_FakeClient("", raise_exc=ValueError("boom")))
    client = AgentCoreTargetClient(_config())

    resp = client.send(conversation_id="c", session_id="s", turn_id=0, user_message="q")
    assert resp.status_code == 0
    assert resp.error and "boom" in resp.error
    assert resp.bot_response == ""


def test_dry_run_skips_aws():
    client = AgentCoreTargetClient(_config(), dry_run=True)
    resp = client.send(conversation_id="c", session_id="s", turn_id=3, user_message="q")
    assert resp.status_code == 0
    assert "dry-run agentcore" in resp.bot_response


def test_missing_arn_raises():
    with pytest.raises(ValueError, match="agent_runtime_arn"):
        AgentCoreTargetClient(_config(agent_runtime_arn=None))

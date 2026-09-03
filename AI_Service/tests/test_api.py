"""The HTTP surface: POST /interview, GET /health, session handling.

Core_LLM is replaced by a fake client throughout -- nothing here needs the
service to be running.
"""
import json

import pytest
from conftest import FakeLLMClient, reply
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client(monkeypatch):
    fake = FakeLLMClient()
    monkeypatch.setattr(main, "CLIENT", fake)
    monkeypatch.setattr(main, "SESSIONS", main.SessionStore())
    test_client = TestClient(main.app)
    test_client.fake = fake
    return test_client


def test_root_reports_the_versioned_behaviour(client):
    body = client.get("/").json()
    assert body["status"] == "ok"
    assert body["prompt_versions_available"] == ["v1", "v2"]
    assert "eye" in body["domains"]


def test_health_is_degraded_when_core_llm_is_unreachable(client, monkeypatch):
    monkeypatch.setattr(main.CLIENT, "health", lambda: False, raising=False)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["core_llm_reachable"] is False


def test_health_is_ok_when_core_llm_answers(client, monkeypatch):
    monkeypatch.setattr(main.CLIENT, "health", lambda: True, raising=False)
    assert client.get("/health").json()["status"] == "ok"


def test_interview_returns_the_task_spec_shape(client):
    body = client.post("/interview", json={"user_message": "چشمم از دیروز قرمز شده"}).json()
    assert body["domain"] == "eye"
    assert body["model"] == "gemma-4-12b"
    assert body["question"]
    assert body["urgency"] in {"routine", "urgent"}


def test_a_session_id_is_issued_and_state_persists(client):
    first = client.post("/interview", json={"user_message": "چشمم قرمز شده"}).json()
    session_id = first["session_id"]
    assert session_id

    second = client.post("/interview", json={"user_message": "از دیروز ناگهانی",
                                             "session_id": session_id}).json()
    assert second["session_id"] == session_id
    assert second["conversation_state"]["slots"]["redness"] == "yes"
    assert second["conversation_state"]["slots"]["onset"] == "sudden"
    assert second["conversation_state"]["turn_count"] == 2


def test_separate_sessions_do_not_share_state(client):
    one = client.post("/interview", json={"user_message": "چشمم قرمز شده"}).json()
    two = client.post("/interview", json={"user_message": "دیدم تار است"}).json()
    assert one["session_id"] != two["session_id"]
    assert two["conversation_state"]["slots"]["redness"] is None


def test_history_grows_across_turns(client):
    first = client.post("/interview", json={"user_message": "چشمم قرمز شده"}).json()
    client.post("/interview", json={"user_message": "از دیروز",
                                    "session_id": first["session_id"]})
    # Third call must see both prior turns in the prompt it builds.
    contents = [m["content"] for m in client.fake.calls[-1]["messages"]]
    assert "چشمم قرمز شده" in contents


def test_urgency_escalates_across_a_conversation(client):
    session_id = None
    for message in ["چشمم قرمز شده", "از دیشب ناگهانی", "دیدم را از دست داده‌ام"]:
        body = client.post("/interview",
                           json={"user_message": message, "session_id": session_id}).json()
        session_id = body["session_id"]
    assert body["urgency"] == "urgent"
    assert body["complete"] is True
    assert any("escalated" in n for n in body["notes"])


def test_caller_can_pass_state_instead_of_a_session(client):
    body = client.post("/interview", json={
        "user_message": "بله قرمز است",
        "conversation_state": {"domain": "eye", "slots": {"onset": "sudden"},
                               "turn_count": 1},
    }).json()
    assert body["conversation_state"]["slots"]["onset"] == "sudden"
    assert body["conversation_state"]["slots"]["redness"] == "yes"


def test_unknown_prompt_version_is_a_400(client):
    r = client.post("/interview", json={"user_message": "چشمم قرمز شده",
                                        "prompt_version": "v99"})
    assert r.status_code == 400
    assert "v99" in r.json()["detail"]


def test_unknown_policy_version_is_a_400(client):
    r = client.post("/interview", json={"user_message": "چشمم قرمز شده",
                                        "policy_version": "v99"})
    assert r.status_code == 400


def test_rejected_turn_is_still_a_200_with_a_usable_body(client):
    """A caller shouldn't have to branch on error shapes to keep a
    conversation going."""
    body = client.post("/interview", json={"user_message": "   "}).json()
    assert body["question"]
    assert body["notes"] == ["turn rejected: empty_input"]


def test_session_can_be_deleted(client):
    session_id = client.post("/interview",
                             json={"user_message": "چشمم قرمز شده"}).json()["session_id"]
    assert client.delete(f"/interview/{session_id}").json()["status"] == "deleted"
    assert client.delete(f"/interview/{session_id}").json()["status"] == "not_found"


def test_session_store_evicts_the_oldest_past_its_limit():
    store = main.SessionStore(limit=2)
    store.get_or_create("a")
    store.get_or_create("b")
    store.get_or_create("a")   # refreshes "a", so "b" is now the oldest
    store.get_or_create("c")
    assert len(store) == 2
    assert store.drop("b") is False
    assert store.drop("a") is True

import asyncio
import json
import pytest

from dptb_pilot.server import app as server_app


def test_workspace_artifact_links_are_detected_and_sanitized():
    link = "/api/download/" + ("A" * 32) + "/outputs/bands.npy"
    assert server_app._workspace_artifact_link(f"use {link}") == link
    assert "[workspace artifact link omitted]" in server_app._sanitize_workspace_artifact_links(link)
    assert "bands.npy" not in server_app._sanitize_workspace_artifact_links(link)


def test_http_chat_rejects_workspace_artifact_link_before_agent_creation():
    link = "/api/download/" + ("B" * 32) + "/outputs/result.npz"
    request = server_app.ChatMessage(message=f"analyze {link}", session_id="B" * 32)
    with pytest.raises(server_app.HTTPException) as exc_info:
        asyncio.run(server_app.chat_with_agent(request))
    assert exc_info.value.status_code == 422


def test_get_user_sessions_preserves_role_content_history(monkeypatch, tmp_path):
    user_id = "U" * 32
    sessions_file = tmp_path / user_id / "sessions.json"
    sessions_file.parent.mkdir()
    sessions_file.write_text(
        json.dumps([
            {
                "chat_id": "chat-current",
                "user_id": user_id,
                "title": "Current format",
                "history": [
                    {"role": "user", "content": "Question", "timestamp": "2026-08-30T00:00:00Z"},
                    {
                        "role": "assistant",
                        "content": "Answer",
                        "usage_metadata": {"total_tokens": 12},
                    },
                ],
            },
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(server_app, "work_path", str(tmp_path))

    response = asyncio.run(server_app.get_user_sessions(user_id))

    history = response["sessions"][0]["history"]
    assert history == [
        {"role": "user", "content": "Question", "timestamp": "2026-08-30T00:00:00Z"},
        {
            "role": "assistant",
            "content": "Answer",
            "usage_metadata": {"total_tokens": 12},
        },
    ]
    assert response["sessions"][0]["message_count"] == 2


def test_get_user_sessions_upgrades_legacy_pair_history(monkeypatch, tmp_path):
    user_id = "L" * 32
    sessions_file = tmp_path / user_id / "sessions.json"
    sessions_file.parent.mkdir()
    sessions_file.write_text(
        json.dumps([
            {
                "chat_id": "chat-legacy",
                "history": [["Question", "Answer"]],
            },
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(server_app, "work_path", str(tmp_path))

    response = asyncio.run(server_app.get_user_sessions(user_id))

    assert response["sessions"][0]["history"] == [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]
    assert response["sessions"][0]["message_count"] == 2


def test_save_sessions_preserves_server_authored_stage_messages(monkeypatch, tmp_path):
    user_id = "S" * 32
    sessions_file = tmp_path / user_id / "sessions.json"
    sessions_file.parent.mkdir()
    sessions_file.write_text(json.dumps([{
        "chat_id": "chat-runtime",
        "title": "Original",
        "history": [
            {"role": "user", "content": "Run", "message_id": "p1"},
            {"role": "assistant", "content": "Stage result", "message_id": "m1", "run_id": "r1"},
        ],
    }]), encoding="utf-8")
    monkeypatch.setattr(server_app, "work_path", str(tmp_path))

    request = server_app.SaveSessionsRequest(sessions=[{
        "chat_id": "chat-runtime",
        "title": "Renamed",
        "history": [{"role": "user", "content": "Run", "message_id": "p1"}],
    }])
    asyncio.run(server_app.save_user_sessions(user_id, request))

    saved = json.loads(sessions_file.read_text(encoding="utf-8"))[0]
    assert saved["title"] == "Renamed"
    assert [item["message_id"] for item in saved["history"]] == ["p1", "m1"]

import asyncio

from dptb_pilot.server import app as server_app


def test_login_does_not_create_agent(monkeypatch):
    session_id = "L" * 32

    def fail_if_created(_session_id):
        raise AssertionError("login must not create an agent")

    monkeypatch.setattr(server_app, "create_session_agent", fail_if_created)
    server_app.active_agents.pop(session_id, None)

    response = asyncio.run(server_app.login(server_app.LoginRequest(session_id=session_id)))

    assert response["session_id"] == session_id
    assert session_id not in server_app.active_agents


def test_agents_are_lazy_reused_and_isolated_per_chat(monkeypatch):
    session_id = "A" * 32
    created = []

    def create_agent(agent_key):
        agent = {"key": agent_key}
        created.append(agent)
        return agent

    monkeypatch.setattr(server_app, "create_session_agent", create_agent)
    server_app.chat_agents.clear()
    server_app.agent_locks.clear()

    async def exercise():
        first, same, second = await asyncio.gather(
            server_app.get_or_create_chat_agent(session_id, "chat-one"),
            server_app.get_or_create_chat_agent(session_id, "chat-one"),
            server_app.get_or_create_chat_agent(session_id, "chat-two"),
        )
        return first, same, second

    first, same, second = asyncio.run(exercise())

    assert first is same
    assert first is not second
    assert len(created) == 2
    assert server_app.agent_status[f"{session_id}:chat-one"]["status"] == "idle"
    assert server_app.agent_status[f"{session_id}:chat-two"]["status"] == "idle"

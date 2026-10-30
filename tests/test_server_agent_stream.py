import asyncio
from types import SimpleNamespace

import pytest

from dptb_pilot.server import app as server_app


class FakeEvent:
    def __init__(self, text=None, *, final=False):
        self.content = (
            SimpleNamespace(parts=[SimpleNamespace(text=text)])
            if text is not None
            else None
        )
        self._final = final

    def get_function_calls(self):
        return []

    def is_final_response(self):
        return self._final


class FakeFunctionResponseEvent(FakeEvent):
    def __init__(self, name, response, call_id=None):
        super().__init__()
        self.function_response = SimpleNamespace(name=name, response=response, id=call_id)

    def get_function_responses(self):
        return [self.function_response]


class FakeFunctionCallEvent(FakeEvent):
    def __init__(self, name, args, call_id=None):
        super().__init__("")
        self.function_call = SimpleNamespace(name=name, args=args, id=call_id)

    def get_function_calls(self):
        return [self.function_call]


class FakeRunner:
    def __init__(self, events):
        self.events = events
        self.run_kwargs = None

    async def run_async(self, **kwargs):
        self.run_kwargs = kwargs
        for event in self.events:
            yield event


def setup_function():
    server_app.termination_requested.clear()
    server_app.workflow_plan_store.clear()
    server_app.workflow_plan_tasks.clear()
    server_app.workflow_plan_decisions.clear()
    server_app.workflow_plan_events.clear()


def collect_agent_responses(events):
    async def collect():
        return [
            response
            async for response in server_app.call_agent_async(
                "continue",
                FakeRunner(events),
                "user",
                "session",
            )
        ]

    return asyncio.run(collect())


def test_empty_final_event_still_unlocks_the_client_response_state():
    responses = collect_agent_responses([
        FakeEvent("Please confirm whether I should continue."),
        FakeEvent(final=True),
    ])

    assert responses == [
        {
            "type": "streaming_response",
            "content": "Please confirm whether I should continue.",
            "is_final": False,
        },
        {
            "type": "final_response",
            "content": "",
            "is_final": True,
        },
    ]


def test_stream_exhaustion_without_final_event_gets_a_terminal_response():
    responses = collect_agent_responses([
        FakeEvent("Please confirm whether I should continue."),
    ])

    assert responses[-1] == {
        "type": "final_response",
        "content": "",
        "is_final": True,
    }


def test_clear_workflow_state_removes_all_review_state():
    server_app.workflow_plan_store["session"] = {"title": "plan"}
    server_app.workflow_plan_tasks["session"] = "task"
    server_app.workflow_plan_decisions["session"] = "approved"
    server_app.workflow_plan_events["session"] = asyncio.Event()

    server_app.clear_workflow_state("session")

    assert "session" not in server_app.workflow_plan_store
    assert "session" not in server_app.workflow_plan_tasks
    assert "session" not in server_app.workflow_plan_decisions
    assert "session" not in server_app.workflow_plan_events


def test_normalize_conversation_history_accepts_vscode_and_legacy_formats():
    history = server_app.normalize_conversation_history([
        {"role": "user", "content": "Use silicon", "timestamp": "ignored"},
        {"role": "assistant", "content": "Okay"},
        ["What lattice?", "Diamond"],
        {"role": "assistant", "content": ""},
    ])

    assert history == [
        {"role": "user", "content": "Use silicon"},
        {"role": "assistant", "content": "Okay"},
        {"role": "user", "content": "What lattice?"},
        {"role": "assistant", "content": "Diamond"},
    ]


def test_normalize_persisted_history_keeps_vscode_metadata():
    history = server_app.normalize_persisted_history([
        {"role": "user", "content": "Use silicon", "timestamp": "2026-08-06"},
        ["What lattice?", "Diamond"],
    ])

    assert history == [
        {"role": "user", "content": "Use silicon", "timestamp": "2026-08-06"},
        {"role": "user", "content": "What lattice?"},
        {"role": "assistant", "content": "Diamond"},
    ]


def test_conversation_query_contains_current_chat_history_and_latest_message():
    query = server_app.build_conversation_query(
        "Continue with that material",
        [{"role": "user", "content": "Use silicon"}],
    )

    assert '"content": "Use silicon"' in query
    assert "<latest_user_message>Continue with that material</latest_user_message>" in query


def test_agent_sessions_are_isolated_by_chat_id():
    first = server_app.agent_session_id("A" * 32, "chat-one")
    second = server_app.agent_session_id("A" * 32, "chat-two")

    assert first != second
    assert first == server_app.agent_session_id("A" * 32, "chat-one")


def test_call_agent_uses_the_conversation_specific_agent_session():
    runner = FakeRunner([FakeEvent(final=True)])

    async def collect():
        return [
            response
            async for response in server_app.call_agent_async(
                "continue",
                runner,
                "user",
                "pilot-session",
                agent_session_id="chat-session",
            )
        ]

    asyncio.run(collect())
    assert runner.run_kwargs["session_id"] == "chat-session"


def test_remote_task_metadata_extracts_job_and_bohrium_ids():
    metadata = server_app._extract_remote_task_info({
        "job_id": "2026-08-31-1/abc123",
        "extra_info": {"bohr_job_id": "bohr-789", "bohr_group_id": "group-1"},
    })

    assert metadata["remote_task_id"] == "2026-08-31-1/abc123"
    assert metadata["bohrium_task_id"] == "bohr-789"
    assert server_app._extract_remote_task_info({"output": "Succeeded"})["remote_task_status"] == "Succeeded"


def test_remote_execution_mode_detects_bohrium_and_slurm_configs():
    assert server_app._remote_execution_mode({
        "executor": {"machine": {"context_type": "Bohrium"}},
    }) == "Bohrium"
    assert server_app._remote_execution_mode({
        "executor": {"machine": {"context_type": "SSHContext"}},
    }) == "Slurm"


def test_tool_responses_emit_a_completed_batch_stage():
    responses = collect_agent_responses([
        FakeFunctionResponseEvent("list_workspace_files", {"files": ["a.json"]}),
        FakeEvent("Done", final=True),
    ])

    assert responses[0]["type"] == "tool_batch_completed"
    assert responses[0]["tools"] == ["list_workspace_files"]
    assert responses[0]["content"].splitlines()[0] == "Tool batch completed: list_workspace_files"
    assert responses[0]["tool_results"][0]["result"] == {"files": ["a.json"]}


def test_tool_batch_pairs_redacted_inputs_with_the_matching_response():
    responses = collect_agent_responses([
        FakeFunctionCallEvent(
            "get_job_results",
            {
                "input_file": "INPUT",
                "password": "secret",
                "executor": {"machine": {"context_type": "SSHContext"}},
            },
            call_id="call-1",
        ),
        FakeFunctionResponseEvent(
            "get_job_results",
            {"job_id": "job-1", "status": "Succeeded"},
            call_id="call-1",
        ),
        FakeEvent("Done", final=True),
    ])

    result = responses[0]["tool_results"][0]
    assert result["input"]["input_file"] == "INPUT"
    assert result["input"]["password"] == "***"
    assert result["execution_mode"] == "Slurm"


def test_each_local_tool_call_publishes_its_current_name(monkeypatch):
    published = []

    async def capture_message(_session_id, message):
        if message.get("type") == "agent_status":
            published.append(message)

    monkeypatch.setattr(server_app.manager, "send_message", capture_message)
    collect_agent_responses([
        FakeFunctionCallEvent("first_local_tool", {}),
        FakeFunctionCallEvent("second_local_tool", {}),
        FakeEvent("Done", final=True),
    ])

    tool_calls = [item for item in published if item.get("last_event") == "tool_call"]
    assert [item["tool_name"] for item in tool_calls] == [
        "first_local_tool",
        "second_local_tool",
    ]


def test_remote_tool_logs_link_to_workspace_files_and_include_a_preview(monkeypatch, tmp_path):
    session_id = "session"
    log_path = tmp_path / session_id / "files" / "remote" / "run.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("remote output\ncompleted\n", encoding="utf-8")
    monkeypatch.setattr(server_app, "work_path", str(tmp_path))

    stage = server_app._tool_batch_stage(
        [SimpleNamespace(
            name="get_job_results",
            response={"job_id": "job-1", "status": "Succeeded", "logs": [{"path": str(log_path)}]},
            id="call-1",
        )],
        pending_tool_calls=[{
            "id": "call-1",
            "tool": "get_job_results",
            "input": {"job_id": "job-1"},
            "execution_mode": None,
        }],
        session_id=session_id,
    )

    log = stage["tool_results"][0]["logs"][0]
    assert log["workspace_path"] == "remote/run.log"
    assert "completed" in log["content"]


def test_workspace_listing_is_not_treated_as_artifact_output():
    responses = collect_agent_responses([
        FakeFunctionResponseEvent(
            "list_workspace_files",
            {"files": ["old-result.csv", "previous.json"]},
        ),
        FakeEvent("I inspected old-result.csv.", final=True),
    ])

    final_response = responses[-1]
    assert "artifact_text" not in final_response
    assert "old-result.csv" in final_response["ignored_artifact_text"]


@pytest.mark.parametrize("tool_name", [
    "list_workspace_files",
    "read_file_content",
    "get_file_path",
    "view_file",
    "preview_workspace_file",
])
def test_file_discovery_tools_are_artifact_read_only(tool_name):
    assert server_app._is_artifact_read_only_tool(tool_name) is True


@pytest.mark.parametrize("tool_name", [
    "download_artifact",
    "download_mp_structure",
    "run_abacus",
    "plot_band",
])
def test_artifact_producing_tools_are_not_filtered(tool_name):
    assert server_app._is_artifact_read_only_tool(tool_name) is False


def test_agent_cannot_finish_while_a_required_remote_job_is_running():
    with pytest.raises(RuntimeError, match="still running"):
        collect_agent_responses([
            FakeFunctionResponseEvent("submit_run_abacus", {"job_id": "job-1", "status": "Running"}),
            FakeEvent("Submitted", final=True),
        ])

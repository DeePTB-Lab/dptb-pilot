from dptb_pilot.server import app as server_app


def test_workflow_mode_overrides_automatic_planner(monkeypatch):
    monkeypatch.setattr(server_app, "should_plan_workflow", lambda _query: False)

    assert server_app.should_generate_workflow("simple task", "always") is True
    assert server_app.should_generate_workflow("simple task", "auto") is False
    assert server_app.should_generate_workflow("simple task", "disabled") is False


def test_workflow_mode_auto_keeps_existing_heuristic(monkeypatch):
    monkeypatch.setattr(server_app, "should_plan_workflow", lambda _query: True)

    assert server_app.should_generate_workflow("simple task", "auto") is True
    assert server_app.normalize_workflow_mode("unknown") == "auto"
    assert server_app.normalize_workflow_mode(None) == "auto"


def test_structured_workflow_tasks_are_preserved_and_validated():
    plan = server_app._normalize_workflow_plan({
        "title": "Parallel",
        "tasks": [
            {"id": "a", "tool": "band_predict", "parallel_safe": True},
            {"id": "b", "tool": "dos_predict", "parallel_safe": True},
        ],
        "execution": {"max_parallelism": 2},
    }, "run both")

    assert [task["tool"] for task in plan["tasks"]] == ["band_predict", "dos_predict"]
    assert server_app.plan_is_executable(plan)

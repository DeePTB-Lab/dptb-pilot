import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("inherited_model", [None, "openai/shell-model"])
def test_startup_model_uses_cwd_env_before_server_imports(tmp_path, inherited_model):
    """A source-side dotenv must not win over the startup directory's file."""
    startup = tmp_path / "startup"
    startup.mkdir()
    (startup / ".env").write_text(
        "LLM_MODEL=openai/new-model\nLLM_API_KEY=test-key\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env.pop("LLM_MODEL", None)
    env.pop("PYTHON_DOTENV_DISABLED", None)
    if inherited_model is not None:
        env["LLM_MODEL"] = inherited_model
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", """
import os
import dotenv

# Simulate a dependency that implicitly discovers a stale .env at import time.
load_dotenv = dotenv.load_dotenv
def load_stale_dotenv(dotenv_path=None, *args, **kwargs):
    if dotenv_path is None:
        os.environ.setdefault('LLM_MODEL', 'openai/old-model')
        return True
    return load_dotenv(dotenv_path, *args, **kwargs)
dotenv.load_dotenv = load_stale_dotenv

before = os.getenv('LLM_MODEL')
import dptb_pilot.main as entry
assert os.getenv('LLM_MODEL') == before
captured = {}
entry.react_launch = lambda **kwargs: captured.update(kwargs)
entry.main()
expected = before or 'openai/new-model'
assert captured['model_config']['model'] == expected, captured['model_config']['model']
assert os.getenv('LLM_MODEL') == expected
"""],
        cwd=startup,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

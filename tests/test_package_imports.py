import subprocess
import sys


def test_package_import_does_not_eagerly_load_agent_stack():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import dptb_pilot; "
                "assert 'dptb_pilot.main' not in sys.modules; "
                "assert 'google.adk' not in sys.modules; "
                "assert callable(dptb_pilot.react_launch); "
                "assert callable(dptb_pilot.launch)"
            ),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

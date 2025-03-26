"""Global tests."""

import subprocess
import sys


def test_main_script() -> None:
    """Test main.py."""
    result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
    assert (
        result.returncode == 0
    ), f"main.py a échoué avec le code de retour {result.returncode}\n{result.stdout}\n{result.stderr}"

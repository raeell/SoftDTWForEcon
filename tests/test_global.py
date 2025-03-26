import subprocess
import sys


def test_main_script():
    result = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)
    assert (
        result.returncode == 0
    ), f"main.py a échoué avec le code de retour {result.returncode}\n{result.stdout}\n{result.stderr}"

"""
Local runner script.
Starts FastAPI server and opens browser.
"""
import subprocess
import sys
import webbrowser
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    host = "127.0.0.1"
    port = 8000

    print("=" * 60)
    print("  Meal Recommender - Hệ thống gợi ý thực đơn")
    print(f"  Starting at http://{host}:{port}")
    print("=" * 60)

    print("\n[1/3] Running food preprocessing...")
    preprocess_script = PROJECT_ROOT / "ml" / "pipelines" / "preprocess_foods.py"
    if preprocess_script.exists():
        subprocess.run(
            [sys.executable, str(preprocess_script)],
            cwd=str(PROJECT_ROOT),
        )

    print("\n[2/3] Opening browser...")
    webbrowser.open(f"http://{host}:{port}")

    print("\n[3/3] Starting FastAPI server...")
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "app:app",
            "--host", host,
            "--port", str(port),
            "--reload",
        ],
        cwd=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()

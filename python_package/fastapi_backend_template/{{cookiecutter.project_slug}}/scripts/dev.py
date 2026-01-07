"""Development script helper."""
import os
import sys
import subprocess


def main() -> None:
    """Run development server."""
    os.environ.setdefault("APP_ENV", "development")
    os.environ.setdefault("DEBUG", "true")
    
    cmd = [
        "uvicorn",
        "app.main:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ]
    
    print("Starting development server...")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()

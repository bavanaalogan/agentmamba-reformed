"""Launch the Streamlit dashboard:  python scripts/launch_dashboard.py [--port 8501] [--profile NAME]

--profile NAME uses the separate memory storage/profiles/NAME (created empty if new) instead of the
application memory storage/app, e.g. a clean test profile for the stage-1/stage-2 protocol."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="8501")
    ap.add_argument("--profile", default="", help="memory profile under storage/profiles/ (default: storage/app)")
    a = ap.parse_args()
    env = dict(os.environ)
    if a.profile:
        env["AGENTMAMBA_STORAGE"] = str(ROOT / "storage" / "profiles" / a.profile)
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
                              "--server.port", a.port, "--server.headless", "true",
                              # see .streamlit/config.toml: the watcher triggers torchvision imports
                              "--server.fileWatcherType", "none"], cwd=ROOT, env=env))

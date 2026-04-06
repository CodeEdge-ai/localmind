#!/usr/bin/env python3
"""
LocalMind — Main Orchestrator  v2.1

GitHub integration now uses the REST API directly (no git binary required).
Works on Railway, Render, Fly.io, and any containerised platform.

Changes from v2.0:
- Removed gitpython dependency entirely
- GitHub commits via requests + GitHub Contents/Trees API
- No git executable needed on the host system
"""

import base64
import json
import os
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

from chat import app as flask_app, ensure_data_dir
from agent import LocalMindAgent

load_dotenv()

DATA_DIR = Path("data")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO  = os.getenv("GITHUB_REPO")   # format: "username/reponame"
AGENT_INTERVAL_HOURS = float(os.getenv("AGENT_INTERVAL_HOURS", "24"))
MAX_RETRIES = 3

GH_API = "https://api.github.com"
GH_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


# ─────────────────────────────────────────────────────────────────────────────
# GITHUB REST API — no git binary needed
# ─────────────────────────────────────────────────────────────────────────────

def _gh_headers():
    """Return auth headers. Fails loudly if token missing."""
    if not GITHUB_TOKEN:
        raise RuntimeError("GITHUB_TOKEN not set — cannot push to GitHub")
    return {**GH_HEADERS, "Authorization": f"Bearer {GITHUB_TOKEN}"}


def _get_default_branch():
    """Fetch the repo's default branch name (usually 'main')."""
    url = f"{GH_API}/repos/{GITHUB_REPO}"
    r = requests.get(url, headers=_gh_headers(), timeout=15)
    r.raise_for_status()
    return r.json().get("default_branch", "main")


def _get_file_sha(path_in_repo):
    """Return the current blob SHA for a file, or None if it doesn't exist yet."""
    url = f"{GH_API}/repos/{GITHUB_REPO}/contents/{path_in_repo}"
    r = requests.get(url, headers=_gh_headers(), timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("sha")


def _push_file(path_in_repo, local_path, commit_message):
    """
    Create or update a single file in the GitHub repo.
    Uses the Contents API — no git binary needed.
    """
    content = local_path.read_bytes()
    encoded = base64.b64encode(content).decode()

    sha = _get_file_sha(path_in_repo)

    payload = {
        "message": commit_message,
        "content": encoded,
        "committer": {
            "name": "LocalMind Agent",
            "email": "localmind@agent.local",
        },
    }
    if sha:
        payload["sha"] = sha  # required for updates

    url = f"{GH_API}/repos/{GITHUB_REPO}/contents/{path_in_repo}"
    r = requests.put(url, headers=_gh_headers(),
                     data=json.dumps(payload), timeout=20)
    r.raise_for_status()
    return r.json()


def push_data_to_github(commit_message):
    """
    Push all files in data/ to GitHub using the REST API.
    No git installation required.
    """
    if not GITHUB_TOKEN or not GITHUB_REPO:
        print("[MAIN] GitHub not configured — skipping push.")
        return False

    data_files = [f for f in DATA_DIR.glob("*") if f.is_file()]
    if not data_files:
        print("[MAIN] No data files to push.")
        return False

    pushed = 0
    for fp in data_files:
        path_in_repo = f"data/{fp.name}"
        try:
            _push_file(path_in_repo, fp, commit_message)
            print(f"[MAIN] ✓ Pushed {path_in_repo}")
            pushed += 1
        except Exception as e:
            print(f"[MAIN] ✗ Failed to push {path_in_repo}: {e}")

    print(f"[MAIN] GitHub sync complete — {pushed}/{len(data_files)} files pushed.")
    return pushed > 0


# ─────────────────────────────────────────────────────────────────────────────
# AGENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_cycle():
    """Run the agent and push results to GitHub. Retries on failure."""
    print(f"\n{'='*60}")
    print(f"[MAIN] Agent cycle starting — {datetime.now()}")
    print(f"{'='*60}")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            agent = LocalMindAgent()
            agent.run()

            commit_msg = (
                f"LocalMind analysis — {datetime.now().strftime('%Y-%m-%d')} — "
                f"{len(agent.logs)} entries — "
                f"score {agent.agent_state.get('learning_score', 0)}/100"
            )
            push_data_to_github(commit_msg)
            print(f"[MAIN] Cycle #{attempt} succeeded.\n")
            return True

        except Exception as e:
            print(f"[MAIN] Cycle attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                wait = 30 * attempt
                print(f"[MAIN] Retrying in {wait}s...")
                time.sleep(wait)

    print(f"[MAIN] All {MAX_RETRIES} attempts failed. Will retry next scheduled run.\n")
    return False


def agent_scheduler():
    print(f"[MAIN] Scheduler started — running every {AGENT_INTERVAL_HOURS}h")

    run_agent_cycle()  # run immediately on startup

    while True:
        next_run = datetime.now() + timedelta(hours=AGENT_INTERVAL_HOURS)
        print(f"[MAIN] Next run: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(AGENT_INTERVAL_HOURS * 3600)
        run_agent_cycle()


# ─────────────────────────────────────────────────────────────────────────────
# FLASK SERVER
# ─────────────────────────────────────────────────────────────────────────────

def start_chat_server():
    ensure_data_dir()
    port = int(os.getenv("PORT", 5000))
    print(f"[MAIN] Chat server → http://0.0.0.0:{port}")
    flask_app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("🧠  LocalMind — Self-Improving AI Agent  v2.1")
    print("=" * 60)
    print(f"   Interval    : every {AGENT_INTERVAL_HOURS}h")
    print(f"   GitHub repo : {GITHUB_REPO or '✗ not set'}")
    print(f"   GitHub token: {'✓ set' if GITHUB_TOKEN else '✗ not set'}")
    print(f"   Anthropic   : {'✓ set' if os.getenv('ANTHROPIC_API_KEY') else '✗ not set'}")
    print(f"   Git binary  : not required (using GitHub REST API)")
    print()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[MAIN] ⚠  ANTHROPIC_API_KEY missing — NLP features disabled")
    if not GITHUB_TOKEN or not GITHUB_REPO:
        print("[MAIN] ⚠  GitHub not configured — data won't be versioned")

    scheduler_thread = threading.Thread(
        target=agent_scheduler,
        daemon=True,
        name="AgentScheduler",
    )
    scheduler_thread.start()
    print("[MAIN] Agent scheduler thread started.")

    start_chat_server()


if __name__ == "__main__":
    main()

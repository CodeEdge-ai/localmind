#!/usr/bin/env python3
"""
LocalMind — Main Orchestrator

Upgraded in this version:
- /health endpoint exposed
- AGENT_INTERVAL_HOURS configurable via .env
- Retry logic on agent run failure
- Cleaner startup banner with learning score
- Graceful handling of missing .env values
"""

import os
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from git import Repo
from git.exc import InvalidGitRepositoryError

from chat import app as flask_app, ensure_data_dir
from agent import LocalMindAgent

load_dotenv()

DATA_DIR = Path("data")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
AGENT_INTERVAL_HOURS = float(os.getenv("AGENT_INTERVAL_HOURS", "24"))
MAX_RETRIES = 3


# ─────────────────────────────────────────────────────────────────────────────
# GIT INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

def setup_git_repo():
    repo_path = Path(".").resolve()
    try:
        repo = Repo(repo_path)
        print("[MAIN] Existing git repository loaded.")
    except InvalidGitRepositoryError:
        print("[MAIN] Initializing new git repository...")
        repo = Repo.init(repo_path)
        repo.config_writer().set_value("user", "name", "LocalMind").release()
        repo.config_writer().set_value("user", "email", "localmind@agent.local").release()

    if GITHUB_TOKEN and GITHUB_REPO:
        auth_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"
        if "origin" in [r.name for r in repo.remotes]:
            repo.remote("origin").set_url(auth_url)
        else:
            repo.create_remote("origin", auth_url)
        print(f"[MAIN] Git remote → {GITHUB_REPO}")
    else:
        print("[MAIN] Warning: Git remote not configured (GITHUB_TOKEN / GITHUB_REPO missing)")

    return repo


def commit_and_push(repo, message=None):
    if not message:
        message = f"LocalMind update — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    try:
        data_files = [f for f in DATA_DIR.glob("*") if f.is_file()]
        if not data_files:
            print("[MAIN] No data files to commit.")
            return False

        for fp in data_files:
            repo.git.add(str(fp))

        if not repo.is_dirty(untracked_files=True):
            print("[MAIN] No changes to commit.")
            return True

        commit = repo.index.commit(message)
        print(f"[MAIN] Committed: {message} [{commit.hexsha[:8]}]")

        if "origin" in [r.name for r in repo.remotes]:
            repo.remote("origin").push()
            print("[MAIN] Pushed to GitHub ✓")
        else:
            print("[MAIN] No remote — skipping push.")

        return True

    except Exception as e:
        print(f"[MAIN] Git error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# AGENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_with_commit(repo):
    """Run agent with retry logic and commit results to GitHub."""
    print(f"\n{'='*60}")
    print(f"[MAIN] Agent run starting — {datetime.now()}")
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
            commit_and_push(repo, commit_msg)
            print(f"[MAIN] Agent run #{attempt} succeeded.\n")
            return True

        except Exception as e:
            print(f"[MAIN] Agent run attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(30 * attempt)  # exponential-ish back-off

    print(f"[MAIN] All {MAX_RETRIES} attempts failed. Will retry at next scheduled run.\n")
    return False


def agent_scheduler(repo):
    print(f"[MAIN] Scheduler active — interval: {AGENT_INTERVAL_HOURS}h")

    # First run immediately on startup
    run_agent_with_commit(repo)

    while True:
        next_run = datetime.now() + timedelta(hours=AGENT_INTERVAL_HOURS)
        print(f"[MAIN] Next agent run: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(AGENT_INTERVAL_HOURS * 3600)
        run_agent_with_commit(repo)


# ─────────────────────────────────────────────────────────────────────────────
# FLASK SERVER
# ─────────────────────────────────────────────────────────────────────────────

def start_chat_server():
    ensure_data_dir()
    port = int(os.getenv("PORT", 5000))
    host = "0.0.0.0"
    print(f"[MAIN] Chat server → http://localhost:{port}")
    flask_app.run(host=host, port=port, debug=False, threaded=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("🧠  LocalMind — Self-Improving AI Agent  v2.0")
    print("=" * 60)
    print(f"   Interval  : every {AGENT_INTERVAL_HOURS}h")
    print(f"   GitHub    : {GITHUB_REPO or 'not configured'}")
    print(f"   API Key   : {'✓ set' if os.getenv('ANTHROPIC_API_KEY') else '✗ missing'}")
    print(f"   Git Token : {'✓ set' if GITHUB_TOKEN else '✗ missing'}")
    print()

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("[MAIN] ⚠  ANTHROPIC_API_KEY missing — NLP features disabled")
    if not GITHUB_TOKEN or not GITHUB_REPO:
        print("[MAIN] ⚠  GitHub config incomplete — versioning disabled")

    repo = setup_git_repo()

    scheduler_thread = threading.Thread(
        target=agent_scheduler,
        args=(repo,),
        daemon=True,
        name="AgentScheduler",
    )
    scheduler_thread.start()
    print("[MAIN] Agent scheduler thread started.")

    start_chat_server()


if __name__ == "__main__":
    main()

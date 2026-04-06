#!/usr/bin/env python3
"""
LocalMind Chatbot — Flask Web Interface with Anthropic NLP

Upgraded in this version:
- Model: claude-haiku-4-5-20251001
- Bug fixes: date parsing crash, None-safe operations, better error handling
- New endpoints:
    DELETE /api/logs/<int:index>        — remove a specific log entry
    GET    /api/export                  — download logs as CSV
    GET    /api/agent-state             — fetch agent's curiosity/predictions/goals
    POST   /api/feedback                — answer one of the agent's curiosity questions
    GET    /api/stats                   — quick summary stats
    GET    /health                      — health check
"""

import csv
import io
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, Response

load_dotenv()

app = Flask(__name__)

# ── Models ────────────────────────────────────────────────────────────────────
CHAT_MODEL = "claude-haiku-4-5-20251001"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
LOGS_PATH = DATA_DIR / "logs.json"
AGENT_STATE_PATH = DATA_DIR / "agent_state.json"
INSIGHTS_PATH = DATA_DIR / "insights.json"
FEEDBACK_PATH = DATA_DIR / "feedback.json"

# ── Anthropic client (lazy init) ──────────────────────────────────────────────
_anthropic_client = None


def get_client():
    global _anthropic_client
    if _anthropic_client is None:
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _anthropic_client = Anthropic(api_key=key)
    return _anthropic_client


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not LOGS_PATH.exists():
        _write_json(LOGS_PATH, [])


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _load_logs():
    ensure_data_dir()
    try:
        logs = _read_json(LOGS_PATH)
        return [e for e in logs if isinstance(e, dict)]
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _save_logs(logs):
    _write_json(LOGS_PATH, logs)


def parse_natural_date(date_str):
    """Convert natural language date to YYYY-MM-DD. Never raises."""
    today = datetime.now()
    if not date_str:
        return today.strftime("%Y-%m-%d")
    ds = str(date_str).lower().strip()
    if ds in ("today", ""):
        return today.strftime("%Y-%m-%d")
    if ds == "yesterday":
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    if ds == "tomorrow":
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    # Try ISO format
    try:
        datetime.strptime(ds, "%Y-%m-%d")
        return ds
    except ValueError:
        pass
    return today.strftime("%Y-%m-%d")


def _safe_date(date_str, fallback=None):
    """Parse a log entry's date safely — returns datetime or fallback."""
    try:
        return datetime.strptime(str(date_str), "%Y-%m-%d")
    except (ValueError, TypeError):
        return fallback or datetime(2000, 1, 1)


# ── NLP: extract structured data ──────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a helpful assistant that extracts structured activity data from natural language messages.

Given a user message about an activity, extract:
- date: YYYY-MM-DD format, or "today"/"yesterday"
- time_start: HH:MM (24-hour)
- time_end: HH:MM (24-hour). If duration is given (e.g. "for 2 hours"), calculate end time from start.
- category: single lowercase word (coding, fitness, reading, meeting, learning, writing, cooking, gaming, socializing, meditation, etc.)
- description: brief description

Respond ONLY with valid JSON, no markdown:
{"date":"...","time_start":"...","time_end":"...","category":"...","description":"..."}

If the message is a QUERY (question about past activities), respond:
{"type":"query","query":"the user's question"}

If the message is a FEEDBACK ANSWER to an agent curiosity question, respond:
{"type":"feedback","answer":"the user's answer"}"""


def extract_with_anthropic(message):
    try:
        client = get_client()
        response = client.messages.create(
            model=CHAT_MODEL,
            max_tokens=350,
            system=EXTRACTION_SYSTEM,
            messages=[{"role": "user", "content": message}],
        )
        raw = response.content[0].text.strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception as e:
        print(f"[CHAT] Extraction error: {e}")
    return None


# ── Log entry processing ──────────────────────────────────────────────────────

def process_log_entry(extracted):
    """Validate, enrich, and save a new log entry. Returns confirmation string."""
    date = parse_natural_date(extracted.get("date", "today"))
    time_start = (extracted.get("time_start") or "09:00").strip()
    time_end = (extracted.get("time_end") or "").strip()
    description = extracted.get("description", "").strip()
    category = (
        extracted.get("category", "general")
        .lower()
        .replace(" ", "_")
        .strip()
    )

    # If no end time, try to infer from description duration
    if time_start and not time_end:
        dur_match = re.search(
            r"(\d+(?:\.\d+)?)\s*hour", description, re.IGNORECASE
        )
        if dur_match:
            try:
                hours = float(dur_match.group(1))
                start_dt = datetime.strptime(f"{date} {time_start}", "%Y-%m-%d %H:%M")
                end_dt = start_dt + timedelta(hours=hours)
                time_end = end_dt.strftime("%H:%M")
            except ValueError:
                pass
        if not time_end:
            # Default: 1 hour session
            try:
                start_dt = datetime.strptime(f"{date} {time_start}", "%Y-%m-%d %H:%M")
                time_end = (start_dt + timedelta(hours=1)).strftime("%H:%M")
            except ValueError:
                time_end = time_start

    entry = {
        "date": date,
        "time_start": time_start,
        "time_end": time_end,
        "category": category,
        "description": description,
    }

    logs = _load_logs()
    logs.append(entry)
    _save_logs(logs)

    # Build confirmation
    duration_str = ""
    try:
        sm = int(time_start.split(":")[0]) * 60 + int(time_start.split(":")[1])
        em = int(time_end.split(":")[0]) * 60 + int(time_end.split(":")[1])
        if em < sm:
            em += 1440
        mins = em - sm
        hrs = mins / 60
        duration_str = f" ({hrs:.1f}h)"
    except Exception:
        pass

    return (
        f"✅ Logged: **{category}** on {date} "
        f"from {time_start} to {time_end}{duration_str}\n"
        f"📝 {description}"
    )


# ── Query answering ───────────────────────────────────────────────────────────

def answer_query(query):
    """Answer a natural-language question about logged activities."""
    logs = _load_logs()
    if not logs:
        return (
            "I don't have any activity logs yet. "
            "Start logging activities and I'll be able to answer questions!"
        )

    q_lower = query.lower()
    today_str = datetime.now().strftime("%Y-%m-%d")
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Filter by time window
    if "today" in q_lower:
        relevant = [e for e in logs if e.get("date") == today_str]
        ctx = "today"
    elif "yesterday" in q_lower:
        relevant = [e for e in logs if e.get("date") == yesterday_str]
        ctx = "yesterday"
    elif "this week" in q_lower or "last 7" in q_lower:
        cutoff = datetime.now() - timedelta(days=7)
        relevant = [e for e in logs if _safe_date(e.get("date")) >= cutoff]
        ctx = "this week"
    elif "this month" in q_lower:
        cutoff = datetime.now() - timedelta(days=30)
        relevant = [e for e in logs if _safe_date(e.get("date")) >= cutoff]
        ctx = "this month"
    else:
        relevant = logs
        ctx = "all time"

    # Further filter by category if mentioned
    all_cats = {e.get("category", "") for e in logs}
    for cat in all_cats:
        if cat and cat in q_lower:
            relevant = [e for e in relevant if e.get("category") == cat]
            break

    if not relevant:
        return f"No logged activities found for {ctx}."

    logs_text = json.dumps(relevant[-30:], indent=2)  # Cap at 30 entries for context

    system = (
        f'You are LocalMind, a helpful AI activity tracker assistant.\n'
        f'Answer this question about the user\'s activities: "{query}"\n\n'
        f"Activity logs ({ctx}):\n{logs_text}\n\n"
        f"Be concise, friendly, and insightful. Summarize key patterns if relevant."
    )

    try:
        client = get_client()
        response = client.messages.create(
            model=CHAT_MODEL,
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": "Answer based on the logs."}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"[CHAT] Query error: {e}")
        # Graceful fallback
        lines = [f"Here are your activities for {ctx}:"]
        for e in relevant[-5:]:
            lines.append(
                f"• {e.get('date')}: {e.get('category')} "
                f"({e.get('time_start','?')}–{e.get('time_end','?')}) — {e.get('description','')}"
            )
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True)
    if not data or not data.get("message", "").strip():
        return jsonify({"error": "No message provided"}), 400

    message = data["message"].strip()
    print(f"[CHAT] → {message}")

    extracted = extract_with_anthropic(message)

    if not extracted:
        return jsonify({
            "response": (
                "I couldn't understand that. Try: "
                "\"I coded for 2 hours\" or \"What did I do today?\""
            )
        })

    msg_type = extracted.get("type")

    if msg_type == "query":
        return jsonify({"response": answer_query(message)})

    if msg_type == "feedback":
        # Save user's answer to a curiosity question
        _save_feedback(message, extracted.get("answer", message))
        return jsonify({
            "response": (
                "💡 Thanks for telling me! I'll factor that into my next analysis. "
                "The more you share, the smarter I get."
            )
        })

    # Default: log entry
    confirmation = process_log_entry(extracted)
    return jsonify({"response": confirmation})


@app.route("/api/logs", methods=["GET"])
def get_logs():
    try:
        logs = _load_logs()
        return jsonify({"logs": logs, "count": len(logs)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs/<int:index>", methods=["DELETE"])
def delete_log(index):
    """Delete a log entry by its 0-based index."""
    logs = _load_logs()
    if index < 0 or index >= len(logs):
        return jsonify({"error": f"Index {index} out of range"}), 404
    removed = logs.pop(index)
    _save_logs(logs)
    return jsonify({"deleted": removed, "remaining": len(logs)})


@app.route("/api/export", methods=["GET"])
def export_logs():
    """Export all logs as a downloadable CSV."""
    logs = _load_logs()
    if not logs:
        return jsonify({"error": "No logs to export"}), 404

    output = io.StringIO()
    fields = ["date", "time_start", "time_end", "category", "description"]
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(logs)

    csv_data = output.getvalue()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={
            "Content-Disposition": (
                f"attachment; filename=localmind_logs_{datetime.now().strftime('%Y%m%d')}.csv"
            )
        },
    )


@app.route("/api/insights", methods=["GET"])
def get_insights():
    try:
        if INSIGHTS_PATH.exists():
            all_insights = _read_json(INSIGHTS_PATH)
            return jsonify({"insights": all_insights[-1] if all_insights else None})
        return jsonify({"insights": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/agent-state", methods=["GET"])
def get_agent_state():
    """Return the agent's current curiosity, predictions, goals, and learning score."""
    try:
        if AGENT_STATE_PATH.exists():
            state = _read_json(AGENT_STATE_PATH)
            return jsonify({"agent_state": state})
        return jsonify({"agent_state": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/feedback", methods=["POST"])
def post_feedback():
    """Accept a user's answer to an agent curiosity question."""
    data = request.get_json(silent=True)
    if not data or not data.get("answer"):
        return jsonify({"error": "No answer provided"}), 400
    question = data.get("question", "")
    answer = data.get("answer", "")
    _save_feedback(question, answer)
    return jsonify({"message": "Feedback saved! The agent will use this in its next run."})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Quick summary statistics."""
    logs = _load_logs()
    if not logs:
        return jsonify({"total_entries": 0, "categories": [], "total_hours": 0})

    from collections import defaultdict
    cat_counts = defaultdict(int)
    cat_mins = defaultdict(int)
    dates = set()

    for e in logs:
        cat = e.get("category", "unknown")
        cat_counts[cat] += 1
        dates.add(e.get("date", ""))
        try:
            ts = e.get("time_start", "00:00").split(":")
            te = e.get("time_end", "00:00").split(":")
            sm = int(ts[0]) * 60 + int(ts[1])
            em = int(te[0]) * 60 + int(te[1])
            if em < sm:
                em += 1440
            cat_mins[cat] += max(0, em - sm)
        except Exception:
            pass

    total_mins = sum(cat_mins.values())
    return jsonify({
        "total_entries": len(logs),
        "unique_days": len(dates),
        "total_hours": round(total_mins / 60, 1),
        "categories": dict(cat_counts),
        "top_category": max(cat_counts, key=cat_counts.get) if cat_counts else None,
    })


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _save_feedback(question, answer):
    ensure_data_dir()
    try:
        fb = _read_json(FEEDBACK_PATH) if FEEDBACK_PATH.exists() else []
    except json.JSONDecodeError:
        fb = []
    fb.append({
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
    })
    _write_json(FEEDBACK_PATH, fb)


def start_chat_server(host="0.0.0.0", port=5000, debug=False):
    ensure_data_dir()
    print(f"[CHAT] Server starting on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    start_chat_server(debug=True)

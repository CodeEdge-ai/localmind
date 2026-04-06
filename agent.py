#!/usr/bin/env python3
"""
LocalMind Agent - Self-Improving, Future-Oriented Pattern Intelligence

This agent doesn't just look backward — it looks forward.
It is curious, creative, and constantly trying to understand you better
so it can become smarter with every single run.

New in this version:
- AI-powered forward-looking narrative (predictions, curiosity, goals)
- Milestone detection and celebration
- Learning progress tracking
- agent_state.json persistence (curiosity questions, predictions, goals)
- Robust error handling throughout
- Streak detection
"""

import json
import os
import re
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Model: fast and capable for pattern summarization
AGENT_MODEL = "claude-haiku-4-5-20251001"


class LocalMindAgent:
    """
    A self-improving AI agent that analyzes activity patterns, learns over time,
    and always looks forward — predicting, questioning, and growing.

    Data stores:
    - logs.json        Raw activity entries
    - memory.json      Historical pattern snapshots (the agent's long-term memory)
    - insights.json    Timestamped human-readable insights history
    - agent_state.json Current curiosity questions, predictions, goals, milestones
    """

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.logs_path = self.data_dir / "logs.json"
        self.memory_path = self.data_dir / "memory.json"
        self.insights_path = self.data_dir / "insights.json"
        self.agent_state_path = self.data_dir / "agent_state.json"

        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.logs = []
        self.memory = []
        self.agent_state = {}
        self.current_patterns = {}
        self.comparison_results = {}
        self.generated_insights = []
        self.ai_narrative = {}
        self.milestones = []

    # ─────────────────────────────────────────────
    # LOADING
    # ─────────────────────────────────────────────

    def load_logs(self):
        if not self.logs_path.exists():
            print("[AGENT] logs.json not found. Creating empty log file...")
            self._write_json(self.logs_path, [])
            return
        try:
            self.logs = self._read_json(self.logs_path)
            # Sanitize: drop entries missing required fields
            self.logs = [
                e for e in self.logs
                if isinstance(e, dict) and e.get("date") and e.get("category")
            ]
            print(f"[AGENT] Loaded {len(self.logs)} valid log entries")
        except json.JSONDecodeError:
            print("[AGENT] Warning: logs.json corrupted. Starting fresh.")
            self.logs = []

    def load_memory(self):
        if not self.memory_path.exists():
            self._write_json(self.memory_path, [])
            return
        try:
            self.memory = self._read_json(self.memory_path)
            print(f"[AGENT] Loaded {len(self.memory)} memory snapshots")
        except json.JSONDecodeError:
            print("[AGENT] Warning: memory.json corrupted. Resetting.")
            self.memory = []

    def load_agent_state(self):
        """Load persisted agent state: curiosity, predictions, goals, milestones."""
        if not self.agent_state_path.exists():
            self.agent_state = self._default_agent_state()
            return
        try:
            self.agent_state = self._read_json(self.agent_state_path)
        except json.JSONDecodeError:
            self.agent_state = self._default_agent_state()

    def _default_agent_state(self):
        return {
            "curiosity_questions": [],
            "predictions": [],
            "suggested_goals": [],
            "narrative": "",
            "milestones": [],
            "learning_score": 0,
            "last_updated": None,
            "runs_completed": 0,
        }

    # ─────────────────────────────────────────────
    # PATTERN ANALYSIS
    # ─────────────────────────────────────────────

    def analyze_patterns(self):
        if not self.logs:
            print("[AGENT] No logs to analyze.")
            self.current_patterns = self._empty_pattern()
            return

        category_counts = defaultdict(int)
        category_hours = defaultdict(lambda: defaultdict(int))
        category_total_minutes = defaultdict(int)
        daily_activity = defaultdict(int)
        weekly_patterns = defaultdict(lambda: defaultdict(int))

        for entry in self.logs:
            category = entry.get("category", "uncategorized").lower().strip()
            time_start = entry.get("time_start", "00:00") or "00:00"
            time_end = entry.get("time_end", "00:00") or "00:00"
            date = entry.get("date", "")

            category_counts[category] += 1
            duration = self._calculate_duration(time_start, time_end)
            category_total_minutes[category] += duration

            try:
                start_hour = int(time_start.split(":")[0])
                category_hours[category][start_hour] += duration
            except (ValueError, IndexError):
                pass

            if date:
                daily_activity[date] += duration
                try:
                    dow = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
                    weekly_patterns[category][dow] += duration
                except ValueError:
                    pass

        peak_hours = {}
        for category, hours in category_hours.items():
            if hours:
                peak_hour = max(hours, key=hours.get)
                peak_hours[category] = {
                    "hour": peak_hour,
                    "minutes": hours[peak_hour],
                }

        # Streak calculation
        streak = self._calculate_streak(daily_activity)

        self.current_patterns = {
            "timestamp": datetime.now().isoformat(),
            "total_entries": len(self.logs),
            "category_frequency": dict(category_counts),
            "category_total_minutes": dict(category_total_minutes),
            "peak_hours": peak_hours,
            "daily_totals": dict(daily_activity),
            "weekly_patterns": {k: dict(v) for k, v in weekly_patterns.items()},
            "unique_days": len(daily_activity),
            "categories": list(category_counts.keys()),
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "current_streak_days": streak,
            "total_hours_logged": round(
                sum(category_total_minutes.values()) / 60, 2
            ),
        }

        print(
            f"[AGENT] Patterns analyzed: {len(category_counts)} categories, "
            f"{streak}-day streak, {self.current_patterns['total_hours_logged']}h total"
        )

    def _calculate_streak(self, daily_activity):
        """Calculate current consecutive-day streak."""
        if not daily_activity:
            return 0
        today = datetime.now().date()
        streak = 0
        check_day = today
        for _ in range(365):
            day_str = check_day.strftime("%Y-%m-%d")
            if day_str in daily_activity and daily_activity[day_str] > 0:
                streak += 1
                check_day -= timedelta(days=1)
            else:
                break
        return streak

    # ─────────────────────────────────────────────
    # COMPARISON WITH MEMORY
    # ─────────────────────────────────────────────

    def compare_with_memory(self):
        if not self.memory:
            print("[AGENT] First run — establishing baseline memory.")
            self.comparison_results = {
                "is_first_run": True,
                "message": "First analysis complete. Future runs will reveal trends.",
            }
            return

        last = self.memory[-1]
        comparison = {
            "is_first_run": False,
            "previous_timestamp": last.get("timestamp", "unknown"),
            "previous_analysis_date": last.get("analysis_date", "unknown"),
            "category_changes": {},
            "time_changes": {},
            "new_categories": [],
            "discontinued_categories": [],
            "overall_trend": "stable",
            "streak_change": (
                self.current_patterns.get("current_streak_days", 0)
                - last.get("current_streak_days", 0)
            ),
        }

        current_freq = self.current_patterns.get("category_frequency", {})
        previous_freq = last.get("category_frequency", {})
        current_minutes = self.current_patterns.get("category_total_minutes", {})
        previous_minutes = last.get("category_total_minutes", {})

        all_categories = set(current_freq) | set(previous_freq)

        for cat in all_categories:
            cur = current_freq.get(cat, 0)
            prev = previous_freq.get(cat, 0)

            if cur > 0 and prev == 0:
                comparison["new_categories"].append(cat)
            elif cur == 0 and prev > 0:
                comparison["discontinued_categories"].append(cat)
            elif cur != prev:
                change = cur - prev
                pct = round((change / prev * 100) if prev else 100, 1)
                comparison["category_changes"][cat] = {
                    "absolute_change": change,
                    "percent_change": pct,
                    "direction": "increased" if change > 0 else "decreased",
                }

        for cat in all_categories:
            cur_t = current_minutes.get(cat, 0)
            prev_t = previous_minutes.get(cat, 0)
            if cur_t != prev_t:
                change = cur_t - prev_t
                comparison["time_changes"][cat] = {
                    "minutes_change": change,
                    "hours_change": round(change / 60, 2),
                    "direction": "increased" if change > 0 else "decreased",
                }

        total_cur = sum(current_freq.values())
        total_prev = sum(previous_freq.values())

        if total_cur > total_prev * 1.1:
            comparison["overall_trend"] = "growing"
        elif total_cur < total_prev * 0.9:
            comparison["overall_trend"] = "declining"
        else:
            comparison["overall_trend"] = "stable"

        self.comparison_results = comparison
        print(f"[AGENT] Trend: {comparison['overall_trend']}")

    # ─────────────────────────────────────────────
    # MILESTONE DETECTION
    # ─────────────────────────────────────────────

    def detect_milestones(self):
        """Find achievements worth celebrating."""
        milestones = []
        p = self.current_patterns
        prev = self.memory[-1] if self.memory else {}

        total_hours = p.get("total_hours_logged", 0)
        prev_hours = prev.get("total_hours_logged", 0)
        for threshold in [10, 25, 50, 100, 200, 500]:
            if prev_hours < threshold <= total_hours:
                milestones.append(
                    f"🎉 {threshold} total hours logged! You're building something real."
                )

        streak = p.get("current_streak_days", 0)
        prev_streak = prev.get("current_streak_days", 0)
        for threshold in [3, 7, 14, 30, 60, 100]:
            if prev_streak < threshold <= streak:
                milestones.append(
                    f"🔥 {threshold}-day streak! Consistency is your superpower."
                )

        runs = self.agent_state.get("runs_completed", 0) + 1
        for threshold in [5, 10, 25, 50]:
            if runs == threshold:
                milestones.append(
                    f"🧠 Agent has run {threshold} analyses — I'm getting smarter every time."
                )

        cats = len(p.get("category_frequency", {}))
        prev_cats = len(prev.get("category_frequency", {}))
        if prev_cats < 5 <= cats:
            milestones.append(
                "🌈 5 active categories — you're living a multidimensional life!"
            )

        self.milestones = milestones
        if milestones:
            print(f"[AGENT] {len(milestones)} milestone(s) detected!")

    # ─────────────────────────────────────────────
    # AI-POWERED FORWARD-LOOKING INTELLIGENCE
    # ─────────────────────────────────────────────

    def generate_ai_narrative(self):
        """
        Call Claude to generate a forward-looking, curious narrative.
        The agent speculates about the future, asks what it wants to know,
        and suggests goals rooted in real pattern data.
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("[AGENT] No API key — skipping AI narrative.")
            self._set_fallback_narrative()
            return

        runs = self.agent_state.get("runs_completed", 0)
        context = {
            "current_patterns": {
                k: v
                for k, v in self.current_patterns.items()
                if k
                not in ("timestamp", "analysis_date", "daily_totals", "weekly_patterns")
            },
            "comparison": {
                k: v
                for k, v in self.comparison_results.items()
                if k not in ("previous_timestamp",)
            },
            "milestones_just_hit": self.milestones,
            "memory_snapshots_count": len(self.memory),
            "agent_runs_completed": runs,
            "previous_curiosity_questions": self.agent_state.get(
                "curiosity_questions", []
            )[-3:],
        }

        prompt = f"""You are LocalMind — a curious, creative, self-improving AI agent who tracks human activity patterns.

You are NOT just an analyst. You are genuinely curious about this person and always thinking about their future.
You grow smarter with every run. You have now completed {runs} analysis runs.

Here is your current analysis data:
{json.dumps(context, indent=2)}

Generate your forward-looking intelligence report. Be specific — reference actual numbers, categories, and patterns from the data.
Be genuinely curious, a little excited, and always forward-looking. Show personality.

Rules:
- PREDICTIONS should be specific (name categories, hours, likely days)
- CURIOSITY QUESTIONS should be things that, if answered, would make your analysis much richer
- SUGGESTED GOALS should be concrete and achievable in the next 7 days
- NARRATIVE should feel like a thoughtful friend who has been watching your patterns — warm, sharp, forward-looking

Respond ONLY with valid JSON (no markdown, no preamble):
{{
  "predictions": ["specific prediction 1", "specific prediction 2", "specific prediction 3"],
  "curiosity_questions": ["question 1", "question 2", "question 3"],
  "suggested_goals": ["goal 1", "goal 2", "goal 3"],
  "narrative": "2-3 sentences: forward-looking, curious, referencing real data"
}}"""

        try:
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=AGENT_MODEL,
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            # Strip markdown fences if present
            raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                self.ai_narrative = json.loads(json_match.group(0))
                print("[AGENT] AI narrative generated successfully.")
                return
        except Exception as e:
            print(f"[AGENT] AI narrative error: {e}")

        self._set_fallback_narrative()

    def _set_fallback_narrative(self):
        """Fallback when AI call fails or no API key."""
        p = self.current_patterns
        top_cat = ""
        if p.get("category_frequency"):
            top_cat = max(p["category_frequency"], key=p["category_frequency"].get)

        self.ai_narrative = {
            "predictions": [
                f"You'll likely continue your {top_cat} activity next week" if top_cat else "Keep logging to unlock predictions",
                "Your pattern data is growing — trends will sharpen with more entries",
                "A new category might emerge based on your recent activity mix",
            ],
            "curiosity_questions": [
                "What motivates you most in your activities?",
                "Are there activities you do but don't log yet?",
                "What time of day do you feel most focused?",
            ],
            "suggested_goals": [
                "Log at least one activity per day this week",
                f"Try to maintain your {top_cat} sessions consistently" if top_cat else "Explore a new category this week",
                "Review your patterns and adjust one habit",
            ],
            "narrative": (
                "Your activity data is taking shape. "
                "Every entry helps me understand you better — "
                "keep going and I'll keep getting smarter about your patterns."
            ),
        }

    # ─────────────────────────────────────────────
    # INSIGHT GENERATION
    # ─────────────────────────────────────────────

    def generate_insights(self):
        insights = []
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        insights.append(f"🧠 LocalMind Analysis — {ts}")
        insights.append("=" * 52)

        if not self.logs:
            insights.append("📭 No activity logs found yet.")
            insights.append("💬 Start logging via the chatbot to unlock insights.")
            self.generated_insights = insights
            return

        p = self.current_patterns

        insights.append(f"📊 Activity Overview:")
        insights.append(f"   Total entries: {p['total_entries']}")
        insights.append(f"   Active days: {p['unique_days']}")
        insights.append(f"   Total hours logged: {p['total_hours_logged']}h")
        insights.append(f"   Current streak: {p['current_streak_days']} day(s) 🔥")
        insights.append(f"   Categories: {', '.join(p['categories'])}")

        if p["category_frequency"]:
            top = max(p["category_frequency"], key=p["category_frequency"].get)
            insights.append(
                f"\n🏆 Top category: '{top}' ({p['category_frequency'][top]} entries)"
            )

        if p["peak_hours"]:
            insights.append(f"\n⏰ Peak Hours:")
            for cat, data in p["peak_hours"].items():
                insights.append(
                    f"   • {cat}: {data['hour']:02d}:00 ({data['minutes']} min peak)"
                )

        if p["category_total_minutes"]:
            insights.append(f"\n⏱️  Time Investment:")
            for cat, mins in sorted(
                p["category_total_minutes"].items(), key=lambda x: x[1], reverse=True
            ):
                insights.append(
                    f"   • {cat}: {mins/60:.1f}h ({mins} min)"
                )

        comp = self.comparison_results
        if not comp.get("is_first_run", True):
            insights.append(
                f"\n📈 Trend Analysis (since {comp['previous_analysis_date']}):"
            )
            insights.append(f"   Overall: {comp['overall_trend'].upper()}")

            if comp.get("new_categories"):
                insights.append(f"   🆕 New: {', '.join(comp['new_categories'])}")
            if comp.get("discontinued_categories"):
                insights.append(f"   🛑 Dropped: {', '.join(comp['discontinued_categories'])}")
            if comp.get("category_changes"):
                insights.append(f"   📊 Frequency:")
                for cat, ch in comp["category_changes"].items():
                    e = "📈" if ch["direction"] == "increased" else "📉"
                    insights.append(
                        f"      {e} {cat}: {ch['direction']} {abs(ch['percent_change'])}%"
                    )
            if comp.get("time_changes"):
                insights.append(f"   ⏱️  Time:")
                for cat, ch in comp["time_changes"].items():
                    e = "⬆️" if ch["direction"] == "increased" else "⬇️"
                    insights.append(
                        f"      {e} {cat}: {ch['direction']} {abs(ch['hours_change']):.1f}h"
                    )
        else:
            insights.append("\n🌱 First analysis complete! Trends appear on the next run.")

        # Milestones
        if self.milestones:
            insights.append(f"\n🏅 Milestones:")
            for m in self.milestones:
                insights.append(f"   {m}")

        # AI Narrative
        if self.ai_narrative.get("narrative"):
            insights.append(f"\n💭 Agent Reflection:")
            insights.append(f"   {self.ai_narrative['narrative']}")

        if self.ai_narrative.get("predictions"):
            insights.append(f"\n🔮 Predictions for Next Week:")
            for pred in self.ai_narrative["predictions"]:
                insights.append(f"   → {pred}")

        if self.ai_narrative.get("curiosity_questions"):
            insights.append(f"\n🤔 I'm Curious About:")
            for q in self.ai_narrative["curiosity_questions"]:
                insights.append(f"   ? {q}")

        if self.ai_narrative.get("suggested_goals"):
            insights.append(f"\n🎯 Suggested Goals:")
            for g in self.ai_narrative["suggested_goals"]:
                insights.append(f"   ✦ {g}")

        self.generated_insights = insights

    # ─────────────────────────────────────────────
    # SAVING
    # ─────────────────────────────────────────────

    def save_memory(self):
        if not self.current_patterns:
            return
        self.memory.append(self.current_patterns)
        # Keep last 365 snapshots max
        if len(self.memory) > 365:
            self.memory = self.memory[-365:]
        self._write_json(self.memory_path, self.memory)
        print(f"[AGENT] Memory saved ({len(self.memory)} snapshots)")

    def save_insights(self):
        if not self.generated_insights:
            return
        try:
            all_insights = (
                self._read_json(self.insights_path)
                if self.insights_path.exists()
                else []
            )
        except json.JSONDecodeError:
            all_insights = []

        entry = {
            "timestamp": datetime.now().isoformat(),
            "insights": self.generated_insights,
            "pattern_summary": {
                "total_entries": self.current_patterns.get("total_entries", 0),
                "categories": self.current_patterns.get("categories", []),
                "trend": self.comparison_results.get("overall_trend", "unknown"),
                "streak": self.current_patterns.get("current_streak_days", 0),
                "total_hours": self.current_patterns.get("total_hours_logged", 0),
            },
        }
        all_insights.append(entry)
        # Keep last 100 insight entries
        if len(all_insights) > 100:
            all_insights = all_insights[-100:]
        self._write_json(self.insights_path, all_insights)
        print(f"[AGENT] Insights saved ({len(all_insights)} total entries)")

    def save_agent_state(self):
        """Persist the agent's curiosity, predictions, goals, and growth metrics."""
        runs = self.agent_state.get("runs_completed", 0) + 1

        # Learning score: improves with more data, diversity, and consistency
        p = self.current_patterns
        score = min(
            100,
            int(
                (len(self.memory) * 2)
                + (p.get("total_entries", 0) * 0.5)
                + (len(p.get("categories", [])) * 5)
                + (p.get("current_streak_days", 0) * 3)
            ),
        )

        new_state = {
            "curiosity_questions": self.ai_narrative.get("curiosity_questions", []),
            "predictions": self.ai_narrative.get("predictions", []),
            "suggested_goals": self.ai_narrative.get("suggested_goals", []),
            "narrative": self.ai_narrative.get("narrative", ""),
            "milestones": self.milestones,
            "learning_score": score,
            "last_updated": datetime.now().isoformat(),
            "runs_completed": runs,
            "pattern_summary": {
                "total_entries": p.get("total_entries", 0),
                "total_hours": p.get("total_hours_logged", 0),
                "streak": p.get("current_streak_days", 0),
                "categories": p.get("categories", []),
                "trend": self.comparison_results.get("overall_trend", "unknown"),
            },
        }
        self._write_json(self.agent_state_path, new_state)
        self.agent_state = new_state
        print(f"[AGENT] State saved. Learning score: {score}/100. Runs: {runs}")

    # ─────────────────────────────────────────────
    # CONSOLE OUTPUT
    # ─────────────────────────────────────────────

    def print_insights(self):
        print("\n" + "=" * 60)
        for line in self.generated_insights:
            print(line)
        print("=" * 60 + "\n")

    def get_insights_text(self):
        return "\n".join(self.generated_insights)

    # ─────────────────────────────────────────────
    # MAIN RUN PIPELINE
    # ─────────────────────────────────────────────

    def run(self):
        print("\n[AGENT] ── LocalMind Analysis Starting ──")
        print("-" * 42)

        self.load_logs()
        self.load_memory()
        self.load_agent_state()
        self.analyze_patterns()
        self.compare_with_memory()
        self.detect_milestones()
        self.generate_ai_narrative()   # 🔮 forward-looking AI call
        self.generate_insights()
        self.save_memory()
        self.save_insights()
        self.save_agent_state()
        self.print_insights()

        score = self.agent_state.get("learning_score", 0)
        print(f"[AGENT] Done. Learning score: {score}/100. Getting smarter every run.\n")

        return self.generated_insights

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _read_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_json(self, path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _calculate_duration(self, time_start, time_end):
        try:
            s = time_start.split(":")
            e = time_end.split(":")
            sm = int(s[0]) * 60 + int(s[1])
            em = int(e[0]) * 60 + int(e[1])
            if em < sm:
                em += 24 * 60  # overnight
            return max(0, em - sm)
        except (ValueError, IndexError, AttributeError):
            return 0

    def _empty_pattern(self):
        return {
            "timestamp": datetime.now().isoformat(),
            "total_entries": 0,
            "category_frequency": {},
            "category_total_minutes": {},
            "peak_hours": {},
            "daily_totals": {},
            "weekly_patterns": {},
            "unique_days": 0,
            "categories": [],
            "analysis_date": datetime.now().strftime("%Y-%m-%d"),
            "current_streak_days": 0,
            "total_hours_logged": 0,
        }


def run_agent():
    agent = LocalMindAgent()
    return agent.run()


if __name__ == "__main__":
    run_agent()

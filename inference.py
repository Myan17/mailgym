"""
Inference Script — MailGym (Email Triage OpenEnv)
==================================================
MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    ENV_BASE_URL   The MailGym server URL (default: http://localhost:7860).

This script uses the OpenAI Client for all LLM calls.
It runs a baseline agent across 3 tasks and emits the required stdout
log format: [START] / [STEP] / [END].

STDOUT FORMAT (strict)
----------------------
    [START] task=<task_name> env=mailgym model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Rules:
    - One [START] per task, one [END] per task (always emitted, even on error).
    - One [STEP] per env.step() call.
    - reward / rewards formatted to 2 decimals, score to 3 decimals.
    - done/success are lowercase booleans.
    - error is the raw error string or the unquoted word null.
    - action is single-line (no embedded newlines).
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"

# Default to the live HF Space so `python inference.py` works with zero setup.
# Override with ENV_BASE_URL=http://localhost:7860 for local development.
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "https://myan9417-mailgym.hf.space"
BENCHMARK = "mailgym"

TEMPERATURE = 0.0          # Deterministic for reproducibility
MAX_TOKENS = 500
SEED = 42                  # Fixed seed for reproducible email selection
SUCCESS_THRESHOLD = 0.5    # A task "succeeds" when score >= threshold

TASKS = ["classify_easy", "triage_medium", "full_triage_hard"]

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert email triage assistant. You will be given an email and
    a task description. Respond with ONLY a valid JSON object matching the
    required fields. Do not include any explanation or text outside the JSON.

    Example responses:
    - Easy task: {"category": "urgent"}
    - Medium task: {"category": "urgent", "priority": "P0", "department": "engineering"}
    - Hard task: {"category": "urgent", "priority": "P0", "department": "engineering", "response_draft": "Hi team, I'm investigating this issue immediately..."}

    Categories: spam, urgent, routine, newsletter, personal
    Priorities: P0 (critical), P1 (high), P2 (medium), P3 (low)
    Departments: engineering, sales, support, hr, legal, marketing, executive
""")


# ── Logging (STRICT format — do not edit casually) ───────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Single-line safety: collapse any newlines/carriage returns in the action string.
    safe_action = action.replace("\n", " ").replace("\r", " ")
    error_val = error.replace("\n", " ").replace("\r", " ") if error else "null"
    done_val = "true" if done else "false"
    print(
        f"[STEP] step={step} action={safe_action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = "true" if success else "false"
    print(
        f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def build_user_prompt(observation: dict) -> str:
    """Build the user prompt from an observation dict."""
    email = observation.get("email") or {}
    task = observation.get("task") or {}

    return textwrap.dedent(f"""\
        EMAIL:
        From: {email.get('sender', 'unknown')}
        Subject: {email.get('subject', 'no subject')}
        Body:
        {email.get('body', '(empty)')}
        Has attachments: {email.get('has_attachments', False)}
        Thread length: {email.get('thread_length', 1)}

        TASK: {task.get('task_name', 'unknown')}
        Difficulty: {task.get('difficulty', 'unknown')}
        Instructions: {task.get('instructions', 'none')}
        Required fields: {json.dumps(task.get('required_fields', []))}

        Respond with ONLY a JSON object containing the required fields.
    """)


def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM response as JSON, handling markdown fences and prose."""
    text = (response_text or "").strip()

    # Strip markdown code fences.
    if text.startswith("```"):
        lines = [line for line in text.split("\n") if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Direct JSON parse.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract first {...} object from the text.
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    return {}


def action_to_str(action: dict) -> str:
    """Compact single-line string form of an action, for [STEP] logging."""
    return json.dumps(action, separators=(",", ":"), ensure_ascii=False)


def call_llm(llm: OpenAI, observation: dict) -> dict:
    """Call the LLM once and return a parsed action dict (possibly empty on failure)."""
    user_prompt = build_user_prompt(observation)
    try:
        completion = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response_text = completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        return {}

    return parse_llm_response(response_text)


# ── Task runner ──────────────────────────────────────────────────────────────

def run_task(llm: OpenAI, env: httpx.Client, task_name: str) -> None:
    """
    Run a single task end-to-end.

    Emits exactly one [START], one [STEP] per env.step(), and one [END],
    even if any stage raises.
    """
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        # 1. Reset.
        reset_resp = env.post("/reset", json={"task_name": task_name, "seed": SEED})
        reset_resp.raise_for_status()
        reset_data = reset_resp.json()
        observation = reset_data["observation"]

        # 2. Ask the LLM for an action.
        action_dict = call_llm(llm, observation)

        # 3. Submit the action. Single-step episodes → steps_taken always 1.
        step_num = 1
        steps_taken = step_num
        error: Optional[str] = None
        reward = 0.0
        done = False

        try:
            step_resp = env.post("/step", json={"action": action_dict})
            step_resp.raise_for_status()
            step_data = step_resp.json()
            reward = float(step_data.get("reward") or 0.0)
            done = bool(step_data.get("done", False))
        except httpx.HTTPStatusError as http_err:
            error = f"HTTP {http_err.response.status_code}: {http_err.response.text[:200]}"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        rewards.append(reward)
        log_step(
            step=step_num,
            action=action_to_str(action_dict),
            reward=reward,
            done=done,
            error=error,
        )

        # 4. Compute final score. Single step → score == reward, already in [0, 1].
        score = max(0.0, min(1.0, reward))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] run_task({task_name}) failed: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    with httpx.Client(base_url=ENV_BASE_URL, timeout=60.0) as env:
        # Sanity-ping the environment before running tasks.
        try:
            health = env.get("/")
            health.raise_for_status()
        except Exception as exc:
            print(
                f"ERROR: Cannot reach environment at {ENV_BASE_URL}: {exc}",
                file=sys.stderr,
            )
            sys.exit(2)

        for task_name in TASKS:
            run_task(llm, env, task_name)


if __name__ == "__main__":
    main()

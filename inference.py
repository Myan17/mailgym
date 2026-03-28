"""
Inference Script — Email Triage OpenEnv Environment
====================================================
MANDATORY ENVIRONMENT VARIABLES:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

This script uses the OpenAI Client for all LLM calls.
It runs a baseline agent against all 3 tasks and prints reproducible scores.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap

from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

TEMPERATURE = 0.0  # Deterministic for reproducibility
MAX_TOKENS = 500
SEED = 42  # Fixed seed for reproducible email selection

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert email triage assistant. You will be given an email and
    a task description. You must respond with ONLY a valid JSON object matching
    the required fields. Do not include any explanation or text outside the JSON.

    Example responses:
    - Easy task: {"category": "urgent"}
    - Medium task: {"category": "urgent", "priority": "P0", "department": "engineering"}
    - Hard task: {"category": "urgent", "priority": "P0", "department": "engineering", "response_draft": "Hi team, I'm investigating this issue immediately..."}

    Categories: spam, urgent, routine, newsletter, personal
    Priorities: P0 (critical), P1 (high), P2 (medium), P3 (low)
    Departments: engineering, sales, support, hr, legal, marketing, executive
""")


def build_user_prompt(observation: dict) -> str:
    """Build the user prompt from an observation."""
    email = observation.get("email", {})
    task = observation.get("task", {})

    prompt = textwrap.dedent(f"""\
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
    return prompt


def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM response as JSON, handling common formatting issues."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (code fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try parsing as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Fallback: return empty dict
    print(f"  WARNING: Could not parse LLM response as JSON: {text[:200]}")
    return {}


# ── Main ─────────────────────────────────────────────────────────────────────

def run_task(client: OpenAI, env_client, task_name: str, seed: int) -> float:
    """Run a single task and return the score."""
    import httpx

    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"{'='*60}")

    # Reset environment
    reset_resp = env_client.post("/reset", json={"task_name": task_name, "seed": seed})
    reset_resp.raise_for_status()
    reset_data = reset_resp.json()

    observation = reset_data["observation"]
    print(f"  Email subject: {observation['email']['subject'][:60]}...")
    print(f"  Task: {observation['task']['difficulty']} — {observation['task']['task_name']}")

    # Build prompt and call LLM
    user_prompt = build_user_prompt(observation)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response_text = completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"  LLM call failed: {exc}")
        response_text = '{"category": "routine"}'

    print(f"  LLM response: {response_text[:200]}")

    # Parse response into action
    action_dict = parse_llm_response(response_text)
    print(f"  Parsed action: {json.dumps(action_dict)}")

    # Submit action to environment
    step_resp = env_client.post("/step", json={"action": action_dict})
    step_resp.raise_for_status()
    step_data = step_resp.json()

    reward = step_data.get("reward", 0.0) or 0.0
    feedback = step_data["observation"].get("feedback", "")
    print(f"  Reward: {reward}")
    print(f"  Feedback: {feedback}")

    return reward


def main() -> None:
    """Run the baseline agent against all 3 tasks."""
    if not API_KEY:
        print("ERROR: HF_TOKEN or API_KEY environment variable not set.")
        sys.exit(1)
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable not set.")
        sys.exit(1)

    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print(f"Environment: {ENV_BASE_URL}")

    # Initialize clients
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    import httpx
    env_client = httpx.Client(base_url=ENV_BASE_URL, timeout=30.0)

    # Verify environment is running
    try:
        health = env_client.get("/")
        health.raise_for_status()
        print(f"Environment health: {health.json()}")
    except Exception as exc:
        print(f"ERROR: Cannot reach environment at {ENV_BASE_URL}: {exc}")
        print("Start the server with: python -m server.app")
        sys.exit(1)

    tasks = ["classify_easy", "triage_medium", "full_triage_hard"]
    scores = {}

    for task_name in tasks:
        score = run_task(llm_client, env_client, task_name, seed=SEED)
        scores[task_name] = score

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE RESULTS")
    print(f"{'='*60}")
    for task, score in scores.items():
        print(f"  {task:25s} → {score:.2f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'Average':25s} → {avg:.2f}")
    print(f"{'='*60}")

    env_client.close()


if __name__ == "__main__":
    main()

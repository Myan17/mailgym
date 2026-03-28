"""
FastAPI server exposing the Email Triage environment via HTTP.

Endpoints (OpenEnv-compliant):
  POST /reset   — start a new episode
  POST /step    — submit an action
  GET  /state   — get current episode metadata
  GET  /tasks   — list available tasks
  GET  /        — health check
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path so we can import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from models import Observation, State, TriageAction
from server.environment import EmailTriageEnvironment, VALID_TASK_NAMES

# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="MailGym",
    description=(
        "An OpenEnv-compliant RL environment that simulates email triage. "
        "Agents classify, prioritize, route, and draft responses to emails."
    ),
    version="1.0.0",
)

# Single environment instance (one episode at a time)
env = EmailTriageEnvironment()


# ── Request / Response Schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = Field(
        "classify_easy",
        description="Task to run: classify_easy, triage_medium, full_triage_hard",
    )
    seed: Optional[int] = Field(
        None,
        description="Optional seed for reproducible email selection",
    )


class StepRequest(BaseModel):
    action: TriageAction = Field(..., description="The agent's triage action")


class ResetResponse(BaseModel):
    observation: Observation
    state: State


class StepResponse(BaseModel):
    observation: Observation
    reward: Optional[float] = None
    done: bool = False
    state: State


class TaskListResponse(BaseModel):
    tasks: list[str]
    descriptions: dict[str, str]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "environment": "mailgym",
        "version": "1.0.0",
    }


@app.get("/tasks", response_model=TaskListResponse)
async def list_tasks():
    """List all available tasks."""
    from server.environment import TASK_DEFINITIONS

    descriptions = {
        name: td.instructions[:100] + "..."
        for name, td in TASK_DEFINITIONS.items()
    }
    return TaskListResponse(tasks=VALID_TASK_NAMES, descriptions=descriptions)


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest):
    """
    Reset the environment and start a new episode.

    Returns the initial observation with the email to triage.
    """
    observation = env.reset(
        task_name=request.task_name,
        seed=request.seed,
    )

    if observation.error_message:
        raise HTTPException(status_code=400, detail=observation.error_message)

    return ResetResponse(
        observation=observation,
        state=env.state(),
    )


@app.post("/step", response_model=StepResponse)
async def step(request: StepRequest):
    """
    Submit a triage action and receive the graded result.

    Returns observation with reward, feedback, and done status.
    """
    state = env.state()
    if state.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is not active. Call POST /reset first.",
        )

    observation = env.step(request.action)

    return StepResponse(
        observation=observation,
        reward=observation.reward,
        done=observation.done,
        state=env.state(),
    )


@app.get("/state", response_model=State)
async def get_state():
    """Return current episode metadata."""
    return env.state()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
    )

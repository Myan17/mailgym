# MailGym

An OpenEnv-compliant reinforcement learning environment that simulates real-world email triage — a task every knowledge worker does daily. Agents must classify, prioritize, route, and draft responses to emails across three difficulty levels.

## Motivation

Email triage is one of the most universal productivity bottlenecks in modern work. An estimated 347 billion emails are sent daily, and knowledge workers spend 28% of their workweek managing email. MailGym models the core decision-making loop of email triage, providing a realistic benchmark for training and evaluating AI agents on a task with immediate practical value.

## Environment Description

The environment presents the agent with a single email and a task description. The agent must analyze the email and submit a triage action. The environment grades the action against a deterministic ground truth and returns a reward score between 0.0 and 1.0 with meaningful partial credit.

### Action Space

The agent submits a `TriageAction` with the following fields:

| Field | Type | Required For | Description |
|-------|------|-------------|-------------|
| `category` | string | All tasks | Email classification: `spam`, `urgent`, `routine`, `newsletter`, `personal` |
| `priority` | string | Medium, Hard | Priority level: `P0` (critical), `P1` (high), `P2` (medium), `P3` (low) |
| `department` | string | Medium, Hard | Routing target: `engineering`, `sales`, `support`, `hr`, `legal`, `marketing`, `executive` |
| `response_draft` | string | Hard | A drafted reply to the email |

### Observation Space

After `reset()`, the agent receives:

| Field | Type | Description |
|-------|------|-------------|
| `email` | EmailData | The email to triage (sender, subject, body, timestamp, attachments, thread length) |
| `task` | TaskInfo | Task name, difficulty, instructions, required fields, available options |
| `feedback` | string | Grading feedback (after `step()`) |
| `reward` | float | Score 0.0–1.0 (after `step()`) |
| `done` | bool | Whether the episode has ended |

## Tasks

### 1. classify_easy (Easy)
- **Objective**: Classify the email into one of 5 categories
- **Required fields**: `category`
- **Scoring**: 1.0 if correct, 0.0 if wrong
- **Expected baseline**: ~0.7–0.9

### 2. triage_medium (Medium)
- **Objective**: Classify + assign priority + route to department
- **Required fields**: `category`, `priority`, `department`
- **Scoring**: Category (0.40) + Priority (0.30) + Department (0.30) = 0.0–1.0
- **Expected baseline**: ~0.5–0.7

### 3. full_triage_hard (Hard)
- **Objective**: Full triage + draft an appropriate response
- **Required fields**: `category`, `priority`, `department`, `response_draft`
- **Scoring**: Category (0.20) + Priority (0.15) + Department (0.15) + Keywords (0.25) + Tone (0.10) + Length (0.15) = 0.0–1.0
- **Expected baseline**: ~0.3–0.6

## Reward Design

Rewards provide meaningful partial-credit signals:
- **Easy**: Binary (correct/incorrect classification)
- **Medium**: Weighted sum of three independent components, so getting 2 of 3 right still earns ~0.60–0.70
- **Hard**: Six-component scoring including response quality metrics (keyword coverage, tone matching, response length), ensuring agents get signal even with partially correct answers

Penalties:
- Missing required fields → immediate 0.0
- No destructive actions possible (read-only email environment)

## Setup and Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
cd mailgym
python -m server.app

# The server runs at http://localhost:7860
```

### Docker

```bash
# Build
docker build -t mailgym:latest -f server/Dockerfile .

# Run
docker run -p 7860:7860 mailgym:latest
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/tasks` | List available tasks |
| POST | `/reset` | Start a new episode (`{"task_name": "classify_easy", "seed": 42}`) |
| POST | `/step` | Submit action (`{"action": {"category": "spam", ...}}`) |
| GET | `/state` | Get episode metadata |

### Running the Baseline Inference Script

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-hf-token"
export ENV_BASE_URL="http://localhost:7860"

python inference.py
```

### Using the Python Client

```python
from client import EmailTriageClient
from models import TriageAction

with EmailTriageClient("http://localhost:7860") as client:
    # List tasks
    print(client.tasks())

    # Easy task
    obs, state = client.reset("classify_easy", seed=42)
    print(f"Email: {obs.email.subject}")

    result = client.step(TriageAction(category="spam"))
    print(f"Reward: {result[1]}")
```

## Project Structure

```
mailgym/
├── openenv.yaml          # OpenEnv manifest
├── models.py             # Pydantic models (Action, Observation, State)
├── client.py             # HTTP client for the environment
├── inference.py          # Baseline inference script (OpenAI client)
├── pyproject.toml        # Project metadata and dependencies
├── requirements.txt      # Pip requirements
├── README.md             # This file
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI server with /reset, /step, /state
    ├── environment.py    # Core environment logic
    ├── data.py           # Synthetic email generator (20 templates)
    ├── graders.py        # Deterministic grading functions
    ├── requirements.txt  # Server-only requirements
    └── Dockerfile        # Container deployment
```

## Baseline Scores

Model: `meta-llama/Llama-3.1-8B-Instruct` via HF Inference Router

| Task | Difficulty | Score |
|------|-----------|-------|
| classify_easy | Easy | **1.00** |
| triage_medium | Medium | **1.00** |
| full_triage_hard | Hard | **0.90** |
| **Average** | | **0.97** |

Scores are reproducible with `seed=42` and `temperature=0.0`.

## License

MIT

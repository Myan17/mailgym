# MailGym — Project Brain

> This file is the single source of truth for what MailGym is, what's done, and what's left.
> Read this first on every new session. Update it when state changes.

---

## 1. Project identity

- **Name:** MailGym
- **What it is:** An OpenEnv-compliant RL environment simulating real-world email triage. Agents classify, prioritize, route, and draft responses to emails across three difficulty tiers.
- **Local path:** `/Users/ayangupta/Documents/Claude/Projects/Meta Hackaton/mailgym`
- **GitHub:** `https://github.com/Myan17/mailgym` (remote: `origin`)
- **HF Space:** `https://myan9417-mailgym.hf.space` (remote: `hf`)
- **Status (as of 2026-04-06):** ~90% complete. Environment deployed and live. Main blocker is inference.py stdout format.

---

## 2. Hackathon submission requirements (the rubric we're grading against)

### Hard blockers — must pass or submission is invalid

1. **HF Space returns 200 on `POST /reset`** — live pingable URL
2. **`openenv validate` passes** — yaml + schemas conform to spec
3. **`docker build` succeeds** — from the repo root or `server/`
4. **`inference.py` runs end-to-end** and produces `[START]`/`[STEP]`/`[END]` logs
5. **Runtime budget:** script must finish in < 20 min on a 2 vCPU / 8 GB box
6. **3+ tasks** with deterministic graders producing scores in [0.0, 1.0]

### Mandatory environment variables (read by `inference.py`)

| Var | Purpose | Default |
|---|---|---|
| `API_BASE_URL` | LLM endpoint | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | LLM identifier | none (must set) |
| `HF_TOKEN` | Auth for LLM | none (must set) |
| `LOCAL_IMAGE_NAME` | Only if using `from_docker_image()` | optional |

### Stdout log spec (STRICT — any deviation disqualifies)

```
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

Rules:
- Exactly one `[START]` at episode begin
- One `[STEP]` per step immediately after `env.step()` returns
- One `[END]` after episode ends — **always** emitted (even on exception, via `finally`)
- `reward` and each entry in `rewards` formatted to **2 decimal places**
- `score` formatted to 3 decimal places
- `done` and `success` are lowercase `true`/`false`
- `error` is the raw error string or the literal word `null` (not the string `"null"` wrapped in quotes — the unquoted word)
- No newlines within a line
- Each task's final score must be in [0, 1]

### Evaluation weights (for context, not enforcement)

| Criterion | Weight | Status |
|---|---|---|
| Real-world utility | 30% | ✅ email triage is a genuine daily workflow |
| Task & grader quality | 25% | ✅ 3 tasks easy→hard with partial credit |
| Environment design | 20% | ✅ clean state, typed I/O |
| Code quality & spec compliance | 15% | ✅ inference.py now emits strict format (verified 2026-04-06) |
| Creativity & novelty | 10% | ✅ email triage not commonly seen in OpenEnv |

---

## 3. Current state — what exists, what's broken

### ✅ Done

- `openenv.yaml` — 3 tasks declared
- `models.py` — Pydantic typed `TriageAction`, `Observation`, `State`, `EmailData`, `TaskInfo`, `GroundTruth`, enums
- `server/app.py` — FastAPI with `/reset`, `/step`, `/state`, `/tasks`, `/health`, `/metadata`, `/schema`, `/mcp`, `/`
- `server/environment.py` — episode logic
- `server/data.py` — 20 synthetic email templates
- `server/graders.py` — deterministic graders with partial credit
- `Dockerfile` (root) + `server/Dockerfile`
- `client.py` — HTTP wrapper around the env
- `tests/` — 181 stress tests
- **HF Space deployed** and responding 200 on `/reset`
- Baseline README scores (Llama 3.1 8B): easy=1.00 medium=1.00 hard=0.90 avg=0.97

### ✅ Blockers — ALL CLEARED (2026-04-06)

1. ~~`inference.py` does NOT emit required stdout format~~ — **FIXED.** Rewrote to emit strict `[START]` / `[STEP]` / `[END]` format. Verified against live HF Space (see §10).
2. ~~`openenv validate` CLI not available on PyPI~~ — **RESOLVED.** Installed the upstream CLI from `git+https://github.com/meta-pytorch/openenv.git` into `./.venv` (requires Python 3.10+, so we built a dedicated venv with `/opt/homebrew/bin/python3.12`). Running `.venv/bin/openenv validate` reports `[OK] mailgym: Ready for multi-mode deployment`.
3. ~~Docker build not yet verified~~ — **DONE.** `docker build -t mailgym:latest .` succeeds cleanly from the repo root; image tagged.
4. ~~`validate-submission.sh` not yet run end-to-end~~ — remaining manual step before submission, but all three sub-checks it would run (HF Space 200, docker build, openenv validate) already passed individually. The script itself is not in the repo; grab it from the hackathon docs if needed.
5. ~~Real baseline re-run pending~~ — **DONE.** Scores reproduce deterministically against the live Space with `meta-llama/Llama-3.1-8B-Instruct` at `seed=42`, `temperature=0.0`:
   - `classify_easy` → **1.000**
   - `triage_medium` → **1.000**
   - `full_triage_hard` → **0.900**
   - average → **0.967**
6. ~~HF token leaked in `.git/config`~~ — **RESOLVED.** Old token was deleted from huggingface.co/settings/tokens by the user. New fine-grained/write token stored only in the gitignored `.env`. `.git/config` remote URL cleaned with `git remote set-url hf https://huggingface.co/spaces/myan9417/mailgym` — no credentials embedded.

### ⚠️ Potential issues (verify during final run)

- Is the `/reset` response shape exactly `{"observation": {...}, "state": {...}}`? ✅ confirmed in `server/app.py:275`
- Is the `/step` response shape `{"observation": ..., "reward": ..., "done": ..., "state": ...}`? ✅ confirmed in `server/app.py:295`
- Does each task actually end in 1 step? Check `server/environment.py` `max_steps`. The `State` default is `max_steps=1` → single-step episodes.

---

## 4. Architecture (what talks to what)

```
┌──────────────────┐         HTTP           ┌─────────────────────────┐
│   inference.py   │ ──────────────────▶   │  server/app.py (FastAPI)│
│  (OpenAI client) │ ◀──────────────────   │   /reset /step /state   │
└──────────────────┘                        └─────────────┬───────────┘
         │                                                │
         ▼                                                ▼
  API_BASE_URL                                 server/environment.py
  (LLM router)                                 server/data.py
                                               server/graders.py
                                                           │
                                                           ▼
                                                     models.py
                                                  (Pydantic types)
```

Each episode = 1 email + 1 agent action + 1 graded reward → done. Single-step.

---

## 5. Action / Observation / Reward

### TriageAction (agent → env)

| Field | Type | Required for |
|---|---|---|
| `category` | `spam` \| `urgent` \| `routine` \| `newsletter` \| `personal` | all tasks |
| `priority` | `P0` \| `P1` \| `P2` \| `P3` | medium, hard |
| `department` | `engineering` \| `sales` \| `support` \| `hr` \| `legal` \| `marketing` \| `executive` | medium, hard |
| `response_draft` | string | hard |

### Observation (env → agent)

- `email`: EmailData (sender, subject, body, timestamp, has_attachments, thread_length)
- `task`: TaskInfo (task_name, difficulty, instructions, required_fields, available_*)
- `feedback`: str (after step)
- `reward`: float (after step)
- `done`: bool
- `error_message`: str | None

### Reward shaping

| Task | Components | Max |
|---|---|---|
| `classify_easy` | Binary category match | 1.0 |
| `triage_medium` | category(0.40) + priority(0.30) + department(0.30) | 1.0 |
| `full_triage_hard` | category(0.20) + priority(0.15) + department(0.15) + keywords(0.25) + tone(0.10) + length(0.15) | 1.0 |

Missing required fields → immediate 0.0. No destructive actions possible.

---

## 6. File map

```
mailgym/
├── CLAUDE.md             ← you are here (brain file)
├── openenv.yaml          # OpenEnv manifest
├── Dockerfile            # Root container (used by HF Space)
├── models.py             # Pydantic types
├── client.py             # HTTP client wrapper
├── inference.py          # Baseline agent — NEEDS FORMAT FIX
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── README.md
├── tests/                # 181 stress tests
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI endpoints
    ├── environment.py    # Episode logic + TASK_DEFINITIONS
    ├── data.py           # Synthetic email generator
    ├── graders.py        # Deterministic grading
    ├── requirements.txt
    └── Dockerfile        # Container for HF Space
```

---

## 7. Commands cheat sheet

```bash
# Local dev — run the env server
cd mailgym
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Build the container
docker build -t mailgym:latest .

# Run the container
docker run -p 7860:7860 mailgym:latest

# Run baseline inference (requires env server running)
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."
export ENV_BASE_URL="http://localhost:7860"
python inference.py

# Validate OpenEnv compliance
openenv validate

# Full pre-submission validator
./validate-submission.sh https://myan9417-mailgym.hf.space

# Ping the live Space
curl -X POST -H "Content-Type: application/json" -d '{}' \
  https://myan9417-mailgym.hf.space/reset

# Push to both remotes
git push origin main
git push hf main
```

---

## 8. Next actions

Everything on the technical checklist is green. Only ship-side tasks remain:

1. ~~Rewrite `inference.py`~~ **DONE.**
2. ~~Full baseline run with real `HF_TOKEN`~~ **DONE** — avg 0.967.
3. ~~Docker build~~ **DONE** — exit 0.
4. ~~Install hackathon-compatible `openenv` CLI~~ **DONE** — upstream build in `.venv`.
5. ~~`openenv validate`~~ **DONE** — `[OK] mailgym: Ready for multi-mode deployment`.
6. ~~Rotate HF token + clean `.git/config`~~ **DONE** by user.
7. **Run `validate-submission.sh`** end-to-end (optional sanity check — sub-checks already all pass individually).
8. **Commit + push** to both `origin` (GitHub) and `hf` (HF Space). Files to commit:
   - `inference.py` (modified — strict stdout format)
   - `CLAUDE.md` (new — this brain file)
   - `.gitignore` (modified — ignores `.env`, caches, editors)
   - `.env.example` (new — safe template)
   - `README.md` (updated — baseline run details + env workflow)
   NOT committing: `.env`, `.venv/`, `__pycache__/` (all gitignored).
9. **Submit.**

---

## 9. Gotchas learned from the code

- `server/app.py:22` does a `sys.path.insert(0, ...)` so the server can import `models` from the project root. This means `models.py` must be at the root, not under `server/`.
- `/step` raises 400 if the episode is already done. The inference script must call `/reset` before every task.
- The env has a single global instance (`env = EmailTriageEnvironment()` in `server/app.py:45`), so episodes are sequential, not concurrent.
- `TriageAction.category` is `str` not an enum in the Pydantic model — the grader does the enum check. Sending a wrong category returns a low reward, not a 422.
- `Observation.reward` is `None` until a step has been taken. Use `reward or 0.0` when reading.
- For the log format, `error=null` is the unquoted word `null`, not `"null"` or `None`. Format it as a plain string.
- Single-step episodes: each task = 1 STEP line + 1 END line. `steps=1` in the END line.

---

## 10. Verification log

### 2026-04-06 — inference.py format dry-run
Ran `inference.py` against the live HF Space (`https://myan9417-mailgym.hf.space`) with a dummy `HF_TOKEN` to confirm the stdout format. LLM 401 and env 422 are expected with a dummy token — the only thing being verified here is the log line shape.

Output (sanitized):
```
[START] task=classify_easy env=mailgym model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action={} reward=0.00 done=false error=HTTP 422: ...
[END] success=false steps=1 score=0.000 rewards=0.00
[START] task=triage_medium env=mailgym model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action={} reward=0.00 done=false error=HTTP 422: ...
[END] success=false steps=1 score=0.000 rewards=0.00
[START] task=full_triage_hard env=mailgym model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action={} reward=0.00 done=false error=HTTP 422: ...
[END] success=false steps=1 score=0.000 rewards=0.00
```

Format checklist — all pass:
- [x] One `[START]` per task
- [x] One `[STEP]` per env call
- [x] One `[END]` per task (emitted via `finally`, so present even with failures)
- [x] `reward` formatted to 2 decimals (`0.00`)
- [x] `score` formatted to 3 decimals (`0.000`)
- [x] `done` / `success` are lowercase booleans
- [x] `error` contains raw error string (not wrapped in quotes)
- [x] Task-to-block ordering preserved: easy → medium → hard
- [x] Single-line log entries (no embedded newlines in action/error)

### HF Space liveness
```
POST https://myan9417-mailgym.hf.space/reset → HTTP 200
```

### 2026-04-06 — `openenv validate`
```
$ .venv/bin/openenv validate
[OK] mailgym: Ready for multi-mode deployment
```

### 2026-04-06 — `docker build`
```
$ docker build -t mailgym:latest .
...
#15 naming to docker.io/library/mailgym:latest done
exit 0
```

### 2026-04-06 — real baseline run (FINAL)
Full run against the live HF Space with a valid `HF_TOKEN`, `model=meta-llama/Llama-3.1-8B-Instruct`, `seed=42`, `temperature=0.0`. Result is deterministic and reproducible on repeat runs.

```
[START] task=classify_easy env=mailgym model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action={"category":"urgent"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00
[START] task=triage_medium env=mailgym model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action={"category":"urgent","priority":"P0","department":"engineering"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.000 rewards=1.00
[START] task=full_triage_hard env=mailgym model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action={"category":"urgent","priority":"P0","department":"engineering","response_draft":"Hi team, I've been alerted to the production database outage. ..."} reward=0.90 done=true error=null
[END] success=true steps=1 score=0.900 rewards=0.90
```

Scores:
| Task | Score |
|---|---|
| classify_easy | **1.000** |
| triage_medium | **1.000** |
| full_triage_hard | **0.900** |
| **Average** | **0.967** |

### 2026-04-06 — credential hygiene
- Old leaked HF token deleted at huggingface.co/settings/tokens (by user)
- `.git/config` cleaned: `hf` remote no longer has embedded credentials
- New token stored only in gitignored `.env`; `.env.example` committed as a safe template

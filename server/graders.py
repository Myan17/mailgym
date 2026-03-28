"""
Grading functions for Email Triage tasks.

Each grader returns a float score in [0.0, 1.0] with meaningful partial credit.
Graders are deterministic and reproducible.
"""

from __future__ import annotations

from models import GroundTruth, TriageAction


def _normalize(value: str | None) -> str:
    """Lowercase and strip a string for comparison."""
    if value is None:
        return ""
    return value.strip().lower()


def _keyword_score(text: str, keywords: list[str]) -> float:
    """Score 0.0–1.0 based on fraction of expected keywords present in text."""
    if not keywords:
        return 1.0  # No keywords expected → full marks
    text_lower = text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in text_lower)
    return matches / len(keywords)


def _tone_score(text: str, expected_tone: str) -> float:
    """
    Heuristic tone check.  Returns 0.0, 0.5, or 1.0.
    """
    text_lower = text.lower()

    tone_signals = {
        "urgent": ["immediately", "asap", "right away", "urgent", "critical", "now"],
        "empathetic": ["sorry", "apolog", "understand", "appreciate", "regret"],
        "professional": ["regards", "thank", "please", "sincerely", "best"],
        "casual": ["hey", "cheers", "cool", "sounds good", "awesome", "!"],
    }

    signals = tone_signals.get(expected_tone, tone_signals["professional"])
    matches = sum(1 for s in signals if s in text_lower)

    if matches >= 2:
        return 1.0
    elif matches == 1:
        return 0.5
    return 0.0


# ── Task Graders ─────────────────────────────────────────────────────────────

def grade_easy(action: TriageAction, truth: GroundTruth) -> tuple[float, str]:
    """
    Easy task: Email classification only.
    Score: 1.0 if correct category, 0.0 otherwise.
    """
    predicted = _normalize(action.category)
    expected = _normalize(truth.category.value)

    if predicted == expected:
        return 1.0, f"Correct! Category is '{truth.category.value}'."
    else:
        return 0.0, (
            f"Incorrect. You predicted '{action.category}', "
            f"but the correct category is '{truth.category.value}'."
        )


def grade_medium(action: TriageAction, truth: GroundTruth) -> tuple[float, str]:
    """
    Medium task: Classification + Priority + Department routing.
    Partial credit:
      - Category correct:   0.40
      - Priority correct:   0.30
      - Department correct:  0.30
    Total: 0.0–1.0
    """
    score = 0.0
    feedback_parts = []

    # Category (0.40)
    if _normalize(action.category) == _normalize(truth.category.value):
        score += 0.40
        feedback_parts.append("Category: correct (+0.40)")
    else:
        feedback_parts.append(
            f"Category: wrong (expected '{truth.category.value}', "
            f"got '{action.category}')"
        )

    # Priority (0.30)
    if _normalize(action.priority) == _normalize(truth.priority.value):
        score += 0.30
        feedback_parts.append("Priority: correct (+0.30)")
    else:
        feedback_parts.append(
            f"Priority: wrong (expected '{truth.priority.value}', "
            f"got '{action.priority}')"
        )

    # Department (0.30)
    if _normalize(action.department) == _normalize(truth.department.value):
        score += 0.30
        feedback_parts.append("Department: correct (+0.30)")
    else:
        feedback_parts.append(
            f"Department: wrong (expected '{truth.department.value}', "
            f"got '{action.department}')"
        )

    feedback = " | ".join(feedback_parts) + f" | Total: {score:.2f}"
    return round(score, 2), feedback


def grade_hard(action: TriageAction, truth: GroundTruth) -> tuple[float, str]:
    """
    Hard task: Full triage + draft response.
    Partial credit:
      - Category correct:       0.20
      - Priority correct:       0.15
      - Department correct:     0.15
      - Response keywords:      0.25 (fraction of expected keywords found)
      - Response tone:          0.10
      - Response length:        0.15 (penalty if too short)
    Total: 0.0–1.0
    """
    score = 0.0
    feedback_parts = []

    # Category (0.20)
    if _normalize(action.category) == _normalize(truth.category.value):
        score += 0.20
        feedback_parts.append("Category: correct (+0.20)")
    else:
        feedback_parts.append(
            f"Category: wrong (expected '{truth.category.value}')"
        )

    # Priority (0.15)
    if _normalize(action.priority) == _normalize(truth.priority.value):
        score += 0.15
        feedback_parts.append("Priority: correct (+0.15)")
    else:
        feedback_parts.append(
            f"Priority: wrong (expected '{truth.priority.value}')"
        )

    # Department (0.15)
    if _normalize(action.department) == _normalize(truth.department.value):
        score += 0.15
        feedback_parts.append("Department: correct (+0.15)")
    else:
        feedback_parts.append(
            f"Department: wrong (expected '{truth.department.value}')"
        )

    # Response quality
    response = action.response_draft or ""

    # Keywords (0.25)
    kw_score = _keyword_score(response, truth.expected_response_keywords)
    kw_points = round(0.25 * kw_score, 2)
    score += kw_points
    feedback_parts.append(
        f"Response keywords: {kw_score:.0%} match (+{kw_points:.2f})"
    )

    # Tone (0.10)
    t_score = _tone_score(response, truth.expected_response_tone)
    t_points = round(0.10 * t_score, 2)
    score += t_points
    feedback_parts.append(
        f"Response tone: {t_score:.0%} match (+{t_points:.2f})"
    )

    # Length (0.15) — at least 50 chars for a meaningful response
    if len(response) >= 150:
        len_points = 0.15
    elif len(response) >= 50:
        len_points = round(0.15 * (len(response) / 150), 2)
    elif len(response) > 0:
        len_points = 0.03
    else:
        len_points = 0.0
    score += len_points
    feedback_parts.append(
        f"Response length ({len(response)} chars): +{len_points:.2f}"
    )

    feedback = " | ".join(feedback_parts) + f" | Total: {score:.2f}"
    return round(min(score, 1.0), 2), feedback


# ── Dispatcher ───────────────────────────────────────────────────────────────

GRADERS = {
    "classify_easy": grade_easy,
    "triage_medium": grade_medium,
    "full_triage_hard": grade_hard,
}


def grade(task_name: str, action: TriageAction, truth: GroundTruth) -> tuple[float, str]:
    """Grade an action for the given task. Returns (score, feedback)."""
    grader = GRADERS.get(task_name)
    if grader is None:
        return 0.0, f"Unknown task: {task_name}"
    return grader(action, truth)

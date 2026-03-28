"""
Synthetic email generator for the Email Triage environment.

Produces realistic emails with known ground-truth labels for deterministic grading.
Each email has a category, priority, department, and (for hard tasks) expected
response keywords.
"""

from __future__ import annotations

import random
from typing import List

from models import (
    Department,
    EmailCategory,
    EmailData,
    GroundTruth,
    Priority,
)

# ── Email Templates ──────────────────────────────────────────────────────────
# Each template: (subject, body, sender, category, priority, department,
#                 response_keywords, response_tone, has_attachments, thread_length)

_EMAIL_POOL = [
    # ── SPAM ──
    {
        "sender": "deals@cheapmeds-online.xyz",
        "subject": "CONGRATULATIONS! You've Won $1,000,000!!!",
        "body": (
            "Dear Lucky Winner,\n\n"
            "You have been selected as our grand prize winner! Click here to "
            "claim your $1,000,000 cash prize. Act NOW before this offer expires!\n\n"
            "No purchase necessary. Limited time only."
        ),
        "category": EmailCategory.SPAM,
        "priority": Priority.P3,
        "department": Department.SUPPORT,
        "response_keywords": [],
        "response_tone": "professional",
        "has_attachments": False,
        "thread_length": 1,
    },
    {
        "sender": "noreply@crypto-gains-4u.net",
        "subject": "Make 500% returns GUARANTEED with this crypto trick",
        "body": (
            "Hi there,\n\n"
            "Our AI trading bot has been generating 500% returns for early adopters. "
            "Deposit just $100 and watch your portfolio explode. Join 50,000 happy "
            "investors today!\n\nVisit http://totally-legit-crypto.biz"
        ),
        "category": EmailCategory.SPAM,
        "priority": Priority.P3,
        "department": Department.SUPPORT,
        "response_keywords": [],
        "response_tone": "professional",
        "has_attachments": False,
        "thread_length": 1,
    },
    {
        "sender": "admin@your-bank-security.com",
        "subject": "URGENT: Your account has been compromised",
        "body": (
            "Dear Valued Customer,\n\n"
            "We detected suspicious activity on your account. Please verify your "
            "identity immediately by clicking the link below and entering your "
            "credentials:\n\nhttp://not-your-bank.phishing.com/verify\n\n"
            "Failure to act within 24 hours will result in account suspension."
        ),
        "category": EmailCategory.SPAM,
        "priority": Priority.P3,
        "department": Department.SUPPORT,
        "response_keywords": [],
        "response_tone": "professional",
        "has_attachments": False,
        "thread_length": 1,
    },
    # ── URGENT ──
    {
        "sender": "ops-alert@company.com",
        "subject": "CRITICAL: Production database is down",
        "body": (
            "Team,\n\n"
            "Our primary production database cluster went offline at 03:42 UTC. "
            "Customer-facing services are returning 500 errors. Estimated impact: "
            "all API traffic (approx. 50k req/min). The on-call DBA has been paged "
            "but we need engineering leads to join the war room.\n\n"
            "War room link: https://meet.company.com/incident-2847\n"
            "Incident ticket: INC-2847"
        ),
        "category": EmailCategory.URGENT,
        "priority": Priority.P0,
        "department": Department.ENGINEERING,
        "response_keywords": ["acknowledge", "join", "war room", "investigating", "ETA"],
        "response_tone": "urgent",
        "has_attachments": False,
        "thread_length": 3,
    },
    {
        "sender": "security@company.com",
        "subject": "SECURITY ALERT: Unauthorized access detected on staging",
        "body": (
            "Hi Engineering,\n\n"
            "Our SIEM flagged 47 unauthorized SSH login attempts to the staging "
            "environment from IP 203.0.113.42 over the last 30 minutes. Two attempts "
            "were successful using compromised credentials (user: deploy-bot).\n\n"
            "Immediate actions needed:\n"
            "1. Rotate all staging credentials\n"
            "2. Block the source IP range\n"
            "3. Audit deploy-bot access logs\n\n"
            "Please respond ASAP with your availability."
        ),
        "category": EmailCategory.URGENT,
        "priority": Priority.P0,
        "department": Department.ENGINEERING,
        "response_keywords": ["acknowledge", "rotate", "credentials", "block", "investigating"],
        "response_tone": "urgent",
        "has_attachments": True,
        "thread_length": 2,
    },
    {
        "sender": "ceo@company.com",
        "subject": "Board meeting moved to tomorrow — need updated financials",
        "body": (
            "Hi Finance team,\n\n"
            "The quarterly board meeting has been moved up to tomorrow at 9 AM. "
            "I need the updated Q1 financial summary, including the revised revenue "
            "projections we discussed last week. Please have this ready by EOD today.\n\n"
            "Also, please flag any material changes from the preliminary numbers.\n\n"
            "Thanks,\nCEO"
        ),
        "category": EmailCategory.URGENT,
        "priority": Priority.P1,
        "department": Department.EXECUTIVE,
        "response_keywords": ["confirmed", "financials", "EOD", "ready", "updated"],
        "response_tone": "professional",
        "has_attachments": False,
        "thread_length": 1,
    },
    # ── ROUTINE ──
    {
        "sender": "jira@company.atlassian.net",
        "subject": "[JIRA] PROJ-1234: Update API documentation for v2 endpoints",
        "body": (
            "Assignee: You\n"
            "Reporter: Sarah Chen\n"
            "Priority: Medium\n\n"
            "Description:\n"
            "The v2 API endpoints launched last sprint but the developer docs still "
            "reference v1. Please update the OpenAPI spec and README to reflect the "
            "new request/response schemas.\n\n"
            "Due date: End of sprint (Friday)"
        ),
        "category": EmailCategory.ROUTINE,
        "priority": Priority.P2,
        "department": Department.ENGINEERING,
        "response_keywords": ["acknowledged", "update", "documentation", "sprint", "Friday"],
        "response_tone": "professional",
        "has_attachments": False,
        "thread_length": 1,
    },
    {
        "sender": "hr@company.com",
        "subject": "Reminder: Submit your timesheet by Friday",
        "body": (
            "Hi team,\n\n"
            "Friendly reminder that timesheets for this pay period are due by "
            "Friday at 5 PM. Please log your hours in the HR portal.\n\n"
            "If you have questions about project codes, check the updated list "
            "on the intranet or reach out to your manager.\n\n"
            "Thanks,\nHR Team"
        ),
        "category": EmailCategory.ROUTINE,
        "priority": Priority.P2,
        "department": Department.HR,
        "response_keywords": ["submitted", "timesheet", "confirm"],
        "response_tone": "professional",
        "has_attachments": False,
        "thread_length": 1,
    },
    {
        "sender": "facilities@company.com",
        "subject": "Office kitchen renovation — temporary closure next week",
        "body": (
            "Hello everyone,\n\n"
            "The 3rd floor kitchen will be closed Monday through Wednesday next "
            "week for renovations. Please use the 2nd floor kitchen during this time. "
            "Coffee service will be available in the lobby.\n\n"
            "We apologize for the inconvenience.\n\n"
            "— Facilities Management"
        ),
        "category": EmailCategory.ROUTINE,
        "priority": Priority.P3,
        "department": Department.HR,
        "response_keywords": ["noted", "thanks"],
        "response_tone": "professional",
        "has_attachments": False,
        "thread_length": 1,
    },
    # ── NEWSLETTER ──
    {
        "sender": "newsletter@techcrunch.com",
        "subject": "TechCrunch Daily: AI startups raise record $2B in Q1",
        "body": (
            "Good morning!\n\n"
            "Today's top stories:\n"
            "• AI startups raised a record $2B in Q1 2026\n"
            "• Apple announces new developer tools at WWDC preview\n"
            "• The future of remote work: 5 trends to watch\n"
            "• European regulators propose new AI transparency rules\n\n"
            "Read more at techcrunch.com"
        ),
        "category": EmailCategory.NEWSLETTER,
        "priority": Priority.P3,
        "department": Department.MARKETING,
        "response_keywords": [],
        "response_tone": "professional",
        "has_attachments": False,
        "thread_length": 1,
    },
    {
        "sender": "digest@github.com",
        "subject": "Your weekly GitHub digest — 12 new stars, 3 PRs merged",
        "body": (
            "Here's your weekly summary:\n\n"
            "Repositories you starred got 12 new stars this week.\n"
            "3 pull requests were merged across your projects.\n"
            "2 new issues were opened.\n\n"
            "Top trending: pytorch/openenv — Agentic execution environments\n\n"
            "See your full digest on GitHub."
        ),
        "category": EmailCategory.NEWSLETTER,
        "priority": Priority.P3,
        "department": Department.ENGINEERING,
        "response_keywords": [],
        "response_tone": "professional",
        "has_attachments": False,
        "thread_length": 1,
    },
    {
        "sender": "updates@company.com",
        "subject": "Company All-Hands Recap — March 2026",
        "body": (
            "Hi team,\n\n"
            "Thanks for joining this month's all-hands! Here's a quick recap:\n\n"
            "• Q1 revenue exceeded targets by 12%\n"
            "• New product launch scheduled for April 15\n"
            "• Engineering headcount increasing by 20%\n"
            "• New parental leave policy effective immediately\n\n"
            "Recording available on the intranet."
        ),
        "category": EmailCategory.NEWSLETTER,
        "priority": Priority.P3,
        "department": Department.EXECUTIVE,
        "response_keywords": [],
        "response_tone": "professional",
        "has_attachments": False,
        "thread_length": 1,
    },
    # ── PERSONAL ──
    {
        "sender": "mike.johnson@gmail.com",
        "subject": "Lunch this Friday?",
        "body": (
            "Hey!\n\n"
            "It's been a while since we caught up. Want to grab lunch this Friday? "
            "I was thinking that new ramen place on Market Street. Let me know "
            "if you're free around noon.\n\n"
            "Cheers,\nMike"
        ),
        "category": EmailCategory.PERSONAL,
        "priority": Priority.P3,
        "department": Department.HR,
        "response_keywords": ["sounds good", "Friday", "noon", "lunch", "confirm"],
        "response_tone": "casual",
        "has_attachments": False,
        "thread_length": 1,
    },
    {
        "sender": "sarah.teammate@company.com",
        "subject": "Re: Your promotion — congrats!!",
        "body": (
            "Hey, just heard the news — huge congratulations on the promotion! "
            "You totally deserve it after leading the platform migration. "
            "Let's celebrate after work sometime this week. Drinks on me!\n\n"
            "— Sarah"
        ),
        "category": EmailCategory.PERSONAL,
        "priority": Priority.P3,
        "department": Department.HR,
        "response_keywords": ["thank", "celebrate", "appreciate"],
        "response_tone": "casual",
        "has_attachments": False,
        "thread_length": 4,
    },
    # ── More URGENT (variety) ──
    {
        "sender": "legal@company.com",
        "subject": "URGENT: Cease & desist received — response needed by EOD",
        "body": (
            "Team,\n\n"
            "We received a cease and desist letter from XCorp alleging patent "
            "infringement on our recommendation engine. Outside counsel needs "
            "a technical brief describing our implementation by end of day.\n\n"
            "Key questions:\n"
            "1. What algorithms does our recommendation engine use?\n"
            "2. When was the current implementation first deployed?\n"
            "3. Are there any third-party libraries involved?\n\n"
            "Please treat this as highest priority."
        ),
        "category": EmailCategory.URGENT,
        "priority": Priority.P0,
        "department": Department.LEGAL,
        "response_keywords": ["acknowledged", "brief", "implementation", "EOD", "counsel"],
        "response_tone": "professional",
        "has_attachments": True,
        "thread_length": 2,
    },
    # ── More ROUTINE (variety) ──
    {
        "sender": "vendor@cloudprovider.com",
        "subject": "Your monthly cloud invoice — March 2026",
        "body": (
            "Hi,\n\n"
            "Your invoice for March 2026 is ready.\n\n"
            "Total: $14,327.89\n"
            "Due date: April 15, 2026\n\n"
            "Breakdown:\n"
            "• Compute: $8,200.00\n"
            "• Storage: $3,127.89\n"
            "• Network: $2,000.00\n"
            "• Support: $1,000.00\n\n"
            "View full invoice at dashboard.cloudprovider.com"
        ),
        "category": EmailCategory.ROUTINE,
        "priority": Priority.P2,
        "department": Department.ENGINEERING,
        "response_keywords": ["received", "invoice", "processed", "payment"],
        "response_tone": "professional",
        "has_attachments": True,
        "thread_length": 1,
    },
    {
        "sender": "recruiting@company.com",
        "subject": "Interview panel request: Senior Engineer candidate Thursday",
        "body": (
            "Hi,\n\n"
            "We have a strong Senior Engineer candidate coming in Thursday at 2 PM. "
            "Could you be on the technical interview panel? The interview will be "
            "45 minutes focused on system design.\n\n"
            "Candidate resume attached. Please confirm your availability.\n\n"
            "Thanks,\nRecruiting Team"
        ),
        "category": EmailCategory.ROUTINE,
        "priority": Priority.P2,
        "department": Department.HR,
        "response_keywords": ["confirm", "available", "Thursday", "interview"],
        "response_tone": "professional",
        "has_attachments": True,
        "thread_length": 1,
    },
    # ── Customer-facing URGENT ──
    {
        "sender": "enterprise-client@bigcorp.com",
        "subject": "SLA breach — our dashboard has been down for 4 hours",
        "body": (
            "Hi Support,\n\n"
            "Our executive dashboard has been returning 502 errors since 6 AM this "
            "morning. This is a Tier 1 SLA violation — our contract guarantees "
            "99.99% uptime and we have a board presentation at noon.\n\n"
            "We need immediate escalation and a status update within 30 minutes.\n\n"
            "Account ID: ENT-4001\n"
            "Contract tier: Enterprise Plus\n\n"
            "Regards,\nVP of Operations, BigCorp"
        ),
        "category": EmailCategory.URGENT,
        "priority": Priority.P0,
        "department": Department.SUPPORT,
        "response_keywords": ["apologies", "investigating", "escalated", "status update", "SLA"],
        "response_tone": "empathetic",
        "has_attachments": False,
        "thread_length": 1,
    },
    # ── Sales ROUTINE ──
    {
        "sender": "prospect@startup.io",
        "subject": "Interested in your enterprise plan — can we schedule a demo?",
        "body": (
            "Hi,\n\n"
            "We're a 50-person startup evaluating tools for our engineering team. "
            "Your enterprise plan looks like a good fit. Could we schedule a "
            "30-minute demo sometime next week?\n\n"
            "We're particularly interested in:\n"
            "• SSO integration\n"
            "• API rate limits on enterprise tier\n"
            "• Custom SLA options\n\n"
            "Looking forward to hearing from you.\n\n"
            "Best,\nCTO, Startup.io"
        ),
        "category": EmailCategory.ROUTINE,
        "priority": Priority.P1,
        "department": Department.SALES,
        "response_keywords": ["demo", "schedule", "enterprise", "happy to", "next week"],
        "response_tone": "professional",
        "has_attachments": False,
        "thread_length": 1,
    },
]


def get_random_email(seed: int | None = None) -> tuple[EmailData, GroundTruth]:
    """Return a random email and its ground truth, optionally seeded for reproducibility."""
    rng = random.Random(seed)
    template = rng.choice(_EMAIL_POOL)

    email = EmailData(
        sender=template["sender"],
        subject=template["subject"],
        body=template["body"],
        timestamp="2026-03-28T10:30:00Z",
        has_attachments=template["has_attachments"],
        thread_length=template["thread_length"],
    )

    ground_truth = GroundTruth(
        category=template["category"],
        priority=template["priority"],
        department=template["department"],
        expected_response_keywords=template["response_keywords"],
        expected_response_tone=template["response_tone"],
    )

    return email, ground_truth


def get_email_by_index(index: int) -> tuple[EmailData, GroundTruth]:
    """Return a specific email by index (for deterministic testing)."""
    template = _EMAIL_POOL[index % len(_EMAIL_POOL)]

    email = EmailData(
        sender=template["sender"],
        subject=template["subject"],
        body=template["body"],
        timestamp="2026-03-28T10:30:00Z",
        has_attachments=template["has_attachments"],
        thread_length=template["thread_length"],
    )

    ground_truth = GroundTruth(
        category=template["category"],
        priority=template["priority"],
        department=template["department"],
        expected_response_keywords=template["response_keywords"],
        expected_response_tone=template["response_tone"],
    )

    return email, ground_truth


def get_pool_size() -> int:
    """Return the number of emails in the pool."""
    return len(_EMAIL_POOL)

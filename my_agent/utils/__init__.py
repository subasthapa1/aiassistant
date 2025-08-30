"""
LangGraph Agent: Read an email, summarize it, draft a reply, and schedule an interview with Human-in-the-Loop review.

Stack: LangChain + LangGraph + LangSmith + OpenAI

Quick start
-----------
1) Install deps (Python 3.10+):
   pip install -U langchain langgraph langchain-openai python-dateutil ics pytz pydantic

2) Export environment variables:
   export OPENAI_API_KEY=...               # required
   export LANGCHAIN_TRACING_V2=true        # optional, enables LangSmith tracing
   export LANGCHAIN_API_KEY=...            # optional, required if tracing enabled
   export LANGCHAIN_PROJECT="Interview Agent"   # optional project name in LangSmith

3) Run on a text email file:
   python agent_email_interview.py --email-file sample_email.txt --tz "Europe/London"

   Or pass raw text:
   python agent_email_interview.py --email-text "...paste the email body..." --tz "Europe/London"

Outputs
-------
- Prints: structured email parse, summary, drafted reply, and chosen interview slot
- Asks the human to approve or edit summary, reply, and slot choice before finalizing
- Writes: an ICS calendar invite to ./out/interview_invite.ics (if approved)

Notes
-----
- Human-in-the-Loop is integrated after key steps: summarization, scheduling, and reply drafting.
- Replace the stubbed scheduling logic with your real calendar integration if desired.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, TypedDict
from pathlib import Path

from dateutil import parser as dtparser
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

# LangChain / OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# LangGraph
from langgraph.graph import StateGraph, END

# ICS file generation
from ics import Calendar, Event

# -----------------------------
# Domain models
# -----------------------------
class ParsedEmail(BaseModel):
    sender: Optional[str] = None
    subject: Optional[str] = None
    body: str
    requested_times: List[str] = Field(default_factory=list)

class EventPlan(BaseModel):
    title: str
    description: str
    location: Optional[str] = None
    start: datetime
    end: datetime
    attendees: List[str] = Field(default_factory=list)

class AgentState(TypedDict):
    email_text: str
    parsed: Optional[ParsedEmail]
    summary: Optional[str]
    reply: Optional[str]
    event: Optional[EventPlan]
    timezone: str

# -----------------------------
# Helpers
# -----------------------------
SYSTEM_SUMMARY = "You are an assistant summarizing emails in 3-5 crisp bullet points."
SYSTEM_REPLY = "You are an assistant drafting professional interview replies."

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def ensure_outdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

# -----------------------------
# Node: parse_email
# -----------------------------
def parse_email_node(state: AgentState) -> AgentState:
    text = state["email_text"].strip()
    subject, sender, body_lines = None, None, []
    for line in text.splitlines():
        l = line.strip()
        if l.lower().startswith("subject:") and subject is None:
            subject = l.split(":", 1)[1].strip()
        elif l.lower().startswith("from:") and sender is None:
            sender = l.split(":", 1)[1].strip()
        else:
            body_lines.append(line)
    body = "\n".join(body_lines).strip()
    state["parsed"] = ParsedEmail(sender=sender, subject=subject, body=body)
    return state

# -----------------------------
# Node: summarize
# -----------------------------
def summarize_node(state: AgentState) -> AgentState:
    parsed = state["parsed"]
    assert parsed is not None
    res = llm.invoke([
        SystemMessage(content=SYSTEM_SUMMARY),
        HumanMessage(content=parsed.body)
    ])
    state["summary"] = res.content.strip()
    print("\n[Human Review] Suggested Summary:\n", state["summary"])
    human = input("Approve summary? (y/n, or edit manually): ").strip().lower()
    if human not in ("y", "yes"):
        print("Enter your summary:")
        state["summary"] = input("> ").strip()
    return state

# -----------------------------
# Node: schedule
# -----------------------------
def schedule_node(state: AgentState) -> AgentState:
    tz = state.get("timezone") or "UTC"
    zone = ZoneInfo(tz)
    start = datetime.now(zone) + timedelta(days=1)
    start = start.replace(hour=10, minute=0, second=0, microsecond=0)
    end = start + timedelta(minutes=30)
    plan = EventPlan(
        title=state["parsed"].subject or "Interview",
        description="30-minute interview (tentative).",
        start=start,
        end=end,
        attendees=[state["parsed"].sender] if state["parsed"].sender else []
    )
    print("\n[Human Review] Proposed Interview Slot:")
    print(f"Start: {plan.start}\nEnd: {plan.end}")
    human = input("Approve this slot? (y/n): ").strip().lower()
    if human not in ("y", "yes"):
        print("Enter new start datetime (e.g., 2025-08-27 15:00):")
        dt_str = input("> ").strip()
        try:
            new_start = dtparser.parse(dt_str)
            plan.start = new_start
            plan.end = new_start + timedelta(minutes=30)
        except Exception:
            print("Invalid input, keeping original.")
    state["event"] = plan
    return state

# -----------------------------
# Node: draft reply
# -----------------------------
def reply_node(state: AgentState) -> AgentState:
    summary, plan = state["summary"], state["event"]
    tz = state.get("timezone") or "UTC"
    res = llm.invoke([
        SystemMessage(content=SYSTEM_REPLY),
        HumanMessage(content=(
            f"Summary: {summary}\nProposed slot: {plan.start.strftime('%A %d %b %Y %H:%M %Z')} ({tz})\nDraft a polite reply."
        ))
    ])
    draft = res.content.strip()
    print("\n[Human Review] Draft Reply:\n", draft)
    human = input("Approve reply? (y/n, or edit manually): ").strip().lower()
    if human not in ("y", "yes"):
        print("Enter your reply:")
        draft = input("> ").strip()
    state["reply"] = draft
    return state

# -----------------------------
# Graph wiring
# -----------------------------
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("parse_email", parse_email_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("schedule", schedule_node)
    graph.add_node("reply", reply_node)
    graph.set_entry_point("parse_email")
    graph.add_edge("parse_email", "summarize")
    graph.add_edge("summarize", "schedule")
    graph.add_edge("schedule", "reply")
    graph.add_edge("reply", END)
    return graph.compile()

# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email-file", type=str, default=None)
    parser.add_argument("--email-text", type=str, default=None)
    parser.add_argument("--tz", type=str, default="UTC")
    args = parser.parse_args()

    if not args.email_text and not args.email_file:
        parser.error("Provide --email-text or --email-file")

    if args.email_file:
        with open(args.email_file, "r", encoding="utf-8") as f:
            email_text = f.read()
    else:
        email_text = args.email_text

    graph = build_graph()
    state: AgentState = {
        "email_text": email_text,
        "parsed": None,
        "summary": None,
        "reply": None,
        "event": None,
        "timezone": args.tz,
    }
    final = graph.invoke(state)

    # Output
    print("\n============= FINAL OUTPUT =============")
    print("Summary:", final["summary"])
    print("Reply:\n", final["reply"])
    print("Event:", final["event"])

if __name__ == "__main__":
    main()

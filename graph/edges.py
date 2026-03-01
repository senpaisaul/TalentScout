"""
graph/edges.py
--------------
NOTE (v2): The main routing logic has been moved inline into graph/graph.py
as part of the dispatcher pattern fix. This file is kept for reference and
documents the routing decisions.

The three edge functions that remain (_last_human_message, _exit_requested)
are utility helpers that are still available for testing.
"""

from graph.state import (
    CandidateState,
    EXIT_KEYWORDS,
    all_info_collected,
    has_more_questions,
    exit_detected,
)


# ---------------------------------------------------------------------------
# Helpers (used in tests / external tooling)
# ---------------------------------------------------------------------------

def _last_human_message(state: CandidateState) -> str:
    """Returns the content of the most recent HumanMessage, lowercased."""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "human":
            return msg.content.lower()
        if isinstance(msg, dict) and msg.get("role") == "human":
            return msg.get("content", "").lower()
    return ""


def _exit_requested(state: CandidateState) -> bool:
    """True if the last human message contains an exit keyword."""
    text = _last_human_message(state).split()
    return any(kw in text for kw in EXIT_KEYWORDS)


# ---------------------------------------------------------------------------
# Legacy routing functions (preserved for reference; no longer wired in graph)
# ---------------------------------------------------------------------------

def route_after_greeting(state: CandidateState) -> str:
    if _exit_requested(state):
        return "farewell"
    return "info_gather"


def route_after_info_gather(state: CandidateState) -> str:
    if _exit_requested(state):
        return "farewell"
    if all_info_collected(state):
        return "tech_stack"
    return "continue_info"


def route_after_tech_stack(state: CandidateState) -> str:
    if _exit_requested(state):
        return "farewell"
    tech = state.get("tech_stack")
    if tech and len(tech) > 0:
        return "generate_questions"
    return "end"


def route_after_interview(state: CandidateState) -> str:
    if _exit_requested(state):
        return "farewell"
    if state.get("is_complete"):
        return "farewell"
    if state.get("current_stage") == "farewell":
        return "farewell"
    if not has_more_questions(state):
        return "farewell"
    return "continue"

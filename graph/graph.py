"""
graph/graph.py
--------------
Assembles the TalentScout conversation graph using LangGraph's StateGraph.

FIX (v2): Replaced the self-looping design (which caused infinite API calls)
with a DISPATCHER pattern. Each .invoke() call now does exactly ONE turn:

    START -> dispatcher (reads current_stage) -> correct node -> END

This means:
  - No node ever loops back to itself within a single .invoke()
  - The Streamlit app calls graph.invoke() once per user message
  - State is persisted between invokes by Streamlit session_state (no checkpointer)

Graph topology (per-invoke, one turn):

    START
      -> dispatcher
           |-> greeting      -> END   (stage: greeting  -> info_gather)
           |-> info_gather   -> END   (stage: info_gather, advances to tech_stack when done)
           |-> tech_stack    -> END   (stage: tech_stack, if no stack yet)
           |        `-------> generate_questions -> END  (if stack provided; presents Q1)
           |-> interview     -> END   (normal turn; off-topic handled inline)
           |        `-------> farewell -> END  (when all Qs answered or exit)
           `-> farewell      -> END   (exit keyword or completion)
"""

from langgraph.graph import StateGraph, START, END

from graph.state import CandidateState, EXIT_KEYWORDS
from graph.nodes import (
    greeting_node,
    info_gather_node,
    tech_stack_node,
    generate_questions_node,
    interview_node,
    farewell_node,
)


# ---------------------------------------------------------------------------
# Dispatcher — reads current_stage, checks exit keywords
# ---------------------------------------------------------------------------

def _dispatcher(state: CandidateState) -> str:
    """
    Central routing function attached to START.
    1. Scans the last human message for exit keywords -> "farewell" if found.
    2. Routes to the node matching current_stage.
    """
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "human":
            words = msg.content.lower().split()
            if any(kw in words for kw in EXIT_KEYWORDS):
                return "farewell"
            break
        if isinstance(msg, dict) and msg.get("role") == "human":
            words = msg.get("content", "").lower().split()
            if any(kw in words for kw in EXIT_KEYWORDS):
                return "farewell"
            break

    stage = state.get("current_stage", "greeting")
    valid = {"greeting", "info_gather", "tech_stack", "generate_questions", "interview", "farewell"}
    return stage if stage in valid else "info_gather"


# ---------------------------------------------------------------------------
# Post-tech_stack routing
# ---------------------------------------------------------------------------

def _route_after_tech_stack(state: CandidateState) -> str:
    """
    If the tech stack was successfully extracted this turn, chain straight to
    generate_questions (no user input needed between them).
    Otherwise END — tech_stack_node already asked the candidate to list their stack.
    """
    tech = state.get("tech_stack")
    if tech and len(tech) > 0:
        return "generate_questions"
    return "end"


# ---------------------------------------------------------------------------
# Post-interview routing
# ---------------------------------------------------------------------------

def _route_after_interview(state: CandidateState) -> str:
    """
    After the interview node runs:
    - is_complete or stage==farewell -> farewell node
    - otherwise                      -> END (wait for next user message)
    """
    if state.get("is_complete") or state.get("current_stage") == "farewell":
        return "farewell"
    return "end"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    """
    Builds and compiles the TalentScout conversation graph.
    No MemorySaver checkpointer — state is managed by Streamlit's session_state.
    Each .invoke() call receives the full current state and returns the updated state.
    """
    builder = StateGraph(CandidateState)

    builder.add_node("greeting",           greeting_node)
    builder.add_node("info_gather",        info_gather_node)
    builder.add_node("tech_stack",         tech_stack_node)
    builder.add_node("generate_questions", generate_questions_node)
    builder.add_node("interview",          interview_node)
    builder.add_node("farewell",           farewell_node)

    # Entry point: dispatcher decides which node to run this turn
    builder.add_conditional_edges(
        START,
        _dispatcher,
        {
            "greeting":           "greeting",
            "info_gather":        "info_gather",
            "tech_stack":         "tech_stack",
            "generate_questions": "generate_questions",
            "interview":          "interview",
            "farewell":           "farewell",
        },
    )

    # greeting -> END (node updates stage to "info_gather")
    builder.add_edge("greeting", END)

    # info_gather -> END (node sets current_stage="tech_stack" when all fields collected)
    builder.add_edge("info_gather", END)

    # tech_stack: if stack parsed -> generate_questions (same turn); else -> END
    builder.add_conditional_edges(
        "tech_stack",
        _route_after_tech_stack,
        {
            "generate_questions": "generate_questions",
            "end":                END,
        },
    )

    # generate_questions -> END
    # (node presents Q1 and sets current_stage="interview"; user answers next turn)
    builder.add_edge("generate_questions", END)

    # interview: complete -> farewell; off-topic handled inline; otherwise -> END
    builder.add_conditional_edges(
        "interview",
        _route_after_interview,
        {
            "farewell": "farewell",
            "end":      END,
        },
    )

    builder.add_edge("farewell", END)

    # Compile WITHOUT a checkpointer — Streamlit manages all state persistence
    return builder.compile()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_graph = None


def get_graph():
    """Returns the compiled graph singleton."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

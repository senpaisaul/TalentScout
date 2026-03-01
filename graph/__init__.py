"""
graph/
------
LangGraph conversation graph modules for TalentScout.

Exposes the graph singleton and state utilities so app.py
only needs to import from the package, not individual submodules.

Usage:
    from graph import get_graph, initial_state, CandidateState
"""

from graph.graph import get_graph, build_graph
from graph.state import (
    CandidateState,
    initial_state,
    STAGES,
    REQUIRED_INFO_FIELDS,
    EXIT_KEYWORDS,
    get_missing_info_fields,
    all_info_collected,
    get_next_question,
    has_more_questions,
    exit_detected,
)

__all__ = [
    # Graph
    "get_graph",
    "build_graph",
    # State
    "CandidateState",
    "initial_state",
    "STAGES",
    "REQUIRED_INFO_FIELDS",
    "EXIT_KEYWORDS",
    "get_missing_info_fields",
    "all_info_collected",
    "get_next_question",
    "has_more_questions",
    "exit_detected",
]
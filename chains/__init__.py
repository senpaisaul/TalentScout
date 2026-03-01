"""
chains/
-------
LangChain 1.x chain modules for TalentScout.

Exposes one public function per chain so the rest of the codebase
imports from the package directly rather than from individual files.

Usage:
    from chains import run_info_chain, run_techq_chain, classify_sentiment
"""

from chains.info_chain       import run_info_chain
from chains.techstack_chain  import run_techstack_chain
from chains.techq_chain      import run_techq_chain, questions_per_tech
from chains.sentiment_chain  import classify_sentiment, get_sentiment_ui
from chains.fallback_chain   import run_fallback_chain

__all__ = [
    "run_info_chain",
    "run_techstack_chain",
    "run_techq_chain",
    "questions_per_tech",
    "classify_sentiment",
    "get_sentiment_ui",
    "run_fallback_chain",
]
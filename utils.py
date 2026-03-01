"""
utils.py
--------
Shared utility functions for TalentScout.

The most critical utility here is extract_json() — a robust JSON extractor
that handles LLM outputs which include preamble text before or after the JSON
block. This is needed because some models (including llama-4-scout) do not
reliably follow "respond ONLY with JSON" instructions and prepend conversational
text like "Sure! Here is the JSON:" before the actual JSON object.
"""

import json
import re
import logging

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict:
    """
    Robustly extracts a JSON object from an LLM response string.

    Handles all common model misbehaviours:
      - Preamble text before JSON:  "Here you go:\n{...}"
      - Postamble text after JSON:  "{...}\nLet me know if..."
      - Markdown fences:            "```json\n{...}\n```"
      - Nested braces in values:    handled by depth-tracking extraction

    Strategy:
      1. Strip markdown code fences if present
      2. Try direct json.loads() on the full text (fast path)
      3. Find the first '{' and last '}', extract that substring, try parse
      4. Use regex to find a JSON-like block and try parse
      5. Return empty dict as last resort (never raise)

    Args:
        text: Raw string content from an LLM response.

    Returns:
        Parsed dict, or {} if no valid JSON found.
    """
    if not text or not text.strip():
        return {}

    # Step 1: Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    # Step 2: Fast path — the whole thing might already be valid JSON
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Step 3: Depth-tracking brace extraction
    # Find the outermost complete {...} block
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(cleaned[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except (json.JSONDecodeError, ValueError):
                        break  # found braces but content invalid, fall through

    # Step 4: Regex sweep for JSON-like patterns
    pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(pattern, cleaned, re.DOTALL)
    for match in sorted(matches, key=len, reverse=True):  # try longest first
        try:
            return json.loads(match)
        except (json.JSONDecodeError, ValueError):
            continue

    # Step 5: Give up gracefully
    logger.warning(f"extract_json: could not parse JSON from: {text[:200]!r}")
    return {}


def extract_json_with_fallback(text: str, fallback: dict) -> dict:
    """
    Like extract_json() but merges the result with a fallback dict.
    Keys present in the parsed JSON override the fallback.
    Useful for chain return values where some fields are always required.
    """
    result = extract_json(text)
    return {**fallback, **result} if result else fallback
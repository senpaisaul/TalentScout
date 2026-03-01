"""
chains/techstack_chain.py
Uses OpenAI JSON mode (response_format=json_object) to guarantee valid JSON output.
"""
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import OPENAI_MODEL, OPENAI_API_KEY
from utils import extract_json_with_fallback

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.2,
    model_kwargs={"response_format": {"type": "json_object"}},
)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """\
You are Alex, a professional hiring assistant for TalentScout.
Your job is to collect and parse the candidate's tech stack.

If the candidate has listed technologies in their last message, extract them all.
Normalise names: JS->JavaScript, TS->TypeScript, K8s->Kubernetes.
Strip version numbers, split combined names like "React/Redux" into separate items.
Capitalise properly.

If the candidate has NOT listed any technologies yet, ask them to do so.

Respond in {language}.

OUTPUT: You MUST return a JSON object with exactly these keys:
- "response": your conversational message (string)
- "tech_stack": array of technology name strings extracted from the candidate's message, or [] if none found

Example when stack provided:
{{"response": "Great stack! I'll prepare questions on Python, FastAPI and PostgreSQL.", "tech_stack": ["Python", "FastAPI", "PostgreSQL"]}}

Example when no stack yet:
{{"response": "Could you list the languages, frameworks and tools you work with?", "tech_stack": []}}\
""",
    ),
    MessagesPlaceholder(variable_name="messages"),
])

_FALLBACK = {"response": "Could you list your technologies? e.g. Python, React, PostgreSQL, Docker.",
             "tech_stack": []}

def run_techstack_chain(messages, language="English"):
    try:
        formatted = _prompt.invoke({"language": language, "messages": messages})
        llm_response = _llm.invoke(formatted)
        result = extract_json_with_fallback(llm_response.content, _FALLBACK)
        logger.info(f"techstack_chain extracted: {result.get('tech_stack')}")

        # Deduplicate and clean
        stack = result.get("tech_stack", [])
        seen, clean = set(), []
        for t in (stack if isinstance(stack, list) else []):
            t = str(t).strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                clean.append(t)
        result["tech_stack"] = clean
        return result
    except Exception as e:
        logger.error(f"run_techstack_chain failed: {e}")
        return _FALLBACK

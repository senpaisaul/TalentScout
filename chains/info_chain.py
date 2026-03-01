"""
chains/info_chain.py
Uses OpenAI JSON mode (response_format=json_object) to guarantee valid JSON output.
"""
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import OPENAI_MODEL, OPENAI_API_KEY
from utils import extract_json_with_fallback

logger = logging.getLogger(__name__)

# response_format forces the model to always return valid JSON — no plain text leakage
_llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.3,
    model_kwargs={"response_format": {"type": "json_object"}},
)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """\
You are Alex, a warm and professional hiring assistant for TalentScout.
YOUR ONLY PURPOSE is to collect candidate information for tech job screening.
Do NOT answer off-topic questions or deviate from this purpose.

FIELDS STILL NEEDED (collect in order, one or two at a time):
{missing_fields}

ALREADY COLLECTED:
{collected_data}

RULES:
- Be conversational — confirm and collect one or two fields at a time.
- VALIDATION: email must contain @, phone must contain digits only, years_of_experience must be a number.
- If the candidate confirms a value, mark it as extracted immediately.
- If validation fails, ask again with a helpful hint.
- Do NOT ask for confirmation twice for the same field. If the candidate already confirmed, extract it.
- Once ALL fields are collected, tell the candidate and ask for their tech stack.

Respond in {language}.

OUTPUT: You MUST return a JSON object with exactly these keys:
- "response": your conversational message to the candidate (string)
- "extracted": an object containing any newly confirmed field values, e.g. {{"email": "x@y.com", "phone": "1234567890"}}
- "validation_failed": true only if a field failed validation, otherwise false

Example when email confirmed:
{{"response": "Got it! Now could you share your phone number?", "extracted": {{"email": "x@y.com"}}, "validation_failed": false}}

Example when asking for next field:
{{"response": "What is your current location?", "extracted": {{}}, "validation_failed": false}}\
""",
    ),
    MessagesPlaceholder(variable_name="messages"),
])

_FALLBACK = {"response": "I'm sorry, I had a small hiccup. Could you repeat that?",
             "extracted": {}, "validation_failed": False}

def run_info_chain(missing_fields, collected_data, messages, language="English"):
    try:
        formatted = _prompt.invoke({
            "missing_fields": ", ".join(missing_fields) if missing_fields else "None — all collected",
            "collected_data": json.dumps(collected_data, indent=2),
            "language": language,
            "messages": messages,
        })
        llm_response = _llm.invoke(formatted)
        result = extract_json_with_fallback(llm_response.content, _FALLBACK)
        logger.info(f"info_chain extracted: {result.get('extracted')}")
        return result
    except Exception as e:
        logger.error(f"run_info_chain failed: {e}")
        return _FALLBACK

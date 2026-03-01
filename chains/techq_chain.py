"""
chains/techq_chain.py
Uses OpenAI JSON mode (response_format=json_object) to guarantee valid JSON output.
"""
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from config import OPENAI_MODEL, OPENAI_API_KEY
from utils import extract_json

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.65,
    model_kwargs={"response_format": {"type": "json_object"}},
)

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """\
You are a senior technical interviewer at TalentScout.

CANDIDATE:
  Name: {full_name} | Experience: {years_of_experience} years | Role: {desired_positions}
  Tech stack: {tech_stack}

Generate exactly {questions_per_tech} technical interview questions PER technology listed.
Calibrate difficulty by experience:
  0-2 yrs: conceptual, syntax, basic patterns
  3-5 yrs: design tradeoffs, debugging, production concerns
  6-9 yrs: architecture, scalability, deep internals
  10+ yrs: system design, cross-team impact

Rules: scenario-based, open-ended, no trivia, no yes/no questions.

OUTPUT: Return a JSON object where each key is a technology name and the value is an array of question strings.
Example:
{{"Python": ["q1", "q2", "q3"], "FastAPI": ["q1", "q2", "q3"]}}\
""",
    ),
    ("human", "Generate the interview questions now."),
])

def questions_per_tech(yoe):
    return 3 if yoe <= 2 else (4 if yoe <= 6 else 5)

def run_techq_chain(full_name, years_of_experience, desired_positions, tech_stack):
    yoe = years_of_experience or 0
    n = questions_per_tech(yoe)
    try:
        formatted = _prompt.invoke({
            "full_name": full_name or "the candidate",
            "years_of_experience": yoe,
            "desired_positions": ", ".join(desired_positions or ["Software Engineer"]),
            "tech_stack": ", ".join(tech_stack or []),
            "questions_per_tech": n,
        })
        llm_response = _llm.invoke(formatted)
        result = extract_json(llm_response.content)
        validated = {}
        for tech in tech_stack:
            qs = result.get(tech) or next(
                (v for k, v in result.items() if k.lower() == tech.lower() and isinstance(v, list)), []
            )
            if not qs:
                qs = [f"Walk me through a real-world project where you used {tech}. What challenges did you face?"]
            validated[tech] = qs[:n]
        return validated
    except Exception as e:
        logger.error(f"run_techq_chain failed: {e}")
        return {t: [f"Describe a challenging problem you solved using {t}."] for t in tech_stack}

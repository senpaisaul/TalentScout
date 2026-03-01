"""
graph/nodes.py
--------------
All LangGraph node functions for TalentScout.
Switched from Groq to OpenAI (langchain_openai).
"""

import logging
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from graph.state import (
    CandidateState,
    REQUIRED_INFO_FIELDS,
    get_missing_info_fields,
    all_info_collected,
    get_next_question,
    has_more_questions,
)
from config import OPENAI_MODEL, OPENAI_API_KEY
from chains.info_chain      import run_info_chain
from chains.techstack_chain import run_techstack_chain
from chains.techq_chain     import run_techq_chain
from chains.fallback_chain  import run_fallback_chain
from utils import extract_json

logger = logging.getLogger(__name__)

# LLM for plain text output (greeting translation)
llm_plain = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.3)

# LLM for interview acknowledgement — must return JSON
llm_precise = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.3,
    model_kwargs={"response_format": {"type": "json_object"}},
)


# ===========================================================================
# NODE 1: GREETING
# ===========================================================================
def greeting_node(state: CandidateState) -> dict:
    """Scripted greeting — minimal LLM usage (only if translation needed)."""
    language = state.get("language", "English")

    greeting_text = (
        "👋 Hello and welcome to **TalentScout**!\n\n"
        "I'm **Alex**, your AI hiring assistant. I'm here to help with your "
        "initial screening for our technology placement roles.\n\n"
        "Here's what we'll cover together:\n"
        "1. **Your background** — a few quick details about you\n"
        "2. **Your tech stack** — the languages, frameworks & tools you work with\n"
        "3. **Technical questions** — tailored specifically to your stack\n\n"
        "The whole process takes about **10–15 minutes**. "
        "You can type `exit` or `bye` at any time to end the session.\n\n"
        "Ready? Let's start — **what's your full name?**"
    )

    if language != "English":
        translate_prompt = ChatPromptTemplate.from_messages([
            ("system", f"Translate the following greeting into {language}. Keep all formatting and emojis. Output only the translation."),
            ("human", "{text}")
        ])
        formatted     = translate_prompt.invoke({"text": greeting_text})
        llm_response  = llm_plain.invoke(formatted)
        greeting_text = StrOutputParser().invoke(llm_response)

    return {
        "messages":      [AIMessage(content=greeting_text)],
        "current_stage": "info_gather",
    }


# ===========================================================================
# NODE 2: INFO GATHER
# ===========================================================================
def info_gather_node(state: CandidateState) -> dict:
    """Conversationally collects all 6 required candidate fields."""
    missing   = get_missing_info_fields(state)
    collected = {f: state.get(f) for f in REQUIRED_INFO_FIELDS
                 if f not in missing and state.get(f) is not None}

    result        = run_info_chain(
        missing_fields=missing,
        collected_data=collected,
        messages=state["messages"],
        language=state.get("language", "English"),
    )
    response_text = result.get("response", "")
    extracted     = result.get("extracted", {}) or {}
    updates       = {"messages": [AIMessage(content=response_text)]}
    fields_done   = list(state.get("info_fields_collected", []))

    for field, value in extracted.items():
        if value is None or value == "null" or field not in REQUIRED_INFO_FIELDS:
            continue
        if field == "years_of_experience":
            try:
                value = int(str(value).split()[0])
            except (ValueError, IndexError):
                continue
        elif field == "desired_positions" and isinstance(value, str):
            value = [v.strip() for v in value.replace(" or ", ",").split(",") if v.strip()]
        updates[field] = value
        if field not in fields_done:
            fields_done.append(field)

    updates["info_fields_collected"] = fields_done
    if set(REQUIRED_INFO_FIELDS).issubset(set(fields_done)):
        updates["current_stage"] = "tech_stack"
    return updates


# ===========================================================================
# NODE 3: TECH STACK
# ===========================================================================
def tech_stack_node(state: CandidateState) -> dict:
    """Elicits and parses the candidate's declared tech stack."""
    result     = run_techstack_chain(
        messages=state["messages"],
        language=state.get("language", "English"),
    )
    tech_stack = list(dict.fromkeys(
        t.strip() for t in (result.get("tech_stack") or [])
        if isinstance(t, str) and t.strip()
    ))
    updates = {"messages": [AIMessage(content=result.get("response", ""))]}
    if tech_stack:
        updates["tech_stack"]    = tech_stack
        updates["current_stage"] = "generate_questions"
    return updates


# ===========================================================================
# NODE 4: GENERATE QUESTIONS
# ===========================================================================
def generate_questions_node(state: CandidateState) -> dict:
    """One-shot question generation; flattens into sequential queue."""
    yoe            = state.get("years_of_experience", 0) or 0
    tech_questions = run_techq_chain(
        full_name=state.get("full_name", "the candidate"),
        years_of_experience=yoe,
        desired_positions=state.get("desired_positions") or [],
        tech_stack=state.get("tech_stack") or [],
    )

    question_queue = [
        {"tech": tech, "q": q}
        for tech, questions in tech_questions.items()
        for q in questions
    ]

    if not question_queue:
        return {
            "messages":      [AIMessage(content="I had trouble generating questions. Could you re-list your tech stack?")],
            "current_stage": "tech_stack",
        }

    tech_list      = ", ".join(tech_questions.keys())
    transition_msg = (
        f"I've prepared your technical questions covering: **{tech_list}**.\n\n"
        f"I'll ask you {len(question_queue)} questions total. Take your time — no trick questions here.\n\n"
        f"Let's begin! 🎯\n\n"
        f"**Question 1 of {len(question_queue)} — {question_queue[0]['tech']}:**\n\n"
        f"{question_queue[0]['q']}"
    )

    return {
        "messages":               [AIMessage(content=transition_msg)],
        "tech_questions":         tech_questions,
        "question_queue":         question_queue,
        "current_question_index": 1,
        "current_stage":          "interview",
    }


# ===========================================================================
# NODE 5: INTERVIEW
# ===========================================================================
_INTERVIEW_SYSTEM_PROMPT = """
You are Alex, a professional but encouraging technical interviewer at TalentScout.

CANDIDATE: {full_name} | {years_of_experience} years experience
CURRENT QUESTION ({question_num} of {total_questions}):
Technology: {current_tech}
Question: "{current_question}"

CANDIDATE'S ANSWER: "{candidate_answer}"

YOUR TASK:
1. Acknowledge their answer in 1-2 sentences. Be encouraging but honest.
   - Strong answer: validate specifically.
   - Incomplete/wrong: note what was good, mention what was missing. Do NOT give the correct answer.
2. Do NOT ask follow-ups on the same topic.
3. If the candidate message is clearly unrelated to the question, set is_off_topic to true.

Respond in {language}.

OUTPUT FORMAT: Return a JSON object with these keys:
- "response": your acknowledgement string
- "is_off_topic": boolean

Example JSON: {{"response": "Nice answer!", "is_off_topic": false}}"""

def interview_node(state: CandidateState) -> dict:
    """Core interview loop — acknowledges answer, records it, presents next question."""
    queue = state.get("question_queue") or []
    idx   = state.get("current_question_index", 0)

    if not queue:
        return {
            "messages":    [AIMessage(content="It looks like we've covered everything!")],
            "is_complete": True,
        }

    prev_idx      = max(0, idx - 1)
    current_q     = queue[prev_idx] if prev_idx < len(queue) else queue[-1]
    candidate_ans = ""
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "human":
            candidate_ans = msg.content
            break
        if isinstance(msg, dict) and msg.get("role") == "human":
            candidate_ans = msg.get("content", "")
            break

    prompt = ChatPromptTemplate.from_messages([
        ("system", _INTERVIEW_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])
    try:
        formatted    = prompt.invoke({
            "full_name":           state.get("full_name", "there"),
            "years_of_experience": state.get("years_of_experience", 0),
            "question_num":        prev_idx + 1,
            "total_questions":     len(queue),
            "current_tech":        current_q["tech"],
            "current_question":    current_q["q"],
            "candidate_answer":    candidate_ans,
            "language":            state.get("language", "English"),
            "messages":            state["messages"],
        })
        llm_response = llm_precise.invoke(formatted)
        result       = extract_json(llm_response.content)
        if not result:
            result = {"response": "Thank you for that answer. Let's continue.", "is_off_topic": False}
    except Exception as e:
        logger.error(f"interview_node LLM error: {e}")
        result = {"response": "Thank you for that answer. Let's continue.", "is_off_topic": False}

    is_off_topic = result.get("is_off_topic", False)

    # Off-topic: redirect inline, don't advance
    if is_off_topic:
        fallback_text = run_fallback_chain(
            candidate_message=candidate_ans,
            current_tech=current_q["tech"],
            current_question=current_q["q"],
            language=state.get("language", "English"),
        )
        return {"messages": [AIMessage(content=fallback_text)]}

    # On-topic: record answer, advance
    answers = list(state.get("candidate_answers") or [])
    answers.append({
        "tech":     current_q["tech"],
        "question": current_q["q"],
        "answer":   candidate_ans,
    })

    response_text      = result.get("response", "")
    next_idx           = idx
    interview_complete = False

    if next_idx < len(queue):
        next_q         = queue[next_idx]
        response_text += (
            f"\n\n**Question {next_idx + 1} of {len(queue)} — {next_q['tech']}:**\n\n"
            f"{next_q['q']}"
        )
        next_idx += 1
    else:
        interview_complete = True

    return {
        "messages":               [AIMessage(content=response_text)],
        "current_question_index": next_idx,
        "candidate_answers":      answers,
        "is_complete":            interview_complete,
        "current_stage":          "farewell" if interview_complete else "interview",
    }


# ===========================================================================
# NODE 6: FALLBACK
# ===========================================================================
def fallback_node(state: CandidateState) -> dict:
    """Redirects off-topic messages back to the current interview question."""
    queue     = state.get("question_queue") or []
    idx       = max(0, state.get("current_question_index", 1) - 1)
    current_q = queue[idx] if idx < len(queue) else {"tech": "General", "q": "your answer"}

    candidate_message = ""
    for msg in reversed(state.get("messages", [])):
        if hasattr(msg, "type") and msg.type == "human":
            candidate_message = msg.content
            break
        if isinstance(msg, dict) and msg.get("role") == "human":
            candidate_message = msg.get("content", "")
            break

    response_text = run_fallback_chain(
        candidate_message=candidate_message,
        current_tech=current_q["tech"],
        current_question=current_q["q"],
        language=state.get("language", "English"),
    )
    return {"messages": [AIMessage(content=response_text)]}


# ===========================================================================
# NODE 7: FAREWELL
# ===========================================================================
def farewell_node(state: CandidateState) -> dict:
    """Terminal node — personalised summary and next steps."""
    name       = state.get("full_name") or "there"
    answers    = state.get("candidate_answers") or []
    tech_stack = state.get("tech_stack") or []
    exit_early = state.get("exit_triggered", False)

    if exit_early or not answers:
        text = (
            f"No problem, {name}! Thanks for your time today. 👋\n\n"
            f"If you'd like to complete the screening in the future, just start a new session.\n\n"
            f"Best of luck with your job search! 🌟"
        )
    else:
        techs = list(dict.fromkeys(a["tech"] for a in answers))
        text  = (
            f"That wraps up your screening, **{name}**! 🎉\n\n"
            f"**Summary:**\n"
            f"- Technologies assessed: {', '.join(techs)}\n"
            f"- Questions answered: {len(answers)}\n\n"
            f"**What happens next:**\n"
            f"1. Our team will review your responses within **2 business days**\n"
            f"2. If your profile matches a role, a recruiter will reach out\n"
            f"3. Watch your email for updates\n\n"
            f"We're impressed by your background in {', '.join(tech_stack[:2])}! Good luck 🚀"
        )

    return {
        "messages":      [AIMessage(content=text)],
        "is_complete":   True,
        "current_stage": "farewell",
    }

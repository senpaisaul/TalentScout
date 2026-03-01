"""
chains/fallback_chain.py — LangChain 1.x, StrOutputParser
"""
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import OPENAI_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)

_llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0.4)
_parser = StrOutputParser()

_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are Alex, a professional hiring assistant for TalentScout. "
     "The candidate went off-topic. In 3-4 sentences: acknowledge briefly, "
     "explain you're focused on technical screening, restate the current question. "
     "Respond in {language}."),
    ("human",
     'Candidate said: "{candidate_message}"\n\n'
     'Current question [{current_tech}]: "{current_question}"\n\nRedirect them.'),
])

def run_fallback_chain(candidate_message, current_tech, current_question, language="English"):
    try:
        formatted = _prompt.invoke({
            "candidate_message": candidate_message or "(no message)",
            "current_tech": current_tech,
            "current_question": current_question,
            "language": language,
        })
        llm_response = _llm.invoke(formatted)
        return _parser.invoke(llm_response)
    except Exception as e:
        logger.error(f"run_fallback_chain failed: {e}")
        return (f"I appreciate your message! Let's get back to the **{current_tech}** question:\n\n"
                f"*{current_question}*")

"""
chains/sentiment_chain.py — LangChain 1.x, StrOutputParser
"""
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import OPENAI_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)

VALID_SENTIMENTS = {"positive", "neutral", "negative"}

_llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, temperature=0)
_parser = StrOutputParser()

_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Classify sentiment as exactly one word: positive, neutral, or negative.\n"
     "positive=enthusiastic/engaged, neutral=factual/brief, negative=frustrated/confused.\n"
     "Reply with ONLY that single word."),
    ("human", "{message}"),
])

def classify_sentiment(message: str) -> str:
    if not message or not message.strip():
        return "neutral"
    try:
        formatted = _prompt.invoke({"message": message})
        llm_response = _llm.invoke(formatted)
        result = _parser.invoke(llm_response).strip().lower().split()[0]
        return result if result in VALID_SENTIMENTS else "neutral"
    except Exception as e:
        logger.error(f"classify_sentiment failed: {e}")
        return "neutral"

SENTIMENT_UI = {
    "positive": {"emoji": "😊", "color": "#22c55e", "label": "Positive"},
    "neutral":  {"emoji": "😐", "color": "#f59e0b", "label": "Neutral"},
    "negative": {"emoji": "😟", "color": "#ef4444", "label": "Needs Support"},
}

def get_sentiment_ui(sentiment: str) -> dict:
    return SENTIMENT_UI.get(sentiment, SENTIMENT_UI["neutral"])

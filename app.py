"""
app.py
------
TalentScout Hiring Assistant — Streamlit frontend.

Wires the LangGraph conversation graph to Streamlit's chat UI.
Key responsibilities:
  - Session initialisation (UUID, initial state, graph singleton)
  - Rendering the chat history from LangGraph state messages
  - Handling user input and invoking the graph
  - Running sentiment analysis asynchronously after each message
  - Persisting state to SQLite via DataHandler after every turn
  - Sidebar: progress tracker, candidate profile, sentiment, language selector
  - Footer: session download + privacy notice

Run with:
    streamlit run app.py

FIX (v2):
  - _invoke_graph now passes the FULL message history (current_state["messages"]
    + new HumanMessage) instead of just [new_message], preventing history loss.
  - Removed graph_config / checkpointer config (MemorySaver removed from graph).
  - Added exc_info=True to error logging for better diagnostics.
"""

import uuid
import logging
import threading
from datetime import datetime

# Thread-safe container for sentiment result (avoids writing to st.session_state
# from a background thread, which causes the ScriptRunContext warning)
_sentiment_result: dict = {"value": "neutral"}

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from graph.graph import get_graph
from graph.state import initial_state, STAGES, REQUIRED_INFO_FIELDS
from data_handler import db
from chains.sentiment_chain import classify_sentiment, get_sentiment_ui

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TalentScout | AI Hiring Assistant",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — professional dark-accent theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  /* ── Global font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* ── Top brand bar ── */
  .brand-bar {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }
  .brand-bar h1 {
    color: white;
    margin: 0;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.5px;
  }
  .brand-bar p {
    color: rgba(255,255,255,0.75);
    margin: 0;
    font-size: 0.85rem;
  }

  /* ── Chat messages ── */
  .stChatMessage {
    border-radius: 10px;
    margin-bottom: 0.4rem;
    padding: 0.1rem 0.5rem;
  }
  [data-testid="stChatMessageContent"] p { line-height: 1.7; }

  /* ── Chat input ── */
  .stChatInput > div {
    border: 2px solid #2563eb !important;
    border-radius: 10px !important;
  }
  .stChatInput > div:focus-within {
    box-shadow: 0 0 0 3px rgba(37,99,235,0.2) !important;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #e2e8f0;
  }
  .sidebar-section {
    background: white;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.85rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }
  .sidebar-section h4 {
    margin: 0 0 0.6rem 0;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #64748b;
  }

  /* ── Progress steps ── */
  .step-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0;
    font-size: 0.88rem;
  }
  .step-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .step-done  .step-dot { background: #22c55e; }
  .step-active .step-dot { background: #2563eb; box-shadow: 0 0 0 3px rgba(37,99,235,0.2); }
  .step-pending .step-dot { background: #cbd5e1; }
  .step-done   span { color: #15803d; font-weight: 500; }
  .step-active span { color: #1d4ed8; font-weight: 600; }
  .step-pending span { color: #94a3b8; }

  /* ── Profile pills ── */
  .profile-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    margin-top: 0.4rem;
  }
  .pill {
    background: #eff6ff;
    color: #1e40af;
    border-radius: 20px;
    padding: 0.2rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 500;
    white-space: nowrap;
  }
  .pill-tech {
    background: #f0fdf4;
    color: #166534;
  }

  /* ── Sentiment badge ── */
  .sentiment-badge {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.75rem;
    border-radius: 8px;
    font-size: 0.85rem;
    font-weight: 500;
    margin-top: 0.25rem;
  }

  /* ── Disabled input overlay ── */
  .done-banner {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
    color: #166534;
    font-weight: 500;
    font-size: 0.9rem;
    margin-top: 0.5rem;
  }

  /* ── Privacy notice ── */
  .privacy-notice {
    font-size: 0.72rem;
    color: #94a3b8;
    text-align: center;
    padding: 0.5rem;
    line-height: 1.5;
  }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Stage display config
# ---------------------------------------------------------------------------
STAGE_LABELS = {
    "greeting":           "Welcome",
    "info_gather":        "Background",
    "tech_stack":         "Tech Stack",
    "generate_questions": "Preparing Questions",
    "interview":          "Technical Interview",
    "farewell":           "Complete",
}

SUPPORTED_LANGUAGES = [
    "English", "Spanish", "French", "German",
    "Portuguese", "Hindi", "Japanese", "Chinese (Simplified)",
]


# ---------------------------------------------------------------------------
# Session initialisation
# ---------------------------------------------------------------------------
def _init_session():
    """
    Initialises Streamlit session_state on first load.
    Subsequent reruns skip this entirely — Streamlit preserves session_state.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "language" not in st.session_state:
        st.session_state.language = "English"

    if "graph_state" not in st.session_state:
        st.session_state.graph_state = initial_state(
            session_id=st.session_state.session_id,
            language=st.session_state.language,
        )

    if "greeting_done" not in st.session_state:
        st.session_state.greeting_done = False

    if "sentiment" not in st.session_state:
        st.session_state.sentiment = "neutral"


# ---------------------------------------------------------------------------
# Graph invocation
# ---------------------------------------------------------------------------
def _invoke_graph(user_message: str | None = None) -> None:
    """
    Invokes the LangGraph graph with an optional user message.
    - If user_message is None: fires the greeting node (session start).
    - Otherwise: appends the human message to history and advances the graph.

    FIX: passes the FULL message history (existing + new message) to graph.invoke().
    Previously `"messages": [HumanMessage(...)]` was passed, discarding all history.
    Without a checkpointer the graph has no stored history, so the full list must
    be supplied on every call.

    Updates st.session_state.graph_state with the new state.
    Persists to DB after every turn.
    """
    graph         = get_graph()
    current_state = st.session_state.graph_state

    if user_message is not None:
        # Build full message history including the new human message
        existing_messages = current_state.get("messages", [])
        input_state = {
            **current_state,
            "messages": existing_messages + [HumanMessage(content=user_message)],
        }
        # Sentiment analysis in the background (non-blocking).
        # Writes to a plain dict — NOT st.session_state — to avoid the
        # "missing ScriptRunContext" warning that occurs when background
        # threads touch Streamlit state directly.
        def _run_sentiment():
            _sentiment_result["value"] = classify_sentiment(user_message)
        threading.Thread(target=_run_sentiment, daemon=True).start()
    else:
        # Greeting turn — pass state as-is (no user message yet)
        input_state = current_state

    try:
        new_state = graph.invoke(input_state)
        st.session_state.graph_state = new_state

        # Persist to DB
        db.save_session(new_state)
        answers = new_state.get("candidate_answers") or []
        if answers:
            db.save_answers(st.session_state.session_id, answers)

    except Exception as e:
        logger.error(f"Graph invocation error: {e}", exc_info=True)
        st.error("Something went wrong. Please try again or refresh the page.")


# ---------------------------------------------------------------------------
# Sidebar rendering
# ---------------------------------------------------------------------------
def _render_sidebar():
    state = st.session_state.graph_state

    with st.sidebar:
        # ── Brand ──
        st.markdown("""
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <span style="font-size:2rem;">🎯</span>
            <h2 style="margin:0.25rem 0 0 0; font-size:1.2rem; font-weight:700; color:#1e3a5f;">
                TalentScout
            </h2>
            <p style="color:#64748b; font-size:0.78rem; margin:0;">AI Hiring Assistant</p>
        </div>
        """, unsafe_allow_html=True)

        # ── Language selector ──
        with st.container():
            st.markdown('<div class="sidebar-section"><h4>🌐 Language</h4>', unsafe_allow_html=True)
            selected_lang = st.selectbox(
                label="Language",
                options=SUPPORTED_LANGUAGES,
                index=SUPPORTED_LANGUAGES.index(
                    st.session_state.get("language", "English")
                ),
                label_visibility="collapsed",
                key="lang_selector",
            )
            if selected_lang != st.session_state.language and not st.session_state.greeting_done:
                st.session_state.language = selected_lang
                st.session_state.graph_state["language"] = selected_lang
            elif selected_lang != st.session_state.language:
                st.caption("Language can only be changed before the session starts.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Progress tracker ──
        current_stage = state.get("current_stage", "greeting")
        stage_order   = list(STAGE_LABELS.keys())
        current_idx   = stage_order.index(current_stage) if current_stage in stage_order else 0

        st.markdown('<div class="sidebar-section"><h4>📋 Progress</h4>', unsafe_allow_html=True)
        for i, (stage_key, stage_label) in enumerate(STAGE_LABELS.items()):
            if stage_key == "generate_questions":
                continue  # internal step, don't show to candidate
            if i < current_idx:
                css_class = "step-done"
            elif stage_key == current_stage:
                css_class = "step-active"
            else:
                css_class = "step-pending"
            st.markdown(
                f'<div class="step-row {css_class}">'
                f'<div class="step-dot"></div>'
                f'<span>{stage_label}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Interview progress bar
        if current_stage == "interview":
            queue = state.get("question_queue") or []
            idx   = state.get("current_question_index", 0)
            total = len(queue)
            if total > 0:
                pct = min(idx / total, 1.0)
                st.progress(pct, text=f"Question {min(idx, total)}/{total}")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Candidate profile ──
        has_any = any([
            state.get("full_name"), state.get("email"),
            state.get("years_of_experience"), state.get("desired_positions"),
            state.get("current_location"), state.get("tech_stack"),
        ])
        if has_any:
            st.markdown('<div class="sidebar-section"><h4>👤 Candidate Profile</h4>', unsafe_allow_html=True)
            if state.get("full_name"):
                st.markdown(f"**{state['full_name']}**")
            if state.get("current_location"):
                st.caption(f"📍 {state['current_location']}")
            if state.get("years_of_experience") is not None:
                yoe   = state["years_of_experience"]
                level = "Junior" if yoe <= 2 else "Mid-level" if yoe <= 6 else "Senior" if yoe <= 10 else "Staff+"
                st.caption(f"💼 {yoe} yrs experience · {level}")
            if state.get("desired_positions"):
                positions = state["desired_positions"]
                pills     = "".join(f'<span class="pill">{p}</span>' for p in positions)
                st.markdown(f'<div class="profile-row">{pills}</div>', unsafe_allow_html=True)
            if state.get("tech_stack"):
                st.markdown("<br>**Tech Stack:**", unsafe_allow_html=True)
                tech_pills = "".join(
                    f'<span class="pill pill-tech">{t}</span>'
                    for t in state["tech_stack"]
                )
                st.markdown(f'<div class="profile-row">{tech_pills}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Sentiment indicator ──
        # Read from the module-level dict (updated by background thread).
        # Fall back to session_state for the initial render.
        sentiment_key = _sentiment_result.get("value", st.session_state.get("sentiment", "neutral"))
        st.session_state.sentiment = sentiment_key  # keep session_state in sync for other uses
        ui            = get_sentiment_ui(sentiment_key)
        st.markdown(
            f'<div class="sidebar-section"><h4>🧠 Candidate Sentiment</h4>'
            f'<div class="sentiment-badge" style="background:{ui["color"]}22; color:{ui["color"]};">'
            f'{ui["emoji"]} {ui["label"]}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # ── Session info ──
        st.markdown(
            f'<div class="sidebar-section"><h4>🔖 Session</h4>'
            f'<code style="font-size:0.7rem; color:#64748b;">{st.session_state.session_id[:18]}…</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Download button (shown after completion) ──
        if state.get("is_complete"):
            export_json = db.export_session_json(st.session_state.session_id)
            if export_json:
                st.download_button(
                    label="⬇️ Download Session Report",
                    data=export_json,
                    file_name=f"talentscout_{st.session_state.session_id[:8]}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        # ── Privacy notice ──
        st.markdown(
            '<div class="privacy-notice">'
            '🔒 Your data is stored locally and used only for '
            'this screening session. You may request deletion at any time.'
            '</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Chat history rendering
# ---------------------------------------------------------------------------
def _render_chat_history():
    """Renders all messages from LangGraph state to Streamlit chat UI."""
    messages = st.session_state.graph_state.get("messages", [])
    for msg in messages:
        if hasattr(msg, "type"):
            role    = "assistant" if msg.type == "ai" else "user"
            content = msg.content
        elif isinstance(msg, dict):
            role    = "assistant" if msg.get("role") in ("ai", "assistant") else "user"
            content = msg.get("content", "")
        else:
            continue

        with st.chat_message(role, avatar="🎯" if role == "assistant" else "👤"):
            st.markdown(content)


# ---------------------------------------------------------------------------
# Main app layout
# ---------------------------------------------------------------------------
def main():
    _init_session()

    # ── Brand header ──
    st.markdown("""
    <div class="brand-bar">
      <div style="font-size:1.8rem; line-height:1;">🎯</div>
      <div>
        <h1>TalentScout</h1>
        <p>AI-Powered Technical Hiring Assistant</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    _render_sidebar()

    # ── Fire greeting on first load ──
    if not st.session_state.greeting_done:
        with st.spinner(""):
            _invoke_graph(user_message=None)
        st.session_state.greeting_done = True
        st.rerun()

    # ── Chat history ──
    _render_chat_history()

    # ── Input area ──
    is_complete = st.session_state.graph_state.get("is_complete", False)

    if is_complete:
        st.markdown(
            '<div class="done-banner">'
            '✅ Screening complete! Download your session report from the sidebar.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        if user_input := st.chat_input(
            placeholder="Type your response here…",
            key="chat_input",
        ):
            # Show user message immediately (optimistic render)
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)

            # Invoke graph and get response
            with st.chat_message("assistant", avatar="🎯"):
                with st.spinner("Alex is typing…"):
                    _invoke_graph(user_message=user_input)

            st.rerun()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()

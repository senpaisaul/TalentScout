# 🎯 TalentScout — AI Hiring Assistant

> An intelligent chatbot that conducts initial technical screening interviews for technology placement candidates, powered by LangGraph, LangChain 1.x, and OpenAI's `gpt-4o-mini`.

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#LiveDemo)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Prompt Design](#prompt-design)
- [Data Privacy](#data-privacy)
- [Technical Details](#technical-details)
- [Challenges & Solutions](#challenges--solutions)

---

## Overview

TalentScout is a conversational AI hiring assistant built for **TalentScout**, a fictional technology recruitment agency. It automates the initial candidate screening process by:

1. Collecting essential candidate information conversationally (name, contact, experience, location, desired roles)
2. Prompting the candidate to declare their tech stack
3. Generating 3–5 calibrated technical interview questions **per technology**, tailored to the candidate's years of experience
4. Conducting the interview, acknowledging answers, and gracefully handling off-topic responses
5. Concluding with a personalised summary and next-steps message

The entire conversation is orchestrated as a **stateful graph** using LangGraph — each stage is an explicit node with typed state, making the flow inspectable, testable, and easy to extend.

---

## Live Demo: [talent-scout-v1.streamlit.app](https://talent-scout-v1.streamlit.app/)**

---

## Features

**Core**
- Conversational info gathering — never asks for all fields at once like a form
- Experience-calibrated questions — junior candidates get conceptual questions; senior candidates get architecture and system design questions
- Sequential question delivery with sidebar progress tracking
- Exit keyword detection at any stage (`exit`, `bye`, `quit`, etc.)
- Graceful fallback for off-topic or unintelligible responses — handled inline without a separate routing step
- Multilingual support — respond in Spanish, French, German, Hindi, Japanese, and more

**Bonus**
- Real-time sentiment analysis — sidebar mood indicator updates after every candidate message
- Session export — download the full screening transcript as JSON after completion
- GDPR-compliant data handling — PII masked in storage, right-to-erasure support
- Persistent SQLite storage — sessions survive page refreshes

---

## Architecture

TalentScout is built on three layers:

```
┌─────────────────────────────────────┐
│           Streamlit UI (app.py)     │  ← Chat interface, sidebar, session management
├─────────────────────────────────────┤
│        LangGraph State Graph        │  ← Conversation flow orchestration
│  START → dispatcher → greeting     │
│  → info_gather → tech_stack        │
│  → generate_questions → interview  │
│  → farewell → END                  │
├─────────────────────────────────────┤
│       LangChain 1.x Chains          │  ← One chain per concern, OpenAI LLM backend
│  info · techstack · techq ·         │
│  sentiment · fallback               │
└─────────────────────────────────────┘
```

### Conversation Graph — Dispatcher Pattern

Each call to `graph.invoke()` executes **exactly one turn** (one node). The `_dispatcher` function at `START` reads `current_stage` from the state and routes to the appropriate node. No node loops back to itself within a single invoke — the Streamlit app controls the turn loop.

```
Per .invoke() call:

START
  └─► _dispatcher (reads current_stage + checks exit keywords)
        ├─► greeting           → END   (sets stage: info_gather)
        ├─► info_gather        → END   (sets stage: tech_stack when all 6 fields done)
        ├─► tech_stack         ──────► generate_questions → END   (if stack extracted)
        │                      → END   (if stack not yet provided, re-ask next turn)
        ├─► interview          ──────► farewell → END   (if complete or exit)
        │                      → END   (normal turn, wait for next answer)
        └─► farewell           → END   (terminal)
```

State flows through the graph as a typed `CandidateState` TypedDict. Streamlit's `session_state` persists the full state between turns — no LangGraph checkpointer is used.

---

## Project Structure

```
talentscout/
│
├── app.py                    # Streamlit entry point — run this
├── config.py                 # Loads OPENAI_API_KEY and OPENAI_MODEL from .env
├── data_handler.py           # SQLite storage, PII masking, GDPR erasure
├── requirements.txt
├── .env                      # Add your OPENAI_API_KEY here (never commit this)
│
├── chains/                   # LangChain 1.x chain modules
│   ├── __init__.py
│   ├── info_chain.py         # Conversational info gathering + JSON extraction
│   ├── techstack_chain.py    # Tech stack elicitation + normalisation
│   ├── techq_chain.py        # Technical question generation
│   ├── sentiment_chain.py    # Sentiment classification (positive/neutral/negative)
│   └── fallback_chain.py     # Off-topic redirect responses
│
└── graph/                    # LangGraph modules
    ├── __init__.py
    ├── state.py              # CandidateState TypedDict + helpers + initial_state()
    ├── nodes.py              # All 7 node functions
    ├── edges.py              # Legacy routing helpers (documentation only)
    └── graph.py              # Dispatcher graph assembly + get_graph() singleton
```

---

## Installation

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys)

### Steps

```bash
# 1. Clone or unzip the project
cd talentscout

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
# Open .env and add your OpenAI API key:
#   OPENAI_API_KEY=sk-proj-...
#   OPENAI_MODEL=gpt-4o-mini

# 5. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | — | Your OpenAI API key from platform.openai.com |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | OpenAI model to use. `gpt-4o` gives higher quality responses. |

---

## Usage

1. **Start the app** — Alex greets you and asks for your full name
2. **Provide your details** — answer each question conversationally; Alex collects all 6 fields naturally over several turns
3. **Declare your tech stack** — list your languages, frameworks, databases, and tools
4. **Answer technical questions** — Alex asks 3–5 questions per technology, calibrated to your experience level
5. **Completion** — Alex summarises the session and explains next steps; download your report from the sidebar

**At any point**, type `exit`, `bye`, or `quit` to end the session gracefully.

**Language** — select your preferred language from the sidebar dropdown before the session begins (language cannot be changed mid-session).

---

## Prompt Design

Prompt engineering is the core of this project. Each chain has a dedicated, carefully crafted prompt. Here's the thinking behind each one:

### Info Gathering (`chains/info_chain.py`)

The key challenge is making field collection feel like a conversation, not a form. The prompt achieves this by:

- Injecting `{missing_fields}` and `{collected_data}` dynamically so the LLM always knows exactly what it still needs and what it already has — this prevents re-asking for already-collected fields
- Instructing the model to collect "one or two at a time" — pacing the conversation naturally
- Including inline validation rules (email format, phone digits, numeric years) directly in the prompt so the LLM handles correction without a separate validation layer
- Requiring JSON output `{"response": "...", "extracted": {...}}` — the response text and the structured extraction happen in a single LLM call, halving the token usage
- Using **OpenAI JSON mode** (`response_format: json_object`) at the API level to guarantee well-formed JSON on every call, eliminating parse failures entirely

### Tech Stack Extraction (`chains/techstack_chain.py`)

Tech stack extraction has a normalisation problem — candidates write "JS", "python", "React 18", "React/Redux" in dozens of ways. The prompt solves this with an explicit normalisation ruleset: `JS → JavaScript`, `K8s → Kubernetes`, strip version numbers, split compound entries. JSON mode ensures the `tech_stack` array is always parseable.

### Technical Question Generation (`chains/techq_chain.py`)

The calibration problem — questions must be appropriate for the candidate's seniority — is solved with an explicit experience ladder in the prompt:

- **0–2 years**: conceptual understanding, syntax, common gotchas
- **3–5 years**: design tradeoffs, debugging, production considerations
- **6–9 years**: architecture, scalability, deep internals
- **10+ years**: system design, cross-team impact, org decisions

The prompt explicitly bans trivia and requires scenario-based questions. Temperature is set to `0.65` here (vs `0.2–0.3` for extraction chains) — enough variety that questions don't feel templated across sessions.

### Interview Acknowledgement (`graph/nodes.py` — `_INTERVIEW_SYSTEM_PROMPT`)

The trickiest prompt. It must: acknowledge the answer, be encouraging but honest, not reveal the correct answer if wrong, detect off-topic replies, and transition cleanly — all in one response. The key constraint is explicit: *"note what was missing — do NOT give the correct answer."* Without this, the LLM defaults to teaching mode. JSON mode guarantees the `is_off_topic` boolean is always present and parseable.

### Fallback (`chains/fallback_chain.py`)

Kept intentionally minimal: 3 rules, 3–4 sentence output cap. The off-topic message is injected as `{candidate_message}` so the LLM can acknowledge it specifically rather than giving a generic redirect. Off-topic handling is now done **inline inside `interview_node`** — when `is_off_topic` is true, `run_fallback_chain` is called directly and the question index is not advanced, so the same question is asked again next turn without any extra graph routing.

### Sentiment Classification (`chains/sentiment_chain.py`)

Single-label classification at `temperature=0`. The prompt defines all three labels explicitly with concrete behavioural descriptors rather than abstract emotional terms, which reduces ambiguous outputs. Runs in a background thread so it never blocks the main chat response.

---

## Data Privacy

TalentScout is designed with GDPR principles in mind:

- **PII masking** — email and phone are masked before any database write (`john@gmail.com` → `jo***@gmail.com`). Raw PII never appears in logs or exports
- **Local storage only** — all data is stored in a local SQLite file (`data/talentscout.db`), resolved to an absolute path at startup. Nothing is sent to third parties beyond the OpenAI API for LLM inference
- **Right to erasure** — `DataHandler.delete_session(session_id)` permanently deletes all session data and Q&A pairs (GDPR Article 17)
- **Minimal collection** — only the fields required for screening are collected; no tracking, analytics, or behavioural profiling
- **Transparency** — candidates are shown a privacy notice in the sidebar on every session

> ⚠️ For production deployment, replace SQLite with a properly secured database (PostgreSQL with encryption at rest) and implement a full data retention policy.

---

## Technical Details

### Libraries & Versions

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.32.0 | Frontend chat UI and sidebar |
| `langchain` | ≥1.0.0 | Chain abstractions and prompt templates |
| `langchain-core` | ≥1.0.0 | Base interfaces (messages, parsers, prompts) |
| `langchain-openai` | ≥0.3.0 | OpenAI API integration via ChatOpenAI |
| `langgraph` | ≥1.0.0 | Stateful conversation graph orchestration |
| `python-dotenv` | ≥1.0.0 | Environment variable loading |
| `typing-extensions` | ≥4.12.0 | TypedDict support for Python 3.11 |

### Model

**`gpt-4o-mini`** via OpenAI API (configurable via `OPENAI_MODEL` in `.env`).

- Fast, cost-efficient model well-suited for structured extraction tasks
- Supports `response_format: json_object` — API-level JSON mode used on all extraction chains
- Swap to `gpt-4o` in `.env` for higher quality interview acknowledgements and question generation
- Two temperature configurations: `0.2–0.3` for extraction/classification, `0.65` for question generation

### OpenAI JSON Mode

All chains that need structured output (`info_chain`, `techstack_chain`, `techq_chain`, and the interview LLM in `nodes.py`) use OpenAI's JSON mode:

```python
_llm = ChatOpenAI(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.3,
    model_kwargs={"response_format": {"type": "json_object"}},
)
```

This is an API-level guarantee — OpenAI will not return a response that isn't valid JSON when this flag is set. It completely eliminates the parse-failure fallback path for these chains.

### LangChain 1.x Invocation Pattern

All chains use the explicit 3-step invocation pattern:

```python
# LangChain 1.x — direct invocation (used throughout this project)
formatted    = _prompt.invoke(inputs)               # Step 1: format prompt
llm_response = _llm.invoke(formatted)               # Step 2: call LLM
result       = extract_json(llm_response.content)   # Step 3: parse output
```

### LangGraph — Dispatcher Pattern (No Checkpointer)

State is owned entirely by Streamlit's `session_state`. On every user message, `app.py` builds the full input state (existing messages + new `HumanMessage`) and passes it to `graph.invoke()`. The graph executes exactly one node per call and returns the updated state, which is written back to `session_state`. No `MemorySaver` checkpointer is used — this avoids doubled message history and infinite loop bugs that arise when checkpointer state and manually-passed state are mixed.

```python
# app.py — full history passed on every invoke
input_state = {
    **current_state,
    "messages": existing_messages + [HumanMessage(content=user_message)],
}
new_state = graph.invoke(input_state)
st.session_state.graph_state = new_state
```

---

## Challenges & Solutions

### Challenge 1: Infinite API call loop

**Problem**: The original graph routed `info_gather → "continue_info" → info_gather` in a tight loop within a single `.invoke()` call. Since no user input arrives mid-invoke, the graph spun indefinitely, making continuous API calls with no response returned to Streamlit.

**Solution**: Replaced the looping design with a **dispatcher pattern**. The graph now executes exactly one node per `.invoke()` call. A `_dispatcher` function at `START` reads `current_stage` from state and routes to the correct node, which runs once and exits to `END`. The Streamlit app controls the turn loop — the graph never loops internally.

---

### Challenge 2: LLM ignoring JSON output instructions

**Problem**: `gpt-4o-mini` consistently returned plain conversational text (e.g. *"Thank you, Abhay! Your email looks good."*) despite prompt instructions saying to output JSON only. This caused `extract_json` to fail on every call, meaning no fields were ever extracted into state — the conversation appeared to work visually but collected nothing internally.

**Solution**: Switched all extraction chains to **OpenAI JSON mode** (`response_format: {"type": "json_object"}`). This is enforced at the API level — the model's output tokeniser is constrained to only produce valid JSON. Prompt-level instructions alone are insufficient for reliable structured output; API-level enforcement is required.

---

### Challenge 3: Message history lost between graph invocations

**Problem**: `app.py` was passing `"messages": [HumanMessage(content=user_message)]` — only the new message — to `graph.invoke()`. Without a checkpointer, the graph has no stored history, so each node received only the single most recent message with no context of what came before.

**Solution**: Pass the full accumulated history on every invoke:
```python
"messages": existing_messages + [HumanMessage(content=user_message)]
```
The `add_messages` reducer in `CandidateState` then appends the node's reply to this list, and the complete updated history is stored back in `session_state`.

---

### Challenge 4: Interview never completing

**Problem**: The completion check in `interview_node` was `next_idx >= len(queue) + 1`, which could never be true (e.g. with 6 questions, the maximum index reached was 6, but the condition required ≥ 7).

**Solution**: Replaced with a conditional check — after recording an answer, if `next_idx < len(queue)` append the next question; otherwise set `interview_complete = True`. This correctly terminates after the last answer is recorded.

---

### Challenge 5: SQLite database path failing on Windows

**Problem**: `data_handler.py` resolved the database path as `Path(__file__).parent / "data"`. On Windows, when Streamlit launches the app, `__file__` can be a relative path, making `.parent` resolve to the current working directory rather than the project folder. SQLite then tried to create the `data/` directory in the wrong location and failed with `unable to open database file`.

**Solution**: Use `.resolve()` to force an absolute path before computing the data directory:
```python
DB_DIR = Path(__file__).resolve().parent / "data"
```
This works correctly regardless of the working directory at launch time.

---

### Challenge 6: Background thread writing to Streamlit session state

**Problem**: Sentiment analysis ran in a `threading.Thread` and wrote directly to `st.session_state.sentiment`. Streamlit's `ScriptRunContext` is not thread-safe — writing to session state from a background thread causes a `missing ScriptRunContext` warning and the write is silently dropped.

**Solution**: The background thread writes to a module-level plain Python dict (`_sentiment_result = {"value": "neutral"}`) instead of `session_state`. The sidebar reads from this dict on each rerun and syncs the value into `session_state` on the main thread, where it is safe to do so.

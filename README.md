# 🎯 TalentScout — AI Hiring Assistant

> An intelligent chatbot that conducts initial technical screening interviews for technology placement candidates, powered by LangGraph, LangChain 1.x, and Groq's `llama-3.3-70b-versatile`.

---

## Table of Contents

- [Overview](#overview)
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

## Features

**Core**
- Conversational info gathering — never asks for all fields at once like a form
- Experience-calibrated questions — junior candidates get conceptual questions; senior candidates get architecture and system design questions
- Sequential question delivery with progress tracking
- Exit keyword detection at any stage (`exit`, `bye`, `quit`, etc.)
- Graceful fallback for off-topic or unintelligible responses
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
│  greeting → info_gather → tech_     │
│  stack → generate_questions →       │
│  interview ↔ fallback → farewell    │
├─────────────────────────────────────┤
│       LangChain 1.x Chains          │  ← One chain per concern, Groq LLM backend
│  info · techstack · techq ·         │
│  sentiment · fallback               │
└─────────────────────────────────────┘
```

### Conversation Graph

```
START
  └─► greeting
        └─► info_gather ◄──────────────┐
              │ (all 6 fields done)     │ (fields still missing)
              ▼                         │
          tech_stack ──────────────────►│
              └─► generate_questions    │
                        └─► interview ──┘
                              │
              ┌───────────────┼────────────────┐
              │ (exit keyword) │ (off-topic)   │ (all Qs answered)
              ▼                ▼               ▼
           farewell         fallback ──► interview
              │
             END
```

State flows through the graph as a typed `CandidateState` TypedDict. LangGraph's `MemorySaver` checkpointer persists the state between turns using the session UUID as the thread key — `app.py` never manually threads state.

---

## Project Structure

```
talentscout/
│
├── app.py                    # Streamlit entry point — run this
├── config.py                 # Loads GROQ_API_KEY from .env
├── data_handler.py           # SQLite storage, PII masking, GDPR erasure
├── requirements.txt
├── .env.example              # Copy to .env and add your API key
│
├── chains/                   # LangChain 1.x chain modules
│   ├── __init__.py           # Public API exports
│   ├── info_chain.py         # Conversational info gathering + JSON extraction
│   ├── techstack_chain.py    # Tech stack elicitation + normalisation
│   ├── techq_chain.py        # Technical question generation
│   ├── sentiment_chain.py    # Sentiment classification (positive/neutral/negative)
│   └── fallback_chain.py     # Off-topic redirect responses
│
└── graph/                    # LangGraph modules
    ├── __init__.py           # Public API exports
    ├── state.py              # CandidateState TypedDict + helpers + initial_state()
    ├── nodes.py              # All 7 node functions
    ├── edges.py              # Conditional routing logic
    └── graph.py              # Graph assembly + get_graph() singleton
```

---

## Installation

### Prerequisites

- Python 3.11+
- A free [Groq API key](https://console.groq.com) (takes ~2 minutes to get)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/talentscout.git
cd talentscout

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Open .env and add your GROQ_API_KEY

# 5. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ Yes | — | Your Groq API key from console.groq.com |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Groq model to use |

---

## Usage

1. **Start the app** — Alex greets you and asks for your full name
2. **Provide your details** — answer each question conversationally; Alex collects all 6 fields naturally over several turns
3. **Declare your tech stack** — list your languages, frameworks, databases, and tools
4. **Answer technical questions** — Alex asks 3–5 questions per technology, calibrated to your experience level
5. **Completion** — Alex summarises the session and explains next steps; download your report from the sidebar

**At any point**, type `exit`, `bye`, or `quit` to end the session gracefully.

**Language** — select your preferred language from the sidebar dropdown before the session begins.

---

## Prompt Design

Prompt engineering is the core of this project. Each chain has a dedicated, carefully crafted prompt. Here's the thinking behind each one:

### Info Gathering (`chains/info_chain.py`)

The key challenge is making field collection feel like a conversation, not a form. The prompt achieves this by:

- Injecting `{missing_fields}` and `{collected_data}` dynamically so the LLM always knows exactly what it still needs and what it already has — this prevents re-asking for already-collected fields
- Instructing the model to collect "one or two at a time" — pacing the conversation naturally
- Including inline validation rules (email format, phone digits, numeric years) directly in the prompt so the LLM handles correction without a separate validation layer
- Requiring JSON output `{"response": "...", "extracted": {...}}` — the response text and the structured extraction happen in a single LLM call, halving the token usage

### Tech Stack Extraction (`chains/techstack_chain.py`)

Tech stack extraction has a normalisation problem — candidates write "JS", "python", "React 18", "React/Redux" in dozens of ways. The prompt solves this with an explicit normalisation ruleset injected directly: `JS → JavaScript`, `K8s → Kubernetes`, strip version numbers, split compound entries. This keeps the output list clean without any post-processing code.

### Technical Question Generation (`chains/techq_chain.py`)

The calibration problem — questions must be appropriate for the candidate's seniority — is solved with an explicit experience ladder in the prompt:

- **0–2 years**: conceptual understanding, syntax, common gotchas
- **3–5 years**: design tradeoffs, debugging, production considerations  
- **6–9 years**: architecture, scalability, deep internals
- **10+ years**: system design, cross-team impact, org decisions

The prompt also explicitly bans trivia ("What does ACID stand for?") and requires scenario-based questions ("You have a service doing X, how would you..."). This produces markedly better questions than a generic "generate interview questions" prompt.

Temperature is set to `0.65` here (vs `0.2–0.3` for extraction chains) — enough variety that questions don't feel templated across sessions.

### Interview Acknowledgement (`graph/nodes.py` — `_INTERVIEW_SYSTEM_PROMPT`)

The trickiest prompt. It must: acknowledge the answer, be encouraging but honest, not reveal the correct answer if wrong, and transition to the next question — all in one response. The key constraint is explicit: *"acknowledge what was good, then gently note what was missing — do NOT give them the correct answer."* Without this, the LLM defaults to teaching mode.

### Fallback (`chains/fallback_chain.py`)

Kept intentionally minimal: 3 rules, 3–4 sentence output cap. The off-topic message is injected as `{candidate_message}` so the LLM can acknowledge it specifically rather than giving a generic redirect — this prevents the candidate from feeling dismissed.

### Sentiment Classification (`chains/sentiment_chain.py`)

Single-label classification at `temperature=0`. The prompt defines all three labels explicitly with concrete behavioural descriptors rather than abstract emotional terms, which dramatically reduces ambiguous outputs.

---

## Data Privacy

TalentScout is designed with GDPR principles in mind:

- **PII masking** — email and phone are masked before any database write (`john@gmail.com` → `jo***@gmail.com`). Raw PII never appears in logs or exports
- **Local storage only** — all data is stored in a local SQLite file (`data/talentscout.db`). Nothing is sent to third parties beyond the Groq API for LLM inference
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
| `langchain-groq` | ≥0.3.0 | Groq API integration via ChatGroq |
| `langgraph` | ≥1.0.0 | Stateful conversation graph orchestration |
| `python-dotenv` | ≥1.0.0 | Environment variable loading |
| `typing-extensions` | ≥4.12.0 | TypedDict support for Python 3.11 |

### Model

**`llama-3.3-70b-versatile`** via Groq API.

- Sub-second inference latency on Groq's LPU hardware
- Strong instruction following for JSON extraction tasks
- Free tier available at [console.groq.com](https://console.groq.com)
- Two temperature configurations: `0.2–0.3` for extraction/classification, `0.65` for question generation

### LangChain 1.x Invocation Pattern

All chains use the explicit 3-step invocation pattern rather than the deprecated LCEL pipe syntax (`prompt | llm | parser`):

```python
# LangChain 1.x — direct invocation (used throughout this project)
formatted    = _prompt.invoke(inputs)           # Step 1: format prompt
llm_response = _llm.invoke(formatted)           # Step 2: call LLM
result       = _parser.parse(llm_response.content)  # Step 3: parse output
```

### LangGraph State Management

`CandidateState` is a `TypedDict` with an `add_messages` reducer on the `messages` field, which appends new messages rather than replacing the list. This gives the LLM full conversation history on every call with zero manual threading. The `MemorySaver` checkpointer persists state between `.invoke()` calls keyed on `thread_id` (the session UUID), so Streamlit reruns never lose state.

---

## Challenges & Solutions

### Challenge 1: Preventing re-asking for collected fields

**Problem**: Without explicit context, the LLM would sometimes ask for fields already provided earlier in the conversation.

**Solution**: The `info_chain` prompt injects two dynamic variables — `{missing_fields}` and `{collected_data}` — on every call. These are computed from `CandidateState` in real time, so the LLM always has an accurate picture of what's been collected and what's still needed, regardless of conversation length.

---

### Challenge 2: JSON extraction reliability

**Problem**: LLMs occasionally wrap JSON responses in markdown fences (` ```json `) or add explanatory text, causing `JsonOutputParser` to fail.

**Solution**: Every JSON-outputting prompt ends with an explicit constraint: *"respond ONLY with valid JSON (no markdown fences, no explanation)"*. The `JsonOutputParser` also handles common fence stripping. All chain invocations are wrapped in try/except with typed fallback returns, so a single bad LLM response never crashes the session.

---

### Challenge 3: LangGraph + Streamlit state synchronisation

**Problem**: Streamlit reruns the entire script on every interaction, which would rebuild the graph and lose conversation state on every message.

**Solution**: Two mechanisms prevent this. First, the compiled graph is a module-level singleton (`get_graph()`) — it's built once and reused across reruns. Second, `MemorySaver` persists all conversation state keyed on `thread_id`, so even if Streamlit reruns the script, the graph resumes from exactly where it left off.

---

### Challenge 4: Calibrating question difficulty

**Problem**: A single prompt asking to "generate technical questions" produces generic, often trivial questions regardless of seniority.

**Solution**: The `techq_chain` prompt includes an explicit four-tier experience ladder with concrete behavioural descriptors for each level. The `questions_per_tech()` helper also varies question count by seniority (3 for junior, 4 for mid, 5 for senior), reflecting that deeper experience warrants deeper exploration.

---

### Challenge 5: Sentiment analysis latency

**Problem**: Running sentiment classification synchronously would add ~300–500ms to every response, making the chat feel sluggish.

**Solution**: Sentiment classification runs in a background `threading.Thread` in `app.py`. It updates `st.session_state.sentiment` asynchronously — the sidebar indicator may lag one turn behind, but the main chat response is never blocked.
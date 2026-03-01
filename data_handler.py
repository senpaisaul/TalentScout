"""
data_handler.py
----------------
Handles all candidate data storage, retrieval, and export for TalentScout.

Storage backend: SQLite (local file `data/talentscout.db`)
  - Zero external dependencies beyond stdlib
  - Single-file database; easy to wipe, backup, or replace with Postgres later
  - Session data is keyed by session_id (UUID from app.py)

Privacy / GDPR compliance:
  - PII fields (email, phone) are masked in application logs
  - Candidate data is stored only for the duration needed (session + export)
  - No data is sent to third parties beyond the Groq API for LLM calls
  - DataHandler.delete_session() allows right-to-erasure requests
  - All DB writes are parameterised (no SQL injection risk)

Schema:
  sessions table   — one row per screening session
  answers  table   — one row per question/answer pair, FK to sessions
"""

import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database path — absolute path so it works regardless of working directory.
# Path(__file__).resolve() guarantees an absolute path on Windows and Linux,
# even when Streamlit changes the working directory at runtime.
# ---------------------------------------------------------------------------
DB_DIR  = Path(__file__).resolve().parent / "data"
DB_PATH = DB_DIR / "talentscout.db"


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
_CREATE_SESSIONS_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    full_name           TEXT,
    email_masked        TEXT,      -- last 2 chars of local-part + domain, e.g. "jo***@gmail.com"
    phone_masked        TEXT,      -- last 4 digits only, e.g. "****-****-1234"
    years_of_experience INTEGER,
    desired_positions   TEXT,      -- JSON array as string
    current_location    TEXT,
    tech_stack          TEXT,      -- JSON array as string
    language            TEXT,
    sentiment           TEXT,
    current_stage       TEXT,
    is_complete         INTEGER,   -- 0 or 1
    started_at          TEXT,      -- ISO 8601
    completed_at        TEXT       -- ISO 8601, NULL until farewell
);
"""

_CREATE_ANSWERS_SQL = """
CREATE TABLE IF NOT EXISTS answers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    technology  TEXT NOT NULL,
    question    TEXT NOT NULL,
    answer      TEXT,
    answered_at TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""


# ---------------------------------------------------------------------------
# PII masking helpers
# ---------------------------------------------------------------------------
def _mask_email(email: str) -> str:
    """
    Masks an email for safe storage in logs / non-sensitive fields.
    "john.doe@example.com" → "jo***@example.com"
    """
    if not email or "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    masked_local = local[:2] + "***" if len(local) > 2 else "***"
    return f"{masked_local}@{domain}"


def _mask_phone(phone: str) -> str:
    """
    Masks a phone number, retaining only the last 4 digits.
    "+44 7911 123456" → "****-****-3456"
    """
    digits = re.sub(r"\D", "", str(phone))
    if len(digits) >= 4:
        return f"****-****-{digits[-4:]}"
    return "****"


# ---------------------------------------------------------------------------
# DataHandler class
# ---------------------------------------------------------------------------
class DataHandler:
    """
    Manages all read/write operations for candidate screening sessions.

    Usage:
        handler = DataHandler()
        handler.save_session(state)
        handler.save_answers(session_id, answers)
        data = handler.export_session(session_id)
        handler.delete_session(session_id)  # GDPR erasure
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Creates the database directory and tables if they don't exist."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Database location: {self.db_path}")
        except OSError as e:
            logger.error(f"Cannot create database directory {self.db_path.parent}: {e}")
            raise
        with self._connect() as conn:
            conn.execute(_CREATE_SESSIONS_SQL)
            conn.execute(_CREATE_ANSWERS_SQL)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Returns a configured SQLite connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row   # dict-like row access
        return conn

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_session(self, state: dict) -> None:
        """
        Upserts a candidate session row from a CandidateState dict.
        Masks PII before writing. Safe to call on every state update.

        Args:
            state: CandidateState dict (or any dict with matching keys).
        """
        session_id = state.get("session_id")
        if not session_id:
            logger.warning("save_session called with no session_id — skipping")
            return

        email = state.get("email") or ""
        phone = state.get("phone") or ""

        now = datetime.utcnow().isoformat()
        completed_at = now if state.get("is_complete") else None

        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, full_name, email_masked, phone_masked,
                        years_of_experience, desired_positions, current_location,
                        tech_stack, language, sentiment, current_stage,
                        is_complete, started_at, completed_at
                    ) VALUES (
                        :session_id, :full_name, :email_masked, :phone_masked,
                        :years_of_experience, :desired_positions, :current_location,
                        :tech_stack, :language, :sentiment, :current_stage,
                        :is_complete, :started_at, :completed_at
                    )
                    ON CONFLICT(session_id) DO UPDATE SET
                        full_name           = excluded.full_name,
                        email_masked        = excluded.email_masked,
                        phone_masked        = excluded.phone_masked,
                        years_of_experience = excluded.years_of_experience,
                        desired_positions   = excluded.desired_positions,
                        current_location    = excluded.current_location,
                        tech_stack          = excluded.tech_stack,
                        language            = excluded.language,
                        sentiment           = excluded.sentiment,
                        current_stage       = excluded.current_stage,
                        is_complete         = excluded.is_complete,
                        completed_at        = COALESCE(excluded.completed_at, sessions.completed_at)
                    """,
                    {
                        "session_id":          session_id,
                        "full_name":           state.get("full_name"),
                        "email_masked":        _mask_email(email),
                        "phone_masked":        _mask_phone(phone),
                        "years_of_experience": state.get("years_of_experience"),
                        "desired_positions":   json.dumps(state.get("desired_positions") or []),
                        "current_location":    state.get("current_location"),
                        "tech_stack":          json.dumps(state.get("tech_stack") or []),
                        "language":            state.get("language", "English"),
                        "sentiment":           state.get("sentiment", "neutral"),
                        "current_stage":       state.get("current_stage", "greeting"),
                        "is_complete":         int(bool(state.get("is_complete"))),
                        "started_at":          now,
                        "completed_at":        completed_at,
                    },
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"save_session DB error for {session_id}: {e}")

    def save_answers(self, session_id: str, answers: list[dict]) -> None:
        """
        Saves candidate Q&A pairs for a session.
        Replaces existing answers for the session (idempotent on re-save).

        Args:
            session_id: Session UUID.
            answers:    List of {"tech", "question", "answer"} dicts.
        """
        if not session_id or not answers:
            return
        now = datetime.utcnow().isoformat()
        try:
            with self._connect() as conn:
                # Delete existing answers for this session first (clean upsert)
                conn.execute(
                    "DELETE FROM answers WHERE session_id = ?", (session_id,)
                )
                conn.executemany(
                    """
                    INSERT INTO answers (session_id, technology, question, answer, answered_at)
                    VALUES (:session_id, :technology, :question, :answer, :answered_at)
                    """,
                    [
                        {
                            "session_id":  session_id,
                            "technology":  a.get("tech", ""),
                            "question":    a.get("question", ""),
                            "answer":      a.get("answer", ""),
                            "answered_at": now,
                        }
                        for a in answers
                    ],
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"save_answers DB error for {session_id}: {e}")

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def load_session(self, session_id: str) -> Optional[dict]:
        """
        Loads a session row by session_id.

        Returns:
            dict with session fields, or None if not found.
        """
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
                ).fetchone()
            if row:
                data = dict(row)
                # Deserialise JSON fields
                data["desired_positions"] = json.loads(data.get("desired_positions") or "[]")
                data["tech_stack"]        = json.loads(data.get("tech_stack") or "[]")
                return data
        except sqlite3.Error as e:
            logger.error(f"load_session DB error for {session_id}: {e}")
        return None

    def load_answers(self, session_id: str) -> list[dict]:
        """
        Loads all Q&A pairs for a session.

        Returns:
            List of {"technology", "question", "answer", "answered_at"} dicts.
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT technology, question, answer, answered_at "
                    "FROM answers WHERE session_id = ? ORDER BY id",
                    (session_id,),
                ).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"load_answers DB error for {session_id}: {e}")
            return []

    def list_sessions(self, completed_only: bool = False) -> list[dict]:
        """
        Lists all sessions in the database (admin/debug use).

        Args:
            completed_only: If True, returns only fully completed sessions.
        """
        try:
            with self._connect() as conn:
                query = "SELECT * FROM sessions"
                if completed_only:
                    query += " WHERE is_complete = 1"
                query += " ORDER BY started_at DESC"
                rows = conn.execute(query).fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"list_sessions DB error: {e}")
            return []

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_session_json(self, session_id: str) -> Optional[str]:
        """
        Exports a complete session (profile + answers) as a formatted JSON string.
        Suitable for download via Streamlit's st.download_button.

        Returns:
            JSON string, or None if session not found.
        """
        session = self.load_session(session_id)
        if not session:
            return None
        answers = self.load_answers(session_id)

        export = {
            "talentscout_screening": {
                "exported_at":  datetime.utcnow().isoformat(),
                "session_id":   session_id,
                "candidate": {
                    "full_name":           session.get("full_name"),
                    "email":               session.get("email_masked"),   # PII masked
                    "phone":               session.get("phone_masked"),   # PII masked
                    "years_of_experience": session.get("years_of_experience"),
                    "desired_positions":   session.get("desired_positions"),
                    "current_location":    session.get("current_location"),
                    "tech_stack":          session.get("tech_stack"),
                },
                "screening": {
                    "language":      session.get("language"),
                    "sentiment":     session.get("sentiment"),
                    "completed":     bool(session.get("is_complete")),
                    "started_at":    session.get("started_at"),
                    "completed_at":  session.get("completed_at"),
                },
                "answers": answers,
            }
        }
        return json.dumps(export, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # GDPR: Right to erasure
    # ------------------------------------------------------------------

    def delete_session(self, session_id: str) -> bool:
        """
        Permanently deletes all data for a session (GDPR Article 17).

        Returns:
            True if deleted, False if not found or error.
        """
        try:
            with self._connect() as conn:
                conn.execute("DELETE FROM answers WHERE session_id = ?", (session_id,))
                result = conn.execute(
                    "DELETE FROM sessions WHERE session_id = ?", (session_id,)
                )
                conn.commit()
            deleted = result.rowcount > 0
            if deleted:
                logger.info(f"Session {session_id} deleted (GDPR erasure)")
            return deleted
        except sqlite3.Error as e:
            logger.error(f"delete_session DB error for {session_id}: {e}")
            return False


# ---------------------------------------------------------------------------
# Module-level singleton — import this in app.py and nodes
# ---------------------------------------------------------------------------
db = DataHandler()
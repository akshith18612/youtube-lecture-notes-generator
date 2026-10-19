"""Small Supabase persistence layer for the public demo.

Only the server-side SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are used here.
Never expose the service-role key to browsers or commit it to Git.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st
from supabase import Client, create_client

TABLE = "generation_history"


def get_db() -> Client | None:
    """Return a Supabase client, or None when database secrets are not configured."""
    try:
        url = str(st.secrets["SUPABASE_URL"]).strip()
        key = str(st.secrets["SUPABASE_SERVICE_ROLE_KEY"]).strip()
    except Exception:
        return None
    if not url or not key:
        return None
    return create_client(url, key)


def save_generation(*, session_id: str, video_id: str, title: str,
                    source_url: str, mode: str, language: str,
                    notes_markdown: str, status: str = "completed") -> None:
    db = get_db()
    if db is None:
        raise RuntimeError("Supabase is not configured. Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to app secrets.")
    db.table(TABLE).insert({
        "session_id": session_id,
        "video_id": video_id,
        "title": title[:500],
        "source_url": source_url[:2000],
        "mode": mode,
        "language": language[:100],
        "notes_markdown": notes_markdown,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }).execute()


def load_session_history(session_id: str, limit: int = 30) -> list[dict[str, Any]]:
    db = get_db()
    if db is None:
        return []
    response = (db.table(TABLE)
                .select("id, video_id, title, source_url, mode, language, notes_markdown, status, created_at")
                .eq("session_id", session_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute())
    return response.data or []

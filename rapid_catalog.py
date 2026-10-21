"""Shared, rate-limited RapidAPI tournament catalogue for TennisEdge IA.

The Odds API and RapidAPI are separate feeds. RapidAPI fixtures are used for
coverage/discovery only: prices without a trustworthy quote timestamp cannot
become approved value signals.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import os

import pandas as pd
import streamlit as st

from rapid_tennis import available, get_upcoming, parse_upcoming

CACHE_SECONDS = 6 * 3600


@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def cached_rapid_catalog(key_fingerprint: str) -> tuple[pd.DataFrame, dict, datetime]:
    """Cache once across the dashboard and secondary page, never cache the key.

    At most two provider pages are read per cache miss (about 8 requests/day per active instance). The fingerprint is used
    solely to invalidate cache when a private server key is rotated.
    """
    items, metadata = get_upcoming(max_pages=2)
    checked = datetime.now(timezone.utc)
    return parse_upcoming(items, retrieved_at=checked), metadata, checked


def current_rapid_catalog() -> tuple[pd.DataFrame, dict, datetime]:
    if not available():
        raise RuntimeError('RapidAPI aún no está configurada en Render.')
    fingerprint = sha256(os.environ['RAPIDAPI_TENNIS_KEY'].strip().encode()).hexdigest()[:16]
    frame, metadata, checked = cached_rapid_catalog(fingerprint)
    if frame.empty:
        return frame, metadata, checked
    # An already-started match must disappear from the proposed-fixtures view,
    # even when the provider response remains cached for several hours.
    now = pd.Timestamp.now(tz='UTC')
    starts = pd.to_datetime(frame['Inicio UTC'], utc=True, errors='coerce')
    frame = frame.loc[starts.isna() | starts.gt(now)].copy()
    return frame.reset_index(drop=True), metadata, checked

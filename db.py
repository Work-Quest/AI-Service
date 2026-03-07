"""
Neon Postgres connection helpers for AI-service.
"""

from __future__ import annotations

import os

import psycopg2


def get_db_url() -> str:
    """
    Returns the Postgres connection URL.
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("Missing DATABASE_URL env var for Neon Postgres connection.")
    return url


def get_connection():
    """
    Open a new psycopg2 connection.

    NOTE: Neon often requires SSL; if you hit SSL errors, add `?sslmode=require` to your DB URL.
    """
    return psycopg2.connect(get_db_url())



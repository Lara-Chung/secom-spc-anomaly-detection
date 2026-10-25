"""Database connection helper; settings match docker-compose.yml and .env.example.

The host is always 127.0.0.1 (the Compose container, which publishes its port on
that address only): the loader drops and recreates tables, so it must not follow
a DATABASE_URL meant for another project.
Precedence for each setting: environment variables, then the repo's .env file,
then defaults. The .env file is parsed with
python-dotenv; keep it to plain KEY=VALUE lines, as in .env.example, so that
Docker Compose and Python read identical values.
"""

import os

import pandas as pd
import psycopg
from dotenv import dotenv_values
from psycopg.conninfo import make_conninfo

from secom_data import REPO_ROOT

DOTENV_FILE = REPO_ROOT / ".env"
DEFAULTS = {
    "POSTGRES_USER": "secom",
    "POSTGRES_PASSWORD": "secom",
    "POSTGRES_DB": "secom",
    "POSTGRES_PORT": "5432",
}


def setting(name: str, dotenv: dict) -> str:
    return os.environ.get(name) or dotenv.get(name) or DEFAULTS[name]


def connection_string(dotenv_file=DOTENV_FILE) -> str:
    dotenv = dotenv_values(dotenv_file) if dotenv_file.exists() else {}
    return make_conninfo(
        host="127.0.0.1",
        port=setting("POSTGRES_PORT", dotenv),
        user=setting("POSTGRES_USER", dotenv),
        password=setting("POSTGRES_PASSWORD", dotenv),
        dbname=setting("POSTGRES_DB", dotenv),
    )


def connect(**kwargs) -> psycopg.Connection:
    return psycopg.connect(connection_string(), **kwargs)


def fetch(conn, query: str, params=None) -> pd.DataFrame:
    """Run a query (optionally with bound parameters) and return a DataFrame."""
    cur = conn.execute(query, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])

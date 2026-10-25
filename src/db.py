"""Database connection helper; settings match docker-compose.yml and .env.example.

Precedence for every setting (including DATABASE_URL): environment variables,
then the repo's .env file, then defaults. The .env file is parsed with
python-dotenv; keep it to plain KEY=VALUE lines, as in .env.example, so that
Docker Compose and Python read identical values.
"""

import os

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


def setting(name: str, dotenv: dict) -> str | None:
    return os.environ.get(name) or dotenv.get(name) or DEFAULTS.get(name)


def connection_string(dotenv_file=DOTENV_FILE) -> str:
    dotenv = dotenv_values(dotenv_file) if dotenv_file.exists() else {}
    if url := setting("DATABASE_URL", dotenv):
        return url
    return make_conninfo(
        host="localhost",
        port=setting("POSTGRES_PORT", dotenv),
        user=setting("POSTGRES_USER", dotenv),
        password=setting("POSTGRES_PASSWORD", dotenv),
        dbname=setting("POSTGRES_DB", dotenv),
    )


def connect(**kwargs) -> psycopg.Connection:
    return psycopg.connect(connection_string(), **kwargs)

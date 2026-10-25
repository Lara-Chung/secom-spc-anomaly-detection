#!/usr/bin/env bash
# Reproduce every output in outputs/ from a clean checkout.
# Requires Docker (with Compose) and uv. Downloads the SECOM archive on first run.
set -euo pipefail
cd "$(dirname "$0")"

docker compose up -d --wait
uv sync
uv run python src/download_or_prepare.py
uv run python src/load_database.py
uv run python src/run_sql.py
uv run python src/train_evaluate.py
uv run python src/make_plots.py
uv run python src/build_investigation_report.py
SECOM_TESTS_REQUIRE_DB=1 uv run pytest -q

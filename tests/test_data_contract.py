"""Checks that the loaded database matches the raw SECOM files.

Requires the database to be running and loaded (src/load_database.py).
"""

import numpy as np
import pandas as pd
import psycopg
import pytest

from db import connect
from secom_data import FEATURES_FILE, LABELS_FILE, read_features, read_labels

TABLES = ("runs", "sensors", "measurements")


@pytest.fixture(scope="module")
def conn():
    try:
        # autocommit: a failing query in one test must not abort the shared session
        c = connect(autocommit=True)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Cannot connect to PostgreSQL ({exc}); check `docker compose up -d --wait` and .env")
    missing = [t for t in TABLES if scalar(c, "SELECT to_regclass(%s)", (t,)) is None]
    if missing:
        c.close()
        pytest.skip(f"Tables {missing} not found; run `uv run python src/load_database.py`")
    yield c
    c.close()


@pytest.fixture(scope="module")
def raw():
    if not (FEATURES_FILE.exists() and LABELS_FILE.exists()):
        pytest.skip("Raw files not found; run `uv run python src/download_or_prepare.py`")
    return read_features(), read_labels()


def scalar(conn, sql, params=None):
    return conn.execute(sql, params).fetchone()[0]


def test_row_counts_match_raw_files(conn, raw):
    features, labels = raw
    assert scalar(conn, "SELECT count(*) FROM runs") == len(labels)
    assert scalar(conn, "SELECT count(*) FROM sensors") == features.shape[1]
    assert scalar(conn, "SELECT count(*) FROM measurements") == features.size


def test_missing_values_are_null(conn, raw):
    features, _ = raw
    nulls = scalar(conn, "SELECT count(*) FROM measurements WHERE value IS NULL")
    assert nulls == int(features.isna().values.sum())
    assert scalar(conn, "SELECT count(*) FROM measurements WHERE value = 'NaN'") == 0


def test_labels_and_fail_flag(conn, raw):
    _, labels = raw
    n_fail = int((labels["label"] == 1).sum())
    assert scalar(conn, "SELECT count(*) FROM runs WHERE is_fail") == n_fail
    assert scalar(conn, "SELECT count(*) FROM runs WHERE label NOT IN (-1, 1)") == 0


def test_timestamps_match_raw(conn, raw):
    _, labels = raw
    lo, hi = conn.execute("SELECT min(ts), max(ts) FROM runs").fetchone()
    assert lo == labels["ts"].min().to_pydatetime()
    assert hi == labels["ts"].max().to_pydatetime()


def test_per_sensor_counts_and_sums_match(conn, raw):
    """Catches shifted or transposed loads that keep the total NULL count."""
    features, _ = raw
    db = pd.DataFrame(
        conn.execute(
            "SELECT sensor_id, count(value), coalesce(sum(value), 0) FROM measurements "
            "GROUP BY sensor_id ORDER BY sensor_id"
        ).fetchall(),
        columns=["sensor_id", "n_present", "total"],
    ).set_index("sensor_id")
    assert list(db.index) == list(features.columns)
    assert (db["n_present"].values == features.count().values).all()
    # Summation order differs (PostgreSQL sequential, numpy pairwise), so the
    # tolerance scales with each sensor's sum of absolute values
    tolerance = 1e-11 * features.abs().sum().values
    assert (np.abs(db["total"].values - features.sum().values) <= tolerance).all()


def test_values_round_trip_exactly(conn, raw):
    features, _ = raw
    rng = np.random.default_rng(0)
    for run_id, sensor_id in zip(rng.integers(0, features.shape[0], 200),
                                 rng.integers(0, features.shape[1], 200)):
        db_value = scalar(
            conn,
            "SELECT value FROM measurements WHERE run_id = %s AND sensor_id = %s",
            (int(run_id), int(sensor_id)),
        )
        raw_value = features.iat[run_id, sensor_id]
        if np.isnan(raw_value):
            assert db_value is None
        else:
            assert db_value == raw_value


def test_keys_and_constraints_are_enforced(conn):
    # force_rollback: if a constraint were missing, the probe row must not persist
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        with conn.transaction(force_rollback=True):
            conn.execute("INSERT INTO measurements VALUES (999999, 0, 1.0)")
    with pytest.raises(psycopg.errors.UniqueViolation):
        with conn.transaction(force_rollback=True):
            conn.execute("INSERT INTO measurements VALUES (0, 0, 1.0)")
    with pytest.raises(psycopg.errors.CheckViolation):
        with conn.transaction(force_rollback=True):
            conn.execute("INSERT INTO runs (run_id, ts, label) VALUES (999999, now(), 0)")

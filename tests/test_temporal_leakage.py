"""Leakage checks for the SQL monitoring views, on a tiny synthetic dataset.

The real schema and sql/0*.sql files are run inside a temporary schema, so the
loaded SECOM tables are untouched.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import psycopg
import pytest

from conftest import skip_or_fail
from db import connect
from run_sql import SQL_DIR, create_views

N_RUNS = 60
N_TRAIN = 42  # floor(0.7 * 60), matches 00_run_split.sql
SCHEMA = "test_temporal_leakage"


@pytest.fixture
def conn():
    try:
        c = connect(autocommit=True)
    except psycopg.OperationalError as exc:
        skip_or_fail(f"Cannot connect to PostgreSQL ({exc})")
    c.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    c.execute(f"CREATE SCHEMA {SCHEMA}")
    c.execute(f"SET search_path TO {SCHEMA}")
    c.execute((SQL_DIR / "schema.sql").read_text())

    rng = np.random.default_rng(0)
    t0 = datetime(2008, 7, 19)
    with c.cursor() as cur:
        cur.executemany(
            "INSERT INTO runs (run_id, ts, label) VALUES (%s, %s, %s)",
            [(i, t0 + timedelta(hours=i), 1 if i % 10 == 0 else -1) for i in range(N_RUNS)],
        )
        cur.executemany(
            "INSERT INTO sensors VALUES (%s, %s)",
            [(j, f"sensor_{j:03d}") for j in range(3)],
        )
        rows = []
        for i in range(N_RUNS):
            rows.append((i, 0, float(rng.normal(10, 1))))       # ordinary sensor
            # flat start; 0.1 is not exactly representable, so stddev_samp of
            # the flat window is a tiny non-zero number rather than 0
            rows.append((i, 1, 0.1 if i < 30 else float(i)))
            rows.append((i, 2, 7.0))                            # constant: excluded
        cur.executemany("INSERT INTO measurements VALUES (%s, %s, %s)", rows)
    create_views(c)
    yield c
    c.execute(f"DROP SCHEMA {SCHEMA} CASCADE")
    c.close()


def refresh(conn):
    """The materialized views are snapshots; rebuild them after changing a base table,
    in dependency order."""
    for view in ("sensor_quality", "frozen_limits", "spc_run_scores"):
        conn.execute(f"REFRESH MATERIALIZED VIEW {view}")


def limits(conn):
    return conn.execute("SELECT sensor_id, mean_train, std_train FROM frozen_limits ORDER BY sensor_id").fetchall()


def rolling(conn):
    return conn.execute(
        "SELECT run_id, sensor_id, roll_mean, roll_std, roll_n, z, is_alert "
        "FROM spc_rolling ORDER BY sensor_id, run_id"
    ).fetchall()


def run_scores(conn):
    return conn.execute(
        "SELECT run_id, frozen_n_checked, frozen_n_alerts, rolling_n_checked, rolling_n_alerts "
        "FROM spc_run_scores ORDER BY run_id"
    ).fetchall()


def test_split_and_exclusion(conn):
    counts = dict(conn.execute("SELECT split, count(*) FROM run_split GROUP BY split").fetchall())
    assert counts == {"train": N_TRAIN, "test": N_RUNS - N_TRAIN}
    eligible = [r[0] for r in conn.execute(
        "SELECT sensor_id FROM sensor_quality WHERE is_eligible ORDER BY sensor_id")]
    assert eligible == [0, 1]


def test_rolling_baseline_uses_previous_rows_only(conn):
    values = [r[0] for r in conn.execute(
        "SELECT value FROM measurements WHERE sensor_id = 0 ORDER BY run_id")]
    (roll_mean,) = conn.execute(
        "SELECT roll_mean FROM spc_rolling WHERE sensor_id = 0 AND run_id = 30").fetchone()
    assert roll_mean == pytest.approx(np.mean(values[:30]))  # runs 0..29, not run 30


def test_insufficient_history_and_zero_variance_are_not_evaluated(conn):
    rows = {(r[0], r[1]): r for r in rolling(conn)}
    assert all(rows[(i, 0)][6] is None for i in range(20))   # fewer than 20 prior values
    assert rows[(20, 0)][6] is not None
    assert all(rows[(i, 1)][6] is None for i in range(20, 31))  # window std = 0
    assert rows[(31, 1)][6] is not None


def test_changing_future_values_does_not_change_past_results(conn):
    before_rolling = rolling(conn)
    before_scores = run_scores(conn)
    before_limits = limits(conn)
    conn.execute("UPDATE measurements SET value = value * 100 + 1000 WHERE run_id >= %s", (N_TRAIN,))
    refresh(conn)
    after_rolling = rolling(conn)
    after_scores = run_scores(conn)
    # The limits themselves must not move: alert counts alone could stay equal by chance
    assert limits(conn) == pytest.approx(before_limits, rel=1e-12)  # a REFRESH may re-aggregate in another order

    assert [r for r in after_rolling if r[0] < N_TRAIN] == [r for r in before_rolling if r[0] < N_TRAIN]
    # Frozen limits come from the training period, so train-period scores are unchanged too
    assert after_scores[:N_TRAIN] == before_scores[:N_TRAIN]
    assert after_scores[N_TRAIN:] != before_scores[N_TRAIN:]  # the change did reach the test period


def test_changing_labels_does_not_change_scores_or_evidence(conn):
    """Outcome isolation: labels feed only the retrospective columns."""
    from investigate_run import sensor_evidence
    before_rolling = rolling(conn)
    before_scores = run_scores(conn)
    before_evidence = sensor_evidence(conn, N_TRAIN + 5, "spc_frozen")
    before_limits = limits(conn)
    conn.execute("UPDATE runs SET label = -label")  # every run, so train-derived limits are covered too
    refresh(conn)
    assert limits(conn) == pytest.approx(before_limits, rel=1e-12)  # a REFRESH may re-aggregate in another order
    assert rolling(conn) == before_rolling
    assert run_scores(conn) == before_scores
    pd.testing.assert_frame_equal(sensor_evidence(conn, N_TRAIN + 5, "spc_frozen"), before_evidence)
    # The flip did reach the outcome column
    (n_fail,) = conn.execute("SELECT count(*) FROM run_split WHERE split = 'test' AND is_fail").fetchone()
    assert n_fail == (N_RUNS - N_TRAIN) - sum(1 for i in range(N_TRAIN, N_RUNS) if i % 10 == 0)


def test_runs_with_equal_timestamps_are_ordered_by_run_id(conn):
    """Ties on ts are broken by run_id: the later run_id sees the earlier one, not vice versa.

    This documents the rule rather than guarding it: without the run_id tie-break
    PostgreSQL's sort order among equal ts values is unspecified, so the test
    could still pass by chance.
    """
    conn.execute("UPDATE runs SET ts = (SELECT ts FROM runs WHERE run_id = 40) WHERE run_id = 41")
    values = [r[0] for r in conn.execute("SELECT value FROM measurements WHERE sensor_id = 0 ORDER BY run_id")]
    rows = {r[0]: r[1:] for r in conn.execute(
        "SELECT run_id, roll_n, roll_mean FROM spc_rolling WHERE sensor_id = 0 AND run_id IN (40, 41)")}
    assert rows[40][0] == 40 and rows[40][1] == pytest.approx(np.mean(values[:40]))  # runs 0..39
    assert rows[41][0] == 41 and rows[41][1] == pytest.approx(np.mean(values[:41]))  # runs 0..40


def test_single_run_evidence_keeps_history_and_matches_view_limits(conn):
    """Filtering the evidence to one run must not shrink the history behind its band,
    and the limits in the evidence must equal the ones the SPC views computed."""
    from investigate_run import sensor_evidence
    view = pd.DataFrame(
        conn.execute("SELECT run_id, sensor_id, roll_mean, roll_std, roll_n, is_alert FROM spc_rolling").fetchall(),
        columns=["run_id", "sensor_id", "roll_mean", "roll_std", "roll_n", "is_alert"],
    ).set_index(["run_id", "sensor_id"])
    for run_id, expected_history in [(30, 30), (55, 50)]:  # 50 = window size
        ev = sensor_evidence(conn, run_id, "spc_rolling").set_index("sensor_id").loc[0]
        v = view.loc[(run_id, 0)]
        assert ev["history_count"] == expected_history == v["roll_n"]
        assert ev["lower_limit"] == pytest.approx(v["roll_mean"] - 3 * v["roll_std"])
        assert ev["upper_limit"] == pytest.approx(v["roll_mean"] + 3 * v["roll_std"])
        assert bool(ev["is_sensor_alert"]) == bool(v["is_alert"])

    train_values = [r[0] for r in conn.execute(
        "SELECT value FROM measurements WHERE sensor_id = 0 AND run_id < %s ORDER BY run_id", (N_TRAIN,))]
    mean, std = np.mean(train_values), np.std(train_values, ddof=1)
    ev = sensor_evidence(conn, 50, "spc_frozen").set_index("sensor_id").loc[0]
    assert ev["history_count"] == N_TRAIN
    assert ev["lower_limit"] == pytest.approx(mean - 3 * std)
    assert ev["upper_limit"] == pytest.approx(mean + 3 * std)

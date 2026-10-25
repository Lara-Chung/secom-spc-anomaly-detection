"""Checks for the alert-investigation queries against the loaded database."""

import numpy as np
import pandas as pd
import psycopg
import pytest

from conftest import skip_or_fail
from db import connect, fetch
from investigate_run import EVIDENCE_SQL, SCORE_COLUMN, InvestigationError, run_header, sensor_evidence, summarise
from secom_data import OUTPUT_DIR
from train_evaluate import EligibleSensors

# First run (nothing assessable under the rolling rule), runs at the 20-reading
# minimum and the 50-run window boundary, the split boundary, the test-period run
# with the most alerts, and the last run.
SAMPLE_RUNS = (0, 19, 20, 50, 1095, 1096, 1458, 1566)


@pytest.fixture(scope="module")
def conn():
    try:
        c = connect(autocommit=True)
    except psycopg.OperationalError as exc:
        skip_or_fail(f"Cannot connect to PostgreSQL ({exc})")
    needed = ("sensor_quality", "frozen_limits", "spc_rolling", "spc_run_scores", "feature_export", "feature_export_all")
    missing = [v for v in needed if c.execute("SELECT to_regclass(%s)", (v,)).fetchone()[0] is None]
    if missing:
        c.close()
        skip_or_fail(f"Views {missing} not found; run `uv run python src/run_sql.py`")
    yield c
    c.close()


@pytest.fixture(scope="module")
def evidence(conn):
    """Evidence tables for every sample run and method, queried once per module."""
    return {(run_id, m): sensor_evidence(conn, run_id, m) for run_id in SAMPLE_RUNS for m in EVIDENCE_SQL}


@pytest.fixture(scope="module")
def run_scores(conn):
    return fetch(conn, "SELECT * FROM spc_run_scores").set_index("run_id")


@pytest.mark.parametrize("method", sorted(EVIDENCE_SQL))
def test_sensor_alerts_sum_to_run_score(run_scores, evidence, method):
    for run_id in SAMPLE_RUNS:
        score = int(run_scores.loc[run_id, SCORE_COLUMN[method]])
        assert int(evidence[(run_id, method)]["is_sensor_alert"].fillna(False).sum()) == score


@pytest.mark.parametrize("method", sorted(EVIDENCE_SQL))
def test_counts_are_nested_and_reasons_explain_the_rest(evidence, method):
    for run_id in SAMPLE_RUNS:
        ev = evidence[(run_id, method)]
        s = summarise(ev)
        assert s["sensor_scope_count"] >= s["n_observed"] >= s["n_assessable"] >= s["n_alerted"]
        reasons = s["n_missing_value"] + s["n_insufficient_history"] + s["n_flat_window"]
        assert reasons == s["sensor_scope_count"] - s["n_assessable"]
        # No z-score may be computed where the reading is not assessable
        assert ev.loc[~ev["is_assessable"], "z_score"].isna().all()
        assert np.isfinite(ev["z_score"].dropna()).all()


def test_unknown_run_and_method_are_rejected(conn):
    with pytest.raises(InvestigationError):
        run_header(conn, 999999, "spc_frozen")
    with pytest.raises(ValueError):
        sensor_evidence(conn, 0, "bogus")


def test_python_eligibility_matches_sql(conn):
    """EligibleSensors fitted on the training period keeps exactly the SQL-eligible sensors."""
    all_rows = fetch(conn, "SELECT sensor_ids, sensor_values FROM feature_export_all "
                           "WHERE split = 'train' ORDER BY ts, run_id")
    sql_ids = fetch(conn, "SELECT sensor_ids FROM feature_export LIMIT 1")["sensor_ids"][0]
    X_train = np.array(all_rows["sensor_values"].tolist(), dtype=float)
    keep = EligibleSensors().fit(X_train).keep_
    python_ids = list(np.array(all_rows["sensor_ids"][0])[keep])
    assert python_ids == list(sql_ids)


def test_workload_totals_match_metrics(conn):
    metrics_path = OUTPUT_DIR / "metrics.csv"
    if not metrics_path.exists():
        skip_or_fail("outputs/metrics.csv not found; run `uv run python src/train_evaluate.py`")
    from build_investigation_report import alert_workload, check_against_metrics
    from investigate_run import run_threshold
    thresholds = {m: run_threshold(m) for m in EVIDENCE_SQL}
    totals = check_against_metrics(alert_workload(conn, thresholds))  # raises on mismatch
    assert set(totals["method"]) == set(EVIDENCE_SQL)
    assert (totals["n_runs"] == totals["n_assessable"] + totals["n_not_assessable"]).all()
    # Frozen limits exist for every run; only the rolling rule has a warm-up period
    assert (totals.loc[totals["method"] == "spc_frozen", "n_not_assessable"] == 0).all()


def test_rolling_evidence_matches_the_view(conn, evidence):
    """The per-run rebuild of the band must reproduce the view's numbers exactly, not approximately."""
    view = fetch(conn, "SELECT run_id, sensor_id, roll_mean, roll_std, roll_n, z, is_alert "
                       "FROM spc_rolling WHERE run_id = ANY(%s)", (list(SAMPLE_RUNS),)).set_index(["run_id", "sensor_id"])
    for run_id in SAMPLE_RUNS:
        ev = evidence[(run_id, "spc_rolling")].set_index("sensor_id").sort_index()
        v = view.loc[run_id].sort_index()
        assert list(ev.index) == list(v.index)
        for ev_col, v_col in [("baseline_mean", "roll_mean"), ("baseline_std", "roll_std"), ("z_score", "z")]:
            a, b = ev[ev_col].to_numpy(dtype=float), v[v_col].to_numpy(dtype=float)
            assert np.array_equal(np.isnan(a), np.isnan(b)) and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
        assert (ev["history_count"].to_numpy() == v["roll_n"].to_numpy()).all()
        flags = lambda s: s.astype(object).where(s.notna(), None).tolist()
        assert flags(ev["is_sensor_alert"]) == flags(v["is_alert"])

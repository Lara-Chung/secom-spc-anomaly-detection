"""Smoke tests for the read-only Streamlit app, run headlessly with Streamlit's AppTest.

They need the exported artifacts under outputs/ (run ./run_all.sh); no database.
"""

import datetime as dt
import shutil

import pandas as pd

import pytest
from streamlit.testing.v1 import AppTest

from conftest import skip_or_fail
from secom_data import OUTPUT_DIR, REPO_ROOT

APP = REPO_ROOT / "app.py"
# The same eight artifacts app.py checks before rendering
REQUIRED = [OUTPUT_DIR / p for p in ("sql/spc_run_scores.csv", "metrics.csv", "ap_differences.csv",
                                     "investigation/alert_workload.csv", "investigation/sensor_evidence.csv",
                                     "investigation/manifest.json", "data_audit.json", "figures/pr_curve_test.png")]


def case_ids() -> set[int]:
    """The documented cases are whatever build_investigation_report.py exported."""
    return set(pd.read_csv(OUTPUT_DIR / "investigation" / "sensor_evidence.csv")["run_id"].unique())


@pytest.fixture(scope="module", autouse=True)
def artifacts_present():
    missing = [p for p in REQUIRED if not p.exists()]
    if missing:
        skip_or_fail(f"Artifacts missing ({missing[0]}); run ./run_all.sh")


def test_app_starts_with_three_tabs_and_metrics():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Overview", "Investigation", "Evaluation"]
    assert app.warning[0].value.startswith("Historical SECOM data")
    labels = [m.label for m in app.metric]
    assert labels[:4] == ["Runs", "Assessable runs", "Alerts", "Alerts per 100 assessable"]
    assert int(app.metric[0].value) == 1567  # whole period, frozen rule


def test_date_filter_changes_the_overview_counts():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.sidebar.slider[0].set_value((dt.date(2008, 10, 1), dt.date(2008, 10, 17))).run()
    assert not at.exception
    assert int(at.metric[0].value) < 1567
    assert int(at.metric[0].value) == 359  # October runs, see outputs/sql/fail_trends_daily.csv


def test_case_run_shows_evidence_and_other_runs_show_cli_hint():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    options = at.selectbox[0].options  # formatted labels, e.g. "1151 (documented case)" or "2"
    cases = case_ids()
    assert sum("documented case" in o for o in options) == len(cases)
    case = sorted(cases)[0]
    at.selectbox[0].set_value(case).run()  # a documented case: evidence table and counts appear
    assert not at.exception
    assert [m.label for m in at.metric][4:8] == ["Sensors in scope", "Observed", "Assessable", "Alerting"]
    assert int(at.metric[4].value) == 444
    other = int(next(o for o in options if "documented case" not in o))
    assert other not in cases
    at.selectbox[0].set_value(other).run()
    assert not at.exception
    assert any(f"--run-id {other} " in info.value for info in at.info)


def test_outcome_toggle_adds_labels():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    assert any("Outcome:" in md.value for md in at.markdown)


def test_missing_artifacts_show_an_error_instead_of_crashing(tmp_path, monkeypatch):
    partial = tmp_path / "outputs"
    shutil.copytree(OUTPUT_DIR, partial, ignore=shutil.ignore_patterns("sensor_evidence.csv"))
    monkeypatch.setenv("SECOM_OUTPUTS_DIR", str(partial))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert at.error and "Artifacts are missing" in at.error[0].value
    assert any("sensor_evidence.csv" in md.value for md in at.markdown)


def test_cases_outside_the_date_range_are_not_offered():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.sidebar.slider[0].set_value((dt.date(2008, 7, 19), dt.date(2008, 7, 20))).run()
    assert not at.exception
    options = at.selectbox[0].options
    assert options and not any("documented case" in o for o in options)
    assert not {int(o) for o in options} & case_ids()
    assert any("outside the selected date range" in info.value for info in at.info)


def test_regenerated_artifacts_are_not_served_from_a_stale_cache(tmp_path, monkeypatch):
    copy = tmp_path / "outputs"
    shutil.copytree(OUTPUT_DIR, copy)
    monkeypatch.setenv("SECOM_OUTPUTS_DIR", str(copy))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert int(at.metric[0].value) == 1567
    workload = copy / "investigation" / "alert_workload.csv"
    lines = workload.read_text().splitlines()
    workload.write_text("\n".join(line for line in lines if not line.startswith("2008-10")) + "\n")  # drop October
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert int(at.metric[0].value) == 1567 - 359

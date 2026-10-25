"""Read-only prototype view of the SECOM alert-investigation artifacts.

Run: uv run streamlit run app.py
Reads files under outputs/ only: no database connection, no training, no live
data. Filters change what is shown, never a score or a threshold.
"""

import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from evidence import CHECKED_COLUMN, REASONS, SCORE_COLUMN, summarise, threshold_from_metrics  # noqa: E402
from secom_data import OUTPUT_DIR  # noqa: E402

OUT = Path(os.environ.get("SECOM_OUTPUTS_DIR", OUTPUT_DIR))  # re-read per run so AppTest can redirect it
ARTIFACTS = {  # name -> (path, command that regenerates it)
    "run_scores": (OUT / "sql" / "spc_run_scores.csv", "uv run python src/run_sql.py"),
    "metrics": (OUT / "metrics.csv", "uv run python src/train_evaluate.py"),
    "differences": (OUT / "ap_differences.csv", "uv run python src/train_evaluate.py"),
    "workload": (OUT / "investigation" / "alert_workload.csv", "uv run python src/build_investigation_report.py"),
    "evidence": (OUT / "investigation" / "sensor_evidence.csv", "uv run python src/build_investigation_report.py"),
    "manifest": (OUT / "investigation" / "manifest.json", "uv run python src/build_investigation_report.py"),
    "audit": (OUT / "data_audit.json", "uv run python src/download_or_prepare.py"),
    "pr_figure": (OUT / "figures" / "pr_curve_test.png", "uv run python src/make_plots.py"),
}
METHODS = {"spc_frozen": "SPC 3σ, frozen train limits", "spc_rolling": "SPC 3σ, rolling past-only"}
DISPLAY = {**METHODS, "isolation_forest": "Isolation Forest", "logistic_regression": "Logistic regression (supervised reference)"}
# Chart colours follow the theme in .streamlit/config.toml (fallbacks match the figures)
BAR = st.get_option("theme.primaryColor") or "#2a78d6"
LINE = st.get_option("theme.textColor") or "#52514e"
HIGHLIGHT = "#dde6f1"  # row tint for alerting sensors; text keeps its own colour


def artifact_stamp() -> tuple:
    """Paths and modification times of every artifact: the cache key, so that a
    regenerated file or a different outputs directory is never served stale."""
    return tuple((str(path), path.stat().st_mtime_ns) for path, _ in ARTIFACTS.values())


@st.cache_data
def load_artifacts(stamp: tuple) -> dict:
    data = {
        "run_scores": pd.read_csv(ARTIFACTS["run_scores"][0], parse_dates=["ts"]),
        "metrics": pd.read_csv(ARTIFACTS["metrics"][0]),
        "differences": pd.read_csv(ARTIFACTS["differences"][0]),
        "workload": pd.read_csv(ARTIFACTS["workload"][0], parse_dates=["day"]),
        "evidence": pd.read_csv(ARTIFACTS["evidence"][0]),
        "manifest": json.loads(ARTIFACTS["manifest"][0].read_text()),
        "audit": json.loads(ARTIFACTS["audit"][0].read_text()),
    }
    data["thresholds"] = {m: threshold_from_metrics(data["metrics"], m) for m in METHODS}
    return data


def missing_artifacts() -> list[tuple[str, Path, str]]:
    return [(name, path, cmd) for name, (path, cmd) in ARTIFACTS.items() if not path.exists()]


def header(data: dict) -> None:
    st.title("SECOM alert investigation")
    st.warning("Historical SECOM data (July to October 2008), offline prototype on exported artifacts. "
               "Not a live feed, and an alert is evidence to investigate, not a root cause.", icon="ℹ️")
    audit, manifest = data["audit"], data["manifest"]
    st.caption(
        f"Data: {audit['n_runs']} runs x {audit['n_sensors']} sensors, {audit['ts_min']} to {audit['ts_max']}, "
        f"archive SHA-256 {audit['archive_sha256'][:12]}. "
        f"Run-level thresholds: frozen {data['thresholds']['spc_frozen']:g}, "
        f"rolling {data['thresholds']['spc_rolling']:g} ({manifest['threshold_source']})."
    )


def sidebar(data: dict) -> dict:
    days = data["workload"]["day"].dt.date
    st.sidebar.header("Filters")
    st.sidebar.caption("Filters change what is shown. No score or threshold is recomputed.")
    date_range = st.sidebar.slider("Date range", min_value=days.min(), max_value=days.max(),
                                   value=(days.min(), days.max()))
    method = st.sidebar.radio("Rule", list(METHODS), format_func=METHODS.get)
    outcome = st.sidebar.toggle("Show retrospective outcomes (pass/fail labels)", value=False)
    return {"start": date_range[0], "end": date_range[1], "method": method, "outcome": outcome}


def overview_tab(data: dict, f: dict) -> None:
    w = data["workload"]
    w = w[(w["method"] == f["method"]) & (w["day"].dt.date >= f["start"]) & (w["day"].dt.date <= f["end"])]
    if w.empty:
        st.info("No runs in the selected date range.")
        return
    n_runs, n_assessable, n_alert = int(w["n_runs"].sum()), int(w["n_assessable"].sum()), int(w["n_alert"].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Runs", n_runs)
    c2.metric("Assessable runs", n_assessable, help="Runs with at least one sensor that could be evaluated")
    c3.metric("Alerts", n_alert)
    c4.metric("Alerts per 100 assessable", f"{100 * n_alert / n_assessable:.1f}" if n_assessable else "n/a",
              help="Alerts per 100 assessable runs in the selected range")
    daily = w.groupby("day")[["n_runs", "n_alert", "n_fail"]].sum().reset_index()
    st.subheader("Alerts per day")
    if n_alert == 0:
        st.info("No alerts in this range.")
    test_start = data["run_scores"].loc[data["run_scores"]["split"] == "test", "ts"].min()  # first test run
    bars = alt.Chart(daily).mark_bar(color=BAR).encode(
        x=alt.X("day:T", title=None),
        y=alt.Y("n_alert:Q", title="Alerts"),
        tooltip=[alt.Tooltip("day:T", title="Day"), alt.Tooltip("n_alert:Q", title="Alerts"),
                 alt.Tooltip("n_runs:Q", title="Runs")],
    )
    chart = bars
    if f["start"] <= test_start.date() <= f["end"]:
        boundary = pd.DataFrame({"day": [test_start], "y": [max(1, int(daily["n_alert"].max()))], "label": ["test period →"]})
        rule = alt.Chart(boundary).mark_rule(color=LINE, strokeDash=[4, 4]).encode(x="day:T")
        label = alt.Chart(boundary).mark_text(align="left", dx=4, color=LINE, fontSize=11).encode(
            x="day:T", y="y:Q", text="label")
        chart = bars + rule + label
    st.altair_chart(chart.properties(height=240), width="stretch")
    st.caption(f"Dashed line: first test-period run, {test_start:%Y-%m-%d %H:%M}; earlier runs on that day "
               "belong to the training period. Thresholds come from the training period.")
    if f["outcome"]:
        st.subheader("Daily fail rate (retrospective)")
        by_day = daily.set_index("day")
        st.line_chart(by_day["n_fail"] / by_day["n_runs"], color=LINE, height=200)


def investigation_tab(data: dict, f: dict) -> None:
    rs, threshold = data["run_scores"], data["thresholds"][f["method"]]
    score_col, checked_col = SCORE_COLUMN[f["method"]], CHECKED_COLUMN[f["method"]]
    in_range = rs[(rs["ts"].dt.date >= f["start"]) & (rs["ts"].dt.date <= f["end"])]
    alerted = in_range[(in_range[score_col] >= threshold) & (in_range[checked_col] > 0)]
    evidence = data["evidence"]
    available = sorted(evidence.loc[evidence["method"] == f["method"], "run_id"].unique())

    st.info(
        "**Alert list** (below): every run in the selected range whose run-level score is at or above the "
        f"threshold ({threshold:g} sensors out of control under {METHODS[f['method']]}).  \n"
        "**Sensor evidence** (further down): pre-exported for the three documented cases only, under both rules. "
        "They are the highest frozen score among alerted-and-failed, alerted-and-passed and not-alerted-and-failed "
        "test runs.  \n"
        "**Any other run**: `uv run python src/investigate_run.py --run-id <id> --method <rule>` builds the same "
        "table from the database."
    )
    st.subheader(f"Alerted runs: {len(alerted)} of {len(in_range)} in range")
    columns = ["run_id", "ts", "split", score_col, checked_col] + (["is_fail"] if f["outcome"] else [])
    if alerted.empty:
        st.info("No alerts in this range.")
    else:
        st.dataframe(
            alerted[columns].rename(columns={score_col: "score", checked_col: "sensors_checked"}),
            hide_index=True, height=240, width="stretch",
            column_config={"run_id": st.column_config.NumberColumn("run", format="%d"),
                           "ts": st.column_config.DatetimeColumn("time stamp", format="YYYY-MM-DD HH:mm"),
                           "score": st.column_config.NumberColumn("sensors out of control", format="%d"),
                           "sensors_checked": st.column_config.NumberColumn("sensors checked", format="%d"),
                           "is_fail": st.column_config.CheckboxColumn("failed (retrospective)")},
        )

    st.subheader("Sensor evidence for one run")
    in_range_ids = set(in_range["run_id"])
    cases_in_range = [r for r in available if r in in_range_ids]
    if available and not cases_in_range:
        st.info(f"The documented cases (runs {', '.join(str(r) for r in available)}, all in the test period) "
                "are outside the selected date range. Widen the range to open their sensor evidence.")
    candidates = sorted(set(cases_in_range) | set(alerted["run_id"]))
    if not candidates:
        st.info("No run to investigate in this range.")
        return
    run_id = st.selectbox("Run", candidates,
                          index=candidates.index(cases_in_range[0]) if cases_in_range else 0,
                          format_func=lambda r: f"{r} (documented case)" if r in available else str(r))
    row = rs.set_index("run_id").loc[run_id]
    score = int(row[score_col])
    verdict = "ALERT" if score >= threshold and row[checked_col] > 0 else "no alert"
    line = f"Run {run_id} at {row['ts']:%Y-%m-%d %H:%M} ({row['split']} period): score {score}, threshold {threshold:g}, {verdict}."
    if f["outcome"]:
        line += f" Outcome: {'fail' if row['is_fail'] else 'pass'}."
    st.write(line)
    if run_id not in available:
        st.info(f"Sensor evidence for run {run_id} is not pre-generated. Produce it with:\n\n"
                f"`uv run python src/investigate_run.py --run-id {run_id} --method {f['method']}`")
        return
    ev = evidence[(evidence["run_id"] == run_id) & (evidence["method"] == f["method"])]
    s = summarise(ev)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sensors in scope", s["sensor_scope_count"])
    c2.metric("Observed", s["n_observed"], help="Reading present")
    c3.metric("Assessable", s["n_assessable"], help="Reading present and a band exists")
    c4.metric("Alerting", s["n_alerted"], help="Outside the band")
    reasons = ", ".join(f"{r} {s[f'n_{r}']}" for r in REASONS if s[f"n_{r}"])
    st.caption(f"Not assessable: {reasons if reasons else 'none'}.")

    is_alert = ev["is_sensor_alert"].fillna(False).astype(bool)  # NULL means not assessable, so not alerting
    col1, col2 = st.columns([1, 3])
    only_alerting = col1.checkbox("Alerting sensors only", value=False)
    table = ev[is_alert] if only_alerting else ev
    if table.empty:
        st.info("No sensor is outside its band for this run.")
        return
    top = len(table) if len(table) <= 5 else col2.slider("Sensors to show (most deviant first)", 5, len(table), min(20, len(table)))
    show = ["sensor_id", "value", "baseline_mean", "baseline_std", "lower_limit", "upper_limit",
            "z_score", "is_sensor_alert", "unassessable_reason", "history_count"]
    view = table[show].head(top)

    def highlight(row):
        return [f"background-color: {HIGHLIGHT}" if is_alert[row.name] else ""] * len(row)

    st.dataframe(
        view.style.apply(highlight, axis=1), hide_index=True, width="stretch",
        column_config={
            "sensor_id": st.column_config.NumberColumn("sensor", format="%d"),
            "value": st.column_config.NumberColumn("value", format="%.4g"),
            "baseline_mean": st.column_config.NumberColumn("baseline mean", format="%.4g"),
            "baseline_std": st.column_config.NumberColumn("baseline std", format="%.4g"),
            "lower_limit": st.column_config.NumberColumn("lower limit", format="%.4g"),
            "upper_limit": st.column_config.NumberColumn("upper limit", format="%.4g"),
            "z_score": st.column_config.NumberColumn("z", format="%.2f"),
            "is_sensor_alert": st.column_config.CheckboxColumn("alert"),
            "unassessable_reason": st.column_config.TextColumn("not assessable"),
            "history_count": st.column_config.NumberColumn("history", format="%d"),
        },
    )
    st.caption("Highlighted rows are sensors outside their band. The limits are baseline mean ± 3 standard "
               "deviations; history is the number of readings behind the band.")


def evaluation_tab(data: dict) -> None:
    m = data["metrics"]
    test = m[m["split"] == "test"].copy()
    test["method"] = test["method"].map(DISPLAY)
    n_runs, n_fail, prevalence = int(test["n_runs"].iloc[0]), int(test["n_fail"].iloc[0]), float(test["prevalence"].iloc[0])
    st.subheader("Average precision on the test period")
    st.caption(f"{n_runs} runs, {n_fail} fails. Random ranking scores AP {prevalence:.4f}, the test fail rate. "
               "Intervals are 95% paired bootstrap intervals; precision and recall are at the training-derived threshold.")
    st.dataframe(
        test[["method", "role", "average_precision", "ap_ci_low", "ap_ci_high", "precision", "recall"]],
        hide_index=True, width="stretch",
        column_config={"method": "method", "role": "role",
                       "average_precision": st.column_config.NumberColumn("AP", format="%.3f"),
                       "ap_ci_low": st.column_config.NumberColumn("AP CI low", format="%.3f"),
                       "ap_ci_high": st.column_config.NumberColumn("AP CI high", format="%.3f"),
                       "precision": st.column_config.NumberColumn("precision", format="%.3f"),
                       "recall": st.column_config.NumberColumn("recall", format="%.3f")},
    )
    rolling_ap = float(test.loc[test["role"] == "supplementary", "average_precision"].iloc[0])
    frozen_ap = float(test.loc[test["method"] == DISPLAY["spc_frozen"], "average_precision"].iloc[0])
    includes_baseline = (test["ap_ci_low"] <= prevalence) & (prevalence <= test["ap_ci_high"])
    above = test.loc[~includes_baseline, "method"].tolist()
    baseline_note = ("Every method's interval includes the random baseline." if not above else
                     "Intervals that exclude the random baseline: " + ", ".join(above) + ".")
    st.markdown(
        f"{baseline_note} "
        f"The rolling rule's AP ({rolling_ap:.3f}) is not evidence that it beats the frozen rule ({frozen_ap:.3f}): "
        "it updates its band with earlier test-period readings, so it follows a different protocol."
    )
    st.subheader("Precision-recall curves")
    st.image(str(ARTIFACTS["pr_figure"][0]))
    with st.expander("Paired AP differences and confusion counts"):
        st.dataframe(data["differences"], hide_index=True, width="stretch",
                     column_config={"ap_difference": st.column_config.NumberColumn("AP difference", format="%+.3f"),
                                    "ci_low": st.column_config.NumberColumn("CI low", format="%+.3f"),
                                    "ci_high": st.column_config.NumberColumn("CI high", format="%+.3f")})
        st.dataframe(test[["method", "alert_rate", "tp", "fp", "fn", "tn"]], hide_index=True, width="stretch",
                     column_config={"alert_rate": st.column_config.NumberColumn("alert rate", format="%.3f")})
    st.caption(
        "Threshold rule: for each method, the training-period score quantile at which the training alert rate "
        "equals the training fail rate. See the README for the full evaluation protocol."
    )


def main() -> None:
    st.set_page_config(page_title="SECOM alert investigation", layout="wide")
    missing = missing_artifacts()
    if missing:
        st.error("Artifacts are missing. Generate them, then reload:")
        for name, path, cmd in missing:
            st.write(f"- `{path.relative_to(OUT.parent) if OUT.parent in path.parents else path}` ({name}): `{cmd}`")
        st.stop()
    try:
        data = load_artifacts(artifact_stamp())
    except LookupError as exc:
        st.error(f"{ARTIFACTS['metrics'][0]} is incomplete ({exc}); run `{ARTIFACTS['metrics'][1]}`")
        st.stop()
    header(data)
    f = sidebar(data)
    tab_overview, tab_investigation, tab_evaluation = st.tabs(["Overview", "Investigation", "Evaluation"])
    with tab_overview:
        overview_tab(data, f)
    with tab_investigation:
        investigation_tab(data, f)
    with tab_evaluation:
        evaluation_tab(data)


main()

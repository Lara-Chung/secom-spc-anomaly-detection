"""Build the alert-investigation artifacts from the database and Phase 1 outputs.

Usage: uv run python src/build_investigation_report.py   (after src/train_evaluate.py)
Writes outputs/investigation/{manifest.json, alert_workload.csv, sensor_evidence.csv}
and docs/investigation_examples.md. Nothing here changes a score or a threshold.
"""

import json
import platform
from importlib.metadata import version

import pandas as pd

from db import connect, fetch
from evidence import check_consistent, summarise
from investigate_run import EVIDENCE_SQL, SQL_DIR, run_header, run_threshold, sensor_evidence
from secom_data import OUTPUT_DIR, REPO_ROOT

INVESTIGATION_DIR = OUTPUT_DIR / "investigation"
DOCS_DIR = REPO_ROOT / "docs"
CASE_METHOD = "spc_frozen"  # cases are chosen under the primary, frozen-limit rule
TOP_SENSORS = 10

# Case groups on the test period: (name, alerted, failed). Within each group the
# run with the highest score is taken; ties go to the earliest (ts, run_id).
CASE_GROUPS = [
    ("alerted_and_failed", True, True),
    ("alerted_and_passed", True, False),
    ("not_alerted_and_failed", False, True),
]

NEXT_STEPS = {
    "alerted_and_failed": [
        "Check whether the sensors listed above belong to one tool or process step; SECOM does not name them, so this needs the sensor map.",
        "Compare the same sensors on the runs just before this one to see whether the deviation built up or appeared at once.",
    ],
    "alerted_and_passed": [
        "The alert is not necessarily wrong: the run passed the final test, but the readings did deviate. Check whether the deviating sensors are known to be noisy or recently recalibrated.",
        "If this pattern repeats on passing runs, the sensors involved are candidates for a wider band or exclusion from the run score.",
    ],
    "not_alerted_and_failed": [
        "The run stayed below the run-level threshold: the sensors that did cross their limits are listed above. The cause of the failure is either not in these sensors or spread across sensors rather than concentrated in enough large deviations.",
        "Look at the sensors with the largest |z| anyway and at the missing readings; a run with many missing values has fewer chances to alert.",
    ],
}


def manifest(thresholds: dict) -> dict:
    """What the artifacts were built from. The code version is the git commit
    that contains them; recording it here would be circular."""
    audit = json.loads((OUTPUT_DIR / "data_audit.json").read_text())
    return {
        "data_archive_sha256": audit["archive_sha256"],
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("pandas", "numpy", "scikit-learn", "psycopg")},
        "run_level_thresholds": thresholds,
        "threshold_source": "training-period score quantile at the training fail rate (outputs/metrics.csv)",
        "case_selection": f"{CASE_METHOD} on the test period; highest score per outcome group, ties by earliest (ts, run_id)",
    }


def alert_workload(conn, thresholds: dict) -> pd.DataFrame:
    sql = (SQL_DIR / "alert_workload.sql").read_text()
    return fetch(conn, sql, {"frozen_threshold": thresholds["spc_frozen"], "rolling_threshold": thresholds["spc_rolling"]})


def check_against_metrics(workload: pd.DataFrame) -> pd.DataFrame:
    """Totals per method and split must agree with the confusion matrices in metrics.csv.

    metrics.csv scores a run with no assessable sensor as 0 (no alert), so it
    counts such runs as TN or FN; the workload report lists them separately.
    """
    columns = ["n_runs", "n_assessable", "n_not_assessable", "n_not_assessable_fail", "n_alert", "tp", "fp", "fn", "tn"]
    totals = workload.groupby(["method", "split"])[columns].sum()
    metrics = pd.read_csv(OUTPUT_DIR / "metrics.csv").set_index(["method", "split"])
    for key, row in totals.iterrows():
        m = metrics.loc[key]
        unassessable_pass = row["n_not_assessable"] - row["n_not_assessable_fail"]
        agree = (row["tp"] == m["tp"] and row["fp"] == m["fp"]
                 and row["fn"] + row["n_not_assessable_fail"] == m["fn"]
                 and row["tn"] + unassessable_pass == m["tn"])
        if not agree:
            raise RuntimeError(f"alert_workload totals for {key} disagree with metrics.csv")
    totals["precision"] = totals["tp"] / (totals["tp"] + totals["fp"])
    totals["recall_assessable"] = totals["tp"] / (totals["tp"] + totals["fn"])
    return totals.reset_index()


def select_cases(conn, threshold: float) -> list[dict]:
    runs = fetch(conn, "SELECT run_id, ts, is_fail, frozen_n_alerts AS score FROM spc_run_scores "
                       "WHERE split = 'test' ORDER BY ts, run_id")
    runs["is_alert"] = runs["score"] >= threshold
    cases = []
    for name, alerted, failed in CASE_GROUPS:
        group = runs[(runs["is_alert"] == alerted) & (runs["is_fail"] == failed)]
        if group.empty:
            cases.append({"group": name, "run_id": None})
            continue
        best = group.sort_values(["score", "ts", "run_id"], ascending=[False, True, True]).iloc[0]
        cases.append({"group": name, "run_id": int(best["run_id"])})
    return cases


def markdown_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    # astype(object) keeps each column's own type; iterating rows would upcast ints to float
    for row in frame.astype(object).itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else f"{v:.4g}")
            else:
                cells.append("" if v is None else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_examples(cases: list[dict], headers: dict, summaries: dict, evidence: dict,
                   totals: pd.DataFrame, threshold: float) -> str:
    out = [
        "# Alert investigation examples",
        "",
        "Three runs from the test period, chosen by a fixed rule before looking at their sensors: "
        f"under the frozen 3σ rule (run alert when at least {threshold:g} sensors are out of control), "
        "the highest-scoring run in each outcome group. They illustrate what the investigation query "
        "returns; they are not a performance sample. All numbers come from "
        "`uv run python src/build_investigation_report.py`; reproduce any table with "
        "`uv run python src/investigate_run.py --run-id <id> --method spc_frozen --outcome`.",
        "",
        "SECOM does not name its sensors, so no table below says what a sensor measures. "
        "A large deviation is evidence to investigate, not a root cause.",
        "",
        "## Alert workload on the test period",
        "",
        "Totals from `outputs/investigation/alert_workload.csv` (per-day rows in the file):",
        "",
        markdown_table(totals[totals["split"] == "test"].drop(columns=["split"])),
        "",
    ]
    titles = {
        "alerted_and_failed": "Case 1: alerted, and the run failed",
        "alerted_and_passed": "Case 2: alerted, but the run passed",
        "not_alerted_and_failed": "Case 3: not alerted, but the run failed",
    }
    for case in cases:
        out += [f"## {titles[case['group']]}", ""]
        if case["run_id"] is None:
            out += ["No run in the test period falls in this group.", ""]
            continue
        h, s, ev = headers[case["run_id"]], summaries[case["run_id"]], evidence[case["run_id"]]
        out += [
            f"Run {h['run_id']} at {h['ts']}: {h['score']} sensors out of control "
            f"(threshold {h['threshold']:g}, {'alert' if h['is_alert'] else 'no alert'}); "
            f"outcome {'fail' if h['is_fail'] else 'pass'}.",
            "",
            f"Sensors in scope {s['sensor_scope_count']}, observed {s['n_observed']}, "
            f"assessable {s['n_assessable']}, alerting {s['n_alerted']}, missing {s['n_missing_value']}.",
            "",
            f"Top {TOP_SENSORS} sensors by |z| (frozen training limits):",
            "",
            markdown_table(ev[["sensor_id", "value", "baseline_mean", "baseline_std", "lower_limit", "upper_limit",
                               "z_score", "is_sensor_alert"]].head(TOP_SENSORS)),
            "",
            "What to check next:",
            "",
            *[f"- {step}" for step in NEXT_STEPS[case["group"]]],
            "",
        ]
    return "\n".join(out)


def main() -> None:
    # Read every input (metrics.csv, data_audit.json, the database) before writing anything,
    # so a missing input cannot leave a half-regenerated outputs/investigation/
    thresholds = {m: run_threshold(m) for m in EVIDENCE_SQL}
    manifest_data = manifest(thresholds)
    with connect() as conn:
        workload = alert_workload(conn, thresholds)
        totals = check_against_metrics(workload)
        cases = select_cases(conn, thresholds[CASE_METHOD])
        # Evidence for every case under both rules (the read-only app can show either),
        # each checked against the run score; the document uses the CASE_METHOD entries
        headers, summaries, evidence, frames = {}, {}, {}, []
        selected = [c for c in cases if c["run_id"] is not None]
        for c in selected:
            rid = c["run_id"]
            for m in EVIDENCE_SQL:
                header, ev = run_header(conn, rid, m), sensor_evidence(conn, rid, m)
                summary = summarise(ev)
                check_consistent(header, summary)
                frames.append(ev.assign(case_group=c["group"]))
                if m == CASE_METHOD:
                    headers[rid], summaries[rid], evidence[rid] = header, summary, ev
        if frames:
            all_evidence = pd.concat(frames)
        else:  # an empty result still carries the query's columns, so the export stays in step
            all_evidence = sensor_evidence(conn, -1, CASE_METHOD).assign(case_group=pd.Series(dtype=str))

    INVESTIGATION_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    workload.to_csv(INVESTIGATION_DIR / "alert_workload.csv", index=False)
    all_evidence.to_csv(INVESTIGATION_DIR / "sensor_evidence.csv", index=False)
    (INVESTIGATION_DIR / "manifest.json").write_text(json.dumps(manifest_data, indent=2) + "\n")
    (DOCS_DIR / "investigation_examples.md").write_text(
        write_examples(cases, headers, summaries, evidence, totals, thresholds[CASE_METHOD]) + "\n")

    print(f"alert_workload.csv: {len(workload)} rows; totals agree with metrics.csv")
    print(totals.round(3).to_string(index=False))
    print("cases: " + ", ".join(f"{c['group']}={c['run_id']}" for c in cases))
    print(f"wrote {INVESTIGATION_DIR.relative_to(REPO_ROOT)}/ and docs/investigation_examples.md")


if __name__ == "__main__":
    main()

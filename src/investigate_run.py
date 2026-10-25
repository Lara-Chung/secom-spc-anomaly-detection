"""Show the sensor evidence behind one run's SPC score.

Usage: uv run python src/investigate_run.py --run-id 1458 --method spc_frozen [--top 15] [--csv out.csv] [--outcome]
Needs the database views (src/run_sql.py) and outputs/metrics.csv (src/train_evaluate.py)
for the run-level threshold. The sensor alerts must add up to the run score in
spc_run_scores. The pass/fail label is shown only with --outcome.
"""

import argparse
import sys

import pandas as pd
import psycopg

from db import connect, fetch
from secom_data import OUTPUT_DIR, REPO_ROOT

SQL_DIR = REPO_ROOT / "sql" / "investigation"
# Method allowlist: each entry is a parameterised SELECT bound with a run_id parameter
EVIDENCE_SQL = {
    "spc_frozen": SQL_DIR / "run_sensor_evidence_frozen.sql",
    "spc_rolling": SQL_DIR / "run_sensor_evidence_rolling.sql",
}
SCORE_COLUMN = {"spc_frozen": "frozen_n_alerts", "spc_rolling": "rolling_n_alerts"}
REASONS = ("missing_value", "insufficient_history", "flat_window")


class InvestigationError(Exception):
    """A request that cannot be served: unknown run, or a missing artifact."""


def run_threshold(method: str) -> float:
    """Run-level alert threshold chosen on the training period (outputs/metrics.csv)."""
    path = OUTPUT_DIR / "metrics.csv"
    if not path.exists():
        raise InvestigationError(f"{path} not found; run `uv run python src/train_evaluate.py` first")
    metrics = pd.read_csv(path)
    row = metrics[(metrics["method"] == method) & (metrics["split"] == "train")]
    if row.empty:
        raise InvestigationError(f"{path} has no training row for {method}; rerun `uv run python src/train_evaluate.py`")
    return float(row["threshold"].iloc[0])


def check_method(method: str) -> None:
    if method not in EVIDENCE_SQL:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(EVIDENCE_SQL)}")


def run_header(conn, run_id: int, method: str) -> dict:
    """Run-level context: time stamp, split, score and alert decision."""
    check_method(method)
    rows = fetch(conn, "SELECT * FROM spc_run_scores WHERE run_id = %(run_id)s", {"run_id": run_id})
    if rows.empty:
        raise InvestigationError(f"run_id {run_id} is not in spc_run_scores; if it exists in runs, "
                                 "the materialized views are stale: run `uv run python src/run_sql.py`")
    row = rows.iloc[0]
    threshold = run_threshold(method)
    score = int(row[SCORE_COLUMN[method]])
    return {
        "run_id": run_id,
        "ts": row["ts"],
        "split": row["split"],
        "method": method,
        "score": score,
        "threshold": threshold,
        "is_alert": score >= threshold,
        "is_fail": bool(row["is_fail"]),
    }


def sensor_evidence(conn, run_id: int, method: str) -> pd.DataFrame:
    """One row per eligible sensor, most deviant first; NULL z-scores sort last."""
    check_method(method)
    evidence = fetch(conn, EVIDENCE_SQL[method].read_text(), {"run_id": run_id})
    # A run whose readings are all NULL would otherwise come back as object columns
    numeric = ["value", "baseline_mean", "baseline_std", "lower_limit", "upper_limit", "z_score", "abs_z_score"]
    evidence[numeric] = evidence[numeric].astype(float)
    return evidence


def summarise(evidence: pd.DataFrame) -> dict:
    """Counts with explicit denominators: scope >= observed >= assessable >= alerted."""
    reasons = evidence["unassessable_reason"].value_counts()
    return {
        "sensor_scope_count": len(evidence),
        "n_observed": int(evidence["value"].notna().sum()),
        "n_assessable": int(evidence["is_assessable"].sum()),
        "n_alerted": int(evidence["is_sensor_alert"].fillna(False).sum()),
        **{f"n_{reason}": int(reasons.get(reason, 0)) for reason in REASONS},
    }


def check_consistent(header: dict, summary: dict) -> None:
    """The sensor-level alerts must add up to the run-level score from the view."""
    if summary["n_alerted"] != header["score"]:
        raise RuntimeError(f"run {header['run_id']}: sensor alerts ({summary['n_alerted']}) disagree "
                           f"with the run score ({header['score']})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--method", choices=sorted(EVIDENCE_SQL), required=True)
    parser.add_argument("--top", type=int, default=15, help="sensors to print (most deviant first)")
    parser.add_argument("--csv", help="write the full evidence table to this path")
    parser.add_argument("--outcome", action="store_true", help="also show the pass/fail label")
    args = parser.parse_args(argv)

    try:
        with connect() as conn:
            header = run_header(conn, args.run_id, args.method)
            evidence = sensor_evidence(conn, args.run_id, args.method)
    except psycopg.OperationalError as exc:
        print(f"cannot connect to PostgreSQL ({exc}); run `docker compose up -d --wait`", file=sys.stderr)
        return 2
    except psycopg.errors.UndefinedTable as exc:
        print(f"database views are missing or outdated ({exc.diag.message_primary}); "
              "run `uv run python src/run_sql.py`", file=sys.stderr)
        return 2
    except InvestigationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    summary = summarise(evidence)
    check_consistent(header, summary)

    print(f"run {header['run_id']}  ts {header['ts']}  split {header['split']}  method {header['method']}")
    print(f"run score {header['score']} sensors out of control; threshold {header['threshold']:g} "
          f"-> {'ALERT' if header['is_alert'] else 'no alert'}")
    if args.outcome:
        print(f"outcome (retrospective): {'fail' if header['is_fail'] else 'pass'}")
    print("sensors: " + ", ".join(f"{k} {v}" for k, v in summary.items()))
    if summary["n_assessable"] == 0:
        print("no sensor is assessable for this run")

    columns = ["sensor_id", "value", "baseline_mean", "baseline_std", "lower_limit", "upper_limit",
               "z_score", "is_sensor_alert", "unassessable_reason", "history_count"]
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(evidence[columns].head(args.top).to_string(index=False))
    if args.csv:
        evidence.to_csv(args.csv, index=False)
        print(f"wrote {args.csv} ({len(evidence)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

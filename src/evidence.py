"""Pure helpers for sensor-evidence tables, shared by the CLI, the report builder
and the read-only app. No database or file access here."""

import pandas as pd

SCORE_COLUMN = {"spc_frozen": "frozen_n_alerts", "spc_rolling": "rolling_n_alerts"}
CHECKED_COLUMN = {"spc_frozen": "frozen_n_checked", "spc_rolling": "rolling_n_checked"}
REASONS = ("missing_value", "insufficient_history", "flat_window")


def threshold_from_metrics(metrics: pd.DataFrame, method: str) -> float:
    """Run-level alert threshold chosen on the training period, from the metrics table."""
    row = metrics[(metrics["method"] == method) & (metrics["split"] == "train")]
    if row.empty:
        raise LookupError(f"metrics table has no training row for {method!r}")
    return float(row["threshold"].iloc[0])


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

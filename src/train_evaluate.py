"""Fit Isolation Forest on the training period and compare it with the SQL SPC scores.

Usage: uv run python src/train_evaluate.py   (after src/run_sql.py)
Reads the feature_export and spc_run_scores views.
Writes outputs/metrics.csv, outputs/ap_differences.csv and outputs/run_scores.csv.

All scores are oriented so that higher = more anomalous. Fail (is_fail) is the
positive class. Every threshold is derived from training-period scores only.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.pipeline import make_pipeline

from db import connect
from secom_data import OUTPUT_DIR

SEED = 0
N_ESTIMATORS = 300
N_BOOTSTRAP = 2000

# method -> role in the comparison
METHODS = {
    "isolation_forest": "primary",   # fitted on train, frozen
    "spc_frozen": "primary",         # 3-sigma limits from train, frozen
    "spc_rolling": "supplementary",  # past-only rolling baseline (online monitoring)
}


def fetch(conn, query) -> pd.DataFrame:
    cur = conn.execute(query)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def load() -> tuple[pd.DataFrame, np.ndarray]:
    with connect() as conn:
        features = fetch(conn, "SELECT run_id, sensor_values FROM feature_export ORDER BY ts, run_id")
        runs = fetch(conn, "SELECT * FROM spc_run_scores ORDER BY ts, run_id")
    if not features["run_id"].equals(runs["run_id"]):
        raise RuntimeError("feature_export and spc_run_scores are not aligned by run")
    X = np.array(features["sensor_values"].tolist(), dtype=float)  # NULL -> NaN
    return runs, X


def score_runs(runs: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    train = (runs["split"] == "train").to_numpy()
    # No scaling: Isolation Forest picks split values between each feature's
    # min and max, so per-feature scale does not change the result
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        IsolationForest(n_estimators=N_ESTIMATORS, random_state=SEED),
    )
    model.fit(X[train])
    scores = runs[["run_id", "ts", "split", "is_fail"]].copy()
    scores["isolation_forest"] = -model.score_samples(X)  # negate: higher = more anomalous
    scores["spc_frozen"] = runs["frozen_n_alerts"]
    scores["spc_rolling"] = runs["rolling_n_alerts"]
    return scores


def evaluate(scores: pd.DataFrame) -> pd.DataFrame:
    train = scores[scores["split"] == "train"]
    # Threshold rule, fixed before looking at the test period: the alert rate on
    # train equals the train fail rate. Ties in integer SPC scores can make the
    # actual alert rate higher; it is reported as alert_rate.
    quantile = 1 - train["is_fail"].mean()
    rows = []
    for method, role in METHODS.items():
        threshold = float(np.quantile(train[method], quantile))
        for split in ("train", "test"):
            part = scores[scores["split"] == split]
            y = part["is_fail"].astype(int)
            alert = part[method] >= threshold
            tn, fp, fn, tp = confusion_matrix(y, alert, labels=[0, 1]).ravel()
            rows.append({
                "method": method,
                "role": role,
                "split": split,
                "n_runs": len(part),
                "n_fail": int(y.sum()),
                "prevalence": y.mean(),
                "average_precision": average_precision_score(y, part[method]),
                "threshold": threshold,
                "threshold_source": f"train quantile {quantile:.4f}",
                "alert_rate": alert.mean(),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": tp / (tp + fp) if tp + fp else np.nan,
                "recall": tp / (tp + fn),
            })
    return pd.DataFrame(rows)


def bootstrap_ap(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired bootstrap over test runs: 95% intervals for AP and for AP differences.

    Every method is scored on the same resample, so the differences are paired.
    """
    test = scores[scores["split"] == "test"]
    y = test["is_fail"].to_numpy(dtype=int)
    if y.sum() == 0:
        raise RuntimeError("Test period has no fail runs; AP is undefined")
    rng = np.random.default_rng(SEED)
    draws = []
    while len(draws) < N_BOOTSTRAP:
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:  # AP is undefined without a positive run
            continue
        draws.append({m: average_precision_score(y[idx], test[m].to_numpy()[idx]) for m in METHODS})
    draws = pd.DataFrame(draws)

    intervals = draws.quantile([0.025, 0.975]).T
    intervals.columns = ["ap_ci_low", "ap_ci_high"]
    methods = list(METHODS)
    diffs = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            d = draws[a] - draws[b]
            diffs.append({
                "comparison": f"{a} - {b}",
                "ap_difference": average_precision_score(y, test[a]) - average_precision_score(y, test[b]),
                "ci_low": d.quantile(0.025),
                "ci_high": d.quantile(0.975),
            })
    return intervals, pd.DataFrame(diffs)


def main() -> None:
    runs, X = load()
    scores = score_runs(runs, X)
    metrics = evaluate(scores)
    intervals, diffs = bootstrap_ap(scores)
    is_test = metrics["split"] == "test"
    metrics.loc[is_test, ["ap_ci_low", "ap_ci_high"]] = intervals.loc[metrics.loc[is_test, "method"]].to_numpy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_csv(OUTPUT_DIR / "run_scores.csv", index=False)
    # threshold is kept unrounded so that applying it to run_scores.csv reproduces tp/fp/fn/tn
    rounded = metrics.round({c: 4 for c in metrics.columns if c != "threshold"})
    rounded.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    diffs.round(4).to_csv(OUTPUT_DIR / "ap_differences.csv", index=False)
    print(f"features: {X.shape[1]} sensors, train runs: {(runs['split'] == 'train').sum()}")
    print(metrics.round(4).to_string(index=False))
    print(f"\nAP differences on test ({N_BOOTSTRAP} paired bootstrap resamples):")
    print(diffs.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

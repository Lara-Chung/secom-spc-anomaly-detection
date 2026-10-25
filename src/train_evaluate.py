"""Fit Isolation Forest on the training period and compare it with the SQL SPC scores.

Usage: uv run python src/train_evaluate.py   (after src/run_sql.py)
Reads the feature_export and spc_run_scores views.
Writes outputs/metrics.csv, outputs/ap_differences.csv, outputs/run_scores.csv
and outputs/reference_cv.csv.

All scores are oriented so that higher = more anomalous. Fail (is_fail) is the
positive class. Every threshold is derived from training-period scores only.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from db import connect
from secom_data import OUTPUT_DIR

SEED = 0
N_ESTIMATORS = 300
N_BOOTSTRAP = 2000
REFERENCE_C_GRID = (1.0, 0.1, 0.01, 0.001)  # inverse L2 strength, searched on train only
REFERENCE_CV_SPLITS = 4

# method -> role in the comparison
METHODS = {
    "isolation_forest": "primary",        # fitted on train, frozen
    "spc_frozen": "primary",              # 3-sigma limits from train, frozen
    "spc_rolling": "supplementary",       # past-only rolling baseline (online monitoring)
    "logistic_regression": "reference",   # supervised: uses training labels, upper reference
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


def reference_pipeline(C: float):
    """Supervised reference: median imputation, standardisation, L2 logistic regression."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", C=C, max_iter=1000),
    )


def score_runs(runs: pd.DataFrame, X: np.ndarray, reference_c: float = 1.0) -> pd.DataFrame:
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

    # Supervised reference: the only method that sees training labels. C comes
    # from select_reference_c (training-period CV); nothing else is searched
    reference = reference_pipeline(reference_c)
    reference.fit(X[train], runs.loc[train, "is_fail"].astype(int))
    scores["logistic_regression"] = reference.predict_proba(X)[:, 1]  # P(fail)
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


def reference_cv(runs: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    """Out-of-fold AP of the supervised reference for each C in REFERENCE_C_GRID,
    under a time-series split of the training period only. No test-period data."""
    train = (runs["split"] == "train").to_numpy()
    X_train = X[train]
    y_train = runs.loc[train, "is_fail"].astype(int).to_numpy()
    rows = []
    for C in REFERENCE_C_GRID:
        splits = TimeSeriesSplit(n_splits=REFERENCE_CV_SPLITS).split(X_train)
        for fold, (fit_idx, val_idx) in enumerate(splits):
            model = reference_pipeline(C).fit(X_train[fit_idx], y_train[fit_idx])
            y_val = y_train[val_idx]
            ap = np.nan  # AP is undefined for a fold without a fail run
            if y_val.sum() > 0:
                ap = average_precision_score(y_val, model.predict_proba(X_train[val_idx])[:, 1])
            rows.append({
                "C": C,
                "fold": fold,
                "n_val": len(val_idx),
                "n_fail_val": int(y_val.sum()),
                "prevalence_val": y_val.mean(),
                "average_precision": ap,
            })
    return pd.DataFrame(rows)


def select_reference_c(cv: pd.DataFrame) -> float:
    """C with the highest mean out-of-fold AP; ties go to the first value in the grid."""
    means = cv.groupby("C")["average_precision"].mean()
    return float(max(REFERENCE_C_GRID, key=lambda c: means[c]))


def main() -> None:
    runs, X = load()
    cv = reference_cv(runs, X)
    reference_c = select_reference_c(cv)
    scores = score_runs(runs, X, reference_c)
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
    cv.round(4).to_csv(OUTPUT_DIR / "reference_cv.csv", index=False)
    print(f"features: {X.shape[1]} sensors, train runs: {(runs['split'] == 'train').sum()}")
    print("supervised reference: training-period CV mean out-of-fold AP by C")
    print(cv.groupby("C")["average_precision"].mean().round(4).to_string())
    print(f"selected C = {reference_c}\n")
    print(metrics.round(4).to_string(index=False))
    print(f"\nAP differences on test ({N_BOOTSTRAP} paired bootstrap resamples):")
    print(diffs.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

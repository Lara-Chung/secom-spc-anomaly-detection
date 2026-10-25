"""Checks for score direction, label mapping and train-only fitting, on synthetic data."""

import numpy as np
import pandas as pd

from train_evaluate import evaluate, score_runs

N_TRAIN, N_TEST, N_SENSORS = 300, 100, 5


def synthetic():
    rng = np.random.default_rng(0)
    n = N_TRAIN + N_TEST
    X = rng.normal(0, 1, size=(n, N_SENSORS))
    X[rng.random(X.shape) < 0.05] = np.nan
    is_fail = np.zeros(n, dtype=bool)
    fail_idx = np.r_[10:310:30, 305:400:10]  # fails in both periods
    is_fail[fail_idx] = True
    X[fail_idx] += 8  # fails are obvious outliers on every sensor
    runs = pd.DataFrame({
        "run_id": range(n),
        "ts": pd.date_range("2008-07-19", periods=n, freq="h"),
        "split": ["train"] * N_TRAIN + ["test"] * N_TEST,
        "is_fail": is_fail,
        "frozen_n_alerts": is_fail.astype(int) * 5,
        "rolling_n_alerts": is_fail.astype(int) * 5,
    })
    return runs, X


def test_higher_score_means_more_anomalous_and_fail_is_positive():
    runs, X = synthetic()
    metrics = evaluate(score_runs(runs, X)).set_index(["method", "split"])
    # Obvious outliers labelled fail must rank at the top; a flipped score or
    # flipped label would push average precision towards or below prevalence
    assert metrics.loc[("isolation_forest", "test"), "average_precision"] > 0.9
    assert metrics.loc[("isolation_forest", "test"), "recall"] > 0.9


def test_test_period_values_do_not_affect_training_fit():
    runs, X = synthetic()
    before = score_runs(runs, X)
    X_changed = X.copy()
    X_changed[N_TRAIN:] = X_changed[N_TRAIN:] * 50 + 100
    after = score_runs(runs, X_changed)
    # Imputer and forest are fitted on train rows only, so train scores are identical
    pd.testing.assert_series_equal(
        before["isolation_forest"][:N_TRAIN], after["isolation_forest"][:N_TRAIN]
    )
    # Thresholds also come from train only
    pd.testing.assert_series_equal(evaluate(before)["threshold"], evaluate(after)["threshold"])

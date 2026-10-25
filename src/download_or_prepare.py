"""Download the UCI SECOM archive, verify it, extract it and write a data audit.

Usage: uv run python src/download_or_prepare.py
Outputs: data/raw/{secom.data, secom_labels.data, secom.names}, outputs/data_audit.json
"""

import hashlib
import json
import urllib.request
import zipfile

import pandas as pd

from secom_data import OUTPUT_DIR, RAW_DIR, read_features, read_labels

SOURCE_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
# SHA-256 of the archive as downloaded from SOURCE_URL on 2026-10-25
EXPECTED_SHA256 = "eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e"
EXPECTED_MEMBERS = {"secom.data", "secom_labels.data", "secom.names"}


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download_and_extract() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive = RAW_DIR / "secom.zip"
    if not archive.exists():
        # Download to a temporary name; only a verified file becomes secom.zip,
        # so an interrupted download never blocks later runs
        partial = archive.with_suffix(".zip.part")
        print(f"Downloading {SOURCE_URL}")
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as resp:
            partial.write_bytes(resp.read())
        digest = sha256(partial)
        if digest != EXPECTED_SHA256:
            raise RuntimeError(f"Checksum mismatch for downloaded file {partial}: {digest}")
        partial.replace(archive)
    else:
        digest = sha256(archive)
        if digest != EXPECTED_SHA256:
            raise RuntimeError(
                f"Checksum mismatch for existing {archive}: {digest}. "
                "Move this file out of data/raw/ and rerun to download a fresh copy."
            )
    with zipfile.ZipFile(archive) as zf:
        members = set(zf.namelist())
        if members != EXPECTED_MEMBERS:
            raise RuntimeError(f"Unexpected archive members: {sorted(members)}")
        zf.extractall(RAW_DIR)


def audit() -> dict:
    features = read_features()
    labels = read_labels()
    if len(features) != len(labels):
        raise RuntimeError("Feature and label files have different row counts")

    n_runs, n_sensors = features.shape
    missing = features.isna()
    missing_rate = missing.mean()
    ts = labels["ts"]
    n_fail = int((labels["label"] == 1).sum())

    return {
        "source_url": SOURCE_URL,
        "archive_sha256": EXPECTED_SHA256,
        "n_runs": n_runs,
        "n_sensors": n_sensors,
        "label_values": {str(k): int(v) for k, v in labels["label"].value_counts().sort_index().items()},
        "label_mapping": {"-1": "pass", "1": "fail"},
        "n_fail": n_fail,
        "fail_prevalence": n_fail / n_runs,
        "missing_cells": int(missing.values.sum()),
        "missing_cell_rate": float(missing.values.mean()),
        "runs_with_any_missing": int(missing.any(axis=1).sum()),
        "sensors_with_any_missing": int((missing_rate > 0).sum()),
        "sensors_missing_rate_gt_50pct": int((missing_rate > 0.5).sum()),
        "sensors_all_missing": int((missing_rate == 1).sum()),
        # Descriptive only (full data); model feature selection is fit on the training period
        "sensors_constant_full_data": int((features.nunique(dropna=True) <= 1).sum()),
        "ts_min": str(ts.min()),
        "ts_max": str(ts.max()),
        "ts_unparsed": int(ts.isna().sum()),
        "ts_sorted_in_file_order": bool(ts.is_monotonic_increasing),
        "ts_out_of_order_rows": int((ts.diff() < pd.Timedelta(0)).sum()),
        "ts_duplicate_rows": int(ts.duplicated(keep=False).sum()),
        "ts_distinct_values": int(ts.nunique()),
        "duplicate_feature_rows": int(features.duplicated(keep=False).sum()),
        "n_days_with_runs": int(ts.dt.normalize().nunique()),
    }


def main() -> None:
    download_and_extract()
    result = audit()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / "data_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

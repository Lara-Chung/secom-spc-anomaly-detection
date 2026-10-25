"""Shared paths and raw-file parsing for the UCI SECOM dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
OUTPUT_DIR = REPO_ROOT / "outputs"

FEATURES_FILE = RAW_DIR / "secom.data"
LABELS_FILE = RAW_DIR / "secom_labels.data"

# Timestamps in secom_labels.data are day-first, e.g. "19/07/2008 11:55:00"
TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"


def read_features(path: Path = FEATURES_FILE) -> pd.DataFrame:
    """Return the run x sensor matrix; columns are 0-based sensor_id, NaN = missing.

    The downloaded file has 590 columns, although the UCI metadata states 591
    attributes; the metadata count appears to include the label column stored
    in secom_labels.data (not documented by the source).
    """
    values = np.loadtxt(path, dtype=float)  # "NaN" tokens parse to np.nan
    return pd.DataFrame(values, columns=range(values.shape[1]))


def read_labels(path: Path = LABELS_FILE) -> pd.DataFrame:
    """Return run_id (0-based file row order), ts and raw label (-1 pass, 1 fail)."""
    labels = pd.read_csv(
        path, sep=" ", header=None, names=["label", "ts"], quotechar='"'
    )
    labels["ts"] = pd.to_datetime(labels["ts"], format=TIMESTAMP_FORMAT)
    labels.insert(0, "run_id", range(len(labels)))
    return labels

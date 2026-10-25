"""Rebuild the SECOM tables in PostgreSQL from the raw files and verify row counts.

Usage: uv run python src/load_database.py
The load is repeatable: sql/schema.sql drops and recreates the tables each run.
"""

import pandas as pd

from db import connect
from secom_data import REPO_ROOT, read_features, read_labels

SCHEMA_FILE = REPO_ROOT / "sql" / "schema.sql"


def copy_frame(cur, table: str, frame: pd.DataFrame) -> None:
    """Bulk-load a DataFrame with COPY; NaN is written as \\N (SQL NULL)."""
    data = frame.to_csv(sep="\t", header=False, index=False, na_rep=r"\N", lineterminator="\n")
    columns = ", ".join(frame.columns)
    with cur.copy(f"COPY {table} ({columns}) FROM STDIN") as copy:
        copy.write(data)


def main() -> None:
    features = read_features()
    labels = read_labels()
    n_runs, n_sensors = features.shape

    sensors = pd.DataFrame({"sensor_id": range(n_sensors)})
    sensors["sensor_name"] = [f"sensor_{i:03d}" for i in sensors["sensor_id"]]

    long = features.stack(future_stack=True).reset_index()  # keeps NaN cells
    long.columns = ["run_id", "sensor_id", "value"]

    with connect() as conn, conn.cursor() as cur:
        cur.execute(SCHEMA_FILE.read_text())
        copy_frame(cur, "runs", labels[["run_id", "ts", "label"]])
        copy_frame(cur, "sensors", sensors)
        copy_frame(cur, "measurements", long)
        cur.execute("ANALYZE runs, sensors, measurements")

        expected = {
            "runs": n_runs,
            "sensors": n_sensors,
            "measurements": n_runs * n_sensors,
        }
        for table, count in expected.items():
            (actual,) = cur.execute(f"SELECT count(*) FROM {table}").fetchone()
            if actual != count:
                raise RuntimeError(f"{table}: expected {count} rows, found {actual}")
            print(f"{table}: {actual} rows")

        (nulls,) = cur.execute(
            "SELECT count(*) FROM measurements WHERE value IS NULL"
        ).fetchone()
        expected_nulls = int(features.isna().values.sum())
        if nulls != expected_nulls:
            raise RuntimeError(f"NULL values: expected {expected_nulls}, found {nulls}")
        print(f"measurements with NULL value: {nulls}")


if __name__ == "__main__":
    main()

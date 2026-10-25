"""Create the analysis views (sql/00-05) and export their results to outputs/sql/.

Usage: uv run python src/run_sql.py
Each sql/0*.sql file creates one view; the files run in name order because
later views build on earlier ones.
"""

import csv

from db import connect
from secom_data import OUTPUT_DIR, REPO_ROOT

SQL_DIR = REPO_ROOT / "sql"
EXPORT_DIR = OUTPUT_DIR / "sql"

# spc_rolling (one row per reading) and feature_export (read by the model) are not exported
EXPORTS = {
    "sensor_quality": "SELECT * FROM sensor_quality ORDER BY sensor_id",
    "fail_trends_daily": "SELECT * FROM fail_trends_daily ORDER BY day",
    "spc_run_scores": "SELECT * FROM spc_run_scores ORDER BY ts, run_id",
}


def create_views(conn) -> None:
    for path in sorted(SQL_DIR.glob("0*.sql")):
        conn.execute(path.read_text())
        print(f"ran {path.name}")


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        create_views(conn)
        for name, query in EXPORTS.items():
            cur = conn.execute(query)
            out = EXPORT_DIR / f"{name}.csv"
            with out.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([c.name for c in cur.description])
                writer.writerows(cur)
            print(f"wrote {out.relative_to(REPO_ROOT)} ({cur.rowcount} rows)")


if __name__ == "__main__":
    main()

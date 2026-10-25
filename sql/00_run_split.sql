-- Chronological 70/30 split: the first 70% of runs by (ts, run_id) are train.
-- run_id breaks ties between runs that share a timestamp.
DROP VIEW IF EXISTS run_split CASCADE;

CREATE VIEW run_split AS
SELECT
    run_id,
    ts,
    is_fail,
    CASE
        WHEN row_number() OVER (ORDER BY ts, run_id) <= floor(0.7 * count(*) OVER ())
        THEN 'train' ELSE 'test'
    END AS split
FROM runs;

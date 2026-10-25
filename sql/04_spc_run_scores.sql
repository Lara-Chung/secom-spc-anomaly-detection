-- Q: How many sensors are out of control in each run, and do flagged runs fail more often?
-- frozen_*: control limits (mean +/- 3 std) estimated on the training period only
--   and held fixed for every run; this is the score compared with Isolation Forest.
-- rolling_*: alert counts from the past-only rolling baseline (03_spc_rolling),
--   reported as a supplementary online-monitoring view.
-- Labels are joined only to report outcomes; no limit uses them.
DROP VIEW IF EXISTS spc_run_scores CASCADE;

CREATE VIEW spc_run_scores AS
WITH limits AS (
    SELECT
        m.sensor_id,
        avg(m.value)          AS mean_train,
        stddev_samp(m.value)  AS std_train
    FROM measurements m
    JOIN run_split s USING (run_id)
    JOIN sensor_quality q USING (sensor_id)
    WHERE s.split = 'train' AND q.is_eligible
    GROUP BY m.sensor_id
),
frozen AS (
    SELECT
        m.run_id,
        count(m.value)                                                 AS frozen_n_checked,
        count(*) FILTER (WHERE abs(m.value - l.mean_train) > 3 * l.std_train) AS frozen_n_alerts
    FROM measurements m
    JOIN limits l USING (sensor_id)
    GROUP BY m.run_id
),
rolling AS (
    SELECT
        run_id,
        count(is_alert)                     AS rolling_n_checked,
        count(*) FILTER (WHERE is_alert)    AS rolling_n_alerts
    FROM spc_rolling
    GROUP BY run_id
)
SELECT
    s.run_id,
    s.ts,
    s.split,
    s.is_fail,
    f.frozen_n_checked,
    f.frozen_n_alerts,
    r.rolling_n_checked,
    r.rolling_n_alerts
FROM run_split s
JOIN frozen f USING (run_id)
JOIN rolling r USING (run_id)
ORDER BY s.ts, s.run_id;

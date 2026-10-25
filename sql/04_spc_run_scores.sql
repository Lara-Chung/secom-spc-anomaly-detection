-- Q: How many sensors are out of control in each run, and do flagged runs fail more often?
-- frozen_*: control limits (mean +/- 3 std) estimated on the training period only
--   and held fixed for every run; this is the score compared with Isolation Forest.
-- rolling_*: alert counts from the past-only rolling baseline (03_spc_rolling),
--   reported as a supplementary online-monitoring view.
-- Labels are joined only to report outcomes; no limit uses them.
-- frozen_limits is its own view because 06_period_shift.sql and the
-- investigation queries in sql/investigation/ reuse the same limits.
-- Both views are materialized. frozen_limits is fixed once the training period
-- is loaded. spc_run_scores reads spc_rolling, which recomputes the rolling
-- window for every reading, so computing it once per run_sql.py run lets the
-- investigation CLI and the report read run scores in milliseconds.
DROP MATERIALIZED VIEW IF EXISTS spc_run_scores CASCADE;
DROP MATERIALIZED VIEW IF EXISTS frozen_limits CASCADE;

CREATE MATERIALIZED VIEW frozen_limits AS
SELECT
    m.sensor_id,
    count(m.value)                                      AS n_train,
    avg(m.value)                                        AS mean_train,
    stddev_samp(m.value)                                AS std_train,
    avg(m.value) - 3 * stddev_samp(m.value)             AS lower_limit,
    avg(m.value) + 3 * stddev_samp(m.value)             AS upper_limit
FROM measurements m
JOIN run_split s USING (run_id)
JOIN sensor_quality q USING (sensor_id)
WHERE s.split = 'train' AND q.is_eligible
GROUP BY m.sensor_id;

CREATE MATERIALIZED VIEW spc_run_scores AS
WITH frozen AS (
    SELECT
        m.run_id,
        count(m.value)                                                 AS frozen_n_checked,
        count(*) FILTER (WHERE abs(m.value - l.mean_train) > 3 * l.std_train) AS frozen_n_alerts
    FROM measurements m
    JOIN frozen_limits l USING (sensor_id)
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

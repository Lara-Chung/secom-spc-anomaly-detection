-- Q: For one run, which sensors are outside their past-only rolling 3-sigma band?
-- Bound parameter: run_id.
-- Same definition as spc_rolling (03_spc_rolling.sql): the band is built from the
-- previous 50 runs in (ts, run_id) order, and a reading is not assessable when it
-- is missing, fewer than 20 of those runs have a value, or all of them share one
-- value (flat window). Filtering the view would recompute the window for every
-- reading, so this query first takes the 50 preceding runs once and then reads
-- their 590 readings each through the primary key. The aggregates take
-- ORDER BY ts, run_id so that floating-point sums accumulate in the same order
-- as the window frame and the numbers match the view exactly.
-- history_end_ts is the time stamp of the last observed reading in the window.
WITH target AS (
    SELECT run_id, ts FROM runs WHERE run_id = %(run_id)s
),
history AS (
    SELECT r.run_id, r.ts
    FROM runs r
    CROSS JOIN target t
    WHERE (r.ts, r.run_id) < (t.ts, t.run_id)
    ORDER BY r.ts DESC, r.run_id DESC
    LIMIT 50
),
band AS (
    SELECT
        m.sensor_id,
        avg(m.value ORDER BY h.ts, h.run_id)          AS roll_mean,
        stddev_samp(m.value ORDER BY h.ts, h.run_id)  AS roll_std,
        count(m.value)                                AS roll_n,
        min(m.value) = max(m.value)                   AS roll_flat,
        max(h.ts) FILTER (WHERE m.value IS NOT NULL)  AS last_observed_ts
    FROM history h
    JOIN measurements m USING (run_id)
    JOIN sensor_quality q USING (sensor_id)
    WHERE q.is_eligible
    GROUP BY m.sensor_id
),
readings AS (
    -- The assessability decision is made once here; every column below derives from it
    SELECT
        t.run_id,
        t.ts,
        q.sensor_id,
        m.value,
        b.roll_mean,
        b.roll_std,
        coalesce(b.roll_n, 0)                         AS roll_n,   -- no band at all for the first run
        b.last_observed_ts,
        CASE WHEN m.value IS NULL              THEN 'missing_value'
             WHEN coalesce(b.roll_n, 0) < 20   THEN 'insufficient_history'
             WHEN b.roll_flat                  THEN 'flat_window' END AS unassessable_reason
    FROM target t
    CROSS JOIN sensor_quality q
    JOIN measurements m ON m.run_id = t.run_id AND m.sensor_id = q.sensor_id
    LEFT JOIN band b ON b.sensor_id = q.sensor_id
    WHERE q.is_eligible
)
SELECT
    run_id,
    ts,
    sensor_id,
    'spc_rolling'                                                         AS method,
    value,
    roll_mean                                                             AS baseline_mean,
    roll_std                                                              AS baseline_std,
    roll_mean - 3 * roll_std                                              AS lower_limit,
    roll_mean + 3 * roll_std                                              AS upper_limit,
    CASE WHEN unassessable_reason IS NULL THEN (value - roll_mean) / roll_std END       AS z_score,
    CASE WHEN unassessable_reason IS NULL THEN abs(value - roll_mean) / roll_std END    AS abs_z_score,
    unassessable_reason IS NULL                                           AS is_assessable,
    CASE WHEN unassessable_reason IS NULL THEN abs(value - roll_mean) > 3 * roll_std END AS is_sensor_alert,
    unassessable_reason,
    roll_n                                                                AS history_count,
    last_observed_ts                                                      AS history_end_ts
FROM readings
ORDER BY abs_z_score DESC NULLS LAST, sensor_id;

-- Q: For one run, which sensors are outside the frozen 3-sigma limits, and by how much?
-- Bound parameter: run_id. Limits come from frozen_limits (training period, held
-- fixed), so history_count is the number of training readings behind each limit
-- and history_end_ts is the end of the training period for every sensor.
-- Not assessable: the reading is missing. Eligible sensors cannot have a zero
-- training standard deviation (constant sensors are excluded upstream).
WITH train_end AS (
    SELECT max(ts) AS ts FROM run_split WHERE split = 'train'
)
SELECT
    r.run_id,
    r.ts,
    m.sensor_id,
    'spc_frozen'                                                  AS method,
    m.value,
    l.mean_train                                                  AS baseline_mean,
    l.std_train                                                   AS baseline_std,
    l.lower_limit,
    l.upper_limit,
    (m.value - l.mean_train) / l.std_train                        AS z_score,
    abs(m.value - l.mean_train) / l.std_train                     AS abs_z_score,
    m.value IS NOT NULL                                           AS is_assessable,
    CASE WHEN m.value IS NULL THEN NULL
         ELSE abs(m.value - l.mean_train) > 3 * l.std_train END   AS is_sensor_alert,
    CASE WHEN m.value IS NULL THEN 'missing_value' END            AS unassessable_reason,
    l.n_train                                                     AS history_count,
    train_end.ts                                                  AS history_end_ts
FROM runs r
JOIN measurements m USING (run_id)
JOIN frozen_limits l USING (sensor_id)
CROSS JOIN train_end
WHERE r.run_id = %(run_id)s
ORDER BY abs_z_score DESC NULLS LAST, m.sensor_id;

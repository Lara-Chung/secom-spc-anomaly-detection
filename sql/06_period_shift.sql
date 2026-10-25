-- Q: Did sensor readings shift between the training and test periods?
-- For each eligible sensor: the change in median from train to test, expressed
-- in training standard deviations, and the share of readings outside the frozen
-- train 3-sigma limits in each period. The median is used because a few extreme
-- readings (e.g. sensor 140 has four readings near 9998 in test) move the mean
-- far while the typical level stays the same. The denominator is the ordinary
-- training standard deviation, not a robust scale, because it is the same value
-- that sets the frozen 3-sigma limits: a shift of 1 means the typical level moved
-- by one third of the control band. Descriptive only; no model or threshold uses it.
DROP VIEW IF EXISTS period_shift CASCADE;
DROP VIEW IF EXISTS period_shift_summary CASCADE;

CREATE VIEW period_shift AS
WITH limits AS (
    SELECT sensor_id, mean_train, std_train FROM frozen_limits
),
per_period AS (
    SELECT
        m.sensor_id,
        s.split,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY m.value)           AS median_value,
        avg((abs(m.value - l.mean_train) > 3 * l.std_train)::int)     AS alert_rate
    FROM measurements m
    JOIN run_split s USING (run_id)
    JOIN limits l USING (sensor_id)
    WHERE m.value IS NOT NULL
    GROUP BY m.sensor_id, s.split
)
SELECT
    tr.sensor_id,
    l.std_train,
    tr.median_value                                    AS median_train,
    te.median_value                                    AS median_test,
    (te.median_value - tr.median_value) / l.std_train  AS shift_in_train_sd,
    tr.alert_rate                                  AS alert_rate_train,
    te.alert_rate                                  AS alert_rate_test
FROM per_period tr
JOIN per_period te ON te.sensor_id = tr.sensor_id AND te.split = 'test'
JOIN limits l ON l.sensor_id = tr.sensor_id
WHERE tr.split = 'train';

-- One-row summary of the view above, used in the README.
CREATE VIEW period_shift_summary AS
SELECT
    count(*)                                                    AS n_sensors,
    count(*) FILTER (WHERE abs(shift_in_train_sd) > 0.5)        AS n_shift_gt_0_5_sd,
    count(*) FILTER (WHERE abs(shift_in_train_sd) > 1)          AS n_shift_gt_1_sd,
    avg(alert_rate_train)                                       AS mean_alert_rate_train,
    avg(alert_rate_test)                                        AS mean_alert_rate_test,
    count(*) FILTER (WHERE alert_rate_test > alert_rate_train)  AS n_alert_rate_up,
    count(*) FILTER (WHERE alert_rate_test < alert_rate_train)  AS n_alert_rate_down,
    count(*) FILTER (WHERE alert_rate_test = alert_rate_train)  AS n_alert_rate_equal
FROM period_shift;

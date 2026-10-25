-- Q: How complete and how variable is each sensor, and which sensors are excluded?
-- Full-data columns are descriptive only. The exclusion decision uses the
-- training period only, so the test period cannot influence feature selection.
-- Excluded: missing rate > 50% in train, or constant in train (<= 1 distinct value).
-- Mostly-constant sensors are kept: rare deviations are what SPC should catch.
DROP VIEW IF EXISTS sensor_quality CASCADE;

CREATE VIEW sensor_quality AS
WITH stats AS (
    SELECT
        m.sensor_id,
        count(*)                                                AS n_runs,
        avg((m.value IS NULL)::int)                             AS missing_rate_all,
        avg(m.value)                                            AS mean_all,
        stddev_samp(m.value)                                    AS std_all,
        avg((m.value IS NULL)::int) FILTER (WHERE s.split = 'train') AS missing_rate_train,
        count(DISTINCT m.value)     FILTER (WHERE s.split = 'train') AS n_distinct_train
    FROM measurements m
    JOIN run_split s USING (run_id)
    GROUP BY m.sensor_id
)
SELECT
    *,
    CASE
        WHEN missing_rate_train > 0.5 THEN 'missing_rate_train > 0.5'
        WHEN n_distinct_train <= 1    THEN 'constant_in_train'
    END AS exclusion_reason,
    NOT (missing_rate_train > 0.5 OR n_distinct_train <= 1) AS is_eligible
FROM stats;

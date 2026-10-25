-- Q: What is the model input? One row per run with the eligible sensors'
-- readings as an array ordered by sensor_id (NULL = missing), plus split and label.
-- The array avoids a dynamic 500-column pivot; Python turns it into a matrix.
-- Imputation and scaling are fitted later in Python on the training rows only.
DROP VIEW IF EXISTS feature_export CASCADE;
DROP VIEW IF EXISTS feature_export_all CASCADE;

CREATE VIEW feature_export AS
SELECT
    s.run_id,
    s.ts,
    s.split,
    s.is_fail,
    array_agg(m.sensor_id ORDER BY m.sensor_id) AS sensor_ids,
    array_agg(m.value     ORDER BY m.sensor_id) AS sensor_values
FROM run_split s
JOIN measurements m USING (run_id)
JOIN sensor_quality q USING (sensor_id)
WHERE q.is_eligible
GROUP BY s.run_id, s.ts, s.split, s.is_fail
ORDER BY s.ts, s.run_id;

-- All 590 sensors in the same row order, for a model that decides sensor
-- eligibility itself inside each cross-validation fold (the supervised reference).
CREATE VIEW feature_export_all AS
SELECT
    s.run_id,
    s.ts,
    s.split,
    s.is_fail,
    array_agg(m.sensor_id ORDER BY m.sensor_id) AS sensor_ids,
    array_agg(m.value     ORDER BY m.sensor_id) AS sensor_values
FROM run_split s
JOIN measurements m USING (run_id)
GROUP BY s.run_id, s.ts, s.split, s.is_fail
ORDER BY s.ts, s.run_id;

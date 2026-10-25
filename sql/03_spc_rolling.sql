-- Q: Which readings fall outside a 3-sigma band built from that sensor's recent history?
-- Online monitoring: the baseline is the previous 50 runs of the same sensor and
-- never includes the current or any later reading (ROWS ... AND 1 PRECEDING).
-- It uses earlier sensor values only, never labels. A reading is not evaluated
-- (is_alert NULL) when it is missing, the window has fewer than 20 non-missing
-- values, or every value in the window is identical. The last case is checked
-- with min = max because stddev_samp of a repeated value such as 0.1 returns a
-- tiny non-zero number from floating-point rounding, not exactly 0.
DROP VIEW IF EXISTS spc_rolling CASCADE;

CREATE VIEW spc_rolling AS
WITH windowed AS (
    SELECT
        m.run_id,
        m.sensor_id,
        r.ts,
        m.value,
        avg(m.value)          OVER w AS roll_mean,
        stddev_samp(m.value)  OVER w AS roll_std,
        count(m.value)        OVER w AS roll_n,
        min(m.value) OVER w = max(m.value) OVER w AS roll_flat
    FROM measurements m
    JOIN runs r USING (run_id)
    JOIN sensor_quality q USING (sensor_id)
    WHERE q.is_eligible
    WINDOW w AS (
        PARTITION BY m.sensor_id
        ORDER BY r.ts, m.run_id
        ROWS BETWEEN 50 PRECEDING AND 1 PRECEDING
    )
)
SELECT
    *,
    CASE
        WHEN value IS NULL OR roll_n < 20 OR roll_flat THEN NULL
        ELSE (value - roll_mean) / roll_std
    END AS z,
    CASE
        WHEN value IS NULL OR roll_n < 20 OR roll_flat THEN NULL
        ELSE abs(value - roll_mean) > 3 * roll_std
    END AS is_alert
FROM windowed;

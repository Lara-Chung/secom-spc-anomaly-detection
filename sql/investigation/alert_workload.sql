-- Q: How many runs per day would each SPC rule send to an engineer, and how many
-- of those runs later failed?
-- Bound parameters: frozen_threshold and rolling_threshold, the run-level
-- thresholds chosen on the training period (outputs/metrics.csv). Alert = score
-- at or above the threshold, the same rule as src/train_evaluate.py.
-- A run is assessable when at least one eligible sensor could be evaluated; runs
-- with no assessable sensor are counted separately and never as true negatives.
-- Aggregation is at run level, so a run with many alerting sensors counts once.
-- is_fail is joined only for the retrospective outcome columns; no alert uses it.
WITH per_run AS (
    SELECT
        s.run_id,
        s.ts::date AS day,
        s.split,
        s.is_fail,
        m.method,
        m.n_checked,
        m.is_alert
    FROM spc_run_scores s
    CROSS JOIN LATERAL (VALUES
        ('spc_frozen',  s.frozen_n_checked,  s.frozen_n_alerts  >= %(frozen_threshold)s),
        ('spc_rolling', s.rolling_n_checked, s.rolling_n_alerts >= %(rolling_threshold)s)
    ) AS m (method, n_checked, is_alert)
)
SELECT
    day,
    method,
    split,
    count(*)                                                        AS n_runs,
    count(*) FILTER (WHERE n_checked > 0)                           AS n_assessable,
    count(*) FILTER (WHERE n_checked = 0)                           AS n_not_assessable,
    count(*) FILTER (WHERE n_checked = 0 AND is_fail)               AS n_not_assessable_fail,
    count(*) FILTER (WHERE n_checked > 0 AND is_alert)              AS n_alert,
    round(100.0 * count(*) FILTER (WHERE n_checked > 0 AND is_alert)
          / nullif(count(*) FILTER (WHERE n_checked > 0), 0), 2)    AS alerts_per_100_assessable,
    count(*) FILTER (WHERE is_fail)                                 AS n_fail,
    count(*) FILTER (WHERE n_checked > 0 AND is_alert AND is_fail)          AS tp,
    count(*) FILTER (WHERE n_checked > 0 AND is_alert AND NOT is_fail)      AS fp,
    count(*) FILTER (WHERE n_checked > 0 AND NOT is_alert AND is_fail)      AS fn,
    count(*) FILTER (WHERE n_checked > 0 AND NOT is_alert AND NOT is_fail)  AS tn
FROM per_run
GROUP BY day, method, split
ORDER BY day, method, split;

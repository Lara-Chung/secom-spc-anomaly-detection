-- Q: How many runs are tested per day, and how does the fail rate move over time?
-- The trailing 7-day rate smooths the noisy daily rate (about 18 runs per day).
DROP VIEW IF EXISTS fail_trends_daily CASCADE;

CREATE VIEW fail_trends_daily AS
WITH daily AS (
    SELECT
        ts::date              AS day,
        count(*)              AS n_runs,
        sum(is_fail::int)     AS n_fail
    FROM runs
    GROUP BY ts::date
)
SELECT
    day,
    n_runs,
    n_fail,
    round(n_fail::numeric / n_runs, 4) AS fail_rate,
    round(
        sum(n_fail) OVER w::numeric / sum(n_runs) OVER w, 4
    ) AS fail_rate_trailing_7d
FROM daily
WINDOW w AS (ORDER BY day RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW)
ORDER BY day;

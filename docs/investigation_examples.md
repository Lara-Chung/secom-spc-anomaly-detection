# Alert investigation examples

Three runs from the test period, chosen by a fixed rule before looking at their sensors: under the frozen 3σ rule (run alert when at least 11 sensors are out of control), the highest-scoring run in each outcome group. They illustrate what the investigation query returns; they are not a performance sample. All numbers come from `uv run python src/build_investigation_report.py`; reproduce any table with `uv run python src/investigate_run.py --run-id <id> --method spc_frozen --outcome`.

SECOM does not name its sensors, so no table below says what a sensor measures. A large deviation is evidence to investigate, not a root cause.

## Alert workload on the test period

Totals from `outputs/investigation/alert_workload.csv` (per-day rows in the file):

| method | n_runs | n_assessable | n_not_assessable | n_not_assessable_fail | n_alert | tp | fp | fn | tn | precision | recall_assessable |
|---|---|---|---|---|---|---|---|---|---|---|---|
| spc_frozen | 471 | 471 | 0 | 0 | 48 | 4 | 44 | 22 | 401 | 0.08333 | 0.1538 |
| spc_rolling | 471 | 471 | 0 | 0 | 29 | 5 | 24 | 21 | 421 | 0.1724 | 0.1923 |

## Case 1: alerted, and the run failed

Run 1151 at 2008-09-28 09:50:00: 18 sensors out of control (threshold 11, alert); outcome fail.

Sensors in scope 444, observed 436, assessable 436, alerting 18, missing 8.

Top 10 sensors by |z| (frozen training limits):

| sensor_id | value | baseline_mean | baseline_std | lower_limit | upper_limit | z_score | is_sensor_alert |
|---|---|---|---|---|---|---|---|
| 270 | 110.3 | 29.6 | 8.042 | 5.471 | 53.72 | 10.03 | True |
| 408 | 28.73 | 5.37 | 2.431 | -1.924 | 12.66 | 9.61 | True |
| 135 | 634 | 118.7 | 54.25 | -43.99 | 281.5 | 9.499 | True |
| 99 | 0.4492 | 0.0001167 | 0.05461 | -0.1637 | 0.1639 | 8.224 | True |
| 104 | -0.006 | 5.297e-06 | 0.0007463 | -0.002234 | 0.002244 | -8.047 | True |
| 417 | 30.11 | 7.361 | 3.4 | -2.84 | 17.56 | 6.69 | True |
| 144 | 0.4455 | 0.1086 | 0.05193 | -0.04715 | 0.2644 | 6.488 | True |
| 279 | 0.148 | 0.03725 | 0.01723 | -0.01443 | 0.08894 | 6.428 | True |
| 100 | 0.0022 | -4.409e-05 | 0.0003617 | -0.001129 | 0.001041 | 6.204 | True |
| 103 | 0.0062 | -0.009283 | 0.003181 | -0.01883 | 0.000261 | 4.867 | True |

What to check next:

- Check whether the sensors listed above belong to one tool or process step; SECOM does not name them, so this needs the sensor map.
- Compare the same sensors on the runs just before this one to see whether the deviation built up or appeared at once.

## Case 2: alerted, but the run passed

Run 1458 at 2008-10-11 07:39:00: 42 sensors out of control (threshold 11, alert); outcome pass.

Sensors in scope 444, observed 440, assessable 440, alerting 42, missing 4.

Top 10 sensors by |z| (frozen training limits):

| sensor_id | value | baseline_mean | baseline_std | lower_limit | upper_limit | z_score | is_sensor_alert |
|---|---|---|---|---|---|---|---|
| 338 | 196.7 | 2.662 | 2.904 | -6.051 | 11.38 | 66.8 | True |
| 202 | 490.6 | 8.893 | 7.955 | -14.97 | 32.76 | 60.55 | True |
| 339 | 197.5 | 9.477 | 4.139 | -2.94 | 21.89 | 45.43 | True |
| 475 | 79.12 | 4.356 | 1.966 | -1.542 | 10.25 | 38.03 | True |
| 203 | 500.3 | 30.71 | 14.17 | -11.8 | 73.22 | 33.14 | True |
| 201 | 126.5 | 7.059 | 3.69 | -4.011 | 18.13 | 32.38 | True |
| 337 | 38.9 | 2.116 | 1.449 | -2.232 | 6.465 | 25.38 | True |
| 286 | 207 | 4.159 | 8.493 | -21.32 | 29.64 | 23.89 | True |
| 471 | 212.7 | 9.444 | 9.573 | -19.28 | 38.16 | 21.23 | True |
| 200 | 183.7 | 17.91 | 8.331 | -7.081 | 42.9 | 19.9 | True |

What to check next:

- The alert is not necessarily wrong: the run passed the final test, but the readings did deviate. Check whether the deviating sensors are known to be noisy or recently recalibrated.
- If this pattern repeats on passing runs, the sensors involved are candidates for a wider band or exclusion from the run score.

## Case 3: not alerted, but the run failed

Run 1227 at 2008-10-02 03:17:00: 9 sensors out of control (threshold 11, no alert); outcome fail.

Sensors in scope 444, observed 444, assessable 444, alerting 9, missing 0.

Top 10 sensors by |z| (frozen training limits):

| sensor_id | value | baseline_mean | baseline_std | lower_limit | upper_limit | z_score | is_sensor_alert |
|---|---|---|---|---|---|---|---|
| 376 | 0.0053 | 0.001612 | 0.0005482 | -3.239e-05 | 0.003257 | 6.727 | True |
| 238 | 0.0159 | 0.004829 | 0.001732 | -0.0003662 | 0.01002 | 6.393 | True |
| 571 | 1.074 | 2.166 | 0.2318 | 1.471 | 2.861 | -4.709 | True |
| 510 | 210.6 | 60.28 | 41.72 | -64.88 | 185.5 | 3.603 | True |
| 308 | 0.3239 | 0.2169 | 0.03288 | 0.1183 | 0.3156 | 3.253 | True |
| 173 | 0.8643 | 0.5711 | 0.09362 | 0.2902 | 0.8519 | 3.132 | True |
| 545 | 10.46 | 7.359 | 0.9927 | 4.381 | 10.34 | 3.126 | True |
| 444 | 1.351 | 0.8868 | 0.1533 | 0.4267 | 1.347 | 3.028 | True |
| 543 | 0.0117 | 0.008146 | 0.001177 | 0.004614 | 0.01168 | 3.019 | True |
| 132 | 2.2 | 2.331 | 0.04762 | 2.188 | 2.473 | -2.752 | False |

What to check next:

- The run stayed below the run-level threshold: the sensors that did cross their limits are listed above. The cause of the failure is either not in these sensors or spread across sensors rather than concentrated in enough large deviations.
- Look at the sensors with the largest |z| anyway and at the missing readings; a run with many missing values has fewer chances to alert.


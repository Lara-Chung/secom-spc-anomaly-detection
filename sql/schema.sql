-- Long-format schema for UCI SECOM: one row per run, one row per sensor,
-- and one measurement row for every run x sensor pair (missing readings = NULL).
-- Re-running this file drops and rebuilds all tables; CASCADE also drops
-- dependent views, which the analysis step recreates.

DROP TABLE IF EXISTS measurements CASCADE;
DROP TABLE IF EXISTS sensors CASCADE;
DROP TABLE IF EXISTS runs CASCADE;

CREATE TABLE runs (
    run_id   INTEGER PRIMARY KEY,           -- 0-based row order in the source files
    ts       TIMESTAMP NOT NULL,            -- test time stamp from secom_labels.data
    label    SMALLINT NOT NULL CHECK (label IN (-1, 1)),  -- source coding: -1 pass, 1 fail
    is_fail  BOOLEAN GENERATED ALWAYS AS (label = 1) STORED
);

CREATE TABLE sensors (
    sensor_id    SMALLINT PRIMARY KEY,      -- 0-based column order in secom.data
    sensor_name  TEXT NOT NULL UNIQUE
);

CREATE TABLE measurements (
    run_id     INTEGER  NOT NULL REFERENCES runs (run_id),
    sensor_id  SMALLINT NOT NULL REFERENCES sensors (sensor_id),
    value      DOUBLE PRECISION,            -- NULL where the source has NaN
    PRIMARY KEY (run_id, sensor_id)
);

-- Chronological ordering and time-based splits; run_id breaks timestamp ties
CREATE INDEX runs_ts_run_id_idx ON runs (ts, run_id);

-- Per-sensor scans (quality summaries, rolling SPC windows partitioned by sensor)
CREATE INDEX measurements_sensor_run_idx ON measurements (sensor_id, run_id);

# Industrial Sensor Data Analytics & Anomaly Detection

Semiconductor process monitoring using the UCI SECOM dataset: PostgreSQL data quality checks, SQL-based statistical process control (SPC), and Isolation Forest anomaly scoring.

> Work in progress. Sections below are filled in as each stage is completed.

## Setup

Requirements: Docker (with Compose) and [uv](https://docs.astral.sh/uv/).

```bash
docker compose up -d --wait
uv sync
```

## Data and license

This project uses the SECOM dataset from the UCI Machine Learning Repository, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The raw files are not redistributed in this repository; they are downloaded from the source.

McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305

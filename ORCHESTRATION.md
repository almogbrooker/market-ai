# Orchestration Options

This repository relies on a simple cron-based schedule by default. For more
coordinated retry and backfill support, evaluate modern orchestration tools.

## Prefect
- **Pros:** Python-native workflows, automatic retries, and built-in support for
  backfills.
- **Cons:** Requires an extra service (Prefect server or cloud) and additional
  dependencies.

## Airflow
- **Pros:** Battle-tested scheduler with DAG-based backfill capabilities and rich
  retry policies.
- **Cons:** Heavier to deploy and maintain compared to Prefect. Best suited for
  larger teams and complex pipelines.

## Cron (Fallback)
- Minimal dependency surface and already part of the deployment playbook.
- Use as a fallback when Prefect or Airflow are unavailable or during simple
  single-machine setups.

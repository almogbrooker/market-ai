# Monitoring and SLOs

This directory contains example dashboards and service level objectives for the
production trading bots.

## Dashboards
- `grafana_dashboard.json` – Prometheus-backed Grafana dashboard visualising
  latency, signal acceptance rate and gross exposure.
- `datadog_dashboard.json` – Equivalent dashboard for Datadog users.

## Availability SLOs
- **Service availability:** 99.5% of requests to `/health` must return `200 OK`.
- **Signal generation latency:** 95th percentile of `bot_latency_seconds` for the
  `generate_signals` operation below 1 second.
- **Risk limits:** `bot_gross_exposure` should remain below `0.60` for 99% of samples.

The monitoring server also exposes a `/kill` endpoint that activates a kill
switch and notifies Slack or Telegram when the environment variables
`SLACK_WEBHOOK_URL`, `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set.

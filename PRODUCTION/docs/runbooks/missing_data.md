# Missing Data Runbook

## Symptoms
- Data pipeline logs show missing or empty datasets.
- Alerts for incomplete bars or missing fields.

## Immediate Actions
1. Verify connectivity to data providers.
2. Re-run the daily pipeline:
   ```bash
   python pipelines/update_daily.py --update-training
   ```
3. If primary provider is down, switch to fallback sources in `data_providers/`.

## Verification
- Ensure `data/` directories contain fresh files.
- Run a quick audit:
   ```bash
   python PRODUCTION/tools/institutional_audit_system.py
   ```

## Escalation
- If data is still missing after two retries, escalate to the data provider and pause automated trading.

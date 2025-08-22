# Safe Shutdown & Resume Runbook

Use this procedure to gracefully stop and restart the automated trading system without duplicating jobs.

## Command
```bash
python PRODUCTION/tools/safe_restart.py
```
This command sends a gentle interrupt, waits for the system to cleanly exit, and then relaunches it. The `trading_system.lock` file ensures exactly one instance is running.

## Verification
- Watch the console for `Trading system stopped` followed by `Trading system started`.
- Confirm a single `trading_system.lock` file exists.

## Escalation
- If the system fails to stop or start within 30 seconds, inspect logs and manually kill the process before retrying.

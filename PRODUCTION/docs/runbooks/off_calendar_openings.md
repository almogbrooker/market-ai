# Off-Calendar Openings Runbook

## Symptoms
- Market appears open on expected holiday or weekend.
- Unexpected price data during closed sessions.

## Immediate Actions
1. Verify official exchange calendar and holiday schedules.
2. Halt trading and prevent auto-restart:
   ```bash
   pkill -f run_automated_trading.py
   touch trading_system.lock
   ```
3. Confirm with broker whether trading is valid.

## Verification
- Resume trading only when market hours are confirmed.
- Remove the manual lock file before restart.

## Escalation
- If discrepancy persists, escalate to exchange or data provider and keep system offline.

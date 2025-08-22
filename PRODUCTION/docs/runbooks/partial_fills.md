# Partial Fills Runbook

## Symptoms
- Orders remain in `partially_filled` status.
- Position sizes differ from intended allocation.

## Immediate Actions
1. Check open orders:
   ```bash
   python check_positions.py
   ```
2. Cancel remaining unfilled orders via broker dashboard or API.
3. Decide whether to re-submit the residual quantity.

## Verification
- Confirm position sizes match targets.
- Review logs for unexpected order activity.

## Escalation
- If partial fills persist across sessions, contact the broker for support.

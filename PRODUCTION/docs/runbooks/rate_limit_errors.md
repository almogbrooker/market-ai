# Rate-Limit Errors Runbook

## Symptoms
- API responses return HTTP 429.
- Logs contain `rate limit` warnings.

## Immediate Actions
1. Pause requests for 60 seconds to respect cooldowns.
2. Enable exponential backoff in affected scripts.
3. Reduce batch sizes or request frequency.

## Verification
- Retry the previous operation and confirm success.
- Monitor logs to ensure rate-limit errors cease.

## Escalation
- If limits continue to be hit, request higher quotas from the provider or stagger jobs.

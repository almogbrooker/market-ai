import os
import threading
from wsgiref.simple_server import make_server

import requests
from prometheus_client import Gauge, Histogram, make_wsgi_app

# Prometheus metrics
LATENCY = Histogram("bot_latency_seconds", "Latency of bot operations", ["operation"])
GROSS_EXPOSURE = Gauge("bot_gross_exposure", "Current gross exposure as fraction of equity")
DAILY_LOSS = Gauge("bot_daily_loss", "Current daily loss as fraction of equity")
SIGNAL_ACCEPT_RATE = Gauge("bot_signal_accept_rate", "Acceptance rate of generated signals")

_kill_switch = False
_server_started = False


def send_slack_alert(message: str) -> None:
    """Send an alert message to Slack using a webhook."""
    url = os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"text": message}, timeout=5)
    except Exception:
        pass


def send_telegram_alert(message: str) -> None:
    """Send an alert message to Telegram using bot credentials."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message},
            timeout=5,
        )
    except Exception:
        pass


def _notify(message: str) -> None:
    send_slack_alert(message)
    send_telegram_alert(message)


def _health_app(environ, start_response):
    """WSGI app serving /health and /kill routes."""
    global _kill_switch
    path = environ.get("PATH_INFO", "/")
    if path == "/health":
        status = "503 Service Unavailable" if _kill_switch else "200 OK"
        start_response(status, [("Content-Type", "text/plain")])
        body = b"DISABLED" if _kill_switch else b"OK"
        return [body]
    if path == "/kill":
        _kill_switch = True
        _notify("Kill switch activated for trading bots")
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"Kill switch activated"]
    start_response("404 Not Found", [("Content-Type", "text/plain")])
    return [b"Not Found"]


def start_monitoring(port: int = 8000) -> None:
    """Start background thread serving Prometheus metrics and health checks."""
    global _server_started
    if _server_started:
        return

    metrics_app = make_wsgi_app()

    def app(environ, start_response):
        if environ.get("PATH_INFO", "").startswith("/metrics"):
            return metrics_app(environ, start_response)
        return _health_app(environ, start_response)

    server = make_server("", port, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _server_started = True

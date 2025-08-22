import sqlite3
import threading
from datetime import datetime, timedelta


class OrderLedger:
    """SQLite-backed ledger for tracking orders.

    The underlying connection uses ``check_same_thread=False`` so the
    connection can be shared across multiple threads. A single
    ``threading.Lock`` serializes access to that connection, making
    operations in this class thread-safe as long as the ledger instance is
    shared and its methods are used for all database interactions.
    """

    def __init__(self, db_path: str = "PRODUCTION/bots/orders.db") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    side TEXT,
                    quantity INTEGER,
                    status TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self.conn.commit()

    def record_order(self, symbol: str, side: str, quantity: int, status: str = "open") -> int:
        """Record a new order in the ledger.

        Returns the row id of the inserted order.
        """
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO orders(symbol, side, quantity, status) VALUES (?, ?, ?, ?)",
                (symbol, side, quantity, status),
            )
            self.conn.commit()
            return cursor.lastrowid

    def has_recent(self, symbol: str, seconds: int = 60) -> bool:
        """Check whether a recent order exists for the given symbol."""
        cutoff = datetime.utcnow() - timedelta(seconds=seconds)
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT 1 FROM orders WHERE symbol = ? AND timestamp >= ? LIMIT 1",
                (symbol, cutoff.isoformat()),
            )
            return cursor.fetchone() is not None

    def mark_filled(self, order_id: int) -> None:
        """Mark an order as filled."""
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE orders SET status = 'filled' WHERE id = ?",
                (order_id,),
            )
            self.conn.commit()

"""CSV trade journal and session PnL aggregates."""

import csv
import os
from datetime import datetime, timezone
from typing import Dict


TRADE_FIELDS = [
    "entry_time", "exit_time", "direction", "figi", "entry_price",
    "exit_price", "lots", "pnl_gross", "commission", "pnl_net", "exit_reason",
]
ONE_WAY_COMMISSION_RATE = 0.00025


def _journal_path() -> str:
    """Return the configured trade journal path."""
    return os.environ.get("TRADES_FILE", "trades.csv")


def calculate_round_trip_commission(
    entry_price: float, exit_price: float, lots: int, lot_size: int = 1
) -> float:
    """Calculate 0.025% commission for both entry and exit notionals."""
    entry_notional = entry_price * lot_size * lots
    exit_notional = exit_price * lot_size * lots
    return (entry_notional + exit_notional) * ONE_WAY_COMMISSION_RATE


def record_trade(
    entry_time: datetime,
    exit_time: datetime,
    direction: str,
    entry_price: float,
    exit_price: float,
    lots: int,
    pnl_gross: float,
    commission: float,
    exit_reason: str,
    figi: str = "FUTGLDRUBF00",
) -> None:
    """Append one completed trade to trades.csv."""
    path = _journal_path()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    row = {
        "entry_time": entry_time.astimezone(timezone.utc).isoformat(),
        "exit_time": exit_time.astimezone(timezone.utc).isoformat(),
        "direction": direction,
        "figi": figi,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "lots": lots,
        "pnl_gross": pnl_gross,
        "commission": commission,
        "pnl_net": pnl_gross - commission,
        "exit_reason": exit_reason,
    }
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=TRADE_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def session_aggregates(session_date: str) -> Dict[str, float]:
    """Calculate PnL, count, win rate and profit factor for a session date."""
    rows = []
    path = _journal_path()
    if os.path.exists(path):
        with open(path, newline="") as file:
            rows = list(csv.DictReader(file))
    pnl = []
    for row in rows:
        if row.get("exit_time", "")[:10] == session_date:
            pnl.append(float(row.get("pnl_net", 0)))
    wins = [value for value in pnl if value > 0]
    losses = [-value for value in pnl if value < 0]
    return {
        "total_pnl": sum(pnl),
        "trade_count": len(pnl),
        "win_rate": len(wins) / len(pnl) if pnl else 0.0,
        "profit_factor": sum(wins) / sum(losses) if losses else (float("inf") if wins else 0.0),
    }

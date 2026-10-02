import csv
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from src.trade_logger import calculate_round_trip_commission, record_trade, session_aggregates


class TestTradeLogger(unittest.TestCase):
    def test_round_trip_commission(self):
        self.assertAlmostEqual(calculate_round_trip_commission(100, 110, 2, 10), 1.05)

    def test_record_and_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "trades.csv")
            with patch.dict(os.environ, {"TRADES_FILE": path}):
                record_trade(datetime(2026, 9, 2, tzinfo=timezone.utc), datetime(2026, 9, 2, 1, tzinfo=timezone.utc), "LONG", 100, 110, 1, 10, 2, "TP")
                record_trade(datetime(2026, 9, 2, 2, tzinfo=timezone.utc), datetime(2026, 9, 2, 3, tzinfo=timezone.utc), "SHORT", 100, 105, 1, -5, 1, "SL")
                result = session_aggregates("2026-09-02")
                self.assertEqual(result["trade_count"], 2)
                self.assertEqual(result["total_pnl"], 2.0)
                self.assertEqual(result["win_rate"], 0.5)
                self.assertEqual(result["profit_factor"], 8 / 6)
                with open(path, newline="") as file:
                    self.assertEqual(len(list(csv.DictReader(file))), 2)


if __name__ == "__main__":
    unittest.main()

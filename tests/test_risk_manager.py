import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.risk_manager import calculate_realized_pnl, check_circuit_breaker, session_date


class TestRiskManager(unittest.TestCase):
    def test_session_date_starts_at_three_moscow(self):
        before_start = datetime(2026, 9, 1, 23, 59, tzinfo=timezone.utc)
        after_start = datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(session_date(before_start), "2026-09-01")
        self.assertEqual(session_date(after_start), "2026-09-02")

    def test_realized_pnl_sums_target_operations(self):
        operations = [
            SimpleNamespace(figi="figi", payment=SimpleNamespace(units=-100, nano=0)),
            SimpleNamespace(figi="other", payment=SimpleNamespace(units=-500, nano=0)),
            SimpleNamespace(figi="figi", payment=SimpleNamespace(units=40, nano=0)),
        ]
        self.assertEqual(calculate_realized_pnl(operations, "figi"), -60.0)

    @patch("src.risk_manager.get_client")
    def test_circuit_breaker_halts_at_limit(self, get_client):
        client = MagicMock()
        get_client.return_value.__enter__.return_value = client
        client.operations.get_portfolio.return_value = SimpleNamespace(
            total_amount_portfolio=1000
        )
        client.operations.get_operations.return_value = SimpleNamespace(
            operations=[SimpleNamespace(figi="figi", payment=-30)]
        )
        state = {}

        decision = check_circuit_breaker(
            token="token",
            account_id="account",
            figi="figi",
            state=state,
            max_daily_loss_pct=0.03,
            now=datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc),
        )

        self.assertFalse(decision.allowed)
        self.assertTrue(state["TRADING_HALTED"])
        self.assertEqual(state["daily_loss_pct"], 0.03)

    @patch("src.risk_manager.get_client")
    def test_new_session_clears_halt(self, get_client):
        client = MagicMock()
        get_client.return_value.__enter__.return_value = client
        client.operations.get_portfolio.return_value = SimpleNamespace(
            total_amount_portfolio=1000
        )
        client.operations.get_operations.return_value = SimpleNamespace(operations=[])
        state = {
            "daily_session_date": "2026-09-01",
            "daily_start_balance": 1000,
            "TRADING_HALTED": True,
        }

        decision = check_circuit_breaker(
            token="token",
            account_id="account",
            figi="figi",
            state=state,
            max_daily_loss_pct=0.03,
            now=datetime(2026, 9, 3, 2, 0, tzinfo=timezone.utc),
        )

        self.assertTrue(decision.allowed)
        self.assertFalse(state["TRADING_HALTED"])
        self.assertEqual(state["daily_session_date"], "2026-09-03")


if __name__ == "__main__":
    unittest.main()

import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.auto_trader import _ensure_market_order_available
from src.market_data import is_trading_time


class TestTradingSchedule(unittest.TestCase):
    def test_before_session_is_closed(self):
        self.assertFalse(is_trading_time(datetime(2026, 9, 2, 23, 59, tzinfo=timezone.utc)))

    def test_after_session_start_is_open(self):
        self.assertTrue(is_trading_time(datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)))

    @patch("src.auto_trader.retry_api_call")
    def test_api_status_blocks_market_orders(self, retry_call):
        client = MagicMock()
        status_method = MagicMock()
        retry_call.return_value = status_method
        status_method.return_value = SimpleNamespace(
            api_trade_available_flag=False,
            market_order_available_flag=False,
            trading_status="SECURITY_TRADING_STATUS_BREAK_IN_TRADING",
        )

        with patch("src.market_data.is_trading_time", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "Market orders unavailable"):
                _ensure_market_order_available(client, "uid")


if __name__ == "__main__":
    unittest.main()

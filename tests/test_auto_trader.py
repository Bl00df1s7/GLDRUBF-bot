import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.auto_trader import (
    calculate_position_size,
    get_account_balance,
    open_position,
    wait_for_order_fill,
)


class TestOrderExecution(unittest.TestCase):
    @patch("src.auto_trader.get_client")
    def test_account_balance_uses_free_rub_cash(self, get_client):
        client = MagicMock()
        get_client.return_value.__enter__.return_value = client
        client.operations.get_positions.return_value = SimpleNamespace(
            money=[SimpleNamespace(currency="rub", units=10000, nano=0)],
            blocked=[SimpleNamespace(currency="rub", units=2500, nano=0)],
        )

        self.assertEqual(get_account_balance("token", "account"), 7500)
        client.operations.get_portfolio.assert_not_called()

    @patch("src.auto_trader._get_margin_per_lot", return_value=1000.0)
    def test_position_size_uses_blocked_margin(self, _margin):
        lots = calculate_position_size(10000, 15000, 1, "uid", "token", "account")
        self.assertEqual(lots, 9)

    def test_wait_for_fill_accepts_filled_order(self):
        client = MagicMock()
        client.orders.get_order_state.return_value = SimpleNamespace(
            execution_report_status="EXECUTION_REPORT_STATUS_FILL",
            lots_executed=3,
        )

        state, lots = wait_for_order_fill(client, "account", "order")

        self.assertEqual(lots, 3)
        self.assertEqual(state.execution_report_status, "EXECUTION_REPORT_STATUS_FILL")
        client.orders.get_order_state.assert_called_once_with(
            account_id="account", order_id="order"
        )

    def test_wait_for_fill_uses_partial_quantity(self):
        client = MagicMock()
        client.orders.get_order_state.return_value = SimpleNamespace(
            execution_report_status="EXECUTION_REPORT_STATUS_PARTIALLYFILL",
            lots_executed=2,
        )

        _, lots = wait_for_order_fill(client, "account", "order")

        self.assertEqual(lots, 2)

    @patch("src.stop_orders.place_protection_orders")
    @patch("src.auto_trader.get_client")
    def test_unfilled_open_does_not_create_protection(
        self, get_client, place_protection_orders
    ):
        client = MagicMock()
        get_client.return_value.__enter__.return_value = client
        client.orders.post_order.return_value = SimpleNamespace(order_id="order")
        client.orders.get_order_state.return_value = SimpleNamespace(
            execution_report_status="EXECUTION_REPORT_STATUS_REJECTED",
            lots_executed=0,
        )

        success, message = open_position(
            token="token",
            account_id="account",
            instrument_uid="uid",
            direction="LONG",
            quantity_lots=3,
            price=100.0,
            stop_loss=90.0,
            take_profit=110.0,
        )

        self.assertFalse(success)
        self.assertIn("not filled", message)
        place_protection_orders.assert_not_called()
        self.assertNotIn("order_id", client.orders.post_order.call_args.kwargs)

    @patch("src.stop_orders.place_protection_orders", return_value={"stop_loss": "sl", "take_profit": "tp"})
    @patch("src.auto_trader.get_client")
    def test_partial_open_protects_actual_quantity(
        self, get_client, place_protection_orders
    ):
        client = MagicMock()
        get_client.return_value.__enter__.return_value = client
        client.orders.post_order.return_value = SimpleNamespace(order_id="order")
        client.orders.get_order_state.return_value = SimpleNamespace(
            execution_report_status="EXECUTION_REPORT_STATUS_PARTIALLYFILL",
            lots_executed=2,
        )
        state = {}

        success, _ = open_position(
            token="token",
            account_id="account",
            instrument_uid="uid",
            direction="LONG",
            quantity_lots=3,
            price=100.0,
            stop_loss=90.0,
            take_profit=110.0,
            state=state,
        )

        self.assertTrue(success)
        self.assertEqual(state["stop_order_id"], "sl")
        self.assertEqual(place_protection_orders.call_args.kwargs["quantity_lots"], 2)
        self.assertNotIn("order_id", client.orders.post_order.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()

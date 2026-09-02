import unittest
from unittest.mock import MagicMock, patch

from src.instruments import quantize_lots, quantize_price
from src.stop_orders import (
    place_protection_orders,
    quotation_from_float,
    cancel_protection_orders,
    replace_protection_orders,
)


class TestStopOrders(unittest.TestCase):
    def test_quantization(self):
        self.assertEqual(quantize_lots(27, 10), 20)
        self.assertEqual(quantize_lots(9, 10), 0)
        self.assertEqual(quantize_price(12055.16, 0.1), 12055.2)

    def test_quotation_conversion(self):
        quotation = quotation_from_float(12055.10)
        self.assertEqual(quotation.units, 12055)
        self.assertEqual(quotation.nano, 100000000)

    @patch("src.stop_orders.get_client")
    def test_place_creates_server_sl_and_tp(self, get_client):
        client = MagicMock()
        get_client.return_value.__enter__.return_value = client
        client.stop_orders.post_stop_order.side_effect = [
            MagicMock(stop_order_id="sl-1"),
            MagicMock(stop_order_id="tp-1"),
        ]

        result = place_protection_orders(
            token="token",
            account_id="account",
            instrument_uid="uid",
            quantity_lots=2,
            position_direction="LONG",
            stop_price=100.0,
            take_profit_price=110.0,
            min_price_increment=0.1,
        )

        self.assertEqual(result, {"stop_loss": "sl-1", "take_profit": "tp-1"})
        calls = client.stop_orders.post_stop_order.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs["quantity"], 2)
        self.assertEqual(calls[0].kwargs["direction"].name, "STOP_ORDER_DIRECTION_SELL")
        self.assertEqual(calls[0].kwargs["stop_order_type"].name, "STOP_ORDER_TYPE_STOP_LOSS")
        self.assertEqual(calls[1].kwargs["stop_order_type"].name, "STOP_ORDER_TYPE_TAKE_PROFIT")

    @patch("src.stop_orders.get_client")
    def test_cancel_protection_orders(self, get_client):
        client = MagicMock()
        get_client.return_value.__enter__.return_value = client

        result = cancel_protection_orders("token", "account", ["sl-1", "tp-1"])

        self.assertEqual(result, ["sl-1", "tp-1"])
        self.assertEqual(client.stop_orders.cancel_stop_order.call_count, 2)

    @patch("src.stop_orders.place_protection_orders")
    @patch("src.stop_orders.cancel_protection_orders")
    def test_replace_protection_orders(self, cancel_orders, place_orders):
        place_orders.return_value = {"stop_loss": "new-sl", "take_profit": "new-tp"}

        result = replace_protection_orders(
            token="token",
            account_id="account",
            instrument_uid="uid",
            quantity_lots=2,
            position_direction="LONG",
            old_stop_order_ids=["old-sl", "old-tp"],
            stop_price=101.0,
            take_profit_price=110.0,
        )

        cancel_orders.assert_called_once_with("token", "account", ["old-sl", "old-tp"])
        self.assertEqual(result["stop_loss"], "new-sl")


if __name__ == "__main__":
    unittest.main()

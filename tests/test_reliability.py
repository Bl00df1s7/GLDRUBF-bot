import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.lock import process_lock
from src.preflight import preflight_check
from src.observability import configure_logging


class TestReliability(unittest.TestCase):
    def test_process_lock_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bot.lock")
            with process_lock(path):
                self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(path))

    @patch("src.preflight.get_client")
    def test_preflight_requires_account(self, get_client):
        client = MagicMock()
        get_client.return_value.__enter__.return_value = client
        client.users.get_accounts.return_value = SimpleNamespace(accounts=[])
        self.assertEqual(preflight_check("token"), (False, "No T-Invest accounts found"))

    def test_json_log_contains_correlation_id(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"LOG_DIR": directory}):
                logger, correlation_id = configure_logging()
                logger.info("test_event", extra={"correlation_id": correlation_id})
                for filename in os.listdir(directory):
                    with open(os.path.join(directory, filename)) as file:
                        records = [json.loads(line) for line in file]
                self.assertTrue(records)
                self.assertEqual(records[-1]["correlation_id"], correlation_id)


if __name__ == "__main__":
    unittest.main()

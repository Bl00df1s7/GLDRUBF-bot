import unittest

import grpc

from src.api_retry import ApiRateLimitError, _is_retryable, _raise_rate_limit


class TestApiRetry(unittest.TestCase):
    def test_transient_statuses_are_retryable(self):
        unavailable = RuntimeError(grpc.StatusCode.UNAVAILABLE, "temporary")
        deadline = RuntimeError(grpc.StatusCode.DEADLINE_EXCEEDED, "timeout")
        self.assertTrue(_is_retryable(unavailable))
        self.assertTrue(_is_retryable(deadline))

    def test_rate_limit_is_not_retryable(self):
        error = RuntimeError(grpc.StatusCode.RESOURCE_EXHAUSTED, "limited")
        self.assertFalse(_is_retryable(error))
        with self.assertRaises(ApiRateLimitError):
            _raise_rate_limit(error)

    def test_other_errors_are_not_retryable(self):
        error = RuntimeError(grpc.StatusCode.INVALID_ARGUMENT, "bad request")
        self.assertFalse(_is_retryable(error))


if __name__ == "__main__":
    unittest.main()

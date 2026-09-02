"""Startup checks required before a trading run."""

from typing import Tuple

from src.client_factory import get_client


def preflight_check(token: str) -> Tuple[bool, str]:
    """Check token access and presence of an account before trading."""
    if not token:
        return False, "T-Invest token is empty"
    try:
        with get_client(token) as client:
            accounts = client.users.get_accounts().accounts
            if not accounts:
                return False, "No T-Invest accounts found"
            return True, f"API reachable; accounts={len(accounts)}"
    except Exception as error:
        return False, f"API preflight failed: {error}"

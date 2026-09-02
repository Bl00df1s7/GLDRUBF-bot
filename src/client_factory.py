"""
T-Invest API client factory.
Works around gRPC DNS resolution issues on macOS by resolving
the hostname via system resolver and connecting by IP with SSL override.
"""

import socket
from contextlib import contextmanager

try:
    from t_tech.invest import Client
    T_TECH_AVAILABLE = True
except ImportError:
    T_TECH_AVAILABLE = False
    Client = None

API_HOST = "invest-public-api.tbank.ru"
API_PORT = 443


def _resolve_api_ip() -> str:
    """Resolve API hostname to IP using system DNS resolver."""
    try:
        ip = socket.getaddrinfo(API_HOST, API_PORT)[0][4][0]
        return ip
    except Exception as e:
        raise RuntimeError(f"Cannot resolve {API_HOST}: {e}")


@contextmanager
def get_client(token: str):
    """
    Context manager that yields a working T-Invest API client.
    Automatically handles DNS resolution workaround for macOS/gRPC.

    Usage:
        with get_client(token) as client:
            response = client.instruments.futures()
    """
    if not T_TECH_AVAILABLE:
        raise RuntimeError("t_tech.invest module not available")

    ip = _resolve_api_ip()
    target = f"{ip}:{API_PORT}"

    channel_options = [
        ("grpc.ssl_target_name_override", API_HOST),
        ("grpc.default_authority", API_HOST),
    ]

    with Client(token, target=target, options=channel_options) as client:
        yield client

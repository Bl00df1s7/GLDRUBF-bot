"""Live margin (ГО) provider for C5 — no hardcoded fallbacks.

Margin is ALWAYS fetched from the broker via
``get_future(CNYRUBF_SPBFUT, responseView=FULL)``:

  * long  → ``dlongClient``
  * short → ``dshortClient``

Values are per-contract initial margin (ГО) in RUB. If the API does not
return a valid current rate, this module raises ``MarginApiUnavailable``
and the caller MUST block entry (``MARGIN_API_UNAVAILABLE`` → NO ENTRY).
Falling back to any remembered, average or hardcoded value
(e.g. 0.0585 / 0.0576) is forbidden by design.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from config.settings import C5_MARGIN_RATE_CHANGE_EPS
from src.client_factory import get_client
from src.instruments import InstrumentIdType
from src.market_data import quotation_to_float

logger = logging.getLogger("golden_bot.margin")

MARGIN_EVENT_CHANGED = "MARGIN_RATE_CHANGED"
MARGIN_ERROR_UNAVAILABLE = "MARGIN_API_UNAVAILABLE"


class MarginApiUnavailable(RuntimeError):
    """Raised when the broker did not provide a valid current ГО."""


@dataclass
class MarginRates:
    """Current per-contract initial margin (ГО) in RUB per direction."""

    long_margin_rub: float
    short_margin_rub: float
    instrument: str

    def for_direction(self, direction: str) -> float:
        if direction == "LONG":
            return self.long_margin_rub
        if direction == "SHORT":
            return self.short_margin_rub
        raise ValueError(f"Invalid direction: {direction}")


def _valid(value: Optional[float]) -> bool:
    # `value == value` excludes NaN
    return value is not None and value > 0.0 and value == value


def fetch_margin_rates(token: str, instrument_id: str) -> MarginRates:
    """Fetch live dlongClient/dshortClient ГО through get_future(FULL).

    Raises:
        MarginApiUnavailable: on API error or missing/invalid rates.
            The reason is logged explicitly; there is NO fallback path.
    """
    try:
        with get_client(token) as client:
            # get_future(CNYRUBF_SPBFUT, responseView=FULL). The installed
            # t-tech-investments SDK exposes this as instruments.future_by(
            # id_type=TICKER, class_code="SPBFUT", id="CNYRUBF"); the dataclass
            # field names are snake_case (dlong_client / dshort_client) and
            # carry the live per-contract ГО in RUB — the direct equivalent of
            # the REST responseView=FULL payload. No other source is permitted.
            #
            # SDK 1.51.0 contract: FutureBy requires id_type; without it the
            # server rejects the request with INVALID_ARGUMENT 30006
            # "Missing parameter: id_type". We look the instrument up by
            # ticker (CNYRUBF), so INSTRUMENT_ID_TYPE_TICKER is required
            # (same shape as src.instruments.get_c5_instrument).
            response = client.instruments.future_by(
                id_type=InstrumentIdType.INSTRUMENT_ID_TYPE_TICKER,
                class_code="SPBFUT",
                id=instrument_id,
            )
    except Exception as exc:  # network / gRPC / SDK errors
        logger.error(
            "%s: get_future(%s, FULL) call failed: %s",
            MARGIN_ERROR_UNAVAILABLE, instrument_id, exc,
        )
        raise MarginApiUnavailable(
            f"{MARGIN_ERROR_UNAVAILABLE}: {exc}"
        ) from exc

    future = getattr(response, "instrument", None) or response

    long_raw = quotation_to_float(
        getattr(future, "dlong_client", None)
        if getattr(future, "dlong_client", None) is not None
        else getattr(future, "dlongClient", None)
    )
    short_raw = quotation_to_float(
        getattr(future, "dshort_client", None)
        if getattr(future, "dshort_client", None) is not None
        else getattr(future, "dshortClient", None)
    )

    if not (_valid(long_raw) and _valid(short_raw)):
        logger.error(
            "%s: invalid margin rates from API for %s "
            "(dlongClient=%r, dshortClient=%r). Entry is blocked.",
            MARGIN_ERROR_UNAVAILABLE, instrument_id, long_raw, short_raw,
        )
        raise MarginApiUnavailable(
            f"{MARGIN_ERROR_UNAVAILABLE}: dlongClient={long_raw!r}, "
            f"dshortClient={short_raw!r} for {instrument_id}"
        )

    return MarginRates(
        long_margin_rub=long_raw,
        short_margin_rub=short_raw,
        instrument=instrument_id,
    )


class MarginTracker:
    """Caches the last fetched live rates and emits change events.

    The cached copy exists only to detect MARGIN_RATE_CHANGED deltas; it
    is never used as a substitute when the API fails — every sizing
    decision re-fetches current values (startup, before entry, after
    position change, after reconnect, after API recovery).
    """

    def __init__(self, token: str, instrument_id: str):
        self.token = token
        self.instrument_id = instrument_id
        self.current: Optional[MarginRates] = None

    def refresh(self) -> MarginRates:
        """Fetch fresh rates; log MARGIN_RATE_CHANGED on any delta."""
        fresh = fetch_margin_rates(self.token, self.instrument_id)

        if self.current is not None:
            changed = (
                abs(fresh.long_margin_rub - self.current.long_margin_rub)
                > C5_MARGIN_RATE_CHANGE_EPS
                or abs(fresh.short_margin_rub - self.current.short_margin_rub)
                > C5_MARGIN_RATE_CHANGE_EPS
            )
            if changed:
                logger.warning(
                    "%s: %s long %.2f→%.2f, short %.2f→%.2f RUB/contract. "
                    "Recalculating required margin, margin_qty, M/E and "
                    "allowed quantity.",
                    MARGIN_EVENT_CHANGED,
                    self.instrument_id,
                    self.current.long_margin_rub, fresh.long_margin_rub,
                    self.current.short_margin_rub, fresh.short_margin_rub,
                )

        self.current = fresh
        return fresh

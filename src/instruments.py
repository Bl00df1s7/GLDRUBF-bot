"""Instrument discovery and price/quantity quantization."""

from decimal import Decimal, ROUND_HALF_UP

try:
    from t_tech.invest import CandleInterval
    T_TECH_AVAILABLE = True
except ImportError:
    T_TECH_AVAILABLE = False
    CandleInterval = None

from src.client_factory import get_client


def _quotation_to_decimal(value) -> Decimal:
    """Convert a numeric value or SDK Quotation to Decimal."""
    if hasattr(value, "units") and hasattr(value, "nano"):
        return Decimal(value.units) + (Decimal(value.nano) / Decimal(1_000_000_000))
    return Decimal(str(value))


def quantize_price(price: float, min_price_increment) -> float:
    """Round a price to the nearest instrument price increment."""
    increment = _quotation_to_decimal(min_price_increment)
    if increment <= 0:
        raise ValueError("min_price_increment must be positive")
    value = _quotation_to_decimal(price)
    steps = (value / increment).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return float(steps * increment)


def quantize_lots(quantity: int, lot_size: int) -> int:
    """Round base-unit quantity down to a whole number of instrument lots."""
    if lot_size <= 0:
        raise ValueError("lot_size must be positive")
    if quantity <= 0:
        return 0
    return (int(quantity) // int(lot_size)) * int(lot_size)


def get_gldrubf_instrument(token: str) -> dict:
    """
    Find GLDRUBF futures instrument.
    
    Args:
        token: T-Invest API token
        
    Returns:
        Instrument object with uid, figi, ticker, etc.
        
    Raises:
        RuntimeError: If t_tech not available or instrument not found
    """
    from config.settings import TARGET_TICKER
    
    if not T_TECH_AVAILABLE:
        raise RuntimeError("t_tech.invest module not available. Install with: pip install t-tech")
    
    with get_client(token) as client:
        response = client.instruments.futures()
        futures = response.instruments
    
    instrument = None
    
    for x in futures:
        if x.ticker.upper() == TARGET_TICKER:
            instrument = x
            break
    
    if instrument is None:
        raise RuntimeError(f"Фьючерс {TARGET_TICKER} не найден")
    
    print("=== INSTRUMENT ===")
    print(f"Ticker:       {instrument.ticker}")
    print(f"Name:         {instrument.name}")
    print(f"UID:          {instrument.uid}")
    print(f"FIGI:         {instrument.figi}")
    print(f"Class code:   {instrument.class_code}")
    print(f"Lot:          {instrument.lot}")
    print(f"Min tick:     {instrument.min_price_increment}")
    
    return instrument

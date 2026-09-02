"""
Position discovery and management.
SIGNAL ONLY MODE - Read-only position data, no trading operations.
"""

import numpy as np

try:
    from t_tech.invest import Client
    T_TECH_AVAILABLE = True
except ImportError:
    T_TECH_AVAILABLE = False
    Client = None

from src.client_factory import get_client

from config.settings import TARGET_TICKER, SL_ATR, TP_PCT, BE_PCT


def find_gldrubf_position(token: str, instrument) -> dict:
    """
    Search for GLDRUBF position across all accounts.
    READ-ONLY MODE - No trading operations.
    
    Args:
        token: T-Invest API token
        instrument: Instrument object with figi
        
    Returns:
        Dictionary with position info or None if no position
        
    Raises:
        RuntimeError: If t_tech not available
    """
    if not T_TECH_AVAILABLE:
        raise RuntimeError("t_tech.invest module not available")
    
    print("\n=== SEARCHING FOR GLDRUBF POSITION ===\n")
    
    gldrubf_position = None
    position_account_id = None
    position_account_name = None
    
    # Normalize target ticker for comparison
    target_ticker_normalized = TARGET_TICKER.upper()
    # Also create base ticker without any suffix for flexible matching
    target_ticker_base = target_ticker_normalized.split('_')[0]
    
    print(f"Searching for ticker: {target_ticker_normalized}")
    print(f"Base ticker for matching: {target_ticker_base}\n")
    
    # Keep context open for all operations
    with get_client(token) as client:
        accounts_response = client.users.get_accounts()
        accounts = accounts_response.accounts
        
        if not accounts:
            raise RuntimeError("Для этого токена не найдено ни одного счёта")
        
        print(f"Найдено счетов: {len(accounts)}\n")
        
        for account in accounts:
            account_id = account.id
            
            print("=" * 60)
            print(f"ACCOUNT: {account_id}")
            print(f"NAME:    {account.name}")
            print(f"TYPE:    {account.type}")
            print(f"STATUS:  {account.status}\n")
            
            try:
                positions_response = client.operations.get_positions(
                    account_id=account_id
                )
            except Exception as e:
                print(f"⚠️ Не удалось получить позиции: {e}")
                print("Счёт пропускаем.\n")
                continue
            
            futures_positions = positions_response.futures
            print(f"Futures positions: {len(futures_positions)}\n")
            
            # Debug: print all futures positions
            if len(futures_positions) > 0:
                print("📋 All futures positions in this account:")
                for pos in futures_positions:
                    pos_ticker = pos.ticker.upper()
                    pos_base = pos_ticker.split('_')[0]
                    match_exact = "✓ EXACT" if pos_ticker == target_ticker_normalized else ""
                    match_base = "✓ BASE" if pos_base == target_ticker_base else ""
                    matches = f"{match_exact} {match_base}".strip()
                    print(f"   - {pos.ticker} (balance: {pos.balance}, figi: {pos.figi}) {matches}")
                print()
            
            # Search for GLDRUBF - EXACT LOGIC FROM WORKING SCRIPT
            for position in futures_positions:
                position_ticker = position.ticker.upper()
                
                # Exact match check first (like old script)
                if position_ticker != target_ticker_normalized:
                    # Try matching without suffix (e.g., GLDRUBF_TOM vs GLDRUBF)
                    base_ticker = position_ticker.split('_')[0]
                    if base_ticker != target_ticker_base:
                        continue
                
                balance = float(position.balance)
                
                print(f"🎯 GLDRUBF FOUND in account {account_id}")
                print(f"   Ticker:   {position.ticker}")
                print(f"   Balance:  {balance}")
                print(f"   Blocked:  {position.blocked}\n")
                
                if balance != 0:
                    gldrubf_position = {
                        "position": position,
                        "account_id": account_id,
                        "account_name": account.name,
                        "balance": balance,
                        "direction": "LONG" if balance > 0 else "SHORT",
                    }
                    
                    print(f"✅ ACTIVE POSITION: {gldrubf_position['direction']} × {balance}\n")
                    break
            
            # Stop searching if position found
            if gldrubf_position is not None:
                break
    
    print("=" * 60)
    print("FINAL POSITION STATE")
    print("=" * 60)
    
    if gldrubf_position is None:
        print("⚪ FINAL: NO GLDRUBF POSITION\n")
    else:
        pos = gldrubf_position
        print(f"Account:     {pos['account_name']}")
        print(f"Account ID:  {pos['account_id']}")
        print(f"Direction:   {pos['direction']}")
        print(f"Quantity:    {pos['balance']}\n")
    
    return gldrubf_position


def get_position_state(
    position_info: dict,
    last_closed: dict,
    current_price: float,
    token: str
) -> dict:
    """
    Get complete position state with entry price and levels.
    READ-ONLY MODE - No trading operations.
    
    Args:
        position_info: Position info from find_gldrubf_position
        last_closed: Last closed candle data
        current_price: Current market price (not used for signals)
        token: T-Invest API token
        
    Returns:
        Dictionary with full position state
        
    Raises:
        RuntimeError: If t_tech not available
    """
    if not T_TECH_AVAILABLE:
        raise RuntimeError("t_tech.invest module not available")
    
    if position_info is None:
        return {
            "direction": "NONE",
            "quantity": 0.0,
            "account_id": None,
            "account_name": None,
            "entry_price": np.nan,
            "entry_atr": float(last_closed["atr"]),
            "sl_price": np.nan,
            "tp_price": np.nan,
            "be_trigger": np.nan,
            "sar_price": float(last_closed["sar"]),
        }
    
    # Get average entry price from portfolio
    account_id = position_info["account_id"]
    figi = position_info["position"].figi
    
    entry_price = np.nan
    
    with get_client(token) as client:
        portfolio = client.operations.get_portfolio(account_id=account_id)
        
        for portfolio_pos in portfolio.positions:
            if portfolio_pos.figi == figi:
                from src.market_data import quotation_to_float
                entry_price = quotation_to_float(portfolio_pos.average_position_price)
                break
    
    if np.isnan(entry_price):
        print("⚠️ Не удалось получить среднюю цену позиции")
    
    # Calculate levels
    entry_atr = float(last_closed["atr"])
    direction = position_info["direction"]
    
    if direction == "LONG":
        sl_price = entry_price - entry_atr * SL_ATR
        tp_price = entry_price * (1 + TP_PCT)
        be_trigger = entry_price * (1 + BE_PCT)
    else:  # SHORT
        sl_price = entry_price + entry_atr * SL_ATR
        tp_price = entry_price * (1 - TP_PCT)
        be_trigger = entry_price * (1 - BE_PCT)
    
    return {
        "direction": direction,
        "quantity": position_info["balance"],
        "account_id": position_info["account_id"],
        "account_name": position_info["account_name"],
        "entry_price": entry_price,
        "entry_atr": entry_atr,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "be_trigger": be_trigger,
        "sar_price": float(last_closed["sar"]),
    }

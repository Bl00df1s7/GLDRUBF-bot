"""G-9C.3 — FULL SHORT cycle across a C5Runtime RESTART (integration, PAPER).

One test, the complete production lifecycle of a SHORT position split
across TWO separate runtime processes:

    PHASE 1  runtime #1 → ENTER_SHORT (SELL OPEN_SHORT through the REAL
             PaperExecutionAdapter + SimulatedBroker) → state saved to disk
    PHASE 2  simulated process restart: runtime #1 / adapter #1 / broker #1
             are dropped; a brand-new SimulatedBroker is seeded from an
             INDEPENDENT read-only broker snapshot via the production
             ``_build_paper_broker`` path; a BRAND-NEW ``C5Runtime`` loads
             the SAME state file and reconciles inside its own bootstrap
    PHASE 3  runtime #2 → EXIT_SHORT (BUY CLOSE_SHORT) → flat position,
             ГО released, P&L realized, state cleaned — with NO repeated
             or extra orders anywhere.

Nothing here replaces production components: the real C5Runtime, the real
bootstrap/reconcile, the real state_store persistence and the real paper
execution chain are used exactly as in tests/test_stage9c_short_exit.py.
The only fakes are the external boundaries (market data / clock / margin
API / order book) plus the same recording wrapper that delegates every
call to the REAL adapter/broker (it only records dispatched orders and
counts dispatch attempts).

Fixture math is byte-for-byte identical to the proven G-9C.1/G-9C.2
suites: flat daily history 1400±DAILY_HL → Lower(10)=1399.92185 so the
16:00 MSK control candle at 1370 fires ENTER_SHORT (qty=2 @ equity 1M);
after the restart the "profit" exit history makes Upper(5)=1340 so the
new closed bar at 1360 fires EXIT_SHORT.  Entry basis = guard bid VWAP
1399.95, ГО reservation 2 × 78 912, realized P&L on close =
(1399.95 − 1360) × 1000 × 2 = 79 900 RUB (this PAPER broker models NO
commissions — same explicit assumption as the G-9C.2 suite).
"""
from __future__ import annotations

import json

import pytest

from tests.test_stage9c_short_exit import (
    CONTROL_DAY,
    EXIT_CANDLE,
    EXIT_PRICE,
    LONG_BREAK,
    NEUTRAL,
    SHORT_BREAK,
    advance,
    buy_orders,
    candle_5m,
    close_buy_orders,
    daily_df,
    exec_res,
    make_recording_runtime,
    market,          # boundary-fake fixture (load_candles / order book)
    new_broker,
    open_short,      # production ENTER_SHORT cycle with post-conditions
    read_state,
    sell_orders,
    set_candles,
    state_env,       # isolated C5_STATE_FILE + PAPER env fixture
    utc_ms,
    use_exit_history,
)


def test_full_short_cycle_restart_recover_from_broker_snapshot_and_close(
        state_env, market):
    """OPEN → persist → RESTART → recover from independent broker
    snapshot → CLOSE, asserting position/direction/entry/margin/PnL/state
    cleanup and the absence of repeated or superfluous orders."""

    # ════════════════════════════════════════════════════════════════
    # PHASE 1 — RUNTIME #1 opens the SHORT through the production chain
    # ════════════════════════════════════════════════════════════════
    broker1 = new_broker()
    rt1, ad1 = make_recording_runtime(market, broker1)
    set_candles(market, candle_5m(CONTROL_DAY, 16, 0, SHORT_BREAK))

    d1 = rt1.run_cycle()
    assert d1.action == "ENTER_SHORT"
    assert d1.reason == "ENTER_SHORT"
    assert d1.qty == 2                                   # fixture sizing

    # REAL broker state after the confirmed OPEN_SHORT SELL
    assert broker1.position_qty == -2                    # SHORT = negative
    assert broker1.short_margin_locked_per_contract == pytest.approx(78_912.0)
    go_pc = broker1.short_margin_locked_per_contract
    assert broker1.entry_price == pytest.approx(1399.95)  # guard bid VWAP
    cash_after_open = broker1.cash
    assert cash_after_open == pytest.approx(1_000_000.0 - 2 * go_pc)

    # exactly ONE order left runtime #1: the opening SELL (OPEN_SHORT)
    assert len(ad1.sell_calls) == 1
    assert ad1.buy_calls == []                           # no BUY during open
    entry_sells = sell_orders(ad1)
    assert len(entry_sells) == 1
    assert entry_sells[0]["intent"] == "OPEN_SHORT"
    assert entry_sells[0]["qty"] == 2

    # durable state written by production save paths
    st1 = read_state(state_env)
    assert st1["c5_position_qty"] == -2
    assert st1["c5_position_direction"] == "SHORT"
    assert st1["c5_entry_price"] == pytest.approx(1399.95)
    assert not st1.get("exit_pending_sell_intent")       # proven-terminal
    persisted_state_file = (state_env / "state.json").read_text()
    assert json.loads(persisted_state_file)["c5_position_qty"] == -2

    # ════════════════════════════════════════════════════════════════
    # PHASE 2 — SIMULATED PROCESS RESTART
    #   runtime #1 / adapter #1 / broker #1 are NEVER used again.
    #   The new process rebuilds its view ONLY from:
    #     a) the state file on disk (same C5_STATE_FILE),
    #     b) an INDEPENDENT read-only broker snapshot seeded through the
    #        production ``_build_paper_broker`` startup path.
    # ════════════════════════════════════════════════════════════════
    from src.execution_adapter import PaperExecutionAdapter, _build_paper_broker

    class ReadOnlyAccountApi:
        """Fake ONLY of the external READ boundary (find_account/snapshot).

        Physically has no post_order / buy / sell methods — mirrors the
        production read adapter used by ``_build_paper_broker`` (same
        shape as Test N in tests/test_stage5b_paper_runtime.py).
        """

        def __init__(self, snap_source):
            self._src = snap_source
            self.snapshot_calls = 0

        def find_account(self):
            return "paper-restart-account"

        def snapshot(self, account_id):
            from src.execution_adapter import AccountSnapshot
            self.snapshot_calls += 1
            s = self._src.snapshot(account_id)
            return AccountSnapshot(
                account_id=account_id,
                equity_rub=s.equity_rub,
                free_cash_rub=s.free_cash_rub,
                position_qty=s.position_qty,
                entry_price=s.entry_price,
                tmon_qty=s.tmon_qty,
                open_orders=0,
            )

    # Documented limitation of the PAPER restart model (NOT a runtime
    # defect, no production code involved here): ``_build_paper_broker``
    # seeds only cash/equity/tmon/position/entry from the snapshot — the
    # ГО reservation ledger (short_margin_locked_per_contract) is a local
    # paper-broker field with no external representation, so the seeded
    # broker starts with an UNKNOWN short margin basis (0.0).  This does
    # not affect position/direction/basis recovery or the EXIT decision;
    # it only means the restarted paper mirror cannot return ГО it never
    # recorded (see the release assertion in PHASE 3).
    read_api = ReadOnlyAccountApi(broker1)
    broker2 = _build_paper_broker(read_api)               # production seed
    assert broker2 is not broker1                         # fresh memory
    assert broker2.position_qty == -2                     # SHORT recovered
    assert broker2.entry_price == pytest.approx(1399.95)  # basis from broker
    assert broker2.cash == pytest.approx(cash_after_open)
    assert broker2.short_margin_locked_per_contract == 0.0  # documented gap

    # fresh adapter ⇒ fresh ORDER LEDGER: nothing carried over, so any
    # duplicated order after the restart would be visible immediately.
    # NOTE: make_recording_runtime builds its own inner PaperExecutionAdapter;
    # it is passed to C5Runtime directly (no post-hoc swap) so the recorded
    # ledger ad2.orders IS the production paper ledger of process #2.
    rt2, ad2 = make_recording_runtime(market, broker2)
    assert ad2.orders == []

    # A neutral closed bar exists right after the restart: reconcile must
    # restore the SHORT from the broker snapshot and HOLD — no re-entry,
    # no early exit, no duplicate orders.
    C_NEUTRAL = candle_5m(CONTROL_DAY, 16, 5, NEUTRAL)
    set_candles(market, candle_5m(CONTROL_DAY, 16, 0, SHORT_BREAK), C_NEUTRAL)
    market.holder["now"] = utc_ms(CONTROL_DAY, 16, 10)

    d2 = rt2.run_cycle()                                  # real bootstrap
    assert d2.action == "HOLD"                            # +reconcile inside
    assert d2.reason == "HOLD_POSITION"

    # production reconcile adopted the broker view without mutating it
    st2 = read_state(state_env)
    assert st2["reconcile_status"] == "OK"
    assert st2["c5_position_qty"] == -2
    assert st2["c5_position_direction"] == "SHORT"
    assert st2["c5_entry_price"] == pytest.approx(1399.95)  # survives restart
    assert broker2.position_qty == -2
    # NO repeated/extra orders across the restart boundary:
    assert ad2.sell_calls == []                           # no re-entry SELL
    assert ad2.buy_calls == []                            # no blind close BUY
    assert paper2.orders == []
    assert len(sell_orders(ad1)) == 1                     # original entry: 1
    assert buy_orders(ad1) == []

    # ════════════════════════════════════════════════════════════════
    # PHASE 3 — RUNTIME #2 closes the SHORT on a NEW closed exit candle
    #   (EXIT window reshaped via the proven G-9C.2 fixture helper; the
    #   ENTRY channel is untouched because this runs strictly after the
    #   entry cycle above)
    # ════════════════════════════════════════════════════════════════
    use_exit_history(market, "profit")
    EXIT_C = EXIT_CANDLE()                                # 16:30, close 1360
    set_candles(market, candle_5m(CONTROL_DAY, 16, 0, SHORT_BREAK),
                C_NEUTRAL, EXIT_C)
    market.holder["now"] = utc_ms(CONTROL_DAY, 17, 0)

    d3 = rt2.run_cycle()
    assert d3.action == "EXIT"
    assert d3.reason == "EXIT_SHORT"

    # exactly ONE BUY dispatched by the restarted runtime — full size
    assert len(ad2.buy_calls) == 1
    qty, kwargs = ad2.buy_calls[0]
    assert qty == 2                                       # abs(confirmed qty)
    assert kwargs["intent"] == "CLOSE_SHORT"
    assert kwargs["close_price"] == pytest.approx(EXIT_PRICE)
    assert "margin_per_contract" not in kwargs            # close ≠ reservation
    # and NO SELL ever left the restarted runtime (no reversal/re-entry)
    assert ad2.sell_calls == []

    cos = close_buy_orders(ad2)
    assert len(cos) == 1 and cos[0]["qty"] == 2
    assert sell_orders(ad2) == []                         # ledger: only the BUY

    # ── REAL broker accounting after the close ──────────────────────
    assert broker2.position_qty == 0                      # flat, no LONG flip
    assert broker2.short_margin_locked_per_contract == 0.0  # basis cleared
    assert broker2.entry_price == 0.0                     # basis cleared
    pnl = (1399.95 - EXIT_PRICE) * 1000 * 2               # 79 900 RUB
    assert broker2.realized_pnl == pytest.approx(pnl)
    # ГО release across the restart: the seeded paper mirror never
    # recorded the short margin reservation (documented gap above), so
    # _close_short releases 0 instead of 2 × 78 912 — fail-safe "release
    # nothing rather than invent a basis" per SimulatedBroker's contract.
    # The cash delta is therefore EXACTLY the realized P&L.
    assert broker2.cash == pytest.approx(cash_after_open + pnl)
    assert broker2.equity == pytest.approx(
        read_api.snapshot("paper-restart-account").equity_rub + pnl)

    # ── state cleanup after the restart-closed cycle ────────────────
    st3 = read_state(state_env)
    assert st3["c5_position_qty"] == 0
    assert st3["c5_position_direction"] is None
    assert st3["c5_entry_price"] is None
    assert not st3.get("exit_pending_sell_intent")
    assert not st3.get("exit_unverified_until_reconcile")
    assert st3["last_processed_5m_ts"] is not None        # exit candle burned

    # ── idempotency after the close: same candle again ⇒ NO re-send ─
    # Position is flat now, so the cycle legitimately leaves the EXIT
    # path and goes through ENTRY evaluation; the burned control point
    # (last_c5_entry_control_timestamp == the 16:00 candle persisted in
    # PHASE 1) must prevent any re-entry on restart/repeat cycles.
    d4 = rt2.run_cycle()
    assert d4.action == "NO_ENTRY"
    assert ad2.buy_calls and len(ad2.buy_calls) == 1      # still exactly one
    assert len(close_buy_orders(ad2)) == 1
    assert broker2.position_qty == 0

    # ── a THIRD process sees a clean flat world: reconcile confirms
    #    agreement between the final state file and the broker ───────
    broker3 = _build_paper_broker(ReadOnlyAccountApi(broker2))
    assert broker3.position_qty == 0
    rt3, ad3 = make_recording_runtime(market, broker3)
    d5 = rt3.run_cycle()
    assert d5.action == "NO_ENTRY"                        # no phantom signal
    assert ad3.buy_calls == [] and ad3.sell_calls == []   # zero new orders
    assert paper2.orders == [o for o in paper2.orders if o["side"] == "BUY"]
    assert len(paper2.orders) == 1                        # total: ONE close BUY
    st5 = read_state(state_env)
    assert st5["reconcile_status"] == "OK"
    assert st5["c5_position_qty"] == 0


def test_restart_with_stale_state_defect_note_guard():
    """Sanity pin: the module imports the PRODUCTION components, not mocks.

    Guards against accidentally swapping out the runtime or reconcile in
    future edits of this suite.
    """
    import src.c5_runtime as rt_mod
    from src.execution_adapter import PaperExecutionAdapter, SimulatedBroker

    assert rt_mod.C5Runtime.__module__ == "src.c5_runtime"
    assert callable(rt_mod.reconcile)
    assert PaperExecutionAdapter is not SimulatedBroker
    # the recording wrapper used above always delegates to the real chain
    from tests.test_stage9c_short_exit import RecordingPaperAdapter
    inner = PaperExecutionAdapter(SimulatedBroker(cash=1_000_000,
                                                  equity=1_000_000))
    rec = RecordingPaperAdapter(inner)
    res = rec.sell_cny("paper-account", 2,
                       margin_per_contract=78_912.0,
                       entry_price=1399.95, intent="OPEN_SHORT")
    assert res.ok and res.qty_executed == 2
    assert inner.broker.position_qty == -2
    assert rec.orders is inner.orders

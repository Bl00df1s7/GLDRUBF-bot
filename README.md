# CNYRUBF (C5) Trading Bot — Golden Bot

Торговый runtime для фьючерса **`CNYRUBF_SPBFUT`** (MOEX SPBFUT) по замороженной
тренд-следящей стратегии **C5** (spec v1.1). Активный путь исполнения —
[`src/c5_runtime.py`](src/c5_runtime.py); он единственный подключён к
[`src/main.py`](src/main.py) и GitHub Actions workflow
[`.github/workflows/gold_strategy.yml`](.github/workflows/gold_strategy.yml).

Legacy-код первой версии бота (GLDRUBF 4H: `strategy.py`, `auto_trader.py`,
`stop_orders.py`, `position_monitor.py`, `indicators.calculate_sar`) остаётся в
репозитории, но **не импортируется активным runtime** — это запрещено тестом
`test_legacy_paths_are_not_importable_from_active_runtime`
([`tests/test_stage2d_runtime.py`](tests/test_stage2d_runtime.py)).

---

## 1. Стратегия C5 (заморожена — не добавлять фильтры/стопы/оптимизацию)

Источник истины: [`src/strategies/c5_core.py`](src/strategies/c5_core.py)
(чистая decision-функция), [`src/strategies/c5_signals.py`](src/strategies/c5_signals.py)
(pandas-адаптер с теми же определениями), параметры —
[`config/settings.py`](config/settings.py) (блок `FROZEN C5 STRATEGY PARAMETERS`).

| Параметр | Значение | Константа |
|---|---|---|
| Инструмент | `CNYRUBF` / class code `SPBFUT`, 1 контракт = 1000 CNY | `C5_TARGET_TICKER`, `C5_CONTRACT_SIZE_CNY` |
| Сигнальный TF | закрытые **дневные** бары (`1d`) | `C5_TIMEFRAME` |
| Контрольная точка | одна закрытая 5m-свеча с close в **16:00 MSK**, раз в день | `C5_CONTROL_TIMEFRAME`, `C5_CONTROL_HOUR_MS` |
| Вход | Donchian breakout **N = 10**: close контрольной свечи > Upper(10) → LONG, < Lower(10) → SHORT | `C5_ENTRY_DONCHIAN` |
| Выход | контр-канал **M = 5** только: LONG закрывается при close < Lower(5), SHORT — при close > Upper(5) | `C5_EXIT_DONCHIAN` |
| ATR14 | только для риск-сайзинга, **не** часть сигнала | `C5_ATR_LEN` |
| Risk multiplier | `clip(ATR_TARGET / ATR14, 0.5, 1.5)` (не инвертируется; ATR_TARGET = 0.1563) | `C5_ATR_TARGET`, `C5_RISK_MULT_MIN/MAX` |
| Risk budget | `equity × 0.20 × multiplier` | `C5_RISK_BUDGET_PCT` |
| Объём | `qty = min(qty_risk, qty_margin, MAX_QTY=40)`, затем stage-cap | `C5_MAX_QTY`, `C5_STAGE_MAX_QTY` |
| Маржа/эквити | `M/E > 0.30 ⇒ NO_ENTRY (ME_LIMIT)` | `C5_ME_LIMIT` |
| Stage-лимит | Stage 1: исполняется не более **5** контрактов | `C5_STAGE_MAX_QTY` |
| Slippage guard | expected slippage > **0.10%** ⇒ вход блокируется | `C5_MAX_EXPECTED_SLIPPAGE` |
| Protections | daily loss 10% / drawdown freeze 20% / kill switch 30% | `C5_DAILY_LOSS_BREAKER`, `C5_DRAWDOWN_FREEZE`, `C5_KILL_SWITCH_DD` |

Правила, зафиксированные в коде (docstring `c5_core.py`):

- каналы строятся из **предыдущих** закрытых баров (`shift(1)`) — lookahead
  запрещён; сегодняшняя дневная свеча исключается из истории
  (`fetch_daily_bars(..., exclude_date=today)`);
- при открытой позиции сигналы входа полностью игнорируются — выход только по
  M=5-каналу;
- **нет** SL / TP / Parabolic SAR / break-even / trailing / time-stop /
  дополнительных фильтров;
- contradictory breakout (long+short одновременно) — anomaly guard, no action
  (`DONCHIAN_ANOMALY_BOTH_SIGNALS` в `c5_signals.py`);
- словарь причин решения — §9 спеки: константы `REASON_*` в начале
  `c5_core.py` (`NO_SIGNAL`, `ENTER_LONG/SHORT`, `EXIT_LONG/SHORT`, `HOLD_POSITION`,
  `QTY_ZERO`, `ME_LIMIT`, `MARGIN_API_UNAVAILABLE`, `DATA_INSUFFICIENT`,
  `STAGE_CAPPED`; внешние блокеры — `DAILY_LOSS_BREAKER`, `DRAWDOWN_FREEZE`,
  `KILL_SWITCH`, `SLIPPAGE_GUARD` через `block_entry()`).

`evaluate(daily_bars, control_bar, position, equity, price, margin_rate)` —
единственная точка принятия решения; возвращает иммутабельный
`Decision(action, qty, reason + полная диагностика: каналы, ATR, ГО, M/E,
qty_risk/margin/liquidity/raw)`. Плохие данные ⇒ `DataError` /
`MarginRateError` — модуль «не угадывает».

## 2. Архитектура runtime

Пайплайн одного цикла (docstring и код [`src/c5_runtime.py`](src/c5_runtime.py)):

```
Market Data (1d + closed 5m)          src/market_data.py
  ↓
Time Gate: ENTRY строго после 16:00 MSK control point (control_point_passed);
           EXIT — на каждой новой закрытой 5m-свече в сессии 10:00–23:40 MSK
  ↓
C5 Core (signal + sizing — никогда не реимплементируется здесь)
  ↓
Margin Provider (live ГО get_future FULL, без fallback)   src/margin_provider.py
  ↓                      long → dlongClient, short → dshortClient;
                         вне диапазона (0, 0.5) ⇒ MARGIN_API_UNAVAILABLE
Execution Guard (стак: bid/ask/spread/depth, slippage ≤ 0.10%)  src/execution_guard.py
  ↓
Protection (daily loss 10% / DD freeze 20% / kill switch 30%)    src/protection.py
  ↓              защиты блокируют ТОЛЬКО вход, никогда выход
Cash Manager (TMON-ликвидация под нужную маржу; строго 1 облигация
              за операцию, cap MAX_TMON_SELLS=20)                 src/cash_manager.py
  ↓
Execution Adapter (PAPER/LIVE различаются ТОЛЬКО здесь)     src/execution_adapter.py
  ↓
State Store (идемпотентность, восстановление)                src/state_store.py
```

Ключевые компоненты:

- **`C5Runtime.bootstrap(state)`** — резолвит инструмент (реальный API в обоих
  режимах — read-only данные), адаптер, счёт, делает reconcile и refresh ГО.
  Классификация сбоев G-1: `ReconcileError` ⇒ `reconcile_status=FAILED`,
  исполнение **запрещено**; `MarginApiUnavailable` после успешного reconcile ⇒
  **EXIT/HOLD разрешены** (позиция восстановлена из снапшота брокера, ГО для
  них не нужен), **ENTRY fail-closed**.
- **`reconcile(adapter, account_id, state)`** — контракт сверки: брокер против
  сохранённого состояния. `broker_qty != stored_qty ⇒ позиция перенимается у
  брокера`; направление выводится из знака qty (отрицательная ⇒ SHORT).
  Открытые заявки при нулевой позиции ⇒ `ReconcileError` (ручной разбор).
  Недействительный equity ⇒ ошибка, состояние не мутируется.
- **`run_cycle()`** — один производственный цикл: safety gate → bootstrap →
  market data → exit-path → entry-path → durable state save. Точка входа
  планировщика — `run_once(token)` / `python -m src.main`.
- **Идемпотентность**: каждая обработанная свеча фиксируется по timestamp
  (`last_processed_5m_ts`, хэш свечи в `state_store.check_candle_already_processed`);
  control point «сжигается» даже при NO_SIGNAL; повторный вход по той же
  свече/точке невозможен, включая рестарт процесса (тесты J/N в
  `test_stage5b_paper_runtime.py`).
- **State** ([`src/state_store.py`](src/state_store.py)) — JSON на диске,
  атомарная запись (`_atomic_write_state`), `save_state_strict` бросает
  `StateSaveError` при сбое (G-7: до dispatch SELL строгий save обязателен —
  при его неудаче ордер НЕ отправляется, `SELL_INTENT_PERSIST_FAILED`).

## 3. Безопасность PAPER/LIVE и обработка исполнения

### Конфигурационный gate (Stage 5A)
[`src/runtime_config.py`](src/runtime_config.py) — единственный парсер
переменных окружения (runtime больше нигде их не читает):

- `TRADING_MODE` — **обязателен**, только `PAPER` или `LIVE`; отсутствует /
  пуст / невалиден ⇒ `TradingConfigError` (fail closed). Дефолтного режима нет,
  LIVE по умолчанию невозможен.
- `AUTO_TRADING_ENABLED` — optional; любое значение кроме `true/1/yes/on` ⇒
  `False`, что блокирует **весь** поток ордеров (включая PAPER) до execution
  слоя (`enforce_trading_gate`).
- Токен: единственное имя переменной — `T_SANDAPI`; legacy названия не
  активируют runtime (тест `test_credential_legacy_names_do_not_enable_runtime`).

### Адаптеры
[`src/execution_adapter.py`](src/execution_adapter.py):

- `PaperExecutionAdapter` + `SimulatedBroker` — локальное зеркало счёта;
  физически не содержит вызовов `post_order` (тесты D/K/G/H/I в
  `test_stage5a_paper_gate.py` и `test_stage5b_paper_runtime.py`).
  SimulatedBroker fail-closed: неизвестное ГО для SHORT, отсутствующая цена
  закрытия, CLOSE_SHORT без позиции, oversell/reversal — отказ с ошибкой.
- `LiveExecutionAdapter` — рыночные ордера полным объёмом, подтверждение через
  `wait_for_order_fill` до мутации состояния; TMON — одна облигация за операцию.

### Классификация fills (G-5 exit / G-9C entry)
`_classify_exit_execution` / `_classify_entry_execution` в `c5_runtime.py`:
`FULL / PARTIAL / ZERO_FILL / UNVERIFIED / FAILED`. Принципы:

- `ok=True` само по себе **не доказывает** исполнение — позиция обновляется
  только по валидному `qty_executed` (confirmed volume only);
- отсутствующий/не-int/bool/отрицательный/больше запрошенного `qty_executed`
  ⇒ `UNVERIFIED`; `0` при `ok=True` ⇒ `ZERO_FILL` (противоречие ⇒ трактуется
  как UNVERIFIED); `ok=False` на exit ⇒ UNVERIFIED, на entry ⇒ `FAILED`
  (доказанная недоставка — безопасен ре-темпл без дубля);
- частичное исполнение оставляет остаток позиции (PARTIAL books executed, not
  requested — тесты `test_g5_partial_fill_keeps_remaining_position`,
  `test_partial_short_books_only_confirmed_qty`).

### Защита от повторной отправки (G-7 / G-9C)
- **Durable intent markers**: перед dispatch SELL/BUY в состояние строго
  пишется маркер `exit_pending_sell_intent` / entry-маркер с полем `operation`
  (`EXIT_LONG`, `OPEN_SHORT`, `CLOSE_SHORT`) и `candle_ts`; снимается только
  proven-terminal результатом или явной операторской конфирмацией.
- **EXIT_UNVERIFIED freeze**: после unverifiable sell все будущие позиционные
  операции блокированы; reconcile **никогда** не снимает freeze автоматически
  (один снапшот не доказывает terminality). Снятие — только
  `state["broker_position_confirmed"] = <int>` (не int-значение отвергается).
- Рестарт процесса не создаёт «своего» in-flight-маркера (`_sell_dispatch_in_flight`,
  per-process seq) — прочитанный с диска маркер всегда считается чужой
  незавершённой операцией и блокирует re-send
  (`test_g6_restart_after_exception_blocks_resend`,
  `test_restart_after_ambiguous_entry_blocks_resend`,
  `test_restart_blocks_repeat_buy_until_operator_resolution`).
- Доверие open-orders (PR #37, [`tests/test_open_orders_trust.py`](tests/test_open_orders_trust.py)):
  ошибка `get_orders()` в LIVE-снапшоте пробрасывается наружу, а не маскируется
  как `open_orders=0`; сбой чтения на старте блокирует исполнение
  (`STARTUP_RECONCILE_FAILED`).

## 4. Тестирование

Запуск: `python -m pytest tests/` (pytest). Сеть и реальные ордера не
используются — всё на boundary-фейках и DI-хуках `C5Runtime`
(`adapter`, `margin_tracker`, `now_fn`, `config`).

| Файл | Что проверяет |
|---|---|
| [`tests/test_c5_core.py`](tests/test_c5_core.py) | Замороженная математика C5: T1–T18 спеки (§13) — ATR vs reference, направление/клип множителя, no-lookahead, сигналы N=10/M=5, игнор входа при позиции, сайзинг/stage-cap/ME-limit, валидация ГО, детерминизм/чистота; отдельно G-1 (exit/HOLD при `margin_rate=None`, entry fail-closed). |
| [`tests/test_stage2d_runtime.py`](tests/test_stage2d_runtime.py) | Цикл runtime A–V: control point, дубли Candle/entries, exit полный объём, restart-идемпотентность, ГО-блокировки, protections (10/20/30%), TMON-секвенции, reconcile «брокер важнее state», slippage/depth guards, legacy-пути не импортируются, main делегирует c5_runtime. |
| [`tests/test_stage5a_paper_gate.py`](tests/test_stage5a_paper_gate.py) | Safety-gate: TRADING_MODE fail-closed/без LIVE-дефолта, AUTO=false блокирует до market data, paper-ветка не конструирует live-клиент, PAPER lifecycle целиком, LIVE маршрутизируется через fake SDK. |
| [`tests/test_stage5b_paper_runtime.py`](tests/test_stage5b_paper_runtime.py) | E2E PAPER A–N + G-1…G-7: идемпотентность, restart+reconcile для LONG (Test N), G-5 классификатор/freeze/residual-survives-restart, G-6 exception-safety, G-7 durability strict-save. |
| [`tests/test_safety_guards_paper_runtime.py`](tests/test_safety_guards_paper_runtime.py) | Guards входа: стакан недоступен/пуст/невалиден, qty=0, TMON-неподтверждение, broker error на BUY. Два кейса помечены `xfail(strict=True)` — см. ограничения ниже. |
| [`tests/test_open_orders_trust.py`](tests/test_open_orders_trust.py) | Надёжность open-orders в LIVE-снапшоте: пустой список = подтверждённый 0, ошибка НЕ маскируется (в т.ч. rate-limit `RequestError`), nonzero open orders блокируют reconcile. |
| [`tests/test_stage9c_short_execution.py`](tests/test_stage9c_short_execution.py) | G-9C.1 SHORT entry: SELL с OPEN_SHORT intent, отрицательная позиция + резервирование ГО только по confirmed qty, partial/unproven/ambiguous-кейсы, маркер через рестарт, запрет flip/stack. |
| [`tests/test_stage9c_short_exit.py`](tests/test_stage9c_short_exit.py) | G-9C.2 SHORT exit: BUY CLOSE_SHORT, release ГО + реализация PnL (в т.ч. убыток), oversize-close/refusal без позиции, unproven close ≠ flat, маркер через рестарт, full-cycle test с partial exit и reconcile. |
| [`tests/test_paper_smoke_clock_override.py`](tests/test_paper_smoke_clock_override.py) | Diagnostic-раннер `scripts/paper_smoke_test.py` (см. §5 CI/workflow). |
| Legacy-тесты (`test_strategy.py`, `test_auto_trader.py`, `test_stop_orders.py`, `test_position_monitor.py`, `test_risk_manager.py`, `test_trading_schedule.py`, `test_api_retry.py`, `test_reliability.py`, `test_trade_logger.py`) | Относятся к GLDRUBF-наследию; проходят, но активный путь C5 ими не покрывается. |

Известные ограничения:

- **GAP A / GAP B** (`test_safety_5`, `test_safety_6`): ветки
  `TMON_SELL_N_NOT_CONFIRMED` и `SimulatedBrokerError` на BUY не достижимы
  через реальную PAPER-цепочку детерминированными хуками; закреплены как
  `xfail(strict=True)` до закрытия gap.
- Часть тестов требует установленного `t_tech.invest` (SDK) — в CI заглушки/
  skip там, где grpc недоступен.
- Workflow `gold_strategy.yml` жёстко `TRADING_MODE=PAPER`; LIVE из GitHub
  Actions невозможен. Ручной `workflow_dispatch` c `paper_smoke_test=true`
  выполняет полный PAPER-прогон; diag-флаги `--diag-market/--diag-margin/--diag-broker`
  работают только все вместе (partial selection = fail closed).

## 5. Состояние проекта

**Реализовано и подтверждено тестами:**

- Полная production-цепочка C5: данные → time gate → сигнал/сайзинг → live ГО →
  guards → protections → TMON → adapter → state (Stages 2D–6D);
- LONG-цикл PAPER end-to-end, включая рестарт процесса + reconcile
  (Test N, G-5 partial residual survives restart);
- SHORT entry (G-9C.1) и SHORT exit (G-9C.2) в PAPER: отрицательная позиция,
  резерв/освобождение ГО по dshortClient, confirmed-volume bookkeeping,
  durable markers через рестарт;
- Safety-gate Stage 5A (fail-closed конфиг, отсутствие LIVE-дефолта, физическая
  изоляция PAPER от `post_order`);
- Trust-модель open-orders для LIVE snapshot (PR #37).

**Ограничения / известные проблемы:**

- SHORT close-out в **LIVE** пока не включён в production-маршрут (реализован и
  проверен только на PAPER-уровне);
- `EXIT_UNVERIFIED` / ambiguous-fill требуют ручного операторского разрешения
  (`broker_position_confirmed`) — автоматического снятия freeze нет by design;
- GAP A / GAP B зафиксированы как strict xfail (§4);
- стратегия заморожена: любые изменения сигналов — только через ревизию спеки.

**Ближайшая задача:** сквозной SHORT-цикл через **рестарт процесса** —
открытие SHORT (фаза 1) → новый экземпляр `C5Runtime` поверх того же state →
bootstrap/reconcile восстанавливает отрицательную позицию → повторный вход по
той же свече заблокирован → EXIT_SHORT даёт ровно один BUY CLOSE_SHORT →
flat, ГО освобождено, PnL корректен. Существующий restart-тест (Test N) покрыт
только для LONG; отдельные G-9C-тесты проверяют entry/exit без полной
две-процессной склейки.

## 6. Запуск

```bash
pip install -r requirements.txt        # включает t_techinvest SDK (whl с GitHub)
export T_SANDAPI="<sandbox token>"     # единственный принятый токен env name
export TRADING_MODE=PAPER              # обязательно; дефолта нет, LIVE — только осознанно
export AUTO_TRADING_ENABLED=true       # без него весь поток ордеров заблокирован

python -m src.main                     # один цикл (для scheduler/cron)
python scripts/paper_smoke_test.py --clock-epoch "2026-10-02T16:01:00+03:00"  # diagnostic PAPER-run
python -m pytest tests/ -q             # тесты (без сети, без реальных ордеров)
```

Секреты (`T_SANDAPI`, `.env*`) не коммитятся (`.gitignore`); токен sandbox
T-Invest выдаётся в кабинете клиента. Реальные торговые заявки возможны только
при явном `TRADING_MODE=LIVE` + `AUTO_TRADING_ENABLED=true` и вне этого README
не демонстрируются.

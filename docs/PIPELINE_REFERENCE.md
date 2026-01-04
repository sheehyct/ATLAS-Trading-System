# ATLAS Trading System - Complete Pipeline Reference

**Last Updated:** January 3, 2026 (Session EQUITY-41)
**Purpose:** Complete documentation of data flow from fetching to execution
**For:** Debugging, agent reference, code navigation

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Equity Options Pipeline](#equity-options-pipeline)
3. [Crypto Pipeline](#crypto-pipeline)
4. [Shared Components](#shared-components)
5. [File Quick Reference](#file-quick-reference)
6. [Debug Checkpoints](#debug-checkpoints)
7. [Common Bug Patterns](#common-bug-patterns)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ATLAS TRADING SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐           ┌─────────────────────┐                 │
│  │   EQUITY OPTIONS    │           │       CRYPTO        │                 │
│  │      PIPELINE       │           │      PIPELINE       │                 │
│  └─────────────────────┘           └─────────────────────┘                 │
│           │                                  │                              │
│           ▼                                  ▼                              │
│  ┌─────────────────────┐           ┌─────────────────────┐                 │
│  │ Alpaca/Tiingo Data  │           │  Coinbase API Data  │                 │
│  └─────────────────────┘           └─────────────────────┘                 │
│           │                                  │                              │
│           ▼                                  ▼                              │
│  ┌─────────────────────┐           ┌─────────────────────┐                 │
│  │   Bar Classifier    │◄──────────│   Bar Classifier    │                 │
│  │  (SHARED: Numba)    │           │    (Same Logic)     │                 │
│  └─────────────────────┘           └─────────────────────┘                 │
│           │                                  │                              │
│           ▼                                  ▼                              │
│  ┌─────────────────────┐           ┌─────────────────────┐                 │
│  │  Pattern Detector   │◄──────────│  Pattern Detector   │                 │
│  │  (SHARED: Numba)    │           │    (Same Logic)     │                 │
│  └─────────────────────┘           └─────────────────────┘                 │
│           │                                  │                              │
│           ▼                                  ▼                              │
│  ┌─────────────────────┐           ┌─────────────────────┐                 │
│  │   Signal Scanner    │           │   Signal Scanner    │                 │
│  │  + TFC Calculation  │           │  + TFC Calculation  │                 │
│  └─────────────────────┘           └─────────────────────┘                 │
│           │                                  │                              │
│           ▼                                  ▼                              │
│  ┌─────────────────────┐           ┌─────────────────────┐                 │
│  │    Signal Store     │           │    Entry Monitor    │                 │
│  │  (Deduplication)    │           │  (Trigger Watch)    │                 │
│  └─────────────────────┘           └─────────────────────┘                 │
│           │                                  │                              │
│           ▼                                  ▼                              │
│  ┌─────────────────────┐           ┌─────────────────────┐                 │
│  │     Daemon          │           │      Daemon         │                 │
│  │  (Orchestrator)     │           │   (Orchestrator)    │                 │
│  └─────────────────────┘           └─────────────────────┘                 │
│           │                                  │                              │
│           ▼                                  ▼                              │
│  ┌─────────────────────┐           ┌─────────────────────┐                 │
│  │  Signal Executor    │           │   Paper Trader      │                 │
│  │  (Alpaca Options)   │           │   (Simulation)      │                 │
│  └─────────────────────┘           └─────────────────────┘                 │
│           │                                  │                              │
│           ▼                                  ▼                              │
│  ┌─────────────────────┐           ┌─────────────────────┐                 │
│  │ Position Monitor    │           │  Position Monitor   │                 │
│  │ (Target/Stop/Exit)  │           │  (Target/Stop/Exit) │                 │
│  └─────────────────────┘           └─────────────────────┘                 │
│           │                                  │                              │
│           ▼                                  ▼                              │
│  ┌─────────────────────┐           ┌─────────────────────┐                 │
│  │  Discord Alerter    │           │  Discord Alerter    │                 │
│  └─────────────────────┘           └─────────────────────┘                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Equity Options Pipeline

### Stage 1: Data Fetching

**File:** `integrations/tiingo_data_fetcher.py`
**Alternative:** `integrations/alpaca_trading_client.py`

```
PURPOSE: Fetch OHLCV data for symbols
INPUT: Symbol, timeframe, lookback period
OUTPUT: pandas DataFrame with OHLCV columns

FLOW:
1. TiingoDataFetcher.fetch_ohlcv(symbol, timeframe, lookback_bars)
2. Returns DataFrame: Date, Open, High, Low, Close, Volume
3. Data is timezone-aware (ET for equities)
```

**Debug Checkpoint:**
- Check if DataFrame is empty
- Verify timezone handling
- Check for gaps in data (holidays, weekends)

---

### Stage 2: Bar Classification

**File:** `strat/bar_classifier.py`
**Function:** `classify_bars_nb(high, low)` (Numba-optimized)

```
PURPOSE: Classify each bar as 1, 2U, 2D, or 3
INPUT: High array, Low array (numpy float64)
OUTPUT: Classifications array (int)

CLASSIFICATION LOGIC:
- First bar: -999 (reference)
- Inside (1): high <= prev_high AND low >= prev_low
- 2U (+2): high > prev_high AND low >= prev_low
- 2D (-2): low < prev_low AND high <= prev_high
- Outside (3): high > prev_high AND low < prev_low
```

**Debug Checkpoint:**
- Verify classifications match visual chart
- Check for -999 at index 0
- Verify no NaN values in input arrays

---

### Stage 3: Pattern Detection

**File:** `strat/unified_pattern_detector.py` (SINGLE SOURCE OF TRUTH)
**Alternative Files:**
- `strat/pattern_detector.py` (Numba core functions)
- `strat/tier1_detector.py` (Legacy, deprecated)

```
PURPOSE: Detect STRAT patterns from classified bars
INPUT: OHLCV DataFrame, PatternDetectionConfig, timeframe
OUTPUT: List of pattern dicts with entry/stop/target

DETECTED PATTERNS:
- 3-1-2 (Outside-Inside-Directional): Reversal
- 2-1-2 (Directional-Inside-Directional): Continuation/Reversal
- 2-2 (Directional-Directional): Reversal only (2D-2U, 2U-2D)
- 3-2 (Outside-Directional): 1.5x measured move
- 3-2-2 (Outside-Directional-Directional): Reversal

KEY FUNCTION: detect_all_patterns(data, config, timeframe)
- Returns patterns sorted CHRONOLOGICALLY (critical fix from EQUITY-38)
- Applies timeframe-specific target adjustment (EQUITY-39)

TARGET METHODOLOGY (apply_timeframe_adjustment):
| Pattern | Timeframe | Target |
|---------|-----------|--------|
| ALL | 1H | 1.0x R:R |
| 3-2 | ALL | 1.5x R:R (fallback) |
| Others | 1D/1W/1M | Structural (bar extreme) |
```

**Debug Checkpoint:**
- Check pattern is in CHRONOLOGICAL order
- Verify target geometry (bullish target > entry)
- Check full bar sequence naming (2D-2U not "2-2 Up")

---

### Stage 4: Signal Scanning

**File:** `strat/paper_signal_scanner.py`
**Class:** `PaperSignalScanner`

```
PURPOSE: Scan symbols for patterns, create signals with context
INPUT: Symbols list, timeframes list
OUTPUT: List of DetectedSignal objects

KEY METHODS:
- scan_symbol_timeframe(): Main scanning loop
- _fetch_data(): Get OHLCV from Tiingo/Alpaca
- _detect_setups(): Find SETUP signals (awaiting trigger)
- _detect_patterns(): Find COMPLETED signals (already triggered)
- get_tfc_score(): Calculate timeframe continuity (0-4)
- get_market_context(): Get VIX, ATR, volume, regime

SIGNAL TYPES:
- COMPLETED: Entry bar has closed (execute immediately)
- SETUP: Awaiting price to break trigger level

FLOW:
1. Fetch 15-minute base data
2. Resample to target timeframes (1H, 1D, 1W, 1M)
3. Classify bars on each timeframe
4. Detect patterns (completed) and setups (pending)
5. Calculate TFC score for each signal
6. Add market context (VIX, ATR, regime)
7. Return DetectedSignal list
```

**Debug Checkpoint:**
- Check `signal_type` is correct (SETUP vs COMPLETED)
- Verify `detected_time` uses scan time (not bar timestamp)
- Check TFC score calculation (0/4 for all = possible bug)

---

### Stage 5: Signal Store (Deduplication)

**File:** `strat/signal_automation/signal_store.py`
**Class:** `SignalStore`

```
PURPOSE: Store signals with deduplication, track lifecycle
INPUT: DetectedSignal from scanner
OUTPUT: StoredSignal with unique key and status

SIGNAL KEY FORMAT: {symbol}_{timeframe}_{pattern}_{direction}_{timestamp}

SIGNAL LIFECYCLE:
- DETECTED: Initial state
- ALERTED: Discord alert sent
- TRIGGERED: Price broke entry level, executing
- HISTORICAL_TRIGGERED: COMPLETED signal marked
- EXPIRED: Exceeded lookback window

KEY METHODS:
- add_signal(): Add with deduplication
- mark_alerted(): Update status after Discord
- mark_triggered(): Update status after execution
- get_signals_by_status(): Query by lifecycle state

LOOKBACK WINDOW: 3 bars (prevents re-alerting same pattern)
```

**Debug Checkpoint:**
- Check for duplicate signals with same key
- Verify status transitions are correct
- Check if signals are expiring too fast

---

### Stage 6: Entry Monitor (SETUP Signals)

**File:** `strat/signal_automation/entry_monitor.py`
**Class:** `EntryMonitor`

```
PURPOSE: Watch SETUP signals for trigger break
INPUT: SETUP signals from scanner
OUTPUT: Callback when trigger price is hit

POLL INTERVAL: 15 seconds (Session EQUITY-29)

TRIGGER LOGIC:
- CALL/LONG: Current price > setup_bar_high
- PUT/SHORT: Current price < setup_bar_low

BIDIRECTIONAL HANDLING:
- 3-? and 2-? setups can trigger in either direction
- Entry monitor detects WHICH bound breaks first
- Callback fires with actual direction

KEY METHODS:
- add_signal(): Add SETUP to watch list
- poll_prices(): Check current prices vs triggers
- _check_trigger(): Determine if trigger hit
- remove_expired(): Clean up old signals
```

**Debug Checkpoint:**
- Check poll interval is 15s (not 60s)
- Verify bidirectional triggers work correctly
- Check if signals are being removed before trigger

---

### Stage 7: Daemon (Orchestrator)

**File:** `strat/signal_automation/daemon.py`
**Class:** `SignalDaemon`

```
PURPOSE: Orchestrate all components, run scan loop
INPUT: Configuration, components
OUTPUT: Executed trades, alerts

SCAN INTERVAL: 15 minutes (resampling architecture)

MAIN LOOP:
1. Check market hours (NYSE calendar)
2. Run scanner (base or per-timeframe)
3. Store new signals (deduplication)
4. Execute COMPLETED signals immediately
5. Add SETUP signals to entry monitor
6. Check position monitor for exits
7. Send Discord alerts

KEY METHODS:
- run_base_scan(): 15-min resampling scan
- run_scan(): Per-timeframe scan
- _execute_triggered_pattern(): Execute COMPLETED signals
- _execute_signals(): DEPRECATED for COMPLETED (EQUITY-41)
- _send_alerts(): Discord + logging alerts
- _is_market_hours(): NYSE calendar check
- _is_intraday_entry_allowed(): "Let market breathe" filter

TIMING FILTERS (1H patterns):
- 2-bar patterns: Not before 10:30 AM ET
- 3-bar patterns: Not before 11:30 AM ET

ALERT CONFIG:
- alert_on_signal_detection: False (noisy)
- alert_on_trigger: False
- alert_on_trade_entry: True
- alert_on_trade_exit: True
```

**Debug Checkpoint:**
- Check COMPLETED signals execute via `_execute_triggered_pattern`
- Verify COMPLETED skip in `_execute_signals` (EQUITY-41 fix)
- Check market hours detection (holidays, early close)

---

### Stage 8: Signal Executor

**File:** `strat/signal_automation/executor.py`
**Class:** `SignalExecutor`

```
PURPOSE: Execute options trades via Alpaca
INPUT: StoredSignal
OUTPUT: ExecutionResult with OSI symbol

OPTIONS SELECTION:
- Target delta: 0.50 (configurable)
- Delta range: 0.45-0.65
- Expiration: 7-21 DTE preference
- Strike selection: Closest to target delta

KEY METHODS:
- execute_signal(): Main execution flow
- _select_option_contract(): Find best option
- _calculate_position_size(): Risk-based sizing
- _submit_order(): Alpaca API order

EXECUTION STATES:
- ORDER_SUBMITTED: Success
- SKIPPED: Filtered out
- FAILED: Error occurred

PAPER MODE: Default True (safe)
```

**Debug Checkpoint:**
- Check delta is in valid range
- Verify position size calculation
- Check if paper mode is enabled

---

### Stage 9: Position Monitor

**File:** `strat/signal_automation/position_monitor.py`
**Class:** `PositionMonitor`

```
PURPOSE: Monitor open positions for exit conditions
INPUT: Open positions from Alpaca
OUTPUT: ExitSignal when condition met

CHECK INTERVAL: 60 seconds

EXIT CONDITIONS (PRIORITY ORDER - EQUITY-41):
1. Minimum hold time (5 minutes)
2. EOD exit for 1H trades (15:55 ET)
3. DTE exit (≤3 days)
4. Stop hit (underlying price)
5. Max loss (timeframe-specific %)
6. TARGET HIT (moved before trailing stop)
7. Trailing stop (only if option P/L >= 0%)
8. Partial exit (multi-contract)
9. Max profit (option premium)

TIMEFRAME-SPECIFIC MAX LOSS (EQUITY-41):
| Timeframe | Max Loss |
|-----------|----------|
| 1M | 75% |
| 1W | 65% |
| 1D | 50% |
| 1H | 40% |

KEY METHODS:
- check_positions(): Main monitoring loop
- _check_position(): Single position checks
- _check_target_hit(): Underlying vs target
- _check_stop_hit(): Underlying vs stop
- _check_trailing_stop(): Trail with min profit check
```

**Debug Checkpoint:**
- Check exit order (target BEFORE trailing stop)
- Verify timeframe-specific max loss thresholds
- Check trailing stop min profit requirement

---

### Stage 10: Discord Alerter

**File:** `strat/signal_automation/alerters/discord_alerter.py`
**Class:** `DiscordAlerter`

```
PURPOSE: Send trade alerts to Discord
INPUT: Signal, execution result, exit signal
OUTPUT: Discord webhook message

ALERT TYPES:
- Pattern detection (disabled by default)
- Entry alert (on execution)
- Exit alert (on position close)
- Scan summary
- Daemon status

MESSAGE FORMAT (Entry):
**Entry: SPY 3-2U 1D Call**
@ $500.00 | Target: $510.00 | Stop: $495.00
Mag: 2.00% | TFC: 3/4

RATE LIMITING: 25 requests per 60 seconds
THROTTLING: 60 seconds between same signal alerts
```

**Debug Checkpoint:**
- Check webhook URL is configured
- Verify alert type flags in config
- Check for duplicate alerts (EQUITY-41 fix)

---

## Crypto Pipeline

### Stage 1: Data Fetching

**File:** `crypto/exchange/coinbase_client.py`
**Class:** `CoinbaseClient`

```
PURPOSE: Fetch OHLCV data from Coinbase
INPUT: Symbol (e.g., BIP-20DEC30-CDE), timeframe, lookback
OUTPUT: pandas DataFrame with OHLCV

MAINTENANCE WINDOW:
- Friday 5-6 PM ET (22:00-23:00 UTC)
- Bars overlapping window marked with has_maintenance_gap
```

---

### Stage 2-3: Bar Classification & Pattern Detection

**Same as Equity** - Uses `strat/bar_classifier.py` and pattern detection from `strat/pattern_detector.py`

---

### Stage 4: Signal Scanning

**File:** `crypto/scanning/signal_scanner.py`
**Class:** `CryptoSignalScanner`

```
PURPOSE: Scan crypto symbols for STRAT patterns
INPUT: Symbols list, timeframes (1w, 1d, 4h, 1h)
OUTPUT: List of CryptoDetectedSignal

DIFFERENCES FROM EQUITY:
- 24/7 trading (no market hours)
- Maintenance window handling (Friday 5-6 PM ET)
- Leverage tiers (intraday vs swing)
- 4 timeframes instead of 5

TFC SCORE: 0-4 (4 timeframes checked)

KEY METHODS:
- scan_symbol_timeframe(): Main scanning
- get_tfc_score(): 4-timeframe continuity
- _detect_setups(): SETUP patterns
- _detect_patterns(): COMPLETED patterns
```

---

### Stage 5: State Management

**File:** `crypto/data/state.py`
**Class:** `CryptoSystemState`

```
PURPOSE: Track system state across scans
INPUT: Bar classifications, patterns, positions
OUTPUT: State queries (continuity, vetoes, triggers)

STATE TRACKED:
- bar_classifications: Dict[timeframe, int]
- current_bars: Dict[timeframe, DataFrame]
- active_patterns: List[Dict]
- account_equity, available_margin, current_position

KEY METHODS:
- get_continuity_score(): TFC calculation
- check_vetoes(): Weekly/Daily inside bar blocks
- check_signal_triggers(): Price vs trigger levels
```

---

### Stage 6: Entry Monitor

**File:** `crypto/scanning/entry_monitor.py`
**Class:** `CryptoEntryMonitor`

```
PURPOSE: Watch SETUP signals for trigger break
POLL INTERVAL: 60 seconds

Same concept as equity entry_monitor
```

---

### Stage 7: Daemon

**File:** `crypto/scanning/daemon.py`
**Class:** `CryptoSignalDaemon`

```
PURPOSE: Orchestrate crypto trading
SCAN INTERVAL: 15 minutes

24/7 OPERATION:
- No market hours check (crypto trades always)
- Maintenance window pause (Friday 5-6 PM ET)

KEY METHODS:
- run_scan(): Main scan loop
- _execute_triggered_pattern(): Execute COMPLETED
- _on_entry_triggered(): Callback from entry monitor
```

---

### Stage 8: Paper Trader

**File:** `crypto/simulation/paper_trader.py`
**Class:** `PaperTrader`

```
PURPOSE: Simulate crypto trade execution
INPUT: Signal, direction, quantity
OUTPUT: SimulatedTrade

FEATURES:
- Account balance tracking
- Margin reservation (leverage)
- P/L calculation
- JSON persistence
```

---

### Stage 9: Position Monitor

**File:** `crypto/simulation/position_monitor.py`
**Class:** `CryptoPositionMonitor`

```
PURPOSE: Monitor crypto positions for exit
INPUT: Open trades from PaperTrader
OUTPUT: ExitSignal when condition met

EXIT CONDITIONS:
- Target hit
- Stop hit
- Manual close
```

---

### Stage 10: Discord Alerter

**File:** `crypto/alerters/discord_alerter.py`
**Class:** `CryptoDiscordAlerter`

```
PURPOSE: Send crypto alerts to Discord

MESSAGE FORMAT:
**ENTRY: BTC LONG**
Pattern: 3-1-2U (1h) | TFC: 3/4
@ $67,000.00
Target: $68,900.00 | Stop: $65,100.00
```

---

## Shared Components

### Bar Classifier
**File:** `strat/bar_classifier.py`
- Used by both equity and crypto
- Numba-optimized for performance
- Returns: 1 (inside), 2 (2U), -2 (2D), 3 (outside)

### Pattern Detector (Numba Core)
**File:** `strat/pattern_detector.py`
- `detect_312_patterns_nb()`: 3-1-2 detection
- `detect_212_patterns_nb()`: 2-1-2 detection
- `detect_22_patterns_nb()`: 2-2 detection
- `detect_32_patterns_nb()`: 3-2 detection
- `detect_322_patterns_nb()`: 3-2-2 detection

### Timeframe Continuity
**File:** `strat/timeframe_continuity.py`
- `TimeframeContinuityChecker` class
- Flexible continuity rules by detection timeframe
- Used by equity; crypto has inline implementation

### Configuration
**Equity:** `strat/signal_automation/config.py`
**Crypto:** `crypto/config.py`

---

## File Quick Reference

### Equity Options Files

| Stage | File | Purpose |
|-------|------|---------|
| 1. Data | `integrations/tiingo_data_fetcher.py` | OHLCV data |
| 2. Classify | `strat/bar_classifier.py` | Bar classification |
| 3. Detect | `strat/unified_pattern_detector.py` | Pattern detection |
| 4. Scan | `strat/paper_signal_scanner.py` | Signal scanning |
| 5. Store | `strat/signal_automation/signal_store.py` | Deduplication |
| 6. Entry | `strat/signal_automation/entry_monitor.py` | Trigger watch |
| 7. Daemon | `strat/signal_automation/daemon.py` | Orchestrator |
| 8. Execute | `strat/signal_automation/executor.py` | Alpaca orders |
| 9. Monitor | `strat/signal_automation/position_monitor.py` | Exit conditions |
| 10. Alert | `strat/signal_automation/alerters/discord_alerter.py` | Discord |

### Crypto Files

| Stage | File | Purpose |
|-------|------|---------|
| 1. Data | `crypto/exchange/coinbase_client.py` | OHLCV data |
| 2. Classify | `strat/bar_classifier.py` | Bar classification |
| 3. Detect | `strat/pattern_detector.py` | Pattern detection |
| 4. Scan | `crypto/scanning/signal_scanner.py` | Signal scanning |
| 5. State | `crypto/data/state.py` | State management |
| 6. Entry | `crypto/scanning/entry_monitor.py` | Trigger watch |
| 7. Daemon | `crypto/scanning/daemon.py` | Orchestrator |
| 8. Trade | `crypto/simulation/paper_trader.py` | Paper trading |
| 9. Monitor | `crypto/simulation/position_monitor.py` | Exit conditions |
| 10. Alert | `crypto/alerters/discord_alerter.py` | Discord |

### Backtest Files

| File | Purpose |
|------|---------|
| `strat/unified_pattern_detector.py` | Single source of truth |
| `scripts/backtest_strat_options_unified.py` | Main backtest script |
| `strategies/strat_options_strategy.py` | Strategy class |
| `validation/validation_runner.py` | Validation framework |

---

## Debug Checkpoints

### When Signals Not Detected

1. **Check data fetching:** Is DataFrame empty?
2. **Check bar classification:** Are bars being classified correctly?
3. **Check pattern detection:** Is `detect_all_patterns()` returning patterns?
4. **Check TFC calculation:** Is TFC always 0/4? (possible bug)
5. **Check signal store:** Is deduplication blocking signals?

### When Signals Not Executing

1. **Check signal type:** Is it SETUP or COMPLETED?
2. **Check daemon flow:** Is COMPLETED going to `_execute_triggered_pattern()`?
3. **Check timing filter:** Is "let market breathe" blocking?
4. **Check market hours:** Is it outside NYSE hours?
5. **Check executor:** Is paper mode enabled?

### When Exits Not Happening

1. **Check exit order:** Is target checked BEFORE trailing stop?
2. **Check underlying price:** Is position monitor getting price?
3. **Check timeframe:** Is max loss threshold correct for timeframe?
4. **Check trailing stop:** Is option P/L above min threshold?

### When Duplicate Alerts

1. **Check daemon:** Is COMPLETED skip in `_execute_signals()`? (EQUITY-41)
2. **Check signal store:** Is deduplication working?
3. **Check alerter:** Is throttling enabled?

---

## Common Bug Patterns

### Bug: Duplicate Entry Alerts (EQUITY-41)
**Symptom:** Same entry alert sent twice
**Root Cause:** `_execute_signals()` executed COMPLETED signals after `_execute_triggered_pattern()`
**Fix:** Skip COMPLETED signals in `_execute_signals()`
**File:** `strat/signal_automation/daemon.py`

### Bug: Target Not Detected (EQUITY-41)
**Symptom:** Trade hits target but exits with TRAIL reason
**Root Cause:** Trailing stop checked BEFORE target hit
**Fix:** Reorder exit checks (target before trailing stop)
**File:** `strat/signal_automation/position_monitor.py`

### Bug: MAX_LOSS on Valid Pattern (EQUITY-41)
**Symptom:** Monthly trade exits at MAX_LOSS when underlying hasn't hit stop
**Root Cause:** Single 50% threshold too aggressive for monthly options
**Fix:** Timeframe-specific thresholds (75% for monthly)
**File:** `strat/signal_automation/position_monitor.py`

### Bug: Trail Exit at Loss (EQUITY-41)
**Symptom:** "TRAIL | P/L: -$X" alerts
**Root Cause:** Trailing stop tracks underlying, but option lost value to theta
**Fix:** Add minimum option profit requirement for trail exit
**File:** `strat/signal_automation/position_monitor.py`

### Bug: Invalid 2-2 Setups (CRYPTO-MONITOR-3)
**Symptom:** Patterns like "1-2D-2D" detected
**Root Cause:** 2-2 setup detection didn't validate reference bar is directional
**Fix:** Skip when `abs(prev_bar_class) != 2`
**Files:** `crypto/scanning/signal_scanner.py`, `strat/paper_signal_scanner.py`

### Bug: Stale 3-? Setups (CRYPTO-MONITOR-2)
**Symptom:** Trades executing 7+ days late
**Root Cause:** 3-? setups never invalidated when range broken
**Fix:** Invalidate when `bar_high > setup_high or bar_low < setup_low`
**Files:** `crypto/scanning/signal_scanner.py`, `strat/paper_signal_scanner.py`

### Bug: Signal Expiration Too Fast (CRYPTO-MONITOR-1)
**Symptom:** Signals expire in 60 seconds
**Root Cause:** `detected_time` used bar timestamp instead of scan time
**Fix:** Use `datetime.now(timezone.utc)` for detected_time
**File:** `crypto/scanning/signal_scanner.py`

---

## Dashboard Pipeline (Visualization Layer)

### Stage 11: Dashboard

**Deployment:** Railway (cloud)
**URL:** Configured per environment
**File:** `dashboard/app.py`

```
PURPOSE: Visualize signals, positions, and performance
TECH: Plotly Dash, dash-bootstrap-components

DATA FLOW:
┌─────────────────────┐     ┌─────────────────────┐
│  VPS Signal API     │────▶│  OptionsDataLoader  │
│  (daemon REST API)  │     │  options_loader.py  │
└─────────────────────┘     └─────────────────────┘
                                      │
┌─────────────────────┐               │
│  Alpaca API         │───────────────┤
│  (live positions)   │               │
└─────────────────────┘               ▼
                            ┌─────────────────────┐
                            │   Options Panel     │
                            │  options_panel.py   │
                            └─────────────────────┘
```

### Dashboard Files

| File | Purpose |
|------|---------|
| `dashboard/app.py` | Main Dash application |
| `dashboard/config.py` | Configuration, colors, intervals |
| `dashboard/data_loaders/options_loader.py` | Fetch signals from VPS API |
| `dashboard/data_loaders/crypto_loader.py` | Fetch crypto signals |
| `dashboard/data_loaders/live_loader.py` | Alpaca live data |
| `dashboard/data_loaders/orders_loader.py` | Order history |
| `dashboard/components/options_panel.py` | Options trading UI |
| `dashboard/components/crypto_panel.py` | Crypto trading UI |
| `dashboard/components/regime_panel.py` | Regime visualization |
| `dashboard/components/risk_panel.py` | Risk metrics |

### Dashboard Data Sources

1. **VPS Signal API** (`VPS_SIGNAL_API_URL` env var)
   - Endpoint: `http://VPS_IP:8080/api/signals`
   - Returns: Active signals, pending entries, triggered signals
   - Used when: `self.use_remote = True`

2. **Local Signal Store** (fallback)
   - File: `strat/signal_automation/signal_store.py`
   - Used when: Running locally, no VPS API configured

3. **Alpaca API**
   - Live option positions
   - Account balance
   - Order history

---

## Pattern Detection Decision Tree

**File:** `strat/pattern_detector.py`

This is the CRITICAL section for debugging pattern misclassification.

### Pattern Type Determination

```
For bar index i, check classifications[i-2], classifications[i-1], classifications[i]:

                            ┌─────────────────────────────────┐
                            │  What is classifications[i-2]?  │
                            └─────────────────────────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
              abs() == 3            abs() == 2            abs() == 1
           (Outside bar)        (Directional)          (Inside bar)
                    │                     │                     │
                    ▼                     ▼                     ▼
        ┌───────────────────┐   ┌─────────────────┐   ┌─────────────────┐
        │ Check i-1 and i   │   │ Check i-1 and i │   │ Check i-1 and i │
        └───────────────────┘   └─────────────────┘   └─────────────────┘
                    │                     │                     │
     ┌──────────────┼──────────────┐      │                     │
     │              │              │      │                     │
     ▼              ▼              ▼      ▼                     ▼
  i-1=±2         i-1=±2         i-1=1   i-1=1               Not a
  i=∓2           i=same dir     i=±2    i=±2              3-bar pattern
  (reversal)     (same dir)             │
     │              │              │     │
     ▼              ▼              ▼     ▼
  3-2-2          3-2           3-1-2   2-1-2
```

### Pattern Detection Functions

| Pattern | Bars | Function | File:Line |
|---------|------|----------|-----------|
| 3-1-2 | 3 | `detect_312_patterns_nb()` | `pattern_detector.py:95` |
| 2-1-2 | 3 | `detect_212_patterns_nb()` | `pattern_detector.py:255` |
| 2-2 | 2 | `detect_22_patterns_nb()` | `pattern_detector.py:488` |
| 3-2 | 2 | `detect_32_patterns_nb()` | `pattern_detector.py:656` |
| 3-2-2 | 3 | `detect_322_patterns_nb()` | `pattern_detector.py:800` |

### 3-2 vs 3-2-2: The Critical Difference

**3-2 Pattern (2 bars):**
```
Bar i-1: Outside bar (abs(classifications[i-1]) == 3)
Bar i:   Directional bar (classifications[i] == ±2) ← TRIGGER

Example: 3-2U (bullish)
  Bar i-1: Type 3 (Outside)
  Bar i:   Type 2U (broke high) ← Entry on this bar

Detection: detect_32_patterns_nb()
  if (bar1_class == -3 or abs(bar1_class) == 3) and bar2_class == 2:
      # 3-2U pattern detected
```

**3-2-2 Pattern (3 bars):**
```
Bar i-2: Outside bar (abs(classifications[i-2]) == 3)
Bar i-1: First directional bar (classifications[i-1] == ±2)
Bar i:   OPPOSITE directional bar (classifications[i] == ∓2) ← TRIGGER

Example: 3-2D-2U (bullish reversal)
  Bar i-2: Type 3 (Outside)
  Bar i-1: Type 2D (failed breakdown)
  Bar i:   Type 2U (reversal confirmed) ← Entry on this bar

Detection: detect_322_patterns_nb()
  if abs(bar_outside) == 3 and bar1_class == -2 and bar2_class == 2:
      # 3-2D-2U pattern detected
```

**Key Distinction:**
- 3-2: Directional bar follows IMMEDIATELY after outside bar
- 3-2-2: TWO directional bars follow outside bar, second is REVERSAL

### 2-2 Pattern: Reference Bar Validation

**Valid 2-2 (REVERSAL ONLY):**
```
Bar i-1: Directional bar (abs(classifications[i-1]) == 2) ← REFERENCE
Bar i:   OPPOSITE directional bar ← TRIGGER

Example: 2D-2U (bullish reversal)
  Bar i-1: Type 2D (reference bar)
  Bar i:   Type 2U (opposite direction) ← Entry

CRITICAL: Reference bar MUST be directional (2U or 2D)
```

**Invalid 2-2 (Bug fixed in CRYPTO-MONITOR-3):**
```
Bar i-1: Inside bar (classifications[i-1] == 1) ← INVALID REFERENCE
Bar i:   Directional bar

Example: 1-2D-2D (INVALID!)
  Bar i-1: Type 1 (inside bar) ← Cannot be reference for 2-2!
  Bar i:   Type 2D

Fix in signal_scanner.py:
  if abs(prev_bar_class) != 2:
      continue  # Skip - reference bar must be directional
```

---

## Debugging Walkthrough: Pattern Misclassification

### Scenario: System detected 3-2D-2U but chart shows 3-2D

**Step 1: Get the bar data**
```python
# In daemon logs or debug, find:
# - Symbol, timeframe, detected timestamp
# - The actual OHLC values for bars i-2, i-1, i
```

**Step 2: Manually classify the bars**
```
For each bar, determine type:
- Inside (1): high <= prev_high AND low >= prev_low
- 2U (2): high > prev_high AND low >= prev_low
- 2D (-2): low < prev_low AND high <= prev_high
- Outside (3): high > prev_high AND low < prev_low
```

**Step 3: Check detection logic**
| If bars are... | Pattern should be... | Function that detected it |
|----------------|---------------------|---------------------------|
| [3, 2U] | 3-2U | `detect_32_patterns_nb()` |
| [3, -2, 2U] | 3-2D-2U | `detect_322_patterns_nb()` |
| [3, 2U, 2U] | NOT 3-2-2 (not reversal) | None (continuation) |

**Step 4: Check if wrong function was called**

In `unified_pattern_detector.py`, `detect_all_patterns()` calls ALL detectors:
```python
patterns_312 = detect_312_patterns_nb(...)
patterns_212 = detect_212_patterns_nb(...)
patterns_22 = detect_22_patterns_nb(...)
patterns_32 = detect_32_patterns_nb(...)
patterns_322 = detect_322_patterns_nb(...)
```

If 3-2D-2U detected when it should be 3-2D:
1. Check if `detect_322_patterns_nb()` has a bug
2. The outside bar at i-2 might have wrong classification
3. Check if there's actually a bar between outside and trigger

**Step 5: Check scanner logic**

In `paper_signal_scanner.py`, the scanner builds full bar sequence:
```python
# _get_full_bar_sequence() constructs the name
# Check if it's using correct indices
```

### Scenario: Target hit but exited with TRAIL

**Step 1: Check exit order in `position_monitor.py`**
```python
# Exit priority (EQUITY-41):
# 4. Target hit ← Should be BEFORE trailing stop
# 5. Trailing stop
```

**Step 2: Verify the fix is deployed**
```python
# In _check_position():
if self._check_target_hit(pos):  # Line ~595
    return ExitSignal(reason=ExitReason.TARGET_HIT, ...)

# Trailing stop check should be AFTER this
if self.config.use_trailing_stop:  # Line ~609
    trailing_signal = self._check_trailing_stop(pos)
```

**Step 3: Check underlying vs target**
```python
# _check_target_hit() compares:
# CALL: pos.underlying_price >= pos.target_price
# PUT: pos.underlying_price <= pos.target_price
```

---

## File Dependencies Map

When debugging or modifying, here's what files affect what:

### Pattern Detection Chain
```
strat/bar_classifier.py
    └── classify_bars_nb()
           │
           ▼
strat/pattern_detector.py
    └── detect_312_patterns_nb()
    └── detect_212_patterns_nb()
    └── detect_22_patterns_nb()
    └── detect_32_patterns_nb()
    └── detect_322_patterns_nb()
           │
           ▼
strat/unified_pattern_detector.py
    └── detect_all_patterns()
           │
           ▼
strat/paper_signal_scanner.py
    └── scan_symbol_timeframe()
    └── _detect_patterns()
    └── _detect_setups()
           │
           ▼
strat/signal_automation/signal_store.py
    └── add_signal()
```

### Execution Chain
```
strat/signal_automation/daemon.py
    └── run_base_scan()
    └── _execute_triggered_pattern()
           │
           ▼
strat/signal_automation/executor.py
    └── execute_signal()
    └── _select_option_contract()
           │
           ▼
integrations/alpaca_trading_client.py
    └── submit_order()
```

### Exit Chain
```
strat/signal_automation/position_monitor.py
    └── check_positions()
    └── _check_position()
           │
           ├── _check_target_hit()
           ├── _check_stop_hit()
           └── _check_trailing_stop()
           │
           ▼
strat/signal_automation/daemon.py
    └── _on_position_exit()
           │
           ▼
strat/signal_automation/alerters/discord_alerter.py
    └── send_simple_exit_alert()
```

---

## Version History

| Session | Changes |
|---------|---------|
| EQUITY-41 | Bug fixes: duplicates, target order, max loss, trail min profit. Added dashboard and pattern decision tree to pipeline docs. |
| EQUITY-40 | TFC integration planning |
| EQUITY-39 | Single source of truth for target methodology |
| EQUITY-38 | Unified pattern detector, chronological ordering |
| EQUITY-36 | EOD exit for 1H, trailing stop, 1.0x R:R for 1H |
| CRYPTO-MONITOR-3 | Invalid 2-2 fix, TRIGGERED execution |
| CRYPTO-MONITOR-2 | Stale 3-? invalidation |
| CRYPTO-MONITOR-1 | Signal expiration fix |

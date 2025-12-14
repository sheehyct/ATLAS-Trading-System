# HANDOFF - ATLAS Trading System Development

**Last Updated:** December 14, 2025 (Session CRYPTO-6)
**Current Branch:** `main`
**Phase:** Paper Trading - MONITORING + Crypto STRAT Integration
**Status:** Crypto module v0.6.0 - Dashboard integration via REST API

---

## Session CRYPTO-6: Dashboard Integration via REST API (COMPLETE)

**Date:** December 14, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - REST API + Dashboard crypto panel

### Objective

Add crypto paper trading panel to dashboard via REST API from VPS daemon.

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `crypto/api/__init__.py` | API module init | 18 |
| `crypto/api/server.py` | Flask REST API server | 232 |
| `dashboard/data_loaders/crypto_loader.py` | Dashboard data loader | 311 |
| `dashboard/components/crypto_panel.py` | Dashboard panel component | 1012 |

### Files Modified

| File | Changes |
|------|---------|
| `crypto/scanning/daemon.py` | Added api_enabled, api_host, api_port config; _start_api_server() method |
| `dashboard/data_loaders/__init__.py` | Export CryptoDataLoader |
| `dashboard/app.py` | Import crypto components; init crypto_loader; add Crypto Trading tab; add callbacks |

### Key Features

1. **REST API (Port 8080)**
   - Runs as daemon thread (single service)
   - Endpoints: /health, /status, /positions, /signals, /performance, /trades
   - Auto-starts with daemon

2. **Dashboard Crypto Panel**
   - Account summary: balance, P&L, return %
   - Daemon status: running, leverage tier, scan counts
   - Open positions with unrealized P&L
   - Pending SETUP signals tab
   - Closed trades tab with summary
   - Performance metrics tab
   - 30-second auto-refresh

3. **Architecture**
   - VPS daemon exposes API on port 8080
   - Railway dashboard calls API via CRYPTO_API_URL env var
   - Clean separation of concerns

### Deployment Steps

**VPS:**
```bash
ssh atlas@178.156.223.251
cd ~/vectorbt-workspace && git pull
sudo ufw allow 8080/tcp
sudo systemctl restart atlas-crypto-daemon
curl http://localhost:8080/health
```

**Railway:**
1. Add env var: `CRYPTO_API_URL=http://178.156.223.251:8080`
2. Push to main (auto-deploys)

### Commits

| Hash | Message |
|------|---------|
| `c391111` | feat(crypto): add REST API and dashboard crypto trading panel |

### Investigation: TradingView vs Coinbase Data

During session, investigated discrepancies between Discord alerts and TradingView charts:

- **Finding:** TradingView was missing Dec 13-14 data, causing visual mismatch
- **Root Cause:** Different data sources (TradingView vs Coinbase INTX)
- **Verification:** Bar classification is CORRECT based on Coinbase data
- **Confirmed:** All 5 timeframes (1w, 1d, 4h, 1h, 15m) are being scanned
- **Note:** Use Coinbase data for analysis since we trade on Coinbase INTX

### Session CRYPTO-7 Priorities

1. **Live Trading Mode** - Enable execution in daemon
2. **Position Exit Tracking** - Track stop/target hits in dashboard
3. **Performance Analytics** - Aggregate P&L by timeframe/pattern

### Plan Mode Recommendation

**PLAN MODE: OFF** - Execution enablement is operational work.

---

## Session CRYPTO-5: VPS Deployment and Discord Alerts (COMPLETE)

**Date:** December 14, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - VPS deployment, 60s position monitoring, Discord alerts

### Objective

Deploy crypto daemon to VPS for 24/7 operation and add Discord alerts for signals.

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `crypto/alerters/__init__.py` | Alerters module init | 10 |
| `crypto/alerters/discord_alerter.py` | CryptoDiscordAlerter class | 520 |

### Files Modified

| File | Changes |
|------|---------|
| `crypto/scanning/daemon.py` | Added discord_webhook_url config, _on_poll callback, Discord integration |
| `crypto/scanning/entry_monitor.py` | Added on_poll callback for 60s position checks |
| `scripts/run_crypto_daemon.py` | Added Discord webhook env var support |
| `deploy/atlas-crypto-daemon.service` | Fixed argument order, added cache paths |

### Key Features

1. **VPS Deployment (LIVE)**
   - Daemon running 24/7 on 178.156.223.251
   - systemd service auto-starts on boot
   - Logs at `/home/atlas/vectorbt-workspace/crypto/logs/daemon.log`

2. **60-Second Position Monitoring**
   - Moved from 5-minute health loop to entry monitor poll
   - Faster stop/target exit detection
   - via `on_poll` callback in entry monitor

3. **Discord Alerts**
   - Rich embeds with color-coded signals (green=LONG, red=SHORT)
   - Leverage tier and TFC score in alerts
   - Trigger alerts for SETUP patterns
   - Separate crypto webhook configured

### Commits

| Hash | Message |
|------|---------|
| `2321b42` | fix(crypto): correct entry_price -> entry_trigger in CLI |
| `0bdadff` | fix(crypto): resolve pandas FutureWarning |
| `75452ea` | fix(deploy): add uv cache paths to systemd |
| `576196f` | fix(deploy): correct argument order |
| `6392b4f` | feat(crypto): add 60s position monitoring via poll |
| `6dd2bf9` | feat(crypto): add Discord alerts for crypto signals |
| `6460f33` | fix(crypto): improve Discord alerter import error logging |
| `67e8981` | fix(crypto): resolve circular import in Discord alerter |
| `1a7a230` | fix(crypto): pass now_et to leverage/intraday functions |

### VPS Status

```bash
# Check daemon status
ssh atlas@178.156.223.251 "sudo systemctl status atlas-crypto-daemon"

# View logs
ssh atlas@178.156.223.251 "sudo journalctl -u atlas-crypto-daemon -f"
```

### Session CRYPTO-6 Priorities

1. **Dashboard Integration** - Add crypto paper trading panel
2. **Live Trading** - Enable execution mode (currently signals only)
3. **Performance Tracking** - Aggregate crypto P&L metrics

### Plan Mode Recommendation

**PLAN MODE: ON** - Dashboard integration requires architectural planning.

---

## Session CRYPTO-4: Intraday Leverage and VPS Deployment (COMPLETE)

**Date:** December 13, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Intraday leverage, VPS deployment, position monitoring

### Objective

Add time-based leverage tier switching and VPS deployment infrastructure.

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `scripts/run_crypto_daemon.py` | CLI entry point for VPS daemon | 370 |
| `deploy/atlas-crypto-daemon.service` | systemd service file | 50 |
| `crypto/simulation/position_monitor.py` | Stop/target exit monitoring | 250 |

### Files Modified

| File | Changes |
|------|---------|
| `crypto/config.py` | Added intraday leverage window (6PM-4PM ET), helper functions |
| `crypto/__init__.py` | Export leverage helpers, bump to v0.4.0 |
| `crypto/scanning/daemon.py` | Time-based leverage in _execute_trade, position monitoring integration |
| `crypto/simulation/paper_trader.py` | Added stop/target/timeframe/pattern to SimulatedTrade |
| `crypto/simulation/__init__.py` | Export CryptoPositionMonitor, ExitSignal |

### Key Features

1. **Time-Based Leverage Tiers**
   - Intraday: 10x available 6PM-4PM ET (22 hours/day)
   - Swing: 4x available 24/7
   - Helper functions: `is_intraday_window()`, `get_max_leverage_for_symbol()`

2. **VPS Deployment**
   - CLI script with start, scan, status, positions, performance, leverage, reset commands
   - systemd service file for production deployment
   - Log file support for daemon mode

3. **Position Monitoring**
   - Stop/target prices stored with trades
   - CryptoPositionMonitor checks exits in health loop
   - Auto-close on stop loss or take profit

### Verified Working

```python
from crypto import is_intraday_window, get_max_leverage_for_symbol

# At 10AM ET (intraday window)
# BTC-PERP-INTX: 10x leverage

# At 5PM ET (4-6PM gap)
# BTC-PERP-INTX: 4x leverage (swing only)
```

```bash
uv run python scripts/run_crypto_daemon.py leverage
# Current Tier: INTRADAY (10x)
# Time until 4PM ET close: 0.1 hours
```

### VPS Deployment Commands

```bash
# On VPS (178.156.223.251)
sudo cp deploy/atlas-crypto-daemon.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable atlas-crypto-daemon
sudo systemctl start atlas-crypto-daemon
sudo journalctl -u atlas-crypto-daemon -f
```

### Bug Fix: SETUP Detection Across Inside Bars

**Issue:** SETUP patterns (2-1, 3-1) were only detected if on the last bar. Missed setups when subsequent bars were also inside bars (e.g., 2D-1-1 structure).

**Fix:** Now checks if inside bar high/low was broken by subsequent bars. If still valid, SETUP is included.

**Commit:** `6ed6169` - fix(crypto): detect SETUP patterns that remain valid across inside bars

### Session CRYPTO-5 Priorities

1. **VPS Deployment Test** - Deploy to VPS and verify 24/7 operation
2. **Entry Monitor Enhancement** - More frequent position checks (every 60s vs 5min)
3. **Discord Alerts** - Add crypto signal alerts to Discord
4. **Performance Tracking** - Dashboard integration for crypto paper trades

### Plan Mode Recommendation

**PLAN MODE: OFF** - VPS deployment is operational work.

---

## Session CRYPTO-3: Entry Monitor and Daemon (COMPLETE)

**Date:** December 13, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Full automation stack implemented

### Objective

Build entry trigger monitoring and daemon orchestration for 24/7 crypto paper trading.

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `crypto/scanning/entry_monitor.py` | 24/7 trigger polling with maintenance window | 320 |
| `crypto/scanning/daemon.py` | Main orchestration daemon | 675 |

### Files Modified

| File | Changes |
|------|---------|
| `crypto/scanning/__init__.py` | Export entry monitor and daemon classes |
| `crypto/__init__.py` | Export new classes, bump to v0.3.0 |

### Key Features

1. **CryptoEntryMonitor** - Polls prices every 60s, checks SETUP triggers
2. **CryptoSignalDaemon** - Orchestrates scanner (15min), monitor (1min), paper trader
3. **24/7 Operation** - No market hours filter
4. **Friday Maintenance Window** - Pauses during 5-6 PM ET
5. **Paper Trading Integration** - Wired to PaperTrader for simulated execution
6. **Signal Deduplication** - Prevents duplicate signals across scans

### Verified Working

```python
from crypto import CryptoSignalDaemon

daemon = CryptoSignalDaemon()
signals = daemon.run_scan_and_monitor()
# BTC: 3-2U LONG (1w) [COMPLETED]
# ETH: 2D-1-? LONG (1d) [SETUP] - trigger: $3,136.30

daemon.start(block=False)  # Background mode works
daemon.stop()
```

### Session CRYPTO-4 Priorities

1. **Intraday Leverage** - Update config for 10x intraday (6PM-4PM ET window)
2. **Paper Balance** - Update default to $1,000
3. **VPS Deployment** - Create CLI script for production daemon
4. **Position Monitoring** - Add stop/target tracking for open trades

### Plan Mode Recommendation

**PLAN MODE: OFF** - Implementation continues from established architecture.

---

## Session CRYPTO-2: Crypto STRAT Signal Scanner (COMPLETE)

**Date:** December 13, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Core scanner implemented and verified

### Objective

Connect crypto module to Atlas STRAT engine for pattern detection on BTC/ETH perpetual futures.

### Files Created

| File | Purpose | Lines |
|------|---------|-------|
| `crypto/scanning/__init__.py` | Module initialization | 15 |
| `crypto/scanning/models.py` | CryptoDetectedSignal, CryptoSignalContext dataclasses | 90 |
| `crypto/scanning/signal_scanner.py` | CryptoSignalScanner - core pattern detection | 650 |

### Files Modified

| File | Changes |
|------|---------|
| `crypto/config.py` | Added MAINTENANCE_WINDOW config, signal filters |
| `crypto/data/state.py` | Added signal tracking methods (add_detected_signal, get_pending_setups, etc.) |
| `crypto/__init__.py` | Export scanning module, bump to v0.2.0 |

### Key Features

1. **CryptoSignalScanner** - Detects all STRAT patterns (2-2, 3-2, 3-2-2, 2-1-2, 3-1-2)
2. **24/7 Operation** - No market hours filter (crypto is 24/7)
3. **Friday Maintenance Window** - Handles 5-6 PM ET Coinbase INTX maintenance
4. **Multi-Timeframe** - Scans 1w, 1d, 4h, 1h, 15m
5. **TFC Score** - Full Timeframe Continuity calculation
6. **SETUP Detection** - Detects X-1 patterns waiting for live break

### Verified Working

```python
from crypto.scanning import CryptoSignalScanner

scanner = CryptoSignalScanner()
signals = scanner.scan_all_timeframes('BTC-PERP-INTX')
scanner.print_signals(signals)

# Found 13 signals across all timeframes
# Weekly: 3-2U LONG detected
# Daily: 3-1-2D SHORT detected
# 4h: Multiple patterns including SETUP signals
```

### Architecture

```
Coinbase OHLCV Data
       |
       v
classify_bars_nb() [from strat/bar_classifier.py - unchanged]
       |
       v
detect_*_patterns_nb() [from strat/pattern_detector.py - unchanged]
       |
       v
CryptoDetectedSignal objects
       |
       v
CryptoSystemState.add_detected_signal()
```

### Session CRYPTO-3 Priorities

1. **Entry Monitor** - Create `crypto/scanning/entry_monitor.py` for 24/7 trigger polling
2. **Daemon** - Create `crypto/scanning/daemon.py` orchestrator
3. **Paper Trading Integration** - Wire triggers to PaperTrader execution

### Plan Mode Recommendation

**PLAN MODE: OFF** - Architecture established, next session is implementation.

---

## Session 83K-82: Crypto Derivatives Module Integration (COMPLETE)

**Date:** December 12, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Core infrastructure ready for paper trading

### Objective

Integrate BTC/ETH/SOL derivatives trading capability into Atlas using Coinbase Advanced Trade API. This complements the existing equities STRAT options strategy with 24/7 crypto trading.

### Source Project

Adapted code from `C:\Cypto_Trading_Bot` (Gemini 3.0 Pro prototype) - kept Coinbase client, discarded duplicate STRAT code (Atlas has better implementation).

### Files Created

| File | Purpose |
|------|---------|
| `crypto/__init__.py` | Module initialization |
| `crypto/config.py` | Configuration (symbols, risk params, timeframes) |
| `crypto/exchange/__init__.py` | Exchange module |
| `crypto/exchange/coinbase_client.py` | Coinbase API client with public API fallback |
| `crypto/data/__init__.py` | Data module |
| `crypto/data/state.py` | System state management (bar classifications, positions) |
| `crypto/trading/__init__.py` | Trading module |
| `crypto/trading/sizing.py` | ATR-based position sizing with leverage limits |
| `crypto/simulation/__init__.py` | Simulation module |
| `crypto/simulation/paper_trader.py` | Paper trading with trade history, P&L tracking |

### Files Modified

| File | Changes |
|------|---------|
| `.env` | Added COINBASE_API_KEY, COINBASE_API_SECRET |
| `pyproject.toml` | Added coinbase-advanced-py>=1.8.2, removed alpaca-trade-api (conflict) |

### Key Features

1. **CoinbaseClient** (`crypto/exchange/coinbase_client.py`)
   - Historical OHLCV data with resampling (4h, 1w)
   - Public API fallback when auth fails (no auth needed for market data)
   - Simulation mode for paper trading (mock orders, positions)
   - Order creation (market, limit, stop)

2. **PaperTrader** (`crypto/simulation/paper_trader.py`)
   - Trade history with P&L calculation
   - FIFO matching for closed trades
   - Performance metrics (win rate, profit factor, expectancy)
   - JSON persistence for session continuity

3. **Position Sizing** (`crypto/trading/sizing.py`)
   - ATR-based sizing with leverage cap (default 8x)
   - Skip trade logic when leverage exceeds limit

4. **State Management** (`crypto/data/state.py`)
   - Multi-timeframe bar classifications
   - Continuity scoring (FTFC)
   - Veto checks (Weekly/Daily inside bars)

### Verified Working

```python
# Current prices fetching (via public API)
BTC-USD: $90,387
ETH-USD: $3,091
SOL-USD: $132

# OHLCV data for all timeframes
15m, 1h, 4h, 1d - all working

# Paper trading simulation
Trades open/close with P&L calculation working
```

### API Credentials - WORKING

New Coinbase API credentials generated and verified working:
- Authenticated API access confirmed
- 16 accounts found
- Perpetual futures products accessible

### Perpetual Futures Access - VERIFIED

| Product | Type | Status |
|---------|------|--------|
| `BTC-PERP-INTX` | Perpetual | Working ($90,177) |
| `ETH-PERP-INTX` | Perpetual | Working ($3,121) |

### Derivatives Infrastructure Added

| File | Purpose |
|------|---------|
| `crypto/config.py` | Updated with INTX symbols, leverage tiers, funding rates, margin |
| `crypto/trading/derivatives.py` | Funding cost calc, liquidation price, margin requirements |

**Leverage Tiers:**
- Intraday: 10x (close before 8h funding)
- Swing: 4x BTC/ETH, 3x SOL

**Funding:** 8h intervals (00:00, 08:00, 16:00 UTC), ~10% APR default

### Session 83K-83 Priorities

1. **Connect to Atlas STRAT** - Wire crypto data to `strat/` module for pattern detection
2. **Create Crypto Signal Scanner** - Similar to `strat/paper_signal_scanner.py`
3. **Test Paper Trading** - Validate simulation with real perp prices
4. **Dashboard Integration** - Add crypto section to Atlas dashboard

### Plan Mode Recommendation

**PLAN MODE: ON** - Next step is connecting crypto to Atlas STRAT engine for pattern detection.

---

## Session 83K-81: Dashboard P&L and Performance Tracking (COMPLETE)

**Date:** December 12, 2025
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - Ready for Railway deployment

### What Was Implemented

#### Priority 1: Closed Position P&L Tracking (COMPLETE)

**Solution:** FIFO matching algorithm using Alpaca `/v2/account/activities/FILL` endpoint

**Files Modified:**
| File | Changes |
|------|---------|
| `integrations/alpaca_trading_client.py` | Added `get_fill_activities()`, `get_closed_trades()` with FIFO matching |
| `dashboard/data_loaders/options_loader.py` | Added `get_closed_trades()`, `get_closed_trades_summary()` |
| `dashboard/components/options_panel.py` | Added 4th tab "Closed Trades", `create_closed_trades_table()` |
| `dashboard/app.py` | Updated `update_options_signals()` callback for closed trades |

**Features:**
- 4th tab "Closed Trades" in Options panel
- FIFO matching for realized P&L calculation
- Summary row: Total P&L, Win Rate, W/L count
- Table columns: Contract, Qty, Entry, Exit, Realized P&L, Duration, Closed Date
- 30-day default lookback

#### Priority 2: Strategy Performance Tab Restructure (COMPLETE)

**Files Modified:**
| File | Changes |
|------|---------|
| `dashboard/config.py` | Added 'strat_options' and 'aggregate' to AVAILABLE_STRATEGIES |
| `dashboard/components/strategy_panel.py` | Changed default to 'strat_options' |
| `dashboard/app.py` | Updated 3 callbacks to handle STRAT Options strategy |

**Strategy Dropdown Options:**
- **STRAT Options (Live)** - Default, shows closed trades performance
- **Aggregate (All Strategies)** - Combined view
- **Opening Range Breakout** - Existing backtest
- **52-Week High Momentum** - Existing backtest

**STRAT Options Displays:**
- Equity Curve: Total P&L, Win Rate, Trade counts, Avg P&L
- Rolling Metrics: Bar chart of last 10 closed trades P&L
- Trade Distribution: Pie chart of wins vs losses

### VPS Deployment

```bash
ssh atlas@178.156.223.251
cd ~/vectorbt-workspace && git pull
# Dashboard auto-deploys via Railway
```

### Session 83K-82 Priorities

1. **Monitor Dashboard** - Verify closed trades display correctly on Railway
2. **Test FIFO Matching** - Verify with actual closed trades in paper account
3. **P3: Trade Progress to Target** (DEFER) - Signal-to-position linkage

### Plan Mode Recommendation

**PLAN MODE: OFF** - Features complete, monitoring phase.

---

## Session 83K-80: HTF Scanning Architecture Fix (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE - Deployed to VPS

### Solution: 15-Minute Base Resampling

For SETUP patterns (2-1-?, 3-1-?), entry is LIVE when price breaks inside bar. Previous fixed schedules missed entries. Now uses 15-min bars as base and resamples to all higher timeframes every 15 minutes.

### Files Modified

| File | Changes |
|------|---------|
| `strat/paper_signal_scanner.py` | Added resampling methods |
| `strat/signal_automation/config.py` | Added `enable_htf_resampling` |
| `strat/signal_automation/scheduler.py` | Added `add_base_scan_job()` |
| `strat/signal_automation/daemon.py` | Added `run_base_scan()` |

### Commits

```
04d8933 feat: implement 15-min base resampling for HTF scanning fix
```

---

## Session 83K-79: Comprehensive Project Audit (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE - Documentation fixed, unused code removed

- Fixed test count (913) and Phase 5 status (deployed)
- Deleted empty stub modules
- Archived 24 exploratory scripts

---

## Session 83K-78: Dashboard Enhancement + Watchlist Expansion (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE - Dashboard redesigned, watchlist expanded to 11 symbols

---

## Session 83K-77: Critical Bug Fix - Rapid Entry/Exit Safeguards (COMPLETE)

**Date:** December 12, 2025
**Status:** COMPLETE - Four safeguards implemented

- Minimum hold time (5 minutes)
- HISTORICAL_TRIGGERED check
- Thread-safe lock
- Market hours check

---

**ARCHIVED SESSIONS:**
- Sessions 1-66: `archives/sessions/HANDOFF_SESSIONS_01-66.md`
- Sessions 83K-2 to 83K-10: `archives/sessions/HANDOFF_SESSIONS_83K-2_to_83K-10.md`
- Sessions 83K-10 to 83K-19: `archives/sessions/HANDOFF_SESSIONS_83K-10_to_83K-19.md`
- Sessions 83K-20 to 83K-39: `archives/sessions/HANDOFF_SESSIONS_83K-20_to_83K-39.md`
- Sessions 83K-40 to 83K-46: `archives/sessions/HANDOFF_SESSIONS_83K-40_to_83K-46.md`
- Sessions 83K-52 to 83K-66: `archives/sessions/HANDOFF_SESSIONS_83K-52_to_83K-66.md`

---

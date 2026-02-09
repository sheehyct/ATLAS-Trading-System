# ATLAS Trading System - Comprehensive Status Guide

**Document Purpose:** Complete status overview for Claude Desktop research project space
**Generated:** January 11, 2026
**Source:** OpenMemory, HANDOFF.md, git history, codebase analysis

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Current Deployment Status](#current-deployment-status)
4. [Implemented Strategies](#implemented-strategies)
5. [Strategies Pending Implementation](#strategies-pending-implementation)
6. [STRAT Options Module](#strat-options-module)
7. [Data Sources and Infrastructure](#data-sources-and-infrastructure)
8. [Recent Development History](#recent-development-history)
9. [Technical Debt and Known Issues](#technical-debt-and-known-issues)
10. [Next Steps and Priorities](#next-steps-and-priorities)

---

## Executive Summary

**System Name:** ATLAS (Adaptive Trading with Layered Asset System)

**Current Phase:** Paper Trading - Active monitoring with live Alpaca connections

**Key Metrics:**
- **Test Coverage:** 413 tests passing (348 STRAT + 65 signal automation)
- **Sessions Completed:** 90+ development sessions since November 2025
- **Live Deployments:** 2 (VPS daemon + Railway dashboard)

**Primary Active Components:**
1. STRAT Options Paper Trading (11-symbol watchlist, autonomous execution)
2. Unified Dashboard (6-tab analytics panel)
3. Daily Audit System (4:30 PM ET webhook reports)

---

## System Architecture Overview

### Four-Layer Architecture

```
Layer 1: ATLAS Regime Detection
    Status: DEPLOYED (validated 77% crash detection)
    Purpose: Market state classification
    States: TREND_BULL, TREND_BEAR, TREND_NEUTRAL, CRASH

Layer 2: STRAT Pattern Recognition
    Status: COMPLETE (56/56 tests passing)
    Purpose: Bar-level pattern detection
    Patterns: 3-1-2, 2-1-2, 3-2, 3-2-2, 2-2 reversal

Layer 3: Options Execution Module
    Status: LIVE (paper trading)
    Purpose: Capital-efficient execution (27x leverage)
    Integration: ThetaData v3 API, Alpaca Paper Trading

Layer 4: Credit Spread Monitoring
    Status: DEFERRED (2026+)
    Purpose: Cross-asset crash validation
    Components: IG-HY spreads, TED spread, VIX term structure
```

### Multi-Layer Integration

| Deployment Mode | Capital Required | Status |
|-----------------|------------------|--------|
| Standalone STRAT (Options) | $3,000+ | LIVE (Paper) |
| Standalone ATLAS (Equity) | $10,000+ | VALIDATED |
| Integrated ATLAS+STRAT | $20,000+ | DEFERRED |

---

## Current Deployment Status

### VPS Deployment (178.156.223.251)

**Service:** `atlas-daemon`
**Branch:** `main`
**Last Deploy:** January 10, 2026 (EQUITY-52-A/B)

**Active Components:**
- Paper signal scanner (continuous)
- Entry monitor (15-second poll interval)
- Position monitor (real-time)
- Daily audit scheduler (4:30 PM ET)

**Watchlist (11 symbols):**
- Core ETFs: SPY, QQQ, IWM, DIA
- Mega-caps: AAPL, TSLA, MSFT, GOOGL
- Retail momentum: HOOD, QBTS, ACHR

### Railway Dashboard

**URL:** Deployed via Railway
**Features:**
- 6-tab STRAT Analytics panel
- Market toggle (Equity Options / Crypto)
- Real-time Alpaca connection
- Portfolio history visualization

**Accounts Connected:**
- Alpaca LARGE: $10,330.95 equity
- Alpaca SMALL: $739.85 equity

### Account Constraints

**Schwab Level 1 Options (Cash Account):**
- CAN: Long stock, long calls/puts, cash-secured puts
- CANNOT: Short stock, naked options, spreads

---

## Implemented Strategies

### 1. 52-Week High Momentum (VALIDATED)

**File:** `strategies/high_momentum_52w.py`
**Status:** VALIDATED - Session 36
**Phase:** Phase 1 (Foundation)

**Logic:**
```python
entry_signal = distance_from_high >= 0.90  # Within 10% of 52-week high
exit_signal = distance_from_high < 0.70    # 30% off highs
rebalance = "semi-annual"                   # February, August
```

**Performance Targets:**
- Sharpe Ratio: 0.8-1.2
- CAGR: 10-15%
- Max Drawdown: -25% to -30%
- Turnover: ~50% semi-annually

**Regime Allocation:**
- TREND_BULL: 30-40%
- TREND_NEUTRAL: 20-25% (still works)
- TREND_BEAR: 0%

### 2. STRAT Options Trading (LIVE)

**Files:**
- `strat/paper_signal_scanner.py` (79,298 lines)
- `strat/signal_automation/daemon.py`
- `strat/unified_pattern_detector.py`

**Status:** LIVE PAPER TRADING

**Pattern Types Detected:**
| Pattern | Description | Direction |
|---------|-------------|-----------|
| 3-1-2U/D | Outside-Inside-Directional | Reversal |
| 2-1-2U/D | Directional-Inside-Directional | Continuation |
| 3-2U/D | Outside-Directional | Reversal |
| 3-2-2 | Outside-Directional-Directional | Extended |
| 2-2 | Directional-Directional | Reversal only |

**Entry Mechanics:**
- Entry: ON THE BREAK (not at bar close)
- Stop: Based on pattern structure
- Target: ATR-based (1.5x for 3-2) or structural

**Target Methodology (EQUITY-39):**
| Pattern | Timeframe | Target |
|---------|-----------|--------|
| ALL | 1H | 1.0x R:R |
| 3-2 | ALL | 1.5x ATR |
| Others | 1D/1W/1M | Structural |

**Recent Enhancements (EQUITY-52-B):**
- ATR-based targets for 3-2 patterns
- ATR trailing stops (0.75 ATR activation, 1.0 ATR trail)
- Daily audit at 4:30 PM ET

### 3. Opening Range Breakout (PARTIAL)

**File:** `strategies/orb.py`
**Status:** Structure complete, needs modifications

**Required Modifications:**
- Add 2x volume confirmation (MANDATORY)
- Restrict to S&P 500 only
- Add transaction cost analysis
- Reduce trading frequency

**Performance Targets:**
- Sharpe Ratio: 1.2-1.8
- Win Rate: 15-25% (asymmetric)
- Average Win: 3-5x average loss

---

## Strategies Pending Implementation

### Phase 1 (Highest Priority)

#### Quality-Momentum Combination

**File:** `strategies/quality_momentum.py`
**Status:** SKELETON (95% complete per MOMENTUM-1)
**Tests:** 36/36 passing

**Discovery (Session MOMENTUM-1):**
- Strategy more complete than documented
- AlphaVantage integration exists (374 lines with 90-day caching)
- Initial backtest: CAGR 26.33% (exceeds 15-22% target)

**Logic:**
```python
quality_score = (
    0.40 * roe_rank +           # Return on Equity
    0.30 * earnings_quality +    # Accruals ratio
    0.30 * (1 / leverage_rank)   # Low leverage preferred
)
quality_filter = quality_score >= 0.50  # Top 50% by quality
momentum_score = price.pct_change(252).shift(21)  # 12-1 momentum
```

**Performance Targets:**
- Sharpe Ratio: 1.3-1.7 (validated 1.55 in research)
- CAGR: 15-22%
- Max Drawdown: -18% to -22%

**Regime Allocation:**
- ALL REGIMES: Works (quality provides downside protection)

### Phase 2

#### Semi-Volatility Momentum

**File:** `strategies/semi_vol_momentum.py`
**Status:** SKELETON (algorithm complete, needs test suite)

**Academic Foundation:** Moreira & Muir (2017) "Volatility-Managed Portfolios"

**Logic:**
```python
realized_vol = returns.rolling(60).std() * np.sqrt(252)
target_vol = 0.15  # 15% annualized
vol_scalar = target_vol / realized_vol
vol_scalar = vol_scalar.clip(0.5, 2.0)  # 50%-200% range
```

**Performance Targets:**
- Sharpe Ratio: 1.4-1.8
- CAGR: 15-20%
- Max Drawdown: -15% to -20%

**Regime Allocation:**
- TREND_BULL + Low Vol: 15-20%
- TREND_BULL + High Vol: 5-10%
- Other: 0%

#### IBS Mean Reversion

**File:** `strategies/ibs_mean_reversion.py`
**Status:** SKELETON

**Academic Foundation:** Connors Research, 20+ years validation

**Logic:**
```python
ibs = (close - low) / (high - low)
entry_signal = (
    (ibs < 0.20) &              # Bottom 20% of range
    (close > sma_200) &          # Uptrend filter
    (volume > volume_ma * 2.0)   # 2x volume MANDATORY
)
```

**Performance Targets:**
- Sharpe Ratio: 1.5-2.0
- Win Rate: 65-75%
- Average Hold: 1-3 days
- Max Drawdown: -10% to -12%

**Regime Allocation:**
- TREND_NEUTRAL/CHOP: 15-20% (thrives in chop)
- TREND_BULL: 5-10%
- TREND_BEAR: 0%

### Phase 3+ (Deferred)

#### Credit Spread Strategy (Layer 4)

**Status:** ACCEPTED but DEFERRED to 2026+

**Research Validation (Session Nov 8, 2025):**
- User spreadsheet: 328x vs B&H 20x (1997-2025)
- Strategy is CRASH INSURANCE, not bull market optimizer
- Avoided 2000 crash: -4% vs B&H -91%
- Avoided 2008 crash: 0% vs B&H -96%

**Entry/Exit Rules:**
- ENTRY: Credit spreads fall 35% from highs -> 100% leveraged ETF
- EXIT: Credit spreads rise 40% from lows AND cross 330-day EMA -> cash

---

## STRAT Options Module

### Architecture

```
strat/
    bar_classifier.py       - Bar classification (1, 2U, 2D, 3)
    pattern_detector.py     - Pattern detection (67,680 lines)
    unified_pattern_detector.py - Single source of truth
    timeframe_continuity.py - TFC scoring
    options_module.py       - Options integration (75,156 lines)
    paper_signal_scanner.py - Signal generation
    signal_automation/
        daemon.py           - Main orchestrator
        entry_monitor.py    - Entry trigger detection
        position_monitor.py - Exit management
        executor.py         - Trade execution
        signal_store.py     - Signal persistence
        config.py           - Configuration
        alerters/
            discord_alerter.py - Discord webhooks
```

### Key Features

**Bar Classification:**
- Type 1: Inside bar (stays within prior range)
- Type 2U: Broke high only (bullish)
- Type 2D: Broke low only (bearish)
- Type 3: Broke both high AND low (outside bar)

**Timeframe Continuity (TFC):**
- Scores: 0-4 (timeframes aligned)
- Threshold: >= 4 for "WITH TFC"
- Type 3 direction: By candle color (green=bullish, red=bearish)

**Entry Timing (CRITICAL):**
- Entry happens ON THE BREAK, not at bar close
- Forming bar classified intrabar
- Bidirectional setups wait for entry_monitor trigger

**Exit Priority (STRAT Methodology):**
1. Hold time check (safety)
2. EOD exit (15:59 ET for 1H)
3. DTE exit (safety)
4. Stop hit
5. Max loss
6. **Target hit**
7. **Pattern invalidation** (Type 3 evolution)
8. Trailing stop
9. Partial exit
10. Max profit

### Recent Bug Fixes (EQUITY-43 to EQUITY-51)

| Session | Fix | Impact |
|---------|-----|--------|
| EQUITY-44 | TFC Type 3 scoring by candle color | Correct TFC alignment |
| EQUITY-44 | Type 3 pattern invalidation exit | Early exit when premise invalidated |
| EQUITY-46 | Stale setup validation | Prevent trading expired setups |
| EQUITY-48 | Real-time Type 3 evolution detection | Intrabar pattern invalidation |
| EQUITY-49 | TFC re-evaluation at entry trigger | Block trades when TFC degraded |
| EQUITY-51 | Stale 1H position detection | Exit overnight positions immediately |

---

## Data Sources and Infrastructure

### Primary Data Sources

| Source | Use Case | Notes |
|--------|----------|-------|
| Alpaca | Primary equity data | `vbt.AlpacaData.pull()` |
| Tiingo | 30+ year historical | SPY since 1993, free tier |
| ThetaData | Options data | v3 REST API, port 25503 |
| yfinance | VIX ONLY | Prohibited for equities |

### Data Requirements

- Timezone: `tz='America/New_York'` (MANDATORY)
- Weekend filtering: `data.index.dayofweek < 5`
- Holiday filtering: `pandas_market_calendars`
- Backtests: `fees=0.001, slippage=0.001` minimum

### Test Coverage

| Suite | Tests | Status |
|-------|-------|--------|
| STRAT tests | 348 | PASSING |
| Signal automation | 65 | PASSING |
| Regime detection | 31 | PASSING |
| ThetaData | 80 | PASSING |
| Total | 413+ | ALL GREEN |

---

## Recent Development History

### Session Timeline (Last 10 Sessions)

| Session | Date | Key Accomplishment |
|---------|------|-------------------|
| EQUITY-53 | Jan 10 | Tests verified, VPS deployment, daily audit |
| EQUITY-52-A | Jan 10 | Unified STRAT Analytics dashboard panel |
| EQUITY-52-B | Jan 10 | ATR-based targets + trailing stops |
| EQUITY-51 | Jan 8-9 | Stale 1H position fix, pipeline analysis |
| EQUITY-50 | Jan 8 | Trade analytics dashboard |
| EQUITY-49 | Jan 8 | TFC re-evaluation at entry |
| EQUITY-48 | Jan 8 | Type 3 evolution detection |
| EQUITY-47 | Jan 7 | TFC + filter rejection logging |
| EQUITY-46 | Jan 7 | Stale setup validation |
| EQUITY-45 | Jan 7 | EOD exit + production bug fixes |

### Major Milestones

**November 2025:**
- ATLAS Phase F validation complete
- STRAT bar classification implemented
- Pattern detection tests passing

**December 2025:**
- Live paper trading deployment
- Dashboard integration
- Options execution module complete
- 11-symbol watchlist expansion

**January 2026:**
- Unified pattern detector
- ATR-based targets
- Daily audit system
- 413 tests passing

---

## Technical Debt and Known Issues

### Resolved (EQUITY-47 to EQUITY-52)

| Item | Status | Session |
|------|--------|---------|
| TFC Logging | DONE | EQUITY-47 |
| Filter Rejection Logging | DONE | EQUITY-47 |
| Type 3 Evolution Detection | DONE | EQUITY-48 |
| Signal Lifecycle Tracing | DONE | EQUITY-48 |
| TFC Re-evaluation at Entry | DONE | EQUITY-49 |
| Trade Analytics Dashboard | DONE | EQUITY-50 |

### Pending Dashboard Work

| Item | Status |
|------|--------|
| Add TFC column to open positions | Pending |
| Test with real closed trades data | Pending |
| Style refinements to match reference | Pending |
| Visual verification of 6-tab structure | Pending |

### Crypto Pipeline Gaps

Bug fixes in equity NOT ported to crypto:

| Fix | Equity Location | Crypto Status |
|-----|-----------------|---------------|
| Stale setup validation | daemon.py:786-877 | MISSING |
| Type 3 invalidation | position_monitor.py:1030-1056 | MISSING |
| TFC re-evaluation | daemon.py:933-1056 | MISSING |
| Stale 1H position | position_monitor.py:1132-1180 | MISSING |

---

## Next Steps and Priorities

### Immediate (EQUITY-54)

1. **Monitor Daily Audit** - Verify 4:30 PM ET Discord webhook
2. **Verify ATR Calculations** - Check `atr_at_detection` in new 3-2 patterns
3. **Dashboard Polish** - TFC column, visual verification

### Short-Term

1. **Quality-Momentum Validation** - Complete VBT 5-step workflow
2. **Semi-Vol Momentum Test Suite** - ~25 tests needed
3. **Crypto Pipeline Unification** - Port equity bug fixes

### Medium-Term

1. **Live Trading Preparation** - 6-month paper trading requirement
2. **Strategy Diversification** - IBS Mean Reversion, ORB modifications
3. **Credit Spread Research** - Layer 4 preparation

### Long-Term (2026+)

1. **Credit Spread Implementation** - Crash protection layer
2. **Integrated ATLAS+STRAT Mode** - Confluence trading
3. **Capital Scaling** - $20k+ multi-layer deployment

---

## Reference Documents

| Document | Location | Purpose |
|----------|----------|---------|
| HANDOFF.md | docs/HANDOFF.md | Session-by-session progress |
| CLAUDE.md | docs/CLAUDE.md | Development rules and workflows |
| Session Startup | .session_startup_prompt.md | Next session context |
| Architecture | docs/SYSTEM_ARCHITECTURE/ | System design docs |
| Pipeline Reference | docs/PIPELINE_REFERENCE.md | Execution pipeline (1100 lines) |

### Key Skills

| Skill | Location | When to Use |
|-------|----------|-------------|
| strat-methodology | ~/.claude/skills/strat-methodology/ | STRAT pattern detection, entry/exit |
| thetadata-api | ~/.claude/skills/thetadata-api/ | ThetaData API calls, debugging |

### Mandatory Workflows

**VBT 5-Step Workflow:**
```
SEARCH  -> mcp__vectorbt-pro__search()
VERIFY  -> mcp__vectorbt-pro__resolve_refnames()
FIND    -> mcp__vectorbt-pro__find()
TEST    -> mcp__vectorbt-pro__run_code()
IMPLEMENT -> Only after steps 1-4 pass
```

**Pre-Commit:**
- Run `/code-review` on changed files
- Run `pr-review-toolkit:silent-failure-hunter` on trade execution code
- Verify tests pass locally

---

**Document Version:** 1.0
**Generated By:** Claude Code (Opus 4.5)
**Source Session:** EQUITY-54 preparation

# ATLAS System Comprehensive Status Guide
## For Claude Desktop Context

**Generated:** November 26, 2025 (Session 81)
**Purpose:** Complete system state for Claude Desktop onboarding
**Branch:** main

---

## SECTION 1: EXECUTIVE SUMMARY

ATLAS (Adaptive Trading with Layered Asset System) is a multi-layer algorithmic trading system.

### Current State at a Glance

| Layer | Component | Status | Live? |
|-------|-----------|--------|-------|
| 1A | Academic Jump Model | COMPLETE | YES |
| 1B | VIX Flash Crash Detection | COMPLETE | YES |
| 2 | STRAT Pattern Recognition | VALIDATED | NO (needs options) |
| 3 | Options Execution | IN PROGRESS | NO |
| 4 | Credit Spread Monitoring | DEFERRED | NO |

### Live Deployment

**System A1** running since November 20, 2025:
- Strategy: 52-Week High Momentum + ATLAS regime filtering
- Positions: CSCO, GOOGL, AMAT, AAPL, CRWD, AVGO (6 stocks)
- Regime: TREND_NEUTRAL (70% allocation)
- Next rebalance: February 1, 2026
- Account: Alpaca paper trading (~$10,000)

---

## SECTION 2: LAYER 1 - REGIME DETECTION (DEPLOYED)

### 2.1 Academic Jump Model

**File:** `regime/academic_jump_model.py`
**Status:** PRODUCTION - 31/31 tests passing

**Academic Foundation:**
- Shu et al. (Princeton, 2024): "Statistical Jump Models for Asset Allocation"
- 33 years empirical validation (1990-2023)
- Sharpe improvement: 20-42% over buy-and-hold
- Max drawdown reduction: ~50%

**6-Phase Implementation (ALL COMPLETE):**
- Phase A: Feature calculation (DD 10d, Sortino 20d, Sortino 60d)
- Phase B: Optimization solver (coordinate descent + dynamic programming)
- Phase C: Cross-validation (lambda selection, 8-year window)
- Phase D: Online inference (3000-day lookback, 6-month parameter updates)
- Phase E: Regime mapping (2-state to 4-regime)
- Phase F: Validation (March 2020: 100% CRASH/BEAR detection)

**Four Regimes:**
```
TREND_BULL   - Jump confidence >70%, positive direction
TREND_BEAR   - Jump confidence >70%, negative direction
TREND_NEUTRAL - Jump confidence 30-70%
CRASH        - Extreme volatility or VIX spike
```

**Regime Allocation Strategy:**
| Regime | Equity Allocation |
|--------|-------------------|
| TREND_BULL | 100% |
| TREND_NEUTRAL | 70% |
| TREND_BEAR | 30% |
| CRASH | 0% (100% cash) |

### 2.2 VIX Flash Crash Detection (Layer 1B)

**Files:**
- `regime/vix_acceleration.py` - Main module
- `regime/vix_spike_detector.py` - Real-time detection class

**Purpose:** Detect flash crashes that academic model misses due to 20-60 day smoothing.

**Key Use Case - August 5, 2024:**
- Academic model: Lagged by days (MISSED)
- VIX acceleration: Detected same day (+64.90%)

**Detection Thresholds:**
```python
INTRADAY_THRESHOLD = 0.20   # 20% from market open
ONE_DAY_THRESHOLD = 0.20    # 20% from yesterday's close
THREE_DAY_THRESHOLD = 0.50  # 50% from 3 days ago
ABSOLUTE_VIX_THRESHOLD = 35.0  # Extreme fear level
```

**Severity Classification:**
- FLASH_CRASH: VIX spike >30% (immediate CRASH regime)
- ELEVATED: VIX spike 15-30% (reduce position sizes)
- NORMAL: VIX spike <15% (use academic regime)

**Key Functions:**
```python
# Historical detection (backtesting)
detect_vix_spike(vix_close, threshold_1d=0.20, threshold_3d=0.50)

# Real-time detection (paper/live trading)
detect_realtime_vix_spike()  # Returns dict with is_crash, triggers, etc.

# Quick regime check
get_current_regime()  # Returns 'CRASH' or 'TREND_NEUTRAL'
```

**Integration Pattern:**
```python
# Academic model baseline
atlas_regime = academic_model.online_inference(spy_data)

# VIX override takes precedence
spikes, severity = get_vix_regime_override(vix_data)
atlas_regime[spikes] = 'CRASH'
```

**Data Sources:**
- Paper Trading: Yahoo Finance (yfinance)
- Live Trading: Upgrade to massive.com required

---

## SECTION 3: LAYER 2 - STRAT PATTERN RECOGNITION (VALIDATED)

### 3.1 Implementation Status

**Status:** CODE COMPLETE, ALL TESTS PASSING (56/56)
**Blocker:** Needs options module for deployment (user has $3k capital)

**Files:**
- `strat/bar_classifier.py` - Bar classification (@njit compiled)
- `strat/pattern_detector.py` - Pattern detection (3-1-2, 2-1-2, 2-2, 3-2-2)
- `strat/timeframe_continuity.py` - Multi-timeframe alignment
- `strat/tier1_detector.py` - Tier 1 pattern detection
- `strat/atlas_integration.py` - ATLAS regime integration

### 3.2 Bar Classification

**Types:**
- Type 1 (Inside): Contained within previous bar's range
- Type 2U (Up): Breaks previous high only
- Type 2D (Down): Breaks previous low only
- Type 3 (Outside): Breaks both previous high and low

**Algorithm:** Uses previous bar comparison (standard STRAT methodology)
- 100% alignment with TradingView
- @njit compiled for performance (3.3M bars/second)

### 3.3 Pattern Detection

**Patterns Implemented:**
| Pattern | Description | Target Calculation |
|---------|-------------|-------------------|
| 3-1-2 | Outside-Inside-Directional reversal | Structural level (bar extreme) |
| 2-1-2 | Directional-Inside-Directional | Structural level (bar extreme) |
| 2-2 | Consecutive directional bars | Structural level |
| 3-2-2 | Outside-Directional-Directional | Outside bar extreme |

**Critical Fix (Session 77):**
- OLD (WRONG): Measured move targets (entry + pattern_range)
- NEW (CORRECT): Structural level targets (bar extreme)
- Validated against TradingView: 100% match

**Entry/Stop/Target Rules:**
```python
# 3-1-2 Bullish Example
entry = inside_bar_high + 0.01
stop = outside_bar_low
target = outside_bar_high  # STRUCTURAL LEVEL
```

### 3.4 Timeframe Continuity

**File:** `strat/timeframe_continuity.py`
**Timeframes:** Monthly, Weekly, Daily, 4H, Hourly (5 levels)

**Output:**
- Continuity strength: 0-5 (count of aligned timeframes)
- Full continuity: Boolean (all timeframes aligned)

**Signal Quality Matrix (Integrated Mode):**
```python
if regime == 'TREND_BULL' and pattern == 'bullish' and continuity >= 0.67:
    signal_quality = 'HIGH'   # Full position
elif regime == 'TREND_NEUTRAL':
    signal_quality = 'MEDIUM' # Half position
elif regime == 'CRASH':
    signal_quality = 'REJECT' # No trade (ATLAS veto)
```

---

## SECTION 4: LAYER 3 - OPTIONS EXECUTION (IN PROGRESS)

### 4.1 Implementation Status

**Status:** ThetaData v3 API integration COMPLETE (Session 81)
**Tests:** 80/80 passing

**Files:**
- `strat/options_module.py` - Options backtester with Black-Scholes
- `strat/greeks.py` - Greeks calculator
- `strat/risk_free_rate.py` - Historical rates (2008-2024)
- `integrations/thetadata_client.py` - ThetaData REST client (v3 API)
- `integrations/thetadata_options_fetcher.py` - High-level fetcher with caching

### 4.2 ThetaData v3 API Migration (Session 81)

**API Changes (v2 to v3):**
| Component | v2 (Old) | v3 (New) |
|-----------|----------|----------|
| Port | 25510 | 25503 |
| Base URL | /v2 | /v3 |
| Strike Format | * 1000 (450000) | dollars (450.0) |
| Right Format | C/P | call/put |
| Symbol Param | root | symbol |
| Response | CSV/flat JSON | Nested JSON |

**Live Validation Results:**
```
Connection: PASS
Expirations: 11 found for SPY
Strikes: 121 found (ATM 580-620)
Quote: SPY241220C00590000 - Bid $8.15, Ask $8.19
```

### 4.3 Options Module Features

**Bug Fixes (Session 78):**
1. Strike boundary check (ITM expansion)
2. Entry slippage modeling (0.2% max cap)
3. Risk-free rate lookup (historical 2008-2024)
4. Theta cost efficiency (75% delta capture)

**Architecture:**
```
ThetaDataProviderBase (ABC)
    |
    +-- ThetaDataRESTClient (historical - IMPLEMENTED)
    |
    +-- ThetaDataWebSocketClient (live - FUTURE)

ThetaDataOptionsFetcher (high-level interface)
    +-- Pickle-based caching (7-day TTL)
    +-- Black-Scholes fallback when unavailable
```

### 4.4 Capital Efficiency

**Why Options for $3k Account:**
- Equity strategies require $10,000+ for full position sizing
- Options provide ~27x capital efficiency
- $3,000 controls ~$80,000 notional exposure

**Strike Selection:**
- Target delta: 0.40-0.55 (slightly OTM)
- DTE: 7-21 days for swing trades

---

## SECTION 5: STRATEGY IMPLEMENTATION STATUS

### 5.1 Strategy Overview

**CRITICAL: Only ONE strategy is fully implemented and deployed.**

| Strategy | Status | File | Priority |
|----------|--------|------|----------|
| 52-Week High Momentum | **DEPLOYED** | `high_momentum_52w.py` | Phase 1 |
| Opening Range Breakout | Structure complete | `orb.py` | Phase 3 |
| Quality-Momentum | SKELETON | `quality_momentum.py` | Phase 1 |
| IBS Mean Reversion | SKELETON | `ibs_mean_reversion.py` | Phase 2 |
| Semi-Volatility Momentum | SKELETON | `semi_vol_momentum.py` | Phase 2 |

### 5.2 52-Week High Momentum (LIVE)

**File:** `strategies/high_momentum_52w.py`
**Status:** Production, deployed as System A1

**Logic:**
```python
# Entry
price_52w_high = close.rolling(252).max()
distance_from_high = close / price_52w_high
entry_signal = distance_from_high >= 0.90  # Within 10% of high

# Exit
exit_signal = distance_from_high < 0.88  # 12% off highs

# Volume confirmation (configurable)
volume_confirmed = volume > volume_ma_20 * 1.25
```

**Performance Targets:**
- Sharpe: 0.8-1.2
- CAGR: 10-15%
- Max DD: -25% to -30%
- Win Rate: 50-60%

**Unique Advantage:** Works in TREND_NEUTRAL (unlike most momentum strategies)

### 5.3 Opening Range Breakout

**File:** `strategies/orb.py`
**Status:** Structure complete, needs research-based modifications

**Implemented:**
- 30-minute opening range calculation
- ATR-based stops (2.5x multiplier)
- Volume confirmation (2.0x MANDATORY)
- Market hours filtering

**Needs:**
- Transaction cost analysis (0.15-0.25% per trade)
- Restriction to S&P 500 only
- Reduced trading frequency

### 5.4 Skeleton Strategies (NOT IMPLEMENTED)

**Quality-Momentum:**
- Academic: Asness et al. (2018) "Quality Minus Junk"
- Logic: Quality filter (ROE, accruals, leverage) + 12-1 momentum
- Target Sharpe: 1.3-1.7
- Unique: Works in ALL regimes

**IBS Mean Reversion:**
- Academic: Connors Research (20+ years validation)
- Logic: IBS < 0.20 entry, >0.80 exit, 3-day time stop
- Target Sharpe: 1.5-2.0
- THRIVES in choppy markets

**Semi-Volatility Momentum:**
- Academic: Moreira & Muir (2017)
- Logic: Position scaling by inverse volatility
- Target Sharpe: 1.4-1.8
- Scaling: 0.5x to 2.0x base position

---

## SECTION 6: TEST SUITE STATUS

### 6.1 Test Coverage Summary

| Component | Tests | Status |
|-----------|-------|--------|
| Regime Detection (Layer 1) | 31/31 | PASSING |
| STRAT Patterns (Layer 2) | 56/56 | PASSING |
| ThetaData Integration | 80/80 | PASSING |
| **Total** | **167+** | **ALL GREEN** |

### 6.2 Key Test Files

```
tests/
+-- test_regime/
|   +-- test_academic_features.py      # Feature calculation
|   +-- test_academic_jump_model.py    # Jump model core
|   +-- test_online_inference.py       # Online inference
|   +-- test_regime_mapping.py         # 4-regime mapping
+-- test_strat/
|   +-- test_bar_classifier.py         # Bar classification (14 tests)
|   +-- test_pattern_detector.py       # Pattern detection (16 tests)
|   +-- test_timeframe_continuity.py   # MTF analysis (21 tests)
|   +-- test_options_pnl.py            # Options P/L
+-- test_integrations/
    +-- test_thetadata_client.py       # REST client (50 tests)
    +-- test_thetadata_options_fetcher.py  # Fetcher (30 tests)
```

### 6.3 Running Tests

```bash
# Run all tests
.venv/Scripts/python.exe -m pytest tests/ -v

# Run specific layer
.venv/Scripts/python.exe -m pytest tests/test_regime/ -v
.venv/Scripts/python.exe -m pytest tests/test_strat/ -v

# Note: uv run may fail due to deephaven-client Python 3.14 marker
# Use .venv directly as workaround
```

---

## SECTION 7: DATA SOURCES & INTEGRATIONS

### 7.1 Authorized Data Sources

| Source | Use Case | Status |
|--------|----------|--------|
| Alpaca | Primary equity data (via VBT Pro) | ACTIVE |
| Tiingo | 30+ year historical data | ACTIVE |
| ThetaData | Options data (v3 API) | ACTIVE |
| yfinance | VIX data ONLY | ACTIVE |

**PROHIBITED:**
- yfinance for equity data (SPY, QQQ, etc.)
- Synthetic/mock price generators
- Random data generation for tests

### 7.2 Data Fetch Patterns

**Equity Data (Alpaca):**
```python
data = vbt.AlpacaData.pull(
    'SPY',
    start='2025-01-01',
    end='2025-11-20',
    timeframe='1d',
    tz='America/New_York',  # MANDATORY
    client_config=dict(api_key=key, secret_key=secret, paper=True)
)
```

**Historical Data (Tiingo):**
```python
from integrations.tiingo_data_fetcher import TiingoDataFetcher
fetcher = TiingoDataFetcher()
data = fetcher.fetch('SPY', start='1993-01-01', end='2025-11-15')
# Returns VBT Pro Data object, 30+ years available
```

**Options Data (ThetaData):**
```python
from integrations.thetadata_client import ThetaDataRESTClient
client = ThetaDataRESTClient()
if client.connect():
    expirations = client.get_expirations('SPY', min_dte=7, max_dte=60)
    strikes = client.get_strikes('SPY', expirations[0])
    quote = client.get_quote('SPY', expirations[0], 590.0, 'call')
```

### 7.3 Tiingo Data Coverage

| Symbol | Trading Days | Date Range |
|--------|-------------|------------|
| SPY | 8,257 | 1993-01-29 to present |
| QQQ | 6,715 | 1999-03-10 to present |
| IWM | 6,407 | 2000-05-26 to present |

---

## SECTION 8: CRITICAL DEVELOPMENT RULES

### 8.1 Mandatory 5-Step VBT Workflow

**ZERO TOLERANCE for skipping ANY step:**

```
1. SEARCH - mcp__vectorbt-pro__search() for patterns/examples
2. VERIFY - resolve_refnames() to confirm methods exist
3. FIND   - mcp__vectorbt-pro__find() for real-world usage
4. TEST   - mcp__vectorbt-pro__run_code() minimal example
5. IMPLEMENT - Only after steps 1-4 pass
```

### 8.2 Date/Timezone Handling

**MANDATORY for all US market data:**
```python
# CORRECT
data = vbt.AlpacaData.pull('SPY', tz='America/New_York', ...)

# WRONG (causes date shifts, 0% TradingView match)
data = vbt.AlpacaData.pull('SPY', ...)  # Missing timezone
```

**Verification:**
```python
# Must be zero weekend dates
assert data.index.dayofweek.max() < 5, "Weekend bars detected!"
# Must be America/New_York
assert data.index.tz.zone == 'America/New_York'
```

### 8.3 Volume Confirmation

**MANDATORY for all breakout strategies:**
```python
volume_ma_20 = data['Volume'].rolling(20).mean()
volume_surge = data['Volume'] > (volume_ma_20 * 2.0)  # 2.0x threshold

# Entry requires volume confirmation
long_entries = price_breakout & volume_surge
```

### 8.4 Professional Standards

- NO emojis (Windows unicode errors)
- NO AI attribution in commits
- Plain ASCII text only
- Professional git commit messages (conventional commits)
- Focus on accomplishments, not problems

### 8.5 Session Workflow

**Start of Session:**
1. Read HANDOFF.md (ALWAYS FIRST)
2. Read CLAUDE.md (development rules)
3. Verify VBT environment
4. Check Next Actions

**End of Session:**
1. Update HANDOFF.md
2. Store facts in OpenMemory
3. Update .session_startup_prompt.md
4. Git commit and push

---

## SECTION 9: FILE STRUCTURE

### 9.1 Core Directories

```
vectorbt-workspace/
|
+-- regime/                    # Layer 1 - Regime Detection
|   +-- academic_jump_model.py # Statistical jump model
|   +-- academic_features.py   # Feature calculation
|   +-- vix_acceleration.py    # Flash crash detection
|   +-- vix_spike_detector.py  # Real-time VIX monitoring
|   +-- regime_allocator.py    # Allocation logic
|
+-- strat/                     # Layer 2 - STRAT Patterns
|   +-- bar_classifier.py      # Bar classification
|   +-- pattern_detector.py    # Pattern detection
|   +-- timeframe_continuity.py # MTF alignment
|   +-- tier1_detector.py      # Tier 1 patterns
|   +-- atlas_integration.py   # Regime integration
|   +-- options_module.py      # Options backtester
|   +-- greeks.py              # Black-Scholes Greeks
|   +-- risk_free_rate.py      # Historical rates
|
+-- strategies/                # Equity Strategies
|   +-- base_strategy.py       # Abstract base class
|   +-- high_momentum_52w.py   # 52-Week High (DEPLOYED)
|   +-- orb.py                 # Opening Range Breakout
|   +-- quality_momentum.py    # SKELETON
|   +-- ibs_mean_reversion.py  # SKELETON
|   +-- semi_vol_momentum.py   # SKELETON
|
+-- integrations/              # External Integrations
|   +-- thetadata_client.py    # ThetaData REST v3
|   +-- thetadata_options_fetcher.py
|   +-- tiingo_data_fetcher.py # 30+ year historical
|
+-- utils/                     # Utilities
|   +-- position_sizing.py     # ATR-based sizing
|   +-- portfolio_heat.py      # Heat management
|   +-- data_fetch.py          # Data fetching
|
+-- tests/                     # Test Suite (167+ tests)
|   +-- test_regime/
|   +-- test_strat/
|   +-- test_integrations/
|
+-- docs/                      # Documentation
    +-- HANDOFF.md             # Session state (READ FIRST)
    +-- CLAUDE.md              # Development rules
    +-- SYSTEM_ARCHITECTURE/   # Architecture docs (1-5)
```

### 9.2 Key Documentation Files

| File | Purpose | Read When |
|------|---------|-----------|
| `docs/HANDOFF.md` | Current session state | EVERY SESSION START |
| `docs/CLAUDE.md` | Development rules | Reference as needed |
| `.session_startup_prompt.md` | Quick context | Session start |
| `docs/SYSTEM_ARCHITECTURE/1_*.md` | Strategy specs | Strategy work |

---

## SECTION 10: CAPITAL DEPLOYMENT

### 10.1 Capital Requirements

| Capital | Recommended Approach | Rationale |
|---------|---------------------|-----------|
| $3,000 | STRAT + Options | 27x leverage, capital efficient |
| $5,000-$9,999 | Options preferred | Equities constrained |
| $10,000+ | Either approach | Full position sizing |
| $25,000+ | Full equity strategies | No constraints |

### 10.2 User's Current Situation

- Capital: ~$3,000
- Account: Schwab Level 1 options (cash account, no margin)
- Paper trading: Alpaca (~$10,000 simulated)

**Level 1 Constraints:**
- CAN: Long stock, long calls/puts, cash-secured puts, straddles/strangles
- CANNOT: Short stock, short options, spreads

### 10.3 Deployment Strategy

**Current (System A1):**
- 52-Week Momentum on Alpaca paper account
- 6 positions, TREND_NEUTRAL regime
- Validates regime detection and execution

**Future (STRAT + Options):**
- Deploy STRAT patterns with options execution
- $3k live capital with 27x efficiency
- Requires ThetaData integration completion

---

## SECTION 11: KNOWN ISSUES & NEXT STEPS

### 11.1 Known Issues

1. **uv Dependency:** `uv run` fails with deephaven-client Python 3.14 marker
   - Workaround: Use `.venv/Scripts/python.exe` directly

2. **README.md Outdated:**
   - States STRAT is "Design phase" (actually code complete)
   - Lists 4 strategies (only 1 implemented)

3. **Deephaven Dashboard:** Duration format bug identified (1-line fix pending)

### 11.2 Session 82 Priorities

**HIGH:**
1. Run full options backtesting with ThetaData real prices
2. Compare synthetic vs real options P/L
3. Update options_module.py to prefer ThetaData

**MEDIUM:**
4. Document pricing discrepancy analysis
5. Fix Deephaven dashboard bug

**LOW:**
6. Update README.md accuracy
7. Implement skeleton strategies

---

## SECTION 12: OPENNEMORY CONTEXT

### 12.1 Session History

81 sessions documented across memory sectors:
- Semantic: System status, architecture decisions
- Procedural: Implementation workflows, bug fixes
- Episodic: Session outcomes, deployment events
- Reflective: Lessons learned, decision rationale

### 12.2 Key Decisions Stored

- Session 20: Multi-layer architecture defined
- Session 31-32: STRAT bar classification and pattern detection
- Session 50: System A1 deployment
- Session 77: Structural level target fix
- Session 81: ThetaData v3 migration

### 12.3 Querying OpenMemory

```python
# Search for context
mcp__openmemory__openmemory_query(
    query="ATLAS regime detection implementation",
    k=10
)

# Store session facts
mcp__openmemory__openmemory_store(
    content="Session XX summary...",
    tags=["session-XX", "feature"]
)
```

---

## SECTION 13: QUICK REFERENCE

### 13.1 Common Commands

```bash
# Verify VBT Pro
uv run python -c "import vectorbtpro as vbt; print(vbt.__version__)"

# Run tests (use venv directly due to uv issue)
.venv/Scripts/python.exe -m pytest tests/ -v

# Start Deephaven dashboard
docker-compose up -d strat-deephaven
# Access: http://localhost:10000/ide
```

### 13.2 Key Imports

```python
# Regime detection
from regime.academic_jump_model import AcademicJumpModel
from regime.vix_acceleration import detect_realtime_vix_spike

# STRAT patterns
from strat.bar_classifier import classify_bars
from strat.pattern_detector import StratPatternDetector

# Options
from integrations.thetadata_client import ThetaDataRESTClient
from strat.options_module import OptionsBacktester

# Strategies
from strategies.high_momentum_52w import HighMomentum52W
```

### 13.3 Environment Variables

```bash
# Required in .env
ALPACA_API_KEY=xxx
ALPACA_SECRET_KEY=xxx
TIINGO_API_KEY=xxx
GITHUB_TOKEN=xxx  # For VBT Pro

# ThetaData (when terminal running)
THETADATA_ENABLED=true
THETADATA_PORT=25503  # v3 API
```

---

## END OF COMPREHENSIVE GUIDE

**Document Stats:**
- Sections: 13
- Strategies covered: 5 (1 deployed, 4 documented)
- Layers covered: 4
- Tests documented: 167+
- Sessions referenced: 81

**Last Updated:** November 26, 2025
**Next Review:** After Session 82 (options backtesting)

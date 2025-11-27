# CLAUDE_REFERENCE.md - Detailed Examples and Patterns

> **Reference document for detailed examples. Only read when needed.**
> **Core rules are in `docs/CLAUDE.md` (read every session).**

---

## 1. Git Commit Message Examples

**CORRECT:**
```
feat: implement BaseStrategy abstract class for multi-strategy system

Add abstract base class that all ATLAS strategies will inherit from.
Includes generate_signals(), calculate_position_size(), and backtest()
methods. Enables portfolio-level orchestration and standardized metrics.
```

**Format:**
- Type: feat/fix/docs/test/refactor
- Brief description (50 chars max, lowercase, no period)
- Blank line
- Detailed explanation (what changed and why)
- NO emojis, NO special characters, NO AI attribution

---

## 2. Data Fetching Patterns

### Alpaca (Primary)
```python
data = vbt.AlpacaData.pull(
    'AAPL',
    start='2025-11-01',
    end='2025-11-20',
    timeframe='1d',
    tz='America/New_York',  # CRITICAL
    client_config=dict(api_key=key, secret_key=secret, paper=True)
)
```

### Tiingo (Secondary)
```python
from integrations.tiingo_data_fetcher import TiingoDataFetcher
fetcher = TiingoDataFetcher()
data = fetcher.fetch('SPY', start='2025-01-01', end='2025-01-31')
```

### VIX Only (yfinance exception)
```python
import yfinance as yf
vix = yf.Ticker("^VIX")
```

### Data Verification
```python
# Verify no weekend bars
assert data.index.dayofweek.max() < 5, "Weekend bars detected!"

# Verify timezone
assert data.index.tz.zone == 'America/New_York', f"Wrong timezone"
```

---

## 3. VectorBT Pro 5-Step Workflow Details

### Step 1: SEARCH
```python
mcp__vectorbt-pro__search(
    query="position sizing risk management from_signals",
    asset_names=["examples", "api", "docs"],
    search_method="hybrid",
    max_tokens=2000
)
```

### Step 2: VERIFY
```python
mcp__vectorbt-pro__resolve_refnames(
    refnames=["vbt.Portfolio", "vbt.PF", "vbt.Portfolio.from_signals"]
)
# Output: OK vbt.Portfolio vectorbtpro.portfolio.base.Portfolio
```

### Step 3: FIND
```python
mcp__vectorbt-pro__find(
    refnames=["vbt.Portfolio.from_signals"],
    asset_names=["examples", "messages"],
    aggregate_messages=True,
    max_tokens=2000
)
```

### Step 4: TEST
```python
mcp__vectorbt-pro__run_code(
    code="""
import vectorbtpro as vbt
import pandas as pd
import numpy as np

np.random.seed(42)
close = pd.Series(100 + np.cumsum(np.random.randn(100)))
entries = pd.Series([True] + [False]*99)
exits = pd.Series([False]*99 + [True])

pf = vbt.PF.from_signals(close=close, entries=entries, exits=exits, init_cash=10000)
print(f"Total Return: {pf.total_return:.2%}")
print("SUCCESS")
""",
    restart=False
)
```

### Step 5: IMPLEMENT
Only after steps 1-4 pass. Use exact data format from step 4.

---

## 4. Volume Confirmation Pattern

All breakout strategies MUST include 2x volume confirmation:

```python
def generate_signals(self, data: pd.DataFrame) -> dict:
    price_breakout_long = data['Close'] > opening_high

    # MANDATORY volume confirmation
    volume_ma_20 = data['Volume'].rolling(20).mean()
    volume_surge = data['Volume'] > (volume_ma_20 * 2.0)

    long_entries = price_breakout_long & volume_surge

    return {'long_entries': long_entries, 'volume_confirmed': volume_surge}
```

---

## 5. STRAT Layer Architecture

```
Layer 1 (ATLAS): Regime Detection
- Input: SPY/market daily OHLCV
- Output: 'TREND_BULL', 'TREND_BEAR', 'TREND_NEUTRAL', 'CRASH'
- Interface: atlas_model.online_inference(data, date) -> pd.Series

Layer 2 (STRAT): Pattern Recognition
- Input: Individual stock/ETF intraday + daily OHLCV
- Output: Pattern signals with entry/stop/target prices
- Interface: strat_detector.run(data) -> dict

Layer 3 (Execution): Capital-Aware Trading
- Input: ATLAS regime + STRAT signals
- Output: Executed trades
- Capital: $3k -> options, $10k+ -> equities
```

### Integration Testing
```python
# Test Layer 1 independently
def test_atlas_regime_detection():
    regime = atlas_model.online_inference(spy_data, date='2020-03-15')
    assert regime == 'CRASH'

# Test Layer 2 independently
def test_strat_pattern_detection():
    pattern = strat_detector.run(spy_data)
    assert pattern['pattern'] == '3-1-2-up'

# Test Layer 3 integration
def test_unified_signal():
    regime = atlas_model.online_inference(spy_data, date='2024-01-15')
    pattern = strat_detector.run(spy_data)
    signal = generate_unified_signal(regime, pattern)

    if regime == 'TREND_BULL' and pattern['direction'] == 'bullish':
        assert signal['quality'] == 'HIGH'
```

---

## 6. Capital Documentation Pattern

```python
class NewStrategy(BaseStrategy):
    """
    CAPITAL REQUIREMENTS:
    - Minimum Viable: $10,000 (full position sizing)
    - Undercapitalized: $3,000-$9,999 (capital constrained)
    - Optimal: $25,000+ (no constraints)

    With $3,000: Use options variant instead
    """
    def generate_signals(self, data):
        return {'long_entries': entries}
```

---

## 7. Multi-Session Plan Template

### Plan File Structure
```markdown
# Project Name - Implementation Plan

**Date:** YYYY-MM-DD
**Objective:** One-line goal
**Estimated Sessions:** X-Y sessions
**Priority:** ACCURACY over speed

---

## Multi-Session Breakdown

| Session | Phase | Focus | Deliverables |
|---------|-------|-------|--------------|
| 83C | 1 | Foundation | protocols.py, config.py |
| 83D | 2 | Walk-Forward | walk_forward.py + tests |

---

## Session Start Protocol
1. Read .session_startup_prompt.md
2. Read HANDOFF.md
3. Read this plan file
4. Query OpenMemory
5. Verify tests passing

## Session End Protocol
1. Commit completed work
2. Update HANDOFF.md
3. Store in OpenMemory
4. Update .session_startup_prompt.md
```

### Session Startup Prompt Template
```markdown
# Session XYZ Startup: [Focus Area]

**Date**: YYYY-MM-DD
**Previous Session**: XYZ-1 ([What was done])
**Status**: [Tests passing, key metrics]

## Session XYZ Mission
- Focus area
- Files to create
- Acceptance criteria

## Key Files to Reference
| File | Purpose |
|------|---------|
...

## Quick Reference
- Test command
- Test counts
```

### OpenMemory Storage Pattern
```
Session [X] - [Title] (Date)

KEY ACCOMPLISHMENTS:
1. [File] - [LOC] - [Purpose]

FILES CREATED:
- [path] ([LOC])

TEST RESULTS:
- [X] tests: ALL PASSING

COMMIT: [hash] [message]

NEXT SESSION:
1. [Task]
```

---

## 8. NYSE Market Hours Filtering

```python
import pandas_market_calendars as mcal

# Get NYSE calendar
nyse = mcal.get_calendar('NYSE')

# Get valid trading days
schedule = nyse.schedule(start_date='2023-01-01', end_date='2023-12-31')
valid_days = schedule.index

# Filter data
filtered_data = data[data.index.normalize().isin(valid_days)]

# Verify
assert filtered_data.index.dayofweek.max() < 5, "Weekend bars!"
assert '2023-12-25' not in str(filtered_data.index), "Holiday bar!"
```

---

## 9. Context Management Triggers

| Condition | Action |
|-----------|--------|
| Context >50% | Prepare handoff |
| Context >70% | MANDATORY handoff |
| >10 files changed | STOP and simplify |
| Repeating questions | Context fatigue |

---

## 10. File Management

- DELETE redundant files (don't archive)
- Keep <15 core Python files
- One test file per component
- One documentation file (HANDOFF.md)

---

## 11. Security Rules

- NO credential harvesting
- NO bulk crawling
- YES to defensive security analysis
- YES to vulnerability explanations
- Verify external code before execution
- Real market data only

---

**Version:** 1.0 (November 27, 2025)
**Purpose:** Detailed examples for CLAUDE.md rules
**Read When:** Need specific implementation patterns

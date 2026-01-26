# Dashboard Development Handoff

> Separate handoff document for dashboard-specific development sessions (DB-x series)

---

## Quick Reference

| Item | Value |
|------|-------|
| Dashboard URL | http://localhost:8050 |
| Run Command | `set PYTHONPATH=. && python -m dashboard.app` |
| Main Entry | `dashboard/app.py` |
| Last Session | DB-1 (2026-01-26) |
| Next Session | DB-2 |

---

## Session History

### DB-1 (2026-01-26) - EQUITY-93B
**Commit:** `dc2b632`

**Completed:**
- Wired account selector to STRAT Analytics callback
- Added strategy selector dropdown (All/STRAT/StatArb)
- Fixed win rate "(0)" bug - `'trades'` → `'total_trades'` key
- Blocked continuation patterns (2U-2U, 2D-2D) in entry_monitor
- Added `/pnl_by_strategy` endpoint to crypto API
- Added `get_pnl_by_strategy()` to crypto_loader

**Issues Identified (Code Review):**
1. OptionsDataLoader reinstantiation (see Known Issues)
2. API response format inconsistency (see Known Issues)

---

## Known Issues (Priority Order)

### Issue 1: OptionsDataLoader Reinstantiation
**Priority:** Medium | **Severity:** Memory Leak | **Session:** DB-1

**Location:** `dashboard/app.py` line 1936

**Problem:**
```python
def render_strat_analytics_tab(active_tab, market, account, strategy, n_intervals):
    # Creates NEW instance every 30 seconds!
    loader = OptionsDataLoader(account=selected_account)
```

**Impact:**
- New HTTP session created every 30-second callback
- ~960 abandoned sessions per 8-hour trading day
- Memory leak (16-32MB/day)
- Potential Alpaca rate limiting

**Proposed Fix (Singleton Pattern):**
```python
# At module level
_options_loaders: Dict[str, OptionsDataLoader] = {}

def get_options_loader(account: str) -> OptionsDataLoader:
    """Get or create OptionsDataLoader for account (singleton per account)."""
    if account not in _options_loaders:
        _options_loaders[account] = OptionsDataLoader(account=account)
    return _options_loaders[account]

# In callback
loader = get_options_loader(selected_account)  # Reuses existing
```

---

### Issue 2: API Response Format Inconsistency
**Priority:** Medium | **Severity:** API Contract Violation | **Session:** DB-1

**Location:** `crypto/api/server.py` lines 211-265

**Problem:**
```
Documented format:    {'total_pnl': float, 'trade_count': int, 'win_rate': float}
Actual PaperTrader:   {'gross': float, 'fees': float, 'funding': float, 'net': float, 'trades': int}
```

**Impact:**
- Clients expecting documented format get KeyError
- Dashboard may silently fail to display strategy P/L

**Proposed Fix:**
```python
def _transform_pnl_response(raw: Dict) -> Dict:
    """Transform PaperTrader format to documented API format."""
    def transform_strategy(data: Dict) -> Dict:
        return {
            'total_pnl': data.get('net', 0),
            'trade_count': data.get('trades', 0),
            'win_rate': 0,  # Calculate from trade history
        }
    return {
        'strat': transform_strategy(raw.get('strat', {})),
        'statarb': transform_strategy(raw.get('statarb', {})),
        'combined': transform_strategy(raw.get('combined', {})),
    }
```

---

### Issue 3: Options Trades Show "Unclassified" Pattern
**Priority:** High | **Severity:** Data Loss | **Session:** DB-1

**Problem:**
Alpaca API doesn't store STRAT pattern metadata. When fetching closed trades, pattern context is lost.

**Current Workaround (Fragile):**
1. Look up in `signal_store` by OSI symbol
2. Fall back to `enriched_trades.json` from backfill script
3. Default to "Unclassified"

**Proposed Fix (DB-2):**
Store pattern metadata locally when trades are entered:
```
Signal Triggered → Save to data/trade_metadata.json → Order placed with Alpaca
                          ↓
Dashboard fetches closed trades → Look up pattern from trade_metadata.json
```

**Files to Modify:**
- `strat/signal_automation/executor.py` - Save metadata on order placement
- `dashboard/data_loaders/options_loader.py` - Load from metadata store first

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        dashboard/app.py                          │
│  - Main Dash application                                         │
│  - All callbacks defined here                                    │
│  - Initializes data loaders at startup                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Loaders                                 │
├─────────────────────────────────────────────────────────────────┤
│ options_loader.py    │ Alpaca API + SignalStore + enriched.json │
│ crypto_loader.py     │ VPS REST API (http://178.156.223.251:8080)│
│ regime_loader.py     │ Academic jump model calculations         │
│ live_loader.py       │ Real-time Alpaca positions               │
│ backtest_loader.py   │ Local results directory                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Components                                  │
├─────────────────────────────────────────────────────────────────┤
│ strat_analytics_panel.py  │ STRAT Analytics tab (Options/Crypto)│
│ options_panel.py          │ Options-specific displays           │
│ crypto_panel.py           │ Crypto-specific displays            │
│ regime_panel.py           │ Regime detection visualizations     │
│ portfolio_panel.py        │ Portfolio overview                  │
│ risk_panel.py             │ Risk management displays            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: STRAT Analytics Tab

```
User selects: Market=Crypto, Account=SMALL, Strategy=StatArb
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ Callback: render_strat_analytics_tab()                          │
│ Inputs: active_tab, market, account, strategy, n_intervals      │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
    market='options'                        market='crypto'
          │                                       │
          ▼                                       ▼
  OptionsDataLoader(account)              crypto_loader (global)
          │                                       │
          ▼                                       ▼
  Alpaca API + SignalStore               VPS /trades endpoint
          │                                       │
          └───────────────────┬───────────────────┘
                              ▼
                    get_closed_trades()
                              │
                              ▼
              Filter by strategy if != 'all'
                              │
                              ▼
         calculate_metrics(trades, strategy)
         calculate_pattern_stats(trades, strategy)
                              │
                              ▼
                   Render tab content
```

---

## Common Gotchas

### 1. Field Name Mismatches
```python
# WRONG - calculate_pattern_stats returns 'total_trades'
trades = stats.get('trades', 0)

# RIGHT
trades = stats.get('total_trades', 0)
```

### 2. Callback Input Order Matters
```python
# Inputs must match function parameter order
@app.callback(
    Output(...),
    [Input('strat-analytics-tabs', 'active_tab'),      # 1st param
     Input('strat-market-selector', 'value'),          # 2nd param
     Input('strat-account-selector', 'value'),         # 3rd param
     Input('strat-strategy-selector', 'value'),        # 4th param
     Input('strat-analytics-refresh', 'n_intervals')]  # 5th param
)
def render_strat_analytics_tab(active_tab, market, account, strategy, n_intervals):
```

### 3. Loader Connection State
```python
# Always check connection before using
if loader is None or not loader._connected:
    return html.Div('Loader not connected')
```

### 4. Pattern Field Names Vary by Source
```python
# Crypto API returns: pattern_type
# Options SignalStore returns: pattern
# Dashboard expects: pattern

# Normalize in loader:
trade['pattern'] = trade.get('pattern_type') or trade.get('pattern') or 'Unclassified'
```

### 5. STRAT Patterns - Continuation vs Reversal
```python
# These should NOT be traded (continuation patterns)
CONTINUATION_PATTERNS = {'2U-2U', '2D-2D'}

# These ARE traded (reversal patterns)
REVERSAL_PATTERNS = {'2U-2D', '2D-2U', '3-2U', '3-2D', ...}
```

---

## Key Files Quick Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `dashboard/app.py` | Main app, callbacks | `render_strat_analytics_tab()` |
| `dashboard/components/strat_analytics_panel.py` | STRAT Analytics UI | `calculate_metrics()`, `calculate_pattern_stats()` |
| `dashboard/data_loaders/options_loader.py` | Alpaca + patterns | `get_closed_trades()`, `_load_enriched_tfc_data()` |
| `dashboard/data_loaders/crypto_loader.py` | VPS API | `get_closed_trades()`, `get_pnl_by_strategy()` |
| `crypto/api/server.py` | VPS REST API | `/trades`, `/pnl_by_strategy` |
| `crypto/scanning/entry_monitor.py` | Entry trigger logic | Continuation pattern blocking |

---

## Testing Checklist

### Before Committing Dashboard Changes:

- [ ] Run dashboard locally: `set PYTHONPATH=. && python -m dashboard.app`
- [ ] Test account selector (SMALL/MID/LARGE) - verify data changes
- [ ] Test market selector (Options/Crypto) - verify data changes
- [ ] Test strategy selector (All/STRAT/StatArb) - verify filtering
- [ ] Check browser console for JavaScript errors
- [ ] Verify no Python exceptions in terminal
- [ ] Check metrics display non-zero values where expected

### Syntax Validation:
```bash
python -m py_compile dashboard/app.py
python -m py_compile dashboard/components/strat_analytics_panel.py
python -m py_compile dashboard/data_loaders/options_loader.py
python -m py_compile dashboard/data_loaders/crypto_loader.py
```

---

## Next Session: DB-2

### Primary Goal
Implement real-time trade metadata storage for options:
- Save pattern/TFC to `data/trade_metadata.json` when orders placed
- Load from metadata store as primary source in options_loader

### Secondary Goals
- Fix OptionsDataLoader reinstantiation (singleton pattern)
- Fix API response format inconsistency

### Files to Modify
- `strat/signal_automation/executor.py` (~30 lines)
- `dashboard/data_loaders/options_loader.py` (~40 lines)
- `dashboard/app.py` (~20 lines for singleton fix)
- `crypto/api/server.py` (~30 lines for format fix)

---

## Session Naming Convention

| Session | Focus |
|---------|-------|
| DB-1 | STRAT Analytics selectors, continuation blocking |
| DB-2 | Trade metadata store, loader fixes |
| DB-3+ | Future dashboard work |

---

*Last Updated: 2026-01-26 (DB-1)*

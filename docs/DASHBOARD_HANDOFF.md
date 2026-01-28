# Dashboard Development Handoff

> Separate handoff document for dashboard-specific development sessions (DB-x series)

---

## Quick Reference

| Item | Value |
|------|-------|
| Dashboard URL | http://localhost:8050 |
| Run Command | `set PYTHONPATH=. && python -m dashboard.app` |
| Main Entry | `dashboard/app.py` |
| Last Session | DB-4 (2026-01-28) |
| Next Session | DB-5 |

---

## Session History

### DB-4 (2026-01-28) - TradingView Charts + Equity Stats

**Completed:**
- **TradingView Lightweight Charts:** Replaced Plotly equity curve with `dash_tvlwc` area chart. Uses emerald line for positive return, crimson for negative. 500px height for better aspect ratio.
- **TVLWC Config:** Added `TVLWC_CHART_OPTIONS` dict to `dashboard/config.py` with dark theme styling (camelCase JS API format).
- **Equity Stats Cards:** Added 4 summary metric cards above equity chart: Total Return ($ and %), Max Drawdown (peak-to-trough %), Starting Equity, Current Equity.
- **Helper Functions:** Added `_calculate_equity_stats()` for stats computation, `_create_stats_cards()` for card rendering.

**Files Modified:**
- `dashboard/config.py` - Added TVLWC_CHART_OPTIONS dict
- `dashboard/components/strat_analytics_panel.py` - Replaced equity chart, added stats cards, new helpers

**Tests:** 224 passed (0 failures)

**Code Simplification Backlog (from code-simplifier review):**
- Extract `_stat_card()` helper from `_create_stats_cards()` (~70 lines duplication)
- Consolidate empty chart placeholders in `_create_equity_chart()`
- DARK_THEME unification (local theme vs config.py COLORS)

---

### DB-3 (2026-01-27) - Bug Fixes, Date Columns, Crypto Alignment

**Completed:**
- **Strategy Performance Tab Fix:** Replaced `dcc.Dropdown` with `dbc.Select` to fix dropdown clipping and dark theme rendering issues. Updated card styling to use consistent `bg_card`/`bg_void`/`border_subtle` config keys.
- **STRAT Analytics Connection Error UX:** Replaced terse error text with informative `_create_connection_error_panel()` showing service name, error detail, troubleshooting hint, and `/diagnostic` endpoint reference.
- **Closed Trades Date Column:** Added "Closed" column to closed trades table showing sell date/time. Sources: `sell_time_display` (options), `exit_time` ISO parse (crypto).
- **Open Positions Date Column:** Added "Opened" column to open positions table showing open date/time. Sources: `entry_time_et` (options from signal store), `entry_time` ISO parse (crypto).
- **Crypto Spot/Derivatives Alignment:** Updated market selector label to "Crypto (Spot/Derivs)", updated panel subtitle to clarify "Pattern detection on underlying/spot -- execution via options or derivatives", added header note to crypto panel "(Spot Detection / Derivatives Execution)".

**Files Modified:**
- `dashboard/components/strategy_panel.py` - Dropdown fix, card styling
- `dashboard/components/strat_analytics_panel.py` - Date columns, crypto label, helpers
- `dashboard/components/crypto_panel.py` - Docstring, header label
- `dashboard/app.py` - Connection error panel helper

**Tests:** 224 passed (0 failures)

**Known Issues Remaining:**
- Strategy Performance Tab callbacks still depend on Alpaca connection for non-STRAT strategies
- TradingView chart integration deferred (dash_tvlwc compatibility test needed)

---

### DB-2 (2026-01-26) - Pattern Persistence & Loader Fixes
**Commits:** `3c28d3f`, `d84b19a`

**Completed:**
- Issue 3 (High): Pattern persistence - save metadata at order placement
  - `executor.py`: Added `_save_trade_metadata()` method
  - `options_loader.py`: Added `_load_trade_metadata()` as primary lookup source
- Issue 1 (Medium): OptionsDataLoader singleton pattern
  - `app.py`: Added `get_options_loader()` with per-account caching
  - Added reconnection logic for cached loaders that lost connection
- Issue 2 (Medium): API response format normalization
  - `server.py`: Added `_transform_pnl_response()` to normalize PaperTrader format

**Pattern Lookup Priority (Updated):**
1. `trade_metadata.json` (saved at order placement - most reliable)
2. Signal store lookup (recent trades where signal hasn't expired)
3. `enriched_trades.json` (backfill data)
4. Parse from signal_key in executions.json (fallback)

---

### DB-1 (2026-01-26) - EQUITY-93B
**Commit:** `dc2b632`

**Completed:**
- Wired account selector to STRAT Analytics callback
- Added strategy selector dropdown (All/STRAT/StatArb)
- Fixed win rate "(0)" bug - `'trades'` -> `'total_trades'` key
- Blocked continuation patterns (2U-2U, 2D-2D) in entry_monitor
- Added `/pnl_by_strategy` endpoint to crypto API
- Added `get_pnl_by_strategy()` to crypto_loader

---

## Known Issues (Priority Order)

### Issue 1: Equity Curve UI/UX - RESOLVED (DB-4)
**Status:** RESOLVED in DB-4

**Solution Implemented:**
- Replaced Plotly with TradingView Lightweight Charts (`dash_tvlwc`)
- Increased chart height to 500px for better aspect ratio
- Added 4 summary stats cards: Total Return, Max Drawdown, Starting Equity, Current Equity
- Color-coded chart line (emerald for positive, crimson for negative return)
- Dark theme styling via `TVLWC_CHART_OPTIONS` config

---

### Issue 2: Overall Dashboard Polish
**Priority:** Low | **Severity:** UX | **Session:** DB-2

**Problem:**
Multiple pages need UI improvements for professional appearance:

| Page | Issues |
|------|--------|
| Overview | Metrics cards could use better spacing/hierarchy |
| Open Positions | Progress bars need better labeling |
| Patterns | Pattern stats table needs better formatting |
| TFC | Alignment scores visualization unclear |
| Closed Trades | Table needs pagination for many trades |
| Pending Patterns | Status indicators need color coding |
| Equity Curve | See Issue 1 above |

---

### Issue 3: Account Selector Not Persisted
**Priority:** Low | **Severity:** Minor UX | **Session:** DB-2

**Problem:**
When switching between tabs, the account/market/strategy selectors reset to defaults instead of persisting user's last selection.

**Proposed Fix:**
Use `dcc.Store` to persist selector state across tab switches.

---

## Future Enhancements

### STRAT Analytics Page

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| Win rate by pattern | Show win rate breakdown per pattern type | Medium |
| Time-based filtering | Add date range selector for analysis period | Medium |
| Export to CSV | Download closed trades data | Low |
| P/L by day of week | Show which days perform best | Low |
| Position sizing analysis | Show if TFC-based sizing improves returns | Medium |

### Equity Curve Page

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| Benchmark comparison | Overlay SPY buy-and-hold for comparison | High |
| Drawdown chart | Add separate drawdown visualization | Medium |
| Rolling Sharpe | Show 30-day rolling Sharpe ratio | Low |
| Trade markers | Mark entry/exit points on equity curve | Medium |

### Open Positions Page

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| Real-time P/L | WebSocket updates for live P/L | High |
| Greeks display | Show delta/theta/vega for options | Medium |
| Risk warnings | Highlight positions near stop loss | High |
| Quick close button | One-click position close | Medium |

### Regime Detection Page

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| Historical regime view | Show regime history over time | Medium |
| Regime-filtered returns | Show strategy returns by regime | High |
| VIX overlay | Show VIX alongside regime detection | Low |

### Portfolio Overview Page

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| Asset allocation pie | Visual breakdown of positions | Medium |
| Sector exposure | Show sector concentration | Low |
| Correlation matrix | Position correlation visualization | Low |

---

## Architecture Overview

```
+---------------------------------------------------------------------+
|                        dashboard/app.py                              |
|  - Main Dash application                                             |
|  - All callbacks defined here                                        |
|  - Initializes data loaders at startup                               |
|  - get_options_loader() singleton (DB-2)                             |
+---------------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------------+
|                     Data Loaders                                     |
+---------------------------------------------------------------------+
| options_loader.py    | Alpaca API + SignalStore + trade_metadata    |
| crypto_loader.py     | VPS REST API (http://178.156.223.251:8080)   |
| regime_loader.py     | Academic jump model calculations              |
| live_loader.py       | Real-time Alpaca positions                    |
| backtest_loader.py   | Local results directory                       |
+---------------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------------+
|                      Components                                      |
+---------------------------------------------------------------------+
| strat_analytics_panel.py  | STRAT Analytics tab (Options/Crypto)    |
| options_panel.py          | Options-specific displays                |
| crypto_panel.py           | Crypto-specific displays                 |
| regime_panel.py           | Regime detection visualizations          |
| portfolio_panel.py        | Portfolio overview                        |
| risk_panel.py             | Risk management displays                  |
+---------------------------------------------------------------------+
```

---

## Data Flow: Pattern Lookup (DB-2)

```
ORDER PLACEMENT:
Signal Triggered -> executor._save_trade_metadata() -> trade_metadata.json
                                                              |
                                                              v
                                            data/executions/trade_metadata.json
                                                              |
DASHBOARD LOAD:                                               |
get_closed_trades() -> _load_trade_metadata() ----------------+
                    -> signal_store lookup (fallback)
                    -> enriched_trades.json (fallback)
                    -> parse signal_key (last resort)
                                |
                                v
                         Pattern displayed
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

### 5. Singleton Reconnection (DB-2)
```python
# get_options_loader() now handles reconnection automatically
loader = get_options_loader(account)  # Will retry connect if cached loader disconnected
```

---

## Key Files Quick Reference

| File | Purpose | Key Functions |
|------|---------|---------------|
| `dashboard/app.py` | Main app, callbacks | `render_strat_analytics_tab()`, `get_options_loader()` |
| `dashboard/components/strat_analytics_panel.py` | STRAT Analytics UI | `calculate_metrics()`, `calculate_pattern_stats()` |
| `dashboard/data_loaders/options_loader.py` | Alpaca + patterns | `get_closed_trades()`, `_load_trade_metadata()` |
| `dashboard/data_loaders/crypto_loader.py` | VPS API | `get_closed_trades()`, `get_pnl_by_strategy()` |
| `strat/signal_automation/executor.py` | Order execution | `execute_signal()`, `_save_trade_metadata()` |
| `crypto/api/server.py` | VPS REST API | `/trades`, `/pnl_by_strategy`, `_transform_pnl_response()` |

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
- [ ] Verify equity curve renders with reasonable aspect ratio

### Syntax Validation:
```bash
python -m py_compile dashboard/app.py
python -m py_compile dashboard/components/strat_analytics_panel.py
python -m py_compile dashboard/data_loaders/options_loader.py
python -m py_compile dashboard/data_loaders/crypto_loader.py
```

---

## Next Session: DB-5

### Primary Goal: Code Simplification & Theme Unification

**From DB-4 Code Simplifier Review:**
1. Extract `_stat_card()` helper from `_create_stats_cards()` - reduce 110 lines to ~40 lines
2. Consolidate empty chart placeholders in `_create_equity_chart()`
3. Unify DARK_THEME dict with config.py COLORS (pre-existing inconsistency)

### Secondary Goals
1. Account selector persistence via `dcc.Store` (Issue 3)
2. Strategy Performance Tab - ensure all strategy types render correctly
3. Apply TradingView charts to regime price chart for consistency
4. Add `line_shape='spline'` to regime detection line for smoother transitions

---

## Session Naming Convention

| Session | Focus |
|---------|-------|
| DB-1 | STRAT Analytics selectors, continuation blocking |
| DB-2 | Trade metadata store, loader fixes, singleton pattern |
| DB-3 | Bug fixes, date columns, crypto alignment |
| DB-4 | TradingView Lightweight Charts, equity stats cards |
| DB-5 | Code simplification, theme unification |
| DB-6+ | Future dashboard work |

---

*Last Updated: 2026-01-28 (DB-4)*

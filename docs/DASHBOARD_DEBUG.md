# ATLAS Dashboard Debugging Guide

**Created**: 2025-11-26
**Purpose**: Debug dashboard data loading issues for fresh context window

## Current Issues (from Screenshots)

### Screenshot 1: Regime Detection Tab
- **Regime Timeline**: "No regime data available for selected range" (RED error)
- **Feature Evolution**: "No feature data available" (RED error)
- **Regime Statistics**: Empty/blank
- **Header shows**: LIVE | CONNECTED | NEUTRAL (regime detected correctly at top level)

### Screenshot 2: Live Portfolio Tab
- **Portfolio Value**: $0.00 (WRONG - should be ~$10,280)
- **P&L Today**: $+0.00 (WRONG)
- **Portfolio Heat gauge**: Shows 4.2% (this IS working)
- **Current Positions table**: Empty (WRONG - should show 6 positions)

### Screenshot 3: Risk Management Tab
- **Portfolio Heat Monitor**: "Live data not available" (RED error)
- **Risk Metrics**: "No data available"
- **Position Allocation**: "Live data not available" (RED error)

---

## Root Cause Analysis

### Issue 1: Live Portfolio Not Loading

The `LiveDataLoader` is initialized but positions aren't reaching the UI.

**File**: `dashboard/data_loaders/live_loader.py`

**Verified Working** (tested in CLI):
```python
from dashboard.data_loaders.live_loader import LiveDataLoader
loader = LiveDataLoader()
positions = loader.get_current_positions()
# Returns 6 positions: AAPL, AMAT, AVGO, CRWD, CSCO, GOOGL
```

**Likely Cause**: The callback in `app.py` may be failing silently or the component IDs don't match.

**Debug Steps**:
1. Check browser console (F12) for JavaScript errors
2. Check terminal running dashboard for Python errors
3. Verify callback output IDs match component IDs in `portfolio_panel.py`

**Relevant Callback** (`dashboard/app.py` lines 449-503):
```python
@app.callback(
    [Output('portfolio-value-card', 'children'),
     Output('positions-table', 'data'),
     Output('portfolio-heat-gauge', 'figure')],
    Input('interval-component', 'n_intervals')
)
def update_live_portfolio(n):
```

**Check**: Does `portfolio_panel.py` define components with IDs:
- `portfolio-value-card`
- `positions-table`
- `portfolio-heat-gauge`

---

### Issue 2: Regime Timeline Not Loading

The RegimeDataLoader requires 1000+ days of data but may be timing out or failing.

**File**: `dashboard/data_loaders/regime_loader.py`

**Verified Working** (tested in CLI):
```python
from dashboard.data_loaders.regime_loader import RegimeDataLoader
loader = RegimeDataLoader()
result = loader.get_current_regime()
# Returns: {'regime': 'TREND_NEUTRAL', 'allocation_pct': 70, ...}
```

**Likely Causes**:
1. Callback timeout (regime detection takes 6-10 seconds)
2. Date range picker format mismatch
3. Exception being caught and returning empty DataFrame

**Relevant Callback** (`dashboard/app.py` lines 326-361):
```python
@app.callback(
    Output('regime-timeline-graph', 'figure'),
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_regime_timeline(start_date, end_date):
```

**Debug Steps**:
1. Add logging to callback to see if it's being called
2. Check if `start_date` and `end_date` are in expected format
3. Test with hardcoded dates instead of date picker values

---

### Issue 3: Risk Management Not Loading

The Risk Management callback depends on `live_loader` which may not be properly initialized at callback time.

**File**: `dashboard/app.py` lines 557-682

**Check**: The callback checks `live_loader.client is None` - but the error shows "Live data not available" which means this check is failing.

**Likely Cause**: `live_loader` global variable may be `None` when callback runs.

**Debug Steps**:
1. Add print/logging at start of callback
2. Check if `live_loader` is initialized before callbacks register
3. Verify Alpaca credentials are loaded from `.env`

---

## Key Files to Investigate

| File | Purpose |
|------|---------|
| `dashboard/app.py` | Main app, callbacks, data loader initialization |
| `dashboard/data_loaders/live_loader.py` | Alpaca positions/account data |
| `dashboard/data_loaders/regime_loader.py` | ATLAS regime detection |
| `dashboard/components/portfolio_panel.py` | Live Portfolio UI components |
| `dashboard/components/risk_panel.py` | Risk Management UI components |
| `dashboard/components/regime_panel.py` | Regime Detection UI components |
| `config/settings.py` | Environment variable loading |
| `.env` | Credentials (DEFAULT_ACCOUNT=LARGE) |

---

## Environment Verification

Run these commands to verify environment is correct:

```bash
cd C:\Strat_Trading_Bot\vectorbt-workspace

# 1. Check .env has correct default account
grep DEFAULT_ACCOUNT .env
# Expected: DEFAULT_ACCOUNT=LARGE

# 2. Test Alpaca connection
uv run python -c "
from config.settings import load_config
load_config()
from integrations.alpaca_trading_client import AlpacaTradingClient
client = AlpacaTradingClient(account='LARGE')
client.connect()
positions = client.list_positions()
print(f'Positions: {len(positions)}')
for p in positions:
    print(f'  {p[\"symbol\"]}: {p[\"qty\"]} shares')
"

# 3. Test LiveDataLoader directly
uv run python -c "
from dashboard.data_loaders.live_loader import LiveDataLoader
loader = LiveDataLoader()
print(f'Account: {loader.account}')
print(f'Client connected: {loader.client is not None}')
positions = loader.get_current_positions()
print(f'Positions: {len(positions)}')
print(positions)
"

# 4. Test RegimeDataLoader
uv run python -c "
from dashboard.data_loaders.regime_loader import RegimeDataLoader
loader = RegimeDataLoader()
result = loader.get_current_regime()
print(result)
"
```

---

## Callback Component ID Verification

The callbacks output to specific component IDs. These must match exactly.

### Live Portfolio Callback Outputs:
- `portfolio-value-card` - Card showing portfolio value
- `positions-table` - DataTable with positions
- `portfolio-heat-gauge` - Gauge figure

### Risk Management Callback Outputs:
- `risk-heat-gauge` - Heat gauge figure
- `risk-metrics-table` - Metrics table
- `position-allocation-chart` - Pie chart figure

### Regime Detection Callback Outputs:
- `regime-timeline-graph` - Timeline figure
- `feature-dashboard-graph` - Feature evolution figure

**Action**: Check `dashboard/components/*.py` files to ensure component IDs match callback outputs.

---

## Quick Fix Attempts

### Fix 1: Add Debug Logging to Callbacks

Edit `dashboard/app.py` and add at the top of each callback:
```python
logger.info(f"Callback triggered: update_live_portfolio, n={n}")
```

### Fix 2: Check Callback Registration Order

Ensure data loaders are initialized BEFORE `@app.callback` decorators run.

In `app.py`, the initialization happens at module level:
```python
# Initialize data loaders (around line 60-80)
try:
    regime_loader = RegimeDataLoader()
    ...
except Exception as e:
    regime_loader = None
```

If an exception occurs here, the loader will be `None` and callbacks will fail.

### Fix 3: Test Dashboard with Verbose Logging

```bash
cd C:\Strat_Trading_Bot\vectorbt-workspace
uv run python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from dashboard.app import app
app.run(debug=True, host='127.0.0.1', port=8050)
"
```

Watch terminal output for errors when switching tabs.

---

## Expected Working State

When working correctly:

**Regime Detection Tab**:
- Timeline shows colored bands (green=BULL, gray=NEUTRAL, orange=BEAR, red=CRASH)
- Current regime: TREND_NEUTRAL
- Feature Evolution shows downside deviation, Sortino ratios

**Live Portfolio Tab**:
- Portfolio Value: ~$10,280
- 6 positions: AAPL (3), AMAT (4), AVGO (2), CRWD (1), CSCO (12), GOOGL (3)
- Heat gauge: ~4-7% (based on max position concentration)

**Risk Management Tab**:
- Heat gauge showing portfolio concentration
- Metrics table with equity, position count, max position %
- Pie chart showing position allocation

---

## Architecture Reference

```
dashboard/
├── app.py                 # Main app, callbacks
├── config.py              # Dashboard configuration
├── components/
│   ├── regime_panel.py    # Regime Detection UI
│   ├── strategy_panel.py  # Strategy Performance UI
│   ├── portfolio_panel.py # Live Portfolio UI
│   ├── risk_panel.py      # Risk Management UI
│   └── options_panel.py   # Options Trading UI
├── data_loaders/
│   ├── regime_loader.py   # ATLAS regime data
│   ├── live_loader.py     # Alpaca live data
│   ├── backtest_loader.py # Backtest results
│   └── orders_loader.py   # Order history
└── visualizations/
    ├── regime_charts.py   # Regime visualization functions
    ├── strategy_charts.py # Strategy visualization functions
    └── enhanced_charts.py # Advanced chart functions
```

---

## Session 76 Changes (Recently Merged)

These files were added in Session 76 and may have integration issues:
- `dashboard/data_loaders/async_data_service.py` (745 lines)
- `dashboard/visualizations/enhanced_charts.py` (877 lines)
- `dashboard/components/options_panel.py` (1049 lines)

Check if these new files are causing import errors that prevent data loaders from initializing.

---

## Contact/Context

- **System A1** deployed to LARGE Alpaca paper account on Nov 20, 2025
- **6 positions**: AAPL, AMAT, AVGO, CRWD, CSCO, GOOGL
- **Current regime**: TREND_NEUTRAL (70% allocation)
- **Next rebalance**: February 1, 2026
- **Dashboard**: Plotly Dash on port 8050

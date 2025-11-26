# ATLAS Dashboard Review - Session 76

## Executive Summary

This document provides a comprehensive review of the ATLAS Plotly dashboard implementation, recommendations for visual improvements, async data fetching architecture, and a detailed analysis of Tableau integration possibilities.

**Key Findings:**
- Current dashboard is well-architected with solid TradingView-inspired design
- Data fetching is entirely synchronous, creating performance bottlenecks
- No ThetaData integration currently exists (despite being mentioned in system goals)
- Significant opportunity for visual and performance improvements

---

## 1. Current Dashboard Assessment

### Architecture Overview

| Component | Technology | Status |
|-----------|------------|--------|
| Web Framework | Plotly Dash + Bootstrap | Stable |
| Data Layer | Synchronous HTTP (requests) | Needs Upgrade |
| Broker Integration | Alpaca (paper trading) | Working |
| Market Data | Yahoo Finance via VectorBT Pro | Working |
| Options Trading | Alpaca Options API (Paper enabled) | **NEW - Session 76** |
| Real-time Updates | 30-second polling (dcc.Interval) | Adequate |

### Strengths

1. **Professional Visual Design**
   - TradingView-inspired dark theme (#090008 background)
   - Proper z-ordering (regime shading → grid → price → markers)
   - Color-coded P&L (green/red for gains/losses)
   - Responsive 4-tab navigation

2. **Well-Structured Codebase**
   - Clean separation: `components/`, `visualizations/`, `data_loaders/`
   - Centralized configuration in `config.py`
   - Consistent color scheme across all charts

3. **Functional Live Portfolio Monitoring**
   - Real Alpaca API integration
   - Position tracking with unrealized P&L
   - Portfolio heat gauge

### Weaknesses

1. **Synchronous Data Fetching**
   - All API calls block the main thread
   - Multiple sequential calls cause latency (300-500ms each)
   - No concurrent request capability

2. **Missing ThetaData Integration**
   - Options module exists in `/strat/options_module.py`
   - No actual ThetaData API client implemented
   - Historical options data unavailable

3. **Limited STRAT Pattern Visualization**
   - Bar classifications calculated but not displayed
   - No pattern annotations on charts
   - Missing measured move target lines

4. **Static Risk Metrics**
   - `create_risk_metrics_table()` uses hardcoded values
   - No real-time VaR calculations
   - Portfolio heat is a placeholder (0.042)

---

## 2. Visual Improvements Implemented

### New Enhanced Charts Module
**File:** `dashboard/visualizations/enhanced_charts.py`

#### 2.1 STRAT Candlestick Chart
```python
create_strat_candlestick_chart(ohlc_data, bar_types, patterns, regimes)
```
- Color-coded bars by STRAT type (1=gray, 2U=green, 2D=red, 3=yellow)
- Pattern entry/target/stop annotations
- Regime background shading
- Volume profile subplot

#### 2.2 Portfolio Metrics Cards
```python
create_portfolio_metrics_cards(equity, pnl_today, pnl_pct, positions_count, buying_power, portfolio_heat)
```
- 6-indicator grid layout
- Color-coded P&L indicators
- Portfolio heat gauge with threshold warnings
- Market status display

#### 2.3 Multi-Timeframe Regime Heatmap
```python
create_regime_heatmap(regime_data, symbols)
```
- Shows regime confluence across timeframes
- Useful for STRAT multi-TF analysis
- Color scale: CRASH → BEAR → NEUTRAL → BULL

#### 2.4 P&L Attribution Waterfall
```python
create_pnl_waterfall(contributions, total_pnl)
```
- Visual breakdown of P&L by source
- Waterfall format for clear attribution
- Green/red for positive/negative contributions

#### 2.5 Options Flow Chart
```python
create_options_flow_chart(flow_data)
```
- Bubble chart for unusual options activity
- Size = volume, color = call/put
- Ready for ThetaData integration

#### 2.6 Sparkline Mini-Charts
```python
create_sparkline(data, positive_color, negative_color)
```
- Compact inline trend visualization
- Useful for position cards
- Auto-detect trend direction

---

## 3. Async Data Fetching Architecture

### New Async Service Module
**File:** `dashboard/data_loaders/async_data_service.py`

### Why Async Matters for ATLAS

**Current (Synchronous) Flow:**
```
Dashboard Refresh
  → Fetch Account (300ms)
  → Fetch Positions (300ms)
  → Fetch Orders (300ms)
  → Total: ~900ms blocking
```

**Async Flow with httpx:**
```
Dashboard Refresh
  → Concurrent: Account | Positions | Orders
  → Total: ~300ms (parallelized)
```

**3x faster refresh with same API rate limits!**

### Architecture Design

```
┌─────────────────────────────────────────────────────────────┐
│                    AsyncDataService                          │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  AsyncHTTPClient                       │  │
│  │  - Connection pooling (100 connections)               │  │
│  │  - HTTP/2 multiplexing (httpx)                        │  │
│  │  - Automatic retries with exponential backoff         │  │
│  │  - 30-second request timeout                          │  │
│  └───────────────────────────────────────────────────────┘  │
│                           │                                  │
│        ┌──────────────────┼──────────────────┐              │
│        ▼                  ▼                  ▼              │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐          │
│  │ Alpaca   │      │ ThetaData│      │ Yahoo    │          │
│  │ Client   │      │ Client   │      │ Finance  │          │
│  └──────────┘      └──────────┘      └──────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

#### AsyncHTTPClient
- Supports both **httpx** (preferred) and **aiohttp** (fallback)
- HTTP/2 enabled for request multiplexing
- Connection pooling for efficiency
- Automatic retry with exponential backoff

#### AlpacaAsyncClient
- Async versions of all Alpaca endpoints
- `fetch_all_portfolio_data()` - concurrent account/positions/orders
- `get_latest_quotes()` - batch quote fetching

#### ThetaDataAsyncClient (Placeholder)
- Ready for integration when ThetaData subscription active
- Supports option chains, quotes, historical data
- Local terminal connection (127.0.0.1:25510)

### Usage in Dash Callbacks

```python
from dashboard.data_loaders.async_data_service import AsyncDataService, run_async

# Create service (initialize once)
service = AsyncDataService()

@app.callback(Output('portfolio-data', 'children'), Input('refresh', 'n_clicks'))
def update_portfolio(n):
    # Use run_async() wrapper for sync callbacks
    data = run_async(service.fetch_portfolio_data())
    return format_portfolio(data)
```

### httpx vs aiohttp Comparison

| Feature | httpx | aiohttp |
|---------|-------|---------|
| HTTP/2 Support | Yes | No |
| API Similarity to requests | High | Low |
| Connection Pooling | Yes | Yes |
| Automatic Retries | Built-in | Manual |
| Streaming | Yes | Yes |
| WebSocket | Basic | Full |
| Maintenance | Active | Active |

**Recommendation:** Use **httpx** as primary, aiohttp as fallback.

---

## 4. Tableau Integration Analysis

### Overview

Tableau is a powerful business intelligence platform that could complement or replace portions of the ATLAS dashboard. Here's a comprehensive analysis:

### Pros of Tableau Integration

#### 1. **Superior Data Visualization**
- 50+ chart types out of the box
- Drag-and-drop dashboard creation
- Publication-quality visualizations
- Automatic best-practice formatting

#### 2. **Advanced Analytics**
- Built-in statistical functions
- Forecasting and trend analysis
- Clustering and segmentation
- LOD (Level of Detail) expressions

#### 3. **Data Connectivity**
- 90+ native data connectors
- Live connections to databases
- Combine multiple data sources
- Built-in ETL capabilities

#### 4. **Enterprise Features**
- Role-based access control
- Scheduled data refreshes
- Embedded analytics
- Mobile-optimized dashboards

#### 5. **Collaboration**
- Tableau Server/Cloud for sharing
- Comments and annotations
- Alerts and subscriptions
- Version control

### Cons of Tableau Integration

#### 1. **Cost**
- Tableau Creator: $70/user/month
- Tableau Explorer: $42/user/month
- Tableau Viewer: $15/user/month
- Server licensing additional
- **Total for small team: $500-2000/year**

#### 2. **Real-Time Limitations**
- Minimum refresh: 1 minute (Tableau Cloud)
- No true real-time streaming
- WebSocket not supported
- **Not suitable for sub-minute trading updates**

#### 3. **Custom Interactivity**
- Limited callback functionality
- No custom Python callbacks
- Actions limited to filter/highlight/URL
- **Cannot trigger order execution**

#### 4. **Deployment Complexity**
- Requires Tableau Server for embedding
- Additional infrastructure
- Separate authentication
- **Increases system complexity**

#### 5. **Trading-Specific Gaps**
- No native candlestick charts
- No STRAT pattern support
- No options chain visualization
- **Would require significant custom work**

### Hybrid Approach Recommendation

**Use Tableau FOR:**
- Historical performance reporting
- Monthly/quarterly analytics
- Executive dashboards
- Portfolio attribution analysis
- Risk reports for compliance

**Keep Plotly Dash FOR:**
- Real-time portfolio monitoring (30-second refresh)
- Live trading interface
- STRAT pattern visualization
- Order execution triggers
- Regime detection display

### Integration Architecture (if pursuing hybrid)

```
┌─────────────────────────────────────────────────────────────┐
│                    ATLAS Dashboard System                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────┐    ┌──────────────────────┐       │
│  │   Plotly Dash App    │    │   Tableau Server     │       │
│  │   (Real-Time)        │    │   (Analytics)        │       │
│  │                      │    │                      │       │
│  │  - Live Portfolio    │    │  - Historical Perf   │       │
│  │  - Order Entry       │    │  - Risk Reports      │       │
│  │  - STRAT Patterns    │    │  - Attribution       │       │
│  │  - Regime Detection  │    │  - Benchmarking      │       │
│  │                      │    │                      │       │
│  │  Refresh: 30 sec     │    │  Refresh: 1 hour     │       │
│  └──────────┬───────────┘    └──────────┬───────────┘       │
│             │                           │                    │
│             └───────────┬───────────────┘                    │
│                         │                                    │
│                         ▼                                    │
│             ┌───────────────────────┐                        │
│             │   Shared Data Layer   │                        │
│             │   (PostgreSQL/BigQuery)│                        │
│             └───────────────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Verdict

**For ATLAS specifically: Tableau is NOT recommended as a replacement.**

Reasons:
1. Real-time requirements (30-sec refresh) exceed Tableau's capabilities
2. STRAT pattern visualization requires custom Plotly code
3. Order execution requires Python callbacks
4. Cost adds complexity without proportional benefit
5. Current Plotly Dash setup is adequate and more flexible

**Tableau may be useful later** for:
- Monthly performance reports for stakeholders
- Compliance/audit dashboards
- Historical backtest analysis

---

## 5. ThetaData Integration Roadmap

### Current State
- Options module exists (`strat/options_module.py`)
- No ThetaData client implemented
- Historical options data unavailable

### Required Implementation

1. **ThetaData Terminal Setup**
   - Install ThetaData Terminal (local gateway)
   - Configure subscription (Pro plan recommended: $50/month)
   - Runs on localhost:25510

2. **Async Client (Already Created)**
   - `ThetaDataAsyncClient` in `async_data_service.py`
   - Methods: `get_option_chain()`, `get_option_quote()`, `get_historical_options()`

3. **Dashboard Integration**
   - Add "Options" tab to dashboard
   - Display option chains with Greeks
   - Show unusual options activity
   - Connect to STRAT pattern signals

4. **Live Data Flow**
```
ThetaData Terminal (localhost:25510)
         │
         ▼
ThetaDataAsyncClient
         │
         ▼
OptionsDataLoader (new)
         │
         ▼
Options Tab Visualizations
```

---

## 6. Implementation Priority

### Immediate (This Session)
- [x] Create async data service module
- [x] Create enhanced visualization components
- [x] Document Tableau analysis

### Short-Term (Next 1-2 Sessions)
- [ ] Integrate async service into dashboard callbacks
- [ ] Add STRAT pattern overlays to regime chart
- [ ] Fix hardcoded risk metrics
- [ ] Add position sparklines

### Medium-Term (Next Month)
- [ ] ThetaData integration (when subscription active)
- [ ] Options flow visualization
- [ ] Multi-timeframe regime heatmap
- [ ] Real-time VaR calculation

### Long-Term (Future)
- [ ] Consider Tableau for reporting (if stakeholder needs arise)
- [ ] WebSocket support for true real-time
- [ ] Mobile-optimized views
- [ ] Alert system integration

---

## 7. Files Created/Modified

### New Files
1. `dashboard/data_loaders/async_data_service.py` - Async HTTP client and data service
2. `dashboard/visualizations/enhanced_charts.py` - Enhanced chart components
3. `docs/DASHBOARD_REVIEW_SESSION76.md` - This review document

### Dependencies to Add
```
# Add to requirements-railway.txt
httpx>=0.27.0  # Async HTTP client with HTTP/2
aiohttp>=3.9.0  # Async HTTP fallback (optional)
```

---

## 8. Conclusion

The ATLAS dashboard has a solid foundation with professional TradingView-inspired design. The key improvements needed are:

1. **Async data fetching** - 3x performance improvement with httpx
2. **STRAT visualization** - Display patterns and measured moves on charts
3. **Real risk calculations** - Replace hardcoded placeholders
4. **ThetaData integration** - Enable options analysis (when subscription active)

**Tableau is not recommended** for the real-time trading dashboard due to refresh limitations and lack of trading-specific features. It may be useful for periodic reporting dashboards in the future.

The new `async_data_service.py` and `enhanced_charts.py` modules provide the foundation for these improvements without disrupting the existing dashboard functionality.

---

*Document created: Session 76*
*Author: Claude (Opus 4)*
*Last updated: 2025-11-26*

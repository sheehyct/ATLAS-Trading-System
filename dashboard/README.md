# ATLAS Trading Dashboard

Professional web-based dashboard for the ATLAS (Adaptive Trading with Layered Asset System) algorithmic trading platform.

## Overview

The ATLAS Dashboard provides real-time monitoring, historical analysis, and risk management visualization for the multi-layer trading architecture:

- **Layer 1**: Regime Detection using academic statistical jump models
- **Layer 2**: STRAT Pattern Recognition (future integration)
- **Layer 3**: Execution layer with equity and options strategies
- **Portfolio Management**: Live monitoring via Alpaca API

## Features

### 📊 Regime Detection Tab
- Interactive regime timeline with HMM background shading
- Feature evolution charts (Downside Deviation, Sortino Ratios)
- Regime statistics tables with performance metrics
- TradingView-quality visualizations with proper z-ordering

### 📈 Strategy Performance Tab
- Equity curves with drawdown visualization
- Rolling performance metrics (Sharpe, Sortino)
- Strategy comparison across market regimes
- Trade distribution analysis

### 💼 Live Portfolio Tab
- Real-time portfolio value and P&L
- Current positions table
- Portfolio heat gauge
- Auto-refresh every 30 seconds

### 🛡️ Risk Management Tab
- Portfolio heat monitoring
- Risk metrics dashboard
- Position allocation charts
- Risk limit indicators

## Installation

### Prerequisites

- Python 3.12+
- UV package manager (recommended) or pip

### Install Dependencies

```bash
# Using UV (recommended)
cd /home/user/ATLAS-Algorithmic-Trading-System-V1
uv sync

# Or using pip
pip install -e .
```

### New Dependencies Added

This dashboard requires the following additional packages (already added to `pyproject.toml`):

- `lightweight-charts>=2.0.0` - TradingView-quality interactive charts
- `mplfinance>=0.12.10b0` - Publication-quality static charts
- `hmmlearn>=0.3.0` - Hidden Markov Models for regime detection

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Alpaca API (for live trading)
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### Dashboard Settings

Edit `dashboard/config.py` to customize:

- Color schemes and themes
- Chart dimensions
- Refresh intervals
- Risk thresholds
- Performance metrics

## Running the Dashboard

### Quick Start

```bash
# From project root
python dashboard/app.py

# Or using UV
uv run python dashboard/app.py
```

Access the dashboard at: **http://localhost:8050**

### Production Deployment

For production, consider using:

```bash
# Gunicorn (recommended for production)
gunicorn dashboard.app:server -b 0.0.0.0:8050 --workers 4

# Or waitress (Windows-compatible)
waitress-serve --host=0.0.0.0 --port=8050 dashboard.app:server
```

## Architecture

### Directory Structure

```
dashboard/
├── app.py                      # Main Dash application
├── config.py                   # Configuration and constants
├── README.md                   # This file
│
├── callbacks/                  # Callback functions (future)
│   └── __init__.py
│
├── components/                 # UI components
│   ├── __init__.py
│   ├── header.py              # Top navigation
│   ├── regime_panel.py        # Regime detection tab
│   ├── strategy_panel.py      # Strategy performance tab
│   ├── portfolio_panel.py     # Live portfolio tab
│   └── risk_panel.py          # Risk management tab
│
├── data_loaders/              # Data integration layer
│   ├── __init__.py
│   ├── regime_loader.py       # Regime detection data
│   ├── backtest_loader.py     # VectorBT Pro results
│   └── live_loader.py         # Alpaca live data
│
├── visualizations/            # Plotly figure generators
│   ├── __init__.py
│   ├── regime_viz.py          # Regime visualizations
│   ├── performance_viz.py     # Performance charts
│   ├── trade_viz.py           # Trade analysis
│   └── risk_viz.py            # Risk charts
│
└── assets/                    # Static assets
    └── custom.css             # Custom styling
```

### Data Flow

1. **Data Loaders** fetch data from:
   - `regime/academic_jump_model.py` (regime classifications)
   - VectorBT Pro backtest results
   - Alpaca API (live trading data)

2. **Visualizations** create Plotly figures with:
   - Proper z-ordering (regime shading → grid → indicators → price → markers)
   - Professional color schemes
   - Mobile-friendly responsive design

3. **Components** assemble UI panels using:
   - Dash Bootstrap Components
   - Custom CSS styling
   - FontAwesome icons

4. **App** orchestrates:
   - Tab navigation
   - Callback management
   - Auto-refresh intervals

## Visualization Best Practices

### Color Scheme (TradingView Professional)

- **Bull/Upward**: `#00ff55` (bright green) or `#26A69A` (teal)
- **Bear/Downward**: `#ed4807` (red-orange) or `#EF5350` (softer red)
- **Background**: `#090008` (near black) or `#2E2E2E` (dark gray)
- **Grid**: `#333333` (subtle lines)

### Z-Ordering Rules

Charts follow proper layering (bottom to top):

1. **Regime shading** - alpha=0.2, layer='below'
2. **Grid lines** - automatic
3. **Indicators** - MA, Bollinger, etc.
4. **Price line** - main chart data
5. **Trade markers** - green/red triangles with white outlines
6. **Annotations** - text labels

### Trade Markers

- **Long entry**: Green triangle-up, white outline, size 12
- **Short entry**: Red triangle-down, white outline, size 12
- **Winning trades**: Full opacity, with P&L annotation
- **Losing trades**: 60% opacity, no annotation (reduces clutter)

## Development

### Current Status

✅ **Completed**:
- Complete directory structure
- Configuration system
- Main app with tab navigation
- All visualization modules
- Data loader placeholders
- UI components
- Custom CSS styling

⏳ **Next Steps**:
1. Connect regime_loader.py to actual `regime/academic_jump_model.py`
2. Implement VectorBT Pro portfolio loading in backtest_loader.py
3. Enable real Alpaca API in live_loader.py
4. Add lightweight-charts-python integration examples
5. Implement callback functions for interactive features
6. Add caching for performance optimization
7. Create mplfinance export functions for reports

### Adding New Visualizations

1. Create function in appropriate `visualizations/*.py` file
2. Add graph component in corresponding `components/*_panel.py`
3. Create callback in `app.py` to update the graph
4. Test with real data from data loaders

### Testing

```bash
# Run with debug mode (default)
python dashboard/app.py

# Check for import errors
python -c "from dashboard.app import app; print('OK')"
```

## Troubleshooting

### Common Issues

**Issue**: Module not found errors
- **Solution**: Ensure dashboard/ is in PYTHONPATH or run from project root

**Issue**: Empty visualizations
- **Solution**: Data loaders currently use placeholder data. Implement actual data loading.

**Issue**: Alpaca connection failed
- **Solution**: Check API credentials in .env file

**Issue**: Slow performance
- **Solution**: Reduce refresh interval, enable caching, or limit date ranges

### Logs

Check console output for:
- Data loader initialization status
- Callback execution errors
- API connection issues

## Performance Optimization

- **Caching**: Implement Redis or disk caching for expensive calculations
- **Lazy loading**: Only load data for active tab
- **Downsampling**: Use datashader for charts with >10k points
- **Background callbacks**: Use `dash.long_callback` for slow operations
- **Clientside callbacks**: Move simple updates to JavaScript

## Security

⚠️ **Important Security Notes**:

1. **Never commit API keys** to version control
2. **Use environment variables** for all credentials
3. **Enable authentication** before production deployment
4. **Use HTTPS** in production
5. **Implement rate limiting** for API calls

## Resources

- [Plotly Dash Documentation](https://dash.plotly.com/)
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/)
- [VectorBT Pro Documentation](../VectorBT%20Pro%20Official%20Documentation/)
- [Alpaca API Documentation](https://alpaca.markets/docs/)
- [lightweight-charts](https://tradingview.github.io/lightweight-charts/)

## License

See main project LICENSE file.

## Support

For issues or questions:
1. Check this README
2. Review PLOTLY_DASH_DASHBOARD_GUIDE.md
3. Check inline code documentation
4. Create an issue in the project repository

---

**Version**: 1.0
**Last Updated**: November 2025
**Status**: Foundation Complete - Ready for Data Integration

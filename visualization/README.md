# ATLAS Visualization Output

This directory contains all visualization outputs organized by category.

## Directory Structure

```
visualization/
├── regime_overlays/          # Price charts with regime background bands
│   ├── SPY_regime_overlay.html
│   ├── QQQ_regime_overlay.html
│   └── Combined_regime_comparison.html
│
├── performance_analysis/     # Regime performance metrics and charts
│   ├── SPY_regime_performance_tearsheet.html (QuantStats)
│   ├── SPY_regime_performance_comparison.html (4-panel charts)
│   ├── SPY_regime_performance_breakdown.csv
│   ├── QQQ_regime_performance_tearsheet.html
│   ├── QQQ_regime_performance_comparison.html
│   └── QQQ_regime_performance_breakdown.csv
│
├── strategy_backtests/       # Strategy backtest results
│   ├── regime_strategies_comparison.html
│   ├── regime_strategies_metrics.csv
│   ├── strategy_1_bull_only_tearsheet.html
│   ├── strategy_2_long_short_tearsheet.html
│   └── strategy_3_conservative_tearsheet.html
│
└── tradingview_exports/      # TradingView CSV exports and Pine Script
    ├── SPY_regimes_tradingview.csv
    ├── QQQ_regimes_tradingview.csv
    ├── regimes_combined_tradingview.csv
    └── ATLAS_TradingView_Indicator.pine

```

## File Descriptions

### Regime Overlays
Interactive Plotly charts showing price action with ATLAS regime detection as colored background bands.

- **Colors**: Red (CRASH), Orange (BEAR), Gray (NEUTRAL), Green (BULL)
- **Features**: VIX spike markers, hover info, zoom/pan
- **Use**: Share with colleagues, visual regime validation

### Performance Analysis
QuantStats tearsheets and performance breakdowns by market regime.

- **Tearsheets**: Professional HTML reports with 40+ metrics
- **Comparison Charts**: 4-panel analysis (cumulative returns, distributions, win rate, Sharpe)
- **CSV Breakdowns**: Metrics by regime (Sharpe, win rate, volatility, etc.)
- **Use**: Validate regime quality, identify best/worst regimes

### Strategy Backtests
Backtest results comparing regime-based strategies vs buy-and-hold.

- **Strategies Tested**:
  - Strategy 1: BULL-Only (long during BULL, cash otherwise)
  - Strategy 2: Long/Short (long BULL, short BEAR)
  - Strategy 3: Conservative (long BULL, 50% NEUTRAL)
  - Benchmark: Buy-and-Hold

- **Key Finding**: Simple regime on/off strategies underperform (see SESSION_36_VISUALIZATION_FINDINGS.md)
- **Use**: Diagnostic data for future strategy development

### TradingView Exports
CSV files and Pine Script for TradingView integration.

- **Note**: Direct CSV import not supported in TradingView web interface
- **Recommended**: Use Pure Pine Script approach instead (see below)
- **Alternative**: Use Plotly HTML visualizations (already professional-grade)

## Regenerating Visualizations

```bash
# Regime overlays
python visualize_regime_overlay.py

# Performance analysis
python regime_performance_analysis.py

# Strategy backtests
python backtest_regime_strategies.py

# TradingView exports
python export_regimes_for_tradingview.py
```

All outputs automatically saved to appropriate subdirectories.

## Sharing Files

All HTML files are self-contained and can be:
- Opened in any browser (Chrome, Firefox, Safari, Edge)
- Shared via email/file sharing
- Hosted on GitHub Pages, Netlify, or Cloudflare Pages (free)

## Notes

- All visualizations use professional color palette (Bootstrap-inspired)
- Charts are fully interactive (zoom, pan, hover)
- QuantStats tearsheets include comprehensive metrics
- CSV files compatible with Excel, Google Sheets, pandas

---

**Last Updated:** November 14, 2025 (Session 36)
**Total Files:** 17 (12 HTML, 4 CSV, 1 Pine Script)

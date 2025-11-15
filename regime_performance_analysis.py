"""
Regime-Aware Performance Analysis for ATLAS Trading System

Generates professional QuantStats tearsheets showing strategy performance
broken down by market regime (CRASH, BEAR, NEUTRAL, BULL).

Usage:
    python regime_performance_analysis.py

Output:
    - regime_performance_tearsheet.html (full QuantStats report)
    - regime_performance_breakdown.csv (metrics by regime)
    - regime_returns_comparison.html (interactive comparison chart)

This demonstrates the value of regime detection: different strategies
perform better in different regimes. Use this to validate strategy selection.
"""

import vectorbtpro as vbt
import pandas as pd
import numpy as np
import quantstats as qs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from regime.academic_jump_model import AcademicJumpModel
from regime.vix_acceleration import fetch_vix_data


def analyze_regime_performance(
    symbol='SPY',
    start_date='2020-01-01',
    end_date='2025-11-14',
    save_prefix='regime_performance'
):
    """
    Analyze buy-and-hold performance broken down by market regime.

    Parameters
    ----------
    symbol : str
        Stock symbol to analyze
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    save_prefix : str
        Prefix for saved files

    Returns
    -------
    dict
        Performance metrics by regime
    """
    print(f"\n{'='*70}")
    print(f"REGIME-AWARE PERFORMANCE ANALYSIS: {symbol}")
    print(f"{'='*70}")
    print(f"Date range: {start_date} to {end_date}\n")

    # Step 1: Fetch data
    print("[1/6] Fetching market data...")
    data = vbt.YFData.pull(symbol, start=start_date, end=end_date).get()
    vix_data = fetch_vix_data(start_date, end_date)
    print(f"   Fetched {len(data)} {symbol} bars")
    print(f"   Fetched {len(vix_data)} VIX bars")

    # Step 2: Run regime detection
    print("\n[2/6] Running regime detection...")
    model = AcademicJumpModel(lambda_penalty=1.5)
    train_size = int(len(data) * 0.6)
    model.fit(data.iloc[:train_size], n_starts=3, random_seed=42)

    lookback = min(252, len(data) - 100)
    regimes, _, _ = model.online_inference(data, lookback=lookback, vix_data=vix_data)
    print(f"   Classified {len(regimes)} days into regimes")

    # Align data with regime dates
    aligned_data = data.loc[regimes.index]

    # Step 3: Calculate returns by regime
    print("\n[3/6] Calculating returns by regime...")
    returns = aligned_data['Close'].pct_change().fillna(0)

    regime_metrics = {}

    for regime in ['CRASH', 'TREND_BEAR', 'TREND_NEUTRAL', 'TREND_BULL']:
        regime_mask = (regimes == regime)
        regime_returns = returns[regime_mask]

        if len(regime_returns) == 0:
            continue

        # Calculate metrics for this regime
        days = len(regime_returns)
        total_return = (1 + regime_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        volatility = regime_returns.std() * np.sqrt(252)
        sharpe = (annualized_return - 0.03) / volatility if volatility > 0 else 0

        # Win rate
        win_rate = (regime_returns > 0).sum() / len(regime_returns) if len(regime_returns) > 0 else 0

        # Max drawdown (simplified)
        cumulative = (1 + regime_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        regime_metrics[regime] = {
            'Days': days,
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Win Rate': win_rate,
            'Max Drawdown': max_dd,
            'Avg Daily Return': regime_returns.mean(),
            'Best Day': regime_returns.max(),
            'Worst Day': regime_returns.min()
        }

        print(f"\n   {regime}:")
        print(f"      Days: {days} ({days/len(regimes)*100:.1f}%)")
        print(f"      Total Return: {total_return:.2%}")
        print(f"      Annualized Return: {annualized_return:.2%}")
        print(f"      Sharpe Ratio: {sharpe:.2f}")
        print(f"      Win Rate: {win_rate:.1%}")

    # Step 4: Save regime breakdown CSV
    print("\n[4/6] Saving regime performance breakdown...")
    metrics_df = pd.DataFrame(regime_metrics).T
    metrics_csv = f'{save_prefix}_breakdown.csv'
    metrics_df.to_csv(metrics_csv)
    print(f"   Saved: {metrics_csv}")

    # Step 5: Create QuantStats tearsheet (overall performance)
    print("\n[5/6] Generating QuantStats tearsheet...")

    # Extend pandas for QuantStats
    qs.extend_pandas()

    # Full strategy returns (remove timezone for QuantStats compatibility)
    strategy_returns = returns.copy()
    strategy_returns.index = strategy_returns.index.tz_localize(None)

    # Benchmark: SPY if analyzing other symbols, or simple market return
    if symbol != 'SPY':
        benchmark_data = vbt.YFData.pull('SPY', start=start_date, end=end_date).get()
        benchmark_returns = benchmark_data['Close'].pct_change().reindex(returns.index).fillna(0)
        benchmark_returns.index = benchmark_returns.index.tz_localize(None)
    else:
        # For SPY, use None benchmark (skip benchmark comparison)
        benchmark_returns = None

    # Generate HTML tearsheet
    tearsheet_path = f'{save_prefix}_tearsheet.html'
    qs.reports.html(
        strategy_returns,
        benchmark=benchmark_returns,
        output=tearsheet_path,
        title=f'{symbol} Regime-Aware Performance Analysis'
    )
    print(f"   Saved: {tearsheet_path}")

    # Step 6: Create regime comparison visualization
    print("\n[6/6] Creating regime comparison chart...")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Cumulative Returns by Regime',
            'Daily Returns Distribution by Regime',
            'Win Rate by Regime',
            'Sharpe Ratio by Regime'
        ],
        specs=[
            [{'type': 'scatter'}, {'type': 'box'}],
            [{'type': 'bar'}, {'type': 'bar'}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    # Color scheme matching regime colors
    regime_colors_map = {
        'CRASH': '#FF0000',
        'TREND_BEAR': '#FFA500',
        'TREND_NEUTRAL': '#808080',
        'TREND_BULL': '#00CC00'
    }

    # Plot 1: Cumulative returns by regime
    for regime in ['CRASH', 'TREND_BEAR', 'TREND_NEUTRAL', 'TREND_BULL']:
        if regime not in regime_metrics:
            continue

        regime_mask = (regimes == regime)
        regime_returns_series = returns[regime_mask]
        cumulative = (1 + regime_returns_series).cumprod()

        fig.add_trace(
            go.Scatter(
                x=cumulative.index,
                y=cumulative.values,
                mode='lines',
                name=regime,
                line=dict(color=regime_colors_map[regime], width=2),
                legendgroup=regime,
                showlegend=True
            ),
            row=1, col=1
        )

    # Plot 2: Distribution box plots
    for regime in ['CRASH', 'TREND_BEAR', 'TREND_NEUTRAL', 'TREND_BULL']:
        if regime not in regime_metrics:
            continue

        regime_mask = (regimes == regime)
        regime_returns_series = returns[regime_mask]

        fig.add_trace(
            go.Box(
                y=regime_returns_series * 100,  # Convert to percentage
                name=regime,
                marker_color=regime_colors_map[regime],
                legendgroup=regime,
                showlegend=False
            ),
            row=1, col=2
        )

    # Plot 3: Win rate bar chart
    win_rates = [regime_metrics[r]['Win Rate'] * 100 for r in ['CRASH', 'TREND_BEAR', 'TREND_NEUTRAL', 'TREND_BULL'] if r in regime_metrics]
    regime_names = [r.replace('TREND_', '') for r in ['CRASH', 'TREND_BEAR', 'TREND_NEUTRAL', 'TREND_BULL'] if r in regime_metrics]
    colors = [regime_colors_map[r] for r in ['CRASH', 'TREND_BEAR', 'TREND_NEUTRAL', 'TREND_BULL'] if r in regime_metrics]

    fig.add_trace(
        go.Bar(
            x=regime_names,
            y=win_rates,
            marker_color=colors,
            showlegend=False,
            text=[f'{w:.1f}%' for w in win_rates],
            textposition='outside'
        ),
        row=2, col=1
    )

    # Plot 4: Sharpe ratio bar chart
    sharpes = [regime_metrics[r]['Sharpe Ratio'] for r in ['CRASH', 'TREND_BEAR', 'TREND_NEUTRAL', 'TREND_BULL'] if r in regime_metrics]

    fig.add_trace(
        go.Bar(
            x=regime_names,
            y=sharpes,
            marker_color=colors,
            showlegend=False,
            text=[f'{s:.2f}' for s in sharpes],
            textposition='outside'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{symbol} Performance Analysis by Regime<br><sub>ATLAS Academic Jump Model + VIX Acceleration (2021-2025)</sub>',
            font=dict(size=18, family='Arial, sans-serif')
        ),
        height=900,
        width=1400,
        template='plotly_white',
        font=dict(family='Arial, sans-serif', size=11),
        hovermode='closest'
    )

    # Update axes
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_yaxes(title_text='Cumulative Return', row=1, col=1)

    fig.update_xaxes(title_text='Regime', row=1, col=2)
    fig.update_yaxes(title_text='Daily Return (%)', row=1, col=2)

    fig.update_xaxes(title_text='Regime', row=2, col=1)
    fig.update_yaxes(title_text='Win Rate (%)', row=2, col=1)

    fig.update_xaxes(title_text='Regime', row=2, col=2)
    fig.update_yaxes(title_text='Sharpe Ratio', row=2, col=2)

    # Save comparison chart
    comparison_path = f'{save_prefix}_comparison.html'
    fig.write_html(comparison_path)
    print(f"   Saved: {comparison_path}")

    # Summary
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print(f"  1. {tearsheet_path} (QuantStats professional tearsheet)")
    print(f"  2. {metrics_csv} (regime performance metrics)")
    print(f"  3. {comparison_path} (regime comparison charts)")
    print("\nKey insights:")

    # Find best/worst regimes
    if regime_metrics:
        best_regime = max(regime_metrics.items(), key=lambda x: x[1]['Sharpe Ratio'])
        worst_regime = min(regime_metrics.items(), key=lambda x: x[1]['Sharpe Ratio'])

        print(f"  - Best Sharpe: {best_regime[0]} ({best_regime[1]['Sharpe Ratio']:.2f})")
        print(f"  - Worst Sharpe: {worst_regime[0]} ({worst_regime[1]['Sharpe Ratio']:.2f})")

        print(f"\n  - Strategy implication: Favor trades during {best_regime[0]} regimes")
        print(f"  - Risk management: Reduce exposure during {worst_regime[0]} regimes")

    print(f"{'='*70}\n")

    return regime_metrics


if __name__ == '__main__':
    # Analyze SPY performance by regime
    spy_metrics = analyze_regime_performance('SPY', '2020-01-01', '2025-11-14', 'SPY_regime_performance')

    # Optionally analyze QQQ as well
    print("\n" + "="*70)
    print("ANALYZING QQQ...")
    print("="*70)
    qqq_metrics = analyze_regime_performance('QQQ', '2020-01-01', '2025-11-14', 'QQQ_regime_performance')

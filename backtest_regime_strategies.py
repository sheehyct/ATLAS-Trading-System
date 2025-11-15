"""
ATLAS Regime-Based Trading Strategy Backtest

Compares regime-based trading strategies against SPY buy-and-hold benchmark:

Strategy 1: BULL-Only (Long only during BULL regimes, cash otherwise)
Strategy 2: Long/Short (Long BULL, Short BEAR, Cash CRASH/NEUTRAL)
Strategy 3: Conservative (Long BULL, 50% NEUTRAL, Cash BEAR/CRASH)
Benchmark: SPY Buy-and-Hold

Generates QuantStats tearsheets and comparison analysis.

Usage:
    python backtest_regime_strategies.py

Output:
    - strategy_1_bull_only_tearsheet.html
    - strategy_2_long_short_tearsheet.html
    - strategy_3_conservative_tearsheet.html
    - regime_strategies_comparison.html
    - regime_strategies_metrics.csv
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


def backtest_regime_strategies(
    symbol='SPY',
    start_date='2020-01-01',
    end_date='2025-11-14',
    initial_cash=10000,
    fees=0.001
):
    """
    Backtest multiple regime-based trading strategies.

    Parameters
    ----------
    symbol : str
        Symbol to trade (SPY, QQQ, etc.)
    start_date : str
        Backtest start date
    end_date : str
        Backtest end date
    initial_cash : float
        Initial capital
    fees : float
        Trading fees (0.001 = 0.1%)

    Returns
    -------
    dict
        Dictionary of portfolio results by strategy
    """
    print(f"\n{'='*70}")
    print(f"ATLAS REGIME-BASED STRATEGY BACKTEST: {symbol}")
    print(f"{'='*70}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Initial capital: ${initial_cash:,.0f}")
    print(f"Trading fees: {fees*100:.2f}%\n")

    # Step 1: Fetch data
    print("[1/6] Fetching market data...")
    data = vbt.YFData.pull(symbol, start=start_date, end=end_date).get()
    vix_data = fetch_vix_data(start_date, end_date)
    print(f"   Fetched {len(data)} {symbol} bars")

    # Step 2: Run regime detection
    print("\n[2/6] Running ATLAS regime detection...")
    model = AcademicJumpModel(lambda_penalty=1.5)
    train_size = int(len(data) * 0.6)
    model.fit(data.iloc[:train_size], n_starts=3, random_seed=42)

    lookback = min(252, len(data) - 100)
    regimes, _, _ = model.online_inference(data, lookback=lookback, vix_data=vix_data)
    print(f"   Classified {len(regimes)} days into regimes")

    # CRITICAL: Align data with regime dates for FAIR comparison
    # Both strategies and buy-and-hold must start on same date
    aligned_data = data.loc[regimes.index]
    aligned_regimes = regimes

    print(f"\n   CRITICAL - FAIR COMPARISON:")
    print(f"   All strategies start: {aligned_data.index[0].date()}")
    print(f"   SPY price at start: ${aligned_data['Close'].iloc[0]:.2f}")
    print(f"   SPY price at end: ${aligned_data['Close'].iloc[-1]:.2f}")
    print(f"   Buy-and-hold opportunity: {(aligned_data['Close'].iloc[-1] / aligned_data['Close'].iloc[0] - 1) * 100:.2f}%")
    print(f"   (No head start - all strategies use same start date)")

    # Regime distribution
    regime_counts = regimes.value_counts()
    print(f"\n   Regime Distribution:")
    for regime, count in regime_counts.items():
        pct = (count / len(regimes)) * 100
        print(f"      {regime}: {count} days ({pct:.1f}%)")

    # Step 3: Create strategy signals
    print("\n[3/6] Generating trading signals...")

    # Strategy 1: BULL-Only (Long during BULL, Cash otherwise)
    entries_s1 = (aligned_regimes == 'TREND_BULL')
    exits_s1 = (aligned_regimes != 'TREND_BULL')

    # Strategy 2: Long/Short (Long BULL, Short BEAR, Cash CRASH/NEUTRAL)
    entries_long_s2 = (aligned_regimes == 'TREND_BULL')
    exits_long_s2 = (aligned_regimes != 'TREND_BULL')
    entries_short_s2 = (aligned_regimes == 'TREND_BEAR')
    exits_short_s2 = (aligned_regimes != 'TREND_BEAR')

    # Strategy 3: Conservative (Long BULL, 50% NEUTRAL, Cash BEAR/CRASH)
    # We'll use custom size for this one
    size_s3 = pd.Series(0.0, index=aligned_regimes.index)
    size_s3[aligned_regimes == 'TREND_BULL'] = 1.0  # 100% long
    size_s3[aligned_regimes == 'TREND_NEUTRAL'] = 0.5  # 50% long
    size_s3[aligned_regimes == 'TREND_BEAR'] = 0.0  # Cash
    size_s3[aligned_regimes == 'CRASH'] = 0.0  # Cash

    print(f"   Strategy 1 (BULL-Only): {entries_s1.sum()} entries, {exits_s1.sum()} exits")
    print(f"      Time in market: {entries_s1.sum()}/{len(aligned_regimes)} days ({entries_s1.sum()/len(aligned_regimes)*100:.1f}%)")
    print(f"   Strategy 2 (Long/Short): {entries_long_s2.sum()} long entries, {entries_short_s2.sum()} short entries")
    print(f"      Time long: {entries_long_s2.sum()}/{len(aligned_regimes)} days ({entries_long_s2.sum()/len(aligned_regimes)*100:.1f}%)")
    print(f"      Time short: {entries_short_s2.sum()}/{len(aligned_regimes)} days ({entries_short_s2.sum()/len(aligned_regimes)*100:.1f}%)")
    print(f"   Strategy 3 (Conservative): Variable sizing based on regime")

    # Step 4: Run backtests
    print("\n[4/6] Running backtests...")

    # Strategy 1: BULL-Only
    pf_s1 = vbt.Portfolio.from_signals(
        close=aligned_data['Close'],
        entries=entries_s1,
        exits=exits_s1,
        init_cash=initial_cash,
        fees=fees,
        size=1.0,
        size_type='valuepercent',
        freq='1D'
    )

    # Strategy 2: Long/Short
    pf_s2 = vbt.Portfolio.from_signals(
        close=aligned_data['Close'],
        entries=entries_long_s2,
        exits=exits_long_s2,
        short_entries=entries_short_s2,
        short_exits=exits_short_s2,
        init_cash=initial_cash,
        fees=fees,
        size=1.0,
        size_type='valuepercent',
        freq='1D'
    )

    # Strategy 3: Conservative (using from_orders for variable sizing)
    orders = pd.DataFrame({
        'size': size_s3.diff().fillna(size_s3),  # Change in position
        'price': aligned_data['Close']
    })

    pf_s3 = vbt.Portfolio.from_orders(
        close=aligned_data['Close'],
        size=orders['size'],
        size_type='targetpercent',
        init_cash=initial_cash,
        fees=fees,
        freq='1D'
    )

    # Benchmark: Buy-and-Hold
    pf_benchmark = vbt.Portfolio.from_holding(
        close=aligned_data['Close'],
        init_cash=initial_cash,
        fees=fees,
        freq='1D'
    )

    print(f"   Strategy 1 final value: ${pf_s1.value.iloc[-1]:,.2f}")
    print(f"   Strategy 2 final value: ${pf_s2.value.iloc[-1]:,.2f}")
    print(f"   Strategy 3 final value: ${pf_s3.value.iloc[-1]:,.2f}")
    print(f"   Buy-and-Hold final value: ${pf_benchmark.value.iloc[-1]:,.2f}")

    # Step 5: Calculate metrics
    print("\n[5/6] Calculating performance metrics...")

    results = {
        'Strategy 1 (BULL-Only)': pf_s1,
        'Strategy 2 (Long/Short)': pf_s2,
        'Strategy 3 (Conservative)': pf_s3,
        'Buy-and-Hold': pf_benchmark
    }

    metrics_data = []

    for name, pf in results.items():
        total_return = pf.total_return.iloc[0] if hasattr(pf.total_return, 'iloc') else pf.total_return
        sharpe = pf.sharpe_ratio.iloc[0] if hasattr(pf.sharpe_ratio, 'iloc') else pf.sharpe_ratio
        sortino = pf.sortino_ratio.iloc[0] if hasattr(pf.sortino_ratio, 'iloc') else pf.sortino_ratio
        max_dd = pf.max_drawdown.iloc[0] if hasattr(pf.max_drawdown, 'iloc') else pf.max_drawdown
        calmar = pf.calmar_ratio.iloc[0] if hasattr(pf.calmar_ratio, 'iloc') else pf.calmar_ratio

        metrics_data.append({
            'Strategy': name,
            'Total Return': total_return,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Max Drawdown': max_dd,
            'Calmar Ratio': calmar,
            'Final Value': pf.value.iloc[-1]
        })

        print(f"\n   {name}:")
        print(f"      Total Return: {total_return:.2%}")
        print(f"      Sharpe Ratio: {sharpe:.2f}")
        print(f"      Max Drawdown: {max_dd:.2%}")

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv('regime_strategies_metrics.csv', index=False)
    print(f"\n   Saved metrics: regime_strategies_metrics.csv")

    # Step 6: Generate visualizations
    print("\n[6/6] Generating visualizations...")

    # Create comparison chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            '<b>Portfolio Value Comparison</b>',
            '<b>Drawdown Comparison</b>'
        ],
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )

    colors = ['#28A745', '#FD7E14', '#007BFF', '#6C757D']

    for idx, (name, pf) in enumerate(results.items()):
        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=pf.value.index,
                y=pf.value.values,
                mode='lines',
                name=name,
                line=dict(color=colors[idx], width=2.5),
                hovertemplate='%{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Drawdown
        drawdown = pf.drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values * 100,
                mode='lines',
                name=name,
                line=dict(color=colors[idx], width=2),
                showlegend=False,
                hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=(
                f'<b>{symbol} ATLAS Regime-Based Strategy Comparison</b><br>'
                f'<span style="font-size:12px; color:#6c757d;">Backtest: {start_date} to {end_date} | Initial Capital: ${initial_cash:,.0f}</span>'
            ),
            font=dict(size=20, family='Arial, sans-serif', color='#2C3E50'),
            x=0.5,
            xanchor='center'
        ),
        height=900,
        width=1500,
        plot_bgcolor='rgba(250, 250, 250, 1)',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=11, color='#2C3E50'),
        hovermode='x unified',
        margin=dict(l=80, r=80, t=120, b=80),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(255, 255, 255, 0.95)',
            bordercolor='rgba(0, 0, 0, 0.2)',
            borderwidth=1,
            font=dict(size=11)
        )
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0, 0, 0, 0.05)',
        showline=True,
        linewidth=1,
        linecolor='rgba(0, 0, 0, 0.2)'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0, 0, 0, 0.05)',
        showline=True,
        linewidth=1,
        linecolor='rgba(0, 0, 0, 0.2)'
    )

    fig.update_yaxes(title_text='Portfolio Value ($)', tickprefix='$', row=1, col=1)
    fig.update_yaxes(title_text='Drawdown (%)', ticksuffix='%', row=2, col=1)
    fig.update_xaxes(title_text='Date', row=2, col=1)

    # Save comparison chart
    fig.write_html('regime_strategies_comparison.html')
    print(f"   Saved comparison chart: regime_strategies_comparison.html")

    # Generate QuantStats tearsheets for each strategy
    qs.extend_pandas()

    for name, pf in results.items():
        if name == 'Buy-and-Hold':
            continue  # Skip benchmark (we'll use it for comparison)

        returns = pf.returns
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        returns.index = returns.index.tz_localize(None)

        benchmark_returns = pf_benchmark.returns
        if isinstance(benchmark_returns, pd.DataFrame):
            benchmark_returns = benchmark_returns.iloc[:, 0]
        benchmark_returns.index = benchmark_returns.index.tz_localize(None)

        filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').replace('/', '_')
        tearsheet_path = f'{filename}_tearsheet.html'

        qs.reports.html(
            returns,
            benchmark=benchmark_returns,
            output=tearsheet_path,
            title=f'{symbol} {name} vs Buy-and-Hold'
        )

        print(f"   Saved QuantStats tearsheet: {tearsheet_path}")

    # Summary
    print(f"\n{'='*70}")
    print("BACKTEST COMPLETE")
    print(f"{'='*70}")
    print(f"\nGenerated files:")
    print(f"  1. regime_strategies_comparison.html (comparison chart)")
    print(f"  2. regime_strategies_metrics.csv (performance metrics)")
    print(f"  3. strategy_1_bull_only_tearsheet.html (Strategy 1 tearsheet)")
    print(f"  4. strategy_2_long_short_tearsheet.html (Strategy 2 tearsheet)")
    print(f"  5. strategy_3_conservative_tearsheet.html (Strategy 3 tearsheet)")

    print(f"\n{'='*70}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*70}\n")
    print(metrics_df.to_string(index=False))

    # Find best strategy
    best_sharpe = metrics_df.loc[metrics_df['Sharpe Ratio'].idxmax()]
    best_return = metrics_df.loc[metrics_df['Total Return'].idxmax()]

    print(f"\n{'='*70}")
    print("KEY INSIGHTS")
    print(f"{'='*70}")
    print(f"\nBest Sharpe Ratio: {best_sharpe['Strategy']} ({best_sharpe['Sharpe Ratio']:.2f})")
    print(f"Best Total Return: {best_return['Strategy']} ({best_return['Total Return']:.2%})")

    # Compare to benchmark
    benchmark_row = metrics_df[metrics_df['Strategy'] == 'Buy-and-Hold'].iloc[0]
    for idx, row in metrics_df.iterrows():
        if row['Strategy'] != 'Buy-and-Hold':
            return_improvement = row['Total Return'] - benchmark_row['Total Return']
            sharpe_improvement = row['Sharpe Ratio'] - benchmark_row['Sharpe Ratio']

            print(f"\n{row['Strategy']} vs Buy-and-Hold:")
            print(f"  Return: {return_improvement:+.2%} ({'+' if return_improvement > 0 else ''}better)")
            print(f"  Sharpe: {sharpe_improvement:+.2f} ({'+' if sharpe_improvement > 0 else ''}better)")

    print(f"{'='*70}\n")

    return results


if __name__ == '__main__':
    # Run backtest
    results = backtest_regime_strategies(
        symbol='SPY',
        start_date='2020-01-01',
        end_date='2025-11-14',
        initial_cash=10000,
        fees=0.001
    )

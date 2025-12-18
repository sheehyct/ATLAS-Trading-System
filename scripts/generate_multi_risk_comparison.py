"""
STRAT Options Multi-Risk Comparison Report

Compares STRAT Options at different risk levels (2%, 5%, 10%) vs SPY Buy-and-Hold.
All strategies use SPY only for an apples-to-apples comparison.

Usage:
    uv run python scripts/generate_multi_risk_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
import quantstats as qs
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from scripts.generate_strat_options_quantstats import (
    load_trade_data, calculate_exit_dates, calculate_position_pnl,
    build_daily_returns, get_benchmark_returns, calculate_trade_metrics
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def generate_multi_strategy_html(all_strategies, comparison_stats, output_path, capital):
    """Generate custom HTML report with all strategies plotted together."""

    # Color scheme
    colors = {
        'STRAT 2%': '#2ecc71',    # Green
        'STRAT 5%': '#3498db',    # Blue
        'STRAT 10%': '#9b59b6',   # Purple
        'SPY Buy&Hold': '#e74c3c'  # Red
    }

    # 1. Cumulative Returns Chart
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    cumulative = (1 + all_strategies).cumprod()
    for col in cumulative.columns:
        ax1.plot(cumulative.index, cumulative[col], label=col, color=colors.get(col, '#333'), linewidth=2)
    ax1.set_title('Cumulative Returns: All Strategies vs SPY Buy-and-Hold', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Growth of $1')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color='black', linestyle='--', alpha=0.3)
    cumulative_chart = fig_to_base64(fig1)

    # 2. Drawdown Chart
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for col in all_strategies.columns:
        cum = (1 + all_strategies[col]).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max * 100
        ax2.fill_between(drawdown.index, drawdown, 0, alpha=0.3, label=col, color=colors.get(col, '#333'))
        ax2.plot(drawdown.index, drawdown, color=colors.get(col, '#333'), linewidth=1)
    ax2.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Drawdown %')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    drawdown_chart = fig_to_base64(fig2)

    # 3. Monthly Returns Comparison (bar chart for final year)
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    monthly = all_strategies.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    last_12 = monthly.tail(12)
    x = np.arange(len(last_12))
    width = 0.2
    for i, col in enumerate(last_12.columns):
        ax3.bar(x + i*width, last_12[col], width, label=col, color=colors.get(col, '#333'), alpha=0.8)
    ax3.set_title('Monthly Returns (Last 12 Months)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Return %')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels([d.strftime('%b %Y') for d in last_12.index], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    monthly_chart = fig_to_base64(fig3)

    # 4. Rolling Sharpe Ratio (252-day)
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    for col in all_strategies.columns:
        rolling_sharpe = all_strategies[col].rolling(252).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        ax4.plot(rolling_sharpe.index, rolling_sharpe, label=col, color=colors.get(col, '#333'), linewidth=1.5)
    ax4.set_title('Rolling Sharpe Ratio (252-day)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.axhline(y=1, color='green', linestyle='--', alpha=0.3)
    sharpe_chart = fig_to_base64(fig4)

    # Format comparison stats for HTML
    stats_html = comparison_stats.to_html(classes='stats-table', float_format=lambda x: f'{x:.2f}' if isinstance(x, float) else x)

    # Build HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>STRAT Options Multi-Risk Comparison</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 28px;
        }}
        .header p {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 25px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .chart-img {{
            width: 100%;
            border-radius: 5px;
        }}
        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .stats-table th, .stats-table td {{
            padding: 12px;
            text-align: right;
            border-bottom: 1px solid #eee;
        }}
        .stats-table th {{
            background: #f8f9fa;
            font-weight: 600;
            text-align: left;
        }}
        .stats-table tr:hover {{
            background: #f8f9fa;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-box {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .summary-box.green {{ border-top: 4px solid #2ecc71; }}
        .summary-box.blue {{ border-top: 4px solid #3498db; }}
        .summary-box.purple {{ border-top: 4px solid #9b59b6; }}
        .summary-box.red {{ border-top: 4px solid #e74c3c; }}
        .summary-box h3 {{
            margin: 0 0 15px 0;
            font-size: 14px;
            color: #666;
        }}
        .summary-box .value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .summary-box .label {{
            font-size: 12px;
            color: #999;
            margin-top: 5px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 20px 0;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 20px;
            height: 4px;
            border-radius: 2px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>STRAT Options Multi-Risk Comparison</h1>
        <p>SPY Only | Starting Capital: ${capital:,} | Jan 2020 - Dec 2024 | 263 Trades</p>
        <p>Comparing 2%, 5%, and 10% risk per trade vs SPY Buy-and-Hold</p>
    </div>

    <div class="summary-grid">
        <div class="summary-box green">
            <h3>STRAT 2% Risk</h3>
            <div class="value">{((1 + all_strategies['STRAT 2%']).prod() - 1) * 100:.1f}%</div>
            <div class="label">Conservative</div>
        </div>
        <div class="summary-box blue">
            <h3>STRAT 5% Risk</h3>
            <div class="value">{((1 + all_strategies['STRAT 5%']).prod() - 1) * 100:.1f}%</div>
            <div class="label">Moderate</div>
        </div>
        <div class="summary-box purple">
            <h3>STRAT 10% Risk</h3>
            <div class="value">{((1 + all_strategies['STRAT 10%']).prod() - 1) * 100:.1f}%</div>
            <div class="label">Aggressive</div>
        </div>
        <div class="summary-box red">
            <h3>SPY Buy & Hold</h3>
            <div class="value">{((1 + all_strategies['SPY Buy&Hold']).prod() - 1) * 100:.1f}%</div>
            <div class="label">Benchmark</div>
        </div>
    </div>

    <div class="card">
        <h2>Cumulative Returns</h2>
        <div class="legend">
            <div class="legend-item"><div class="legend-color" style="background: #2ecc71;"></div> STRAT 2%</div>
            <div class="legend-item"><div class="legend-color" style="background: #3498db;"></div> STRAT 5%</div>
            <div class="legend-item"><div class="legend-color" style="background: #9b59b6;"></div> STRAT 10%</div>
            <div class="legend-item"><div class="legend-color" style="background: #e74c3c;"></div> SPY Buy&Hold</div>
        </div>
        <img src="data:image/png;base64,{cumulative_chart}" class="chart-img" alt="Cumulative Returns">
    </div>

    <div class="card">
        <h2>Drawdown Analysis</h2>
        <img src="data:image/png;base64,{drawdown_chart}" class="chart-img" alt="Drawdown">
    </div>

    <div class="card">
        <h2>Rolling Sharpe Ratio (252-day)</h2>
        <img src="data:image/png;base64,{sharpe_chart}" class="chart-img" alt="Rolling Sharpe">
    </div>

    <div class="card">
        <h2>Monthly Returns (Last 12 Months)</h2>
        <img src="data:image/png;base64,{monthly_chart}" class="chart-img" alt="Monthly Returns">
    </div>

    <div class="card">
        <h2>QuantStats Comparison Metrics</h2>
        {stats_html}
    </div>

    <div class="card">
        <h2>Risk Level Guide</h2>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #e8f5e9;">
                <td style="padding: 15px; border: 1px solid #ddd; width: 20%;"><strong>2% Risk</strong><br>~$500/trade</td>
                <td style="padding: 15px; border: 1px solid #ddd;">
                    <strong>Conservative</strong> - Best for capital preservation and learning. Max drawdown limited, slower growth, lower stress.
                </td>
            </tr>
            <tr style="background: #e3f2fd;">
                <td style="padding: 15px; border: 1px solid #ddd;"><strong>5% Risk</strong><br>~$1,250/trade</td>
                <td style="padding: 15px; border: 1px solid #ddd;">
                    <strong>Moderate</strong> - Balanced growth and risk management. Good for experienced traders seeking steady returns.
                </td>
            </tr>
            <tr style="background: #f3e5f5;">
                <td style="padding: 15px; border: 1px solid #ddd;"><strong>10% Risk</strong><br>~$2,500/trade</td>
                <td style="padding: 15px; border: 1px solid #ddd;">
                    <strong>Aggressive</strong> - Maximum growth potential. Higher drawdowns possible, requires strong conviction.
                </td>
            </tr>
        </table>
    </div>

    <p style="text-align: center; color: #999; margin-top: 30px;">
        Generated by ATLAS STRAT Options Backtest | Data: Jan 2020 - Dec 2024
    </p>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


def main():
    print('=' * 80)
    print('STRAT OPTIONS MULTI-RISK COMPARISON')
    print('SPY Only: 2% vs 5% vs 10% Risk vs SPY Buy-and-Hold')
    print('=' * 80)

    # Configuration
    CAPITAL = 25000
    TIMEFRAMES = ['1D', '1W', '1M']
    SYMBOLS = ['SPY']
    MAX_HOLDING = {'1D': 18, '1W': 4, '1M': 2}
    RISK_LEVELS = [0.02, 0.05, 0.10]  # 2%, 5%, 10%

    # Load trades once
    print('\n[1/5] Loading SPY trade data...')
    trades_base = load_trade_data(timeframes=TIMEFRAMES, symbols=SYMBOLS)
    trades_base = calculate_exit_dates(trades_base, MAX_HOLDING)
    print(f'  Loaded {len(trades_base)} SPY trades')

    # Generate returns for each risk level
    all_returns = {}
    all_metrics = {}

    print('\n[2/5] Calculating returns for each risk level...')
    for risk_pct in RISK_LEVELS:
        label = f'STRAT_{int(risk_pct*100)}pct'
        print(f'  Processing {label}...')

        # Calculate P&L for this risk level
        trades = trades_base.copy()
        trades = calculate_position_pnl(trades, capital=CAPITAL, risk_pct=risk_pct)

        # Calculate metrics
        metrics = calculate_trade_metrics(trades, CAPITAL)
        all_metrics[label] = metrics

        # Build daily returns
        returns = build_daily_returns(trades, CAPITAL)
        returns.name = label
        all_returns[label] = returns

        print(f'    Total P&L: ${metrics["total_pnl"]:,.0f} ({metrics["total_return_pct"]:.1f}%)')

    # Get SPY benchmark
    print('\n[3/5] Downloading SPY benchmark...')
    first_returns = list(all_returns.values())[0]
    spy_returns = get_benchmark_returns(
        start_date=first_returns.index.min().strftime('%Y-%m-%d'),
        end_date=first_returns.index.max().strftime('%Y-%m-%d')
    )
    spy_returns.name = 'SPY_BuyHold'

    # Align all returns to common dates
    print('\n[4/5] Aligning returns...')
    common_start = max([r.index.min() for r in all_returns.values()] + [spy_returns.index.min()])
    common_end = min([r.index.max() for r in all_returns.values()] + [spy_returns.index.max()])

    aligned_returns = {}
    for name, returns in all_returns.items():
        aligned = returns[common_start:common_end]
        aligned_returns[name] = aligned

    spy_aligned = spy_returns[common_start:common_end]
    spy_aligned = spy_aligned.reindex(list(aligned_returns.values())[0].index).fillna(0)

    print(f'  Date range: {common_start.date()} to {common_end.date()}')
    print(f'  {len(list(aligned_returns.values())[0])} trading days')

    # Create combined DataFrame
    print('\n[5/5] Generating combined report...')
    combined_df = pd.DataFrame(aligned_returns)
    combined_df['SPY_BuyHold'] = spy_aligned

    # Generate HTML report
    output_path = Path('reports/strat_options_multi_risk_comparison.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extend pandas for QuantStats
    qs.extend_pandas()

    # Create combined DataFrame with all strategies + benchmark
    all_strategies = pd.DataFrame({
        'STRAT 2%': aligned_returns['STRAT_2pct'],
        'STRAT 5%': aligned_returns['STRAT_5pct'],
        'STRAT 10%': aligned_returns['STRAT_10pct'],
        'SPY Buy&Hold': spy_aligned
    })

    # Generate comparison stats table using individual stats
    print('  Generating comparison statistics...')
    stats_list = []
    for col in all_strategies.columns:
        stats = {
            'Strategy': col,
            'Total Return': f"{((1 + all_strategies[col]).prod() - 1) * 100:.1f}%",
            'CAGR': f"{qs.stats.cagr(all_strategies[col]) * 100:.1f}%",
            'Sharpe': f"{qs.stats.sharpe(all_strategies[col]):.2f}",
            'Sortino': f"{qs.stats.sortino(all_strategies[col]):.2f}",
            'Max Drawdown': f"{qs.stats.max_drawdown(all_strategies[col]) * 100:.1f}%",
            'Volatility': f"{qs.stats.volatility(all_strategies[col]) * 100:.1f}%",
            'Calmar': f"{qs.stats.calmar(all_strategies[col]):.2f}",
        }
        stats_list.append(stats)
    comparison_stats = pd.DataFrame(stats_list).set_index('Strategy').T

    # Generate custom HTML report with all strategies
    generate_multi_strategy_html(
        all_strategies,
        comparison_stats,
        output_path,
        CAPITAL
    )

    print(f'  Report generated: {output_path}')

    # Calculate SPY buy-and-hold metrics
    spy_total_return = (1 + spy_aligned).prod() - 1
    spy_final = CAPITAL * (1 + spy_total_return)

    metrics_2 = all_metrics['STRAT_2pct']
    metrics_5 = all_metrics['STRAT_5pct']
    metrics_10 = all_metrics['STRAT_10pct']

    # Append custom trade metrics comparison
    print('  Appending trade metrics comparison...')
    append_comparison_table(output_path, metrics_2, metrics_5, metrics_10, spy_final, spy_total_return, CAPITAL)

    # Print summary
    print('\n' + '=' * 80)
    print('COMPLETE')
    print('=' * 80)

    print('\n--- SUMMARY ---')
    print(f'{"Strategy":<20} {"Final Equity":>15} {"Return":>10} {"P&L":>12}')
    print('-' * 60)
    print(f'{"STRAT 2% Risk":<20} ${metrics_2["final_equity"]:>14,.0f} {metrics_2["total_return_pct"]:>9.1f}% ${metrics_2["total_pnl"]:>11,.0f}')
    print(f'{"STRAT 5% Risk":<20} ${metrics_5["final_equity"]:>14,.0f} {metrics_5["total_return_pct"]:>9.1f}% ${metrics_5["total_pnl"]:>11,.0f}')
    print(f'{"STRAT 10% Risk":<20} ${metrics_10["final_equity"]:>14,.0f} {metrics_10["total_return_pct"]:>9.1f}% ${metrics_10["total_pnl"]:>11,.0f}')
    print(f'{"SPY Buy & Hold":<20} ${spy_final:>14,.0f} {spy_total_return*100:>9.1f}% ${spy_final - CAPITAL:>11,.0f}')

    print(f'\nReport saved to: {output_path}')


def append_comparison_table(output_path, metrics_2, metrics_5, metrics_10, spy_final, spy_total_return, capital):
    """Append comparison table to HTML report."""

    html_addition = f'''
    <div style="margin: 40px auto; max-width: 1100px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <h2 style="border-bottom: 2px solid #333; padding-bottom: 10px;">Risk Level Comparison Summary</h2>

        <p style="color: #666; margin-bottom: 20px;">
            Starting Capital: $25,000 | SPY Only | Jan 2020 - Dec 2024 | 263 Trades
        </p>

        <h3>Performance by Risk Level</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <tr style="background: #f8f9fa;">
                <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">Metric</th>
                <th style="padding: 12px; border: 1px solid #ddd; text-align: right;">2% Risk</th>
                <th style="padding: 12px; border: 1px solid #ddd; text-align: right;">5% Risk</th>
                <th style="padding: 12px; border: 1px solid #ddd; text-align: right;">10% Risk</th>
                <th style="padding: 12px; border: 1px solid #ddd; text-align: right; background: #e3f2fd;">SPY B&H</th>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Final Equity</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">${metrics_2['final_equity']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">${metrics_5['final_equity']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; font-weight: bold; color: #28a745;">${metrics_10['final_equity']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; background: #e3f2fd;">${spy_final:,.0f}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Total Return</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_2['total_return_pct']:.1f}%</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_5['total_return_pct']:.1f}%</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; font-weight: bold; color: #28a745;">{metrics_10['total_return_pct']:.1f}%</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; background: #e3f2fd;">{spy_total_return*100:.1f}%</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;"><strong>Total P&L</strong></td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">${metrics_2['total_pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">${metrics_5['total_pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; font-weight: bold; color: #28a745;">${metrics_10['total_pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; background: #e3f2fd;">${spy_final - capital:,.0f}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">Win Rate</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_2['win_rate']:.1%}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_5['win_rate']:.1%}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_10['win_rate']:.1%}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; background: #e3f2fd;">N/A</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">Profit Factor</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_2['profit_factor']:.2f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_5['profit_factor']:.2f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_10['profit_factor']:.2f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; background: #e3f2fd;">N/A</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">Avg Win</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: #28a745;">${metrics_2['avg_win']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: #28a745;">${metrics_5['avg_win']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: #28a745;">${metrics_10['avg_win']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; background: #e3f2fd;">N/A</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">Avg Loss</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: #dc3545;">${metrics_2['avg_loss']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: #dc3545;">${metrics_5['avg_loss']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: #dc3545;">${metrics_10['avg_loss']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; background: #e3f2fd;">N/A</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">Max Win Streak</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_2['max_win_streak']} trades</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_5['max_win_streak']} trades</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_10['max_win_streak']} trades</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; background: #e3f2fd;">N/A</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">Max Loss Streak</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_2['max_loss_streak']} trades</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_5['max_loss_streak']} trades</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_10['max_loss_streak']} trades</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; background: #e3f2fd;">N/A</td>
            </tr>
        </table>

        <h3 style="margin-top: 30px;">CALLs vs PUTs Performance</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <tr style="background: #f8f9fa;">
                <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">Direction</th>
                <th style="padding: 12px; border: 1px solid #ddd; text-align: right;">Trades</th>
                <th style="padding: 12px; border: 1px solid #ddd; text-align: right;">2% P&L</th>
                <th style="padding: 12px; border: 1px solid #ddd; text-align: right;">5% P&L</th>
                <th style="padding: 12px; border: 1px solid #ddd; text-align: right;">10% P&L</th>
                <th style="padding: 12px; border: 1px solid #ddd; text-align: right;">Win Rate</th>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">CALLs (Bullish)</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_2['call_count']}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: {'#28a745' if metrics_2['call_pnl'] > 0 else '#dc3545'};">${metrics_2['call_pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: {'#28a745' if metrics_5['call_pnl'] > 0 else '#dc3545'};">${metrics_5['call_pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: {'#28a745' if metrics_10['call_pnl'] > 0 else '#dc3545'};">${metrics_10['call_pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_2['call_win_rate']:.1%}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">PUTs (Bearish)</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_2['put_count']}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: {'#28a745' if metrics_2['put_pnl'] > 0 else '#dc3545'};">${metrics_2['put_pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: {'#28a745' if metrics_5['put_pnl'] > 0 else '#dc3545'};">${metrics_5['put_pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: {'#28a745' if metrics_10['put_pnl'] > 0 else '#dc3545'};">${metrics_10['put_pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{metrics_2['put_win_rate']:.1%}</td>
            </tr>
        </table>

        <h3 style="margin-top: 30px;">Best Pattern: 2-2 Up (Bullish Reversal)</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <tr style="background: #f8f9fa;">
                <th style="padding: 12px; border: 1px solid #ddd; text-align: left;">Risk Level</th>
                <th style="padding: 12px; border: 1px solid #ddd; text-align: right;">Trades</th>
                <th style="padding: 12px; border: 1px solid #ddd; text-align: right;">P&L</th>
                <th style="padding: 12px; border: 1px solid #ddd; text-align: right;">Win Rate</th>
            </tr>
'''

    # Add pattern rows for 2-2 Up (best performer)
    for label, metrics in [('2%', metrics_2), ('5%', metrics_5), ('10%', metrics_10)]:
        if '2-2 Up' in metrics['by_pattern']:
            p = metrics['by_pattern']['2-2 Up']
            html_addition += f'''
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd;">{label} Risk</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{p['count']}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right; color: #28a745;">${p['pnl']:,.0f}</td>
                <td style="padding: 10px; border: 1px solid #ddd; text-align: right;">{p['win_rate']:.1%}</td>
            </tr>'''

    html_addition += '''
        </table>

        <h3 style="margin-top: 30px;">Risk-Adjusted Scaling Guide</h3>
        <table style="width: 100%; border-collapse: collapse; margin: 10px 0;">
            <tr style="background: #e8f5e9;">
                <td style="padding: 15px; border: 1px solid #ddd; width: 25%;"><strong>2% Risk</strong><br>~$500/trade</td>
                <td style="padding: 15px; border: 1px solid #ddd;">
                    <strong>Conservative</strong> - Best for capital preservation and learning.<br>
                    Max drawdown limited, slower growth, lower stress.
                </td>
            </tr>
            <tr style="background: #fff3e0;">
                <td style="padding: 15px; border: 1px solid #ddd;"><strong>5% Risk</strong><br>~$1,250/trade</td>
                <td style="padding: 15px; border: 1px solid #ddd;">
                    <strong>Moderate</strong> - Balanced growth and risk management.<br>
                    Good for experienced traders seeking steady returns.
                </td>
            </tr>
            <tr style="background: #ffebee;">
                <td style="padding: 15px; border: 1px solid #ddd;"><strong>10% Risk</strong><br>~$2,500/trade</td>
                <td style="padding: 15px; border: 1px solid #ddd;">
                    <strong>Aggressive</strong> - Maximum growth potential.<br>
                    Higher drawdowns possible, requires strong conviction.
                </td>
            </tr>
        </table>

        <p style="margin-top: 30px; color: #666; font-size: 12px;">
            Generated by ATLAS STRAT Options Backtest | Data: Jan 2020 - Dec 2024 | 263 SPY Trades
        </p>
    </div>
    '''

    # Read and modify HTML
    with open(output_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    html_content = html_content.replace('</body>', html_addition + '</body>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == '__main__':
    main()

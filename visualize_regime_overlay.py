"""
Regime Overlay Visualization for ATLAS Trading System

Creates interactive Plotly charts showing price action with ATLAS regime detection
background bands for SPY and QQQ. Demonstrates academic jump model + VIX acceleration
layer working together to identify market regimes.

Usage:
    python visualize_regime_overlay.py

Output:
    - SPY_regime_overlay.html (interactive chart)
    - QQQ_regime_overlay.html (interactive chart)
    - Combined_regime_comparison.html (side-by-side)

Share these HTML files to demonstrate regime detection to other traders.
"""

import vectorbtpro as vbt
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from regime.academic_jump_model import AcademicJumpModel
from regime.vix_acceleration import fetch_vix_data


def plot_regime_overlay(
    symbol='SPY',
    start_date='2020-01-01',
    end_date='2025-11-14',
    save_path=None
):
    """
    Create interactive Plotly chart with regime background bands.

    Parameters
    ----------
    symbol : str
        Stock symbol (SPY, QQQ, etc.)
    start_date : str
        Start date for chart (YYYY-MM-DD)
    end_date : str
        End date for chart (YYYY-MM-DD)
    save_path : str, optional
        Path to save HTML file (default: {symbol}_regime_overlay.html)

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Interactive plotly figure
    """
    print(f"\n=== Regime Overlay Visualization: {symbol} ===")
    print(f"Date range: {start_date} to {end_date}")

    # Fetch price data
    print(f"\n[1/5] Fetching {symbol} data...")
    data = vbt.YFData.pull(symbol, start=start_date, end=end_date).get()
    print(f"   Fetched {len(data)} bars")

    # Fetch VIX data
    print("\n[2/5] Fetching VIX data...")
    vix_data = fetch_vix_data(start_date, end_date)
    print(f"   Fetched {len(vix_data)} VIX bars")

    # Initialize and fit academic model
    print("\n[3/5] Fitting academic jump model...")
    model = AcademicJumpModel(lambda_penalty=1.5)

    # Use first 60% of data for training
    train_size = int(len(data) * 0.6)
    train_data = data.iloc[:train_size]

    model.fit(train_data, n_starts=3, random_seed=42)
    print(f"   Model fitted on {len(train_data)} training bars")

    # Run online inference with VIX acceleration
    print("\n[4/5] Running regime detection (Academic + VIX)...")
    lookback = min(1000, len(data) - 100)  # Adaptive lookback
    regimes, lambda_series, theta_df = model.online_inference(
        data,
        lookback=lookback,
        vix_data=vix_data
    )
    print(f"   Detected {len(regimes)} regime classifications")

    # Count regime distribution
    regime_counts = regimes.value_counts()
    print("\n   Regime Distribution:")
    for regime, count in regime_counts.items():
        pct = (count / len(regimes)) * 100
        print(f"      {regime}: {count} days ({pct:.1f}%)")

    # Create Plotly figure
    print("\n[5/5] Creating interactive Plotly chart...")
    fig = go.Figure()

    # Define regime colors (transparency via rgba)
    regime_colors = {
        'CRASH': 'rgba(255, 0, 0, 0.2)',      # Red
        'TREND_BEAR': 'rgba(255, 165, 0, 0.2)',  # Orange
        'TREND_NEUTRAL': 'rgba(128, 128, 128, 0.15)',  # Gray
        'TREND_BULL': 'rgba(0, 255, 0, 0.2)'   # Green
    }

    # Add regime background bands
    for regime_name, color in regime_colors.items():
        regime_mask = (regimes == regime_name)

        if regime_mask.sum() == 0:
            continue  # Skip if regime never appears

        # Get regime periods
        regime_dates = regimes[regime_mask].index

        # Create continuous segments (handle gaps)
        segments = []
        current_segment = [regime_dates[0]]

        for i in range(1, len(regime_dates)):
            # Check if dates are consecutive (within 5 days for market gaps)
            days_diff = (regime_dates[i] - regime_dates[i-1]).days
            if days_diff <= 5:
                current_segment.append(regime_dates[i])
            else:
                segments.append(current_segment)
                current_segment = [regime_dates[i]]

        segments.append(current_segment)  # Add last segment

        # Plot each continuous segment
        for segment in segments:
            if len(segment) < 2:
                continue  # Skip single-day segments

            # Get price range for this segment
            segment_data = data.loc[segment]
            y_min = segment_data['Low'].min() * 0.98
            y_max = segment_data['High'].max() * 1.02

            fig.add_trace(go.Scatter(
                x=[segment[0], segment[0], segment[-1], segment[-1], segment[0]],
                y=[y_min, y_max, y_max, y_min, y_min],
                fill='toself',
                fillcolor=color,
                line=dict(width=0),
                showlegend=True if segment == segments[0] else False,
                name=regime_name,
                hoverinfo='skip'
            ))

    # Add price line (on top of regime bands)
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name=f'{symbol} Price',
        line=dict(color='black', width=2),
        hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
    ))

    # Add VIX spike markers (flash crash detection)
    from regime.vix_acceleration import detect_vix_spike
    vix_aligned = vix_data.reindex(data.index, method='ffill')
    vix_spikes = detect_vix_spike(vix_aligned)

    if vix_spikes.sum() > 0:
        spike_dates = vix_spikes[vix_spikes].index
        spike_prices = data.loc[spike_dates, 'Close']

        fig.add_trace(go.Scatter(
            x=spike_dates,
            y=spike_prices,
            mode='markers',
            name='VIX Flash Crash',
            marker=dict(
                color='red',
                size=12,
                symbol='x',
                line=dict(width=2, color='darkred')
            ),
            hovertemplate='VIX Spike<br>%{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=f'{symbol} Price with ATLAS Regime Detection<br><sub>Academic Jump Model + VIX Acceleration Layer (2020-2025)</sub>',
            font=dict(size=20)
        ),
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=700,
        width=1400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        ),
        annotations=[
            dict(
                text=(
                    f"Regime Distribution: "
                    f"CRASH {regime_counts.get('CRASH', 0)} days, "
                    f"BEAR {regime_counts.get('TREND_BEAR', 0)} days, "
                    f"NEUTRAL {regime_counts.get('TREND_NEUTRAL', 0)} days, "
                    f"BULL {regime_counts.get('TREND_BULL', 0)} days"
                ),
                xref="paper", yref="paper",
                x=0.5, y=-0.15,
                showarrow=False,
                font=dict(size=12, color='gray'),
                xanchor='center'
            )
        ]
    )

    # Save to HTML
    if save_path is None:
        save_path = f'{symbol}_regime_overlay.html'

    fig.write_html(save_path)
    print(f"\n[SUCCESS] Chart saved to: {save_path}")

    return fig


def create_comparison_chart(symbols=['SPY', 'QQQ'], start_date='2020-01-01', end_date='2025-11-14'):
    """
    Create side-by-side comparison of multiple symbols with regime overlays.

    Parameters
    ----------
    symbols : list
        List of symbols to compare
    start_date : str
        Start date
    end_date : str
        End date

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Combined comparison figure
    """
    print(f"\n=== Creating Comparison Chart: {', '.join(symbols)} ===")

    # Create subplots
    fig = make_subplots(
        rows=len(symbols),
        cols=1,
        subplot_titles=[f'{sym} with ATLAS Regime Detection' for sym in symbols],
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # Fetch VIX data once (shared across symbols)
    print("\nFetching VIX data...")
    vix_data = fetch_vix_data(start_date, end_date)

    regime_colors = {
        'CRASH': 'rgba(255, 0, 0, 0.2)',
        'TREND_BEAR': 'rgba(255, 165, 0, 0.2)',
        'TREND_NEUTRAL': 'rgba(128, 128, 128, 0.15)',
        'TREND_BULL': 'rgba(0, 255, 0, 0.2)'
    }

    for idx, symbol in enumerate(symbols, start=1):
        print(f"\n[{idx}/{len(symbols)}] Processing {symbol}...")

        # Fetch data
        data = vbt.YFData.pull(symbol, start=start_date, end=end_date).get()

        # Run regime detection
        model = AcademicJumpModel(lambda_penalty=1.5)
        train_size = int(len(data) * 0.6)
        model.fit(data.iloc[:train_size], n_starts=3, random_seed=42)

        lookback = min(1000, len(data) - 100)
        regimes, _, _ = model.online_inference(data, lookback=lookback, vix_data=vix_data)

        # Add regime bands
        for regime_name, color in regime_colors.items():
            regime_mask = (regimes == regime_name)
            if regime_mask.sum() == 0:
                continue

            regime_dates = regimes[regime_mask].index
            segments = []
            current_segment = [regime_dates[0]]

            for i in range(1, len(regime_dates)):
                if (regime_dates[i] - regime_dates[i-1]).days <= 5:
                    current_segment.append(regime_dates[i])
                else:
                    segments.append(current_segment)
                    current_segment = [regime_dates[i]]
            segments.append(current_segment)

            for seg_idx, segment in enumerate(segments):
                if len(segment) < 2:
                    continue

                segment_data = data.loc[segment]
                y_min = segment_data['Low'].min() * 0.98
                y_max = segment_data['High'].max() * 1.02

                fig.add_trace(go.Scatter(
                    x=[segment[0], segment[0], segment[-1], segment[-1], segment[0]],
                    y=[y_min, y_max, y_max, y_min, y_min],
                    fill='toself',
                    fillcolor=color,
                    line=dict(width=0),
                    showlegend=(idx == 1 and seg_idx == 0),  # Only show legend once
                    name=regime_name,
                    legendgroup=regime_name,
                    hoverinfo='skip'
                ), row=idx, col=1)

        # Add price line
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['Close'],
            mode='lines',
            name=symbol,
            line=dict(color='black', width=2),
            showlegend=False,
            hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
        ), row=idx, col=1)

        # Update y-axis title
        fig.update_yaxes(title_text=f'{symbol} Price ($)', row=idx, col=1)

    # Update layout
    fig.update_layout(
        title=dict(
            text='ATLAS Regime Detection Comparison<br><sub>Academic Jump Model + VIX Acceleration (2020-2025)</sub>',
            font=dict(size=20)
        ),
        height=400 * len(symbols),
        width=1400,
        hovermode='x unified',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    fig.update_xaxes(title_text='Date', row=len(symbols), col=1)

    # Save
    save_path = 'Combined_regime_comparison.html'
    fig.write_html(save_path)
    print(f"\n[SUCCESS] Comparison chart saved to: {save_path}")

    return fig


if __name__ == '__main__':
    print("=" * 70)
    print("ATLAS Regime Overlay Visualization")
    print("Academic Jump Model + VIX Acceleration Layer")
    print("=" * 70)

    # Configuration
    START_DATE = '2020-01-01'
    END_DATE = '2025-11-14'

    # Generate individual charts
    print("\n" + "=" * 70)
    print("GENERATING INDIVIDUAL CHARTS")
    print("=" * 70)

    spy_fig = plot_regime_overlay('SPY', START_DATE, END_DATE)
    qqq_fig = plot_regime_overlay('QQQ', START_DATE, END_DATE)

    # Generate comparison chart
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON CHART")
    print("=" * 70)

    comparison_fig = create_comparison_chart(['SPY', 'QQQ'], START_DATE, END_DATE)

    # Summary
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  1. SPY_regime_overlay.html (SPY with regime bands)")
    print("  2. QQQ_regime_overlay.html (QQQ with regime bands)")
    print("  3. Combined_regime_comparison.html (side-by-side comparison)")
    print("\nShare these HTML files to demonstrate ATLAS regime detection!")
    print("Open in any browser - interactive charts with zoom/pan/hover.")
    print("\nKey features:")
    print("  - Green = TREND_BULL (buy signals)")
    print("  - Orange = TREND_BEAR (sell/short signals)")
    print("  - Gray = TREND_NEUTRAL (sideways/range-bound)")
    print("  - Red = CRASH (exit all, VIX spike detected)")
    print("  - Red X markers = VIX flash crash events")
    print("=" * 70)

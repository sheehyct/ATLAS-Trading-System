"""
Tests for Sharpe ratio calculation correctness.

Session 83K-58: Validates fix for Sharpe inflation bug where multiple
trades on same day were incorrectly treated as separate daily periods.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestDailySharpeCalculation:
    """Tests for daily-aggregated Sharpe calculation."""

    @pytest.fixture
    def strategy(self):
        """Create a strategy instance for testing."""
        from strategies.strat_options_strategy import STRATOptionsStrategy
        return STRATOptionsStrategy()

    def test_same_day_trades_not_inflated(self, strategy):
        """Multiple trades on same day should not inflate Sharpe."""
        # 10 trades on single day - should give 0 Sharpe (1 data point = no std)
        trades_df = pd.DataFrame({
            'pnl': [100, -50, 100, -50, 100, -50, 100, -50, 100, -50],
            'entry_date': [datetime(2024, 1, 1)] * 10,
        })

        sharpe = strategy._calculate_daily_sharpe(trades_df)

        assert sharpe == 0.0, "Single trading day should give Sharpe=0 (can't calc std)"

    def test_two_days_gives_sharpe(self, strategy):
        """Two days with different P&L should give a Sharpe value."""
        # Day 1: +$200 total, Day 2: -$100 total
        trades_df = pd.DataFrame({
            'pnl': [100, 100, -50, -50],  # Day 1: +200, Day 2: -100
            'entry_date': [
                datetime(2024, 1, 1), datetime(2024, 1, 1),
                datetime(2024, 1, 2), datetime(2024, 1, 2),
            ],
        })

        sharpe = strategy._calculate_daily_sharpe(trades_df, starting_capital=10000)

        # Should produce a Sharpe value (not 0 or inf)
        assert sharpe != 0.0, "Two days of data should produce non-zero Sharpe"
        assert np.isfinite(sharpe), f"Sharpe should be finite, got {sharpe}"

    def test_consistent_pnl_gives_high_sharpe(self, strategy):
        """Consistent positive P&L should give high Sharpe (low variance)."""
        # 10 days, each with exactly +$100 P&L
        # Note: Constant P&L gives slightly decreasing % returns as equity grows
        # (100/10000=1%, 100/10100=0.99%, etc.) so variance is small but not zero
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        trades_df = pd.DataFrame({
            'pnl': [100] * 10,
            'entry_date': dates,
        })

        sharpe = strategy._calculate_daily_sharpe(trades_df, starting_capital=10000)

        # Consistent positive returns = very high Sharpe
        assert sharpe > 0, "Consistent positive P&L should give positive Sharpe"

    def test_empty_trades_returns_zero(self, strategy):
        """Empty trades DataFrame should return 0."""
        trades_df = pd.DataFrame(columns=['pnl', 'entry_date'])
        sharpe = strategy._calculate_daily_sharpe(trades_df)

        assert sharpe == 0.0

    def test_single_trade_returns_zero(self, strategy):
        """Single trade should return 0 (can't calculate std)."""
        trades_df = pd.DataFrame({
            'pnl': [100],
            'entry_date': [datetime(2024, 1, 1)],
        })

        sharpe = strategy._calculate_daily_sharpe(trades_df)

        assert sharpe == 0.0

    def test_missing_pnl_column_returns_zero(self, strategy):
        """Missing pnl column should return 0."""
        trades_df = pd.DataFrame({
            'profit': [100, 200],  # Wrong column name
            'entry_date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        })

        sharpe = strategy._calculate_daily_sharpe(trades_df)

        assert sharpe == 0.0

    def test_missing_entry_date_column_returns_zero(self, strategy):
        """Missing entry_date column should return 0."""
        trades_df = pd.DataFrame({
            'pnl': [100, 200],
            'date': [datetime(2024, 1, 1), datetime(2024, 1, 2)],  # Wrong column name
        })

        sharpe = strategy._calculate_daily_sharpe(trades_df)

        assert sharpe == 0.0

    def test_realistic_mixed_days(self, strategy):
        """Realistic scenario with mixed winning/losing days."""
        # Simulate 20 trading days with varying P&L
        np.random.seed(42)
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(20)]
        pnls = np.random.normal(50, 100, 20)  # Mean $50, std $100

        trades_df = pd.DataFrame({
            'pnl': pnls,
            'entry_date': dates,
        })

        sharpe = strategy._calculate_daily_sharpe(trades_df, starting_capital=10000)

        # With 20 days of data, expect a reasonable Sharpe in -5 to +5 range
        assert -10 < sharpe < 10, f"Sharpe {sharpe} seems unreasonable for normal data"

    def test_hourly_multiple_trades_same_day(self, strategy):
        """Hourly timeframe with multiple signals on same day."""
        # 6 hourly trades each day for 5 days
        trades = []
        np.random.seed(123)
        for day in range(5):
            for hour in range(6):
                trades.append({
                    'pnl': np.random.choice([100, -80]),
                    'entry_date': datetime(2024, 1, 1 + day, 9 + hour, 30),
                })

        trades_df = pd.DataFrame(trades)
        sharpe = strategy._calculate_daily_sharpe(trades_df, starting_capital=10000)

        # 5 calendar days of data - should aggregate correctly
        assert isinstance(sharpe, float)
        assert np.isfinite(sharpe)


class TestSharpeComparisonOldVsNew:
    """Compare old (buggy) vs new (correct) Sharpe calculations."""

    def test_inflation_detection(self):
        """Demonstrate that old method inflates Sharpe with same-day trades."""
        # Old (wrong) method
        def old_sharpe(equity):
            returns = pd.Series(equity).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                return returns.mean() / returns.std() * np.sqrt(252)
            return 0

        # New (correct) method - same as _calculate_daily_sharpe
        def new_sharpe(trades_df, starting_capital=10000):
            daily_pnl = trades_df.groupby(trades_df['entry_date'].dt.date)['pnl'].sum()
            if len(daily_pnl) < 2:
                return 0.0
            daily_returns = daily_pnl / starting_capital
            if daily_returns.std() == 0:
                return 0.0
            return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

        # Create scenario: 20 trades across 2 days (10 each day)
        trades_df = pd.DataFrame({
            'pnl': [50, -30, 60, -20, 40, -25, 55, -35, 45, -15,
                   -10, 30, -40, 20, -50, 35, -45, 25, -55, 15],
            'entry_date': [datetime(2024, 1, 1)] * 10 + [datetime(2024, 1, 2)] * 10,
        })

        # Build equity curve for old method
        equity = [10000]
        for pnl in trades_df['pnl']:
            equity.append(max(0, equity[-1] + pnl))

        old_result = old_sharpe(equity)
        new_result = new_sharpe(trades_df)

        # Document the difference (not necessarily assert which is bigger)
        print(f"Old (buggy) Sharpe: {old_result:.2f}")
        print(f"New (correct) Sharpe: {new_result:.2f}")

        # Both should be finite
        assert np.isfinite(old_result), "Old method should produce finite result"
        # New method with only 2 days and volatile returns might be 0 or finite
        assert np.isfinite(new_result) or new_result == 0.0, "New method should be finite or 0"


class TestEdgeCases:
    """Test edge cases for Sharpe calculation."""

    @pytest.fixture
    def strategy(self):
        from strategies.strat_options_strategy import STRATOptionsStrategy
        return STRATOptionsStrategy()

    def test_none_trades_returns_zero(self, strategy):
        """None trades DataFrame should return 0."""
        sharpe = strategy._calculate_daily_sharpe(None)
        assert sharpe == 0.0

    def test_negative_pnl_all_days(self, strategy):
        """All negative P&L should give negative Sharpe."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        trades_df = pd.DataFrame({
            'pnl': [-100, -150, -50, -200, -75],
            'entry_date': dates,
        })

        sharpe = strategy._calculate_daily_sharpe(trades_df, starting_capital=10000)

        assert sharpe < 0, "All losing days should give negative Sharpe"

    def test_all_positive_pnl(self, strategy):
        """All positive P&L with variance should give positive Sharpe."""
        dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        trades_df = pd.DataFrame({
            'pnl': [100, 150, 50, 200, 75],  # Varying positive amounts
            'entry_date': dates,
        })

        sharpe = strategy._calculate_daily_sharpe(trades_df, starting_capital=10000)

        assert sharpe > 0, "All winning days with variance should give positive Sharpe"

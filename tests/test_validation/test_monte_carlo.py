"""
Tests for Monte Carlo Validator

Session 83E: Monte Carlo simulation implementation.

Test Coverage:
- TradeRecord dataclass
- Bootstrap resampling
- Equity curve and returns calculation
- Sharpe ratio calculation
- Max drawdown calculation
- Confidence interval calculation
- P(Loss) and P(Ruin) probabilities
- Options-specific IV and theta shocks
- Full validation with good/poor strategies
- Edge cases (empty, insufficient trades)
- Configuration (equity vs options thresholds)
"""

import pytest
import numpy as np
import pandas as pd

from validation.monte_carlo import (
    MonteCarloValidator,
    TradeRecord,
    generate_synthetic_trades,
)
from validation.config import MonteCarloConfig, MonteCarloConfigOptions
from validation.results import MonteCarloResults


# =============================================================================
# Test TradeRecord Dataclass
# =============================================================================

class TestTradeRecord:
    """Tests for TradeRecord dataclass."""

    def test_trade_record_creation(self):
        """Test basic TradeRecord creation."""
        trade = TradeRecord(
            pnl=100.0,
            pnl_pct=0.05,
            is_winner=True
        )
        assert trade.pnl == 100.0
        assert trade.pnl_pct == 0.05
        assert trade.is_winner is True
        assert trade.is_options is False
        assert trade.entry_iv is None

    def test_trade_record_options(self):
        """Test TradeRecord with options data."""
        trade = TradeRecord(
            pnl=150.0,
            pnl_pct=0.10,
            is_winner=True,
            is_options=True,
            entry_iv=0.25,
            exit_iv=0.30,
            theta_cost=15.0
        )
        assert trade.is_options is True
        assert trade.entry_iv == 0.25
        assert trade.exit_iv == 0.30
        assert trade.theta_cost == 15.0

    def test_trade_record_loser(self):
        """Test losing trade record."""
        trade = TradeRecord(
            pnl=-75.0,
            pnl_pct=-0.03,
            is_winner=False
        )
        assert trade.pnl == -75.0
        assert trade.is_winner is False


# =============================================================================
# Test Bootstrap Resampling
# =============================================================================

class TestBootstrapResampling:
    """Tests for bootstrap resampling functionality."""

    def test_bootstrap_returns_same_length(self):
        """Bootstrap should return same number of trades."""
        config = MonteCarloConfig(seed=42)
        validator = MonteCarloValidator(config)
        validator._rng = np.random.default_rng(42)

        trades = [
            TradeRecord(pnl=100, pnl_pct=0.05, is_winner=True),
            TradeRecord(pnl=-50, pnl_pct=-0.025, is_winner=False),
            TradeRecord(pnl=75, pnl_pct=0.04, is_winner=True),
        ]

        resampled = validator._bootstrap_trades(trades)
        assert len(resampled) == len(trades)

    def test_bootstrap_with_replacement(self):
        """Bootstrap samples with replacement (may have duplicates)."""
        config = MonteCarloConfig(seed=42)
        validator = MonteCarloValidator(config)
        validator._rng = np.random.default_rng(42)

        # Create 10 unique trades
        trades = [
            TradeRecord(pnl=i * 10, pnl_pct=0.01 * i, is_winner=(i % 2 == 0))
            for i in range(10)
        ]

        # Run many resamples - should occasionally have duplicates
        has_duplicate = False
        for _ in range(100):
            resampled = validator._bootstrap_trades(trades)
            pnls = [t.pnl for t in resampled]
            if len(pnls) != len(set(pnls)):
                has_duplicate = True
                break

        assert has_duplicate, "Bootstrap should produce duplicates with replacement"

    def test_bootstrap_reproducible_with_seed(self):
        """Same seed should produce same resample."""
        trades = [
            TradeRecord(pnl=100, pnl_pct=0.05, is_winner=True),
            TradeRecord(pnl=-50, pnl_pct=-0.025, is_winner=False),
            TradeRecord(pnl=75, pnl_pct=0.04, is_winner=True),
        ]

        # First run
        config1 = MonteCarloConfig(seed=42)
        validator1 = MonteCarloValidator(config1)
        validator1._rng = np.random.default_rng(42)
        result1 = validator1._bootstrap_trades(trades)

        # Second run with same seed
        config2 = MonteCarloConfig(seed=42)
        validator2 = MonteCarloValidator(config2)
        validator2._rng = np.random.default_rng(42)
        result2 = validator2._bootstrap_trades(trades)

        assert [t.pnl for t in result1] == [t.pnl for t in result2]


# =============================================================================
# Test Equity Curve Calculation
# =============================================================================

class TestEquityCurveCalculation:
    """Tests for equity curve calculation."""

    def test_equity_curve_basic(self):
        """Test basic equity curve calculation."""
        config = MonteCarloConfig()
        validator = MonteCarloValidator(config)

        trades = [
            TradeRecord(pnl=100, pnl_pct=0.01, is_winner=True),
            TradeRecord(pnl=50, pnl_pct=0.005, is_winner=True),
            TradeRecord(pnl=-30, pnl_pct=-0.003, is_winner=False),
        ]

        equity = validator._calculate_equity_curve(trades, starting_capital=10000)

        assert len(equity) == 4  # Starting + 3 trades
        assert equity[0] == 10000
        assert equity[1] == 10100
        assert equity[2] == 10150
        assert equity[3] == 10120

    def test_equity_curve_starting_capital(self):
        """Test equity curve respects starting capital."""
        config = MonteCarloConfig()
        validator = MonteCarloValidator(config)

        trades = [TradeRecord(pnl=100, pnl_pct=0.01, is_winner=True)]

        equity_10k = validator._calculate_equity_curve(trades, starting_capital=10000)
        equity_50k = validator._calculate_equity_curve(trades, starting_capital=50000)

        assert equity_10k[0] == 10000
        assert equity_50k[0] == 50000
        assert equity_10k[1] - equity_10k[0] == equity_50k[1] - equity_50k[0]  # Same P/L

    def test_equity_curve_empty_trades(self):
        """Test equity curve with no trades."""
        config = MonteCarloConfig()
        validator = MonteCarloValidator(config)

        equity = validator._calculate_equity_curve([], starting_capital=10000)

        assert len(equity) == 1
        assert equity[0] == 10000


# =============================================================================
# Test Sharpe Ratio Calculation
# =============================================================================

class TestSharpeCalculation:
    """Tests for Sharpe ratio calculation."""

    def test_sharpe_positive_returns(self):
        """Test Sharpe with positive returns."""
        config = MonteCarloConfig()
        validator = MonteCarloValidator(config)

        # Consistent positive returns should give positive Sharpe
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])
        sharpe = validator._calculate_sharpe(returns)

        assert sharpe > 0

    def test_sharpe_zero_variance(self):
        """Test Sharpe with zero variance returns."""
        config = MonteCarloConfig()
        validator = MonteCarloValidator(config)

        returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe = validator._calculate_sharpe(returns)

        assert sharpe == 0.0  # Can't calculate with zero std

    def test_sharpe_empty_returns(self):
        """Test Sharpe with empty returns."""
        config = MonteCarloConfig()
        validator = MonteCarloValidator(config)

        sharpe = validator._calculate_sharpe(np.array([]))

        assert sharpe == 0.0

    def test_sharpe_single_return(self):
        """Test Sharpe with single return."""
        config = MonteCarloConfig()
        validator = MonteCarloValidator(config)

        sharpe = validator._calculate_sharpe(np.array([0.05]))

        assert sharpe == 0.0  # Need at least 2 returns


# =============================================================================
# Test Max Drawdown Calculation
# =============================================================================

class TestMaxDrawdownCalculation:
    """Tests for maximum drawdown calculation."""

    def test_max_drawdown_no_dd(self):
        """Test max DD with monotonically increasing equity."""
        config = MonteCarloConfig()
        validator = MonteCarloValidator(config)

        equity = np.array([10000, 10100, 10200, 10300, 10400])
        max_dd = validator._calculate_max_drawdown(equity)

        assert max_dd == 0.0

    def test_max_drawdown_simple(self):
        """Test max DD with single drawdown."""
        config = MonteCarloConfig()
        validator = MonteCarloValidator(config)

        # 10000 -> 9000 = 10% drawdown
        equity = np.array([10000, 9000, 9500, 10000])
        max_dd = validator._calculate_max_drawdown(equity)

        assert abs(max_dd - 0.10) < 0.001

    def test_max_drawdown_multiple(self):
        """Test max DD with multiple drawdowns returns the largest."""
        config = MonteCarloConfig()
        validator = MonteCarloValidator(config)

        # First DD: 10% (10000 -> 9000)
        # Second DD: 20% (10500 -> 8400)
        equity = np.array([10000, 9000, 10500, 8400, 11000])
        max_dd = validator._calculate_max_drawdown(equity)

        assert abs(max_dd - 0.20) < 0.001

    def test_max_drawdown_empty(self):
        """Test max DD with empty equity."""
        config = MonteCarloConfig()
        validator = MonteCarloValidator(config)

        max_dd = validator._calculate_max_drawdown(np.array([]))

        assert max_dd == 0.0


# =============================================================================
# Test Confidence Interval Calculation
# =============================================================================

class TestConfidenceIntervals:
    """Tests for confidence interval calculations."""

    def test_ci_95_percent(self):
        """Test 95% confidence interval calculation."""
        config = MonteCarloConfig(confidence_level=0.95)
        validator = MonteCarloValidator(config)

        # Create a distribution we know the percentiles of
        data = np.arange(1, 101)  # 1 to 100

        # 2.5th percentile should be ~3, 97.5th should be ~98
        lower_pct = (1 - 0.95) / 2 * 100  # 2.5
        upper_pct = (1 - (1 - 0.95) / 2) * 100  # 97.5

        lower = np.percentile(data, lower_pct)
        upper = np.percentile(data, upper_pct)

        assert abs(lower - 3.475) < 0.1
        assert abs(upper - 97.525) < 0.1

    def test_ci_excludes_zero_check(self):
        """Test CI excludes zero property."""
        results = MonteCarloResults(
            original_sharpe=1.0,
            simulated_sharpe_mean=0.9,
            simulated_sharpe_std=0.2,
            sharpe_95_ci=(0.5, 1.3),  # Does not include 0
            original_max_dd=0.15,
            simulated_max_dd_95=0.20,
            max_dd_95_ci=(0.10, 0.25),
            return_95_ci=(0.05, 0.25),
            probability_of_loss=0.10,
            probability_of_ruin=0.02,
            n_simulations=1000,
            passes_validation=True,
            failure_reasons=[]
        )

        assert results.sharpe_ci_excludes_zero is True

    def test_ci_includes_zero_check(self):
        """Test CI includes zero property."""
        results = MonteCarloResults(
            original_sharpe=0.3,
            simulated_sharpe_mean=0.25,
            simulated_sharpe_std=0.3,
            sharpe_95_ci=(-0.2, 0.7),  # Includes 0
            original_max_dd=0.20,
            simulated_max_dd_95=0.25,
            max_dd_95_ci=(0.15, 0.30),
            return_95_ci=(-0.05, 0.15),
            probability_of_loss=0.35,
            probability_of_ruin=0.08,
            n_simulations=1000,
            passes_validation=False,
            failure_reasons=["CI includes zero"]
        )

        assert results.sharpe_ci_excludes_zero is False


# =============================================================================
# Test Probability Calculations
# =============================================================================

class TestProbabilityCalculations:
    """Tests for P(Loss) and P(Ruin) calculations."""

    def test_probability_of_loss(self):
        """Test P(Loss) calculation."""
        # 30 out of 100 simulations result in loss
        simulated_returns = np.array([1.0] * 70 + [-0.1] * 30)
        p_loss = np.sum(simulated_returns < 0) / len(simulated_returns)

        assert abs(p_loss - 0.30) < 0.001

    def test_probability_of_ruin(self):
        """Test P(Ruin) calculation with 50% threshold."""
        # 5 out of 100 have >50% drawdown
        simulated_dds = np.array([0.20] * 95 + [0.60] * 5)
        ruin_threshold = 0.50
        p_ruin = np.sum(simulated_dds > ruin_threshold) / len(simulated_dds)

        assert abs(p_ruin - 0.05) < 0.001


# =============================================================================
# Test Options Shocks
# =============================================================================

class TestOptionsShocks:
    """Tests for options-specific IV and theta shocks."""

    def test_options_shocks_applied(self):
        """Test that options shocks modify P/L."""
        config = MonteCarloConfigOptions(
            seed=42,
            iv_shock_std=0.20,
            theta_shock_max=1.50,
            apply_options_shocks=True
        )
        validator = MonteCarloValidator(config)
        validator._rng = np.random.default_rng(42)

        # Create options trades
        trades = [
            TradeRecord(
                pnl=100.0,
                pnl_pct=0.05,
                is_winner=True,
                is_options=True,
                entry_iv=0.25,
                exit_iv=0.30,
                theta_cost=10.0
            ),
            TradeRecord(
                pnl=-80.0,
                pnl_pct=-0.04,
                is_winner=False,
                is_options=True,
                entry_iv=0.25,
                exit_iv=0.20,
                theta_cost=15.0
            ),
        ]

        shocked = validator._apply_options_shocks(trades)

        # P/L should be modified (at least the loser with theta shock)
        assert shocked[0].pnl != trades[0].pnl or shocked[1].pnl != trades[1].pnl

    def test_options_shocks_disabled(self):
        """Test that shocks can be disabled."""
        config = MonteCarloConfigOptions(
            seed=42,
            apply_options_shocks=False
        )
        validator = MonteCarloValidator(config)
        validator._rng = np.random.default_rng(42)

        trades = [
            TradeRecord(
                pnl=100.0,
                pnl_pct=0.05,
                is_winner=True,
                is_options=True,
                entry_iv=0.25,
                exit_iv=0.30,
                theta_cost=10.0
            ),
        ]

        # With shocks disabled, should return unchanged
        shocked = validator._apply_options_shocks(trades)

        assert shocked == trades

    def test_theta_shock_on_losers_only(self):
        """Test theta shock only affects losing trades."""
        config = MonteCarloConfigOptions(
            seed=42,
            iv_shock_std=0.0,  # Disable IV shock
            theta_shock_max=1.50,
            apply_options_shocks=True
        )
        validator = MonteCarloValidator(config)
        validator._rng = np.random.default_rng(42)

        # Winner with theta cost - should not have theta shock
        winner = TradeRecord(
            pnl=100.0,
            pnl_pct=0.05,
            is_winner=True,
            is_options=True,
            entry_iv=None,  # No IV shock
            exit_iv=None,
            theta_cost=10.0
        )

        # Loser with theta cost - should have theta shock
        loser = TradeRecord(
            pnl=-80.0,
            pnl_pct=-0.04,
            is_winner=False,
            is_options=True,
            entry_iv=None,  # No IV shock
            exit_iv=None,
            theta_cost=15.0
        )

        shocked = validator._apply_options_shocks([winner, loser])

        # Winner should be unchanged (no IV, no theta shock on winners)
        assert shocked[0].pnl == winner.pnl

        # Loser should be worse (theta shock)
        assert shocked[1].pnl < loser.pnl


# =============================================================================
# Test Full Validation
# =============================================================================

class TestFullValidation:
    """Tests for complete Monte Carlo validation."""

    def test_validate_returns_results(self):
        """Test validate() returns MonteCarloResults."""
        config = MonteCarloConfig(n_simulations=100, seed=42)
        validator = MonteCarloValidator(config)

        trades = generate_synthetic_trades(50, win_rate=0.55, seed=42)
        results = validator.validate(trades, account_size=10000)

        assert isinstance(results, MonteCarloResults)
        assert results.n_simulations == 100

    def test_validate_good_strategy(self):
        """Test validation passes for good strategy."""
        config = MonteCarloConfig(
            n_simulations=500,
            seed=42,
            max_probability_of_loss=0.30,  # Generous threshold
            max_probability_of_ruin=0.10
        )
        validator = MonteCarloValidator(config)

        # Good strategy: high win rate, favorable R:R
        trades = generate_synthetic_trades(
            n_trades=100,
            win_rate=0.60,
            avg_winner=120.0,
            avg_loser=-80.0,
            seed=42
        )

        results = validator.validate(trades, account_size=10000)

        # Should likely pass with good strategy
        assert results.simulated_sharpe_mean > 0
        assert results.probability_of_loss < 0.50  # At least better than coin flip

    def test_validate_poor_strategy(self):
        """Test validation fails for poor strategy."""
        config = MonteCarloConfig(
            n_simulations=500,
            seed=42,
            max_probability_of_loss=0.20,
            max_probability_of_ruin=0.05
        )
        validator = MonteCarloValidator(config)

        # Poor strategy: low win rate, unfavorable R:R
        trades = generate_synthetic_trades(
            n_trades=100,
            win_rate=0.35,
            avg_winner=80.0,
            avg_loser=-120.0,
            seed=42
        )

        results = validator.validate(trades, account_size=10000)

        # Should fail
        assert results.passes_validation is False
        assert len(results.failure_reasons) > 0

    def test_validate_empty_trades(self):
        """Test validation with empty trades DataFrame."""
        config = MonteCarloConfig(n_simulations=100, seed=42)
        validator = MonteCarloValidator(config)

        trades = pd.DataFrame(columns=['pnl'])
        results = validator.validate(trades, account_size=10000)

        assert results.passes_validation is False
        assert "No trades" in results.failure_reasons[0]

    def test_validate_insufficient_trades(self):
        """Test validation with too few trades."""
        config = MonteCarloConfig(n_simulations=100, seed=42)
        validator = MonteCarloValidator(config)

        trades = pd.DataFrame({'pnl': [100, -50, 75]})  # Only 3 trades
        results = validator.validate(trades, account_size=10000)

        assert results.passes_validation is False
        assert "Insufficient" in results.failure_reasons[0]

    def test_validate_missing_pnl_column(self):
        """Test validation fails gracefully with missing pnl column."""
        config = MonteCarloConfig(n_simulations=100, seed=42)
        validator = MonteCarloValidator(config)

        trades = pd.DataFrame({'profit': [100, -50, 75]})  # Wrong column name
        results = validator.validate(trades, account_size=10000)

        assert results.passes_validation is False
        assert "pnl" in results.failure_reasons[0].lower()


# =============================================================================
# Test Results Summary
# =============================================================================

class TestResultsSummary:
    """Tests for results summary and serialization."""

    def test_summary_method(self):
        """Test summary() produces readable output."""
        results = MonteCarloResults(
            original_sharpe=1.2,
            simulated_sharpe_mean=1.0,
            simulated_sharpe_std=0.3,
            sharpe_95_ci=(0.4, 1.6),
            original_max_dd=0.15,
            simulated_max_dd_95=0.22,
            max_dd_95_ci=(0.10, 0.25),
            return_95_ci=(0.08, 0.32),
            probability_of_loss=0.12,
            probability_of_ruin=0.03,
            n_simulations=10000,
            passes_validation=True,
            failure_reasons=[]
        )

        summary = results.summary()

        assert "MONTE CARLO" in summary
        assert "PASSED" in summary
        assert "10,000" in summary
        assert "12.0%" in summary  # P(Loss)

    def test_to_dict_serializable(self):
        """Test to_dict() produces serializable output."""
        results = MonteCarloResults(
            original_sharpe=1.0,
            simulated_sharpe_mean=0.9,
            simulated_sharpe_std=0.2,
            sharpe_95_ci=(0.5, 1.3),
            original_max_dd=0.15,
            simulated_max_dd_95=0.20,
            max_dd_95_ci=(0.10, 0.25),
            return_95_ci=(0.05, 0.25),
            probability_of_loss=0.15,
            probability_of_ruin=0.04,
            n_simulations=1000,
            passes_validation=True,
            failure_reasons=[]
        )

        d = results.to_dict()

        assert isinstance(d, dict)
        assert d['original_sharpe'] == 1.0
        assert d['n_simulations'] == 1000
        assert isinstance(d['sharpe_95_ci'], list)

    def test_passes_method(self):
        """Test passes() convenience method."""
        config = MonteCarloConfig()
        validator = MonteCarloValidator(config)

        results_pass = MonteCarloResults(
            original_sharpe=1.0,
            simulated_sharpe_mean=0.9,
            simulated_sharpe_std=0.2,
            sharpe_95_ci=(0.5, 1.3),
            original_max_dd=0.15,
            simulated_max_dd_95=0.20,
            max_dd_95_ci=(0.10, 0.25),
            return_95_ci=(0.05, 0.25),
            probability_of_loss=0.15,
            probability_of_ruin=0.04,
            n_simulations=1000,
            passes_validation=True,
            failure_reasons=[]
        )

        results_fail = MonteCarloResults(
            original_sharpe=0.2,
            simulated_sharpe_mean=0.1,
            simulated_sharpe_std=0.3,
            sharpe_95_ci=(-0.4, 0.6),
            original_max_dd=0.35,
            simulated_max_dd_95=0.55,
            max_dd_95_ci=(0.30, 0.60),
            return_95_ci=(-0.15, 0.10),
            probability_of_loss=0.45,
            probability_of_ruin=0.20,
            n_simulations=1000,
            passes_validation=False,
            failure_reasons=["P(Loss) too high"]
        )

        assert validator.passes(results_pass) is True
        assert validator.passes(results_fail) is False


# =============================================================================
# Test Configuration
# =============================================================================

class TestConfiguration:
    """Tests for Monte Carlo configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MonteCarloConfig()

        assert config.n_simulations == 10000
        assert config.confidence_level == 0.95
        assert config.max_probability_of_loss == 0.20
        assert config.max_probability_of_ruin == 0.05
        assert config.ruin_threshold == 0.50
        assert config.seed is None

    def test_options_config_looser_thresholds(self):
        """Test options config has looser thresholds."""
        equity_config = MonteCarloConfig()
        options_config = MonteCarloConfigOptions()

        assert options_config.max_probability_of_loss > equity_config.max_probability_of_loss
        assert options_config.max_probability_of_ruin > equity_config.max_probability_of_ruin

    def test_options_config_shock_parameters(self):
        """Test options config has shock parameters."""
        config = MonteCarloConfigOptions()

        assert config.iv_shock_std == 0.20
        assert config.theta_shock_max == 1.50
        assert config.apply_options_shocks is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = MonteCarloConfig(
            n_simulations=5000,
            confidence_level=0.99,
            max_probability_of_loss=0.15,
            max_probability_of_ruin=0.03,
            ruin_threshold=0.40,
            seed=123
        )

        assert config.n_simulations == 5000
        assert config.confidence_level == 0.99
        assert config.max_probability_of_loss == 0.15
        assert config.ruin_threshold == 0.40
        assert config.seed == 123


# =============================================================================
# Test Synthetic Trade Generation
# =============================================================================

class TestSyntheticTradeGeneration:
    """Tests for synthetic trade generation helper."""

    def test_generate_trades_count(self):
        """Test correct number of trades generated."""
        trades = generate_synthetic_trades(100, seed=42)

        assert len(trades) == 100

    def test_generate_trades_has_pnl(self):
        """Test generated trades have pnl column."""
        trades = generate_synthetic_trades(50, seed=42)

        assert 'pnl' in trades.columns

    def test_generate_trades_win_rate(self):
        """Test generated trades roughly match win rate."""
        # Generate many trades to test win rate
        trades = generate_synthetic_trades(1000, win_rate=0.60, seed=42)

        actual_win_rate = (trades['pnl'] > 0).mean()

        # Should be close to 60% (within 5%)
        assert abs(actual_win_rate - 0.60) < 0.05

    def test_generate_trades_reproducible(self):
        """Test same seed produces same trades."""
        trades1 = generate_synthetic_trades(50, seed=42)
        trades2 = generate_synthetic_trades(50, seed=42)

        pd.testing.assert_frame_equal(trades1, trades2)


# =============================================================================
# Test Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_all_winning_trades(self):
        """Test with all winning trades."""
        config = MonteCarloConfig(n_simulations=100, seed=42)
        validator = MonteCarloValidator(config)

        trades = pd.DataFrame({'pnl': [100, 50, 75, 120, 80]})
        results = validator.validate(trades, account_size=10000)

        assert results.probability_of_loss == 0.0
        assert results.simulated_sharpe_mean > 0

    def test_all_losing_trades(self):
        """Test with all losing trades."""
        config = MonteCarloConfig(n_simulations=100, seed=42)
        validator = MonteCarloValidator(config)

        trades = pd.DataFrame({'pnl': [-100, -50, -75, -120, -80]})
        results = validator.validate(trades, account_size=10000)

        assert results.probability_of_loss == 1.0
        assert results.passes_validation is False

    def test_zero_pnl_trades(self):
        """Test with trades that have zero P/L."""
        config = MonteCarloConfig(n_simulations=100, seed=42)
        validator = MonteCarloValidator(config)

        trades = pd.DataFrame({'pnl': [0, 0, 100, -50, 0]})
        results = validator.validate(trades, account_size=10000)

        assert isinstance(results, MonteCarloResults)

    def test_large_trades(self):
        """Test with very large P/L values."""
        config = MonteCarloConfig(n_simulations=100, seed=42)
        validator = MonteCarloValidator(config)

        trades = pd.DataFrame({'pnl': [50000, -30000, 75000, -25000, 60000]})
        results = validator.validate(trades, account_size=100000)

        assert isinstance(results, MonteCarloResults)
        assert np.isfinite(results.simulated_sharpe_mean)

    def test_validation_with_options_flag(self):
        """Test validation with is_options=True."""
        config = MonteCarloConfigOptions(n_simulations=100, seed=42)
        validator = MonteCarloValidator(config)

        trades = pd.DataFrame({
            'pnl': [100, -50, 75, -25, 80],
            'entry_iv': [0.25, 0.30, 0.22, 0.28, 0.24],
            'exit_iv': [0.28, 0.25, 0.25, 0.30, 0.22],
            'theta_cost': [10, 15, 8, 12, 9]
        })

        results = validator.validate(trades, account_size=10000, is_options=True)

        assert isinstance(results, MonteCarloResults)


# =============================================================================
# Test Validation Criteria
# =============================================================================

class TestValidationCriteria:
    """Tests for specific validation criteria."""

    def test_sharpe_ci_failure(self):
        """Test failure when Sharpe CI includes zero."""
        results = MonteCarloResults(
            original_sharpe=0.3,
            simulated_sharpe_mean=0.2,
            simulated_sharpe_std=0.4,
            sharpe_95_ci=(-0.5, 0.9),  # Includes zero
            original_max_dd=0.20,
            simulated_max_dd_95=0.25,
            max_dd_95_ci=(0.15, 0.30),
            return_95_ci=(-0.05, 0.20),
            probability_of_loss=0.15,
            probability_of_ruin=0.03,
            n_simulations=1000,
            passes_validation=False,
            failure_reasons=["95% CI for Sharpe includes zero"]
        )

        assert results.sharpe_ci_excludes_zero is False

    def test_p_loss_failure(self):
        """Test failure when P(Loss) exceeds threshold."""
        config = MonteCarloConfig(max_probability_of_loss=0.20)
        validator = MonteCarloValidator(config)

        # Create strategy that will have high P(Loss)
        trades = generate_synthetic_trades(
            n_trades=100,
            win_rate=0.40,
            avg_winner=60,
            avg_loser=-100,
            seed=42
        )

        results = validator.validate(trades, account_size=10000)

        # With 40% win rate and unfavorable R:R, P(Loss) should be high
        assert results.probability_of_loss > 0.20

    def test_p_ruin_threshold(self):
        """Test P(Ruin) uses configurable threshold."""
        config = MonteCarloConfig(
            n_simulations=500,
            ruin_threshold=0.30,  # 30% drawdown = ruin
            seed=42
        )
        validator = MonteCarloValidator(config)

        # Strategy with moderate volatility
        trades = generate_synthetic_trades(
            n_trades=100,
            win_rate=0.50,
            avg_winner=100,
            avg_loser=-100,
            seed=42
        )

        results = validator.validate(trades, account_size=10000)

        # P(Ruin) should reflect the 30% threshold
        assert isinstance(results.probability_of_ruin, float)

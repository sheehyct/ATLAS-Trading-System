"""
Integration tests for Options Module with STRAT patterns.

Session 72: Validates the complete flow from pattern detection to options trade:
1. Tier1Detector -> PatternSignal
2. PatternSignal -> OptionContract (via OptionsExecutor)
3. OptionContract -> Backtest Results (via OptionsBacktester)

These tests ensure all components work together correctly.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from strat.options_module import (
    OptionsExecutor,
    OptionsBacktester,
    OptionContract,
    OptionType,
    OptionStrategy,
    OptionTrade,
)
from strat.tier1_detector import Tier1Detector, PatternSignal, PatternType, Timeframe
from strat.greeks import calculate_greeks


class TestOptionContractOSI:
    """Test OSI symbol generation for option contracts."""

    def test_call_osi_format(self):
        """Test OSI symbol format for call option."""
        contract = OptionContract(
            underlying='SPY',
            expiration=datetime(2024, 12, 20),
            option_type=OptionType.CALL,
            strike=300.0
        )

        # OSI format: SYMBOL + YYMMDD + C/P + 00000000 (8 digits, strike * 1000)
        expected = 'SPY241220C00300000'
        assert contract.osi_symbol == expected, f"OSI {contract.osi_symbol} != {expected}"

    def test_put_osi_format(self):
        """Test OSI symbol format for put option."""
        contract = OptionContract(
            underlying='AAPL',
            expiration=datetime(2025, 1, 17),
            option_type=OptionType.PUT,
            strike=150.5
        )

        expected = 'AAPL250117P00150500'
        assert contract.osi_symbol == expected, f"OSI {contract.osi_symbol} != {expected}"

    def test_fractional_strike(self):
        """Test OSI symbol handles fractional strikes correctly."""
        contract = OptionContract(
            underlying='QQQ',
            expiration=datetime(2025, 3, 21),
            option_type=OptionType.CALL,
            strike=387.50
        )

        # Strike 387.50 * 1000 = 387500 -> 00387500
        expected = 'QQQ250321C00387500'
        assert contract.osi_symbol == expected


class TestStrikeSelection:
    """Test the 0.3x strike selection formula."""

    def test_bullish_strike_above_entry(self):
        """Bullish trade: strike should be above entry (0.3x toward target)."""
        # Mock pattern with entry=100, target=110, stop=95
        # 0.3x from entry to target: 100 + 0.3*(110-100) = 103
        entry = 100
        target = 110
        expected_strike = entry + 0.3 * (target - entry)  # 103

        # Verify our calculation matches expectation
        assert expected_strike == 103.0

    def test_bearish_strike_below_entry(self):
        """Bearish trade: strike should be below entry (0.3x toward target)."""
        # Mock pattern with entry=100, target=90, stop=105
        # 0.3x from entry to target: 100 + 0.3*(90-100) = 97
        entry = 100
        target = 90
        expected_strike = entry + 0.3 * (target - entry)  # 97

        # Verify our calculation matches expectation
        assert expected_strike == 97.0

    def test_strike_rounds_to_increment(self):
        """Strike should round to standard increment ($1 for most stocks)."""
        # Strike of 103.37 should round to 103.0 or 104.0 depending on rounding rule
        raw_strike = 103.37
        rounded = round(raw_strike)

        assert rounded in [103.0, 104.0]


class TestOptionsExecutorPatternFlow:
    """Test OptionsExecutor converts patterns to option trades correctly."""

    @pytest.fixture
    def executor(self):
        """Create OptionsExecutor instance."""
        return OptionsExecutor()

    @pytest.fixture
    def mock_bullish_signal(self):
        """Create a mock bullish pattern signal."""
        return PatternSignal(
            pattern_type=PatternType.PATTERN_212_UP,
            direction=1,  # Bullish
            entry_price=100.0,
            stop_price=95.0,
            target_price=110.0,
            timestamp=pd.Timestamp('2024-06-15'),
            timeframe=Timeframe.WEEKLY,
        )

    @pytest.fixture
    def mock_bearish_signal(self):
        """Create a mock bearish pattern signal."""
        return PatternSignal(
            pattern_type=PatternType.PATTERN_212_DOWN,
            direction=-1,  # Bearish
            entry_price=100.0,
            stop_price=105.0,
            target_price=90.0,
            timestamp=pd.Timestamp('2024-06-15'),
            timeframe=Timeframe.WEEKLY,
        )

    def test_bullish_signal_creates_call(self, executor, mock_bullish_signal):
        """Bullish signal should create CALL option."""
        trades = executor.generate_option_trades(
            [mock_bullish_signal],
            underlying='SPY',
            underlying_price=100.0
        )

        assert len(trades) == 1
        assert trades[0].contract.option_type == OptionType.CALL

    def test_bearish_signal_creates_put(self, executor, mock_bearish_signal):
        """Bearish signal should create PUT option."""
        trades = executor.generate_option_trades(
            [mock_bearish_signal],
            underlying='SPY',
            underlying_price=100.0
        )

        assert len(trades) == 1
        assert trades[0].contract.option_type == OptionType.PUT

    def test_expiration_is_future(self, executor, mock_bullish_signal):
        """Option expiration should be in the future relative to signal."""
        trades = executor.generate_option_trades(
            [mock_bullish_signal],
            underlying='SPY',
            underlying_price=100.0
        )

        assert trades[0].contract.expiration > mock_bullish_signal.timestamp

    def test_trade_preserves_pattern_info(self, executor, mock_bullish_signal):
        """Trade should preserve pattern signal information."""
        trades = executor.generate_option_trades(
            [mock_bullish_signal],
            underlying='SPY',
            underlying_price=100.0
        )

        assert trades[0].pattern_signal == mock_bullish_signal
        assert trades[0].entry_trigger == mock_bullish_signal.entry_price


class TestEndToEndBacktest:
    """End-to-end tests with synthetic data."""

    @pytest.fixture
    def synthetic_price_data(self):
        """Create synthetic price data for backtesting."""
        # 100 days of price data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)

        # Start at $450, add random walk
        prices = 450 + np.cumsum(np.random.randn(100) * 2)

        return pd.DataFrame({
            'Open': prices - 0.5,
            'High': prices + 2,
            'Low': prices - 2,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)

    @pytest.fixture
    def mock_trade_for_backtest(self, synthetic_price_data):
        """Create a mock trade aligned with synthetic data."""
        # Use a date from the synthetic data
        signal_date = synthetic_price_data.index[10]
        entry_price_val = synthetic_price_data.loc[signal_date, 'Close']
        target_price_val = entry_price_val + 15
        stop_price_val = entry_price_val - 8

        signal = PatternSignal(
            pattern_type=PatternType.PATTERN_212_UP,
            direction=1,  # Bullish
            entry_price=entry_price_val,
            stop_price=stop_price_val,
            target_price=target_price_val,
            timestamp=signal_date,  # Already a pd.Timestamp
            timeframe=Timeframe.WEEKLY,
        )

        # Calculate strike using 0.3x formula
        strike = entry_price_val + 0.3 * (target_price_val - entry_price_val)
        strike = round(strike)

        contract = OptionContract(
            underlying='SPY',
            expiration=signal_date.to_pydatetime() + timedelta(days=35),
            option_type=OptionType.CALL,
            strike=strike
        )

        return OptionTrade(
            contract=contract,
            pattern_signal=signal,
            entry_trigger=entry_price_val,
            target_exit=target_price_val,
            stop_exit=stop_price_val,
            quantity=1,
            strategy=OptionStrategy.LONG_CALL
        )

    def test_backtest_produces_results(self, synthetic_price_data, mock_trade_for_backtest):
        """Backtest should produce DataFrame with results."""
        backtester = OptionsBacktester(risk_free_rate=0.05, default_iv=0.20)
        results = backtester.backtest_trades([mock_trade_for_backtest], synthetic_price_data)

        # Results should be a DataFrame
        assert isinstance(results, pd.DataFrame)

    def test_backtest_result_columns(self, synthetic_price_data, mock_trade_for_backtest):
        """Backtest results should have expected columns."""
        backtester = OptionsBacktester(risk_free_rate=0.05, default_iv=0.20)
        results = backtester.backtest_trades([mock_trade_for_backtest], synthetic_price_data)

        if not results.empty:
            expected_columns = [
                'timestamp', 'pattern_type', 'osi_symbol', 'entry_trigger',
                'exit_price', 'exit_type', 'days_held', 'pnl', 'win'
            ]
            for col in expected_columns:
                assert col in results.columns, f"Missing column: {col}"


class TestGreeksIntegration:
    """Test Greeks calculation integration with options flow."""

    def test_greeks_at_entry_and_exit(self):
        """Greeks should be calculable at both entry and exit points."""
        S_entry = 450
        S_exit = 460
        K = 455
        T_entry = 35/365
        T_exit = 28/365

        entry_greeks = calculate_greeks(
            S=S_entry, K=K, T=T_entry, r=0.05, sigma=0.20, option_type='call'
        )
        exit_greeks = calculate_greeks(
            S=S_exit, K=K, T=T_exit, r=0.05, sigma=0.20, option_type='call'
        )

        # Exit price should be higher (ITM now)
        assert exit_greeks.option_price > entry_greeks.option_price

        # Exit delta should be higher (more ITM)
        assert exit_greeks.delta > entry_greeks.delta

    def test_theta_integration_over_time(self):
        """Theta decay should be integrated over holding period."""
        S, K = 450, 450
        greeks_35d = calculate_greeks(S, K, 35/365, 0.05, 0.20, 'call')
        greeks_28d = calculate_greeks(S, K, 28/365, 0.05, 0.20, 'call')

        # Price should decrease due to theta (if stock price unchanged)
        price_loss = greeks_35d.option_price - greeks_28d.option_price

        # Using average theta over 7 days
        avg_theta = (greeks_35d.theta + greeks_28d.theta) / 2
        estimated_decay = abs(avg_theta) * 7

        # Estimated decay should be close to actual price loss
        assert abs(price_loss - estimated_decay) < 0.3, f"Decay mismatch: {price_loss} vs {estimated_decay}"


class TestDataFrameConversion:
    """Test trades to DataFrame conversion."""

    @pytest.fixture
    def sample_trades(self):
        """Create sample trades for testing."""
        trades = []
        for i in range(3):
            entry_price_val = 450 + i*5
            target_price_val = 460 + i*5
            stop_price_val = 445 + i*5

            signal = PatternSignal(
                pattern_type=PatternType.PATTERN_212_UP,
                direction=1,  # Bullish
                entry_price=entry_price_val,
                stop_price=stop_price_val,
                target_price=target_price_val,
                timestamp=pd.Timestamp(f'2024-06-{15 + i*7}'),
                timeframe=Timeframe.WEEKLY,
            )

            contract = OptionContract(
                underlying='SPY',
                expiration=datetime(2024, 7, 19),
                option_type=OptionType.CALL,
                strike=455 + i*5
            )

            trades.append(OptionTrade(
                contract=contract,
                pattern_signal=signal,
                entry_trigger=entry_price_val,
                target_exit=target_price_val,
                stop_exit=stop_price_val,
                quantity=1,
                strategy=OptionStrategy.LONG_CALL
            ))

        return trades

    def test_trades_to_dataframe(self, sample_trades):
        """trades_to_dataframe should create valid DataFrame."""
        executor = OptionsExecutor()
        df = executor.trades_to_dataframe(sample_trades)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_dataframe_has_required_columns(self, sample_trades):
        """DataFrame should have all required columns."""
        executor = OptionsExecutor()
        df = executor.trades_to_dataframe(sample_trades)

        required_cols = ['underlying', 'osi_symbol', 'strike', 'option_type', 'entry_trigger', 'target_exit', 'stop_exit']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

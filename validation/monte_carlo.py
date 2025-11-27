"""
ATLAS Production Readiness Validation - Monte Carlo Simulator

Implements Monte Carlo simulation for strategy robustness testing per ATLAS Checklist Section 1.7.

Key Features:
- Bootstrap resampling with trade order randomization
- Options-specific IV shock and theta uncertainty modeling
- Confidence interval calculation for Sharpe ratio
- P(Loss) and P(Ruin) probability estimation
- Configurable simulation count (default 10,000)

Session 83E: Monte Carlo simulation implementation.

Usage:
    from validation.monte_carlo import MonteCarloValidator
    from validation import MonteCarloConfig

    validator = MonteCarloValidator(MonteCarloConfig())
    results = validator.validate(trades_df, account_size=10000)

    if results.passes_validation:
        print("Strategy passes Monte Carlo robustness testing")
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from validation.config import MonteCarloConfig, MonteCarloConfigOptions
from validation.results import MonteCarloResults

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """
    Simplified trade record for Monte Carlo simulation.

    Attributes:
        pnl: Profit/loss for the trade
        pnl_pct: P/L as percentage of entry value
        is_winner: Whether trade was profitable
        is_options: Whether this is an options trade
        entry_iv: Entry implied volatility (options only)
        exit_iv: Exit implied volatility (options only)
        theta_cost: Theta decay cost (options only)
    """
    pnl: float
    pnl_pct: float
    is_winner: bool
    is_options: bool = False
    entry_iv: Optional[float] = None
    exit_iv: Optional[float] = None
    theta_cost: Optional[float] = None


class MonteCarloValidator:
    """
    Monte Carlo simulation for strategy robustness testing.

    Implements bootstrap resampling per ATLAS Checklist Section 1.7:
    - Shuffle trade order to test robustness to sequence
    - Bootstrap with replacement for statistical significance
    - Options-specific IV and theta shock modeling
    - Calculate confidence intervals and probability metrics

    Acceptance Criteria (equities):
    - 95% CI for Sharpe excludes 0
    - P(Loss) < 20%
    - P(Ruin >50% DD) < 5%

    For options strategies, use MonteCarloConfigOptions with looser thresholds.

    Example:
        config = MonteCarloConfig(n_simulations=10000)
        validator = MonteCarloValidator(config)

        results = validator.validate(trades, account_size=10000)
        print(results.summary())

        if results.passes_validation:
            print("Strategy is robust to trade order variations")
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """
        Initialize Monte Carlo validator.

        Args:
            config: Monte Carlo configuration. Uses defaults if None.
        """
        self.config = config or MonteCarloConfig()
        self._rng: Optional[np.random.Generator] = None

    def validate(
        self,
        trades: pd.DataFrame,
        account_size: float,
        is_options: bool = False
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation on trade sequence.

        Args:
            trades: DataFrame with columns: 'pnl' (required), 'pnl_pct' (optional).
                    For options: 'entry_iv', 'exit_iv', 'theta_cost' (optional).
            account_size: Starting account size for simulation.
            is_options: Whether trades are options (enables IV/theta shocks).

        Returns:
            MonteCarloResults with probabilities and confidence intervals.
        """
        logger.info(f"Starting Monte Carlo simulation: {len(trades)} trades, {self.config.n_simulations} iterations")

        # Initialize random generator
        self._rng = np.random.default_rng(self.config.seed)

        # Validate input
        if trades is None or len(trades) == 0:
            logger.warning("No trades provided for Monte Carlo simulation")
            return self._create_empty_results("No trades provided")

        if 'pnl' not in trades.columns:
            logger.warning("Trades DataFrame missing 'pnl' column")
            return self._create_empty_results("Missing 'pnl' column")

        # Convert trades to list of trade records
        trade_records = self._parse_trades(trades, is_options)

        if len(trade_records) < 5:
            logger.warning(f"Insufficient trades ({len(trade_records)}) for Monte Carlo")
            return self._create_empty_results("Insufficient trades (minimum 5)")

        # Calculate original metrics
        original_equity = self._calculate_equity_curve(trade_records, account_size)
        original_returns = self._calculate_returns(original_equity)
        original_sharpe = self._calculate_sharpe(original_returns)
        original_max_dd = self._calculate_max_drawdown(original_equity)

        logger.info(f"Original metrics: Sharpe={original_sharpe:.2f}, MaxDD={original_max_dd:.1%}")

        # Run simulations
        simulated_sharpes = []
        simulated_max_dds = []
        simulated_final_returns = []

        for i in range(self.config.n_simulations):
            # Bootstrap resample with replacement
            sim_trades = self._bootstrap_trades(trade_records)

            # Apply options-specific shocks if applicable
            if is_options and isinstance(self.config, MonteCarloConfigOptions):
                if self.config.apply_options_shocks:
                    sim_trades = self._apply_options_shocks(sim_trades)

            # Calculate simulation metrics
            sim_equity = self._calculate_equity_curve(sim_trades, account_size)
            sim_returns = self._calculate_returns(sim_equity)

            simulated_sharpes.append(self._calculate_sharpe(sim_returns))
            simulated_max_dds.append(self._calculate_max_drawdown(sim_equity))
            simulated_final_returns.append(
                (sim_equity[-1] / account_size - 1) if sim_equity[-1] > 0 else -1.0
            )

            # Log progress every 1000 iterations
            if (i + 1) % 1000 == 0:
                logger.debug(f"Completed {i + 1}/{self.config.n_simulations} simulations")

        # Convert to numpy arrays for calculations
        simulated_sharpes = np.array(simulated_sharpes)
        simulated_max_dds = np.array(simulated_max_dds)
        simulated_final_returns = np.array(simulated_final_returns)

        # Calculate statistics
        results = self._calculate_statistics(
            original_sharpe=original_sharpe,
            original_max_dd=original_max_dd,
            simulated_sharpes=simulated_sharpes,
            simulated_max_dds=simulated_max_dds,
            simulated_final_returns=simulated_final_returns
        )

        logger.info(f"Monte Carlo complete: {'PASSED' if results.passes_validation else 'FAILED'}")

        return results

    def _parse_trades(
        self,
        trades: pd.DataFrame,
        is_options: bool
    ) -> List[TradeRecord]:
        """
        Parse trades DataFrame into TradeRecord list.

        Args:
            trades: DataFrame with trade data
            is_options: Whether trades are options

        Returns:
            List of TradeRecord objects
        """
        records = []

        for _, row in trades.iterrows():
            pnl = float(row['pnl'])

            # Get pnl_pct if available, else calculate estimate
            if 'pnl_pct' in row and pd.notna(row['pnl_pct']):
                pnl_pct = float(row['pnl_pct'])
            else:
                # Estimate as percentage of typical trade size
                pnl_pct = 0.0  # Will use absolute P/L

            record = TradeRecord(
                pnl=pnl,
                pnl_pct=pnl_pct,
                is_winner=(pnl > 0),
                is_options=is_options,
                entry_iv=float(row['entry_iv']) if 'entry_iv' in row and pd.notna(row.get('entry_iv')) else None,
                exit_iv=float(row['exit_iv']) if 'exit_iv' in row and pd.notna(row.get('exit_iv')) else None,
                theta_cost=float(row['theta_cost']) if 'theta_cost' in row and pd.notna(row.get('theta_cost')) else None
            )
            records.append(record)

        return records

    def _bootstrap_trades(self, trades: List[TradeRecord]) -> List[TradeRecord]:
        """
        Bootstrap resample trades with replacement.

        This randomizes trade order and can duplicate/omit trades,
        testing robustness to trade sequence.

        Args:
            trades: Original trade list

        Returns:
            Resampled trade list (same length as original)
        """
        n = len(trades)
        indices = self._rng.choice(n, size=n, replace=True)
        return [trades[i] for i in indices]

    def _apply_options_shocks(self, trades: List[TradeRecord]) -> List[TradeRecord]:
        """
        Apply IV and theta shocks to options trades.

        Per ATLAS Checklist Section 9.5.2:
        - IV shock: +/- 20% standard deviation on entry/exit IV
        - Theta shock: Up to 50% worse theta decay on losing trades

        Args:
            trades: Trade list with options data

        Returns:
            Modified trade list with shocked P/L values
        """
        if not isinstance(self.config, MonteCarloConfigOptions):
            return trades

        # Check if shocks are disabled
        if not self.config.apply_options_shocks:
            return trades

        shocked_trades = []

        for trade in trades:
            new_pnl = trade.pnl

            # Apply IV shock
            if trade.entry_iv is not None and trade.exit_iv is not None:
                # Random IV shock applied to entry
                iv_shock_pct = self._rng.normal(0, self.config.iv_shock_std)

                # Estimate P/L impact from IV change (vega exposure)
                # Simplified: assume vega impact proportional to IV shock
                iv_impact_pct = iv_shock_pct * 0.5  # 50% of IV shock affects P/L
                iv_impact = abs(new_pnl) * iv_impact_pct
                new_pnl += iv_impact

            # Apply theta shock to losers
            if not trade.is_winner and trade.theta_cost is not None:
                # Losers can have up to 50% worse theta decay
                theta_multiplier = self._rng.uniform(1.0, self.config.theta_shock_max)
                additional_theta = trade.theta_cost * (theta_multiplier - 1.0)
                new_pnl -= abs(additional_theta)

            shocked_trade = TradeRecord(
                pnl=new_pnl,
                pnl_pct=trade.pnl_pct * (new_pnl / trade.pnl) if trade.pnl != 0 else 0.0,
                is_winner=(new_pnl > 0),
                is_options=trade.is_options,
                entry_iv=trade.entry_iv,
                exit_iv=trade.exit_iv,
                theta_cost=trade.theta_cost
            )
            shocked_trades.append(shocked_trade)

        return shocked_trades

    def _calculate_equity_curve(
        self,
        trades: List[TradeRecord],
        starting_capital: float
    ) -> np.ndarray:
        """
        Calculate equity curve from trade sequence.

        Args:
            trades: List of trades in execution order
            starting_capital: Initial account value

        Returns:
            Array of equity values (length = len(trades) + 1)
        """
        equity = [starting_capital]

        for trade in trades:
            new_equity = equity[-1] + trade.pnl
            equity.append(new_equity)

        return np.array(equity)

    def _calculate_returns(self, equity: np.ndarray) -> np.ndarray:
        """
        Calculate return series from equity curve.

        Args:
            equity: Equity curve array

        Returns:
            Array of periodic returns
        """
        if len(equity) < 2:
            return np.array([])

        # Avoid division by zero
        equity_safe = np.where(equity[:-1] == 0, 1e-10, equity[:-1])
        returns = (equity[1:] - equity[:-1]) / equity_safe

        return returns

    def _calculate_sharpe(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio from returns.

        Args:
            returns: Array of periodic returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Periods per year for annualization

        Returns:
            Annualized Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        # Remove any NaN or infinite values
        returns = returns[np.isfinite(returns)]

        if len(returns) < 2:
            return 0.0

        # Convert annual risk-free to periodic
        rf_per_period = risk_free_rate / periods_per_year

        excess_returns = returns - rf_per_period
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)

        if std_excess == 0 or np.isnan(std_excess):
            return 0.0

        sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)

        return float(sharpe) if np.isfinite(sharpe) else 0.0

    def _calculate_max_drawdown(self, equity: np.ndarray) -> float:
        """
        Calculate maximum drawdown from equity curve.

        Args:
            equity: Equity curve array

        Returns:
            Maximum drawdown as positive decimal (0.25 = 25%)
        """
        if len(equity) < 2:
            return 0.0

        # Calculate running maximum
        running_max = np.maximum.accumulate(equity)

        # Avoid division by zero
        running_max_safe = np.where(running_max == 0, 1e-10, running_max)

        # Calculate drawdown
        drawdown = (running_max - equity) / running_max_safe

        max_dd = np.max(drawdown)

        return float(max_dd) if np.isfinite(max_dd) else 0.0

    def _calculate_statistics(
        self,
        original_sharpe: float,
        original_max_dd: float,
        simulated_sharpes: np.ndarray,
        simulated_max_dds: np.ndarray,
        simulated_final_returns: np.ndarray
    ) -> MonteCarloResults:
        """
        Calculate final statistics and determine pass/fail.

        Args:
            original_sharpe: Sharpe from actual trade sequence
            original_max_dd: Max DD from actual trade sequence
            simulated_sharpes: Array of simulated Sharpe ratios
            simulated_max_dds: Array of simulated max drawdowns
            simulated_final_returns: Array of simulated final returns

        Returns:
            MonteCarloResults with all metrics
        """
        # Calculate confidence intervals
        alpha = 1 - self.config.confidence_level
        lower_pct = alpha / 2 * 100
        upper_pct = (1 - alpha / 2) * 100

        sharpe_ci = (
            float(np.percentile(simulated_sharpes, lower_pct)),
            float(np.percentile(simulated_sharpes, upper_pct))
        )

        max_dd_ci = (
            float(np.percentile(simulated_max_dds, 5)),
            float(np.percentile(simulated_max_dds, 95))
        )

        return_ci = (
            float(np.percentile(simulated_final_returns, 5)),
            float(np.percentile(simulated_final_returns, 95))
        )

        # Calculate mean and std of simulated Sharpe
        simulated_sharpe_mean = float(np.mean(simulated_sharpes))
        simulated_sharpe_std = float(np.std(simulated_sharpes, ddof=1))

        # Calculate 95th percentile max DD
        simulated_max_dd_95 = float(np.percentile(simulated_max_dds, 95))

        # Calculate probabilities
        probability_of_loss = float(np.sum(simulated_final_returns < 0) / len(simulated_final_returns))
        probability_of_ruin = float(np.sum(simulated_max_dds > self.config.ruin_threshold) / len(simulated_max_dds))

        # Determine pass/fail with reasons
        failure_reasons = []

        # Check if 95% CI excludes zero
        if sharpe_ci[0] <= 0:
            failure_reasons.append(
                f"95% CI for Sharpe [{sharpe_ci[0]:.2f}, {sharpe_ci[1]:.2f}] includes zero"
            )

        # Check probability of loss
        if probability_of_loss > self.config.max_probability_of_loss:
            failure_reasons.append(
                f"P(Loss) {probability_of_loss:.1%} exceeds maximum {self.config.max_probability_of_loss:.1%}"
            )

        # Check probability of ruin
        if probability_of_ruin > self.config.max_probability_of_ruin:
            failure_reasons.append(
                f"P(Ruin) {probability_of_ruin:.1%} exceeds maximum {self.config.max_probability_of_ruin:.1%}"
            )

        passes_validation = len(failure_reasons) == 0

        return MonteCarloResults(
            original_sharpe=original_sharpe,
            simulated_sharpe_mean=simulated_sharpe_mean,
            simulated_sharpe_std=simulated_sharpe_std,
            sharpe_95_ci=sharpe_ci,
            original_max_dd=original_max_dd,
            simulated_max_dd_95=simulated_max_dd_95,
            max_dd_95_ci=max_dd_ci,
            return_95_ci=return_ci,
            probability_of_loss=probability_of_loss,
            probability_of_ruin=probability_of_ruin,
            n_simulations=self.config.n_simulations,
            passes_validation=passes_validation,
            failure_reasons=failure_reasons,
            simulated_sharpes=simulated_sharpes,
            simulated_max_dds=simulated_max_dds,
            simulated_returns=simulated_final_returns,
            max_probability_of_loss=self.config.max_probability_of_loss,
            max_probability_of_ruin=self.config.max_probability_of_ruin,
        )

    def _create_empty_results(self, reason: str) -> MonteCarloResults:
        """Create empty results for failed validation."""
        return MonteCarloResults(
            original_sharpe=0.0,
            simulated_sharpe_mean=0.0,
            simulated_sharpe_std=0.0,
            sharpe_95_ci=(0.0, 0.0),
            original_max_dd=0.0,
            simulated_max_dd_95=0.0,
            max_dd_95_ci=(0.0, 0.0),
            return_95_ci=(0.0, 0.0),
            probability_of_loss=1.0,
            probability_of_ruin=1.0,
            n_simulations=0,
            passes_validation=False,
            failure_reasons=[reason],
            simulated_sharpes=None,
            simulated_max_dds=None,
            simulated_returns=None,
            max_probability_of_loss=self.config.max_probability_of_loss,
            max_probability_of_ruin=self.config.max_probability_of_ruin,
        )

    def passes(self, results: MonteCarloResults) -> bool:
        """
        Check if results meet acceptance criteria.

        Convenience method matching ValidatorProtocol.

        Args:
            results: Results from validate() method

        Returns:
            True if validation passes
        """
        return results.passes_validation


def generate_synthetic_trades(
    n_trades: int,
    win_rate: float = 0.55,
    avg_winner: float = 100.0,
    avg_loser: float = -80.0,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate synthetic trades for testing Monte Carlo validator.

    Args:
        n_trades: Number of trades to generate
        win_rate: Probability of winning trade
        avg_winner: Average winning trade P/L
        avg_loser: Average losing trade P/L (should be negative)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with 'pnl' column
    """
    rng = np.random.default_rng(seed)

    trades = []
    for _ in range(n_trades):
        is_winner = rng.random() < win_rate
        if is_winner:
            pnl = rng.normal(avg_winner, avg_winner * 0.3)
        else:
            pnl = rng.normal(avg_loser, abs(avg_loser) * 0.3)
        trades.append({'pnl': pnl})

    return pd.DataFrame(trades)

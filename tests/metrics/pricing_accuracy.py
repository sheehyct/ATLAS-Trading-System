"""
Session 82: Pricing Accuracy Metrics Module

Provides standardized metrics for comparing option pricing accuracy
between ThetaData (real market data) and Black-Scholes (synthetic).

Metrics:
- MAE (Mean Absolute Error): Average absolute price difference
- RMSE (Root Mean Squared Error): Penalizes larger errors
- MAPE (Mean Absolute Percentage Error): Scale-independent accuracy
- Correlation: How well prices move together
- Bias: Systematic over/under-pricing

Usage:
    from tests.metrics.pricing_accuracy import PricingAccuracyMetrics

    metrics = PricingAccuracyMetrics()
    metrics.add_comparison(real_price=5.25, synthetic_price=5.10, option_type='call')
    results = metrics.calculate()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class PricingComparison:
    """Single pricing comparison between real and synthetic."""
    real_price: float
    synthetic_price: float
    underlying_price: float
    strike: float
    dte: int  # Days to expiration
    option_type: str  # 'call' or 'put'
    moneyness: str = 'atm'  # 'itm', 'atm', 'otm'
    real_iv: Optional[float] = None
    synthetic_iv: Optional[float] = None

    def __post_init__(self):
        """Calculate moneyness if not set."""
        if self.moneyness == 'atm':
            ratio = self.underlying_price / self.strike
            if self.option_type == 'call':
                if ratio > 1.02:
                    self.moneyness = 'itm'
                elif ratio < 0.98:
                    self.moneyness = 'otm'
            else:  # put
                if ratio < 0.98:
                    self.moneyness = 'itm'
                elif ratio > 1.02:
                    self.moneyness = 'otm'


@dataclass
class AccuracyResults:
    """Results of pricing accuracy analysis."""
    # Core metrics
    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error (%)
    correlation: float  # Pearson correlation
    bias: float  # Mean error (synthetic - real), positive = over-pricing

    # Sample info
    n_samples: int
    real_mean: float
    synthetic_mean: float

    # Breakdown by option type
    call_mae: Optional[float] = None
    put_mae: Optional[float] = None

    # Breakdown by moneyness
    itm_mae: Optional[float] = None
    atm_mae: Optional[float] = None
    otm_mae: Optional[float] = None

    # IV comparison
    iv_mae: Optional[float] = None
    iv_correlation: Optional[float] = None

    # Thresholds
    within_1pct: float = 0.0  # % of samples within 1%
    within_5pct: float = 0.0  # % of samples within 5%
    within_10pct: float = 0.0  # % of samples within 10%

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'mape': self.mape,
            'correlation': self.correlation,
            'bias': self.bias,
            'n_samples': self.n_samples,
            'real_mean': self.real_mean,
            'synthetic_mean': self.synthetic_mean,
            'call_mae': self.call_mae,
            'put_mae': self.put_mae,
            'itm_mae': self.itm_mae,
            'atm_mae': self.atm_mae,
            'otm_mae': self.otm_mae,
            'iv_mae': self.iv_mae,
            'iv_correlation': self.iv_correlation,
            'within_1pct': self.within_1pct,
            'within_5pct': self.within_5pct,
            'within_10pct': self.within_10pct,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def passes_threshold(self, max_mape: float = 15.0, min_correlation: float = 0.9) -> bool:
        """Check if results meet accuracy thresholds."""
        return self.mape <= max_mape and self.correlation >= min_correlation


class PricingAccuracyMetrics:
    """
    Calculate pricing accuracy metrics for ThetaData vs Black-Scholes comparison.

    Designed for Session 82 validation of ThetaData integration.
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.comparisons: List[PricingComparison] = []

    def add_comparison(
        self,
        real_price: float,
        synthetic_price: float,
        underlying_price: float = 100.0,
        strike: float = 100.0,
        dte: int = 30,
        option_type: str = 'call',
        real_iv: Optional[float] = None,
        synthetic_iv: Optional[float] = None,
    ) -> None:
        """
        Add a single pricing comparison.

        Args:
            real_price: ThetaData market price
            synthetic_price: Black-Scholes calculated price
            underlying_price: Current underlying price
            strike: Option strike price
            dte: Days to expiration
            option_type: 'call' or 'put'
            real_iv: ThetaData implied volatility
            synthetic_iv: Black-Scholes assumed IV
        """
        if real_price <= 0 or synthetic_price <= 0:
            return  # Skip invalid prices

        comparison = PricingComparison(
            real_price=real_price,
            synthetic_price=synthetic_price,
            underlying_price=underlying_price,
            strike=strike,
            dte=dte,
            option_type=option_type.lower(),
            real_iv=real_iv,
            synthetic_iv=synthetic_iv,
        )
        self.comparisons.append(comparison)

    def add_comparisons_from_dataframe(
        self,
        df,
        real_price_col: str = 'market_price',
        synthetic_price_col: str = 'bs_price',
        underlying_col: str = 'underlying_price',
        strike_col: str = 'strike',
        dte_col: str = 'days_held',
        option_type_col: str = 'option_type',
    ) -> int:
        """
        Add comparisons from a pandas DataFrame.

        Returns number of comparisons added.
        """
        added = 0
        for _, row in df.iterrows():
            try:
                self.add_comparison(
                    real_price=row[real_price_col],
                    synthetic_price=row[synthetic_price_col],
                    underlying_price=row.get(underlying_col, 100.0),
                    strike=row.get(strike_col, 100.0),
                    dte=int(row.get(dte_col, 30)),
                    option_type=row.get(option_type_col, 'call'),
                )
                added += 1
            except (KeyError, ValueError):
                continue
        return added

    def calculate(self) -> AccuracyResults:
        """
        Calculate all accuracy metrics.

        Returns:
            AccuracyResults with comprehensive metrics
        """
        if len(self.comparisons) == 0:
            return AccuracyResults(
                mae=float('inf'),
                rmse=float('inf'),
                mape=100.0,
                correlation=0.0,
                bias=0.0,
                n_samples=0,
                real_mean=0.0,
                synthetic_mean=0.0,
            )

        # Extract arrays
        real = np.array([c.real_price for c in self.comparisons])
        synthetic = np.array([c.synthetic_price for c in self.comparisons])
        errors = synthetic - real  # positive = over-pricing

        # Core metrics
        mae = np.abs(errors).mean()
        rmse = np.sqrt((errors ** 2).mean())

        # MAPE (handle division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_errors = np.abs(errors) / real * 100
            pct_errors = pct_errors[~np.isinf(pct_errors) & ~np.isnan(pct_errors)]
            mape = pct_errors.mean() if len(pct_errors) > 0 else 100.0

        # Correlation
        if len(real) > 1:
            correlation = np.corrcoef(real, synthetic)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # Bias
        bias = errors.mean()

        # Threshold analysis
        pct_diff = np.abs(errors) / real * 100
        within_1pct = (pct_diff <= 1.0).mean() * 100
        within_5pct = (pct_diff <= 5.0).mean() * 100
        within_10pct = (pct_diff <= 10.0).mean() * 100

        # Call/Put breakdown
        call_errors = [e for c, e in zip(self.comparisons, errors) if c.option_type == 'call']
        put_errors = [e for c, e in zip(self.comparisons, errors) if c.option_type == 'put']
        call_mae = np.abs(call_errors).mean() if call_errors else None
        put_mae = np.abs(put_errors).mean() if put_errors else None

        # Moneyness breakdown
        itm_errors = [e for c, e in zip(self.comparisons, errors) if c.moneyness == 'itm']
        atm_errors = [e for c, e in zip(self.comparisons, errors) if c.moneyness == 'atm']
        otm_errors = [e for c, e in zip(self.comparisons, errors) if c.moneyness == 'otm']
        itm_mae = np.abs(itm_errors).mean() if itm_errors else None
        atm_mae = np.abs(atm_errors).mean() if atm_errors else None
        otm_mae = np.abs(otm_errors).mean() if otm_errors else None

        # IV comparison
        real_ivs = [c.real_iv for c in self.comparisons if c.real_iv is not None]
        synth_ivs = [c.synthetic_iv for c in self.comparisons if c.synthetic_iv is not None]
        if len(real_ivs) > 0 and len(synth_ivs) > 0:
            min_len = min(len(real_ivs), len(synth_ivs))
            iv_errors = np.array(synth_ivs[:min_len]) - np.array(real_ivs[:min_len])
            iv_mae = np.abs(iv_errors).mean()
            iv_correlation = np.corrcoef(real_ivs[:min_len], synth_ivs[:min_len])[0, 1] if min_len > 1 else 0.0
        else:
            iv_mae = None
            iv_correlation = None

        return AccuracyResults(
            mae=mae,
            rmse=rmse,
            mape=mape,
            correlation=correlation,
            bias=bias,
            n_samples=len(self.comparisons),
            real_mean=real.mean(),
            synthetic_mean=synthetic.mean(),
            call_mae=call_mae,
            put_mae=put_mae,
            itm_mae=itm_mae,
            atm_mae=atm_mae,
            otm_mae=otm_mae,
            iv_mae=iv_mae,
            iv_correlation=iv_correlation if iv_correlation and not np.isnan(iv_correlation) else None,
            within_1pct=within_1pct,
            within_5pct=within_5pct,
            within_10pct=within_10pct,
        )

    def clear(self) -> None:
        """Clear all comparisons."""
        self.comparisons = []

    def summary(self) -> str:
        """Generate human-readable summary."""
        results = self.calculate()

        lines = [
            "=" * 50,
            "PRICING ACCURACY SUMMARY",
            "=" * 50,
            f"Samples: {results.n_samples}",
            "",
            "--- Core Metrics ---",
            f"MAE:         ${results.mae:.4f}",
            f"RMSE:        ${results.rmse:.4f}",
            f"MAPE:        {results.mape:.2f}%",
            f"Correlation: {results.correlation:.4f}",
            f"Bias:        ${results.bias:+.4f} ({'over' if results.bias > 0 else 'under'}-pricing)",
            "",
            "--- Threshold Analysis ---",
            f"Within 1%:   {results.within_1pct:.1f}%",
            f"Within 5%:   {results.within_5pct:.1f}%",
            f"Within 10%:  {results.within_10pct:.1f}%",
        ]

        if results.call_mae is not None or results.put_mae is not None:
            lines.extend([
                "",
                "--- By Option Type ---",
            ])
            if results.call_mae is not None:
                lines.append(f"Call MAE:    ${results.call_mae:.4f}")
            if results.put_mae is not None:
                lines.append(f"Put MAE:     ${results.put_mae:.4f}")

        if results.iv_mae is not None:
            lines.extend([
                "",
                "--- IV Comparison ---",
                f"IV MAE:      {results.iv_mae:.4f}",
            ])
            if results.iv_correlation is not None:
                lines.append(f"IV Corr:     {results.iv_correlation:.4f}")

        lines.append("=" * 50)

        return "\n".join(lines)


# Convenience functions for quick analysis
def compare_prices(
    real_prices: List[float],
    synthetic_prices: List[float],
) -> AccuracyResults:
    """
    Quick comparison of two price lists.

    Args:
        real_prices: List of real (market) prices
        synthetic_prices: List of synthetic (model) prices

    Returns:
        AccuracyResults
    """
    metrics = PricingAccuracyMetrics()
    for real, synth in zip(real_prices, synthetic_prices):
        metrics.add_comparison(real_price=real, synthetic_price=synth)
    return metrics.calculate()


def validate_pricing_accuracy(
    results: AccuracyResults,
    max_mape: float = 15.0,
    min_correlation: float = 0.9,
    min_samples: int = 10,
) -> Tuple[bool, str]:
    """
    Validate pricing accuracy meets thresholds.

    Returns:
        (passed: bool, message: str)
    """
    issues = []

    if results.n_samples < min_samples:
        issues.append(f"Insufficient samples: {results.n_samples} < {min_samples}")

    if results.mape > max_mape:
        issues.append(f"MAPE too high: {results.mape:.2f}% > {max_mape}%")

    if results.correlation < min_correlation:
        issues.append(f"Correlation too low: {results.correlation:.4f} < {min_correlation}")

    if issues:
        return False, "; ".join(issues)

    return True, f"Passed: MAPE={results.mape:.2f}%, Corr={results.correlation:.4f}"


if __name__ == "__main__":
    # Self-test
    print("Testing PricingAccuracyMetrics...")

    metrics = PricingAccuracyMetrics()

    # Add some test comparisons
    test_data = [
        (5.25, 5.10, 'call'),  # Real slightly higher
        (3.50, 3.45, 'call'),
        (2.10, 2.30, 'put'),   # Synthetic over-pricing
        (4.00, 3.80, 'put'),
        (6.75, 6.70, 'call'),
        (1.50, 1.55, 'put'),
    ]

    for real, synth, opt_type in test_data:
        metrics.add_comparison(
            real_price=real,
            synthetic_price=synth,
            option_type=opt_type,
        )

    print(metrics.summary())

    # Validate
    results = metrics.calculate()
    passed, msg = validate_pricing_accuracy(results, max_mape=15.0)
    print(f"\nValidation: {'PASSED' if passed else 'FAILED'} - {msg}")

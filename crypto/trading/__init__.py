"""
Trading logic module for crypto derivatives.

Provides position sizing, fee calculations, and beta analysis.
"""

from .sizing import (
    calculate_position_size,
    calculate_position_size_leverage_first,
    should_skip_trade,
    calculate_stop_distance_for_leverage,
)

from .fees import (
    calculate_fee,
    calculate_round_trip_fee,
    calculate_breakeven_move,
    calculate_num_contracts,
    calculate_notional_from_contracts,
    create_coinbase_fee_func,
    create_fixed_pct_fee_func,
    analyze_fee_impact,
    TAKER_FEE_RATE,
    MAKER_FEE_RATE,
    MIN_FEE_PER_CONTRACT,
    CONTRACT_MULTIPLIERS,
)

from .beta import (
    calculate_effective_multiplier,
    get_effective_multipliers,
    rank_by_capital_efficiency,
    project_pnl_on_btc_move,
    compare_instruments_on_btc_move,
    calculate_rolling_beta,
    calculate_beta_from_ranges,
    update_beta_from_current_levels,
    calculate_beta_adjusted_size,
    select_best_instrument,
    CRYPTO_BETA_TO_BTC,
    INTRADAY_LEVERAGE,
    SWING_LEVERAGE,
)

__all__ = [
    # Sizing
    "calculate_position_size",
    "calculate_position_size_leverage_first",
    "should_skip_trade",
    "calculate_stop_distance_for_leverage",
    # Fees
    "calculate_fee",
    "calculate_round_trip_fee",
    "calculate_breakeven_move",
    "calculate_num_contracts",
    "calculate_notional_from_contracts",
    "create_coinbase_fee_func",
    "create_fixed_pct_fee_func",
    "analyze_fee_impact",
    "TAKER_FEE_RATE",
    "MAKER_FEE_RATE",
    "MIN_FEE_PER_CONTRACT",
    "CONTRACT_MULTIPLIERS",
    # Beta
    "calculate_effective_multiplier",
    "get_effective_multipliers",
    "rank_by_capital_efficiency",
    "project_pnl_on_btc_move",
    "compare_instruments_on_btc_move",
    "calculate_rolling_beta",
    "calculate_beta_from_ranges",
    "update_beta_from_current_levels",
    "calculate_beta_adjusted_size",
    "select_best_instrument",
    "CRYPTO_BETA_TO_BTC",
    "INTRADAY_LEVERAGE",
    "SWING_LEVERAGE",
]

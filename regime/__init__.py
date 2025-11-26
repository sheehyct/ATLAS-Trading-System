"""
Regime Detection Framework

This module provides market regime detection and regime-based capital allocation
for the ATLAS trading system.

Modules:
    academic_jump_model: Academic clustering-based regime detection (PRODUCTION)
    academic_features: Feature calculation for regime detection
    vix_acceleration: VIX spike detection for flash crash override
    regime_allocator: Regime-based capital allocation logic

Market Regimes:
    TREND_BULL: Strong bullish trend (positive Sortino ratio)
    TREND_NEUTRAL: Choppy/sideways market (low volatility)
    TREND_BEAR: Strong bearish trend (negative Sortino ratio)
    CRASH: Extreme volatility event (VIX spike or high downside deviation)

Note: The legacy JumpModel was deprecated in Session 12 and removed in Session 80+.
All production code uses AcademicJumpModel (Sessions 12-19).
"""

from regime.academic_jump_model import AcademicJumpModel

__all__ = [
    'AcademicJumpModel',
]

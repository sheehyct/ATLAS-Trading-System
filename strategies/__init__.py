"""
Strategies module for ATLAS algorithmic trading strategies.

Contains:
- base_strategy.py: Abstract base class for all strategies
- orb.py: Opening Range Breakout (ORB) strategy with ATR-based position sizing
- high_momentum_52w.py: 52-Week High Momentum strategy (foundation)
- quality_momentum.py: Quality-Momentum Combination strategy (all-weather)
- semi_vol_momentum.py: Semi-Volatility Momentum strategy (volatility-scaled)
- ibs_mean_reversion.py: IBS Mean Reversion strategy (short-term reversals)

Strategy Implementation Status:
- BaseStrategy: COMPLETE (abstract base class)
- ORBStrategy: COMPLETE (needs volume confirmation modification)
- HighMomentum52W: COMPLETE (validated Session 36)
- QualityMomentum: SKELETON (needs fundamental data integration)
- SemiVolMomentum: SKELETON (ready for testing)
- IBSMeanReversion: SKELETON (ready for testing)
"""

from .base_strategy import BaseStrategy, StrategyConfig
from .orb import ORBStrategy
from .high_momentum_52w import HighMomentum52W
from .quality_momentum import QualityMomentum
from .semi_vol_momentum import SemiVolMomentum
from .ibs_mean_reversion import IBSMeanReversion

__all__ = [
    'BaseStrategy',
    'StrategyConfig',
    'ORBStrategy',
    'HighMomentum52W',
    'QualityMomentum',
    'SemiVolMomentum',
    'IBSMeanReversion'
]

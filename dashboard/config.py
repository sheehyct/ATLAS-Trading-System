"""
ATLAS Dashboard Configuration

This module contains all configuration constants for the ATLAS Plotly Dash dashboard,
including visualization settings, color schemes, z-ordering rules, and API configurations.

Based on:
- PLOTLY_DASH_DASHBOARD_GUIDE.md
- Visualization best practices for trading dashboards
- TradingView professional standards

Session 70: Updated to use centralized config.settings for all credentials.
"""

from pathlib import Path
from typing import Dict, Any

# Use centralized config (loads from root .env with all credentials)
import os
import warnings

# ============================================
# PATH CONFIGURATION
# ============================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
ASSETS_DIR = BASE_DIR / 'dashboard' / 'assets'
REGIME_DIR = BASE_DIR / 'regime'
STRATEGIES_DIR = BASE_DIR / 'strategies'

# ============================================
# API CONFIGURATION
# ============================================

# Get Alpaca credentials from centralized config
# Wrapped in try/except for Railway deployment where credentials may be missing
try:
    from config.settings import get_alpaca_credentials, get_default_account
    _default_account = get_default_account()  # Uses DEFAULT_ACCOUNT env var (LARGE, MID, or SMALL)
    _alpaca_creds = get_alpaca_credentials(_default_account)
    ALPACA_CONFIG = {
        'api_key': _alpaca_creds['api_key'],
        'secret_key': _alpaca_creds['secret_key'],
        'base_url': _alpaca_creds['base_url'],
        'paper': True  # Use paper trading by default
    }
except Exception as e:
    warnings.warn(f"Could not load Alpaca credentials: {e}. Live data features disabled.")
    ALPACA_CONFIG = {
        'api_key': None,
        'secret_key': None,
        'base_url': 'https://paper-api.alpaca.markets',
        'paper': True
    }

# ============================================
# DASHBOARD SETTINGS
# ============================================

DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.getenv('PORT', 8050)),  # Railway sets PORT env var
    'debug': os.getenv('DASH_DEBUG', 'false').lower() == 'true',  # False for production
    'refresh_interval': 120000,  # 2 minutes for casual monitoring (increase frequency when strategies deployed)
    'cache_timeout': 300,  # 5 minutes cache
}

# ============================================
# COLOR SCHEMES (TradingView Professional)
# ============================================

# Primary Trading Colors
COLORS = {
    # Bull/Upward Movement
    'bull_primary': '#00ff55',      # Bright green
    'bull_secondary': '#26A69A',    # Teal green (softer)
    'bull_fill': 'rgba(0, 255, 85, 0.2)',

    # Bear/Downward Movement
    'bear_primary': '#ed4807',      # Bright red-orange
    'bear_secondary': '#EF5350',    # Softer red
    'bear_fill': 'rgba(237, 72, 7, 0.2)',

    # Neutral/Background
    'background_dark': '#090008',   # Near black
    'background_medium': '#2E2E2E', # Dark gray
    'background_light': '#3E3E3E',  # Medium gray

    # Grid and Accents
    'grid': '#333333',              # Subtle grid lines
    'grid_major': '#444444',        # Major grid lines
    'text_primary': '#E0E0E0',      # Light gray text
    'text_secondary': '#A0A0A0',    # Muted gray text

    # Status Colors
    'success': '#00C853',           # Success green
    'warning': '#FFC107',           # Warning amber
    'danger': '#FF1744',            # Danger red
    'info': '#2196F3',              # Info blue

    # Chart Elements
    'price_line': '#FFFFFF',        # White for main price
    'volume_up': 'rgba(38, 166, 154, 0.6)',   # Teal volume bars
    'volume_down': 'rgba(239, 83, 80, 0.6)',  # Red volume bars
}

# ============================================
# REGIME DETECTION COLORS
# ============================================

REGIME_COLORS = {
    'TREND_BULL': 'rgba(0, 255, 85, 0.2)',      # Bright green with 20% opacity
    'TREND_NEUTRAL': 'rgba(128, 128, 128, 0.2)', # Gray with 20% opacity
    'TREND_BEAR': 'rgba(255, 165, 0, 0.2)',     # Orange with 20% opacity
    'CRASH': 'rgba(237, 72, 7, 0.3)',           # Red-orange with 30% opacity
}

REGIME_BORDER_COLORS = {
    'TREND_BULL': '#00ff55',
    'TREND_NEUTRAL': '#808080',
    'TREND_BEAR': '#FFA500',
    'CRASH': '#ed4807',
}

REGIME_TEXT_COLORS = {
    'TREND_BULL': '#00ff55',
    'TREND_NEUTRAL': '#A0A0A0',
    'TREND_BEAR': '#FFA500',
    'CRASH': '#ed4807',
}

# ============================================
# STRAT PATTERN COLORS
# ============================================

STRAT_COLORS = {
    '1': 'rgba(128, 128, 128, 0.3)',    # Inside bar - Gray
    '2U': 'rgba(0, 255, 85, 0.3)',      # Directional Up - Green
    '2D': 'rgba(237, 72, 7, 0.3)',      # Directional Down - Red
    '3': 'rgba(255, 255, 0, 0.3)',      # Outside bar - Yellow
}

STRAT_BORDER_COLORS = {
    '1': '#808080',
    '2U': '#00ff55',
    '2D': '#ed4807',
    '3': '#FFFF00',
}

# ============================================
# Z-ORDERING RULES (Bottom to Top)
# ============================================

# Plotly layering rules for proper visualization
Z_ORDER = {
    'regime_shading': 0,      # Background regime shading (lowest)
    'grid': 1,                # Grid lines
    'volume': 2,              # Volume bars
    'indicators': 3,          # Technical indicators (MA, Bollinger, etc.)
    'price_line': 4,          # Main price line/candlesticks
    'trade_markers': 5,       # Entry/exit markers
    'annotations': 6,         # Text annotations (highest)
}

# Plotly uses 'layer' parameter: 'below' (below traces) or 'above' (above traces)
PLOTLY_LAYERS = {
    'regime_shading': 'below',
    'grid': 'below',
    'volume': 'below',
    'indicators': 'above',
    'price_line': 'above',
    'trade_markers': 'above',
    'annotations': 'above',
}

# ============================================
# TRADE MARKER STYLES
# ============================================

TRADE_MARKERS = {
    # Entry markers
    'long_entry': {
        'symbol': 'triangle-up',
        'size': 12,
        'color': COLORS['bull_primary'],
        'line': {'color': '#FFFFFF', 'width': 2},  # White outline
    },
    'short_entry': {
        'symbol': 'triangle-down',
        'size': 12,
        'color': COLORS['bear_primary'],
        'line': {'color': '#FFFFFF', 'width': 2},  # White outline
    },

    # Exit markers
    'long_exit': {
        'symbol': 'triangle-down',
        'size': 10,
        'color': COLORS['bear_secondary'],
        'line': {'color': '#FFFFFF', 'width': 1},
    },
    'short_exit': {
        'symbol': 'triangle-up',
        'size': 10,
        'color': COLORS['bull_secondary'],
        'line': {'color': '#FFFFFF', 'width': 1},
    },

    # Winning trades (filled)
    'win': {
        'opacity': 1.0,
        'show_annotation': True,  # Show P&L annotation
    },

    # Losing trades (semi-transparent)
    'loss': {
        'opacity': 0.6,
        'show_annotation': False,  # Don't clutter with loss annotations
    },
}

# ============================================
# CHART TEMPLATES
# ============================================

# Plotly template settings
PLOTLY_TEMPLATE = {
    'layout': {
        'template': 'plotly_dark',
        'paper_bgcolor': COLORS['background_dark'],
        'plot_bgcolor': COLORS['background_dark'],
        'font': {
            'family': 'Segoe UI, Arial, sans-serif',
            'size': 12,
            'color': COLORS['text_primary'],
        },
        'xaxis': {
            'gridcolor': COLORS['grid'],
            'showgrid': True,
            'zeroline': False,
        },
        'yaxis': {
            'gridcolor': COLORS['grid'],
            'showgrid': True,
            'zeroline': False,
        },
        'hovermode': 'x unified',
        'hoverlabel': {
            'bgcolor': COLORS['background_medium'],
            'font_size': 13,
        },
    }
}

# ============================================
# LIGHTWEIGHT CHARTS CONFIG
# ============================================

# Configuration for lightweight-charts-python (TradingView-quality)
LIGHTWEIGHT_CHARTS_CONFIG = {
    'layout': {
        'background': {'color': COLORS['background_dark']},
        'text_color': COLORS['text_primary'],
    },
    'grid': {
        'vert_enabled': True,
        'horz_enabled': True,
        'color': COLORS['grid'],
    },
    'crosshair': {
        'mode': 0,  # Normal crosshair
    },
    'price_scale': {
        'border_color': COLORS['grid'],
    },
    'time_scale': {
        'border_color': COLORS['grid'],
        'time_visible': True,
        'seconds_visible': False,
    },
}

# ============================================
# PERFORMANCE THRESHOLDS
# ============================================

PERFORMANCE_THRESHOLDS = {
    # Sharpe Ratio
    'sharpe_good': 1.0,
    'sharpe_excellent': 1.5,
    'sharpe_outstanding': 2.0,

    # Sortino Ratio
    'sortino_good': 1.5,
    'sortino_excellent': 2.0,

    # Risk Limits
    'max_drawdown_limit': 0.25,      # 25% max drawdown
    'portfolio_heat_limit': 0.08,     # 8% portfolio heat (2R total risk)
    'position_size_limit': 0.05,      # 5% max position size
    'daily_loss_limit': 0.03,         # 3% daily loss limit

    # Win Rate
    'win_rate_good': 0.50,            # 50% win rate
    'win_rate_excellent': 0.60,       # 60% win rate

    # Profit Factor
    'profit_factor_good': 1.5,
    'profit_factor_excellent': 2.0,
}

# ============================================
# AVAILABLE STRATEGIES
# ============================================

AVAILABLE_STRATEGIES = {
    'orb': {
        'name': 'Opening Range Breakout',
        'file': 'strategies/orb.py',
        'class': 'ORBStrategy',
        'description': 'Trades breakouts from opening range with regime filtering',
    },
    '52w_high': {
        'name': '52-Week High Momentum',
        'file': 'strategies/high_momentum_52w.py',
        'class': 'HighMomentum52W',
        'description': 'Long-only momentum strategy on 52-week highs',
    },
    'portfolio': {
        'name': 'Multi-Strategy Portfolio',
        'file': None,
        'class': None,
        'description': 'Combined portfolio of all strategies',
    }
}

# ============================================
# CHART DIMENSIONS (Mobile-Friendly)
# ============================================

CHART_DIMENSIONS = {
    # Desktop
    'desktop': {
        'regime_timeline': 800,
        'feature_dashboard': 700,
        'equity_curve': 700,
        'trade_distribution': 500,
        'portfolio_heat': 400,
    },

    # Mobile (responsive)
    'mobile': {
        'regime_timeline': 500,
        'feature_dashboard': 450,
        'equity_curve': 450,
        'trade_distribution': 350,
        'portfolio_heat': 300,
    },
}

# Default to desktop, will be overridden by responsive callbacks
CHART_HEIGHT = CHART_DIMENSIONS['desktop']

# ============================================
# DATA REFRESH INTERVALS
# ============================================

REFRESH_INTERVALS = {
    'live_positions': 30000,      # 30 seconds
    'portfolio_metrics': 60000,   # 1 minute
    'regime_update': 300000,      # 5 minutes
    'backtest_cache': 3600000,    # 1 hour
}

# ============================================
# REGIME FEATURE THRESHOLDS
# ============================================

# Thresholds from academic_jump_model.py
REGIME_THRESHOLDS = {
    'downside_dev_crash': 0.02,       # CRASH threshold
    'sortino_20d_bear': -0.5,         # BEAR threshold
    'sortino_20d_bull': 0.5,          # BULL threshold
}

# ============================================
# CHART EXPORT SETTINGS
# ============================================

EXPORT_CONFIG = {
    'dpi': 300,                       # Publication quality
    'format': 'png',                  # Default format
    'width': 1920,                    # Full HD width
    'height': 1080,                   # Full HD height
    'scale': 2,                       # Retina display scale
}

# ============================================
# MOBILE BREAKPOINTS
# ============================================

BREAKPOINTS = {
    'mobile': 576,      # px
    'tablet': 768,      # px
    'desktop': 992,     # px
    'large': 1200,      # px
}

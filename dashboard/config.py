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
# COLOR SCHEMES (Premium OLED Luxury)
# ============================================

# Premium Trading Colors - Swiss Minimalism + OLED Luxury
COLORS = {
    # Bull/Upward Movement (Emerald)
    'bull_primary': '#00DC82',      # Vibrant emerald
    'bull_secondary': '#10B981',    # Softer emerald
    'bull_fill': 'rgba(0, 220, 130, 0.08)',
    'bull_glow': 'rgba(0, 220, 130, 0.15)',

    # Bear/Downward Movement (Crimson)
    'bear_primary': '#FF3B5C',      # Refined crimson
    'bear_secondary': '#F43F5E',    # Softer rose
    'bear_fill': 'rgba(255, 59, 92, 0.08)',
    'bear_glow': 'rgba(255, 59, 92, 0.15)',

    # Neutral/Background (OLED Optimized)
    'background_dark': '#000000',   # True black for OLED
    'background_medium': '#141414', # Elevated surface
    'background_light': '#1f1f1f',  # Active state

    # Additional surface levels
    'bg_void': '#000000',           # Primary background
    'bg_surface': '#0a0a0a',        # Barely visible surface
    'bg_elevated': '#111111',       # Elevated elements
    'bg_card': '#141414',           # Card backgrounds
    'bg_hover': '#1a1a1a',          # Hover state

    # Grid and Borders (Subtle)
    'grid': '#1f1f1f',              # Very subtle grid
    'grid_major': '#27272A',        # Major grid lines
    'border_subtle': '#27272A',     # Subtle borders
    'border_default': '#3F3F46',    # Default borders

    # Text Hierarchy
    'text_primary': '#FAFAFA',      # High contrast primary
    'text_secondary': '#A1A1AA',    # Muted secondary
    'text_tertiary': '#71717A',     # Very muted

    # Accent Colors (Vibrant)
    'accent_emerald': '#00DC82',    # Primary success
    'accent_crimson': '#FF3B5C',    # Primary danger
    'accent_amber': '#FFB800',      # Warning/caution
    'accent_electric': '#3B82F6',   # Info/action
    'accent_violet': '#8B5CF6',     # Premium/options
    'accent_cyan': '#22D3EE',       # Data/metrics

    # Status Colors
    'success': '#00DC82',           # Emerald success
    'warning': '#FFB800',           # Amber warning
    'danger': '#FF3B5C',            # Crimson danger
    'info': '#3B82F6',              # Electric blue info

    # Chart Elements
    'price_line': '#FAFAFA',        # High contrast price
    'volume_up': 'rgba(0, 220, 130, 0.4)',    # Emerald volume
    'volume_down': 'rgba(255, 59, 92, 0.4)',  # Crimson volume
    'crosshair': '#52525B',         # Subtle crosshair
}

# ============================================
# REGIME DETECTION COLORS (Luxury Edition)
# ============================================

REGIME_COLORS = {
    'TREND_BULL': 'rgba(0, 220, 130, 0.12)',     # Emerald with subtle opacity
    'TREND_NEUTRAL': 'rgba(113, 113, 122, 0.08)', # Zinc with minimal opacity
    'TREND_BEAR': 'rgba(255, 184, 0, 0.12)',     # Amber with subtle opacity
    'CRASH': 'rgba(255, 59, 92, 0.15)',          # Crimson with moderate opacity
}

REGIME_BORDER_COLORS = {
    'TREND_BULL': '#00DC82',     # Emerald
    'TREND_NEUTRAL': '#71717A',  # Zinc
    'TREND_BEAR': '#FFB800',     # Amber
    'CRASH': '#FF3B5C',          # Crimson
}

REGIME_TEXT_COLORS = {
    'TREND_BULL': '#00DC82',     # Emerald
    'TREND_NEUTRAL': '#A1A1AA',  # Zinc-400
    'TREND_BEAR': '#FFB800',     # Amber
    'CRASH': '#FF3B5C',          # Crimson
}

# ============================================
# STRAT PATTERN COLORS (Luxury Edition)
# ============================================

STRAT_COLORS = {
    '1': 'rgba(113, 113, 122, 0.15)',   # Inside bar - Zinc
    '2U': 'rgba(0, 220, 130, 0.15)',    # Directional Up - Emerald
    '2D': 'rgba(255, 59, 92, 0.15)',    # Directional Down - Crimson
    '3': 'rgba(255, 184, 0, 0.15)',     # Outside bar - Amber
}

STRAT_BORDER_COLORS = {
    '1': '#71717A',     # Zinc
    '2U': '#00DC82',    # Emerald
    '2D': '#FF3B5C',    # Crimson
    '3': '#FFB800',     # Amber
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
# CHART TEMPLATES (Premium Luxury)
# ============================================

# Premium Typography
FONTS = {
    'display': '"Clash Display", "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif',
    'body': '"Satoshi", "SF Pro Text", -apple-system, BlinkMacSystemFont, sans-serif',
    'mono': '"JetBrains Mono", "SF Mono", "Fira Code", Consolas, monospace',
}

# Plotly template settings
PLOTLY_TEMPLATE = {
    'layout': {
        'template': 'plotly_dark',
        'paper_bgcolor': COLORS['background_dark'],
        'plot_bgcolor': COLORS['background_dark'],
        'font': {
            'family': FONTS['body'],
            'size': 12,
            'color': COLORS['text_primary'],
        },
        'title': {
            'font': {
                'family': FONTS['display'],
                'size': 18,
                'color': COLORS['text_primary'],
            },
            'x': 0,
            'xanchor': 'left',
        },
        'xaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': 1,
            'showgrid': True,
            'zeroline': False,
            'linecolor': COLORS['border_subtle'],
            'tickfont': {'family': FONTS['mono'], 'size': 10},
        },
        'yaxis': {
            'gridcolor': COLORS['grid'],
            'gridwidth': 1,
            'showgrid': True,
            'zeroline': False,
            'linecolor': COLORS['border_subtle'],
            'tickfont': {'family': FONTS['mono'], 'size': 10},
        },
        'hovermode': 'x unified',
        'hoverlabel': {
            'bgcolor': COLORS['bg_card'],
            'bordercolor': COLORS['border_default'],
            'font': {
                'family': FONTS['body'],
                'size': 12,
                'color': COLORS['text_primary'],
            },
        },
        'legend': {
            'bgcolor': 'rgba(0,0,0,0)',
            'bordercolor': 'rgba(0,0,0,0)',
            'font': {
                'family': FONTS['body'],
                'size': 11,
                'color': COLORS['text_secondary'],
            },
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'right',
            'x': 1,
        },
        'margin': {'l': 60, 'r': 30, 't': 60, 'b': 40},
        'colorway': [
            COLORS['accent_emerald'],
            COLORS['accent_electric'],
            COLORS['accent_amber'],
            COLORS['accent_violet'],
            COLORS['accent_cyan'],
            COLORS['accent_crimson'],
        ],
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

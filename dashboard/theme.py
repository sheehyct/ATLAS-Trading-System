"""
ATLAS Dashboard Premium Theme

Combines Swiss Minimalism with Dark OLED Luxury aesthetic:
- True black (#000000) backgrounds for OLED optimization
- Vibrant accent colors with subtle glow effects
- Typography-first design with premium fonts
- Razor-sharp visual hierarchy
- Cinematic micro-interactions

Design Philosophy:
- Minimalism: Remove everything non-essential
- Swiss Grid: Mathematical precision in layout
- OLED Luxury: Deep blacks, vibrant accents, subtle luminosity
"""

from typing import Dict, Any

# ============================================
# PREMIUM TYPOGRAPHY
# ============================================

# IBM Plex: Professional, corporate, highly readable
# Used by IBM, Carbon Design System - Bloomberg/financial industry feel
FONTS = {
    'display': '"IBM Plex Sans", "SF Pro Display", -apple-system, BlinkMacSystemFont, sans-serif',
    'body': '"IBM Plex Sans", "SF Pro Text", -apple-system, BlinkMacSystemFont, sans-serif',
    'mono': '"IBM Plex Mono", "JetBrains Mono", "SF Mono", monospace',
    'fallback': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
}

# Font imports for external stylesheets (Google Fonts)
FONT_URLS = [
    'https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap',
    'https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap',
]

# ============================================
# OLED LUXURY COLOR PALETTE
# ============================================

# Core: True black for maximum OLED efficiency
# Accents: Refined, vibrant colors with controlled saturation

COLORS_LUXURY = {
    # === BACKGROUNDS (OLED Optimized) ===
    'bg_void': '#000000',           # True black - primary background
    'bg_surface': '#0a0a0a',        # Barely visible surface
    'bg_elevated': '#111111',       # Elevated elements
    'bg_card': '#141414',           # Card backgrounds
    'bg_hover': '#1a1a1a',          # Hover state
    'bg_active': '#1f1f1f',         # Active/pressed state

    # === ACCENT COLORS (Vibrant but refined) ===
    'accent_emerald': '#00DC82',    # Primary success/bull - emerald green
    'accent_crimson': '#FF3B5C',    # Primary danger/bear - refined crimson
    'accent_amber': '#FFB800',      # Warning/caution - rich amber
    'accent_electric': '#3B82F6',   # Info/neutral action - electric blue
    'accent_violet': '#8B5CF6',     # Premium/options - violet
    'accent_cyan': '#22D3EE',       # Data/metrics - cyan

    # === TEXT HIERARCHY ===
    'text_primary': '#FAFAFA',      # High contrast primary text
    'text_secondary': '#A1A1AA',    # Muted secondary text (zinc-400)
    'text_tertiary': '#71717A',     # Very muted (zinc-500)
    'text_disabled': '#52525B',     # Disabled state (zinc-600)

    # === BORDERS & DIVIDERS ===
    'border_subtle': '#27272A',     # Subtle divider (zinc-800)
    'border_default': '#3F3F46',    # Default border (zinc-700)
    'border_focus': '#52525B',      # Focus ring base

    # === TRADING SPECIFIC ===
    'bull_primary': '#00DC82',      # Emerald for bullish
    'bull_secondary': '#10B981',    # Softer emerald
    'bull_glow': 'rgba(0, 220, 130, 0.15)',
    'bull_fill': 'rgba(0, 220, 130, 0.08)',

    'bear_primary': '#FF3B5C',      # Crimson for bearish
    'bear_secondary': '#F43F5E',    # Softer rose
    'bear_glow': 'rgba(255, 59, 92, 0.15)',
    'bear_fill': 'rgba(255, 59, 92, 0.08)',

    'neutral_primary': '#71717A',   # Zinc for neutral
    'neutral_fill': 'rgba(113, 113, 122, 0.08)',

    # === CHART ELEMENTS ===
    'grid': '#1f1f1f',              # Very subtle grid
    'grid_major': '#27272A',        # Major grid lines
    'crosshair': '#52525B',         # Crosshair color
    'volume_up': 'rgba(0, 220, 130, 0.4)',
    'volume_down': 'rgba(255, 59, 92, 0.4)',

    # === GLOW EFFECTS (Signature detail) ===
    'glow_emerald': '0 0 20px rgba(0, 220, 130, 0.3)',
    'glow_crimson': '0 0 20px rgba(255, 59, 92, 0.3)',
    'glow_amber': '0 0 20px rgba(255, 184, 0, 0.3)',
    'glow_electric': '0 0 20px rgba(59, 130, 246, 0.3)',
    'glow_subtle': '0 0 40px rgba(255, 255, 255, 0.02)',
}

# ============================================
# REGIME COLORS (Luxury Edition)
# ============================================

REGIME_COLORS_LUXURY = {
    'TREND_BULL': 'rgba(0, 220, 130, 0.12)',
    'TREND_NEUTRAL': 'rgba(113, 113, 122, 0.08)',
    'TREND_BEAR': 'rgba(255, 184, 0, 0.12)',
    'CRASH': 'rgba(255, 59, 92, 0.15)',
}

REGIME_BORDERS_LUXURY = {
    'TREND_BULL': '#00DC82',
    'TREND_NEUTRAL': '#71717A',
    'TREND_BEAR': '#FFB800',
    'CRASH': '#FF3B5C',
}

# ============================================
# STRAT PATTERN COLORS (Luxury Edition)
# ============================================

STRAT_COLORS_LUXURY = {
    '1': 'rgba(113, 113, 122, 0.15)',   # Inside - zinc
    '2U': 'rgba(0, 220, 130, 0.15)',    # Up - emerald
    '2D': 'rgba(255, 59, 92, 0.15)',    # Down - crimson
    '3': 'rgba(255, 184, 0, 0.15)',     # Outside - amber
}

STRAT_BORDERS_LUXURY = {
    '1': '#71717A',
    '2U': '#00DC82',
    '2D': '#FF3B5C',
    '3': '#FFB800',
}

# ============================================
# PLOTLY THEME CONFIGURATION
# ============================================

def get_plotly_template() -> Dict[str, Any]:
    """Get premium Plotly template configuration."""
    return {
        'layout': {
            'template': 'plotly_dark',
            'paper_bgcolor': COLORS_LUXURY['bg_void'],
            'plot_bgcolor': COLORS_LUXURY['bg_void'],
            'font': {
                'family': FONTS['body'],
                'size': 12,
                'color': COLORS_LUXURY['text_primary'],
            },
            'title': {
                'font': {
                    'family': FONTS['display'],
                    'size': 18,
                    'color': COLORS_LUXURY['text_primary'],
                },
                'x': 0,
                'xanchor': 'left',
            },
            'xaxis': {
                'gridcolor': COLORS_LUXURY['grid'],
                'gridwidth': 1,
                'showgrid': True,
                'zeroline': False,
                'linecolor': COLORS_LUXURY['border_subtle'],
                'tickfont': {'family': FONTS['mono'], 'size': 10},
            },
            'yaxis': {
                'gridcolor': COLORS_LUXURY['grid'],
                'gridwidth': 1,
                'showgrid': True,
                'zeroline': False,
                'linecolor': COLORS_LUXURY['border_subtle'],
                'tickfont': {'family': FONTS['mono'], 'size': 10},
            },
            'hovermode': 'x unified',
            'hoverlabel': {
                'bgcolor': COLORS_LUXURY['bg_card'],
                'bordercolor': COLORS_LUXURY['border_default'],
                'font': {
                    'family': FONTS['body'],
                    'size': 12,
                    'color': COLORS_LUXURY['text_primary'],
                },
            },
            'legend': {
                'bgcolor': 'rgba(0,0,0,0)',
                'bordercolor': 'rgba(0,0,0,0)',
                'font': {
                    'family': FONTS['body'],
                    'size': 11,
                    'color': COLORS_LUXURY['text_secondary'],
                },
                'orientation': 'h',
                'yanchor': 'bottom',
                'y': 1.02,
                'xanchor': 'right',
                'x': 1,
            },
            'margin': {'l': 60, 'r': 30, 't': 60, 'b': 40},
            'colorway': [
                COLORS_LUXURY['accent_emerald'],
                COLORS_LUXURY['accent_electric'],
                COLORS_LUXURY['accent_amber'],
                COLORS_LUXURY['accent_violet'],
                COLORS_LUXURY['accent_cyan'],
                COLORS_LUXURY['accent_crimson'],
            ],
        }
    }


def get_indicator_style(value_type: str = 'neutral') -> Dict[str, Any]:
    """Get styling for Plotly indicator/gauge components."""
    color_map = {
        'positive': COLORS_LUXURY['accent_emerald'],
        'negative': COLORS_LUXURY['accent_crimson'],
        'warning': COLORS_LUXURY['accent_amber'],
        'neutral': COLORS_LUXURY['accent_electric'],
    }
    return {
        'font': {'family': FONTS['display'], 'color': color_map.get(value_type, COLORS_LUXURY['text_primary'])},
        'title_font': {'family': FONTS['body'], 'size': 13, 'color': COLORS_LUXURY['text_secondary']},
        'number_font': {'family': FONTS['mono'], 'size': 28},
    }


def get_gauge_colors(threshold_type: str = 'heat') -> Dict[str, Any]:
    """Get gauge color configuration for different threshold types."""
    if threshold_type == 'heat':
        return {
            'steps': [
                {'range': [0, 4], 'color': 'rgba(0, 220, 130, 0.15)'},
                {'range': [4, 6], 'color': 'rgba(255, 184, 0, 0.15)'},
                {'range': [6, 10], 'color': 'rgba(255, 59, 92, 0.15)'},
            ],
            'bar_color': COLORS_LUXURY['accent_emerald'],
            'threshold_color': COLORS_LUXURY['accent_crimson'],
        }
    elif threshold_type == 'performance':
        return {
            'steps': [
                {'range': [-50, 0], 'color': 'rgba(255, 59, 92, 0.15)'},
                {'range': [0, 50], 'color': 'rgba(0, 220, 130, 0.15)'},
            ],
            'bar_color': COLORS_LUXURY['accent_electric'],
            'threshold_color': COLORS_LUXURY['accent_amber'],
        }
    return {}


# ============================================
# CSS VARIABLES EXPORT
# ============================================

def get_css_variables() -> str:
    """Generate CSS custom properties for the theme."""
    css_vars = """
    :root {
        /* Backgrounds */
        --bg-void: #000000;
        --bg-surface: #0a0a0a;
        --bg-elevated: #111111;
        --bg-card: #141414;
        --bg-hover: #1a1a1a;
        --bg-active: #1f1f1f;

        /* Accents */
        --accent-emerald: #00DC82;
        --accent-crimson: #FF3B5C;
        --accent-amber: #FFB800;
        --accent-electric: #3B82F6;
        --accent-violet: #8B5CF6;
        --accent-cyan: #22D3EE;

        /* Text */
        --text-primary: #FAFAFA;
        --text-secondary: #A1A1AA;
        --text-tertiary: #71717A;
        --text-disabled: #52525B;

        /* Borders */
        --border-subtle: #27272A;
        --border-default: #3F3F46;
        --border-focus: #52525B;

        /* Trading */
        --bull: #00DC82;
        --bear: #FF3B5C;
        --neutral: #71717A;

        /* Typography - IBM Plex (Professional) */
        --font-display: 'IBM Plex Sans', 'SF Pro Display', -apple-system, sans-serif;
        --font-body: 'IBM Plex Sans', 'SF Pro Text', -apple-system, sans-serif;
        --font-mono: 'IBM Plex Mono', 'JetBrains Mono', monospace;

        /* Transitions */
        --transition-fast: 150ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-base: 200ms cubic-bezier(0.4, 0, 0.2, 1);
        --transition-slow: 300ms cubic-bezier(0.4, 0, 0.2, 1);

        /* Shadows/Glows */
        --glow-emerald: 0 0 20px rgba(0, 220, 130, 0.3);
        --glow-crimson: 0 0 20px rgba(255, 59, 92, 0.3);
        --glow-subtle: 0 0 40px rgba(255, 255, 255, 0.02);
        --shadow-card: 0 4px 24px rgba(0, 0, 0, 0.5);
    }
    """
    return css_vars


# ============================================
# DASH BOOTSTRAP THEME OVERRIDES
# ============================================

# Custom class names for dbc components with luxury styling
DBC_CLASSES = {
    'card': 'atlas-card',
    'card_header': 'atlas-card-header',
    'nav_tabs': 'atlas-nav-tabs',
    'nav_link': 'atlas-nav-link',
    'badge_bull': 'atlas-badge-bull',
    'badge_bear': 'atlas-badge-bear',
    'badge_neutral': 'atlas-badge-neutral',
    'metric_card': 'atlas-metric-card',
    'metric_value': 'atlas-metric-value',
    'metric_label': 'atlas-metric-label',
}

# Export all for easy import
__all__ = [
    'FONTS',
    'FONT_URLS',
    'COLORS_LUXURY',
    'REGIME_COLORS_LUXURY',
    'REGIME_BORDERS_LUXURY',
    'STRAT_COLORS_LUXURY',
    'STRAT_BORDERS_LUXURY',
    'get_plotly_template',
    'get_indicator_style',
    'get_gauge_colors',
    'get_css_variables',
    'DBC_CLASSES',
]

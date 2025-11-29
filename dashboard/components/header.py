"""
Dashboard Header Component - Premium Luxury Edition

Minimalist navigation bar with:
- Swiss-inspired typography hierarchy
- Subtle glow effects on status indicators
- True black OLED-optimized background
- Refined spacing and proportions
"""

import dash_bootstrap_components as dbc
from dash import html
from dashboard.config import COLORS


def create_header():
    """
    Create premium dashboard header with minimalist luxury aesthetic.

    Features:
    - Display font for brand identity
    - Status badges with subtle glow effects
    - Clean, uncluttered design
    - Responsive layout

    Returns:
        Bootstrap navbar component
    """

    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                # Logo and Brand - Swiss Typography
                dbc.Col([
                    html.Div([
                        # Minimal icon with glow
                        html.Span(
                            html.I(
                                className='fas fa-chart-line',
                                style={
                                    'fontSize': '1.25rem',
                                    'filter': f'drop-shadow(0 0 8px {COLORS["accent_emerald"]})',
                                }
                            ),
                            style={
                                'color': COLORS['accent_emerald'],
                                'marginRight': '0.75rem',
                            }
                        ),
                        # Brand name - Display font
                        html.Span(
                            'ATLAS',
                            style={
                                'fontFamily': '"Clash Display", sans-serif',
                                'fontWeight': '600',
                                'fontSize': '1.375rem',
                                'letterSpacing': '-0.02em',
                                'color': COLORS['text_primary'],
                            }
                        ),
                        # Subtitle - Body font, muted
                        html.Span(
                            'Trading',
                            style={
                                'fontFamily': '"Satoshi", sans-serif',
                                'fontWeight': '400',
                                'fontSize': '1.375rem',
                                'letterSpacing': '-0.01em',
                                'color': COLORS['text_tertiary'],
                                'marginLeft': '0.5rem',
                            }
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], width='auto'),

                # Status Indicators - Refined Badges
                dbc.Col([
                    html.Div([
                        # Live Status - Emerald Glow
                        html.Span(
                            'LIVE',
                            id='live-status-badge',
                            className='badge',
                            style={
                                'background': f'linear-gradient(135deg, {COLORS["accent_emerald"]}, #00B36B)',
                                'color': COLORS['background_dark'],
                                'fontFamily': '"JetBrains Mono", monospace',
                                'fontSize': '0.65rem',
                                'fontWeight': '500',
                                'letterSpacing': '0.05em',
                                'padding': '0.25rem 0.5rem',
                                'marginRight': '0.5rem',
                                'boxShadow': f'0 0 12px {COLORS["bull_glow"]}',
                                'borderRadius': '4px',
                            }
                        ),

                        # Connection Status - Electric Blue
                        html.Span(
                            'CONNECTED',
                            id='connection-badge',
                            className='badge',
                            style={
                                'background': f'linear-gradient(135deg, {COLORS["accent_electric"]}, #2563EB)',
                                'color': COLORS['text_primary'],
                                'fontFamily': '"JetBrains Mono", monospace',
                                'fontSize': '0.65rem',
                                'fontWeight': '500',
                                'letterSpacing': '0.05em',
                                'padding': '0.25rem 0.5rem',
                                'marginRight': '0.5rem',
                                'boxShadow': '0 0 12px rgba(59, 130, 246, 0.25)',
                                'borderRadius': '4px',
                            }
                        ),

                        # Current Regime - Dynamic color
                        html.Span(
                            'NEUTRAL',
                            id='current-regime-badge',
                            className='badge',
                            style={
                                'background': COLORS['bg_elevated'],
                                'color': COLORS['text_secondary'],
                                'fontFamily': '"JetBrains Mono", monospace',
                                'fontSize': '0.65rem',
                                'fontWeight': '500',
                                'letterSpacing': '0.05em',
                                'padding': '0.25rem 0.5rem',
                                'marginRight': '0.5rem',
                                'border': f'1px solid {COLORS["border_default"]}',
                                'borderRadius': '4px',
                            }
                        ),

                        # Market Time - Subtle
                        html.Span(
                            id='market-time-badge',
                            className='badge',
                            style={
                                'background': COLORS['bg_surface'],
                                'color': COLORS['text_tertiary'],
                                'fontFamily': '"JetBrains Mono", monospace',
                                'fontSize': '0.65rem',
                                'fontWeight': '400',
                                'padding': '0.25rem 0.5rem',
                                'border': f'1px solid {COLORS["border_subtle"]}',
                                'borderRadius': '4px',
                            }
                        ),
                    ], style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'gap': '0.25rem',
                    })
                ], width='auto', className='ms-auto')
            ], align='center', className='g-0 w-100', justify='between')
        ], fluid=True),
        color='dark',
        dark=True,
        className='mb-4',
        style={
            'background': f'linear-gradient(180deg, {COLORS["bg_card"]} 0%, {COLORS["bg_surface"]} 100%)',
            'borderBottom': f'1px solid {COLORS["border_subtle"]}',
            'padding': '0.75rem 1.25rem',
            'boxShadow': '0 4px 24px rgba(0, 0, 0, 0.5)',
        }
    )


def create_regime_badge(regime: str) -> dict:
    """
    Get badge styling for a specific regime.

    Args:
        regime: Regime name (TREND_BULL, TREND_BEAR, TREND_NEUTRAL, CRASH)

    Returns:
        Style dict for the badge
    """
    regime_styles = {
        'TREND_BULL': {
            'background': f'linear-gradient(135deg, {COLORS["accent_emerald"]}, #00B36B)',
            'color': COLORS['background_dark'],
            'boxShadow': f'0 0 12px {COLORS["bull_glow"]}',
        },
        'TREND_BEAR': {
            'background': f'linear-gradient(135deg, {COLORS["accent_amber"]}, #E6A600)',
            'color': COLORS['background_dark'],
            'boxShadow': '0 0 12px rgba(255, 184, 0, 0.25)',
        },
        'TREND_NEUTRAL': {
            'background': COLORS['bg_elevated'],
            'color': COLORS['text_secondary'],
            'border': f'1px solid {COLORS["border_default"]}',
        },
        'CRASH': {
            'background': f'linear-gradient(135deg, {COLORS["accent_crimson"]}, #E6365A)',
            'color': COLORS['text_primary'],
            'boxShadow': f'0 0 12px {COLORS["bear_glow"]}',
            'animation': 'pulse-glow 2s ease-in-out infinite',
        },
    }

    base_style = {
        'fontFamily': '"JetBrains Mono", monospace',
        'fontSize': '0.65rem',
        'fontWeight': '500',
        'letterSpacing': '0.05em',
        'padding': '0.25rem 0.5rem',
        'borderRadius': '4px',
    }

    regime_specific = regime_styles.get(regime, regime_styles['TREND_NEUTRAL'])
    return {**base_style, **regime_specific}

"""
Multi-Sector Stock Universe for Momentum Strategies

This module defines the shared stock universe used by Quality-Momentum and
Semi-Vol Momentum strategies. Centralizes universe management to avoid duplication.

Design Criteria:
- Minimum 2 stocks per GICS sector for diversification
- High liquidity (avg volume > 1M shares)
- Data availability from 2010+ (for 15-year backtest)
- Quality bias: Established, profitable companies

Created: Session MOMENTUM-5, January 2026
"""

from collections import Counter
from typing import Dict, List


# =============================================================================
# SECTOR SYMBOL DEFINITIONS (36 Stocks Across 11 GICS Sectors)
# =============================================================================

# Information Technology (6 stocks) - Core sector for momentum
TECH_SYMBOLS = [
    'AAPL',   # Consumer Electronics - $3T market cap
    'MSFT',   # Enterprise Software - $3T market cap
    'NVDA',   # Semiconductors/AI - High momentum
    'AVGO',   # Semiconductors - Dividend growth
    'CRM',    # Cloud Software - SaaS leader
    'CSCO',   # Networking - Value/Quality
]

# Communication Services (3 stocks)
COMM_SYMBOLS = [
    'GOOGL',  # Search/Advertising - Quality growth
    'META',   # Social Media - High momentum
    'DIS',    # Entertainment - Consumer brand
]

# Consumer Discretionary (4 stocks)
CONSUMER_DISC_SYMBOLS = [
    'AMZN',   # E-commerce/Cloud - Growth leader
    'TSLA',   # EVs - High momentum/volatility
    'HD',     # Home Improvement - Quality retailer
    'NKE',    # Apparel - Global brand
]

# Consumer Staples (3 stocks)
CONSUMER_STAPLES_SYMBOLS = [
    'PG',     # Household Products - Dividend aristocrat
    'KO',     # Beverages - Defensive quality
    'COST',   # Retail - Growth staple
]

# Energy (3 stocks)
ENERGY_SYMBOLS = [
    'XOM',    # Integrated Oil - Major
    'CVX',    # Integrated Oil - Quality
    'COP',    # Exploration/Production - Momentum
]

# Financials (4 stocks)
FINANCIAL_SYMBOLS = [
    'JPM',    # Diversified Banks - Quality leader
    'BRK-B',  # Insurance/Conglomerate - Value
    'V',      # Payment Networks - Growth quality
    'MA',     # Payment Networks - Growth quality
]

# Health Care (4 stocks)
HEALTHCARE_SYMBOLS = [
    'UNH',    # Health Insurance - Quality growth
    'JNJ',    # Pharmaceuticals - Dividend aristocrat
    'LLY',    # Pharmaceuticals - High momentum
    'ABBV',   # Pharmaceuticals - Dividend growth
]

# Industrials (3 stocks)
INDUSTRIAL_SYMBOLS = [
    'CAT',    # Machinery - Cyclical quality
    'UNP',    # Railroads - Moat
    'HON',    # Diversified - Quality conglomerate
]

# Materials (2 stocks)
MATERIALS_SYMBOLS = [
    'LIN',    # Industrial Gases - Quality leader
    'APD',    # Industrial Gases - Dividend growth
]

# Real Estate (2 stocks)
REAL_ESTATE_SYMBOLS = [
    'PLD',    # Industrial REITs - E-commerce exposure
    'AMT',    # Cell Tower REITs - Digital infrastructure
]

# Utilities (2 stocks)
UTILITIES_SYMBOLS = [
    'NEE',    # Electric Utilities - Renewable leader
    'SO',     # Electric Utilities - Defensive yield
]


# =============================================================================
# COMBINED UNIVERSE
# =============================================================================

# Full universe (36 stocks)
UNIVERSE_SYMBOLS = (
    TECH_SYMBOLS +
    COMM_SYMBOLS +
    CONSUMER_DISC_SYMBOLS +
    CONSUMER_STAPLES_SYMBOLS +
    ENERGY_SYMBOLS +
    FINANCIAL_SYMBOLS +
    HEALTHCARE_SYMBOLS +
    INDUSTRIAL_SYMBOLS +
    MATERIALS_SYMBOLS +
    REAL_ESTATE_SYMBOLS +
    UTILITIES_SYMBOLS
)

# Sector mapping for analysis
SECTOR_MAP: Dict[str, str] = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
    'AVGO': 'Technology', 'CRM': 'Technology', 'CSCO': 'Technology',
    'GOOGL': 'Communication', 'META': 'Communication', 'DIS': 'Communication',
    'AMZN': 'Consumer Disc', 'TSLA': 'Consumer Disc', 'HD': 'Consumer Disc', 'NKE': 'Consumer Disc',
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'COST': 'Consumer Staples',
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
    'JPM': 'Financials', 'BRK-B': 'Financials', 'V': 'Financials', 'MA': 'Financials',
    'UNH': 'Healthcare', 'JNJ': 'Healthcare', 'LLY': 'Healthcare', 'ABBV': 'Healthcare',
    'CAT': 'Industrials', 'UNP': 'Industrials', 'HON': 'Industrials',
    'LIN': 'Materials', 'APD': 'Materials',
    'PLD': 'Real Estate', 'AMT': 'Real Estate',
    'NEE': 'Utilities', 'SO': 'Utilities',
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_validation_symbols() -> List[str]:
    """
    Get 16-symbol validation subset with representation from all 11 sectors.

    Returns:
        List of 16 symbols for faster validation runs while maintaining
        sector diversity across all 11 GICS sectors.
    """
    return list(
        TECH_SYMBOLS[:3] +        # 3 tech
        COMM_SYMBOLS[:1] +        # 1 comm
        CONSUMER_DISC_SYMBOLS[:2] +  # 2 consumer disc
        CONSUMER_STAPLES_SYMBOLS[:1] +  # 1 staples
        ENERGY_SYMBOLS[:1] +      # 1 energy
        FINANCIAL_SYMBOLS[:2] +   # 2 financials
        HEALTHCARE_SYMBOLS[:2] +  # 2 healthcare
        INDUSTRIAL_SYMBOLS[:1] +  # 1 industrials
        MATERIALS_SYMBOLS[:1] +   # 1 materials
        REAL_ESTATE_SYMBOLS[:1] + # 1 real estate
        UTILITIES_SYMBOLS[:1]     # 1 utilities
    )  # Total: 16 stocks across 11 sectors


def get_sector(symbol: str) -> str:
    """
    Get the GICS sector for a given symbol.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Sector name or 'Unknown' if not in universe
    """
    return SECTOR_MAP.get(symbol, 'Unknown')


def get_symbols_by_sector(sector: str) -> List[str]:
    """
    Get all symbols belonging to a specific sector.

    Args:
        sector: Sector name (e.g., 'Technology', 'Healthcare')

    Returns:
        List of symbols in that sector
    """
    return [sym for sym, sec in SECTOR_MAP.items() if sec == sector]


def get_sector_counts() -> Dict[str, int]:
    """
    Get count of symbols per sector.

    Returns:
        Dictionary mapping sector name to symbol count
    """
    return dict(Counter(SECTOR_MAP.values()))


def print_sector_distribution(symbols: List[str]) -> None:
    """
    Print sector distribution of a symbol list.

    Args:
        symbols: List of stock symbols to analyze
    """
    sectors = [SECTOR_MAP.get(s, 'Unknown') for s in symbols]
    distribution = Counter(sectors)
    print("\nSector Distribution:")
    for sector, count in sorted(distribution.items(), key=lambda x: -x[1]):
        pct = count / len(symbols) * 100
        print(f"  {sector:<18} {count:>2} stocks ({pct:>5.1f}%)")


def get_all_sectors() -> List[str]:
    """
    Get list of all sectors in the universe.

    Returns:
        List of unique sector names
    """
    return sorted(set(SECTOR_MAP.values()))

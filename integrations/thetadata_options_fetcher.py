"""
ThetaData Options Fetcher for ATLAS Trading System.

Session 79: High-level interface for options data in STRAT backtesting.

Features:
- Abstracts ThetaData specifics from options_module
- Caches quotes to minimize API calls (pickle-based, following tiingo pattern)
- Graceful fallback to Black-Scholes when data unavailable
- Designed for dependency injection (testable with mock provider)

Usage:
    from integrations.thetadata_options_fetcher import ThetaDataOptionsFetcher

    fetcher = ThetaDataOptionsFetcher()
    if fetcher.is_available:
        price = fetcher.get_option_price('SPY', datetime(2024,12,20), 450.0, 'C', datetime(2024,11,15))
        greeks = fetcher.get_option_greeks('SPY', datetime(2024,12,20), 450.0, 'C', datetime(2024,11,15))
"""

import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Any

from integrations.thetadata_client import (
    ThetaDataProviderBase,
    ThetaDataRESTClient,
    OptionsQuote,
    create_thetadata_client
)

# Optional import for Black-Scholes fallback
try:
    from strat.greeks import calculate_greeks, Greeks
    GREEKS_AVAILABLE = True
except ImportError:
    GREEKS_AVAILABLE = False

# Session 82: Import risk-free rate for dynamic fallback
try:
    from strat.risk_free_rate import get_risk_free_rate
    RISK_FREE_RATE_AVAILABLE = True
except ImportError:
    RISK_FREE_RATE_AVAILABLE = False


class ThetaDataOptionsFetcher:
    """
    High-level interface for options data in STRAT backtesting.

    Provides a clean abstraction layer between ThetaData and the options module.
    Handles caching, connection management, and fallback to Black-Scholes.

    Attributes:
        cache_dir: Directory for pickle cache files
        use_cache: Whether to use caching
        fallback_to_bs: Whether to fall back to Black-Scholes when ThetaData unavailable
        cache_ttl_days: How long cached data is valid (default: 7 days)
    """

    DEFAULT_CACHE_DIR = "./data/thetadata_cache"
    DEFAULT_CACHE_TTL_DAYS = 7

    def __init__(
        self,
        provider: Optional[ThetaDataProviderBase] = None,
        cache_dir: str = None,
        use_cache: bool = True,
        fallback_to_bs: bool = True,
        cache_ttl_days: int = None,
        auto_connect: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize ThetaData options fetcher.

        Args:
            provider: ThetaData provider instance (creates REST client if None)
            cache_dir: Directory for cache files (default: ./data/thetadata_cache)
            use_cache: Enable caching (default: True)
            fallback_to_bs: Fall back to Black-Scholes if ThetaData fails (default: True)
            cache_ttl_days: Cache validity period (default: 7 days)
            auto_connect: Automatically connect on init (default: True)
            logger: Optional logger instance
        """
        self._provider = provider
        self._use_cache = use_cache
        self._fallback_to_bs = fallback_to_bs
        self._cache_ttl_days = cache_ttl_days or self.DEFAULT_CACHE_TTL_DAYS
        self._connected = False

        # Set up cache directory
        self.cache_dir = Path(cache_dir or self.DEFAULT_CACHE_DIR)
        if use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = logger or self._create_default_logger()

        # Create provider if not provided
        # Session 83 BUG FIX: Narrow exception handling - don't catch KeyboardInterrupt, SystemExit
        if self._provider is None:
            try:
                self._provider = create_thetadata_client(mode='rest')
            except (ImportError, ModuleNotFoundError) as e:
                # Missing dependencies or config module
                self.logger.warning(f"ThetaData client import error: {e}")
                self._provider = None
            except ValueError as e:
                # Invalid configuration
                self.logger.warning(f"ThetaData client config error: {e}")
                self._provider = None
            except OSError as e:
                # File system issues (e.g., config file access)
                self.logger.warning(f"ThetaData client file error: {e}")
                self._provider = None
            # Note: Don't catch generic Exception - let programming errors propagate

        # Auto-connect if requested
        if auto_connect and self._provider is not None:
            self._connect()

    def _create_default_logger(self) -> logging.Logger:
        """Create default logger."""
        logger = logging.getLogger('thetadata_fetcher')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _connect(self) -> bool:
        """Attempt to connect to ThetaData terminal."""
        if self._provider is None:
            return False

        try:
            self._connected = self._provider.connect()
            return self._connected
        except Exception as e:
            self.logger.warning(f"Connection failed: {e}")
            self._connected = False
            return False

    @property
    def is_available(self) -> bool:
        """
        Check if ThetaData is available for use.

        Returns:
            True if connected and ready for requests
        """
        return self._connected and self._provider is not None

    # =========================================================================
    # Caching Methods
    # =========================================================================

    def _get_cache_key(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        date: datetime,
        data_type: str = 'quote'
    ) -> str:
        """
        Generate cache key for option data.

        Format: {underlying}_{expiration}_{strike}_{type}_{date}_{data_type}.pkl

        Args:
            underlying: Underlying symbol
            expiration: Expiration date
            strike: Strike price
            option_type: 'C' or 'P'
            date: Historical date
            data_type: Type of data ('quote' or 'greeks')

        Returns:
            Cache key string
        """
        exp_str = expiration.strftime('%Y%m%d')
        date_str = date.strftime('%Y%m%d')
        strike_str = f"{strike:.2f}".replace('.', '_')
        return f"{underlying}_{exp_str}_{strike_str}_{option_type}_{date_str}_{data_type}.pkl"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get full path for cache file."""
        return self.cache_dir / cache_key

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """
        Check if cached file exists and is not expired.

        Args:
            cache_path: Path to cache file

        Returns:
            True if cache is valid and not expired
        """
        if not cache_path.exists():
            return False

        # Check file age
        # Session 80 BUG FIX: Use timedelta comparison instead of .days truncation
        # Old code: age_days = (datetime.now() - file_mtime).days  # truncates to whole days
        file_mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - file_mtime
        ttl = timedelta(days=self._cache_ttl_days)

        return age < ttl

    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Load data from cache if valid.

        Args:
            cache_key: Cache key

        Returns:
            Cached data or None if not available/expired
        """
        if not self._use_cache:
            return None

        cache_path = self._get_cache_path(cache_key)

        if not self._is_cache_valid(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """
        Save data to cache.

        Args:
            cache_key: Cache key
            data: Data to cache
        """
        if not self._use_cache:
            return

        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_key}: {e}")

    def clear_cache(self, underlying: str = None) -> int:
        """
        Clear cached data.

        Args:
            underlying: If provided, only clear cache for this underlying.
                       If None, clear all cache.

        Returns:
            Number of cache files cleared
        """
        if underlying:
            pattern = f"{underlying}_*.pkl"
        else:
            pattern = "*.pkl"

        files = list(self.cache_dir.glob(pattern))
        for f in files:
            f.unlink()

        self.logger.info(f"Cleared {len(files)} cache files")
        return len(files)

    # =========================================================================
    # Public Interface Methods
    # =========================================================================

    def get_option_price(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        as_of: datetime,
        underlying_price: Optional[float] = None
    ) -> Optional[float]:
        """
        Get option mid-price at specific timestamp.

        Uses ThetaData if available, falls back to Black-Scholes.

        Args:
            underlying: Underlying symbol (e.g., 'SPY')
            expiration: Option expiration date
            strike: Strike price in dollars
            option_type: 'C' for call, 'P' for put
            as_of: Historical date for quote
            underlying_price: Current underlying price (for Black-Scholes fallback)

        Returns:
            Option mid-price or None if unavailable
        """
        # Check cache first
        cache_key = self._get_cache_key(
            underlying, expiration, strike, option_type, as_of, 'quote'
        )
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached.get('mid')

        # Try ThetaData
        if self.is_available:
            quote = self._provider.get_quote(
                underlying, expiration, strike, option_type, as_of
            )
            if quote is not None:
                # Cache the quote data
                self._save_to_cache(cache_key, {
                    'bid': quote.bid,
                    'ask': quote.ask,
                    'mid': quote.mid,
                    'timestamp': as_of,
                })
                return quote.mid

        # Fallback to Black-Scholes
        if self._fallback_to_bs and GREEKS_AVAILABLE and underlying_price:
            return self._calculate_bs_price(
                underlying_price, strike, expiration, as_of, option_type
            )

        return None

    def get_option_greeks(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        as_of: datetime,
        underlying_price: Optional[float] = None
    ) -> Optional[Dict[str, float]]:
        """
        Get option Greeks from ThetaData.

        Falls back to calculated Greeks using Black-Scholes if unavailable.

        Args:
            underlying: Underlying symbol
            expiration: Option expiration date
            strike: Strike price in dollars
            option_type: 'C' or 'P'
            as_of: Historical date
            underlying_price: Current underlying price (for fallback)

        Returns:
            Dict with delta, gamma, theta, vega, iv or None if unavailable
        """
        # Check cache first
        cache_key = self._get_cache_key(
            underlying, expiration, strike, option_type, as_of, 'greeks'
        )
        cached = self._load_from_cache(cache_key)
        if cached is not None:
            return cached

        # Try ThetaData
        if self.is_available:
            greeks = self._provider.get_greeks(
                underlying, expiration, strike, option_type, as_of
            )
            if greeks is not None:
                # Cache the Greeks
                self._save_to_cache(cache_key, greeks)
                return greeks

        # Fallback to Black-Scholes
        if self._fallback_to_bs and GREEKS_AVAILABLE and underlying_price:
            return self._calculate_bs_greeks(
                underlying_price, strike, expiration, as_of, option_type
            )

        return None

    def get_quote_with_greeks(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        as_of: datetime,
        underlying_price: Optional[float] = None
    ) -> Optional[OptionsQuote]:
        """
        Get complete quote with Greeks data.

        Combines quote and Greeks into single OptionsQuote object.

        Args:
            underlying: Underlying symbol
            expiration: Option expiration date
            strike: Strike price
            option_type: 'C' or 'P'
            as_of: Historical date
            underlying_price: Underlying price for fallback

        Returns:
            OptionsQuote with all available data, or None
        """
        if self.is_available:
            quote = self._provider.get_quote_with_greeks(
                underlying, expiration, strike, option_type, as_of
            )
            if quote is not None:
                return quote

        # Fallback to Black-Scholes
        if self._fallback_to_bs and GREEKS_AVAILABLE and underlying_price:
            return self._create_bs_quote(
                underlying, expiration, strike, option_type, as_of, underlying_price
            )

        return None

    # =========================================================================
    # Black-Scholes Fallback Methods
    # =========================================================================

    def _calculate_bs_price(
        self,
        underlying_price: float,
        strike: float,
        expiration: datetime,
        as_of: datetime,
        option_type: str
    ) -> Optional[float]:
        """Calculate option price using Black-Scholes."""
        if not GREEKS_AVAILABLE:
            return None

        # Calculate time to expiration
        dte = (expiration - as_of).days
        if dte <= 0:
            return max(0, underlying_price - strike) if option_type == 'C' else max(0, strike - underlying_price)

        T = dte / 365.0

        # Session 82: Use dynamic risk-free rate when available
        if RISK_FREE_RATE_AVAILABLE:
            try:
                r = get_risk_free_rate(as_of)
            except Exception:
                r = 0.045  # Reasonable fallback
        else:
            r = 0.045  # Reasonable fallback

        # Session 82: More realistic default IV (was 0.20, caused 40-75% pricing errors)
        sigma = 0.15

        greeks = calculate_greeks(
            S=underlying_price,
            K=strike,
            T=T,
            r=r,
            sigma=sigma,
            option_type='call' if option_type == 'C' else 'put'
        )

        return greeks.option_price

    def _calculate_bs_greeks(
        self,
        underlying_price: float,
        strike: float,
        expiration: datetime,
        as_of: datetime,
        option_type: str
    ) -> Optional[Dict[str, float]]:
        """Calculate Greeks using Black-Scholes."""
        if not GREEKS_AVAILABLE:
            return None

        dte = (expiration - as_of).days
        if dte <= 0:
            return None

        T = dte / 365.0

        # Session 82: Use dynamic risk-free rate when available
        if RISK_FREE_RATE_AVAILABLE:
            try:
                r = get_risk_free_rate(as_of)
            except Exception:
                r = 0.045  # Reasonable fallback
        else:
            r = 0.045  # Reasonable fallback

        # Session 82: More realistic default IV (was 0.20, caused 40-75% pricing errors)
        sigma = 0.15

        greeks = calculate_greeks(
            S=underlying_price,
            K=strike,
            T=T,
            r=r,
            sigma=sigma,
            option_type='call' if option_type == 'C' else 'put'
        )

        return {
            'delta': greeks.delta,
            'gamma': greeks.gamma,
            'theta': greeks.theta,
            'vega': greeks.vega,
            'iv': sigma,
            'underlying_price': underlying_price,
        }

    def _estimate_spread_pct(
        self,
        option_price: float,
        underlying_price: float,
        strike: float,
        dte: int
    ) -> float:
        """
        Calculate realistic bid-ask spread per ATLAS checklist Section 9.1.1.

        Session 83 BUG FIX: Replace hardcoded 2% spread with realistic model
        based on moneyness, DTE, and option price.

        Args:
            option_price: Mid-price of option
            underlying_price: Current stock price
            strike: Strike price
            dte: Days to expiration

        Returns:
            Expected spread as decimal (0.02 = 2%)
        """
        # Base spread - even liquid options have minimum spread
        base_spread = 0.02  # 2% minimum

        # DTE adjustment: shorter dated = wider spreads
        if dte < 7:
            dte_adj = 0.03  # Near expiration = wider
        elif dte < 21:
            dte_adj = 0.01  # Sweet spot
        else:
            dte_adj = 0.02  # Longer dated = less liquid

        # Moneyness adjustment: OTM options have wider spreads
        moneyness = strike / underlying_price if underlying_price > 0 else 1.0
        otm_distance = abs(moneyness - 1.0)
        moneyness_adj = otm_distance * 0.10  # 10% of OTM distance

        # Price adjustment: cheap options have wider relative spreads
        if option_price < 1.0:
            price_adj = 0.05  # $0.05 spread on $1 option = 5%
        elif option_price < 5.0:
            price_adj = 0.02
        else:
            price_adj = 0.01

        total_spread = base_spread + dte_adj + moneyness_adj + price_adj

        # Cap at 20% (options with >20% spread should not be traded)
        return min(total_spread, 0.20)

    def _create_bs_quote(
        self,
        underlying: str,
        expiration: datetime,
        strike: float,
        option_type: str,
        as_of: datetime,
        underlying_price: float
    ) -> Optional[OptionsQuote]:
        """Create OptionsQuote using Black-Scholes."""
        if not GREEKS_AVAILABLE:
            return None

        greeks_dict = self._calculate_bs_greeks(
            underlying_price, strike, expiration, as_of, option_type
        )
        if greeks_dict is None:
            return None

        price = self._calculate_bs_price(
            underlying_price, strike, expiration, as_of, option_type
        )
        if price is None:
            return None

        # Create OSI symbol
        symbol_part = underlying.upper()
        date_part = expiration.strftime('%y%m%d')
        strike_part = str(int(strike * 1000)).zfill(8)
        osi_symbol = f"{symbol_part}{date_part}{option_type}{strike_part}"

        # Session 83 BUG FIX: Calculate realistic spread per ATLAS checklist Section 9.1.1
        dte = max(0, (expiration - as_of).days)
        spread_pct = self._estimate_spread_pct(price, underlying_price, strike, dte)
        half_spread = spread_pct / 2

        return OptionsQuote(
            symbol=osi_symbol,
            underlying=underlying.upper(),
            strike=strike,
            expiration=expiration,
            option_type=option_type,
            timestamp=as_of,
            bid=price * (1 - half_spread),  # ATLAS-compliant realistic spread
            ask=price * (1 + half_spread),
            mid=price,
            iv=greeks_dict['iv'],
            delta=greeks_dict['delta'],
            gamma=greeks_dict['gamma'],
            theta=greeks_dict['theta'],
            vega=greeks_dict['vega'],
            underlying_price=underlying_price,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("ThetaData Options Fetcher Test")
    print("=" * 60)

    # Create fetcher
    fetcher = ThetaDataOptionsFetcher(auto_connect=True)

    print(f"\n[STATUS] ThetaData available: {fetcher.is_available}")

    # Test parameters
    underlying = 'SPY'
    expiration = datetime(2024, 12, 20)
    strike = 450.0
    option_type = 'C'
    as_of = datetime(2024, 11, 15)

    # Test get_option_price
    print("\n[TEST 1] Getting option price...")
    price = fetcher.get_option_price(
        underlying, expiration, strike, option_type, as_of,
        underlying_price=450.0
    )
    if price:
        print(f"  PASS - Price: ${price:.2f}")
    else:
        print("  FAIL - No price returned")

    # Test get_option_greeks
    print("\n[TEST 2] Getting option Greeks...")
    greeks = fetcher.get_option_greeks(
        underlying, expiration, strike, option_type, as_of,
        underlying_price=450.0
    )
    if greeks:
        print(f"  PASS - Greeks received")
        print(f"  Delta: {greeks['delta']:.4f}")
        print(f"  Theta: {greeks['theta']:.4f}")
        print(f"  IV: {greeks['iv']:.2%}")
    else:
        print("  FAIL - No Greeks returned")

    # Test caching
    print("\n[TEST 3] Testing cache...")
    price2 = fetcher.get_option_price(
        underlying, expiration, strike, option_type, as_of,
        underlying_price=450.0
    )
    print(f"  Second call price: ${price2:.2f if price2 else 0:.2f}")
    print("  (Should be from cache if ThetaData was available)")

    # Test cache clearing
    print("\n[TEST 4] Clearing cache...")
    cleared = fetcher.clear_cache(underlying)
    print(f"  Cleared {cleared} cache files")

    print("\n" + "=" * 60)
    print("ThetaData Options Fetcher Test Complete")
    print("=" * 60)

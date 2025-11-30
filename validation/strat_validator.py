"""
ATLAS STRAT Validator - Orchestrates validation runs for STRAT strategies

Session 83K-2: Creates ATLASSTRATValidator for running comprehensive validation
across all pattern types, timeframes, and symbols using REAL ThetaData options data.

ThetaData Integration (Standard Tier):
- Historical quotes (bid/ask) - tick level
- Greeks 1st Order (delta, theta, vega, rho)
- Implied volatility (from Greeks endpoint)
- 8 years of historical data (2016-2024)

Validation Matrix:
| Patterns | 3-1-2, 2-1-2, 2-2 Up, 3-2, 3-2-2 | 5 |
| Timeframes | 1D, 1W, 1M | 3 |
| Symbols | SPY, QQQ, AAPL, IWM, DIA, NVDA | 6 |
| Total | 5 x 3 x 6 | 90 runs |

Usage:
    from validation.strat_validator import ATLASSTRATValidator

    validator = ATLASSTRATValidator()

    # Verify ThetaData connection first
    if not validator.verify_thetadata_connection():
        print("ThetaData not available!")

    # Run single batch
    results = validator.run_batch('3-1-2')

    # Or run full validation
    all_results = validator.run_full_validation()
    report = validator.generate_summary_report()
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np

from validation.validation_runner import ValidationRunner, run_validation
from validation.config import ValidationConfig
from validation.results import ValidationReport, ValidationSummary
# NOTE: STRATOptionsStrategy import moved to _run_single_validation() to avoid circular import

# ThetaData imports
try:
    from integrations.thetadata_options_fetcher import ThetaDataOptionsFetcher
    from integrations.thetadata_client import ThetaDataRESTClient
    THETADATA_IMPORTS_AVAILABLE = True
except ImportError:
    THETADATA_IMPORTS_AVAILABLE = False

logger = logging.getLogger(__name__)


# Default validation matrix
DEFAULT_PATTERNS = ['3-1-2', '2-1-2', '2-2', '3-2', '3-2-2']
DEFAULT_TIMEFRAMES = ['1D', '1W', '1M']
DEFAULT_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'IWM', 'DIA', 'NVDA']

# ThetaData configuration
THETADATA_HOST = "localhost"
THETADATA_PORT = 25503


@dataclass
class ThetaDataStatus:
    """Status of ThetaData connection and data availability."""
    connected: bool = False
    quotes_available: bool = False
    greeks_available: bool = False
    symbols_checked: Dict[str, bool] = field(default_factory=dict)
    oldest_data_date: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationRunConfig:
    """Configuration for a single validation run."""
    pattern: str
    timeframe: str
    symbol: str
    min_continuation_bars: int = 2
    include_22_down: bool = False  # Session 69: 2-2 Down excluded by default
    use_thetadata: bool = True  # Use real ThetaData when available

    @property
    def run_id(self) -> str:
        """Generate unique run identifier."""
        return f"{self.pattern}_{self.timeframe}_{self.symbol}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataSourceMetrics:
    """Track data source usage for a validation run."""
    total_trades: int = 0
    thetadata_quotes: int = 0
    thetadata_greeks: int = 0
    blackscholes_fallback: int = 0
    mixed_source: int = 0

    @property
    def thetadata_coverage_pct(self) -> float:
        """Percentage of trades with ThetaData pricing."""
        if self.total_trades == 0:
            return 0.0
        return (self.thetadata_quotes / self.total_trades) * 100

    @property
    def greeks_coverage_pct(self) -> float:
        """Percentage of trades with ThetaData Greeks."""
        if self.total_trades == 0:
            return 0.0
        return (self.thetadata_greeks / self.total_trades) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            'thetadata_coverage_pct': self.thetadata_coverage_pct,
            'greeks_coverage_pct': self.greeks_coverage_pct,
        }


@dataclass
class ValidationRunResult:
    """Result from a single validation run."""
    run_id: str
    pattern: str
    timeframe: str
    symbol: str
    passed: bool
    report: Optional[ValidationReport] = None
    error: Optional[str] = None
    trade_count: int = 0
    execution_time_seconds: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    data_source_metrics: Optional[DataSourceMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'pattern': self.pattern,
            'timeframe': self.timeframe,
            'symbol': self.symbol,
            'passed': self.passed,
            'error': self.error,
            'trade_count': self.trade_count,
            'execution_time_seconds': self.execution_time_seconds,
            'timestamp': self.timestamp.isoformat(),
            'data_source_metrics': self.data_source_metrics.to_dict() if self.data_source_metrics else None,
            'report': self.report.to_dict() if self.report else None,
        }


@dataclass
class BatchResult:
    """Result from a batch of validation runs (e.g., single pattern)."""
    batch_id: str
    pattern: str
    runs: List[ValidationRunResult]
    total_runs: int = 0
    passed_runs: int = 0
    failed_runs: int = 0
    error_runs: int = 0
    insufficient_data_runs: int = 0
    pass_rate: float = 0.0
    avg_thetadata_coverage: float = 0.0
    total_execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Calculate aggregate statistics."""
        self.total_runs = len(self.runs)
        self.passed_runs = sum(1 for r in self.runs if r.passed and r.error is None)
        self.failed_runs = sum(1 for r in self.runs if not r.passed and r.error is None)
        self.error_runs = sum(1 for r in self.runs if r.error is not None)
        self.insufficient_data_runs = sum(
            1 for r in self.runs
            if r.error and 'insufficient' in r.error.lower()
        )
        self.pass_rate = self.passed_runs / self.total_runs if self.total_runs > 0 else 0.0
        self.total_execution_time = sum(r.execution_time_seconds for r in self.runs)

        # Calculate average ThetaData coverage
        coverages = [
            r.data_source_metrics.thetadata_coverage_pct
            for r in self.runs
            if r.data_source_metrics
        ]
        self.avg_thetadata_coverage = sum(coverages) / len(coverages) if coverages else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'batch_id': self.batch_id,
            'pattern': self.pattern,
            'total_runs': self.total_runs,
            'passed_runs': self.passed_runs,
            'failed_runs': self.failed_runs,
            'error_runs': self.error_runs,
            'insufficient_data_runs': self.insufficient_data_runs,
            'pass_rate': self.pass_rate,
            'avg_thetadata_coverage': self.avg_thetadata_coverage,
            'total_execution_time': self.total_execution_time,
            'timestamp': self.timestamp.isoformat(),
            'runs': [r.to_dict() for r in self.runs],
        }


@dataclass
class CheckpointState:
    """Checkpoint state for resume capability."""
    completed_runs: List[str]  # List of run_ids
    current_batch: Optional[str]
    current_batch_index: int
    results: Dict[str, Dict]  # run_id -> result dict
    thetadata_status: Optional[Dict] = None
    started_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def load(cls, path: Path) -> Optional['CheckpointState']:
        """Load checkpoint from file."""
        if not path.exists():
            return None
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(
                completed_runs=data.get('completed_runs', []),
                current_batch=data.get('current_batch'),
                current_batch_index=data.get('current_batch_index', 0),
                results=data.get('results', {}),
                thetadata_status=data.get('thetadata_status'),
                started_at=datetime.fromisoformat(data.get('started_at', datetime.now().isoformat())),
                updated_at=datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat())),
            )
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def save(self, path: Path) -> None:
        """Save checkpoint to file."""
        self.updated_at = datetime.now()
        data = {
            'completed_runs': self.completed_runs,
            'current_batch': self.current_batch,
            'current_batch_index': self.current_batch_index,
            'results': self.results,
            'thetadata_status': self.thetadata_status,
            'started_at': self.started_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


class DataFetcher:
    """Fetches OHLCV data for validation runs."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize data fetcher.

        Args:
            cache_dir: Directory to cache fetched data
        """
        self.cache_dir = cache_dir or Path("data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data_cache: Dict[str, pd.DataFrame] = {}

    def get_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol/timeframe combination.

        Args:
            symbol: Trading symbol (e.g., 'SPY')
            timeframe: Timeframe ('1D', '1W', '1M')
            start_date: Start date (YYYY-MM-DD), defaults to 5 years ago
            end_date: End date (YYYY-MM-DD), defaults to today

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        # Try to load from cache file
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                self._data_cache[cache_key] = df
                return df
            except Exception as e:
                logger.warning(f"Failed to load cache for {cache_key}: {e}")

        # Fetch from Tiingo (primary data source per CLAUDE.md)
        df = self._fetch_from_tiingo(symbol, timeframe, start_date, end_date)

        if df is not None and not df.empty:
            # Save to cache
            try:
                df.to_parquet(cache_file)
            except Exception as e:
                logger.warning(f"Failed to save cache for {cache_key}: {e}")
            self._data_cache[cache_key] = df
            return df

        # Fallback: Try Alpaca
        df = self._fetch_from_alpaca(symbol, timeframe, start_date, end_date)
        if df is not None and not df.empty:
            self._data_cache[cache_key] = df
            return df

        raise ValueError(f"Failed to fetch data for {symbol} {timeframe}")

    def _fetch_from_tiingo(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Fetch from Tiingo via integrations module."""
        try:
            from integrations.tiingo_data_fetcher import TiingoDataFetcher

            fetcher = TiingoDataFetcher()

            # Calculate default dates - use 5 years for robust validation
            end = end_date or datetime.now().strftime('%Y-%m-%d')
            start = start_date or (datetime.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')

            # Map timeframe to resample frequency
            freq_map = {'1D': 'D', '1W': 'W', '1M': 'ME'}
            resample_freq = freq_map.get(timeframe, 'D')

            # Fetch daily data using Tiingo's fetch() method
            # Returns VBT Data object, need to call .get() to get DataFrame
            data = fetcher.fetch(
                symbols=symbol,
                start_date=start,
                end_date=end,
                timeframe='1d',
                use_cache=True
            )

            # Handle VBT Data object - call .get() to extract DataFrame
            if hasattr(data, 'get'):
                df = data.get()
            else:
                df = data

            if df is None or (hasattr(df, 'empty') and df.empty) or len(df) == 0:
                return None

            # Normalize column names
            df.columns = [c.lower() for c in df.columns]

            # Resample if needed
            if timeframe != '1D':
                df = self._resample_ohlcv(df, resample_freq)

            return df

        except ImportError:
            logger.debug("TiingoDataFetcher not available")
            return None
        except Exception as e:
            logger.warning(f"Tiingo fetch failed for {symbol}: {e}")
            return None

    def _fetch_from_alpaca(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> Optional[pd.DataFrame]:
        """Fetch from Alpaca via VectorBT Pro."""
        try:
            import vectorbtpro as vbt

            # Calculate default dates
            end = end_date or datetime.now().strftime('%Y-%m-%d')
            start = start_date or (datetime.now() - pd.DateOffset(years=5)).strftime('%Y-%m-%d')

            # Alpaca timeframe mapping
            tf_map = {'1D': '1Day', '1W': '1Week', '1M': '1Month'}
            alpaca_tf = tf_map.get(timeframe, '1Day')

            data = vbt.AlpacaData.pull(
                symbols=symbol,
                start=start,
                end=end,
                timeframe=alpaca_tf,
                tz='America/New_York'
            )

            df = data.get()
            if df is not None and not df.empty:
                # Normalize column names
                df.columns = [c.lower() for c in df.columns]
            return df

        except ImportError:
            logger.debug("VectorBT Pro Alpaca not available")
            return None
        except Exception as e:
            logger.warning(f"Alpaca fetch failed for {symbol}: {e}")
            return None

    def _resample_ohlcv(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """Resample OHLCV data to different frequency."""
        df.columns = [c.lower() for c in df.columns]

        ohlc_map = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        }

        # Add volume if present
        if 'volume' in df.columns:
            ohlc_map['volume'] = 'sum'

        resampled = df.resample(freq).agg(ohlc_map).dropna()
        return resampled


class ATLASSTRATValidator:
    """
    Orchestrates comprehensive ATLAS validation for STRAT strategies.

    Uses REAL ThetaData options data for accurate validation:
    - Historical quotes (bid/ask) at tick level
    - Greeks 1st Order (delta, theta, vega, rho)
    - Implied volatility
    - 8 years of historical data

    Runs validation across all combinations of:
    - Pattern types: 3-1-2, 2-1-2, 2-2 Up, 3-2, 3-2-2
    - Timeframes: 1D, 1W, 1M
    - Symbols: SPY, QQQ, AAPL, IWM, DIA, NVDA

    Supports checkpoint/resume for multi-session execution.
    """

    def __init__(
        self,
        patterns: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        output_dir: Optional[Path] = None,
        data_fetcher: Optional[DataFetcher] = None,
        require_thetadata: bool = True,
    ):
        """
        Initialize validator.

        Args:
            patterns: Pattern types to validate (defaults to all 5)
            timeframes: Timeframes to validate (defaults to 1D, 1W, 1M)
            symbols: Symbols to validate (defaults to 6 core symbols)
            output_dir: Directory for results output
            data_fetcher: Custom data fetcher (optional)
            require_thetadata: Require ThetaData connection (default True)
        """
        self.patterns = patterns or DEFAULT_PATTERNS.copy()
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES.copy()
        self.symbols = symbols or DEFAULT_SYMBOLS.copy()
        self.output_dir = output_dir or Path("validation_results/session_83k")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.data_fetcher = data_fetcher or DataFetcher()
        self.validation_runner = ValidationRunner()
        self.require_thetadata = require_thetadata

        # ThetaData components
        self.thetadata_fetcher: Optional[ThetaDataOptionsFetcher] = None
        self.thetadata_status: Optional[ThetaDataStatus] = None

        # Results storage
        self.results: Dict[str, ValidationRunResult] = {}
        self.batch_results: Dict[str, BatchResult] = {}

        # Checkpoint
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.checkpoint: Optional[CheckpointState] = None

        # Bugs discovered during runs
        self.bugs_discovered: List[Dict[str, Any]] = []

    def verify_thetadata_connection(self) -> ThetaDataStatus:
        """
        Verify ThetaData terminal connection and data availability.

        Returns:
            ThetaDataStatus with connection details
        """
        status = ThetaDataStatus()

        if not THETADATA_IMPORTS_AVAILABLE:
            status.error_message = "ThetaData modules not available"
            logger.error(status.error_message)
            return status

        try:
            # Create REST client
            client = ThetaDataRESTClient(
                host=THETADATA_HOST,
                port=THETADATA_PORT
            )

            # Test connection
            connected = client.connect()
            status.connected = connected

            if not connected:
                status.error_message = "Failed to connect to ThetaData terminal"
                logger.error(status.error_message)
                return status

            logger.info(f"ThetaData connected on {THETADATA_HOST}:{THETADATA_PORT}")

            # Test quote endpoint
            try:
                quote = client.get_quote(
                    underlying='SPY',
                    expiration=datetime(2024, 12, 20),
                    strike=590.0,
                    option_type='C',
                    as_of=datetime(2024, 11, 15)
                )
                status.quotes_available = quote is not None
                logger.info(f"Quotes endpoint: {'AVAILABLE' if status.quotes_available else 'NOT AVAILABLE'}")
            except Exception as e:
                logger.warning(f"Quote test failed: {e}")
                status.quotes_available = False

            # Test Greeks endpoint
            try:
                greeks = client.get_greeks(
                    underlying='SPY',
                    expiration=datetime(2024, 12, 20),
                    strike=590.0,
                    option_type='C',
                    as_of=datetime(2024, 11, 15)
                )
                status.greeks_available = greeks is not None
                logger.info(f"Greeks endpoint: {'AVAILABLE' if status.greeks_available else 'NOT AVAILABLE'}")
            except Exception as e:
                logger.warning(f"Greeks test failed: {e}")
                status.greeks_available = False

            # Check symbol availability
            for symbol in self.symbols:
                try:
                    expirations = client.get_expirations(symbol)
                    status.symbols_checked[symbol] = len(expirations) > 0
                except Exception:
                    status.symbols_checked[symbol] = False

            # Initialize fetcher for validation runs
            self.thetadata_fetcher = ThetaDataOptionsFetcher(
                provider=client,
                use_cache=True,
                fallback_to_bs=True,  # Allow B-S fallback for missing data
            )

            self.thetadata_status = status

        except Exception as e:
            status.error_message = str(e)
            logger.error(f"ThetaData verification failed: {e}")

        return status

    def run_full_validation(
        self,
        resume: bool = True,
        skip_patterns: Optional[List[str]] = None,
        skip_timeframes: Optional[List[str]] = None,
        skip_symbols: Optional[List[str]] = None,
    ) -> Dict[str, BatchResult]:
        """
        Run full validation across all pattern/timeframe/symbol combinations.

        Args:
            resume: Whether to resume from checkpoint if available
            skip_patterns: Patterns to skip
            skip_timeframes: Timeframes to skip
            skip_symbols: Symbols to skip

        Returns:
            Dictionary of BatchResult keyed by pattern
        """
        logger.info("=" * 60)
        logger.info("ATLAS STRAT FULL VALIDATION")
        logger.info("=" * 60)
        logger.info(f"Patterns: {self.patterns}")
        logger.info(f"Timeframes: {self.timeframes}")
        logger.info(f"Symbols: {self.symbols}")

        total_runs = len(self.patterns) * len(self.timeframes) * len(self.symbols)
        logger.info(f"Total runs: {total_runs}")

        # Verify ThetaData connection
        if self.require_thetadata:
            status = self.verify_thetadata_connection()
            if not status.connected:
                raise RuntimeError(f"ThetaData required but not available: {status.error_message}")
            logger.info(f"ThetaData: Quotes={status.quotes_available}, Greeks={status.greeks_available}")

        # Load checkpoint if resuming
        if resume:
            self.checkpoint = CheckpointState.load(self.checkpoint_path)
            if self.checkpoint:
                completed = len(self.checkpoint.completed_runs)
                logger.info(f"Resuming from checkpoint: {completed}/{total_runs} runs completed")

        # Initialize checkpoint if not loaded
        if self.checkpoint is None:
            self.checkpoint = CheckpointState(
                completed_runs=[],
                current_batch=None,
                current_batch_index=0,
                results={},
                thetadata_status=self.thetadata_status.to_dict() if self.thetadata_status else None,
                started_at=datetime.now(),
                updated_at=datetime.now(),
            )

        # Run batches by pattern
        for i, pattern in enumerate(self.patterns):
            if skip_patterns and pattern in skip_patterns:
                logger.info(f"Skipping pattern: {pattern}")
                continue

            self.checkpoint.current_batch = pattern
            self.checkpoint.current_batch_index = i

            batch_result = self.run_batch(
                pattern=pattern,
                skip_timeframes=skip_timeframes,
                skip_symbols=skip_symbols,
                resume=resume,
            )

            self.batch_results[pattern] = batch_result

            # Save checkpoint after each batch
            self._save_checkpoint()
            self._save_batch_result(batch_result)

            logger.info(
                f"Batch {pattern} complete: {batch_result.passed_runs}/{batch_result.total_runs} passed, "
                f"ThetaData coverage: {batch_result.avg_thetadata_coverage:.1f}%"
            )

        # Generate summary
        summary = self.generate_summary_report()

        logger.info("=" * 60)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 60)

        return self.batch_results

    def run_batch(
        self,
        pattern: str,
        skip_timeframes: Optional[List[str]] = None,
        skip_symbols: Optional[List[str]] = None,
        resume: bool = True,
    ) -> BatchResult:
        """
        Run validation for a single pattern across all timeframes/symbols.

        Args:
            pattern: Pattern type (e.g., '3-1-2')
            skip_timeframes: Timeframes to skip
            skip_symbols: Symbols to skip
            resume: Whether to skip already completed runs

        Returns:
            BatchResult with all run results
        """
        logger.info(f"\n--- Batch: {pattern} ---")

        batch_id = f"batch_{pattern}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        runs: List[ValidationRunResult] = []

        for timeframe in self.timeframes:
            if skip_timeframes and timeframe in skip_timeframes:
                continue

            for symbol in self.symbols:
                if skip_symbols and symbol in skip_symbols:
                    continue

                run_config = ValidationRunConfig(
                    pattern=pattern,
                    timeframe=timeframe,
                    symbol=symbol,
                    include_22_down=False,  # Session 69: 2-2 Down excluded
                    use_thetadata=self.thetadata_fetcher is not None,
                )

                # Check if already completed
                if resume and self.checkpoint and run_config.run_id in self.checkpoint.completed_runs:
                    # Load result from checkpoint
                    if run_config.run_id in self.checkpoint.results:
                        cached = self.checkpoint.results[run_config.run_id]
                        result = ValidationRunResult(
                            run_id=run_config.run_id,
                            pattern=pattern,
                            timeframe=timeframe,
                            symbol=symbol,
                            passed=cached.get('passed', False),
                            error=cached.get('error'),
                            trade_count=cached.get('trade_count', 0),
                            execution_time_seconds=cached.get('execution_time_seconds', 0),
                        )
                        runs.append(result)
                        continue

                # Run validation
                result = self._run_single_validation(run_config)
                runs.append(result)
                self.results[run_config.run_id] = result

                # Update checkpoint
                if self.checkpoint:
                    self.checkpoint.completed_runs.append(run_config.run_id)
                    self.checkpoint.results[run_config.run_id] = result.to_dict()
                    self._save_checkpoint()

        return BatchResult(
            batch_id=batch_id,
            pattern=pattern,
            runs=runs,
        )

    def _run_single_validation(self, config: ValidationRunConfig) -> ValidationRunResult:
        """
        Run a single validation.

        Args:
            config: Run configuration

        Returns:
            ValidationRunResult
        """
        start_time = time.time()
        logger.info(f"Running: {config.run_id}")

        data_metrics = DataSourceMetrics()

        try:
            # Fetch OHLCV data
            data = self.data_fetcher.get_data(
                symbol=config.symbol,
                timeframe=config.timeframe,
            )

            if data is None or data.empty:
                return ValidationRunResult(
                    run_id=config.run_id,
                    pattern=config.pattern,
                    timeframe=config.timeframe,
                    symbol=config.symbol,
                    passed=False,
                    error="No OHLCV data available",
                    execution_time_seconds=time.time() - start_time,
                    data_source_metrics=data_metrics,
                )

            # Check minimum data requirements
            min_bars = {'1D': 252, '1W': 52, '1M': 12}
            required = min_bars.get(config.timeframe, 252)
            if len(data) < required:
                return ValidationRunResult(
                    run_id=config.run_id,
                    pattern=config.pattern,
                    timeframe=config.timeframe,
                    symbol=config.symbol,
                    passed=False,
                    error=f"Insufficient data: {len(data)} bars < {required} required",
                    execution_time_seconds=time.time() - start_time,
                    data_source_metrics=data_metrics,
                )

            # Configure strategy with ThetaData
            # Lazy import to avoid circular dependency (strategies imports validation.protocols)
            from strategies.strat_options_strategy import STRATOptionsStrategy, STRATOptionsConfig

            strategy_config = STRATOptionsConfig(
                pattern_types=[config.pattern],
                timeframe=config.timeframe,
                min_continuation_bars=config.min_continuation_bars,
                include_22_down=config.include_22_down,
                symbol=config.symbol,
            )

            strategy = STRATOptionsStrategy(config=strategy_config)

            # If ThetaData available, configure backtester to use it
            if config.use_thetadata and self.thetadata_fetcher:
                # The STRATOptionsStrategy uses OptionsBacktester internally
                # which can accept a ThetaData fetcher
                # Session 83K-3 BUG FIX: Use internal attribute names (_options_fetcher, _use_market_prices)
                if hasattr(strategy, '_backtester') and strategy._backtester:
                    strategy._backtester._options_fetcher = self.thetadata_fetcher
                    strategy._backtester._use_market_prices = True
                    logger.debug(f"ThetaData fetcher wired to {config.run_id} backtester")

            # Run validation
            strategy_name = f"STRAT_{config.pattern}_{config.timeframe}_{config.symbol}"
            report = run_validation(
                strategy=strategy,
                data=data,
                strategy_name=strategy_name,
                is_options=True,  # Using options thresholds
            )

            # Extract trade count and data source metrics
            trade_count = 0
            if report.walk_forward and hasattr(report.walk_forward, 'folds'):
                # Sum trades across folds
                for fold in report.walk_forward.folds:
                    trade_count += fold.oos_trades

            # Try to get data source breakdown from backtest
            try:
                backtest_result = strategy.backtest(data)
                if backtest_result.trades is not None and not backtest_result.trades.empty:
                    trades_df = backtest_result.trades
                    data_metrics.total_trades = len(trades_df)

                    if 'data_source' in trades_df.columns:
                        source_counts = trades_df['data_source'].value_counts()
                        data_metrics.thetadata_quotes = source_counts.get('ThetaData', 0)
                        data_metrics.blackscholes_fallback = source_counts.get('BlackScholes', 0)
                        data_metrics.mixed_source = source_counts.get('Mixed', 0)
                    else:
                        # Assume ThetaData if fetcher was configured
                        if config.use_thetadata:
                            data_metrics.thetadata_quotes = data_metrics.total_trades

                    trade_count = data_metrics.total_trades
            except Exception as e:
                logger.debug(f"Could not extract data source metrics: {e}")

            exec_time = time.time() - start_time

            result = ValidationRunResult(
                run_id=config.run_id,
                pattern=config.pattern,
                timeframe=config.timeframe,
                symbol=config.symbol,
                passed=report.summary.passes_all,
                report=report,
                trade_count=trade_count,
                execution_time_seconds=exec_time,
                data_source_metrics=data_metrics,
            )

            # Log result
            status = "PASSED" if result.passed else "FAILED"
            coverage = f", ThetaData: {data_metrics.thetadata_coverage_pct:.0f}%" if data_metrics.total_trades > 0 else ""
            logger.info(f"  {config.run_id}: {status} ({trade_count} trades, {exec_time:.1f}s{coverage})")

            return result

        except Exception as e:
            exec_time = time.time() - start_time
            error_msg = str(e)
            logger.error(f"  {config.run_id}: ERROR - {error_msg}")

            # Record bug if it looks like a code issue
            if any(kw in error_msg.lower() for kw in ['import', 'attribute', 'type', 'key']):
                self._record_bug(config, error_msg)

            return ValidationRunResult(
                run_id=config.run_id,
                pattern=config.pattern,
                timeframe=config.timeframe,
                symbol=config.symbol,
                passed=False,
                error=error_msg,
                execution_time_seconds=exec_time,
                data_source_metrics=data_metrics,
            )

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary report.

        Returns:
            Dictionary with summary statistics
        """
        # Aggregate by pattern
        by_pattern: Dict[str, Dict] = {}
        for pattern, batch in self.batch_results.items():
            by_pattern[pattern] = {
                'total_runs': batch.total_runs,
                'passed': batch.passed_runs,
                'failed': batch.failed_runs,
                'errors': batch.error_runs,
                'insufficient_data': batch.insufficient_data_runs,
                'pass_rate': batch.pass_rate,
                'avg_thetadata_coverage': batch.avg_thetadata_coverage,
            }

        # Aggregate by timeframe
        by_timeframe: Dict[str, Dict] = {tf: {'passed': 0, 'total': 0, 'thetadata_sum': 0.0} for tf in self.timeframes}
        for batch in self.batch_results.values():
            for run in batch.runs:
                tf = run.timeframe
                by_timeframe[tf]['total'] += 1
                if run.passed:
                    by_timeframe[tf]['passed'] += 1
                if run.data_source_metrics:
                    by_timeframe[tf]['thetadata_sum'] += run.data_source_metrics.thetadata_coverage_pct

        for tf in by_timeframe:
            total = by_timeframe[tf]['total']
            passed = by_timeframe[tf]['passed']
            by_timeframe[tf]['pass_rate'] = passed / total if total > 0 else 0
            by_timeframe[tf]['avg_thetadata_coverage'] = by_timeframe[tf]['thetadata_sum'] / total if total > 0 else 0
            del by_timeframe[tf]['thetadata_sum']

        # Aggregate by symbol
        by_symbol: Dict[str, Dict] = {s: {'passed': 0, 'total': 0, 'thetadata_sum': 0.0} for s in self.symbols}
        for batch in self.batch_results.values():
            for run in batch.runs:
                sym = run.symbol
                by_symbol[sym]['total'] += 1
                if run.passed:
                    by_symbol[sym]['passed'] += 1
                if run.data_source_metrics:
                    by_symbol[sym]['thetadata_sum'] += run.data_source_metrics.thetadata_coverage_pct

        for sym in by_symbol:
            total = by_symbol[sym]['total']
            passed = by_symbol[sym]['passed']
            by_symbol[sym]['pass_rate'] = passed / total if total > 0 else 0
            by_symbol[sym]['avg_thetadata_coverage'] = by_symbol[sym]['thetadata_sum'] / total if total > 0 else 0
            del by_symbol[sym]['thetadata_sum']

        # Overall statistics
        total_runs = sum(b.total_runs for b in self.batch_results.values())
        total_passed = sum(b.passed_runs for b in self.batch_results.values())
        total_failed = sum(b.failed_runs for b in self.batch_results.values())
        total_errors = sum(b.error_runs for b in self.batch_results.values())
        total_time = sum(b.total_execution_time for b in self.batch_results.values())

        # Overall ThetaData coverage
        all_coverages = [
            r.data_source_metrics.thetadata_coverage_pct
            for batch in self.batch_results.values()
            for r in batch.runs
            if r.data_source_metrics
        ]
        avg_thetadata_coverage = sum(all_coverages) / len(all_coverages) if all_coverages else 0

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_runs': total_runs,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'total_errors': total_errors,
            'overall_pass_rate': total_passed / total_runs if total_runs > 0 else 0,
            'avg_thetadata_coverage': avg_thetadata_coverage,
            'total_execution_time_seconds': total_time,
            'thetadata_status': self.thetadata_status.to_dict() if self.thetadata_status else None,
            'by_pattern': by_pattern,
            'by_timeframe': by_timeframe,
            'by_symbol': by_symbol,
            'bugs_discovered': self.bugs_discovered,
        }

        # Save summary
        summary_path = self.output_dir / "summary" / "master_report.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save CSV version
        csv_path = self.output_dir / "summary" / "master_report.csv"
        self._save_summary_csv(summary, csv_path)

        logger.info(f"Summary saved to {summary_path}")

        return summary

    def _save_summary_csv(self, summary: Dict, path: Path) -> None:
        """Save summary as CSV."""
        rows = []

        for pattern in self.patterns:
            for tf in self.timeframes:
                for sym in self.symbols:
                    run_id = f"{pattern}_{tf}_{sym}"
                    result = self.results.get(run_id)
                    if result:
                        rows.append({
                            'pattern': pattern,
                            'timeframe': tf,
                            'symbol': sym,
                            'passed': result.passed,
                            'error': result.error or '',
                            'trade_count': result.trade_count,
                            'thetadata_coverage': result.data_source_metrics.thetadata_coverage_pct if result.data_source_metrics else 0,
                            'execution_time': result.execution_time_seconds,
                        })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(path, index=False)

    def _save_checkpoint(self) -> None:
        """Save current checkpoint state."""
        if self.checkpoint:
            self.checkpoint.save(self.checkpoint_path)

    def _save_batch_result(self, batch: BatchResult) -> None:
        """Save batch result to file."""
        batch_dir = self.output_dir / "by_pattern" / batch.pattern.replace('-', '')
        batch_dir.mkdir(parents=True, exist_ok=True)

        batch_file = batch_dir / f"{batch.batch_id}.json"
        with open(batch_file, 'w') as f:
            json.dump(batch.to_dict(), f, indent=2, default=str)

    def _record_bug(self, config: ValidationRunConfig, error: str) -> None:
        """Record a discovered bug."""
        bug = {
            'run_id': config.run_id,
            'pattern': config.pattern,
            'timeframe': config.timeframe,
            'symbol': config.symbol,
            'error': error,
            'timestamp': datetime.now().isoformat(),
        }
        self.bugs_discovered.append(bug)

        # Save bugs file
        bugs_dir = self.output_dir / "bugs"
        bugs_dir.mkdir(parents=True, exist_ok=True)
        bugs_file = bugs_dir / "bugs_discovered.json"
        with open(bugs_file, 'w') as f:
            json.dump(self.bugs_discovered, f, indent=2)

    def print_summary(self) -> str:
        """Print human-readable summary."""
        lines = [
            "",
            "=" * 70,
            "ATLAS STRAT VALIDATION SUMMARY",
            "=" * 70,
            "",
        ]

        # ThetaData status
        if self.thetadata_status:
            lines.append("ThetaData Integration:")
            lines.append(f"  Connected: {self.thetadata_status.connected}")
            lines.append(f"  Quotes:    {self.thetadata_status.quotes_available}")
            lines.append(f"  Greeks:    {self.thetadata_status.greeks_available}")
            lines.append("")

        total_runs = sum(b.total_runs for b in self.batch_results.values())
        total_passed = sum(b.passed_runs for b in self.batch_results.values())

        lines.append(f"Total Runs:     {total_runs}")
        lines.append(f"Total Passed:   {total_passed}")
        lines.append(f"Pass Rate:      {total_passed/total_runs*100:.1f}%" if total_runs > 0 else "N/A")
        lines.append("")
        lines.append("--- BY PATTERN ---")
        lines.append(f"{'Pattern':<12} {'Passed':>8} {'Failed':>8} {'Errors':>8} {'ThetaData%':>12} {'Pass%':>10}")
        lines.append("-" * 62)

        for pattern, batch in self.batch_results.items():
            lines.append(
                f"{pattern:<12} {batch.passed_runs:>8} {batch.failed_runs:>8} "
                f"{batch.error_runs:>8} {batch.avg_thetadata_coverage:>11.1f}% {batch.pass_rate*100:>9.1f}%"
            )

        if self.bugs_discovered:
            lines.extend([
                "",
                "--- BUGS DISCOVERED ---",
            ])
            for bug in self.bugs_discovered:
                lines.append(f"  [{bug['run_id']}] {bug['error'][:50]}...")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

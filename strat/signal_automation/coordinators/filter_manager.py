"""
EQUITY-87: FilterManager - Extracted from SignalDaemon

Manages signal quality filtering with configurable thresholds.
Applies magnitude, R:R, pattern, and TFC (timeframe continuity) filters.

Responsibilities:
- Apply magnitude threshold filters (SETUP vs COMPLETED)
- Apply risk/reward ratio filters
- Apply pattern whitelist filters
- Apply timeframe continuity (TFC) filters with 1D alignment for 1H patterns
- Log filter rejections with actual vs threshold values
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from strat.paper_signal_scanner import DetectedSignal

logger = logging.getLogger(__name__)


@dataclass
class FilterConfig:
    """
    Configuration for signal quality filters.

    EQUITY-87: Externalized from hardcoded values in SignalDaemon._passes_filters().

    Attributes:
        setup_min_magnitude: Minimum magnitude % for SETUP signals (relaxed threshold)
        completed_min_magnitude: Minimum magnitude % for COMPLETED signals
        setup_min_rr: Minimum R:R for SETUP signals (relaxed threshold)
        completed_min_rr: Minimum R:R for COMPLETED signals
        tfc_enabled: Whether TFC filtering is enabled
        tfc_minimums: Per-timeframe minimum TFC scores
        allowed_patterns: List of allowed pattern types (empty = all allowed)
    """
    # SETUP signals use relaxed thresholds (Session 83K-71)
    # SETUP signals ending in inside bar have naturally lower magnitude
    setup_min_magnitude: float = 0.1
    setup_min_rr: float = 0.3

    # COMPLETED signals use strict thresholds
    completed_min_magnitude: float = 0.5
    completed_min_rr: float = 1.0

    # TFC (Timeframe Continuity) settings
    tfc_enabled: bool = True
    tfc_minimums: Dict[str, int] = field(default_factory=lambda: {
        '1H': 3,  # 3/4 aligned for hourly (special handling for 2/4+1D)
        '4H': 2,  # 2/3 aligned
        '1D': 2,  # 2/3 aligned
        '1W': 1,  # 1/2 aligned
        '1M': 1,  # 1/1 aligned
    })

    # Pattern whitelist (empty = all allowed)
    allowed_patterns: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> 'FilterConfig':
        """
        Create FilterConfig from environment variables.

        Environment variables:
            SIGNAL_SETUP_MIN_MAGNITUDE: float (default 0.1)
            SIGNAL_SETUP_MIN_RR: float (default 0.3)
            SIGNAL_TFC_FILTER_ENABLED: bool (default true)
        """
        return cls(
            setup_min_magnitude=float(os.environ.get('SIGNAL_SETUP_MIN_MAGNITUDE', '0.1')),
            setup_min_rr=float(os.environ.get('SIGNAL_SETUP_MIN_RR', '0.3')),
            tfc_enabled=os.environ.get('SIGNAL_TFC_FILTER_ENABLED', 'true').lower() == 'true',
        )


class FilterManager:
    """
    Manages signal quality filtering for signal automation.

    Extracted from SignalDaemon as part of EQUITY-87 Phase 2 refactoring.
    Uses Facade pattern - daemon delegates filtering to this coordinator.

    Args:
        config: FilterConfig with threshold settings
        scan_config: ScanConfig for COMPLETED thresholds (min_magnitude_pct, min_risk_reward)
    """

    def __init__(
        self,
        config: Optional[FilterConfig] = None,
        scan_config: Optional[Any] = None,  # ScanConfig from config.py
    ):
        self._config = config or FilterConfig.from_env()
        self._scan_config = scan_config

    def passes_filters(self, signal: DetectedSignal) -> bool:
        """
        Check if signal passes all quality filters.

        Session EQUITY-47: Added logging for filter rejections with actual vs threshold values.
        Session 83K-71: SETUP signals use relaxed thresholds.
        Session EQUITY-62/65: TFC filtering with 1D alignment for 1H patterns.

        Args:
            signal: Signal to check

        Returns:
            True if passes all filters
        """
        # Human-readable key for logging (not database signal_key)
        signal_key = f"{signal.symbol}_{signal.timeframe}_{signal.pattern_type}_{signal.direction}"

        # Check magnitude filter
        if not self._check_magnitude(signal, signal_key):
            return False

        # Check R:R filter
        if not self._check_rr(signal, signal_key):
            return False

        # Check pattern filter
        if not self._check_pattern(signal, signal_key):
            return False

        # Check TFC filter
        if not self._check_tfc(signal, signal_key):
            return False

        return True

    def _is_setup_signal(self, signal: DetectedSignal) -> bool:
        """Check if signal is a SETUP type (vs COMPLETED)."""
        return getattr(signal, 'signal_type', 'COMPLETED') == 'SETUP'

    def _check_magnitude(self, signal: DetectedSignal, signal_key: str) -> bool:
        """
        Check if signal meets minimum magnitude threshold.

        SETUP signals use relaxed thresholds since inside bars have
        naturally lower magnitude (target is first directional bar's extreme).

        Note: SETUP thresholds are read from environment at call time to support
        runtime configuration changes (matching original daemon.py behavior).

        Args:
            signal: Signal to check
            signal_key: Human-readable key for logging

        Returns:
            True if passes magnitude filter
        """
        is_setup = self._is_setup_signal(signal)

        if is_setup:
            # Read env var at call time to support dynamic configuration
            min_magnitude = float(os.environ.get(
                'SIGNAL_SETUP_MIN_MAGNITUDE',
                str(self._config.setup_min_magnitude)
            ))
            if signal.magnitude_pct < min_magnitude:
                logger.info(
                    f"FILTER REJECTED: {signal_key} - "
                    f"magnitude {signal.magnitude_pct:.3f}% < min {min_magnitude}% (SETUP)"
                )
                return False
        else:
            # COMPLETED signals use config threshold or default
            min_magnitude = (
                self._scan_config.min_magnitude_pct
                if self._scan_config
                else self._config.completed_min_magnitude
            )
            if signal.magnitude_pct < min_magnitude:
                logger.info(
                    f"FILTER REJECTED: {signal_key} - "
                    f"magnitude {signal.magnitude_pct:.3f}% < min {min_magnitude}%"
                )
                return False

        return True

    def _check_rr(self, signal: DetectedSignal, signal_key: str) -> bool:
        """
        Check if signal meets minimum risk/reward ratio threshold.

        Note: SETUP thresholds are read from environment at call time to support
        runtime configuration changes (matching original daemon.py behavior).

        Args:
            signal: Signal to check
            signal_key: Human-readable key for logging

        Returns:
            True if passes R:R filter
        """
        is_setup = self._is_setup_signal(signal)

        if is_setup:
            # Read env var at call time to support dynamic configuration
            min_rr = float(os.environ.get(
                'SIGNAL_SETUP_MIN_RR',
                str(self._config.setup_min_rr)
            ))
            if signal.risk_reward < min_rr:
                logger.info(
                    f"FILTER REJECTED: {signal_key} - "
                    f"R:R {signal.risk_reward:.2f} < min {min_rr} (SETUP)"
                )
                return False
        else:
            # COMPLETED signals use config threshold or default
            min_rr = (
                self._scan_config.min_risk_reward
                if self._scan_config
                else self._config.completed_min_rr
            )
            if signal.risk_reward < min_rr:
                logger.info(
                    f"FILTER REJECTED: {signal_key} - "
                    f"R:R {signal.risk_reward:.2f} < min {min_rr}"
                )
                return False

        return True

    def _check_pattern(self, signal: DetectedSignal, signal_key: str) -> bool:
        """
        Check if signal's pattern type is in allowed patterns list.

        Converts directional pattern names to base patterns for comparison:
        - '2U-2D' -> '2-2'
        - '2D-1-2U' -> '2-1-2'
        - '3-2U' -> '3-2'
        - '3-1-?' -> '3-1-2' (SETUP patterns)

        Args:
            signal: Signal to check
            signal_key: Human-readable key for logging

        Returns:
            True if passes pattern filter (or if no patterns configured)
        """
        # Get allowed patterns from scan_config or filter config
        allowed_patterns = (
            self._scan_config.patterns
            if self._scan_config and self._scan_config.patterns
            else self._config.allowed_patterns
        )

        # If no patterns configured, allow all
        if not allowed_patterns:
            return True

        # Convert directional pattern to base pattern
        base_pattern = self._normalize_pattern(signal.pattern_type)

        if base_pattern not in allowed_patterns:
            logger.info(
                f"FILTER REJECTED: {signal_key} - "
                f"pattern '{base_pattern}' not in allowed patterns {allowed_patterns}"
            )
            return False

        return True

    def _normalize_pattern(self, pattern_type: str) -> str:
        """
        Normalize directional pattern to base pattern for comparison.

        Examples:
            '2U-2D' -> '2-2'
            '3-2U' -> '3-2'
            '2D-1-2U' -> '2-1-2'
            '3-1-?' -> '3-1-2' (SETUP patterns)

        Args:
            pattern_type: Raw pattern type with direction suffixes

        Returns:
            Normalized base pattern
        """
        return pattern_type.replace('2U', '2').replace('2D', '2').replace('-?', '-2')

    def _check_tfc(self, signal: DetectedSignal, signal_key: str) -> bool:
        """
        Check if signal meets timeframe continuity (TFC) requirements.

        Session EQUITY-62/65: TFC filtering with special handling for 1H patterns.
        Per STRAT methodology: Trade WITH the higher timeframes, not against.

        1H Enhancement (Session EQUITY-65):
        - 3/4 or 4/4: PASS (strong alignment)
        - 2/4 with 1D aligned: PASS (daily "control" supports trade)
        - 2/4 without 1D: FAIL (fighting daily or in chop)
        - 0/4 or 1/4: FAIL (weak alignment)

        Note: TFC enabled flag is read from environment at call time to support
        runtime configuration changes (matching original daemon.py behavior).

        Args:
            signal: Signal to check
            signal_key: Human-readable key for logging

        Returns:
            True if passes TFC filter (or if TFC disabled)
        """
        # Read env var at call time to support dynamic configuration
        # Use config as default when env var not set
        env_tfc = os.environ.get('SIGNAL_TFC_FILTER_ENABLED')
        if env_tfc is not None:
            tfc_enabled = env_tfc.lower() == 'true'
        else:
            tfc_enabled = self._config.tfc_enabled
        if not tfc_enabled:
            return True

        # Extract TFC data from signal context
        tfc_score = getattr(signal.context, 'tfc_score', 0) if signal.context else 0
        tfc_alignment = getattr(signal.context, 'tfc_alignment', '') if signal.context else ''
        aligned_timeframes = getattr(signal.context, 'aligned_timeframes', []) if signal.context else []

        timeframe = signal.timeframe

        # Special handling for 1H patterns (Session EQUITY-65)
        if timeframe == '1H':
            return self._check_tfc_hourly(
                tfc_score, aligned_timeframes, tfc_alignment, signal_key
            )

        # Non-1H patterns use standard minimums
        min_aligned = self._config.tfc_minimums.get(timeframe, 2)

        if tfc_score < min_aligned:
            logger.info(
                f"FILTER REJECTED (TFC): {signal_key} - "
                f"TFC score {tfc_score} < min {min_aligned} ({tfc_alignment or 'N/A'})"
            )
            return False

        return True

    def _check_tfc_hourly(
        self,
        tfc_score: int,
        aligned_timeframes: List[str],
        tfc_alignment: str,
        signal_key: str,
    ) -> bool:
        """
        Check TFC for 1H patterns with special 1D alignment requirement.

        Session EQUITY-65: "Control" concept for 1H patterns:
        - 1D represents "who is in control right now" for intraday trades
        - Type 1 (inside) on 1D = "chop" = dangerous for options (theta decay)

        Args:
            tfc_score: TFC alignment score (0-4)
            aligned_timeframes: List of aligned timeframe names
            tfc_alignment: TFC alignment description string
            signal_key: Human-readable key for logging

        Returns:
            True if passes 1H TFC filter
        """
        if tfc_score >= 3:
            # 3/4 or 4/4 always passes (strong alignment)
            return True
        elif tfc_score == 2:
            # 2/4 only passes if 1D is aligned (daily "control" supports trade)
            if '1D' not in aligned_timeframes:
                logger.info(
                    f"FILTER REJECTED (TFC): {signal_key} - "
                    f"TFC 2/4 but 1D not aligned (aligned: {aligned_timeframes}, {tfc_alignment or 'N/A'})"
                )
                return False
            return True
        else:
            # 0/4 or 1/4 fails (weak alignment)
            logger.info(
                f"FILTER REJECTED (TFC): {signal_key} - "
                f"TFC {tfc_score}/4 below minimum ({tfc_alignment or 'N/A'})"
            )
            return False

    @property
    def config(self) -> FilterConfig:
        """Return current filter configuration."""
        return self._config

    def update_config(self, config: FilterConfig) -> None:
        """
        Update filter configuration.

        Args:
            config: New FilterConfig to use
        """
        self._config = config

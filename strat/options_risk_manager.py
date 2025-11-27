"""
Options Risk Manager for ATLAS Trading System.

Session 83H: Implements Greeks-based position limits, circuit breakers, and
pre-trade validation per ATLAS Checklist Section 9.3.

Features:
- Greeks-based portfolio limits (delta, gamma, theta, vega)
- Circuit breaker state machine (NORMAL -> CAUTION -> REDUCED -> HALTED)
- Pre-trade validation hook
- DTE limits with forced exit logic
- Spread width validation

ATLAS Compliance (Section 9.3):
- max_portfolio_delta: 30%
- max_portfolio_gamma: 5%
- max_portfolio_theta: -2% daily
- max_portfolio_vega: 10%
- min_dte_entry: 7 days
- max_dte_entry: 45 days
- forced_exit_dte: 3 days

Usage:
    from strat.options_risk_manager import OptionsRiskManager, OptionsRiskConfig

    config = OptionsRiskConfig()
    risk_manager = OptionsRiskManager(config, account_size=10000.0)

    # Pre-trade validation
    result = risk_manager.validate_new_position(
        delta=0.55,
        gamma=0.02,
        theta=-0.15,
        vega=0.08,
        premium=250.0,
        dte=21,
        spread_pct=0.05
    )

    if result.passed:
        # Execute trade
        pass
    else:
        print(f"Trade rejected: {result.reasons}")
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """
    Circuit breaker states for options risk management.

    States progress from NORMAL to more restrictive levels as
    risk metrics exceed thresholds.

    State Machine:
        NORMAL -> CAUTION -> REDUCED -> HALTED -> EMERGENCY

    Transitions:
        NORMAL: All systems operational, full position sizing
        CAUTION: Warning state, monitor closely
        REDUCED: Reduce position sizes by 50%
        HALTED: No new positions, exit-only mode
        EMERGENCY: Force exit all positions immediately
    """
    NORMAL = "NORMAL"
    CAUTION = "CAUTION"
    REDUCED = "REDUCED"
    HALTED = "HALTED"
    EMERGENCY = "EMERGENCY"


@dataclass
class OptionsRiskConfig:
    """
    Configuration for options risk management.

    Per ATLAS Checklist Section 9.3.1 - Greeks-Based Position Limits.

    Attributes:
        max_portfolio_delta: Max net delta exposure (0.30 = 30%)
        max_portfolio_gamma: Max gamma exposure (0.05 = 5%)
        max_portfolio_theta: Max daily theta bleed as pct of account (-0.02 = -2%)
        max_portfolio_vega: Max vega exposure (0.10 = 10%)
        max_position_delta: Max delta per single position (0.10 = 10%)
        max_contracts_per_symbol: Max contracts per underlying
        max_premium_at_risk: Max premium per position as pct of account (0.05 = 5%)
        min_dte_entry: Minimum DTE for new positions
        max_dte_entry: Maximum DTE for new positions
        forced_exit_dte: DTE at which to force exit
        max_spread_pct: Max bid-ask spread as pct of mid price (0.10 = 10%)
        position_loss_exit_pct: Exit if position loses this much (0.50 = 50%)
        vix_halt_threshold: VIX level to trigger halt (25.0)
        vix_spike_pct: VIX daily spike % to trigger halt (0.25 = 25%)
    """
    # Portfolio-level Greeks limits
    max_portfolio_delta: float = 0.30
    max_portfolio_gamma: float = 0.05
    max_portfolio_theta: float = -0.02  # Negative = daily bleed
    max_portfolio_vega: float = 0.10

    # Position-level limits
    max_position_delta: float = 0.10
    max_contracts_per_symbol: int = 10
    max_premium_at_risk: float = 0.05

    # DTE limits
    min_dte_entry: int = 7
    max_dte_entry: int = 45
    forced_exit_dte: int = 3

    # Spread limits
    max_spread_pct: float = 0.10

    # Exit triggers
    position_loss_exit_pct: float = 0.50

    # VIX-based circuit breakers
    vix_halt_threshold: float = 25.0
    vix_spike_pct: float = 0.25

    # Circuit breaker thresholds (as multipliers of limits)
    caution_threshold: float = 0.80  # 80% of limit -> CAUTION
    reduced_threshold: float = 1.00  # 100% of limit -> REDUCED
    halted_threshold: float = 1.20   # 120% of limit -> HALTED

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'max_portfolio_delta': self.max_portfolio_delta,
            'max_portfolio_gamma': self.max_portfolio_gamma,
            'max_portfolio_theta': self.max_portfolio_theta,
            'max_portfolio_vega': self.max_portfolio_vega,
            'max_position_delta': self.max_position_delta,
            'max_contracts_per_symbol': self.max_contracts_per_symbol,
            'max_premium_at_risk': self.max_premium_at_risk,
            'min_dte_entry': self.min_dte_entry,
            'max_dte_entry': self.max_dte_entry,
            'forced_exit_dte': self.forced_exit_dte,
            'max_spread_pct': self.max_spread_pct,
            'position_loss_exit_pct': self.position_loss_exit_pct,
            'vix_halt_threshold': self.vix_halt_threshold,
            'vix_spike_pct': self.vix_spike_pct,
        }


@dataclass
class ValidationResult:
    """
    Result from pre-trade validation.

    Attributes:
        passed: Whether validation passed
        reasons: List of failure reasons (empty if passed)
        warnings: List of warnings (even if passed)
        adjusted_size: Suggested adjusted position size (if REDUCED state)
        circuit_state: Current circuit breaker state
    """
    passed: bool
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    adjusted_size: Optional[float] = None
    circuit_state: CircuitBreakerState = CircuitBreakerState.NORMAL

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'passed': self.passed,
            'reasons': self.reasons,
            'warnings': self.warnings,
            'adjusted_size': self.adjusted_size,
            'circuit_state': self.circuit_state.value,
        }


@dataclass
class PortfolioGreeks:
    """
    Aggregated Greeks for the entire portfolio.

    Attributes:
        net_delta: Net delta exposure (sum of all position deltas)
        total_gamma: Total gamma exposure
        total_theta: Total daily theta (time decay)
        total_vega: Total vega (IV sensitivity)
        position_count: Number of open positions
        total_premium_at_risk: Total premium invested
    """
    net_delta: float = 0.0
    total_gamma: float = 0.0
    total_theta: float = 0.0
    total_vega: float = 0.0
    position_count: int = 0
    total_premium_at_risk: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert Greeks to dictionary."""
        return {
            'net_delta': self.net_delta,
            'total_gamma': self.total_gamma,
            'total_theta': self.total_theta,
            'total_vega': self.total_vega,
            'position_count': self.position_count,
            'total_premium_at_risk': self.total_premium_at_risk,
        }


@dataclass
class PositionRisk:
    """
    Risk metrics for a single options position.

    Attributes:
        symbol: Underlying symbol
        option_symbol: Full OSI symbol
        delta: Position delta
        gamma: Position gamma
        theta: Position theta (daily)
        vega: Position vega
        premium: Premium paid
        current_value: Current position value
        dte: Days to expiration
        entry_date: When position was opened
        contracts: Number of contracts
        pnl_pct: Current P/L percentage
    """
    symbol: str
    option_symbol: str
    delta: float
    gamma: float
    theta: float
    vega: float
    premium: float
    current_value: float
    dte: int
    entry_date: datetime
    contracts: int = 1
    pnl_pct: float = 0.0

    def __post_init__(self):
        """Calculate P/L percentage."""
        if self.premium > 0:
            self.pnl_pct = (self.current_value - self.premium) / self.premium


@dataclass
class ForceExitSignal:
    """
    Signal indicating a position should be force exited.

    Attributes:
        position: The position to exit
        reason: Why exit is being forced
        urgency: How urgent (1=normal, 2=high, 3=immediate)
    """
    position: PositionRisk
    reason: str
    urgency: int = 1


class OptionsRiskManager:
    """
    Options risk manager with Greeks-based limits and circuit breakers.

    Implements ATLAS Checklist Section 9.3 requirements:
    - Greeks-based position limits
    - Circuit breaker state machine
    - Pre-trade validation
    - DTE-based forced exits

    Attributes:
        config: Risk configuration parameters
        account_size: Current account size
        circuit_state: Current circuit breaker state
        portfolio_greeks: Aggregated portfolio Greeks
        positions: Dict of current positions by symbol
        last_vix: Last known VIX level
        state_history: History of state transitions

    Usage:
        risk_mgr = OptionsRiskManager(config, account_size=10000.0)

        # Before each trade
        result = risk_mgr.validate_new_position(...)
        if result.passed:
            execute_trade()
            risk_mgr.add_position(position)

        # Periodic checks
        exits = risk_mgr.check_circuit_breakers(current_vix)
        for exit_signal in exits:
            close_position(exit_signal.position)
    """

    def __init__(
        self,
        config: Optional[OptionsRiskConfig] = None,
        account_size: float = 10000.0
    ):
        """
        Initialize options risk manager.

        Args:
            config: Risk configuration (uses defaults if None)
            account_size: Current account equity

        Raises:
            ValueError: If account_size <= 0
        """
        if account_size <= 0:
            raise ValueError(f"account_size must be positive, got {account_size}")

        self.config = config or OptionsRiskConfig()
        self.account_size = account_size
        self.circuit_state = CircuitBreakerState.NORMAL
        self.portfolio_greeks = PortfolioGreeks()
        self.positions: Dict[str, PositionRisk] = {}
        self.last_vix: Optional[float] = None
        self.state_history: List[Tuple[datetime, CircuitBreakerState, str]] = []

        # Log initial state
        self._log_state_change(CircuitBreakerState.NORMAL, "Initialized")

    def _log_state_change(self, new_state: CircuitBreakerState, reason: str):
        """Record circuit breaker state change."""
        self.state_history.append((datetime.now(), new_state, reason))
        logger.info(f"Circuit breaker state: {new_state.value} - {reason}")

    def update_account_size(self, new_size: float):
        """
        Update account size (call after deposits/withdrawals or P/L changes).

        Args:
            new_size: New account equity

        Raises:
            ValueError: If new_size <= 0
        """
        if new_size <= 0:
            raise ValueError(f"account_size must be positive, got {new_size}")
        self.account_size = new_size

    def validate_new_position(
        self,
        delta: float,
        gamma: float,
        theta: float,
        vega: float,
        premium: float,
        dte: int,
        spread_pct: float,
        symbol: Optional[str] = None,
        contracts: int = 1
    ) -> ValidationResult:
        """
        Validate a proposed new position against risk limits.

        This is the pre-trade validation hook per ATLAS Section 9.3.

        Args:
            delta: Position delta (e.g., 0.55 for 55% delta)
            gamma: Position gamma
            theta: Position theta (daily, negative for long options)
            vega: Position vega
            premium: Total premium for position
            dte: Days to expiration
            spread_pct: Bid-ask spread as % of mid price
            symbol: Underlying symbol (for per-symbol limits)
            contracts: Number of contracts

        Returns:
            ValidationResult with pass/fail status and reasons
        """
        reasons = []
        warnings = []

        # Check circuit breaker state first
        if self.circuit_state == CircuitBreakerState.HALTED:
            return ValidationResult(
                passed=False,
                reasons=["Trading HALTED - no new positions allowed"],
                circuit_state=self.circuit_state
            )

        if self.circuit_state == CircuitBreakerState.EMERGENCY:
            return ValidationResult(
                passed=False,
                reasons=["EMERGENCY state - exiting all positions"],
                circuit_state=self.circuit_state
            )

        # DTE validation
        if dte < self.config.min_dte_entry:
            reasons.append(
                f"DTE {dte} below minimum {self.config.min_dte_entry}"
            )

        if dte > self.config.max_dte_entry:
            reasons.append(
                f"DTE {dte} above maximum {self.config.max_dte_entry}"
            )

        # Spread width validation
        if spread_pct > self.config.max_spread_pct:
            reasons.append(
                f"Spread {spread_pct:.1%} exceeds max {self.config.max_spread_pct:.1%}"
            )

        # Position delta validation
        position_delta_pct = abs(delta * contracts) / self.account_size
        if position_delta_pct > self.config.max_position_delta:
            reasons.append(
                f"Position delta {position_delta_pct:.1%} exceeds max {self.config.max_position_delta:.1%}"
            )

        # Premium at risk validation
        premium_pct = premium / self.account_size
        if premium_pct > self.config.max_premium_at_risk:
            reasons.append(
                f"Premium {premium_pct:.1%} exceeds max {self.config.max_premium_at_risk:.1%}"
            )

        # Per-symbol contract limit
        if symbol and symbol in self.positions:
            existing = self.positions[symbol]
            total_contracts = existing.contracts + contracts
            if total_contracts > self.config.max_contracts_per_symbol:
                reasons.append(
                    f"Total contracts {total_contracts} would exceed "
                    f"max {self.config.max_contracts_per_symbol} for {symbol}"
                )

        # Portfolio Greeks validation (after this trade)
        new_portfolio_delta = self.portfolio_greeks.net_delta + (delta * contracts)
        new_portfolio_gamma = self.portfolio_greeks.total_gamma + (gamma * contracts)
        new_portfolio_theta = self.portfolio_greeks.total_theta + (theta * contracts)
        new_portfolio_vega = self.portfolio_greeks.total_vega + (vega * contracts)

        # Normalize by account size for percentage checks
        delta_pct = abs(new_portfolio_delta) / self.account_size
        gamma_pct = abs(new_portfolio_gamma) / self.account_size
        theta_pct = new_portfolio_theta / self.account_size  # Keep sign
        vega_pct = abs(new_portfolio_vega) / self.account_size

        # Check portfolio limits
        if delta_pct > self.config.max_portfolio_delta:
            reasons.append(
                f"Portfolio delta would be {delta_pct:.1%}, "
                f"exceeds max {self.config.max_portfolio_delta:.1%}"
            )

        if gamma_pct > self.config.max_portfolio_gamma:
            reasons.append(
                f"Portfolio gamma would be {gamma_pct:.1%}, "
                f"exceeds max {self.config.max_portfolio_gamma:.1%}"
            )

        # Theta check (negative theta = losing money)
        if theta_pct < self.config.max_portfolio_theta:
            reasons.append(
                f"Portfolio theta would be {theta_pct:.1%} daily, "
                f"exceeds max bleed {self.config.max_portfolio_theta:.1%}"
            )

        if vega_pct > self.config.max_portfolio_vega:
            reasons.append(
                f"Portfolio vega would be {vega_pct:.1%}, "
                f"exceeds max {self.config.max_portfolio_vega:.1%}"
            )

        # Warnings for approaching limits
        if delta_pct > self.config.max_portfolio_delta * self.config.caution_threshold:
            warnings.append(
                f"Approaching delta limit ({delta_pct:.1%} / {self.config.max_portfolio_delta:.1%})"
            )

        # REDUCED state: suggest smaller position
        adjusted_size = None
        if self.circuit_state == CircuitBreakerState.REDUCED:
            adjusted_size = 0.5  # 50% of requested size
            warnings.append("REDUCED state: position size halved")

        passed = len(reasons) == 0

        return ValidationResult(
            passed=passed,
            reasons=reasons,
            warnings=warnings,
            adjusted_size=adjusted_size,
            circuit_state=self.circuit_state
        )

    def add_position(self, position: PositionRisk):
        """
        Add a position to the portfolio and update aggregated Greeks.

        Args:
            position: Position to add
        """
        self.positions[position.option_symbol] = position
        self._recalculate_portfolio_greeks()

    def remove_position(self, option_symbol: str):
        """
        Remove a position from the portfolio.

        Args:
            option_symbol: OSI symbol of position to remove
        """
        if option_symbol in self.positions:
            del self.positions[option_symbol]
            self._recalculate_portfolio_greeks()

    def update_position(self, option_symbol: str, current_value: float, dte: int):
        """
        Update position with current market data.

        Args:
            option_symbol: OSI symbol
            current_value: Current position value
            dte: Current days to expiration
        """
        if option_symbol in self.positions:
            pos = self.positions[option_symbol]
            pos.current_value = current_value
            pos.dte = dte
            # Recalculate P/L
            if pos.premium > 0:
                pos.pnl_pct = (current_value - pos.premium) / pos.premium

    def _recalculate_portfolio_greeks(self):
        """Recalculate aggregated portfolio Greeks from all positions."""
        self.portfolio_greeks = PortfolioGreeks(
            net_delta=sum(p.delta * p.contracts for p in self.positions.values()),
            total_gamma=sum(p.gamma * p.contracts for p in self.positions.values()),
            total_theta=sum(p.theta * p.contracts for p in self.positions.values()),
            total_vega=sum(p.vega * p.contracts for p in self.positions.values()),
            position_count=len(self.positions),
            total_premium_at_risk=sum(p.premium for p in self.positions.values())
        )

    def aggregate_portfolio_greeks(self) -> PortfolioGreeks:
        """
        Get current aggregated portfolio Greeks.

        Returns:
            PortfolioGreeks with current totals
        """
        self._recalculate_portfolio_greeks()
        return self.portfolio_greeks

    def check_circuit_breakers(
        self,
        current_vix: Optional[float] = None
    ) -> List[ForceExitSignal]:
        """
        Check circuit breaker conditions and return any force exit signals.

        Per ATLAS Section 9.3.2 - Options Circuit Breakers.

        Args:
            current_vix: Current VIX level (optional)

        Returns:
            List of ForceExitSignal for positions that need closing
        """
        force_exits = []
        reasons = []

        # VIX-based checks
        if current_vix is not None:
            if current_vix >= self.config.vix_halt_threshold:
                reasons.append(f"VIX {current_vix:.1f} >= halt threshold {self.config.vix_halt_threshold}")

            if self.last_vix is not None:
                vix_change = (current_vix - self.last_vix) / self.last_vix
                if vix_change >= self.config.vix_spike_pct:
                    reasons.append(f"VIX spike {vix_change:.1%} >= {self.config.vix_spike_pct:.1%}")

            self.last_vix = current_vix

        # Portfolio Greeks checks
        greeks = self.aggregate_portfolio_greeks()

        delta_pct = abs(greeks.net_delta) / self.account_size if self.account_size > 0 else 0
        gamma_pct = abs(greeks.total_gamma) / self.account_size if self.account_size > 0 else 0
        theta_pct = greeks.total_theta / self.account_size if self.account_size > 0 else 0
        vega_pct = abs(greeks.total_vega) / self.account_size if self.account_size > 0 else 0

        # Determine new state based on how far over limits we are
        max_breach = 0.0

        if delta_pct > 0:
            delta_ratio = delta_pct / self.config.max_portfolio_delta
            max_breach = max(max_breach, delta_ratio)

        if gamma_pct > 0:
            gamma_ratio = gamma_pct / self.config.max_portfolio_gamma
            max_breach = max(max_breach, gamma_ratio)

        # Theta is negative, so compare magnitudes
        if theta_pct < 0 and self.config.max_portfolio_theta < 0:
            theta_ratio = theta_pct / self.config.max_portfolio_theta
            max_breach = max(max_breach, theta_ratio)

        if vega_pct > 0:
            vega_ratio = vega_pct / self.config.max_portfolio_vega
            max_breach = max(max_breach, vega_ratio)

        # Determine state from breach level
        old_state = self.circuit_state
        new_state = CircuitBreakerState.NORMAL

        if max_breach >= self.config.halted_threshold or len(reasons) > 0:
            new_state = CircuitBreakerState.HALTED
        elif max_breach >= self.config.reduced_threshold:
            new_state = CircuitBreakerState.REDUCED
        elif max_breach >= self.config.caution_threshold:
            new_state = CircuitBreakerState.CAUTION

        # EMERGENCY state can only be cleared manually via reset_to_normal()
        # Don't allow automatic downgrade from EMERGENCY
        if old_state == CircuitBreakerState.EMERGENCY:
            new_state = CircuitBreakerState.EMERGENCY

        # Log state change (only if actually changing)
        if new_state != old_state:
            reason = f"Greeks breach {max_breach:.1%}" if max_breach > 0 else "; ".join(reasons)
            self._log_state_change(new_state, reason)
            self.circuit_state = new_state

        # Check individual positions for force exits
        for pos in self.positions.values():
            # DTE-based forced exit
            if pos.dte <= self.config.forced_exit_dte:
                force_exits.append(ForceExitSignal(
                    position=pos,
                    reason=f"DTE {pos.dte} <= forced exit threshold {self.config.forced_exit_dte}",
                    urgency=2
                ))

            # Loss-based forced exit
            if pos.pnl_pct <= -self.config.position_loss_exit_pct:
                force_exits.append(ForceExitSignal(
                    position=pos,
                    reason=f"Position loss {pos.pnl_pct:.1%} >= {self.config.position_loss_exit_pct:.1%}",
                    urgency=2
                ))

        # EMERGENCY state: exit all positions (whether newly triggered or already set)
        if self.circuit_state == CircuitBreakerState.EMERGENCY:
            for pos in self.positions.values():
                if not any(fe.position.option_symbol == pos.option_symbol for fe in force_exits):
                    force_exits.append(ForceExitSignal(
                        position=pos,
                        reason="EMERGENCY: Exit all positions",
                        urgency=3
                    ))

        return force_exits

    def get_portfolio_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio risk summary.

        Returns:
            Dictionary with all risk metrics and status
        """
        greeks = self.aggregate_portfolio_greeks()

        # Calculate percentages
        delta_pct = abs(greeks.net_delta) / self.account_size if self.account_size > 0 else 0
        gamma_pct = abs(greeks.total_gamma) / self.account_size if self.account_size > 0 else 0
        theta_pct = greeks.total_theta / self.account_size if self.account_size > 0 else 0
        vega_pct = abs(greeks.total_vega) / self.account_size if self.account_size > 0 else 0
        premium_pct = greeks.total_premium_at_risk / self.account_size if self.account_size > 0 else 0

        return {
            'circuit_state': self.circuit_state.value,
            'account_size': self.account_size,
            'position_count': greeks.position_count,
            'greeks': greeks.to_dict(),
            'utilization': {
                'delta': {
                    'value': delta_pct,
                    'limit': self.config.max_portfolio_delta,
                    'pct_of_limit': delta_pct / self.config.max_portfolio_delta if self.config.max_portfolio_delta > 0 else 0,
                },
                'gamma': {
                    'value': gamma_pct,
                    'limit': self.config.max_portfolio_gamma,
                    'pct_of_limit': gamma_pct / self.config.max_portfolio_gamma if self.config.max_portfolio_gamma > 0 else 0,
                },
                'theta': {
                    'value': theta_pct,
                    'limit': self.config.max_portfolio_theta,
                    'pct_of_limit': theta_pct / self.config.max_portfolio_theta if self.config.max_portfolio_theta != 0 else 0,
                },
                'vega': {
                    'value': vega_pct,
                    'limit': self.config.max_portfolio_vega,
                    'pct_of_limit': vega_pct / self.config.max_portfolio_vega if self.config.max_portfolio_vega > 0 else 0,
                },
                'premium': {
                    'value': premium_pct,
                    'total': greeks.total_premium_at_risk,
                },
            },
            'last_vix': self.last_vix,
            'config': self.config.to_dict(),
        }

    def can_trade(self) -> Tuple[bool, str]:
        """
        Quick check if trading is currently allowed.

        Returns:
            (can_trade, reason) tuple
        """
        if self.circuit_state == CircuitBreakerState.HALTED:
            return False, "Trading HALTED"
        if self.circuit_state == CircuitBreakerState.EMERGENCY:
            return False, "EMERGENCY state"
        return True, "OK"

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on circuit state.

        Returns:
            Multiplier (1.0 = full size, 0.5 = half, 0.0 = no trading)
        """
        multipliers = {
            CircuitBreakerState.NORMAL: 1.0,
            CircuitBreakerState.CAUTION: 1.0,
            CircuitBreakerState.REDUCED: 0.5,
            CircuitBreakerState.HALTED: 0.0,
            CircuitBreakerState.EMERGENCY: 0.0,
        }
        return multipliers.get(self.circuit_state, 1.0)

    def reset_to_normal(self, reason: str = "Manual reset"):
        """
        Reset circuit breaker to NORMAL state.

        Use with caution - only after market conditions stabilize.

        Args:
            reason: Reason for reset (logged)
        """
        if self.circuit_state != CircuitBreakerState.NORMAL:
            self._log_state_change(CircuitBreakerState.NORMAL, reason)
            self.circuit_state = CircuitBreakerState.NORMAL

    def print_summary(self) -> str:
        """
        Generate human-readable risk summary.

        Returns:
            Formatted string summary
        """
        summary = self.get_portfolio_risk_summary()
        greeks = summary['greeks']
        util = summary['utilization']

        lines = [
            "=" * 60,
            "OPTIONS RISK MANAGER SUMMARY",
            "=" * 60,
            f"Circuit State:      {summary['circuit_state']}",
            f"Account Size:       ${summary['account_size']:,.2f}",
            f"Open Positions:     {summary['position_count']}",
            "",
            "--- PORTFOLIO GREEKS ---",
            f"Net Delta:          {greeks['net_delta']:.2f} ({util['delta']['pct_of_limit']:.1%} of limit)",
            f"Total Gamma:        {greeks['total_gamma']:.4f} ({util['gamma']['pct_of_limit']:.1%} of limit)",
            f"Total Theta:        ${greeks['total_theta']:.2f}/day ({util['theta']['pct_of_limit']:.1%} of limit)",
            f"Total Vega:         {greeks['total_vega']:.2f} ({util['vega']['pct_of_limit']:.1%} of limit)",
            "",
            f"Premium at Risk:    ${util['premium']['total']:,.2f} ({util['premium']['value']:.1%} of account)",
        ]

        if summary['last_vix'] is not None:
            lines.append(f"Last VIX:           {summary['last_vix']:.2f}")

        lines.append("=" * 60)
        return "\n".join(lines)

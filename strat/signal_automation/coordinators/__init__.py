# EQUITY-84: Coordinator classes extracted from SignalDaemon god class
# Phase 4 refactoring - Facade pattern with focused coordinators

"""
Coordinators package for signal automation.

Extracted components:
- AlertManager: Signal and trade alert delivery
- HealthMonitor: Metrics collection and daily audits
- FilterManager: Signal quality gates
- ExecutionCoordinator: Trade submission logic
- StaleSetupValidator: Setup freshness validation
- ExitConditionEvaluator: Position exit logic
- TrailingStopManager: Trailing stop calculations
- PartialExitManager: Multi-contract partial exits
"""

# Coordinators imported as they are extracted
from strat.signal_automation.coordinators.alert_manager import AlertManager
from strat.signal_automation.coordinators.health_monitor import HealthMonitor, DaemonStats
from strat.signal_automation.coordinators.filter_manager import FilterManager, FilterConfig
from strat.signal_automation.coordinators.execution_coordinator import ExecutionCoordinator
from strat.signal_automation.coordinators.stale_setup_validator import (
    StaleSetupValidator,
    StalenessConfig,
)
from strat.signal_automation.coordinators.exit_evaluator import (
    ExitConditionEvaluator,
    TrailingStopChecker,
    PartialExitChecker,
)
from strat.signal_automation.coordinators.trailing_stop_manager import (
    TrailingStopManager,
)
from strat.signal_automation.coordinators.partial_exit_manager import (
    PartialExitManager,
)

__all__ = [
    'AlertManager',
    'HealthMonitor',
    'DaemonStats',
    'FilterManager',
    'FilterConfig',
    'ExecutionCoordinator',
    'StaleSetupValidator',
    'StalenessConfig',
    'ExitConditionEvaluator',
    'TrailingStopChecker',
    'PartialExitChecker',
    'TrailingStopManager',
    'PartialExitManager',
]

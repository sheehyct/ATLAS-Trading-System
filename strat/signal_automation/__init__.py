"""
STRAT Signal Automation - Session 83K-45/49

Automated signal detection, storage, alerting, execution, and position monitoring
system for autonomous paper trading deployment to Alpaca.

Phase 1 (Signal Detection) - COMPLETE:
- Automated scanning on schedule (market hours)
- Signal deduplication (no duplicate alerts)
- Discord webhook alerts + structured logging
- VPS-ready daemon architecture

Phase 2 (Options Execution) - COMPLETE:
- Options order submission to Alpaca paper account
- Delta-targeted strike selection (0.40-0.55)
- DTE optimization (7-21 day range)
- Executor wired to daemon (Session 83K-48)

Phase 3 (Position Monitoring) - IN PROGRESS:
- P&L tracking for open positions
- Target/stop monitoring with auto-exit
- DTE-based exit management
- Exit alerting via Discord

Components:
- config.py: Configuration dataclasses (ScanConfig, ScheduleConfig, AlertConfig, ExecutionConfig, MonitoringConfig)
- signal_store.py: Signal persistence with deduplication
- scheduler.py: APScheduler setup and job definitions
- alerters/: Alert delivery mechanisms (Discord, logging, email)
- daemon.py: Main daemon entry point
- executor.py: Signal-to-order execution (Session 83K-47)
- position_monitor.py: Position monitoring and auto-exit (Session 83K-49)
"""

from strat.signal_automation.config import (
    ScanConfig,
    ScheduleConfig,
    AlertConfig,
    ExecutionConfig,
    MonitoringConfig,
    SignalAutomationConfig,
)
from strat.signal_automation.signal_store import (
    SignalStatus,
    StoredSignal,
    SignalStore,
)
from strat.signal_automation.scheduler import SignalScheduler
from strat.signal_automation.daemon import SignalDaemon
from strat.signal_automation.executor import (
    ExecutionState,
    ExecutionResult,
    ExecutorConfig,
    SignalExecutor,
    create_paper_executor,
)
from strat.signal_automation.position_monitor import (
    ExitReason,
    MonitoringConfig as MonitoringConfigClass,
    TrackedPosition,
    ExitSignal,
    PositionMonitor,
    create_position_monitor,
)

__all__ = [
    # Config
    'ScanConfig',
    'ScheduleConfig',
    'AlertConfig',
    'ExecutionConfig',
    'MonitoringConfig',
    'SignalAutomationConfig',
    # Signal Store
    'SignalStatus',
    'StoredSignal',
    'SignalStore',
    # Scheduler
    'SignalScheduler',
    # Daemon
    'SignalDaemon',
    # Executor (Session 83K-47)
    'ExecutionState',
    'ExecutionResult',
    'ExecutorConfig',
    'SignalExecutor',
    'create_paper_executor',
    # Position Monitor (Session 83K-49)
    'ExitReason',
    'TrackedPosition',
    'ExitSignal',
    'PositionMonitor',
    'create_position_monitor',
]

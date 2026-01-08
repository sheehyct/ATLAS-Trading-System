"""
STRAT Signal Automation Configuration - Session 83K-45/48

Configuration dataclasses for the signal automation system.
Follows established patterns from paper_trading.py.

Configuration Categories:
1. ScanConfig - Pattern scanning parameters
2. ScheduleConfig - APScheduler timing configuration
3. AlertConfig - Alert delivery settings
4. ExecutionConfig - Signal-to-order execution settings (Session 83K-48)
5. SignalAutomationConfig - Master configuration
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum
import os


class ScanInterval(str, Enum):
    """Scan interval presets matching trading timeframes."""
    FIFTEEN_MIN = '15m'    # For 15m timeframe patterns (Session EQUITY-18)
    THIRTY_MIN = '30m'     # For 30m timeframe patterns (Session EQUITY-18)
    HOURLY = 'hourly'      # For 1H timeframe patterns
    DAILY = 'daily'        # For 1D timeframe patterns
    WEEKLY = 'weekly'      # For 1W timeframe patterns
    MONTHLY = 'monthly'    # For 1M timeframe patterns


class AlertChannel(str, Enum):
    """Available alert channels."""
    DISCORD = 'discord'
    EMAIL = 'email'
    LOGGING = 'logging'


@dataclass
class ScanConfig:
    """
    Configuration for pattern scanning.

    Attributes:
        symbols: List of symbols to scan (default: SPY, QQQ, IWM, DIA, AAPL)
        timeframes: List of timeframes to scan (default: all)
        patterns: List of patterns to detect (default: all STRAT patterns)
        lookback_bars: Number of bars to fetch for pattern detection
        signal_age_bars: Only report signals from last N bars (freshness filter)
        min_magnitude_pct: Minimum magnitude to report (filter noise)
        min_risk_reward: Minimum R:R ratio to report
    """
    symbols: List[str] = field(default_factory=lambda: [
        # Core ETFs
        'SPY', 'QQQ', 'IWM', 'DIA',
        # Mega-caps (high volume, liquid options)
        'AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
        # High-beta tech (good for STRAT momentum plays)
        'AMD', 'NFLX', 'MU', 'PLTR',
        # Crypto-correlated / retail momentum (high ATR, good premium)
        'COIN', 'MSTR', 'HOOD', 'ACHR', 'QBTS'
    ])

    timeframes: List[str] = field(default_factory=lambda: [
        '1H', '1D', '1W', '1M'
    ])

    patterns: List[str] = field(default_factory=lambda: [
        '2-2', '3-2', '3-2-2', '2-1-2', '3-1-2'
    ])

    # Detection parameters
    lookback_bars: int = 50          # Bars to fetch for pattern detection
    signal_age_bars: int = 3         # Only report signals from last N bars

    # Quality filters
    min_magnitude_pct: float = 0.5   # Minimum expected move (backtest: <0.5% loses money)
    min_risk_reward: float = 1.0     # Minimum risk/reward ratio


@dataclass
class ScheduleConfig:
    """
    APScheduler timing configuration.

    All times in America/New_York timezone (market hours).
    Uses cron expressions for precise scheduling.

    Attributes:
        fifteen_min_cron: Cron for 15m scans (every 15 min during market hours)
        thirty_min_cron: Cron for 30m scans (every 30 min during market hours)
        hourly_cron: Cron for hourly scans (9:30-15:30 ET Mon-Fri)
        daily_cron: Cron for daily scans (after market close)
        weekly_cron: Cron for weekly scans (Friday after close)
        monthly_cron: Cron for monthly scans (last trading day)
        scan_15m: Enable/disable 15m timeframe scanning
        scan_30m: Enable/disable 30m timeframe scanning
        scan_hourly: Enable/disable hourly timeframe scanning
        scan_daily: Enable/disable daily timeframe scanning
        scan_weekly: Enable/disable weekly timeframe scanning
        scan_monthly: Enable/disable monthly timeframe scanning
        timezone: Timezone for all schedules
    """
    # =========================================================================
    # Session EQUITY-18: 15m and 30m Timeframe Scanning
    # =========================================================================
    # 15m: Scan at :00, :15, :30, :45 during market hours
    # Note: :00,:15,:30,:45 at hours 9-15 gives us scans at market-aligned times
    fifteen_min_cron: str = '0,15,30,45 9-15 * * mon-fri'

    # 30m: Scan at :00, :30 during market hours
    thirty_min_cron: str = '0,30 9-15 * * mon-fri'

    # =========================================================================
    # Cron expressions (minute hour day month day_of_week)
    # NOTE: APScheduler uses 0=Mon, 1=Tue, ..., 6=Sun. Use 'mon-fri' for clarity.
    # Hourly: Run at :30 each hour from 9:30-15:30 ET, Mon-Fri
    hourly_cron: str = '30 9-15 * * mon-fri'

    # Daily: Run at 5 PM ET after market close, Mon-Fri
    daily_cron: str = '0 17 * * mon-fri'

    # Weekly: Run at 6 PM ET on Friday
    weekly_cron: str = '0 18 * * fri'

    # Monthly: Run at 6 PM ET on 28th of month (APScheduler doesn't support 'L')
    monthly_cron: str = '0 18 28 * *'

    # =========================================================================
    # Session 83K-80: 15-Minute Base Resampling (HTF Scanning Architecture Fix)
    # =========================================================================
    # When enable_htf_resampling=True, uses a single 15-minute scan that
    # resamples to all higher timeframes (1H, 1D, 1W, 1M).
    # This fixes the bug where HTF scans missed entry opportunities.

    # Enable unified multi-TF resampling (replaces separate HTF scan jobs)
    enable_htf_resampling: bool = True

    # Base timeframe for resampling
    base_timeframe: str = '15min'

    # 15-minute scan schedule (market hours, market-aligned)
    # Run at :30, :45, :00, :15 (after each 15-min bar closes)
    # This detects setups within 15 minutes of inside bar close
    base_scan_cron: str = '30,45,0,15 9-15 * * mon-fri'

    # =========================================================================
    # Legacy: Individual timeframe scans (used when enable_htf_resampling=False)
    # Session EQUITY-18: Added scan_15m and scan_30m flags
    # =========================================================================
    # Enable/disable individual timeframe scans
    # Session EQUITY-34: Disabled - not in ScanConfig.timeframes, not used for STRAT resampling
    scan_15m: bool = False     # 15-minute timeframe scanning (disabled)
    scan_30m: bool = False     # 30-minute timeframe scanning (disabled)
    scan_hourly: bool = True
    scan_daily: bool = True
    scan_weekly: bool = True
    scan_monthly: bool = True

    # Timezone for all schedules
    timezone: str = 'America/New_York'

    # Grace period for missed jobs (in seconds)
    misfire_grace_time: int = 300  # 5 minutes


@dataclass
class AlertConfig:
    """
    Alert delivery configuration.

    Supports multiple channels with throttling to prevent spam.

    Session EQUITY-34: Added explicit alert type flags matching crypto daemon pattern.

    Attributes:
        discord_webhook_url: Discord webhook URL (from env if not set)
        discord_enabled: Enable Discord alerts
        alert_on_signal_detection: Send alerts for pattern detection (noisy, disabled)
        alert_on_trigger: Send alerts when SETUP price hit (disabled)
        alert_on_trade_entry: Send alerts when trade executes (enabled)
        alert_on_trade_exit: Send alerts when trade closes (enabled)
        email_enabled: Enable email alerts (future)
        email_smtp_server: SMTP server for email
        email_from: From address for email
        email_to: List of recipient addresses
        logging_enabled: Enable structured logging (always recommended)
        log_file: Path to signal log file
        min_alert_interval_seconds: Throttle duplicate alerts
    """
    # Discord configuration
    discord_webhook_url: Optional[str] = None
    discord_enabled: bool = True

    # Discord alert types (Session EQUITY-34: Explicit control matching crypto daemon)
    alert_on_signal_detection: bool = False  # Pattern detection (noisy)
    alert_on_trigger: bool = False           # When SETUP price hit
    alert_on_trade_entry: bool = True        # When trade executes
    alert_on_trade_exit: bool = True         # When trade closes with P&L

    # Email configuration (future)
    email_enabled: bool = False
    email_smtp_server: str = ''
    email_from: str = ''
    email_to: List[str] = field(default_factory=list)

    # Logging configuration (always active)
    logging_enabled: bool = True
    log_file: str = 'logs/signals.log'
    log_level: str = 'INFO'

    # Alert throttling
    min_alert_interval_seconds: int = 60  # Don't re-alert same signal within 60s

    def __post_init__(self):
        """Load Discord webhook from environment if not provided."""
        if self.discord_webhook_url is None:
            # Prefer equity-specific webhook, fall back to generic
            self.discord_webhook_url = (
                os.environ.get('DISCORD_EQUITY_WEBHOOK_URL') or
                os.environ.get('DISCORD_WEBHOOK_URL', '')
            )

        # Disable Discord if no URL configured
        if not self.discord_webhook_url:
            self.discord_enabled = False


@dataclass
class ExecutionConfig:
    """
    Options execution configuration - Session 83K-48.

    Controls signal-to-order execution via Alpaca.

    Attributes:
        enabled: Master switch for execution
        account: Alpaca account to use ('SMALL' = $3k paper)
        max_capital_per_trade: Maximum capital per trade
        max_concurrent_positions: Maximum open positions
        target_delta: Target delta for strike selection
        delta_range_min: Minimum delta threshold
        delta_range_max: Maximum delta threshold
        min_dte: Minimum days to expiration
        max_dte: Maximum days to expiration
        target_dte: Target days to expiration
        use_limit_orders: Use limit orders vs market orders
        limit_price_buffer: Buffer above ask for limit orders
    """
    # Master switch
    enabled: bool = False  # Off by default for safety

    # Account settings
    account: str = 'SMALL'              # Alpaca account to use
    max_capital_per_trade: float = 300.0  # Max $ per trade
    max_concurrent_positions: int = 5   # Max open positions

    # Delta targeting (Session 83K-64: User-approved middle ground)
    target_delta: float = 0.55          # Target delta for strikes
    delta_range_min: float = 0.45       # Minimum delta
    delta_range_max: float = 0.65       # Maximum delta

    # DTE targeting (7-21 day range per spec)
    min_dte: int = 7                    # Minimum DTE
    max_dte: int = 21                   # Maximum DTE
    target_dte: int = 14                # Target DTE

    # Order settings
    use_limit_orders: bool = True       # Use limit vs market orders
    limit_price_buffer: float = 0.02    # Buffer above ask for limits

    # Session EQUITY-49: TFC Re-evaluation at Entry
    # TFC can change between pattern detection and entry trigger (hours/days later).
    # Re-evaluate TFC at entry time and optionally block if alignment degraded.
    tfc_reeval_enabled: bool = True           # Enable TFC re-evaluation at entry
    tfc_reeval_min_strength: int = 2          # Block entry if TFC strength drops below this
    tfc_reeval_block_on_flip: bool = True     # Block entry if TFC direction flipped
    tfc_reeval_log_always: bool = True        # Log TFC comparison even when not blocking


@dataclass
class MonitoringConfig:
    """
    Position monitoring configuration - Session 83K-49.

    Controls exit condition detection and auto-closing.

    Attributes:
        enabled: Master switch for monitoring
        check_interval: Seconds between position checks
        minimum_hold_seconds: Minimum time to hold before checking exits (Session 83K-77)
        exit_dte: Close positions at or below this DTE
        max_loss_pct: Close if loss exceeds this % (0.50 = 50%)
        max_profit_pct: Close if profit exceeds this % (1.00 = 100%)
        alert_on_exit: Send alerts when positions exit
    """
    # Master switch
    enabled: bool = True  # On by default when execution enabled

    # Monitoring intervals
    check_interval: int = 60            # Check positions every N seconds
    minimum_hold_seconds: int = 300     # 5 min before exit checks (Session 83K-77)

    # Exit thresholds
    exit_dte: int = 3                   # Close at or below this DTE
    max_loss_pct: float = 0.50          # Max loss as % of premium (50%)
    max_profit_pct: float = 1.00        # Take profit at 100% gain

    # Alerting
    alert_on_exit: bool = True          # Send alerts for exits


@dataclass
class ApiConfig:
    """
    REST API server configuration - Session EQUITY-33.

    Controls the optional HTTP API for remote dashboard access.

    Attributes:
        enabled: Enable API server
        host: Host to bind to
        port: Port to bind to
    """
    enabled: bool = False  # Off by default
    host: str = '0.0.0.0'  # Listen on all interfaces
    port: int = 8081       # Default port (8080 used by crypto)


@dataclass
class SignalAutomationConfig:
    """
    Master configuration for the signal automation system.

    Combines all configuration categories into a single config object.
    Can be loaded from environment, file, or defaults.

    Attributes:
        scan: Pattern scanning configuration
        schedule: APScheduler timing configuration
        alerts: Alert delivery configuration
        execution: Options execution configuration (Session 83K-48)
        monitoring: Position monitoring configuration (Session 83K-49)
        api: REST API server configuration (Session EQUITY-33)
        store_path: Path to signal store directory
        graceful_shutdown_timeout: Seconds to wait for graceful shutdown
        health_check_interval: Seconds between health checks
    """
    scan: ScanConfig = field(default_factory=ScanConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    # Signal store location
    store_path: str = 'data/signals'

    # Daemon settings
    graceful_shutdown_timeout: int = 30  # Seconds to wait for shutdown
    health_check_interval: int = 300     # 5 minutes between health checks

    @classmethod
    def from_env(cls) -> 'SignalAutomationConfig':
        """
        Create configuration from environment variables.

        Environment variables:
            SIGNAL_SYMBOLS: Comma-separated list of symbols
            SIGNAL_TIMEFRAMES: Comma-separated list of timeframes
            SIGNAL_SCAN_HOURLY: Enable hourly scans (true/false)
            SIGNAL_SCAN_DAILY: Enable daily scans (true/false)
            SIGNAL_SCAN_WEEKLY: Enable weekly scans (true/false)
            SIGNAL_SCAN_MONTHLY: Enable monthly scans (true/false)
            DISCORD_WEBHOOK_URL: Discord webhook for alerts
            SIGNAL_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
            SIGNAL_STORE_PATH: Path to signal store directory
            SIGNAL_EXECUTION_ENABLED: Enable order execution (true/false)
            SIGNAL_EXECUTION_ACCOUNT: Alpaca account (SMALL, MEDIUM, LARGE)
            SIGNAL_MAX_CAPITAL_PER_TRADE: Maximum capital per trade
            SIGNAL_MAX_CONCURRENT_POSITIONS: Maximum open positions
        """
        scan_config = ScanConfig()
        schedule_config = ScheduleConfig()
        alert_config = AlertConfig()
        execution_config = ExecutionConfig()

        # Override from environment
        if symbols := os.environ.get('SIGNAL_SYMBOLS'):
            scan_config.symbols = [s.strip() for s in symbols.split(',')]

        if timeframes := os.environ.get('SIGNAL_TIMEFRAMES'):
            scan_config.timeframes = [t.strip() for t in timeframes.split(',')]

        # Schedule enables/disables
        schedule_config.scan_hourly = os.environ.get(
            'SIGNAL_SCAN_HOURLY', 'true'
        ).lower() == 'true'
        schedule_config.scan_daily = os.environ.get(
            'SIGNAL_SCAN_DAILY', 'true'
        ).lower() == 'true'
        schedule_config.scan_weekly = os.environ.get(
            'SIGNAL_SCAN_WEEKLY', 'true'
        ).lower() == 'true'
        schedule_config.scan_monthly = os.environ.get(
            'SIGNAL_SCAN_MONTHLY', 'true'
        ).lower() == 'true'

        # Alert configuration - prefer equity-specific webhook for signal daemon
        alert_config.discord_webhook_url = (
            os.environ.get('DISCORD_EQUITY_WEBHOOK_URL') or
            os.environ.get('DISCORD_WEBHOOK_URL', '')
        )
        alert_config.discord_enabled = bool(alert_config.discord_webhook_url)
        alert_config.log_level = os.environ.get('SIGNAL_LOG_LEVEL', 'INFO')

        # Execution configuration (Session 83K-48)
        execution_config.enabled = os.environ.get(
            'SIGNAL_EXECUTION_ENABLED', 'false'
        ).lower() == 'true'
        execution_config.account = os.environ.get(
            'SIGNAL_EXECUTION_ACCOUNT', 'SMALL'
        )
        if max_capital := os.environ.get('SIGNAL_MAX_CAPITAL_PER_TRADE'):
            execution_config.max_capital_per_trade = float(max_capital)
        if max_positions := os.environ.get('SIGNAL_MAX_CONCURRENT_POSITIONS'):
            execution_config.max_concurrent_positions = int(max_positions)

        # Monitoring configuration (Session 83K-49)
        monitoring_config = MonitoringConfig()
        monitoring_config.enabled = os.environ.get(
            'SIGNAL_MONITORING_ENABLED', 'true'
        ).lower() == 'true'
        if check_interval := os.environ.get('SIGNAL_MONITOR_INTERVAL'):
            monitoring_config.check_interval = int(check_interval)
        if min_hold := os.environ.get('SIGNAL_MIN_HOLD_SECONDS'):
            monitoring_config.minimum_hold_seconds = int(min_hold)
        if exit_dte := os.environ.get('SIGNAL_EXIT_DTE'):
            monitoring_config.exit_dte = int(exit_dte)
        if max_loss := os.environ.get('SIGNAL_MAX_LOSS_PCT'):
            monitoring_config.max_loss_pct = float(max_loss)
        if max_profit := os.environ.get('SIGNAL_MAX_PROFIT_PCT'):
            monitoring_config.max_profit_pct = float(max_profit)

        # Store path
        store_path = os.environ.get('SIGNAL_STORE_PATH', 'data/signals')

        # API configuration (Session EQUITY-33)
        api_config = ApiConfig()
        api_config.enabled = os.environ.get(
            'SIGNAL_API_ENABLED', 'false'
        ).lower() == 'true'
        if api_host := os.environ.get('SIGNAL_API_HOST'):
            api_config.host = api_host
        if api_port := os.environ.get('SIGNAL_API_PORT'):
            api_config.port = int(api_port)

        return cls(
            scan=scan_config,
            schedule=schedule_config,
            alerts=alert_config,
            execution=execution_config,
            monitoring=monitoring_config,
            api=api_config,
            store_path=store_path,
        )

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid)
        """
        issues = []

        # Check symbols
        if not self.scan.symbols:
            issues.append('No symbols configured for scanning')

        # Check timeframes
        # Session EQUITY-18: Added 15m and 30m as valid timeframes
        valid_timeframes = {'15m', '30m', '1H', '1D', '1W', '1M'}
        for tf in self.scan.timeframes:
            if tf not in valid_timeframes:
                issues.append(f'Invalid timeframe: {tf}')

        # Check alert configuration
        if not self.alerts.discord_enabled and not self.alerts.logging_enabled:
            issues.append('No alert channels enabled (both Discord and logging disabled)')

        # Check Discord webhook format
        if self.alerts.discord_enabled:
            if not self.alerts.discord_webhook_url:
                issues.append('Discord enabled but no webhook URL configured')
            elif not self.alerts.discord_webhook_url.startswith('https://discord.com/api/webhooks/'):
                issues.append('Discord webhook URL appears invalid')

        # Check execution configuration (Session 83K-48)
        if self.execution.enabled:
            if self.execution.max_capital_per_trade <= 0:
                issues.append('Execution max_capital_per_trade must be positive')
            if self.execution.max_concurrent_positions <= 0:
                issues.append('Execution max_concurrent_positions must be positive')
            if self.execution.min_dte >= self.execution.max_dte:
                issues.append('Execution min_dte must be less than max_dte')
            if not (0 < self.execution.delta_range_min < self.execution.delta_range_max < 1):
                issues.append('Execution delta range must be between 0 and 1')

        # Check monitoring configuration (Session 83K-49)
        if self.monitoring.enabled:
            if self.monitoring.check_interval < 10:
                issues.append('Monitoring check_interval must be at least 10 seconds')
            if self.monitoring.exit_dte < 0:
                issues.append('Monitoring exit_dte must be non-negative')
            if not (0 < self.monitoring.max_loss_pct <= 1.0):
                issues.append('Monitoring max_loss_pct must be between 0 and 1')
            if not (0 < self.monitoring.max_profit_pct <= 10.0):
                issues.append('Monitoring max_profit_pct must be between 0 and 10')

        return issues

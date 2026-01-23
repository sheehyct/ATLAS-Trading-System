"""
Tests for strat/signal_automation/config.py

Comprehensive tests for configuration dataclasses used by the signal automation system.

Session: EQUITY-80
"""

import os
import pytest
from unittest.mock import patch

from strat.signal_automation.config import (
    ScanInterval,
    AlertChannel,
    ScanConfig,
    ScheduleConfig,
    AlertConfig,
    ExecutionConfig,
    MonitoringConfig,
    ApiConfig,
    SignalAutomationConfig,
)


# =============================================================================
# ScanInterval Enum Tests
# =============================================================================

class TestScanInterval:
    """Tests for ScanInterval enum."""

    def test_fifteen_min_value(self):
        """Test 15m interval value."""
        assert ScanInterval.FIFTEEN_MIN.value == '15m'

    def test_thirty_min_value(self):
        """Test 30m interval value."""
        assert ScanInterval.THIRTY_MIN.value == '30m'

    def test_hourly_value(self):
        """Test hourly interval value."""
        assert ScanInterval.HOURLY.value == 'hourly'

    def test_daily_value(self):
        """Test daily interval value."""
        assert ScanInterval.DAILY.value == 'daily'

    def test_weekly_value(self):
        """Test weekly interval value."""
        assert ScanInterval.WEEKLY.value == 'weekly'

    def test_monthly_value(self):
        """Test monthly interval value."""
        assert ScanInterval.MONTHLY.value == 'monthly'

    def test_all_intervals_are_strings(self):
        """Test all intervals inherit from str."""
        for interval in ScanInterval:
            assert isinstance(interval, str)
            assert isinstance(interval.value, str)

    def test_enum_count(self):
        """Test correct number of intervals."""
        assert len(ScanInterval) == 6


# =============================================================================
# AlertChannel Enum Tests
# =============================================================================

class TestAlertChannel:
    """Tests for AlertChannel enum."""

    def test_discord_value(self):
        """Test Discord channel value."""
        assert AlertChannel.DISCORD.value == 'discord'

    def test_email_value(self):
        """Test email channel value."""
        assert AlertChannel.EMAIL.value == 'email'

    def test_logging_value(self):
        """Test logging channel value."""
        assert AlertChannel.LOGGING.value == 'logging'

    def test_all_channels_are_strings(self):
        """Test all channels inherit from str."""
        for channel in AlertChannel:
            assert isinstance(channel, str)

    def test_enum_count(self):
        """Test correct number of channels."""
        assert len(AlertChannel) == 3


# =============================================================================
# ScanConfig Tests
# =============================================================================

class TestScanConfig:
    """Tests for ScanConfig dataclass."""

    def test_default_symbols(self):
        """Test default symbols list contains expected tickers."""
        config = ScanConfig()
        assert 'SPY' in config.symbols
        assert 'QQQ' in config.symbols
        assert 'AAPL' in config.symbols
        assert 'NVDA' in config.symbols
        assert len(config.symbols) >= 10

    def test_default_timeframes(self):
        """Test default timeframes."""
        config = ScanConfig()
        assert config.timeframes == ['1H', '1D', '1W', '1M']

    def test_default_patterns(self):
        """Test default patterns."""
        config = ScanConfig()
        expected = ['2-2', '3-2', '3-2-2', '2-1-2', '3-1-2']
        assert config.patterns == expected

    def test_default_lookback_bars(self):
        """Test default lookback bars."""
        config = ScanConfig()
        assert config.lookback_bars == 50

    def test_default_signal_age_bars(self):
        """Test default signal age bars."""
        config = ScanConfig()
        assert config.signal_age_bars == 3

    def test_default_min_magnitude(self):
        """Test default minimum magnitude."""
        config = ScanConfig()
        assert config.min_magnitude_pct == 0.5

    def test_default_min_risk_reward(self):
        """Test default minimum risk/reward."""
        config = ScanConfig()
        assert config.min_risk_reward == 1.0

    def test_custom_symbols(self):
        """Test custom symbols list."""
        config = ScanConfig(symbols=['TSLA', 'AMZN'])
        assert config.symbols == ['TSLA', 'AMZN']

    def test_custom_timeframes(self):
        """Test custom timeframes."""
        config = ScanConfig(timeframes=['1H', '1D'])
        assert config.timeframes == ['1H', '1D']

    def test_custom_magnitude(self):
        """Test custom magnitude threshold."""
        config = ScanConfig(min_magnitude_pct=1.0)
        assert config.min_magnitude_pct == 1.0

    def test_custom_risk_reward(self):
        """Test custom risk/reward threshold."""
        config = ScanConfig(min_risk_reward=2.0)
        assert config.min_risk_reward == 2.0


# =============================================================================
# ScheduleConfig Tests
# =============================================================================

class TestScheduleConfig:
    """Tests for ScheduleConfig dataclass."""

    def test_default_fifteen_min_cron(self):
        """Test default 15-minute cron."""
        config = ScheduleConfig()
        assert config.fifteen_min_cron == '0,15,30,45 9-15 * * mon-fri'

    def test_default_thirty_min_cron(self):
        """Test default 30-minute cron."""
        config = ScheduleConfig()
        assert config.thirty_min_cron == '0,30 9-15 * * mon-fri'

    def test_default_hourly_cron(self):
        """Test default hourly cron."""
        config = ScheduleConfig()
        assert config.hourly_cron == '30 9-15 * * mon-fri'

    def test_default_daily_cron(self):
        """Test default daily cron."""
        config = ScheduleConfig()
        assert config.daily_cron == '0 17 * * mon-fri'

    def test_default_weekly_cron(self):
        """Test default weekly cron."""
        config = ScheduleConfig()
        assert config.weekly_cron == '0 18 * * fri'

    def test_default_monthly_cron(self):
        """Test default monthly cron."""
        config = ScheduleConfig()
        assert config.monthly_cron == '0 18 28 * *'

    def test_htf_resampling_enabled_by_default(self):
        """Test HTF resampling is enabled by default."""
        config = ScheduleConfig()
        assert config.enable_htf_resampling is True

    def test_base_timeframe(self):
        """Test base timeframe for resampling."""
        config = ScheduleConfig()
        assert config.base_timeframe == '15min'

    def test_base_scan_cron(self):
        """Test base scan cron."""
        config = ScheduleConfig()
        assert config.base_scan_cron == '30,45,0,15 9-15 * * mon-fri'

    def test_scan_15m_disabled_by_default(self):
        """Test 15m scan disabled by default."""
        config = ScheduleConfig()
        assert config.scan_15m is False

    def test_scan_30m_disabled_by_default(self):
        """Test 30m scan disabled by default."""
        config = ScheduleConfig()
        assert config.scan_30m is False

    def test_scan_hourly_enabled(self):
        """Test hourly scan enabled by default."""
        config = ScheduleConfig()
        assert config.scan_hourly is True

    def test_scan_daily_enabled(self):
        """Test daily scan enabled by default."""
        config = ScheduleConfig()
        assert config.scan_daily is True

    def test_scan_weekly_enabled(self):
        """Test weekly scan enabled by default."""
        config = ScheduleConfig()
        assert config.scan_weekly is True

    def test_scan_monthly_enabled(self):
        """Test monthly scan enabled by default."""
        config = ScheduleConfig()
        assert config.scan_monthly is True

    def test_default_timezone(self):
        """Test default timezone is New York."""
        config = ScheduleConfig()
        assert config.timezone == 'America/New_York'

    def test_misfire_grace_time(self):
        """Test misfire grace time is 5 minutes."""
        config = ScheduleConfig()
        assert config.misfire_grace_time == 300

    def test_custom_timezone(self):
        """Test custom timezone."""
        config = ScheduleConfig(timezone='UTC')
        assert config.timezone == 'UTC'


# =============================================================================
# AlertConfig Tests
# =============================================================================

class TestAlertConfig:
    """Tests for AlertConfig dataclass."""

    def test_discord_enabled_by_default(self):
        """Test Discord enabled by default."""
        with patch.dict(os.environ, {'DISCORD_WEBHOOK_URL': 'https://discord.com/api/webhooks/123/abc'}):
            config = AlertConfig()
            assert config.discord_enabled is True

    def test_discord_disabled_without_webhook(self):
        """Test Discord disabled when no webhook."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing webhook vars
            env = {k: v for k, v in os.environ.items() if 'DISCORD' not in k}
            with patch.dict(os.environ, env, clear=True):
                config = AlertConfig()
                assert config.discord_enabled is False

    def test_alert_on_signal_detection_disabled(self):
        """Test signal detection alerts disabled by default."""
        config = AlertConfig()
        assert config.alert_on_signal_detection is False

    def test_alert_on_trigger_disabled(self):
        """Test trigger alerts disabled by default."""
        config = AlertConfig()
        assert config.alert_on_trigger is False

    def test_alert_on_trade_entry_enabled(self):
        """Test trade entry alerts enabled by default."""
        config = AlertConfig()
        assert config.alert_on_trade_entry is True

    def test_alert_on_trade_exit_enabled(self):
        """Test trade exit alerts enabled by default."""
        config = AlertConfig()
        assert config.alert_on_trade_exit is True

    def test_email_disabled_by_default(self):
        """Test email disabled by default."""
        config = AlertConfig()
        assert config.email_enabled is False

    def test_email_to_empty_by_default(self):
        """Test email recipients empty by default."""
        config = AlertConfig()
        assert config.email_to == []

    def test_logging_enabled_by_default(self):
        """Test logging enabled by default."""
        config = AlertConfig()
        assert config.logging_enabled is True

    def test_default_log_file(self):
        """Test default log file path."""
        config = AlertConfig()
        assert config.log_file == 'logs/signals.log'

    def test_default_log_level(self):
        """Test default log level is INFO."""
        config = AlertConfig()
        assert config.log_level == 'INFO'

    def test_min_alert_interval(self):
        """Test minimum alert interval is 60 seconds."""
        config = AlertConfig()
        assert config.min_alert_interval_seconds == 60

    def test_post_init_loads_equity_webhook(self):
        """Test __post_init__ loads DISCORD_EQUITY_WEBHOOK_URL."""
        with patch.dict(os.environ, {'DISCORD_EQUITY_WEBHOOK_URL': 'https://discord.com/api/webhooks/equity/abc'}):
            config = AlertConfig()
            assert config.discord_webhook_url == 'https://discord.com/api/webhooks/equity/abc'

    def test_post_init_falls_back_to_generic_webhook(self):
        """Test __post_init__ falls back to DISCORD_WEBHOOK_URL."""
        env = {k: v for k, v in os.environ.items() if 'DISCORD' not in k}
        env['DISCORD_WEBHOOK_URL'] = 'https://discord.com/api/webhooks/generic/def'
        with patch.dict(os.environ, env, clear=True):
            config = AlertConfig()
            assert config.discord_webhook_url == 'https://discord.com/api/webhooks/generic/def'

    def test_post_init_disables_discord_without_url(self):
        """Test __post_init__ disables Discord when no URL."""
        env = {k: v for k, v in os.environ.items() if 'DISCORD' not in k}
        with patch.dict(os.environ, env, clear=True):
            config = AlertConfig()
            assert config.discord_enabled is False

    def test_explicit_webhook_url_used(self):
        """Test explicit webhook URL takes precedence."""
        config = AlertConfig(discord_webhook_url='https://discord.com/api/webhooks/explicit/xyz')
        assert config.discord_webhook_url == 'https://discord.com/api/webhooks/explicit/xyz'


# =============================================================================
# ExecutionConfig Tests
# =============================================================================

class TestExecutionConfig:
    """Tests for ExecutionConfig dataclass."""

    def test_execution_disabled_by_default(self):
        """Test execution disabled by default for safety."""
        config = ExecutionConfig()
        assert config.enabled is False

    def test_default_account(self):
        """Test default account is SMALL."""
        config = ExecutionConfig()
        assert config.account == 'SMALL'

    def test_default_max_capital(self):
        """Test default max capital per trade."""
        config = ExecutionConfig()
        assert config.max_capital_per_trade == 300.0

    def test_default_max_positions(self):
        """Test default max concurrent positions."""
        config = ExecutionConfig()
        assert config.max_concurrent_positions == 5

    def test_default_target_delta(self):
        """Test default target delta."""
        config = ExecutionConfig()
        assert config.target_delta == 0.55

    def test_default_delta_range(self):
        """Test default delta range."""
        config = ExecutionConfig()
        assert config.delta_range_min == 0.45
        assert config.delta_range_max == 0.65

    def test_default_dte_range(self):
        """Test default DTE range."""
        config = ExecutionConfig()
        assert config.min_dte == 7
        assert config.max_dte == 21
        assert config.target_dte == 14

    def test_use_limit_orders_by_default(self):
        """Test limit orders enabled by default."""
        config = ExecutionConfig()
        assert config.use_limit_orders is True

    def test_default_limit_price_buffer(self):
        """Test default limit price buffer."""
        config = ExecutionConfig()
        assert config.limit_price_buffer == 0.02

    def test_tfc_reeval_enabled_by_default(self):
        """Test TFC re-evaluation enabled by default."""
        config = ExecutionConfig()
        assert config.tfc_reeval_enabled is True

    def test_tfc_reeval_min_strength(self):
        """Test TFC re-eval minimum strength is 3."""
        config = ExecutionConfig()
        assert config.tfc_reeval_min_strength == 3

    def test_tfc_reeval_block_on_flip(self):
        """Test TFC re-eval blocks on direction flip."""
        config = ExecutionConfig()
        assert config.tfc_reeval_block_on_flip is True

    def test_tfc_reeval_log_always(self):
        """Test TFC re-eval always logs."""
        config = ExecutionConfig()
        assert config.tfc_reeval_log_always is True

    def test_custom_capital(self):
        """Test custom max capital."""
        config = ExecutionConfig(max_capital_per_trade=500.0)
        assert config.max_capital_per_trade == 500.0


# =============================================================================
# MonitoringConfig Tests
# =============================================================================

class TestMonitoringConfig:
    """Tests for MonitoringConfig dataclass."""

    def test_monitoring_enabled_by_default(self):
        """Test monitoring enabled by default."""
        config = MonitoringConfig()
        assert config.enabled is True

    def test_default_check_interval(self):
        """Test default check interval is 60 seconds."""
        config = MonitoringConfig()
        assert config.check_interval == 60

    def test_default_minimum_hold(self):
        """Test default minimum hold is 5 minutes."""
        config = MonitoringConfig()
        assert config.minimum_hold_seconds == 300

    def test_default_exit_dte(self):
        """Test default exit DTE is 3."""
        config = MonitoringConfig()
        assert config.exit_dte == 3

    def test_default_max_loss_pct(self):
        """Test default max loss is 50%."""
        config = MonitoringConfig()
        assert config.max_loss_pct == 0.50

    def test_default_max_profit_pct(self):
        """Test default max profit is 100%."""
        config = MonitoringConfig()
        assert config.max_profit_pct == 1.00

    def test_alert_on_exit_enabled(self):
        """Test alert on exit enabled by default."""
        config = MonitoringConfig()
        assert config.alert_on_exit is True

    def test_custom_check_interval(self):
        """Test custom check interval."""
        config = MonitoringConfig(check_interval=30)
        assert config.check_interval == 30


# =============================================================================
# ApiConfig Tests
# =============================================================================

class TestApiConfig:
    """Tests for ApiConfig dataclass."""

    def test_api_disabled_by_default(self):
        """Test API disabled by default."""
        config = ApiConfig()
        assert config.enabled is False

    def test_default_host(self):
        """Test default host is 0.0.0.0."""
        config = ApiConfig()
        assert config.host == '0.0.0.0'

    def test_default_port(self):
        """Test default port is 8081."""
        config = ApiConfig()
        assert config.port == 8081

    def test_custom_port(self):
        """Test custom port."""
        config = ApiConfig(port=9000)
        assert config.port == 9000


# =============================================================================
# SignalAutomationConfig Tests
# =============================================================================

class TestSignalAutomationConfig:
    """Tests for SignalAutomationConfig master configuration."""

    def test_default_creates_all_subconfigs(self):
        """Test default creates all sub-configurations."""
        config = SignalAutomationConfig()
        assert isinstance(config.scan, ScanConfig)
        assert isinstance(config.schedule, ScheduleConfig)
        assert isinstance(config.alerts, AlertConfig)
        assert isinstance(config.execution, ExecutionConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert isinstance(config.api, ApiConfig)

    def test_default_store_path(self):
        """Test default store path."""
        config = SignalAutomationConfig()
        assert config.store_path == 'data/signals'

    def test_default_graceful_shutdown_timeout(self):
        """Test default shutdown timeout is 30 seconds."""
        config = SignalAutomationConfig()
        assert config.graceful_shutdown_timeout == 30

    def test_default_health_check_interval(self):
        """Test default health check interval is 5 minutes."""
        config = SignalAutomationConfig()
        assert config.health_check_interval == 300

    def test_custom_store_path(self):
        """Test custom store path."""
        config = SignalAutomationConfig(store_path='/custom/path')
        assert config.store_path == '/custom/path'


# =============================================================================
# SignalAutomationConfig.from_env() Tests
# =============================================================================

class TestSignalAutomationConfigFromEnv:
    """Tests for SignalAutomationConfig.from_env() classmethod."""

    def test_from_env_default_values(self):
        """Test from_env with no environment variables."""
        env = {k: v for k, v in os.environ.items() if not k.startswith('SIGNAL_') and 'DISCORD' not in k}
        with patch.dict(os.environ, env, clear=True):
            config = SignalAutomationConfig.from_env()
            assert isinstance(config, SignalAutomationConfig)
            assert config.execution.enabled is False

    def test_from_env_custom_symbols(self):
        """Test from_env parses SIGNAL_SYMBOLS."""
        with patch.dict(os.environ, {'SIGNAL_SYMBOLS': 'AAPL,MSFT,GOOG'}):
            config = SignalAutomationConfig.from_env()
            assert config.scan.symbols == ['AAPL', 'MSFT', 'GOOG']

    def test_from_env_custom_timeframes(self):
        """Test from_env parses SIGNAL_TIMEFRAMES."""
        with patch.dict(os.environ, {'SIGNAL_TIMEFRAMES': '1H,1D'}):
            config = SignalAutomationConfig.from_env()
            assert config.scan.timeframes == ['1H', '1D']

    def test_from_env_scan_hourly_true(self):
        """Test from_env parses SIGNAL_SCAN_HOURLY=true."""
        with patch.dict(os.environ, {'SIGNAL_SCAN_HOURLY': 'true'}):
            config = SignalAutomationConfig.from_env()
            assert config.schedule.scan_hourly is True

    def test_from_env_scan_hourly_false(self):
        """Test from_env parses SIGNAL_SCAN_HOURLY=false."""
        with patch.dict(os.environ, {'SIGNAL_SCAN_HOURLY': 'false'}):
            config = SignalAutomationConfig.from_env()
            assert config.schedule.scan_hourly is False

    def test_from_env_scan_daily(self):
        """Test from_env parses SIGNAL_SCAN_DAILY."""
        with patch.dict(os.environ, {'SIGNAL_SCAN_DAILY': 'false'}):
            config = SignalAutomationConfig.from_env()
            assert config.schedule.scan_daily is False

    def test_from_env_scan_weekly(self):
        """Test from_env parses SIGNAL_SCAN_WEEKLY."""
        with patch.dict(os.environ, {'SIGNAL_SCAN_WEEKLY': 'false'}):
            config = SignalAutomationConfig.from_env()
            assert config.schedule.scan_weekly is False

    def test_from_env_scan_monthly(self):
        """Test from_env parses SIGNAL_SCAN_MONTHLY."""
        with patch.dict(os.environ, {'SIGNAL_SCAN_MONTHLY': 'false'}):
            config = SignalAutomationConfig.from_env()
            assert config.schedule.scan_monthly is False

    def test_from_env_discord_webhook(self):
        """Test from_env loads Discord webhook."""
        with patch.dict(os.environ, {'DISCORD_WEBHOOK_URL': 'https://discord.com/api/webhooks/123/abc'}):
            config = SignalAutomationConfig.from_env()
            assert config.alerts.discord_webhook_url == 'https://discord.com/api/webhooks/123/abc'
            assert config.alerts.discord_enabled is True

    def test_from_env_equity_webhook_priority(self):
        """Test from_env prefers equity-specific webhook."""
        with patch.dict(os.environ, {
            'DISCORD_WEBHOOK_URL': 'https://discord.com/api/webhooks/generic/123',
            'DISCORD_EQUITY_WEBHOOK_URL': 'https://discord.com/api/webhooks/equity/456'
        }):
            config = SignalAutomationConfig.from_env()
            assert config.alerts.discord_webhook_url == 'https://discord.com/api/webhooks/equity/456'

    def test_from_env_log_level(self):
        """Test from_env parses SIGNAL_LOG_LEVEL."""
        with patch.dict(os.environ, {'SIGNAL_LOG_LEVEL': 'DEBUG'}):
            config = SignalAutomationConfig.from_env()
            assert config.alerts.log_level == 'DEBUG'

    def test_from_env_execution_enabled(self):
        """Test from_env parses SIGNAL_EXECUTION_ENABLED."""
        with patch.dict(os.environ, {'SIGNAL_EXECUTION_ENABLED': 'true'}):
            config = SignalAutomationConfig.from_env()
            assert config.execution.enabled is True

    def test_from_env_execution_account(self):
        """Test from_env parses SIGNAL_EXECUTION_ACCOUNT."""
        with patch.dict(os.environ, {'SIGNAL_EXECUTION_ACCOUNT': 'MEDIUM'}):
            config = SignalAutomationConfig.from_env()
            assert config.execution.account == 'MEDIUM'

    def test_from_env_max_capital(self):
        """Test from_env parses SIGNAL_MAX_CAPITAL_PER_TRADE."""
        with patch.dict(os.environ, {'SIGNAL_MAX_CAPITAL_PER_TRADE': '500'}):
            config = SignalAutomationConfig.from_env()
            assert config.execution.max_capital_per_trade == 500.0

    def test_from_env_max_positions(self):
        """Test from_env parses SIGNAL_MAX_CONCURRENT_POSITIONS."""
        with patch.dict(os.environ, {'SIGNAL_MAX_CONCURRENT_POSITIONS': '10'}):
            config = SignalAutomationConfig.from_env()
            assert config.execution.max_concurrent_positions == 10

    def test_from_env_monitoring_enabled(self):
        """Test from_env parses SIGNAL_MONITORING_ENABLED."""
        with patch.dict(os.environ, {'SIGNAL_MONITORING_ENABLED': 'false'}):
            config = SignalAutomationConfig.from_env()
            assert config.monitoring.enabled is False

    def test_from_env_monitor_interval(self):
        """Test from_env parses SIGNAL_MONITOR_INTERVAL."""
        with patch.dict(os.environ, {'SIGNAL_MONITOR_INTERVAL': '30'}):
            config = SignalAutomationConfig.from_env()
            assert config.monitoring.check_interval == 30

    def test_from_env_min_hold_seconds(self):
        """Test from_env parses SIGNAL_MIN_HOLD_SECONDS."""
        with patch.dict(os.environ, {'SIGNAL_MIN_HOLD_SECONDS': '600'}):
            config = SignalAutomationConfig.from_env()
            assert config.monitoring.minimum_hold_seconds == 600

    def test_from_env_exit_dte(self):
        """Test from_env parses SIGNAL_EXIT_DTE."""
        with patch.dict(os.environ, {'SIGNAL_EXIT_DTE': '5'}):
            config = SignalAutomationConfig.from_env()
            assert config.monitoring.exit_dte == 5

    def test_from_env_max_loss_pct(self):
        """Test from_env parses SIGNAL_MAX_LOSS_PCT."""
        with patch.dict(os.environ, {'SIGNAL_MAX_LOSS_PCT': '0.75'}):
            config = SignalAutomationConfig.from_env()
            assert config.monitoring.max_loss_pct == 0.75

    def test_from_env_max_profit_pct(self):
        """Test from_env parses SIGNAL_MAX_PROFIT_PCT."""
        with patch.dict(os.environ, {'SIGNAL_MAX_PROFIT_PCT': '2.0'}):
            config = SignalAutomationConfig.from_env()
            assert config.monitoring.max_profit_pct == 2.0

    def test_from_env_store_path(self):
        """Test from_env parses SIGNAL_STORE_PATH."""
        with patch.dict(os.environ, {'SIGNAL_STORE_PATH': '/custom/signals'}):
            config = SignalAutomationConfig.from_env()
            assert config.store_path == '/custom/signals'

    def test_from_env_api_enabled(self):
        """Test from_env parses SIGNAL_API_ENABLED."""
        with patch.dict(os.environ, {'SIGNAL_API_ENABLED': 'true'}):
            config = SignalAutomationConfig.from_env()
            assert config.api.enabled is True

    def test_from_env_api_host(self):
        """Test from_env parses SIGNAL_API_HOST."""
        with patch.dict(os.environ, {'SIGNAL_API_HOST': '127.0.0.1'}):
            config = SignalAutomationConfig.from_env()
            assert config.api.host == '127.0.0.1'

    def test_from_env_api_port(self):
        """Test from_env parses SIGNAL_API_PORT."""
        with patch.dict(os.environ, {'SIGNAL_API_PORT': '9999'}):
            config = SignalAutomationConfig.from_env()
            assert config.api.port == 9999


# =============================================================================
# SignalAutomationConfig.validate() Tests
# =============================================================================

class TestSignalAutomationConfigValidate:
    """Tests for SignalAutomationConfig.validate() method."""

    def test_valid_default_config(self):
        """Test default config is valid."""
        env = {k: v for k, v in os.environ.items() if 'DISCORD' not in k}
        with patch.dict(os.environ, env, clear=True):
            config = SignalAutomationConfig()
            # Default config has logging enabled, so it's valid
            issues = config.validate()
            # May have Discord warning but no critical issues
            assert not any('No symbols' in issue for issue in issues)
            assert not any('Invalid timeframe' in issue for issue in issues)

    def test_validate_empty_symbols(self):
        """Test validation catches empty symbols."""
        config = SignalAutomationConfig()
        config.scan.symbols = []
        issues = config.validate()
        assert any('No symbols' in issue for issue in issues)

    def test_validate_invalid_timeframe(self):
        """Test validation catches invalid timeframe."""
        config = SignalAutomationConfig()
        config.scan.timeframes = ['1H', 'INVALID']
        issues = config.validate()
        assert any('Invalid timeframe: INVALID' in issue for issue in issues)

    def test_validate_valid_15m_timeframe(self):
        """Test validation accepts 15m timeframe."""
        config = SignalAutomationConfig()
        config.scan.timeframes = ['15m', '1H']
        issues = config.validate()
        assert not any('Invalid timeframe: 15m' in issue for issue in issues)

    def test_validate_valid_30m_timeframe(self):
        """Test validation accepts 30m timeframe."""
        config = SignalAutomationConfig()
        config.scan.timeframes = ['30m', '1H']
        issues = config.validate()
        assert not any('Invalid timeframe: 30m' in issue for issue in issues)

    def test_validate_no_alert_channels(self):
        """Test validation catches no alert channels."""
        config = SignalAutomationConfig()
        config.alerts.discord_enabled = False
        config.alerts.logging_enabled = False
        issues = config.validate()
        assert any('No alert channels enabled' in issue for issue in issues)

    def test_validate_discord_no_url(self):
        """Test validation catches Discord enabled without URL."""
        config = SignalAutomationConfig()
        config.alerts.discord_enabled = True
        config.alerts.discord_webhook_url = ''
        issues = config.validate()
        assert any('Discord enabled but no webhook URL' in issue for issue in issues)

    def test_validate_discord_invalid_url(self):
        """Test validation catches invalid Discord URL."""
        config = SignalAutomationConfig()
        config.alerts.discord_enabled = True
        config.alerts.discord_webhook_url = 'https://invalid.com/webhook'
        issues = config.validate()
        assert any('Discord webhook URL appears invalid' in issue for issue in issues)

    def test_validate_execution_negative_capital(self):
        """Test validation catches negative capital."""
        config = SignalAutomationConfig()
        config.execution.enabled = True
        config.execution.max_capital_per_trade = -100
        issues = config.validate()
        assert any('max_capital_per_trade must be positive' in issue for issue in issues)

    def test_validate_execution_zero_positions(self):
        """Test validation catches zero max positions."""
        config = SignalAutomationConfig()
        config.execution.enabled = True
        config.execution.max_concurrent_positions = 0
        issues = config.validate()
        assert any('max_concurrent_positions must be positive' in issue for issue in issues)

    def test_validate_execution_dte_range(self):
        """Test validation catches invalid DTE range."""
        config = SignalAutomationConfig()
        config.execution.enabled = True
        config.execution.min_dte = 30
        config.execution.max_dte = 21
        issues = config.validate()
        assert any('min_dte must be less than max_dte' in issue for issue in issues)

    def test_validate_execution_delta_range_invalid(self):
        """Test validation catches invalid delta range."""
        config = SignalAutomationConfig()
        config.execution.enabled = True
        config.execution.delta_range_min = 0.8
        config.execution.delta_range_max = 0.6
        issues = config.validate()
        assert any('delta range must be between 0 and 1' in issue for issue in issues)

    def test_validate_monitoring_short_interval(self):
        """Test validation catches too short check interval."""
        config = SignalAutomationConfig()
        config.monitoring.enabled = True
        config.monitoring.check_interval = 5
        issues = config.validate()
        assert any('check_interval must be at least 10 seconds' in issue for issue in issues)

    def test_validate_monitoring_negative_exit_dte(self):
        """Test validation catches negative exit DTE."""
        config = SignalAutomationConfig()
        config.monitoring.enabled = True
        config.monitoring.exit_dte = -1
        issues = config.validate()
        assert any('exit_dte must be non-negative' in issue for issue in issues)

    def test_validate_monitoring_invalid_max_loss(self):
        """Test validation catches invalid max loss."""
        config = SignalAutomationConfig()
        config.monitoring.enabled = True
        config.monitoring.max_loss_pct = 1.5
        issues = config.validate()
        assert any('max_loss_pct must be between 0 and 1' in issue for issue in issues)

    def test_validate_monitoring_invalid_max_profit(self):
        """Test validation catches excessive max profit."""
        config = SignalAutomationConfig()
        config.monitoring.enabled = True
        config.monitoring.max_profit_pct = 15.0
        issues = config.validate()
        assert any('max_profit_pct must be between 0 and 10' in issue for issue in issues)

    def test_validate_returns_empty_for_valid(self):
        """Test validation returns empty list for valid config."""
        config = SignalAutomationConfig()
        config.alerts.discord_enabled = False  # Disable Discord to avoid URL check
        config.alerts.logging_enabled = True   # Keep logging enabled
        issues = config.validate()
        assert issues == []

# ATLAS Trading System - UI/Alert System Specification

**Version:** 1.0
**Date:** 2025-11-25
**Status:** Design Specification (Ready for Implementation)
**Author:** Claude Code Analysis Session

---

## Executive Summary

This document specifies a real-time portfolio monitoring and alert system for the ATLAS Trading System. The design integrates with existing infrastructure (AlpacaTradingClient, ExecutionLogger, OptionsExecutor) and provides multi-channel notifications for trade events.

**Target Implementation Time:** 2-3 sessions (6-9 hours)
**Dependencies:** Existing execution layer, config/settings.py, strat/options_module.py

---

## Requirements Summary

| Requirement | Priority | Complexity |
|-------------|----------|------------|
| Real-time position status display | HIGH | Medium |
| P/L tracking (position + portfolio) | HIGH | Low |
| Email alerts (entries, exits, stops, targets) | HIGH | Medium |
| Mobile push notifications | MEDIUM | Medium |
| Pattern detection alerts | MEDIUM | Low |
| Web dashboard (optional) | LOW | High |

---

## System Architecture

### Component Diagram

```
+------------------+     +-------------------+     +------------------+
|                  |     |                   |     |                  |
|  Data Sources    |---->|  Alert Engine     |---->|  Notification    |
|                  |     |  (Core)           |     |  Channels        |
+------------------+     +-------------------+     +------------------+
        |                        |                        |
        v                        v                        v
+------------------+     +-------------------+     +------------------+
| - Alpaca API     |     | - Event Queue     |     | - Email (SMTP)   |
| - Pattern        |     | - Alert Rules     |     | - Push (Pushover)|
|   Detector       |     | - Rate Limiter    |     | - Console        |
| - Options Module |     | - Alert History   |     | - Webhook        |
+------------------+     +-------------------+     +------------------+
```

### Data Flow

```
1. EVENT GENERATION
   OptionsBacktester.backtest_trades() --> TradeEvent
   Tier1Detector.detect_patterns() --> PatternEvent
   AlpacaTradingClient.get_positions() --> PositionEvent

2. EVENT PROCESSING
   EventQueue.enqueue(event)
   AlertEngine.process(event)
   AlertRules.evaluate(event) --> Alert[]

3. NOTIFICATION DISPATCH
   NotificationRouter.route(alert)
   Channel.send(alert) --> DeliveryStatus
   AlertHistory.record(alert, status)
```

---

## Component Specifications

### 1. AlertEngine (Core)

**Location:** `alerts/alert_engine.py`

**Purpose:** Central event processing and alert generation

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Callable
from queue import PriorityQueue
import threading

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = 1      # Pattern detected, position opened
    WARNING = 2   # Approaching stop, theta decay warning
    CRITICAL = 3  # Stop hit, large loss, system error

class AlertType(Enum):
    """Types of alerts."""
    TRADE_ENTRY = "trade_entry"
    TRADE_EXIT = "trade_exit"
    STOP_HIT = "stop_hit"
    TARGET_HIT = "target_hit"
    PATTERN_DETECTED = "pattern_detected"
    POSITION_UPDATE = "position_update"
    PORTFOLIO_SUMMARY = "portfolio_summary"
    SYSTEM_ERROR = "system_error"

@dataclass
class Alert:
    """Represents an alert to be sent."""
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    symbol: Optional[str] = None
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'type': self.alert_type.value,
            'severity': self.severity.name,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'data': self.data
        }

@dataclass
class AlertRule:
    """Defines when to trigger an alert."""
    name: str
    condition: Callable[[dict], bool]
    alert_type: AlertType
    severity: AlertSeverity
    message_template: str
    channels: List[str] = field(default_factory=lambda: ['console', 'email'])
    cooldown_seconds: int = 0  # Prevent spam

class AlertEngine:
    """
    Central alert processing engine.

    Responsibilities:
    - Process incoming events from data sources
    - Evaluate alert rules against events
    - Route alerts to appropriate channels
    - Maintain alert history and rate limiting
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.rules: List[AlertRule] = []
        self.channels: dict = {}
        self.history: List[Alert] = []
        self.last_alert_times: dict = {}  # For cooldown tracking
        self._lock = threading.Lock()

    def register_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        self.rules.append(rule)

    def register_channel(self, name: str, channel: 'NotificationChannel') -> None:
        """Register a notification channel."""
        self.channels[name] = channel

    def process_event(self, event: dict) -> List[Alert]:
        """
        Process an event and generate alerts.

        Args:
            event: Dict containing event data

        Returns:
            List of generated alerts
        """
        alerts = []

        for rule in self.rules:
            if self._check_cooldown(rule.name):
                continue

            try:
                if rule.condition(event):
                    alert = self._create_alert(rule, event)
                    alerts.append(alert)
                    self._dispatch_alert(alert, rule.channels)
                    self._update_cooldown(rule.name, rule.cooldown_seconds)
            except Exception as e:
                # Log but don't crash on rule evaluation errors
                self._log_error(f"Rule {rule.name} failed: {e}")

        return alerts

    def _create_alert(self, rule: AlertRule, event: dict) -> Alert:
        """Create an alert from a rule and event."""
        message = rule.message_template.format(**event)
        return Alert(
            alert_type=rule.alert_type,
            severity=rule.severity,
            title=f"ATLAS: {rule.name}",
            message=message,
            symbol=event.get('symbol'),
            data=event
        )

    def _dispatch_alert(self, alert: Alert, channel_names: List[str]) -> None:
        """Send alert to specified channels."""
        for name in channel_names:
            if name in self.channels:
                try:
                    self.channels[name].send(alert)
                except Exception as e:
                    self._log_error(f"Channel {name} failed: {e}")

        with self._lock:
            self.history.append(alert)

    def _check_cooldown(self, rule_name: str) -> bool:
        """Check if rule is in cooldown period."""
        if rule_name not in self.last_alert_times:
            return False
        last_time, cooldown = self.last_alert_times[rule_name]
        return (datetime.now() - last_time).total_seconds() < cooldown

    def _update_cooldown(self, rule_name: str, cooldown: int) -> None:
        """Update cooldown tracking for a rule."""
        if cooldown > 0:
            self.last_alert_times[rule_name] = (datetime.now(), cooldown)

    def _log_error(self, message: str) -> None:
        """Log an error (integrate with ExecutionLogger)."""
        print(f"[ALERT_ENGINE_ERROR] {message}")
```

---

### 2. Notification Channels

**Location:** `alerts/channels/`

#### 2.1 Base Channel Interface

```python
# alerts/channels/base.py

from abc import ABC, abstractmethod
from typing import Optional

class NotificationChannel(ABC):
    """Base class for notification channels."""

    @abstractmethod
    def send(self, alert: 'Alert') -> bool:
        """
        Send an alert through this channel.

        Args:
            alert: Alert to send

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test that the channel is configured correctly."""
        pass
```

#### 2.2 Console Channel

```python
# alerts/channels/console.py

from datetime import datetime
from .base import NotificationChannel

class ConsoleChannel(NotificationChannel):
    """Print alerts to console with color coding."""

    COLORS = {
        'INFO': '\033[94m',      # Blue
        'WARNING': '\033[93m',   # Yellow
        'CRITICAL': '\033[91m',  # Red
        'RESET': '\033[0m'
    }

    def send(self, alert: 'Alert') -> bool:
        color = self.COLORS.get(alert.severity.name, '')
        reset = self.COLORS['RESET']

        timestamp = alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        print(f"{color}[{timestamp}] [{alert.severity.name}] {alert.title}{reset}")
        print(f"  {alert.message}")
        if alert.symbol:
            print(f"  Symbol: {alert.symbol}")
        print()
        return True

    def test_connection(self) -> bool:
        return True  # Console always available
```

#### 2.3 Email Channel (SMTP)

```python
# alerts/channels/email.py

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from .base import NotificationChannel

class EmailChannel(NotificationChannel):
    """
    Send alerts via email using SMTP.

    Configuration (.env):
        ALERT_EMAIL_SMTP_HOST=smtp.gmail.com
        ALERT_EMAIL_SMTP_PORT=587
        ALERT_EMAIL_USERNAME=your-email@gmail.com
        ALERT_EMAIL_PASSWORD=app-specific-password
        ALERT_EMAIL_FROM=your-email@gmail.com
        ALERT_EMAIL_TO=recipient@example.com
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_tls: bool = True
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls

    def send(self, alert: 'Alert') -> bool:
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[ATLAS {alert.severity.name}] {alert.title}"
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(self.to_addrs)

            # Plain text version
            text_body = self._format_text(alert)
            msg.attach(MIMEText(text_body, 'plain'))

            # HTML version
            html_body = self._format_html(alert)
            msg.attach(MIMEText(html_body, 'html'))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            return True

        except Exception as e:
            print(f"[EMAIL_ERROR] Failed to send: {e}")
            return False

    def _format_text(self, alert: 'Alert') -> str:
        """Format alert as plain text."""
        lines = [
            f"ATLAS Trading System Alert",
            f"=" * 40,
            f"Type: {alert.alert_type.value}",
            f"Severity: {alert.severity.name}",
            f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S ET')}",
            f"",
            f"Message:",
            f"{alert.message}",
        ]

        if alert.symbol:
            lines.append(f"\nSymbol: {alert.symbol}")

        if alert.data:
            lines.append(f"\nDetails:")
            for key, value in alert.data.items():
                if key not in ['symbol', 'message']:
                    lines.append(f"  {key}: {value}")

        return '\n'.join(lines)

    def _format_html(self, alert: 'Alert') -> str:
        """Format alert as HTML email."""
        severity_colors = {
            'INFO': '#3498db',
            'WARNING': '#f39c12',
            'CRITICAL': '#e74c3c'
        }
        color = severity_colors.get(alert.severity.name, '#333')

        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: {color}; color: white; padding: 15px; border-radius: 5px 5px 0 0;">
                <h2 style="margin: 0;">ATLAS Trading Alert</h2>
                <p style="margin: 5px 0 0 0;">{alert.severity.name} - {alert.alert_type.value}</p>
            </div>
            <div style="border: 1px solid #ddd; border-top: none; padding: 20px; border-radius: 0 0 5px 5px;">
                <p style="font-size: 16px; color: #333;">{alert.message}</p>
                <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">Time</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S ET')}</td>
                    </tr>
        """

        if alert.symbol:
            html += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">Symbol</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; font-weight: bold;">{alert.symbol}</td>
                    </tr>
            """

        for key, value in alert.data.items():
            if key not in ['symbol', 'message'] and not key.startswith('_'):
                html += f"""
                    <tr>
                        <td style="padding: 8px; border-bottom: 1px solid #eee; color: #666;">{key.replace('_', ' ').title()}</td>
                        <td style="padding: 8px; border-bottom: 1px solid #eee;">{value}</td>
                    </tr>
                """

        html += """
                </table>
            </div>
            <p style="color: #999; font-size: 12px; margin-top: 20px; text-align: center;">
                ATLAS Algorithmic Trading System
            </p>
        </body>
        </html>
        """
        return html

    def test_connection(self) -> bool:
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
            return True
        except Exception as e:
            print(f"[EMAIL_TEST_ERROR] {e}")
            return False
```

#### 2.4 Push Notification Channel (Pushover)

```python
# alerts/channels/push.py

import requests
from .base import NotificationChannel

class PushoverChannel(NotificationChannel):
    """
    Send push notifications via Pushover.

    Configuration (.env):
        PUSHOVER_USER_KEY=your-user-key
        PUSHOVER_API_TOKEN=your-api-token

    Setup:
        1. Create account at https://pushover.net
        2. Install Pushover app on iOS/Android
        3. Create application token for ATLAS
        4. Add keys to .env file
    """

    API_URL = "https://api.pushover.net/1/messages.json"

    PRIORITY_MAP = {
        'INFO': -1,      # Lowest priority (no sound/vibration)
        'WARNING': 0,    # Normal priority
        'CRITICAL': 1    # High priority (bypass quiet hours)
    }

    def __init__(self, user_key: str, api_token: str):
        self.user_key = user_key
        self.api_token = api_token

    def send(self, alert: 'Alert') -> bool:
        try:
            priority = self.PRIORITY_MAP.get(alert.severity.name, 0)

            payload = {
                'token': self.api_token,
                'user': self.user_key,
                'title': alert.title,
                'message': alert.message,
                'priority': priority,
                'timestamp': int(alert.timestamp.timestamp())
            }

            # High priority requires retry/expire params
            if priority >= 1:
                payload['retry'] = 60    # Retry every 60 seconds
                payload['expire'] = 300  # Stop after 5 minutes

            response = requests.post(self.API_URL, data=payload, timeout=10)
            response.raise_for_status()
            return True

        except Exception as e:
            print(f"[PUSHOVER_ERROR] Failed to send: {e}")
            return False

    def test_connection(self) -> bool:
        try:
            response = requests.post(
                "https://api.pushover.net/1/users/validate.json",
                data={
                    'token': self.api_token,
                    'user': self.user_key
                },
                timeout=10
            )
            return response.status_code == 200
        except Exception:
            return False
```

#### 2.5 Webhook Channel

```python
# alerts/channels/webhook.py

import requests
import json
from .base import NotificationChannel

class WebhookChannel(NotificationChannel):
    """
    Send alerts to a webhook URL (Discord, Slack, custom).

    Configuration (.env):
        ALERT_WEBHOOK_URL=https://discord.com/api/webhooks/...
        ALERT_WEBHOOK_FORMAT=discord  # or 'slack', 'json'
    """

    def __init__(self, webhook_url: str, format_type: str = 'json'):
        self.webhook_url = webhook_url
        self.format_type = format_type

    def send(self, alert: 'Alert') -> bool:
        try:
            if self.format_type == 'discord':
                payload = self._format_discord(alert)
            elif self.format_type == 'slack':
                payload = self._format_slack(alert)
            else:
                payload = alert.to_dict()

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return True

        except Exception as e:
            print(f"[WEBHOOK_ERROR] Failed to send: {e}")
            return False

    def _format_discord(self, alert: 'Alert') -> dict:
        """Format for Discord webhook."""
        colors = {
            'INFO': 3447003,      # Blue
            'WARNING': 16776960,   # Yellow
            'CRITICAL': 15158332   # Red
        }

        return {
            'embeds': [{
                'title': alert.title,
                'description': alert.message,
                'color': colors.get(alert.severity.name, 0),
                'timestamp': alert.timestamp.isoformat(),
                'fields': [
                    {'name': 'Type', 'value': alert.alert_type.value, 'inline': True},
                    {'name': 'Severity', 'value': alert.severity.name, 'inline': True},
                    {'name': 'Symbol', 'value': alert.symbol or 'N/A', 'inline': True}
                ]
            }]
        }

    def _format_slack(self, alert: 'Alert') -> dict:
        """Format for Slack webhook."""
        return {
            'text': f"*{alert.title}*",
            'attachments': [{
                'color': {'INFO': 'good', 'WARNING': 'warning', 'CRITICAL': 'danger'}.get(alert.severity.name, '#333'),
                'text': alert.message,
                'fields': [
                    {'title': 'Type', 'value': alert.alert_type.value, 'short': True},
                    {'title': 'Symbol', 'value': alert.symbol or 'N/A', 'short': True}
                ],
                'ts': int(alert.timestamp.timestamp())
            }]
        }

    def test_connection(self) -> bool:
        try:
            # Send a test message
            response = requests.post(
                self.webhook_url,
                json={'content': 'ATLAS Alert System Test'},
                timeout=10
            )
            return response.status_code in [200, 204]
        except Exception:
            return False
```

---

### 3. Portfolio Monitor

**Location:** `alerts/portfolio_monitor.py`

**Purpose:** Track positions and generate P/L alerts

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from decimal import Decimal

@dataclass
class PositionSnapshot:
    """Snapshot of a single position."""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    entry_time: Optional[datetime] = None

    @property
    def at_risk(self) -> bool:
        """Check if position is approaching stop."""
        if self.stop_price is None:
            return False
        if self.quantity > 0:  # Long
            return self.current_price <= self.stop_price * 1.02  # Within 2%
        else:  # Short
            return self.current_price >= self.stop_price * 0.98

    @property
    def near_target(self) -> bool:
        """Check if position is approaching target."""
        if self.target_price is None:
            return False
        if self.quantity > 0:  # Long
            return self.current_price >= self.target_price * 0.98  # Within 2%
        else:  # Short
            return self.current_price <= self.target_price * 1.02

@dataclass
class PortfolioSnapshot:
    """Snapshot of entire portfolio."""
    timestamp: datetime
    total_equity: float
    cash: float
    buying_power: float
    positions: Dict[str, PositionSnapshot] = field(default_factory=dict)
    total_unrealized_pnl: float = 0.0
    total_unrealized_pnl_pct: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0

    @property
    def position_count(self) -> int:
        return len(self.positions)

    @property
    def deployed_pct(self) -> float:
        """Percentage of capital deployed."""
        if self.total_equity <= 0:
            return 0.0
        return (self.total_equity - self.cash) / self.total_equity * 100

class PortfolioMonitor:
    """
    Monitors portfolio and generates alerts for significant events.

    Events tracked:
    - Position P/L changes > threshold
    - Stop price approached/hit
    - Target price approached/hit
    - Daily P/L summary
    - Portfolio concentration warnings
    """

    def __init__(
        self,
        alert_engine: 'AlertEngine',
        pnl_alert_threshold_pct: float = 5.0,
        concentration_threshold_pct: float = 25.0
    ):
        self.alert_engine = alert_engine
        self.pnl_alert_threshold = pnl_alert_threshold_pct
        self.concentration_threshold = concentration_threshold_pct
        self.previous_snapshot: Optional[PortfolioSnapshot] = None
        self._setup_rules()

    def _setup_rules(self) -> None:
        """Register alert rules for portfolio monitoring."""

        # Stop hit alert
        self.alert_engine.register_rule(AlertRule(
            name="Stop Hit",
            condition=lambda e: e.get('event_type') == 'stop_hit',
            alert_type=AlertType.STOP_HIT,
            severity=AlertSeverity.CRITICAL,
            message_template="STOP HIT: {symbol} at ${current_price:.2f} (stop was ${stop_price:.2f}). Loss: ${loss:.2f} ({loss_pct:.1f}%)",
            channels=['console', 'email', 'push'],
            cooldown_seconds=0  # Always alert on stops
        ))

        # Target hit alert
        self.alert_engine.register_rule(AlertRule(
            name="Target Hit",
            condition=lambda e: e.get('event_type') == 'target_hit',
            alert_type=AlertType.TARGET_HIT,
            severity=AlertSeverity.INFO,
            message_template="TARGET HIT: {symbol} at ${current_price:.2f} (target was ${target_price:.2f}). Profit: ${profit:.2f} ({profit_pct:.1f}%)",
            channels=['console', 'email', 'push'],
            cooldown_seconds=0
        ))

        # Large P/L change alert
        self.alert_engine.register_rule(AlertRule(
            name="Large P/L Change",
            condition=lambda e: e.get('event_type') == 'pnl_change' and abs(e.get('pnl_change_pct', 0)) >= self.pnl_alert_threshold,
            alert_type=AlertType.POSITION_UPDATE,
            severity=AlertSeverity.WARNING,
            message_template="{symbol}: P/L changed by {pnl_change_pct:+.1f}% (${pnl_change:+.2f}). Current P/L: {total_pnl_pct:+.1f}%",
            channels=['console', 'email'],
            cooldown_seconds=3600  # Max once per hour per position
        ))

        # Approaching stop warning
        self.alert_engine.register_rule(AlertRule(
            name="Approaching Stop",
            condition=lambda e: e.get('event_type') == 'approaching_stop',
            alert_type=AlertType.POSITION_UPDATE,
            severity=AlertSeverity.WARNING,
            message_template="WARNING: {symbol} approaching stop. Current: ${current_price:.2f}, Stop: ${stop_price:.2f} ({distance_pct:.1f}% away)",
            channels=['console', 'push'],
            cooldown_seconds=1800  # Max once per 30 min
        ))

    def update(self, snapshot: PortfolioSnapshot) -> List['Alert']:
        """
        Update portfolio state and check for alert conditions.

        Args:
            snapshot: Current portfolio snapshot

        Returns:
            List of generated alerts
        """
        alerts = []

        for symbol, position in snapshot.positions.items():
            # Check stop/target hits
            if position.stop_price and position.current_price <= position.stop_price:
                alerts.extend(self.alert_engine.process_event({
                    'event_type': 'stop_hit',
                    'symbol': symbol,
                    'current_price': position.current_price,
                    'stop_price': position.stop_price,
                    'loss': abs(position.unrealized_pnl),
                    'loss_pct': abs(position.unrealized_pnl_pct)
                }))

            elif position.target_price and position.current_price >= position.target_price:
                alerts.extend(self.alert_engine.process_event({
                    'event_type': 'target_hit',
                    'symbol': symbol,
                    'current_price': position.current_price,
                    'target_price': position.target_price,
                    'profit': position.unrealized_pnl,
                    'profit_pct': position.unrealized_pnl_pct
                }))

            elif position.at_risk:
                distance = abs(position.current_price - position.stop_price) / position.stop_price * 100
                alerts.extend(self.alert_engine.process_event({
                    'event_type': 'approaching_stop',
                    'symbol': symbol,
                    'current_price': position.current_price,
                    'stop_price': position.stop_price,
                    'distance_pct': distance
                }))

            # Check P/L changes vs previous snapshot
            if self.previous_snapshot and symbol in self.previous_snapshot.positions:
                prev_pos = self.previous_snapshot.positions[symbol]
                pnl_change = position.unrealized_pnl - prev_pos.unrealized_pnl
                pnl_change_pct = position.unrealized_pnl_pct - prev_pos.unrealized_pnl_pct

                if abs(pnl_change_pct) >= self.pnl_alert_threshold:
                    alerts.extend(self.alert_engine.process_event({
                        'event_type': 'pnl_change',
                        'symbol': symbol,
                        'pnl_change': pnl_change,
                        'pnl_change_pct': pnl_change_pct,
                        'total_pnl_pct': position.unrealized_pnl_pct
                    }))

        self.previous_snapshot = snapshot
        return alerts

    def generate_summary(self, snapshot: PortfolioSnapshot) -> 'Alert':
        """Generate daily portfolio summary alert."""
        position_lines = []
        for symbol, pos in sorted(snapshot.positions.items(), key=lambda x: x[1].unrealized_pnl, reverse=True):
            position_lines.append(
                f"  {symbol}: ${pos.market_value:,.0f} ({pos.unrealized_pnl_pct:+.1f}%)"
            )

        message = f"""
Daily Portfolio Summary
=======================
Total Equity: ${snapshot.total_equity:,.2f}
Cash: ${snapshot.cash:,.2f}
Deployed: {snapshot.deployed_pct:.1f}%

Daily P/L: ${snapshot.daily_pnl:+,.2f} ({snapshot.daily_pnl_pct:+.2f}%)
Unrealized P/L: ${snapshot.total_unrealized_pnl:+,.2f} ({snapshot.total_unrealized_pnl_pct:+.2f}%)

Positions ({snapshot.position_count}):
{chr(10).join(position_lines) if position_lines else '  (No positions)'}
""".strip()

        return Alert(
            alert_type=AlertType.PORTFOLIO_SUMMARY,
            severity=AlertSeverity.INFO,
            title="Daily Portfolio Summary",
            message=message,
            data={
                'equity': snapshot.total_equity,
                'daily_pnl': snapshot.daily_pnl,
                'position_count': snapshot.position_count
            }
        )
```

---

### 4. Pattern Alert Integration

**Location:** `alerts/pattern_alerts.py`

**Purpose:** Generate alerts when STRAT patterns are detected

```python
from typing import List
from strat.tier1_detector import PatternSignal, PatternType, Timeframe

class PatternAlertIntegration:
    """
    Integrates pattern detection with alert system.

    Generates alerts for:
    - New Tier 1 patterns detected
    - High-confidence patterns (confluence with regime)
    - Pattern entry triggers hit
    """

    def __init__(self, alert_engine: 'AlertEngine'):
        self.alert_engine = alert_engine
        self._setup_rules()

    def _setup_rules(self) -> None:
        """Register pattern-related alert rules."""

        # New pattern detected
        self.alert_engine.register_rule(AlertRule(
            name="Pattern Detected",
            condition=lambda e: e.get('event_type') == 'pattern_detected',
            alert_type=AlertType.PATTERN_DETECTED,
            severity=AlertSeverity.INFO,
            message_template="New {pattern_type} pattern on {symbol} ({timeframe}). Entry: ${entry_price:.2f}, Target: ${target_price:.2f}, R:R {risk_reward:.1f}:1",
            channels=['console', 'email'],
            cooldown_seconds=300  # Max once per 5 min per pattern type
        ))

        # High-confidence pattern (with regime alignment)
        self.alert_engine.register_rule(AlertRule(
            name="High Confidence Pattern",
            condition=lambda e: e.get('event_type') == 'pattern_detected' and e.get('signal_quality') == 'HIGH',
            alert_type=AlertType.PATTERN_DETECTED,
            severity=AlertSeverity.WARNING,  # Higher priority
            message_template="HIGH CONFIDENCE: {pattern_type} on {symbol} ({timeframe}). Regime: {regime}, Continuity: {continuity_score:.0%}. Entry: ${entry_price:.2f}",
            channels=['console', 'email', 'push'],
            cooldown_seconds=0  # Always alert on high confidence
        ))

    def process_signals(
        self,
        signals: List[PatternSignal],
        symbol: str,
        regime: str = None,
        continuity_score: float = None
    ) -> List['Alert']:
        """
        Process pattern signals and generate alerts.

        Args:
            signals: List of detected PatternSignal objects
            symbol: Underlying symbol
            regime: Current ATLAS regime (optional)
            continuity_score: Timeframe continuity score (optional)

        Returns:
            List of generated alerts
        """
        alerts = []

        for signal in signals:
            # Determine signal quality if regime provided
            signal_quality = None
            if regime:
                signal_quality = self._evaluate_signal_quality(signal, regime, continuity_score)

            event = {
                'event_type': 'pattern_detected',
                'symbol': symbol,
                'pattern_type': signal.pattern_type.value,
                'timeframe': signal.timeframe.value,
                'direction': 'BULL' if signal.direction == 1 else 'BEAR',
                'entry_price': signal.entry_price,
                'target_price': signal.target_price,
                'stop_price': signal.stop_price,
                'risk_reward': signal.risk_reward,
                'continuation_bars': signal.continuation_bars,
                'signal_quality': signal_quality,
                'regime': regime,
                'continuity_score': continuity_score
            }

            alerts.extend(self.alert_engine.process_event(event))

        return alerts

    def _evaluate_signal_quality(
        self,
        signal: PatternSignal,
        regime: str,
        continuity_score: float
    ) -> str:
        """
        Evaluate signal quality based on regime alignment.

        Returns: 'HIGH', 'MEDIUM', or 'REJECT'
        """
        is_bullish = signal.direction == 1

        # CRASH regime veto
        if regime == 'CRASH' and is_bullish:
            return 'REJECT'

        # Perfect alignment
        if regime == 'TREND_BULL' and is_bullish:
            if continuity_score and continuity_score >= 0.67:
                return 'HIGH'
            return 'MEDIUM'

        if regime == 'TREND_BEAR' and not is_bullish:
            if continuity_score and continuity_score >= 0.67:
                return 'HIGH'
            return 'MEDIUM'

        # Neutral regime
        if regime == 'TREND_NEUTRAL':
            return 'MEDIUM'

        # Counter-trend
        return 'REJECT'
```

---

### 5. Configuration

**Location:** `config/alert_settings.py`

```python
"""
Alert system configuration.

Add these to your .env file:

# Email Configuration
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_SMTP_HOST=smtp.gmail.com
ALERT_EMAIL_SMTP_PORT=587
ALERT_EMAIL_USERNAME=your-email@gmail.com
ALERT_EMAIL_PASSWORD=your-app-password
ALERT_EMAIL_FROM=your-email@gmail.com
ALERT_EMAIL_TO=recipient1@example.com,recipient2@example.com

# Pushover Configuration
ALERT_PUSH_ENABLED=true
PUSHOVER_USER_KEY=your-user-key
PUSHOVER_API_TOKEN=your-api-token

# Webhook Configuration (Discord/Slack)
ALERT_WEBHOOK_ENABLED=false
ALERT_WEBHOOK_URL=https://discord.com/api/webhooks/...
ALERT_WEBHOOK_FORMAT=discord

# Alert Thresholds
ALERT_PNL_THRESHOLD_PCT=5.0
ALERT_CONCENTRATION_THRESHOLD_PCT=25.0
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

def get_alert_config() -> dict:
    """Load alert configuration from environment."""
    return {
        'email': {
            'enabled': os.getenv('ALERT_EMAIL_ENABLED', 'false').lower() == 'true',
            'smtp_host': os.getenv('ALERT_EMAIL_SMTP_HOST', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('ALERT_EMAIL_SMTP_PORT', '587')),
            'username': os.getenv('ALERT_EMAIL_USERNAME', ''),
            'password': os.getenv('ALERT_EMAIL_PASSWORD', ''),
            'from_addr': os.getenv('ALERT_EMAIL_FROM', ''),
            'to_addrs': os.getenv('ALERT_EMAIL_TO', '').split(',')
        },
        'push': {
            'enabled': os.getenv('ALERT_PUSH_ENABLED', 'false').lower() == 'true',
            'user_key': os.getenv('PUSHOVER_USER_KEY', ''),
            'api_token': os.getenv('PUSHOVER_API_TOKEN', '')
        },
        'webhook': {
            'enabled': os.getenv('ALERT_WEBHOOK_ENABLED', 'false').lower() == 'true',
            'url': os.getenv('ALERT_WEBHOOK_URL', ''),
            'format': os.getenv('ALERT_WEBHOOK_FORMAT', 'json')
        },
        'thresholds': {
            'pnl_alert_pct': float(os.getenv('ALERT_PNL_THRESHOLD_PCT', '5.0')),
            'concentration_pct': float(os.getenv('ALERT_CONCENTRATION_THRESHOLD_PCT', '25.0'))
        }
    }

def create_alert_engine() -> 'AlertEngine':
    """Factory function to create configured AlertEngine."""
    from alerts.alert_engine import AlertEngine
    from alerts.channels.console import ConsoleChannel
    from alerts.channels.email import EmailChannel
    from alerts.channels.push import PushoverChannel
    from alerts.channels.webhook import WebhookChannel

    config = get_alert_config()
    engine = AlertEngine(config)

    # Always register console
    engine.register_channel('console', ConsoleChannel())

    # Register email if enabled
    if config['email']['enabled']:
        engine.register_channel('email', EmailChannel(
            smtp_host=config['email']['smtp_host'],
            smtp_port=config['email']['smtp_port'],
            username=config['email']['username'],
            password=config['email']['password'],
            from_addr=config['email']['from_addr'],
            to_addrs=config['email']['to_addrs']
        ))

    # Register push if enabled
    if config['push']['enabled']:
        engine.register_channel('push', PushoverChannel(
            user_key=config['push']['user_key'],
            api_token=config['push']['api_token']
        ))

    # Register webhook if enabled
    if config['webhook']['enabled']:
        engine.register_channel('webhook', WebhookChannel(
            webhook_url=config['webhook']['url'],
            format_type=config['webhook']['format']
        ))

    return engine
```

---

## Directory Structure

```
alerts/
├── __init__.py
├── alert_engine.py          # Core AlertEngine class
├── portfolio_monitor.py     # PortfolioMonitor class
├── pattern_alerts.py        # PatternAlertIntegration class
└── channels/
    ├── __init__.py
    ├── base.py              # NotificationChannel ABC
    ├── console.py           # ConsoleChannel
    ├── email.py             # EmailChannel (SMTP)
    ├── push.py              # PushoverChannel
    └── webhook.py           # WebhookChannel (Discord/Slack)

config/
└── alert_settings.py        # Configuration and factory

scripts/
└── monitor_portfolio.py     # CLI script for real-time monitoring
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure (Session 1)
- [ ] Create `alerts/` directory structure
- [ ] Implement `AlertEngine` class
- [ ] Implement `ConsoleChannel`
- [ ] Implement `Alert` and `AlertRule` dataclasses
- [ ] Add basic tests for AlertEngine

### Phase 2: Notification Channels (Session 2)
- [ ] Implement `EmailChannel` with SMTP
- [ ] Implement `PushoverChannel`
- [ ] Implement `WebhookChannel`
- [ ] Add configuration to `config/alert_settings.py`
- [ ] Add `.env.template` updates

### Phase 3: Integration (Session 3)
- [ ] Implement `PortfolioMonitor`
- [ ] Implement `PatternAlertIntegration`
- [ ] Create `scripts/monitor_portfolio.py`
- [ ] Integration tests with existing OptionsExecutor
- [ ] Documentation updates

---

## Testing Requirements

### Unit Tests

```python
# tests/test_alerts/test_alert_engine.py

def test_alert_rule_condition():
    """Test that alert rules correctly evaluate conditions."""
    rule = AlertRule(
        name="Test Rule",
        condition=lambda e: e.get('value') > 10,
        alert_type=AlertType.POSITION_UPDATE,
        severity=AlertSeverity.INFO,
        message_template="Value is {value}"
    )

    assert rule.condition({'value': 15}) == True
    assert rule.condition({'value': 5}) == False

def test_alert_cooldown():
    """Test that cooldown prevents spam."""
    engine = AlertEngine()
    engine.register_channel('console', ConsoleChannel())
    engine.register_rule(AlertRule(
        name="Cooldown Test",
        condition=lambda e: True,
        alert_type=AlertType.INFO,
        severity=AlertSeverity.INFO,
        message_template="Test",
        cooldown_seconds=60
    ))

    # First alert should fire
    alerts1 = engine.process_event({})
    assert len(alerts1) == 1

    # Second alert within cooldown should not fire
    alerts2 = engine.process_event({})
    assert len(alerts2) == 0
```

### Integration Tests

```python
# tests/test_alerts/test_integration.py

def test_pattern_to_alert_flow():
    """Test full flow from pattern detection to alert."""
    from strat.tier1_detector import Tier1Detector, Timeframe
    from alerts.pattern_alerts import PatternAlertIntegration

    # Detect patterns
    detector = Tier1Detector()
    signals = detector.detect_patterns(test_data, timeframe=Timeframe.WEEKLY)

    # Generate alerts
    engine = AlertEngine()
    engine.register_channel('console', ConsoleChannel())
    integration = PatternAlertIntegration(engine)

    alerts = integration.process_signals(signals, symbol='SPY', regime='TREND_BULL')

    # Verify alerts generated
    assert len(alerts) > 0
    assert all(a.alert_type == AlertType.PATTERN_DETECTED for a in alerts)
```

---

## Technology Recommendations

| Component | Recommended | Alternative | Rationale |
|-----------|-------------|-------------|-----------|
| Email | SMTP (Gmail) | SendGrid API | SMTP is simpler, no extra dependency |
| Push | Pushover | Telegram Bot | Pushover has better iOS support |
| Webhook | Discord | Slack | Discord webhooks are free, no app approval |
| Queue | threading.Queue | Redis | No extra infrastructure needed |
| Scheduling | APScheduler | Celery | Lightweight, no broker needed |

---

## Future Enhancements

1. **Web Dashboard** - Flask/FastAPI with real-time WebSocket updates
2. **SMS Alerts** - Twilio integration for critical alerts
3. **Alert Analytics** - Track alert frequency, response times
4. **Smart Batching** - Combine multiple alerts into single notification
5. **Escalation Policies** - Re-alert if no acknowledgment

---

## Handoff Notes

**For implementing Claude instance:**

1. Start with Phase 1 (Core Infrastructure) - this is self-contained
2. Use `config/settings.py` pattern for credentials (already established)
3. Follow existing code style in `strat/` and `integrations/`
4. Run existing 141 tests after implementation to verify no regressions
5. Add new tests to `tests/test_alerts/`

**Dependencies to add to pyproject.toml:**
```toml
dependencies = [
    # ... existing ...
    "requests>=2.28.0",  # For Pushover/Webhook (likely already present)
]
```

**No new major dependencies required** - uses standard library (smtplib, email) for email.

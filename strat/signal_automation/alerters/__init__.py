"""
STRAT Signal Alerters - Session 83K-45

Multi-channel alert delivery for detected STRAT signals.

Supported Channels:
- Discord: Primary alert channel via webhook (instant, rich formatting)
- Logging: Structured JSON logging (always active, audit trail)
- Email: Optional for daily summaries (future)

Design:
- Abstract base class for pluggable alerters
- Throttling to prevent alert spam
- Async delivery for non-blocking operation
"""

from strat.signal_automation.alerters.base import BaseAlerter
from strat.signal_automation.alerters.logging_alerter import LoggingAlerter
from strat.signal_automation.alerters.discord_alerter import DiscordAlerter

__all__ = [
    'BaseAlerter',
    'LoggingAlerter',
    'DiscordAlerter',
]

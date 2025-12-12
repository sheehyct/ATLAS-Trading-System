"""
STRAT Signal Scheduler - Session 83K-46

APScheduler-based job scheduling for automated signal scanning.
Market-hours aware with timezone support.

Features:
- Cron-based scheduling per timeframe
- Market hours awareness (9:30-16:00 ET)
- Graceful job management
- Health monitoring
"""

import logging
from datetime import datetime, time
from typing import Optional, Callable, Dict, Any, List

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import (
    EVENT_JOB_EXECUTED,
    EVENT_JOB_ERROR,
    EVENT_JOB_MISSED,
    JobEvent,
)
import pytz

from strat.signal_automation.config import ScheduleConfig

logger = logging.getLogger(__name__)


class SignalScheduler:
    """
    APScheduler-based job scheduler for signal scanning.

    Manages scheduled jobs for:
    - Hourly scans (during market hours)
    - Daily scans (after market close)
    - Weekly scans (Friday after close)
    - Monthly scans (last trading day)

    Features:
    - Market hours awareness
    - Timezone-safe scheduling (America/New_York)
    - Job persistence (optional)
    - Health monitoring
    - Graceful shutdown

    Usage:
        scheduler = SignalScheduler()
        scheduler.add_hourly_job(scan_hourly_callback)
        scheduler.add_daily_job(scan_daily_callback)
        scheduler.start()
        # ... later ...
        scheduler.shutdown()
    """

    # Market hours in ET
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)

    def __init__(
        self,
        config: Optional[ScheduleConfig] = None,
        timezone: str = 'America/New_York'
    ):
        """
        Initialize signal scheduler.

        Args:
            config: Schedule configuration
            timezone: Timezone for scheduling (default: America/New_York)
        """
        self.config = config or ScheduleConfig()
        self.timezone = pytz.timezone(timezone)

        # Create scheduler with timezone
        self._scheduler = BackgroundScheduler(
            timezone=self.timezone,
            job_defaults={
                'coalesce': True,  # Combine missed jobs
                'max_instances': 1,  # Only one instance per job
                'misfire_grace_time': self.config.misfire_grace_time,
            }
        )

        # Job tracking
        self._jobs: Dict[str, str] = {}  # name -> job_id
        self._job_stats: Dict[str, Dict[str, Any]] = {}
        self._is_running = False

        # Add event listeners
        self._scheduler.add_listener(
            self._on_job_executed,
            EVENT_JOB_EXECUTED
        )
        self._scheduler.add_listener(
            self._on_job_error,
            EVENT_JOB_ERROR
        )
        self._scheduler.add_listener(
            self._on_job_missed,
            EVENT_JOB_MISSED
        )

    def _on_job_executed(self, event: JobEvent) -> None:
        """Handle successful job execution."""
        job_id = event.job_id
        if job_id in self._job_stats:
            self._job_stats[job_id]['last_run'] = datetime.now()
            self._job_stats[job_id]['run_count'] += 1
            self._job_stats[job_id]['last_status'] = 'success'
        logger.info(f"Job executed: {job_id}")

    def _on_job_error(self, event: JobEvent) -> None:
        """Handle job execution error."""
        job_id = event.job_id
        if job_id in self._job_stats:
            self._job_stats[job_id]['last_run'] = datetime.now()
            self._job_stats[job_id]['error_count'] += 1
            self._job_stats[job_id]['last_status'] = 'error'
            self._job_stats[job_id]['last_error'] = str(event.exception)
        logger.error(f"Job error: {job_id} - {event.exception}")

    def _on_job_missed(self, event: JobEvent) -> None:
        """Handle missed job."""
        job_id = event.job_id
        if job_id in self._job_stats:
            self._job_stats[job_id]['missed_count'] += 1
            self._job_stats[job_id]['last_status'] = 'missed'
        logger.warning(f"Job missed: {job_id}")

    def _parse_cron(self, cron_expr: str) -> Dict[str, Any]:
        """
        Parse cron expression to APScheduler trigger kwargs.

        Format: minute hour day month day_of_week

        Args:
            cron_expr: Cron expression string

        Returns:
            Dictionary for CronTrigger
        """
        parts = cron_expr.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {cron_expr}")

        return {
            'minute': parts[0],
            'hour': parts[1],
            'day': parts[2],
            'month': parts[3],
            'day_of_week': parts[4],
        }

    def add_hourly_job(
        self,
        callback: Callable,
        job_id: str = 'scan_hourly'
    ) -> Optional[str]:
        """
        Add hourly scan job during market hours.

        Runs at :30 each hour from 9:30 to 15:30 ET, Mon-Fri.

        Args:
            callback: Function to call on trigger
            job_id: Unique job identifier

        Returns:
            Job ID if added, None if disabled
        """
        if not self.config.scan_hourly:
            logger.info("Hourly scanning disabled in config")
            return None

        cron_kwargs = self._parse_cron(self.config.hourly_cron)

        job = self._scheduler.add_job(
            callback,
            trigger=CronTrigger(**cron_kwargs, timezone=self.timezone),
            id=job_id,
            name='Hourly Signal Scan',
            replace_existing=True,
        )

        self._jobs['hourly'] = job.id
        self._job_stats[job.id] = {
            'name': 'Hourly Scan',
            'run_count': 0,
            'error_count': 0,
            'missed_count': 0,
            'last_run': None,
            'last_status': 'pending',
            'last_error': None,
        }

        logger.info(f"Added hourly job: {job.id}")
        return job.id

    # =========================================================================
    # Session 83K-80: 15-Minute Base Scan (HTF Resampling Architecture)
    # =========================================================================

    def add_base_scan_job(
        self,
        callback: Callable,
        job_id: str = 'scan_base'
    ) -> Optional[str]:
        """
        Add 15-minute base scan job for unified multi-TF scanning.

        Session 83K-80: HTF Scanning Architecture Fix.

        This job runs every 15 minutes during market hours and:
        - Fetches 15-min data once per symbol
        - Resamples to 1H, 1D, 1W, 1M
        - Detects SETUP patterns on ALL timeframes

        Replaces separate hourly/daily/weekly/monthly jobs when
        enable_htf_resampling=True.

        Schedule: :30, :45, :00, :15 of each hour (9:30-15:45 ET)

        Args:
            callback: Function to call on trigger
            job_id: Unique job identifier

        Returns:
            Job ID if added, None if HTF resampling disabled
        """
        if not self.config.enable_htf_resampling:
            logger.info("HTF resampling disabled - use legacy scan jobs instead")
            return None

        cron_kwargs = self._parse_cron(self.config.base_scan_cron)

        job = self._scheduler.add_job(
            callback,
            trigger=CronTrigger(**cron_kwargs, timezone=self.timezone),
            id=job_id,
            name='15-Min Multi-TF Scan (Resampled)',
            replace_existing=True,
        )

        self._jobs['base'] = job.id
        self._job_stats[job.id] = {
            'name': '15-Min Multi-TF Scan',
            'run_count': 0,
            'error_count': 0,
            'missed_count': 0,
            'last_run': None,
            'last_status': 'pending',
            'last_error': None,
        }

        logger.info(f"Added 15-min base scan job: {job.id} (HTF resampling enabled)")
        return job.id

    def add_daily_job(
        self,
        callback: Callable,
        job_id: str = 'scan_daily'
    ) -> Optional[str]:
        """
        Add daily scan job after market close.

        Runs at 5 PM ET, Mon-Fri.

        Args:
            callback: Function to call on trigger
            job_id: Unique job identifier

        Returns:
            Job ID if added, None if disabled
        """
        if not self.config.scan_daily:
            logger.info("Daily scanning disabled in config")
            return None

        cron_kwargs = self._parse_cron(self.config.daily_cron)

        job = self._scheduler.add_job(
            callback,
            trigger=CronTrigger(**cron_kwargs, timezone=self.timezone),
            id=job_id,
            name='Daily Signal Scan',
            replace_existing=True,
        )

        self._jobs['daily'] = job.id
        self._job_stats[job.id] = {
            'name': 'Daily Scan',
            'run_count': 0,
            'error_count': 0,
            'missed_count': 0,
            'last_run': None,
            'last_status': 'pending',
            'last_error': None,
        }

        logger.info(f"Added daily job: {job.id}")
        return job.id

    def add_weekly_job(
        self,
        callback: Callable,
        job_id: str = 'scan_weekly'
    ) -> Optional[str]:
        """
        Add weekly scan job on Friday after close.

        Runs at 6 PM ET on Friday.

        Args:
            callback: Function to call on trigger
            job_id: Unique job identifier

        Returns:
            Job ID if added, None if disabled
        """
        if not self.config.scan_weekly:
            logger.info("Weekly scanning disabled in config")
            return None

        cron_kwargs = self._parse_cron(self.config.weekly_cron)

        job = self._scheduler.add_job(
            callback,
            trigger=CronTrigger(**cron_kwargs, timezone=self.timezone),
            id=job_id,
            name='Weekly Signal Scan',
            replace_existing=True,
        )

        self._jobs['weekly'] = job.id
        self._job_stats[job.id] = {
            'name': 'Weekly Scan',
            'run_count': 0,
            'error_count': 0,
            'missed_count': 0,
            'last_run': None,
            'last_status': 'pending',
            'last_error': None,
        }

        logger.info(f"Added weekly job: {job.id}")
        return job.id

    def add_monthly_job(
        self,
        callback: Callable,
        job_id: str = 'scan_monthly'
    ) -> Optional[str]:
        """
        Add monthly scan job on last day of month.

        Runs at 6 PM ET on last day of month.

        Args:
            callback: Function to call on trigger
            job_id: Unique job identifier

        Returns:
            Job ID if added, None if disabled
        """
        if not self.config.scan_monthly:
            logger.info("Monthly scanning disabled in config")
            return None

        cron_kwargs = self._parse_cron(self.config.monthly_cron)

        job = self._scheduler.add_job(
            callback,
            trigger=CronTrigger(**cron_kwargs, timezone=self.timezone),
            id=job_id,
            name='Monthly Signal Scan',
            replace_existing=True,
        )

        self._jobs['monthly'] = job.id
        self._job_stats[job.id] = {
            'name': 'Monthly Scan',
            'run_count': 0,
            'error_count': 0,
            'missed_count': 0,
            'last_run': None,
            'last_status': 'pending',
            'last_error': None,
        }

        logger.info(f"Added monthly job: {job.id}")
        return job.id

    def add_health_check_job(
        self,
        callback: Callable,
        interval_seconds: int = 300,
        job_id: str = 'health_check'
    ) -> str:
        """
        Add periodic health check job.

        Args:
            callback: Health check function
            interval_seconds: Check interval in seconds
            job_id: Unique job identifier

        Returns:
            Job ID
        """
        job = self._scheduler.add_job(
            callback,
            trigger='interval',
            seconds=interval_seconds,
            id=job_id,
            name='Health Check',
            replace_existing=True,
        )

        self._jobs['health_check'] = job.id
        logger.info(f"Added health check job: {job.id}")
        return job.id

    def add_interval_job(
        self,
        callback: Callable,
        interval_seconds: int,
        job_id: str,
        job_name: Optional[str] = None
    ) -> str:
        """
        Add periodic interval job (Session 83K-49).

        Args:
            callback: Function to call on interval
            interval_seconds: Interval in seconds
            job_id: Unique job identifier
            job_name: Human-readable job name

        Returns:
            Job ID
        """
        job = self._scheduler.add_job(
            callback,
            trigger='interval',
            seconds=interval_seconds,
            id=job_id,
            name=job_name or job_id,
            replace_existing=True,
        )

        self._jobs[job_id] = job.id
        self._job_stats[job.id] = {
            'name': job_name or job_id,
            'run_count': 0,
            'error_count': 0,
            'missed_count': 0,
            'last_run': None,
            'last_status': 'pending',
            'last_error': None,
        }

        logger.info(f"Added interval job: {job.id} (every {interval_seconds}s)")
        return job.id

    def run_job_now(self, job_name: str) -> bool:
        """
        Manually trigger a job immediately.

        Args:
            job_name: Name of job to run (hourly, daily, weekly, monthly)

        Returns:
            True if job triggered
        """
        if job_name not in self._jobs:
            logger.error(f"Job not found: {job_name}")
            return False

        job_id = self._jobs[job_name]
        job = self._scheduler.get_job(job_id)

        if job:
            job.modify(next_run_time=datetime.now(self.timezone))
            logger.info(f"Triggered job: {job_name}")
            return True

        logger.error(f"Job not found in scheduler: {job_id}")
        return False

    def is_market_hours(self) -> bool:
        """
        Check if currently in US market hours.

        Returns:
            True if market is open (9:30-16:00 ET, Mon-Fri)
        """
        now = datetime.now(self.timezone)

        # Check weekday (0=Monday, 6=Sunday)
        if now.weekday() >= 5:
            return False

        # Check time
        current_time = now.time()
        return self.MARKET_OPEN <= current_time <= self.MARKET_CLOSE

    def start(self) -> None:
        """Start the scheduler."""
        if self._is_running:
            logger.warning("Scheduler already running")
            return

        self._scheduler.start()
        self._is_running = True
        logger.info("Scheduler started")

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the scheduler gracefully.

        Args:
            wait: Wait for running jobs to complete
        """
        if not self._is_running:
            logger.warning("Scheduler not running")
            return

        self._scheduler.shutdown(wait=wait)
        self._is_running = False
        logger.info("Scheduler shutdown complete")

    def pause(self) -> None:
        """Pause all scheduled jobs."""
        self._scheduler.pause()
        logger.info("Scheduler paused")

    def resume(self) -> None:
        """Resume all scheduled jobs."""
        self._scheduler.resume()
        logger.info("Scheduler resumed")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._is_running

    def get_next_run_times(self) -> Dict[str, Optional[datetime]]:
        """
        Get next run time for each job.

        Returns:
            Dictionary of job_name -> next_run_time
        """
        result = {}
        for name, job_id in self._jobs.items():
            job = self._scheduler.get_job(job_id)
            if job:
                result[name] = job.next_run_time
            else:
                result[name] = None
        return result

    def get_job_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all jobs.

        Returns:
            Dictionary of job_id -> stats
        """
        # Update next run times
        for name, job_id in self._jobs.items():
            if job_id in self._job_stats:
                job = self._scheduler.get_job(job_id)
                if job:
                    self._job_stats[job_id]['next_run'] = job.next_run_time

        return self._job_stats.copy()

    def get_status(self) -> Dict[str, Any]:
        """
        Get overall scheduler status.

        Returns:
            Status dictionary
        """
        return {
            'running': self._is_running,
            'timezone': str(self.timezone),
            'market_hours': self.is_market_hours(),
            'jobs_count': len(self._jobs),
            'jobs': list(self._jobs.keys()),
            'next_runs': {
                name: str(dt) if dt else None
                for name, dt in self.get_next_run_times().items()
            },
        }

    def get_jobs_summary(self) -> List[Dict[str, Any]]:
        """
        Get summary of all scheduled jobs.

        Returns:
            List of job summaries
        """
        summaries = []
        for name, job_id in self._jobs.items():
            job = self._scheduler.get_job(job_id)
            stats = self._job_stats.get(job_id, {})

            summary = {
                'name': name,
                'job_id': job_id,
                'next_run': str(job.next_run_time) if job else None,
                'run_count': stats.get('run_count', 0),
                'error_count': stats.get('error_count', 0),
                'last_status': stats.get('last_status', 'unknown'),
            }
            summaries.append(summary)

        return summaries

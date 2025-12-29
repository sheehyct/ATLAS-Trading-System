"""
Version Tracker - Embeds code version in every trade for audit traceability.

PURPOSE:
When a fix is deployed, every trade after that fix will carry the commit hash.
This allows you to answer: "Was this trade before or after the fix?"

Usage:
    from utils.version_tracker import get_version_info, VersionInfo

    # Get current version (call once at daemon startup)
    version = get_version_info()

    # Embed in trade record
    trade['code_version'] = version.commit_short
    trade['code_session'] = version.session_id
    trade['code_branch'] = version.branch
"""

import subprocess
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import re

# Cache for version info (computed once per process)
_version_cache: Optional['VersionInfo'] = None


@dataclass
class VersionInfo:
    """Code version metadata to embed in every trade."""

    commit_hash: str          # Full 40-char hash
    commit_short: str         # Short 7-char hash for display
    commit_message: str       # First line of commit message
    commit_timestamp: str     # When commit was made (ISO format)
    branch: str               # Current branch name
    session_id: str           # Extracted session ID (e.g., "EQUITY-35")
    is_dirty: bool            # True if uncommitted changes exist
    tags: list                # Git tags on this commit
    captured_at: str          # When this info was captured (ISO format)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @property
    def version_string(self) -> str:
        """Human-readable version string for logs."""
        dirty_marker = " (dirty)" if self.is_dirty else ""
        return f"{self.commit_short}/{self.session_id}{dirty_marker}"

    @property
    def trade_metadata(self) -> Dict[str, Any]:
        """Minimal metadata to embed in each trade record."""
        return {
            'code_version': self.commit_short,
            'code_session': self.session_id,
            'code_branch': self.branch,
            'code_dirty': self.is_dirty,
            'code_timestamp': self.commit_timestamp,
        }


def _run_git_command(cmd: list, cwd: Optional[str] = None) -> str:
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            ['git'] + cmd,
            capture_output=True,
            text=True,
            cwd=cwd or _get_repo_root(),
            timeout=5
        )
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        return ""


def _get_repo_root() -> str:
    """Get the git repository root directory."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except Exception:
        # Fallback to current working directory
        return os.getcwd()


def _extract_session_id(commit_message: str) -> str:
    """
    Extract session ID from commit message.

    Patterns recognized:
    - (EQUITY-35) -> EQUITY-35
    - (CRYPTO-MONITOR-2) -> CRYPTO-MONITOR-2
    - (SESSION-83K) -> SESSION-83K
    - EQUITY-35: -> EQUITY-35
    """
    # Pattern 1: (SESSION-ID) at end of message
    match = re.search(r'\(([A-Z]+-[\w-]+)\)\s*$', commit_message)
    if match:
        return match.group(1)

    # Pattern 2: SESSION-ID: at start
    match = re.search(r'^([A-Z]+-[\w-]+):', commit_message)
    if match:
        return match.group(1)

    # Pattern 3: Any SESSION-ID pattern in message
    match = re.search(r'([A-Z]+-\d+(?:-\w+)?)', commit_message)
    if match:
        return match.group(1)

    return "UNKNOWN"


def get_version_info(force_refresh: bool = False) -> VersionInfo:
    """
    Get current code version information.

    Results are cached for the lifetime of the process unless force_refresh=True.
    Call this once at daemon startup and pass to all trade recording functions.

    Args:
        force_refresh: If True, re-read from git even if cached

    Returns:
        VersionInfo dataclass with all version metadata
    """
    global _version_cache

    if _version_cache is not None and not force_refresh:
        return _version_cache

    # Get commit hash
    commit_hash = _run_git_command(['rev-parse', 'HEAD'])
    if not commit_hash:
        commit_hash = "0" * 40  # Fallback for non-git environments

    commit_short = commit_hash[:7]

    # Get commit message
    commit_message = _run_git_command(['log', '-1', '--format=%s'])

    # Get commit timestamp
    commit_timestamp = _run_git_command(['log', '-1', '--format=%aI'])

    # Get branch name
    branch = _run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'])
    if not branch:
        branch = "unknown"

    # Check for uncommitted changes
    status = _run_git_command(['status', '--porcelain'])
    is_dirty = bool(status)

    # Get tags
    tags_output = _run_git_command(['tag', '--points-at', 'HEAD'])
    tags = tags_output.split('\n') if tags_output else []

    # Extract session ID
    session_id = _extract_session_id(commit_message)

    # Build version info
    _version_cache = VersionInfo(
        commit_hash=commit_hash,
        commit_short=commit_short,
        commit_message=commit_message,
        commit_timestamp=commit_timestamp or datetime.now().isoformat(),
        branch=branch,
        session_id=session_id,
        is_dirty=is_dirty,
        tags=tags,
        captured_at=datetime.now().isoformat()
    )

    return _version_cache


def get_version_string() -> str:
    """Quick helper to get version string for logging."""
    return get_version_info().version_string


def get_trade_version_metadata() -> Dict[str, Any]:
    """Quick helper to get metadata dict for embedding in trades."""
    return get_version_info().trade_metadata


# Component registry for tracking which components are affected by fixes
COMPONENT_DEPENDENCIES = {
    'pattern_detection': {
        'files': [
            'strat/bar_classifier.py',
            'strat/pattern_detector.py',
            'strat/paper_signal_scanner.py',
        ],
        'affects': ['equity', 'crypto'],
        'description': 'STRAT bar pattern recognition'
    },
    'options_pricing': {
        'files': [
            'strat/options_module.py',
            'strat/greeks.py',
        ],
        'affects': ['equity'],
        'description': 'Options strike selection and Greeks'
    },
    'equity_daemon': {
        'files': [
            'strat/signal_automation/daemon.py',
            'strat/signal_automation/signal_store.py',
            'strat/signal_automation/entry_monitor.py',
            'strat/signal_automation/position_monitor.py',
        ],
        'affects': ['equity'],
        'description': 'Equity/options trading daemon'
    },
    'crypto_daemon': {
        'files': [
            'crypto/scanning/daemon.py',
            'crypto/scanning/signal_scanner.py',
            'crypto/simulation/paper_trader.py',
            'crypto/simulation/position_monitor.py',
        ],
        'affects': ['crypto'],
        'description': 'Crypto perpetual futures daemon'
    },
    'alerting': {
        'files': [
            'strat/signal_automation/alerters/',
            'crypto/alerters/',
        ],
        'affects': ['equity', 'crypto'],
        'description': 'Discord and logging alerts'
    },
    'execution': {
        'files': [
            'strat/signal_automation/executor.py',
            'crypto/exchange/',
        ],
        'affects': ['equity', 'crypto'],
        'description': 'Order execution and exchange integration'
    },
}


def get_affected_components(changed_files: list) -> list:
    """
    Determine which components are affected by file changes.

    Args:
        changed_files: List of changed file paths (from git diff)

    Returns:
        List of affected component names
    """
    affected = set()

    for component, config in COMPONENT_DEPENDENCIES.items():
        for file_pattern in config['files']:
            for changed_file in changed_files:
                if file_pattern in changed_file or changed_file.startswith(file_pattern):
                    affected.add(component)
                    break

    return list(affected)


def get_affected_systems(changed_files: list) -> list:
    """
    Determine which trading systems (equity, crypto) are affected.

    Args:
        changed_files: List of changed file paths

    Returns:
        List of affected systems ['equity', 'crypto']
    """
    affected_components = get_affected_components(changed_files)
    systems = set()

    for component in affected_components:
        if component in COMPONENT_DEPENDENCIES:
            systems.update(COMPONENT_DEPENDENCIES[component]['affects'])

    return list(systems)


if __name__ == '__main__':
    # Quick test
    info = get_version_info()
    print(f"Version: {info.version_string}")
    print(f"Commit: {info.commit_hash}")
    print(f"Message: {info.commit_message}")
    print(f"Session: {info.session_id}")
    print(f"Branch: {info.branch}")
    print(f"Dirty: {info.is_dirty}")
    print(f"\nTrade metadata: {json.dumps(info.trade_metadata, indent=2)}")

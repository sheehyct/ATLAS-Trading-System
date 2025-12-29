"""
Fix Manifest - Structured tracking of code fixes and their expected impact.

PURPOSE:
Track every fix with:
- What was changed
- Which components are affected
- Expected impact on trades
- Before/after trade validation status

Usage:
    from utils.fix_manifest import FixManifest, FixEntry

    manifest = FixManifest()

    # Record a new fix
    manifest.add_fix(
        session_id="EQUITY-35",
        description="Add EOD exit for 1H trades",
        components=["equity_daemon", "position_monitor"],
        affected_systems=["equity"],
        expected_impact="1H trades now exit at market close instead of next day",
        trade_fields_affected=["exit_timestamp", "exit_reason"]
    )

    # Check if a trade happened after a specific fix
    if manifest.trade_after_fix(trade_timestamp, "EQUITY-35"):
        print("Trade includes EOD exit logic")
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import subprocess


MANIFEST_FILE = Path(__file__).parent.parent / 'data' / 'fix_manifest.json'


@dataclass
class FixEntry:
    """A single fix entry in the manifest."""

    # Identity
    session_id: str              # e.g., "EQUITY-35"
    commit_hash: str             # Git commit that contains this fix
    commit_short: str            # Short hash for display

    # Timing
    deployed_at: str             # ISO timestamp when fix was deployed
    created_at: str              # ISO timestamp when entry was created

    # Description
    description: str             # What was fixed
    components: List[str]        # Which components were changed
    affected_systems: List[str]  # ['equity', 'crypto', 'both']

    # Expected impact
    expected_impact: str         # How trades should behave after this fix
    trade_fields_affected: List[str]  # Which trade record fields are affected

    # Verification
    verified: bool = False       # Has fix been verified with a trade?
    verification_notes: str = "" # Notes from verification
    verified_at: str = ""        # When verified

    # Files changed
    files_changed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FixEntry':
        return cls(**data)


class FixManifest:
    """
    Manages the fix manifest - a chronological log of all code fixes.

    Provides:
    - Add new fixes with affected components
    - Query which fixes apply to a trade by timestamp
    - Mark fixes as verified
    - Generate audit reports
    """

    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifest_path = manifest_path or MANIFEST_FILE
        self.entries: List[FixEntry] = []
        self._load()

    def _load(self) -> None:
        """Load manifest from disk."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    data = json.load(f)
                    self.entries = [FixEntry.from_dict(e) for e in data.get('fixes', [])]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load fix manifest: {e}")
                self.entries = []
        else:
            # Create directory if needed
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            self.entries = []

    def _save(self) -> None:
        """Save manifest to disk."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, 'w') as f:
            json.dump({
                'version': '1.0',
                'last_updated': datetime.now().isoformat(),
                'fixes': [e.to_dict() for e in self.entries]
            }, f, indent=2)

    def add_fix(
        self,
        session_id: str,
        description: str,
        components: List[str],
        affected_systems: List[str],
        expected_impact: str,
        trade_fields_affected: Optional[List[str]] = None,
        commit_hash: Optional[str] = None,
        deployed_at: Optional[str] = None,
        files_changed: Optional[List[str]] = None,
    ) -> FixEntry:
        """
        Add a new fix to the manifest.

        Args:
            session_id: Session identifier (e.g., "EQUITY-35")
            description: What was fixed
            components: List of affected components
            affected_systems: List of affected systems ['equity', 'crypto']
            expected_impact: How trades should behave after this fix
            trade_fields_affected: Which trade record fields changed
            commit_hash: Git commit hash (auto-detected if not provided)
            deployed_at: Deployment timestamp (uses now if not provided)
            files_changed: List of changed files

        Returns:
            The created FixEntry
        """
        # Auto-detect commit if not provided
        if not commit_hash:
            try:
                result = subprocess.run(
                    ['git', 'rev-parse', 'HEAD'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                commit_hash = result.stdout.strip()
            except Exception:
                commit_hash = "unknown"

        entry = FixEntry(
            session_id=session_id,
            commit_hash=commit_hash,
            commit_short=commit_hash[:7] if commit_hash else "unknown",
            deployed_at=deployed_at or datetime.now().isoformat(),
            created_at=datetime.now().isoformat(),
            description=description,
            components=components,
            affected_systems=affected_systems,
            expected_impact=expected_impact,
            trade_fields_affected=trade_fields_affected or [],
            files_changed=files_changed or []
        )

        self.entries.append(entry)
        self._save()
        return entry

    def get_fix(self, session_id: str) -> Optional[FixEntry]:
        """Get a fix entry by session ID."""
        for entry in self.entries:
            if entry.session_id == session_id:
                return entry
        return None

    def get_fixes_after(self, timestamp: datetime) -> List[FixEntry]:
        """Get all fixes deployed after a given timestamp."""
        result = []
        for entry in self.entries:
            deployed = datetime.fromisoformat(entry.deployed_at.replace('Z', '+00:00'))
            if deployed > timestamp:
                result.append(entry)
        return result

    def get_fixes_before(self, timestamp: datetime) -> List[FixEntry]:
        """Get all fixes deployed before a given timestamp."""
        result = []
        for entry in self.entries:
            try:
                deployed = datetime.fromisoformat(entry.deployed_at.replace('Z', '+00:00'))
                if deployed < timestamp:
                    result.append(entry)
            except ValueError:
                continue
        return result

    def trade_after_fix(self, trade_timestamp: datetime, session_id: str) -> bool:
        """
        Check if a trade happened after a specific fix was deployed.

        Args:
            trade_timestamp: When the trade was executed
            session_id: The fix session ID to check

        Returns:
            True if trade was after the fix, False otherwise
        """
        fix = self.get_fix(session_id)
        if not fix:
            return False

        try:
            deployed = datetime.fromisoformat(fix.deployed_at.replace('Z', '+00:00'))
            return trade_timestamp > deployed
        except ValueError:
            return False

    def get_applicable_fixes(self, trade_timestamp: datetime, system: str = 'equity') -> List[FixEntry]:
        """
        Get all fixes that apply to a trade (deployed before the trade and affect the system).

        Args:
            trade_timestamp: When the trade was executed
            system: Which system ('equity' or 'crypto')

        Returns:
            List of applicable FixEntry objects
        """
        result = []
        for entry in self.entries:
            try:
                deployed = datetime.fromisoformat(entry.deployed_at.replace('Z', '+00:00'))
                if deployed < trade_timestamp and system in entry.affected_systems:
                    result.append(entry)
            except ValueError:
                continue
        return result

    def mark_verified(self, session_id: str, notes: str = "") -> bool:
        """
        Mark a fix as verified with a successful trade.

        Args:
            session_id: The fix session ID
            notes: Verification notes

        Returns:
            True if fix was found and marked, False otherwise
        """
        for entry in self.entries:
            if entry.session_id == session_id:
                entry.verified = True
                entry.verification_notes = notes
                entry.verified_at = datetime.now().isoformat()
                self._save()
                return True
        return False

    def get_unverified_fixes(self, system: Optional[str] = None) -> List[FixEntry]:
        """Get all fixes that haven't been verified yet."""
        result = []
        for entry in self.entries:
            if not entry.verified:
                if system is None or system in entry.affected_systems:
                    result.append(entry)
        return result

    def get_fixes_by_component(self, component: str) -> List[FixEntry]:
        """Get all fixes affecting a specific component."""
        return [e for e in self.entries if component in e.components]

    def generate_audit_report(self) -> str:
        """Generate a human-readable audit report."""
        lines = [
            "=" * 70,
            "FIX MANIFEST AUDIT REPORT",
            f"Generated: {datetime.now().isoformat()}",
            "=" * 70,
            "",
            f"Total Fixes Tracked: {len(self.entries)}",
            f"Verified: {sum(1 for e in self.entries if e.verified)}",
            f"Unverified: {sum(1 for e in self.entries if not e.verified)}",
            "",
        ]

        # Group by system
        equity_fixes = [e for e in self.entries if 'equity' in e.affected_systems]
        crypto_fixes = [e for e in self.entries if 'crypto' in e.affected_systems]

        lines.append(f"Equity System Fixes: {len(equity_fixes)}")
        lines.append(f"Crypto System Fixes: {len(crypto_fixes)}")
        lines.append("")

        # List recent fixes
        lines.append("-" * 70)
        lines.append("RECENT FIXES (last 10)")
        lines.append("-" * 70)

        for entry in sorted(self.entries, key=lambda e: e.deployed_at, reverse=True)[:10]:
            verified_marker = "[x]" if entry.verified else "[ ]"
            lines.append(f"{verified_marker} [{entry.session_id}] {entry.commit_short}")
            lines.append(f"   {entry.description}")
            lines.append(f"   Deployed: {entry.deployed_at[:19]}")
            lines.append(f"   Affects: {', '.join(entry.affected_systems)}")
            if entry.expected_impact:
                lines.append(f"   Impact: {entry.expected_impact[:60]}...")
            lines.append("")

        # List unverified fixes
        unverified = self.get_unverified_fixes()
        if unverified:
            lines.append("-" * 70)
            lines.append(f"UNVERIFIED FIXES ({len(unverified)})")
            lines.append("-" * 70)
            for entry in unverified:
                lines.append(f"  [{entry.session_id}] {entry.description}")
            lines.append("")

        lines.append("=" * 70)
        return "\n".join(lines)

    def to_json(self) -> str:
        """Export manifest as JSON string."""
        return json.dumps({
            'version': '1.0',
            'exported_at': datetime.now().isoformat(),
            'fixes': [e.to_dict() for e in self.entries]
        }, indent=2)


# Convenience function for auto-recording fixes from git
def record_current_commit_as_fix(
    description: str,
    components: List[str],
    affected_systems: List[str],
    expected_impact: str,
    trade_fields_affected: Optional[List[str]] = None
) -> FixEntry:
    """
    Record the current git commit as a fix.

    Convenience function that auto-detects:
    - Commit hash
    - Session ID from commit message
    - Changed files

    Args:
        description: What was fixed
        components: List of affected components
        affected_systems: List of affected systems
        expected_impact: How trades should behave after this fix
        trade_fields_affected: Which trade record fields changed

    Returns:
        The created FixEntry
    """
    # Get commit info
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%H|%s'],
            capture_output=True,
            text=True,
            timeout=5
        )
        commit_hash, commit_message = result.stdout.strip().split('|', 1)
    except Exception:
        commit_hash = "unknown"
        commit_message = ""

    # Extract session ID
    import re
    match = re.search(r'\(([A-Z]+-[\w-]+)\)', commit_message)
    session_id = match.group(1) if match else "UNKNOWN"

    # Get changed files
    try:
        result = subprocess.run(
            ['git', 'diff', '--name-only', 'HEAD~1', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        files_changed = [f for f in result.stdout.strip().split('\n') if f]
    except Exception:
        files_changed = []

    manifest = FixManifest()
    return manifest.add_fix(
        session_id=session_id,
        description=description,
        components=components,
        affected_systems=affected_systems,
        expected_impact=expected_impact,
        trade_fields_affected=trade_fields_affected,
        commit_hash=commit_hash,
        files_changed=files_changed
    )


if __name__ == '__main__':
    # Example usage
    manifest = FixManifest()
    print(manifest.generate_audit_report())

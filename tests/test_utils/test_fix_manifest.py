"""
Tests for utils/fix_manifest.py - Fix tracking and audit functionality.

EQUITY-83: Phase 3 test coverage for fix manifest module.

Tests cover:
- FixEntry dataclass
- FixManifest load/save
- Add/get/query fixes
- Timestamp comparisons
- Verification workflow
- Audit report generation
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

from utils.fix_manifest import (
    FixEntry,
    FixManifest,
    record_current_commit_as_fix,
    MANIFEST_FILE,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_manifest_path(tmp_path):
    """Create a temporary manifest file path."""
    return tmp_path / "test_fix_manifest.json"


@pytest.fixture
def sample_fix_entry():
    """Create a sample FixEntry for testing."""
    return FixEntry(
        session_id="EQUITY-35",
        commit_hash="abc1234567890def1234567890abcdef12345678",
        commit_short="abc1234",
        deployed_at="2025-01-20T10:00:00",
        created_at="2025-01-20T10:00:00",
        description="Add EOD exit for 1H trades",
        components=["equity_daemon", "position_monitor"],
        affected_systems=["equity"],
        expected_impact="1H trades now exit at market close",
        trade_fields_affected=["exit_timestamp", "exit_reason"],
        verified=False,
        verification_notes="",
        verified_at="",
        files_changed=["daemon.py", "position_monitor.py"]
    )


@pytest.fixture
def manifest_with_entries(temp_manifest_path):
    """Create a manifest with pre-populated entries."""
    manifest = FixManifest(temp_manifest_path)

    # Add multiple entries
    manifest.add_fix(
        session_id="EQUITY-30",
        description="Fix pattern detection",
        components=["pattern_detector"],
        affected_systems=["equity"],
        expected_impact="Correct 3-2 pattern detection",
        commit_hash="aaa1111111111111111111111111111111111111",
        deployed_at="2025-01-15T09:00:00"
    )

    manifest.add_fix(
        session_id="EQUITY-31",
        description="Add TFC filter",
        components=["daemon"],
        affected_systems=["equity", "crypto"],
        expected_impact="Filter weak TFC signals",
        commit_hash="bbb2222222222222222222222222222222222222",
        deployed_at="2025-01-16T10:00:00"
    )

    manifest.add_fix(
        session_id="CRYPTO-5",
        description="Fix funding rate calculation",
        components=["crypto_daemon"],
        affected_systems=["crypto"],
        expected_impact="Correct funding costs",
        commit_hash="ccc3333333333333333333333333333333333333",
        deployed_at="2025-01-17T11:00:00"
    )

    return manifest


# =============================================================================
# FixEntry Dataclass Tests
# =============================================================================

class TestFixEntry:
    """Tests for FixEntry dataclass."""

    def test_fix_entry_creation(self, sample_fix_entry):
        """Test basic FixEntry creation."""
        assert sample_fix_entry.session_id == "EQUITY-35"
        assert sample_fix_entry.commit_short == "abc1234"
        assert sample_fix_entry.description == "Add EOD exit for 1H trades"
        assert sample_fix_entry.verified is False

    def test_fix_entry_to_dict(self, sample_fix_entry):
        """Test FixEntry serialization to dict."""
        data = sample_fix_entry.to_dict()

        assert isinstance(data, dict)
        assert data["session_id"] == "EQUITY-35"
        assert data["commit_hash"] == "abc1234567890def1234567890abcdef12345678"
        assert data["components"] == ["equity_daemon", "position_monitor"]
        assert data["affected_systems"] == ["equity"]

    def test_fix_entry_from_dict(self, sample_fix_entry):
        """Test FixEntry deserialization from dict."""
        data = sample_fix_entry.to_dict()
        restored = FixEntry.from_dict(data)

        assert restored.session_id == sample_fix_entry.session_id
        assert restored.commit_hash == sample_fix_entry.commit_hash
        assert restored.description == sample_fix_entry.description
        assert restored.components == sample_fix_entry.components

    def test_fix_entry_default_values(self):
        """Test FixEntry with default values."""
        entry = FixEntry(
            session_id="TEST-1",
            commit_hash="test123",
            commit_short="test123",
            deployed_at="2025-01-01T00:00:00",
            created_at="2025-01-01T00:00:00",
            description="Test fix",
            components=["test"],
            affected_systems=["equity"],
            expected_impact="Test impact",
            trade_fields_affected=[]
        )

        assert entry.verified is False
        assert entry.verification_notes == ""
        assert entry.verified_at == ""
        assert entry.files_changed == []


# =============================================================================
# FixManifest Initialization Tests
# =============================================================================

class TestFixManifestInit:
    """Tests for FixManifest initialization."""

    def test_manifest_creates_empty(self, temp_manifest_path):
        """Test manifest creates with empty entries."""
        manifest = FixManifest(temp_manifest_path)
        assert manifest.entries == []

    def test_manifest_creates_directory(self, tmp_path):
        """Test manifest creates parent directory if needed."""
        nested_path = tmp_path / "subdir" / "manifest.json"
        manifest = FixManifest(nested_path)

        # Directory should be created on save
        manifest.add_fix(
            session_id="TEST-1",
            description="Test",
            components=["test"],
            affected_systems=["equity"],
            expected_impact="Test"
        )

        assert nested_path.parent.exists()

    def test_manifest_uses_default_path(self):
        """Test manifest uses default path when not specified."""
        with patch.object(Path, 'exists', return_value=False):
            manifest = FixManifest()
            assert manifest.manifest_path == MANIFEST_FILE

    def test_manifest_loads_existing(self, temp_manifest_path):
        """Test manifest loads existing entries."""
        # Create manifest with entries
        data = {
            "version": "1.0",
            "last_updated": "2025-01-20T10:00:00",
            "fixes": [{
                "session_id": "EQUITY-10",
                "commit_hash": "abc123",
                "commit_short": "abc123",
                "deployed_at": "2025-01-01T00:00:00",
                "created_at": "2025-01-01T00:00:00",
                "description": "Test fix",
                "components": ["test"],
                "affected_systems": ["equity"],
                "expected_impact": "Test",
                "trade_fields_affected": [],
                "verified": False,
                "verification_notes": "",
                "verified_at": "",
                "files_changed": []
            }]
        }

        with open(temp_manifest_path, 'w') as f:
            json.dump(data, f)

        manifest = FixManifest(temp_manifest_path)
        assert len(manifest.entries) == 1
        assert manifest.entries[0].session_id == "EQUITY-10"

    def test_manifest_handles_corrupt_json(self, temp_manifest_path):
        """Test manifest handles corrupt JSON gracefully."""
        with open(temp_manifest_path, 'w') as f:
            f.write("not valid json {{{")

        manifest = FixManifest(temp_manifest_path)
        assert manifest.entries == []


# =============================================================================
# FixManifest Add/Get Tests
# =============================================================================

class TestFixManifestAddGet:
    """Tests for adding and getting fixes."""

    def test_add_fix_basic(self, temp_manifest_path):
        """Test adding a basic fix."""
        manifest = FixManifest(temp_manifest_path)

        entry = manifest.add_fix(
            session_id="EQUITY-40",
            description="Fix bug",
            components=["daemon"],
            affected_systems=["equity"],
            expected_impact="Bug fixed"
        )

        assert entry.session_id == "EQUITY-40"
        assert entry.description == "Fix bug"
        assert len(manifest.entries) == 1

    def test_add_fix_with_all_fields(self, temp_manifest_path):
        """Test adding fix with all optional fields."""
        manifest = FixManifest(temp_manifest_path)

        entry = manifest.add_fix(
            session_id="EQUITY-41",
            description="Complete fix",
            components=["daemon", "monitor"],
            affected_systems=["equity", "crypto"],
            expected_impact="Full impact",
            trade_fields_affected=["field1", "field2"],
            commit_hash="xyz789",
            deployed_at="2025-01-20T12:00:00",
            files_changed=["file1.py", "file2.py"]
        )

        assert entry.trade_fields_affected == ["field1", "field2"]
        assert entry.commit_hash == "xyz789"
        assert entry.files_changed == ["file1.py", "file2.py"]

    def test_add_fix_auto_detects_commit(self, temp_manifest_path):
        """Test add_fix auto-detects commit when not provided."""
        manifest = FixManifest(temp_manifest_path)

        with patch('utils.fix_manifest.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout="fedcba9876543210fedcba9876543210fedcba98\n"
            )

            entry = manifest.add_fix(
                session_id="EQUITY-42",
                description="Auto-detect",
                components=["test"],
                affected_systems=["equity"],
                expected_impact="Test"
            )

            assert entry.commit_hash == "fedcba9876543210fedcba9876543210fedcba98"
            assert entry.commit_short == "fedcba9"

    def test_add_fix_handles_git_failure(self, temp_manifest_path):
        """Test add_fix handles git command failure."""
        manifest = FixManifest(temp_manifest_path)

        with patch('utils.fix_manifest.subprocess.run', side_effect=Exception("Git error")):
            entry = manifest.add_fix(
                session_id="EQUITY-43",
                description="No git",
                components=["test"],
                affected_systems=["equity"],
                expected_impact="Test"
            )

            assert entry.commit_hash == "unknown"

    def test_get_fix_found(self, manifest_with_entries):
        """Test getting existing fix by session ID."""
        fix = manifest_with_entries.get_fix("EQUITY-31")

        assert fix is not None
        assert fix.session_id == "EQUITY-31"
        assert fix.description == "Add TFC filter"

    def test_get_fix_not_found(self, manifest_with_entries):
        """Test getting non-existent fix returns None."""
        fix = manifest_with_entries.get_fix("EQUITY-99")
        assert fix is None


# =============================================================================
# FixManifest Timestamp Query Tests
# =============================================================================

class TestFixManifestTimestampQueries:
    """Tests for timestamp-based queries."""

    def test_get_fixes_after(self, manifest_with_entries):
        """Test getting fixes after a timestamp."""
        cutoff = datetime(2025, 1, 16, 0, 0, 0)
        fixes = manifest_with_entries.get_fixes_after(cutoff)

        assert len(fixes) == 2
        session_ids = [f.session_id for f in fixes]
        assert "EQUITY-31" in session_ids
        assert "CRYPTO-5" in session_ids

    def test_get_fixes_before(self, manifest_with_entries):
        """Test getting fixes before a timestamp."""
        cutoff = datetime(2025, 1, 16, 12, 0, 0)
        fixes = manifest_with_entries.get_fixes_before(cutoff)

        assert len(fixes) == 2
        session_ids = [f.session_id for f in fixes]
        assert "EQUITY-30" in session_ids
        assert "EQUITY-31" in session_ids

    def test_trade_after_fix_true(self, manifest_with_entries):
        """Test trade_after_fix returns True when trade is after."""
        trade_time = datetime(2025, 1, 18, 12, 0, 0)
        result = manifest_with_entries.trade_after_fix(trade_time, "CRYPTO-5")
        assert result is True

    def test_trade_after_fix_false(self, manifest_with_entries):
        """Test trade_after_fix returns False when trade is before."""
        trade_time = datetime(2025, 1, 16, 8, 0, 0)
        result = manifest_with_entries.trade_after_fix(trade_time, "CRYPTO-5")
        assert result is False

    def test_trade_after_fix_unknown_session(self, manifest_with_entries):
        """Test trade_after_fix returns False for unknown session."""
        trade_time = datetime(2025, 1, 20, 12, 0, 0)
        result = manifest_with_entries.trade_after_fix(trade_time, "UNKNOWN-99")
        assert result is False

    def test_get_applicable_fixes_equity(self, manifest_with_entries):
        """Test getting applicable fixes for equity system."""
        trade_time = datetime(2025, 1, 18, 12, 0, 0)
        fixes = manifest_with_entries.get_applicable_fixes(trade_time, "equity")

        assert len(fixes) == 2
        session_ids = [f.session_id for f in fixes]
        assert "EQUITY-30" in session_ids
        assert "EQUITY-31" in session_ids
        assert "CRYPTO-5" not in session_ids  # Crypto-only fix

    def test_get_applicable_fixes_crypto(self, manifest_with_entries):
        """Test getting applicable fixes for crypto system."""
        trade_time = datetime(2025, 1, 18, 12, 0, 0)
        fixes = manifest_with_entries.get_applicable_fixes(trade_time, "crypto")

        assert len(fixes) == 2
        session_ids = [f.session_id for f in fixes]
        assert "EQUITY-31" in session_ids  # Affects both
        assert "CRYPTO-5" in session_ids


# =============================================================================
# FixManifest Verification Tests
# =============================================================================

class TestFixManifestVerification:
    """Tests for fix verification workflow."""

    def test_mark_verified(self, manifest_with_entries):
        """Test marking a fix as verified."""
        result = manifest_with_entries.mark_verified(
            "EQUITY-30",
            notes="Verified with SPY trade on 2025-01-20"
        )

        assert result is True

        fix = manifest_with_entries.get_fix("EQUITY-30")
        assert fix.verified is True
        assert fix.verification_notes == "Verified with SPY trade on 2025-01-20"
        assert fix.verified_at != ""

    def test_mark_verified_unknown_session(self, manifest_with_entries):
        """Test marking unknown session returns False."""
        result = manifest_with_entries.mark_verified("UNKNOWN-99", "Notes")
        assert result is False

    def test_get_unverified_fixes_all(self, manifest_with_entries):
        """Test getting all unverified fixes."""
        unverified = manifest_with_entries.get_unverified_fixes()
        assert len(unverified) == 3  # All initially unverified

    def test_get_unverified_fixes_by_system(self, manifest_with_entries):
        """Test getting unverified fixes for specific system."""
        unverified = manifest_with_entries.get_unverified_fixes("crypto")

        assert len(unverified) == 2
        session_ids = [f.session_id for f in unverified]
        assert "EQUITY-31" in session_ids  # Affects both
        assert "CRYPTO-5" in session_ids

    def test_get_unverified_after_verification(self, manifest_with_entries):
        """Test unverified count decreases after verification."""
        manifest_with_entries.mark_verified("EQUITY-30")
        unverified = manifest_with_entries.get_unverified_fixes()

        assert len(unverified) == 2
        session_ids = [f.session_id for f in unverified]
        assert "EQUITY-30" not in session_ids


# =============================================================================
# FixManifest Component Query Tests
# =============================================================================

class TestFixManifestComponentQueries:
    """Tests for component-based queries."""

    def test_get_fixes_by_component(self, manifest_with_entries):
        """Test getting fixes by affected component."""
        fixes = manifest_with_entries.get_fixes_by_component("daemon")

        assert len(fixes) == 1
        assert fixes[0].session_id == "EQUITY-31"

    def test_get_fixes_by_component_not_found(self, manifest_with_entries):
        """Test getting fixes for unknown component returns empty."""
        fixes = manifest_with_entries.get_fixes_by_component("unknown_component")
        assert fixes == []


# =============================================================================
# FixManifest Report Generation Tests
# =============================================================================

class TestFixManifestReports:
    """Tests for report generation."""

    def test_generate_audit_report(self, manifest_with_entries):
        """Test audit report generation."""
        report = manifest_with_entries.generate_audit_report()

        assert "FIX MANIFEST AUDIT REPORT" in report
        assert "Total Fixes Tracked: 3" in report
        assert "Verified: 0" in report
        assert "Unverified: 3" in report
        assert "EQUITY-30" in report
        assert "EQUITY-31" in report
        assert "CRYPTO-5" in report

    def test_generate_audit_report_with_verified(self, manifest_with_entries):
        """Test audit report shows verified count."""
        manifest_with_entries.mark_verified("EQUITY-30")
        report = manifest_with_entries.generate_audit_report()

        assert "Verified: 1" in report
        assert "Unverified: 2" in report

    def test_to_json(self, manifest_with_entries):
        """Test JSON export."""
        json_str = manifest_with_entries.to_json()
        data = json.loads(json_str)

        assert data["version"] == "1.0"
        assert "exported_at" in data
        assert len(data["fixes"]) == 3


# =============================================================================
# FixManifest Persistence Tests
# =============================================================================

class TestFixManifestPersistence:
    """Tests for save/load functionality."""

    def test_save_and_reload(self, temp_manifest_path):
        """Test manifest saves and reloads correctly."""
        # Create and populate manifest
        manifest1 = FixManifest(temp_manifest_path)
        manifest1.add_fix(
            session_id="PERSIST-1",
            description="Persistence test",
            components=["test"],
            affected_systems=["equity"],
            expected_impact="Test persistence"
        )

        # Create new manifest from same file
        manifest2 = FixManifest(temp_manifest_path)

        assert len(manifest2.entries) == 1
        assert manifest2.entries[0].session_id == "PERSIST-1"

    def test_auto_saves_on_add(self, temp_manifest_path):
        """Test manifest auto-saves after adding entry."""
        manifest = FixManifest(temp_manifest_path)
        manifest.add_fix(
            session_id="AUTO-SAVE",
            description="Test",
            components=["test"],
            affected_systems=["equity"],
            expected_impact="Test"
        )

        # File should exist and contain data
        with open(temp_manifest_path, 'r') as f:
            data = json.load(f)

        assert len(data["fixes"]) == 1
        assert data["fixes"][0]["session_id"] == "AUTO-SAVE"


# =============================================================================
# record_current_commit_as_fix Tests
# =============================================================================

class TestRecordCurrentCommit:
    """Tests for record_current_commit_as_fix convenience function."""

    def test_extracts_session_from_parentheses(self, temp_manifest_path):
        """Test session ID extraction from (SESSION-ID) format."""
        with patch('utils.fix_manifest.subprocess.run') as mock_run:
            # First call: commit info
            # Second call: changed files
            mock_run.side_effect = [
                MagicMock(stdout="abc123|feat: add feature (EQUITY-50)\n"),
                MagicMock(stdout="file1.py\nfile2.py\n")
            ]

            with patch('utils.fix_manifest.FixManifest') as mock_manifest_class:
                mock_manifest = MagicMock()
                mock_manifest_class.return_value = mock_manifest

                record_current_commit_as_fix(
                    description="Test",
                    components=["test"],
                    affected_systems=["equity"],
                    expected_impact="Test"
                )

                mock_manifest.add_fix.assert_called_once()
                call_kwargs = mock_manifest.add_fix.call_args[1]
                assert call_kwargs["session_id"] == "EQUITY-50"

    def test_extracts_session_from_middle(self, temp_manifest_path):
        """Test session ID extraction from EQUITY-XX in middle of message."""
        with patch('utils.fix_manifest.subprocess.run') as mock_run:
            mock_run.side_effect = [
                MagicMock(stdout="abc123|fix: EQUITY-51 bug resolved\n"),
                MagicMock(stdout="file1.py\n")
            ]

            with patch('utils.fix_manifest.FixManifest') as mock_manifest_class:
                mock_manifest = MagicMock()
                mock_manifest_class.return_value = mock_manifest

                record_current_commit_as_fix(
                    description="Test",
                    components=["test"],
                    affected_systems=["equity"],
                    expected_impact="Test"
                )

                call_kwargs = mock_manifest.add_fix.call_args[1]
                # fix_manifest only supports (SESSION-ID) format, so this returns UNKNOWN
                # unless message contains pattern in parentheses
                assert call_kwargs["session_id"] == "UNKNOWN"

    def test_handles_git_failure(self, temp_manifest_path):
        """Test graceful handling of git failures."""
        with patch('utils.fix_manifest.subprocess.run', side_effect=Exception("Git error")):
            with patch('utils.fix_manifest.FixManifest') as mock_manifest_class:
                mock_manifest = MagicMock()
                mock_manifest_class.return_value = mock_manifest

                record_current_commit_as_fix(
                    description="Test",
                    components=["test"],
                    affected_systems=["equity"],
                    expected_impact="Test"
                )

                call_kwargs = mock_manifest.add_fix.call_args[1]
                assert call_kwargs["session_id"] == "UNKNOWN"
                assert call_kwargs["commit_hash"] == "unknown"

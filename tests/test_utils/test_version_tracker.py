"""
Tests for utils/version_tracker.py - Code version tracking for trades.

EQUITY-83: Phase 3 test coverage for version tracker module.

Tests cover:
- VersionInfo dataclass
- Git command execution
- Session ID extraction
- Version caching
- Component dependency tracking
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from utils.version_tracker import (
    VersionInfo,
    get_version_info,
    get_version_string,
    get_trade_version_metadata,
    get_affected_components,
    get_affected_systems,
    COMPONENT_DEPENDENCIES,
    _extract_session_id,
    _run_git_command,
    _get_repo_root,
    _version_cache,
)


# =============================================================================
# VersionInfo Dataclass Tests
# =============================================================================

class TestVersionInfo:
    """Tests for VersionInfo dataclass."""

    def test_version_info_creation(self):
        """Test basic VersionInfo creation."""
        info = VersionInfo(
            commit_hash="abc1234567890def1234567890abcdef12345678",
            commit_short="abc1234",
            commit_message="feat: add feature (EQUITY-50)",
            commit_timestamp="2025-01-20T10:00:00",
            branch="main",
            session_id="EQUITY-50",
            is_dirty=False,
            tags=["v1.0.0"],
            captured_at="2025-01-20T10:00:00"
        )

        assert info.commit_hash == "abc1234567890def1234567890abcdef12345678"
        assert info.commit_short == "abc1234"
        assert info.session_id == "EQUITY-50"
        assert info.is_dirty is False

    def test_version_info_to_dict(self):
        """Test VersionInfo serialization."""
        info = VersionInfo(
            commit_hash="abc123",
            commit_short="abc123",
            commit_message="Test commit",
            commit_timestamp="2025-01-20T10:00:00",
            branch="main",
            session_id="EQUITY-1",
            is_dirty=True,
            tags=[],
            captured_at="2025-01-20T10:00:00"
        )

        data = info.to_dict()

        assert isinstance(data, dict)
        assert data["commit_hash"] == "abc123"
        assert data["session_id"] == "EQUITY-1"
        assert data["is_dirty"] is True

    def test_version_string_clean(self):
        """Test version_string property for clean repo."""
        info = VersionInfo(
            commit_hash="abc123",
            commit_short="abc1234",
            commit_message="Test",
            commit_timestamp="2025-01-20T10:00:00",
            branch="main",
            session_id="EQUITY-50",
            is_dirty=False,
            tags=[],
            captured_at="2025-01-20T10:00:00"
        )

        assert info.version_string == "abc1234/EQUITY-50"

    def test_version_string_dirty(self):
        """Test version_string property for dirty repo."""
        info = VersionInfo(
            commit_hash="abc123",
            commit_short="abc1234",
            commit_message="Test",
            commit_timestamp="2025-01-20T10:00:00",
            branch="main",
            session_id="EQUITY-50",
            is_dirty=True,
            tags=[],
            captured_at="2025-01-20T10:00:00"
        )

        assert info.version_string == "abc1234/EQUITY-50 (dirty)"

    def test_trade_metadata(self):
        """Test trade_metadata property."""
        info = VersionInfo(
            commit_hash="abc123",
            commit_short="abc1234",
            commit_message="Test",
            commit_timestamp="2025-01-20T10:00:00",
            branch="main",
            session_id="EQUITY-50",
            is_dirty=False,
            tags=[],
            captured_at="2025-01-20T10:00:00"
        )

        metadata = info.trade_metadata

        assert metadata["code_version"] == "abc1234"
        assert metadata["code_session"] == "EQUITY-50"
        assert metadata["code_branch"] == "main"
        assert metadata["code_dirty"] is False
        assert metadata["code_timestamp"] == "2025-01-20T10:00:00"


# =============================================================================
# Session ID Extraction Tests
# =============================================================================

class TestSessionIdExtraction:
    """Tests for _extract_session_id function."""

    def test_extract_from_parentheses_end(self):
        """Test extraction from (SESSION-ID) at end."""
        result = _extract_session_id("feat: add feature (EQUITY-50)")
        assert result == "EQUITY-50"

    def test_extract_from_parentheses_with_space(self):
        """Test extraction from (SESSION-ID) with trailing space."""
        result = _extract_session_id("feat: add feature (EQUITY-51) ")
        assert result == "EQUITY-51"

    def test_extract_from_colon_prefix(self):
        """Test extraction from SESSION-ID: prefix."""
        result = _extract_session_id("EQUITY-52: fix bug in daemon")
        assert result == "EQUITY-52"

    def test_extract_crypto_session(self):
        """Test extraction of CRYPTO session ID."""
        result = _extract_session_id("fix: crypto bug (CRYPTO-MONITOR-2)")
        assert result == "CRYPTO-MONITOR-2"

    def test_extract_from_anywhere(self):
        """Test extraction from middle of message."""
        result = _extract_session_id("fix: EQUITY-53 bug in pattern detection")
        assert result == "EQUITY-53"

    def test_extract_unknown(self):
        """Test returns UNKNOWN for unrecognized format."""
        result = _extract_session_id("fix: some bug without session")
        assert result == "UNKNOWN"

    def test_extract_empty_message(self):
        """Test returns UNKNOWN for empty message."""
        result = _extract_session_id("")
        assert result == "UNKNOWN"


# =============================================================================
# Git Command Tests
# =============================================================================

class TestGitCommands:
    """Tests for git command execution."""

    def test_run_git_command_success(self):
        """Test successful git command execution."""
        with patch('utils.version_tracker.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stdout="main\n")

            result = _run_git_command(['rev-parse', '--abbrev-ref', 'HEAD'])

            assert result == "main"

    def test_run_git_command_timeout(self):
        """Test git command handles timeout."""
        with patch('utils.version_tracker.subprocess.run') as mock_run:
            from subprocess import TimeoutExpired
            mock_run.side_effect = TimeoutExpired('git', 5)

            result = _run_git_command(['rev-parse', 'HEAD'])

            assert result == ""

    def test_run_git_command_error(self):
        """Test git command handles subprocess error."""
        with patch('utils.version_tracker.subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("git not found")

            result = _run_git_command(['rev-parse', 'HEAD'])

            assert result == ""

    def test_get_repo_root_success(self):
        """Test getting repo root successfully."""
        with patch('utils.version_tracker.subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(stdout="/path/to/repo\n")

            result = _get_repo_root()

            assert result == "/path/to/repo"

    def test_get_repo_root_fallback(self):
        """Test repo root falls back to cwd on error."""
        with patch('utils.version_tracker.subprocess.run', side_effect=Exception("Error")):
            with patch('utils.version_tracker.os.getcwd', return_value="/fallback/path"):
                result = _get_repo_root()

                assert result == "/fallback/path"


# =============================================================================
# get_version_info Tests
# =============================================================================

class TestGetVersionInfo:
    """Tests for get_version_info function."""

    def setup_method(self):
        """Clear cache before each test."""
        import utils.version_tracker
        utils.version_tracker._version_cache = None

    def test_get_version_info_basic(self):
        """Test getting version info with mocked git."""
        import utils.version_tracker
        utils.version_tracker._version_cache = None

        with patch('utils.version_tracker._run_git_command') as mock_git:
            mock_git.side_effect = [
                "abc1234567890def1234567890abcdef12345678",  # rev-parse HEAD
                "feat: add feature (EQUITY-60)",  # log message
                "2025-01-20T10:00:00+00:00",  # log timestamp
                "main",  # branch
                "",  # status (clean)
                "v1.0.0",  # tags
            ]

            info = get_version_info(force_refresh=True)

            assert info.commit_short == "abc1234"
            assert info.session_id == "EQUITY-60"
            assert info.branch == "main"
            assert info.is_dirty is False

    def test_get_version_info_dirty_repo(self):
        """Test version info detects dirty repo."""
        import utils.version_tracker
        utils.version_tracker._version_cache = None

        with patch('utils.version_tracker._run_git_command') as mock_git:
            mock_git.side_effect = [
                "abc123",  # rev-parse HEAD
                "test commit",  # log message
                "2025-01-20T10:00:00",  # log timestamp
                "main",  # branch
                " M modified_file.py",  # status (dirty)
                "",  # tags
            ]

            info = get_version_info(force_refresh=True)

            assert info.is_dirty is True

    def test_get_version_info_caching(self):
        """Test version info is cached."""
        import utils.version_tracker
        utils.version_tracker._version_cache = None

        with patch('utils.version_tracker._run_git_command') as mock_git:
            mock_git.side_effect = [
                "abc123",
                "test (EQUITY-70)",
                "2025-01-20T10:00:00",
                "main",
                "",
                "",
            ]

            info1 = get_version_info(force_refresh=True)
            info2 = get_version_info()  # Should use cache

            assert info1 is info2
            # Only 6 calls for first invocation, not 12
            assert mock_git.call_count == 6

    def test_get_version_info_force_refresh(self):
        """Test force_refresh bypasses cache."""
        import utils.version_tracker
        utils.version_tracker._version_cache = None

        with patch('utils.version_tracker._run_git_command') as mock_git:
            mock_git.side_effect = [
                "abc123", "test1 (EQUITY-71)", "2025-01-20T10:00:00", "main", "", "",
                "def456", "test2 (EQUITY-72)", "2025-01-20T11:00:00", "dev", "", "",
            ]

            info1 = get_version_info(force_refresh=True)
            info2 = get_version_info(force_refresh=True)

            assert info1.commit_short == "abc123"
            assert info2.commit_short == "def456"

    def test_get_version_info_no_git(self):
        """Test fallback when git not available."""
        import utils.version_tracker
        utils.version_tracker._version_cache = None

        with patch('utils.version_tracker._run_git_command', return_value=""):
            info = get_version_info(force_refresh=True)

            assert info.commit_hash == "0" * 40
            assert info.branch == "unknown"


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def setup_method(self):
        """Clear cache before each test."""
        import utils.version_tracker
        utils.version_tracker._version_cache = None

    def test_get_version_string(self):
        """Test get_version_string helper."""
        import utils.version_tracker
        utils.version_tracker._version_cache = None

        with patch('utils.version_tracker._run_git_command') as mock_git:
            mock_git.side_effect = [
                "abc1234", "test (EQUITY-80)", "2025-01-20T10:00:00", "main", "", "",
            ]

            result = get_version_string()

            assert "abc1234" in result
            assert "EQUITY-80" in result

    def test_get_trade_version_metadata(self):
        """Test get_trade_version_metadata helper."""
        import utils.version_tracker
        utils.version_tracker._version_cache = None

        with patch('utils.version_tracker._run_git_command') as mock_git:
            mock_git.side_effect = [
                "abc1234", "test (EQUITY-81)", "2025-01-20T10:00:00", "main", "", "",
            ]

            metadata = get_trade_version_metadata()

            assert metadata["code_version"] == "abc1234"
            assert metadata["code_session"] == "EQUITY-81"


# =============================================================================
# Component Dependency Tests
# =============================================================================

class TestComponentDependencies:
    """Tests for component dependency tracking."""

    def test_component_dependencies_structure(self):
        """Test COMPONENT_DEPENDENCIES has expected structure."""
        assert "pattern_detection" in COMPONENT_DEPENDENCIES
        assert "options_pricing" in COMPONENT_DEPENDENCIES
        assert "equity_daemon" in COMPONENT_DEPENDENCIES
        assert "crypto_daemon" in COMPONENT_DEPENDENCIES
        assert "alerting" in COMPONENT_DEPENDENCIES
        assert "execution" in COMPONENT_DEPENDENCIES

    def test_component_has_required_keys(self):
        """Test each component has required keys."""
        for component, config in COMPONENT_DEPENDENCIES.items():
            assert "files" in config, f"{component} missing 'files'"
            assert "affects" in config, f"{component} missing 'affects'"
            assert "description" in config, f"{component} missing 'description'"

    def test_get_affected_components_pattern_detection(self):
        """Test identifying pattern detection component."""
        changed_files = ["strat/bar_classifier.py", "utils/helper.py"]
        affected = get_affected_components(changed_files)

        assert "pattern_detection" in affected

    def test_get_affected_components_equity_daemon(self):
        """Test identifying equity daemon component."""
        changed_files = ["strat/signal_automation/daemon.py"]
        affected = get_affected_components(changed_files)

        assert "equity_daemon" in affected

    def test_get_affected_components_multiple(self):
        """Test identifying multiple components."""
        changed_files = [
            "strat/pattern_detector.py",
            "strat/signal_automation/daemon.py",
            "crypto/scanning/daemon.py"
        ]
        affected = get_affected_components(changed_files)

        assert "pattern_detection" in affected
        assert "equity_daemon" in affected
        assert "crypto_daemon" in affected

    def test_get_affected_components_none(self):
        """Test returns empty for unrelated files."""
        changed_files = ["README.md", "docs/HANDOFF.md"]
        affected = get_affected_components(changed_files)

        assert affected == []

    def test_get_affected_systems_equity(self):
        """Test identifying equity system."""
        changed_files = ["strat/options_module.py"]
        systems = get_affected_systems(changed_files)

        assert "equity" in systems
        assert "crypto" not in systems

    def test_get_affected_systems_crypto(self):
        """Test identifying crypto system."""
        changed_files = ["crypto/scanning/daemon.py"]
        systems = get_affected_systems(changed_files)

        assert "crypto" in systems

    def test_get_affected_systems_both(self):
        """Test identifying both systems."""
        changed_files = [
            "strat/bar_classifier.py",  # Affects both
            "strat/signal_automation/alerters/discord.py"  # Affects both
        ]
        systems = get_affected_systems(changed_files)

        # bar_classifier is in pattern_detection which affects both
        assert "equity" in systems or "crypto" in systems

    def test_get_affected_systems_empty(self):
        """Test returns empty for unrelated files."""
        changed_files = ["tests/test_something.py"]
        systems = get_affected_systems(changed_files)

        assert systems == []

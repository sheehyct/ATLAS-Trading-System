"""
Tests for strat/pattern_registry.py

Covers:
- PatternMetadata dataclass
- PATTERN_REGISTRY definitions
- get_pattern_metadata function
- is_bidirectional_pattern function with heuristics
- get_valid_directions function
- extract_setup_pattern_type function
"""

import pytest
from strat.pattern_registry import (
    PatternMetadata,
    PATTERN_REGISTRY,
    get_pattern_metadata,
    is_bidirectional_pattern,
    get_valid_directions,
    extract_setup_pattern_type,
)


# =============================================================================
# PatternMetadata Dataclass Tests
# =============================================================================

class TestPatternMetadata:
    """Test PatternMetadata dataclass."""

    def test_create_pattern_metadata(self):
        """Create PatternMetadata with all fields."""
        metadata = PatternMetadata(
            name="test-pattern",
            is_bidirectional=True,
            bars_required=2,
            description="Test description",
            valid_directions=('CALL', 'PUT'),
        )

        assert metadata.name == "test-pattern"
        assert metadata.is_bidirectional is True
        assert metadata.bars_required == 2
        assert metadata.description == "Test description"
        assert metadata.valid_directions == ('CALL', 'PUT')

    def test_pattern_metadata_is_frozen(self):
        """PatternMetadata is frozen (immutable)."""
        metadata = PatternMetadata(
            name="test",
            is_bidirectional=True,
            bars_required=1,
            description="test",
            valid_directions=('CALL',),
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            metadata.name = "changed"

    def test_pattern_metadata_hashable(self):
        """PatternMetadata is hashable (can be used in sets)."""
        metadata = PatternMetadata(
            name="test",
            is_bidirectional=True,
            bars_required=1,
            description="test",
            valid_directions=('CALL',),
        )

        # Should not raise
        hash(metadata)
        {metadata}  # Can be added to set


# =============================================================================
# PATTERN_REGISTRY Tests
# =============================================================================

class TestPatternRegistry:
    """Test PATTERN_REGISTRY definitions."""

    def test_registry_is_not_empty(self):
        """PATTERN_REGISTRY is not empty."""
        assert len(PATTERN_REGISTRY) > 0

    def test_registry_has_bidirectional_patterns(self):
        """Registry contains bidirectional patterns."""
        bidirectional = [
            k for k, v in PATTERN_REGISTRY.items() if v.is_bidirectional
        ]
        assert len(bidirectional) > 0
        assert "3-?" in bidirectional
        assert "3-1-?" in bidirectional

    def test_registry_has_unidirectional_patterns(self):
        """Registry contains unidirectional patterns."""
        unidirectional = [
            k for k, v in PATTERN_REGISTRY.items() if not v.is_bidirectional
        ]
        assert len(unidirectional) > 0
        assert "3-2D-?" in unidirectional
        assert "3-2U-?" in unidirectional

    def test_all_registry_values_are_pattern_metadata(self):
        """All registry values are PatternMetadata instances."""
        for key, value in PATTERN_REGISTRY.items():
            assert isinstance(value, PatternMetadata), f"{key} is not PatternMetadata"

    def test_registry_3_question_mark_pattern(self):
        """3-? pattern is correctly defined."""
        metadata = PATTERN_REGISTRY["3-?"]

        assert metadata.name == "3-?"
        assert metadata.is_bidirectional is True
        assert metadata.bars_required == 1
        assert metadata.valid_directions == ('CALL', 'PUT')

    def test_registry_3_1_question_mark_pattern(self):
        """3-1-? pattern is correctly defined."""
        metadata = PATTERN_REGISTRY["3-1-?"]

        assert metadata.name == "3-1-?"
        assert metadata.is_bidirectional is True
        assert metadata.bars_required == 2
        assert metadata.valid_directions == ('CALL', 'PUT')

    def test_registry_3_2D_question_mark_pattern(self):
        """3-2D-? pattern is correctly defined (unidirectional)."""
        metadata = PATTERN_REGISTRY["3-2D-?"]

        assert metadata.name == "3-2D-?"
        assert metadata.is_bidirectional is False
        assert metadata.bars_required == 2
        assert metadata.valid_directions == ('CALL',)  # Reversal UP = CALL only

    def test_registry_3_2U_question_mark_pattern(self):
        """3-2U-? pattern is correctly defined (unidirectional)."""
        metadata = PATTERN_REGISTRY["3-2U-?"]

        assert metadata.name == "3-2U-?"
        assert metadata.is_bidirectional is False
        assert metadata.bars_required == 2
        assert metadata.valid_directions == ('PUT',)  # Reversal DOWN = PUT only

    def test_registry_2U_1_question_mark_pattern(self):
        """2U-1-? pattern is correctly defined."""
        metadata = PATTERN_REGISTRY["2U-1-?"]

        assert metadata.is_bidirectional is True
        assert metadata.valid_directions == ('CALL', 'PUT')

    def test_registry_2D_1_question_mark_pattern(self):
        """2D-1-? pattern is correctly defined."""
        metadata = PATTERN_REGISTRY["2D-1-?"]

        assert metadata.is_bidirectional is True
        assert metadata.valid_directions == ('CALL', 'PUT')

    def test_registry_2_2_reversal_patterns(self):
        """2-2 reversal patterns are unidirectional."""
        # 2D-2U pattern - reversal DOWN = PUT
        metadata_2d_2u = PATTERN_REGISTRY.get("2D-2U-?")
        if metadata_2d_2u:
            assert metadata_2d_2u.is_bidirectional is False
            assert metadata_2d_2u.valid_directions == ('PUT',)

        # 2U-2D pattern - reversal UP = CALL
        metadata_2u_2d = PATTERN_REGISTRY.get("2U-2D-?")
        if metadata_2u_2d:
            assert metadata_2u_2d.is_bidirectional is False
            assert metadata_2u_2d.valid_directions == ('CALL',)


# =============================================================================
# get_pattern_metadata Tests
# =============================================================================

class TestGetPatternMetadata:
    """Test get_pattern_metadata function."""

    def test_get_existing_pattern(self):
        """Get metadata for existing pattern."""
        metadata = get_pattern_metadata("3-?")

        assert metadata is not None
        assert metadata.name == "3-?"
        assert metadata.is_bidirectional is True

    def test_get_nonexistent_pattern_returns_none(self):
        """Get metadata for nonexistent pattern returns None."""
        metadata = get_pattern_metadata("nonexistent-pattern")

        assert metadata is None

    def test_get_all_registered_patterns(self):
        """Get metadata for all registered patterns."""
        for pattern_name in PATTERN_REGISTRY.keys():
            metadata = get_pattern_metadata(pattern_name)
            assert metadata is not None
            assert metadata.name == pattern_name


# =============================================================================
# is_bidirectional_pattern Tests
# =============================================================================

class TestIsBidirectionalPattern:
    """Test is_bidirectional_pattern function."""

    def test_registered_bidirectional_pattern(self):
        """Registered bidirectional pattern returns True."""
        assert is_bidirectional_pattern("3-?") is True
        assert is_bidirectional_pattern("3-1-?") is True
        assert is_bidirectional_pattern("2U-1-?") is True
        assert is_bidirectional_pattern("2D-1-?") is True

    def test_registered_unidirectional_pattern(self):
        """Registered unidirectional pattern returns False."""
        assert is_bidirectional_pattern("3-2D-?") is False
        assert is_bidirectional_pattern("3-2U-?") is False

    def test_heuristic_inside_bar_pattern_bidirectional(self):
        """Unregistered pattern ending in -1-? is bidirectional."""
        # Not in registry, but ends with -1-?
        assert is_bidirectional_pattern("X-1-?") is True
        assert is_bidirectional_pattern("custom-1-?") is True

    def test_heuristic_2D_pattern_unidirectional(self):
        """Unregistered pattern with -2D-? is unidirectional."""
        # Not in registry, but has -2D-?
        assert is_bidirectional_pattern("custom-2D-?") is False
        assert is_bidirectional_pattern("X-2D-?") is False

    def test_heuristic_2U_pattern_unidirectional(self):
        """Unregistered pattern with -2U-? is unidirectional."""
        # Not in registry, but has -2U-?
        assert is_bidirectional_pattern("custom-2U-?") is False
        assert is_bidirectional_pattern("X-2U-?") is False

    def test_unknown_pattern_defaults_to_bidirectional(self):
        """Unknown pattern with no heuristic match defaults to bidirectional."""
        # No matching heuristic - default to True (safer)
        assert is_bidirectional_pattern("unknown-pattern") is True
        assert is_bidirectional_pattern("3") is True

    def test_complex_pattern_heuristics(self):
        """Complex patterns use heuristics correctly."""
        # Pattern with -2D-? in the middle
        assert is_bidirectional_pattern("3-2D-2U-?") is False

        # Pattern ending in -1-? takes precedence
        assert is_bidirectional_pattern("X-Y-1-?") is True


# =============================================================================
# get_valid_directions Tests
# =============================================================================

class TestGetValidDirections:
    """Test get_valid_directions function."""

    def test_registered_bidirectional_both_directions(self):
        """Registered bidirectional pattern returns both directions."""
        directions = get_valid_directions("3-?")

        assert 'CALL' in directions
        assert 'PUT' in directions
        assert len(directions) == 2

    def test_registered_unidirectional_call_only(self):
        """3-2D-? pattern returns CALL only (reversal UP)."""
        directions = get_valid_directions("3-2D-?")

        assert directions == ('CALL',)

    def test_registered_unidirectional_put_only(self):
        """3-2U-? pattern returns PUT only (reversal DOWN)."""
        directions = get_valid_directions("3-2U-?")

        assert directions == ('PUT',)

    def test_heuristic_bidirectional_both_directions(self):
        """Unregistered bidirectional pattern returns both directions."""
        directions = get_valid_directions("custom-1-?")

        assert 'CALL' in directions
        assert 'PUT' in directions

    def test_heuristic_2D_pattern_call_only(self):
        """Unregistered -2D-? pattern returns CALL only."""
        directions = get_valid_directions("X-2D-?")

        assert directions == ('CALL',)

    def test_heuristic_2U_pattern_put_only(self):
        """Unregistered -2U-? pattern returns PUT only."""
        directions = get_valid_directions("X-2U-?")

        assert directions == ('PUT',)

    def test_unknown_pattern_defaults_to_both(self):
        """Unknown pattern defaults to both directions."""
        directions = get_valid_directions("unknown")

        assert 'CALL' in directions
        assert 'PUT' in directions


# =============================================================================
# extract_setup_pattern_type Tests
# =============================================================================

class TestExtractSetupPatternType:
    """Test extract_setup_pattern_type function."""

    def test_already_setup_pattern_unchanged(self):
        """Pattern already in setup form is returned unchanged."""
        assert extract_setup_pattern_type("3-?") == "3-?"
        assert extract_setup_pattern_type("3-1-?") == "3-1-?"
        assert extract_setup_pattern_type("3-2D-?") == "3-2D-?"
        assert extract_setup_pattern_type("2U-1-?") == "2U-1-?"

    def test_completed_3_2U_pattern_to_setup(self):
        """Completed 3-2U pattern converts to 3-? setup."""
        # Wait, looking at the code more carefully:
        # "3-1-2U" -> "3-1-?" (removes the final -2U or -2D)
        result = extract_setup_pattern_type("3-1-2U")
        assert result == "3-1-?"

    def test_completed_3_2D_pattern_to_setup(self):
        """Completed 3-2D pattern converts to setup."""
        result = extract_setup_pattern_type("3-1-2D")
        assert result == "3-1-?"

    def test_completed_2U_pattern_to_setup(self):
        """Completed pattern ending in 2U converts to setup."""
        result = extract_setup_pattern_type("2D-2U")
        assert result == "2D-?"

    def test_completed_2D_pattern_to_setup(self):
        """Completed pattern ending in 2D converts to setup."""
        result = extract_setup_pattern_type("2U-2D")
        assert result == "2U-?"

    def test_pattern_without_match_unchanged(self):
        """Pattern not matching regex is returned unchanged."""
        result = extract_setup_pattern_type("3")
        assert result == "3"

        result = extract_setup_pattern_type("unknown")
        assert result == "unknown"

    def test_longer_pattern_sequences(self):
        """Longer pattern sequences convert correctly."""
        result = extract_setup_pattern_type("3-2D-2U")
        assert result == "3-2D-?"

        result = extract_setup_pattern_type("2U-1-2D")
        assert result == "2U-1-?"


# =============================================================================
# Integration Tests
# =============================================================================

class TestPatternRegistryIntegration:
    """Integration tests for pattern registry."""

    def test_bidirectional_patterns_have_both_directions(self):
        """All bidirectional patterns have both CALL and PUT valid."""
        for name, metadata in PATTERN_REGISTRY.items():
            if metadata.is_bidirectional:
                assert 'CALL' in metadata.valid_directions, f"{name} missing CALL"
                assert 'PUT' in metadata.valid_directions, f"{name} missing PUT"

    def test_unidirectional_patterns_have_single_direction(self):
        """All unidirectional patterns have exactly one valid direction."""
        for name, metadata in PATTERN_REGISTRY.items():
            if not metadata.is_bidirectional:
                assert len(metadata.valid_directions) == 1, f"{name} should have 1 direction"

    def test_get_pattern_metadata_matches_registry(self):
        """get_pattern_metadata returns same objects as registry."""
        for name, expected in PATTERN_REGISTRY.items():
            actual = get_pattern_metadata(name)
            assert actual is expected

    def test_is_bidirectional_matches_metadata(self):
        """is_bidirectional_pattern matches metadata.is_bidirectional."""
        for name, metadata in PATTERN_REGISTRY.items():
            assert is_bidirectional_pattern(name) == metadata.is_bidirectional, \
                f"{name} mismatch"

    def test_get_valid_directions_matches_metadata(self):
        """get_valid_directions matches metadata.valid_directions."""
        for name, metadata in PATTERN_REGISTRY.items():
            assert get_valid_directions(name) == metadata.valid_directions, \
                f"{name} mismatch"

    def test_reversal_patterns_logic(self):
        """Reversal patterns have correct direction logic.

        STRAT reversal patterns:
        - 3-2D-? (bearish bar) -> reversal UP = CALL
        - 3-2U-? (bullish bar) -> reversal DOWN = PUT
        """
        # 3-2D-? = bearish directional bar, reversal is CALL (break UP)
        metadata_3_2d = get_pattern_metadata("3-2D-?")
        assert metadata_3_2d.valid_directions == ('CALL',)

        # 3-2U-? = bullish directional bar, reversal is PUT (break DOWN)
        metadata_3_2u = get_pattern_metadata("3-2U-?")
        assert metadata_3_2u.valid_directions == ('PUT',)

    def test_inside_bar_patterns_bidirectional(self):
        """All inside bar patterns (-1-?) are bidirectional."""
        inside_bar_patterns = [
            name for name in PATTERN_REGISTRY.keys() if "-1-?" in name
        ]

        for pattern in inside_bar_patterns:
            metadata = get_pattern_metadata(pattern)
            assert metadata.is_bidirectional is True, f"{pattern} should be bidirectional"

    def test_extract_then_lookup(self):
        """Extract setup type then lookup in registry."""
        # Completed pattern -> setup form -> lookup
        completed = "3-1-2U"
        setup_type = extract_setup_pattern_type(completed)
        metadata = get_pattern_metadata(setup_type)

        assert metadata is not None
        assert metadata.name == "3-1-?"
        assert metadata.is_bidirectional is True

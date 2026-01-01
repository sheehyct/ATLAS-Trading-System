"""
Pattern Registry - Single Source of Truth for STRAT Pattern Metadata

Session EQUITY-41: Created to fix Bug #6 (bidirectional logic applied incorrectly)

This registry defines which patterns are BIDIRECTIONAL vs UNIDIRECTIONAL:

BIDIRECTIONAL patterns (break determines direction):
    - 3-?     : Outside bar - break up = 3-2U CALL, break down = 3-2D PUT
    - 3-1-?   : Outside-Inside - break determines direction
    - 2-1-?   : Directional-Inside - break determines direction
    - X-1-?   : Any bar followed by Inside - break determines direction

UNIDIRECTIONAL patterns (reversal only, opposite break invalidates):
    - 3-2D-?  : Only break UP triggers (3-2D-2U CALL reversal)
    - 3-2U-?  : Only break DOWN triggers (3-2U-2D PUT reversal)
    - X-2D-?  : Only break UP triggers (X-2D-2U CALL reversal)
    - X-2U-?  : Only break DOWN triggers (X-2U-2D PUT reversal)

Usage:
    from strat.pattern_registry import is_bidirectional_pattern, get_pattern_metadata

    if is_bidirectional_pattern(signal.pattern_type):
        # Check both directions
    else:
        # Only check declared direction
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass(frozen=True)
class PatternMetadata:
    """Metadata for a STRAT pattern type."""
    name: str
    is_bidirectional: bool
    bars_required: int
    description: str
    valid_directions: tuple  # ('CALL', 'PUT') or ('CALL',) or ('PUT',)


# =============================================================================
# PATTERN REGISTRY - Single Source of Truth
# =============================================================================

PATTERN_REGISTRY = {
    # =========================================================================
    # BIDIRECTIONAL patterns (break determines direction)
    # These patterns can trigger either CALL or PUT based on which way breaks
    # =========================================================================

    "3-?": PatternMetadata(
        name="3-?",
        is_bidirectional=True,
        bars_required=1,
        description="Outside bar setup - break up = 3-2U CALL, break down = 3-2D PUT",
        valid_directions=('CALL', 'PUT'),
    ),

    "3-1-?": PatternMetadata(
        name="3-1-?",
        is_bidirectional=True,
        bars_required=2,
        description="Outside-Inside setup - break up = 3-1-2U CALL, break down = 3-1-2D PUT",
        valid_directions=('CALL', 'PUT'),
    ),

    "2U-1-?": PatternMetadata(
        name="2U-1-?",
        is_bidirectional=True,
        bars_required=2,
        description="2U-Inside setup - break determines direction",
        valid_directions=('CALL', 'PUT'),
    ),

    "2D-1-?": PatternMetadata(
        name="2D-1-?",
        is_bidirectional=True,
        bars_required=2,
        description="2D-Inside setup - break determines direction",
        valid_directions=('CALL', 'PUT'),
    ),

    # =========================================================================
    # UNIDIRECTIONAL patterns (reversal only)
    # These patterns only trigger in ONE direction - opposite break INVALIDATES
    # =========================================================================

    # 3-2-2 reversal patterns (3-bar patterns)
    "3-2D-?": PatternMetadata(
        name="3-2D-?",
        is_bidirectional=False,
        bars_required=2,
        description="3-2D setup - ONLY break UP triggers (3-2D-2U CALL reversal)",
        valid_directions=('CALL',),
    ),

    "3-2U-?": PatternMetadata(
        name="3-2U-?",
        is_bidirectional=False,
        bars_required=2,
        description="3-2U setup - ONLY break DOWN triggers (3-2U-2D PUT reversal)",
        valid_directions=('PUT',),
    ),

    # 2-2 reversal patterns (2-bar patterns becoming 3-bar)
    "2D-2U-?": PatternMetadata(
        name="2D-2U-?",
        is_bidirectional=False,
        bars_required=2,
        description="2D-2U setup - ONLY break DOWN triggers (reversal PUT)",
        valid_directions=('PUT',),
    ),

    "2U-2D-?": PatternMetadata(
        name="2U-2D-?",
        is_bidirectional=False,
        bars_required=2,
        description="2U-2D setup - ONLY break UP triggers (reversal CALL)",
        valid_directions=('CALL',),
    ),
}


# =============================================================================
# LOOKUP FUNCTIONS
# =============================================================================

def get_pattern_metadata(pattern_name: str) -> Optional[PatternMetadata]:
    """
    Get metadata for a pattern by exact name.

    Args:
        pattern_name: Pattern name (e.g., "3-2D-?", "3-1-?")

    Returns:
        PatternMetadata if found, None otherwise
    """
    return PATTERN_REGISTRY.get(pattern_name)


def is_bidirectional_pattern(pattern_name: str) -> bool:
    """
    Check if a pattern is bidirectional.

    For patterns not in the registry, uses heuristics:
    - Patterns ending in "-1-?" are bidirectional (inside bar setups)
    - Patterns ending in "-2D-?" or "-2U-?" are unidirectional (reversal only)
    - Default to True for safety (check both directions)

    Args:
        pattern_name: Pattern name or bar_sequence (e.g., "3-2D-?", "2U-1-?")

    Returns:
        True if bidirectional, False if unidirectional
    """
    # Check registry first
    metadata = PATTERN_REGISTRY.get(pattern_name)
    if metadata:
        return metadata.is_bidirectional

    # Heuristics for patterns not in registry
    # Pattern ending with inside bar ("-1-?") = bidirectional
    if pattern_name.endswith("-1-?"):
        return True

    # Pattern with directional bar before "?" = unidirectional reversal
    # e.g., "X-2D-?" or "X-2U-?" or "3-2D-?" or "2U-2D-?"
    if "-2D-?" in pattern_name or "-2U-?" in pattern_name:
        return False

    # Default to bidirectional (safer - checks both directions)
    return True


def get_valid_directions(pattern_name: str) -> tuple:
    """
    Get valid directions for a pattern.

    Args:
        pattern_name: Pattern name

    Returns:
        Tuple of valid directions, e.g., ('CALL', 'PUT') or ('CALL',)
    """
    metadata = PATTERN_REGISTRY.get(pattern_name)
    if metadata:
        return metadata.valid_directions

    # Heuristics for unregistered patterns
    if is_bidirectional_pattern(pattern_name):
        return ('CALL', 'PUT')

    # Unidirectional - determine from pattern
    if "-2D-?" in pattern_name:
        return ('CALL',)  # 2D reversal = break UP = CALL
    if "-2U-?" in pattern_name:
        return ('PUT',)   # 2U reversal = break DOWN = PUT

    return ('CALL', 'PUT')  # Default


def extract_setup_pattern_type(bar_sequence: str) -> str:
    """
    Extract the setup pattern type from a bar sequence.

    Examples:
        "3-2D-?" -> "3-2D-?"
        "2U-1-?" -> "2U-1-?"
        "3-1-2U" -> "3-1-?" (completed pattern -> setup form)

    Args:
        bar_sequence: Full bar sequence string

    Returns:
        Setup pattern type for registry lookup
    """
    # Already a setup pattern
    if bar_sequence.endswith("-?"):
        return bar_sequence

    # Convert completed pattern to setup form
    # e.g., "3-1-2U" -> "3-1-?"
    if re.match(r'.*-2[UD]$', bar_sequence):
        return re.sub(r'-2[UD]$', '-?', bar_sequence)

    return bar_sequence

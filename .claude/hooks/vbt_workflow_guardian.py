#!/usr/bin/env python3
"""
VBT 5-Step Workflow Guardian Hook (PreToolUse)

Purpose: Enforce the VBT 5-Step Workflow (SEARCH -> VERIFY -> FIND -> TEST -> IMPLEMENT)
when writing code that uses VectorBT Pro functions.

Hook Type: PreToolUse
Exit Code: 2 (strict block - stops the tool call AND stops Claude from continuing)

The 5-Step Workflow prevents implementation failures from:
- Using deprecated/non-existent VBT methods
- Incorrect parameter usage
- Missing proper VBT patterns

Required markers in code to indicate workflow compliance:
- VBT_VERIFIED: <method_name> - indicates method was verified via resolve_refnames
- VBT_TESTED: <description> - indicates code was tested via run_code
- Reference to VBT documentation in comments

This hook fires BEFORE a tool is executed and checks for workflow compliance.
"""

import json
import sys
import re
from pathlib import Path

# VBT function patterns that require verification
VBT_PATTERNS = [
    # Data fetching
    r'vbt\.\w*Data\.pull',
    r'vbt\.YFData',
    r'vbt\.AlpacaData',
    r'vbt\.BinanceData',
    r'vbt\.PolygonData',

    # Portfolio
    r'vbt\.Portfolio',
    r'vbt\.PF',
    r'\.from_signals\s*\(',
    r'\.from_order_func\s*\(',
    r'\.from_holding\s*\(',

    # Indicators
    r'vbt\.talib\s*\(',
    r'vbt\.ATR',
    r'vbt\.RSI',
    r'vbt\.MACD',
    r'vbt\.MA\.',
    r'vbt\.BBANDS',
    r'vbt\.indicators',

    # Analytics
    r'\.total_return',
    r'\.sharpe_ratio',
    r'\.max_drawdown',
    r'\.calmar_ratio',
    r'\.sortino_ratio',

    # Accessors
    r'\.vbt\.',
    r'\.vbt\.plot',
    r'\.vbt\.returns',

    # Run configurations
    r'vbt\.run\s*\(',
    r'vbt\.optimize\s*\(',
]

# Files that should be checked for VBT workflow compliance
STRATEGY_FILE_PATTERNS = [
    r'strategies/.*\.py$',
    r'scripts/backtest.*\.py$',
    r'tests/test_strategies/.*\.py$',
]

# Tools that modify files
WRITE_TOOLS = ['Write', 'Edit', 'NotebookEdit']

# Verification markers that indicate workflow compliance
VERIFICATION_MARKERS = [
    r'VBT_VERIFIED:\s*\w+',           # VBT_VERIFIED: method_name
    r'VBT_TESTED:\s*.+',               # VBT_TESTED: description
    r'#\s*Verified via.*vbt',          # # Verified via vbt MCP
    r'#\s*5-step workflow complete',   # Explicit marker
    r'#\s*mcp__vectorbt-pro',          # Reference to MCP tool usage
    r'vectorbt\.pro.*docs',            # Reference to VBT docs
    r'vbt\.search\s*\(',               # Used vbt.search
]


def is_strategy_file(file_path: str) -> bool:
    """Check if the file is a strategy-related file that needs VBT verification."""
    if not file_path:
        return False

    normalized = file_path.replace('\\', '/')

    for pattern in STRATEGY_FILE_PATTERNS:
        if re.search(pattern, normalized):
            return True

    return False


def contains_vbt_functions(content: str) -> list:
    """Check if content contains VBT function calls that need verification."""
    found_patterns = []

    for pattern in VBT_PATTERNS:
        matches = re.findall(pattern, content)
        if matches:
            found_patterns.extend(matches)

    return found_patterns


def has_verification_markers(content: str) -> bool:
    """Check if content has markers indicating VBT 5-step workflow was followed."""
    for marker in VERIFICATION_MARKERS:
        if re.search(marker, content, re.IGNORECASE):
            return True
    return False


def check_conversation_context(hook_input: dict) -> bool:
    """
    Check if VBT MCP tools were used recently in the conversation.

    This looks for markers that indicate proper workflow was followed.
    Since hooks don't have direct conversation access, we check for
    patterns in the content being written.
    """
    tool_input = hook_input.get('tool_input', {})
    content = tool_input.get('content', '') or tool_input.get('new_string', '')

    # Check for workflow markers
    workflow_indicators = [
        'resolve_refnames',
        'mcp__vectorbt-pro',
        'run_code',
        'vbt.search',
        'verified',
        '5-step',
        'VBT_VERIFIED',
        'VBT_TESTED',
    ]

    content_lower = content.lower()
    for indicator in workflow_indicators:
        if indicator.lower() in content_lower:
            return True

    return False


def main():
    """Main hook entry point."""
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        # If we can't parse input, allow (fail open)
        sys.exit(0)

    # Get tool name and input
    tool_name = hook_input.get('tool_name', '')
    tool_input = hook_input.get('tool_input', {})

    # Only check write operations
    if tool_name not in WRITE_TOOLS:
        sys.exit(0)

    # Get the file path being modified
    file_path = tool_input.get('file_path', '')

    # Only check strategy files
    if not is_strategy_file(file_path):
        sys.exit(0)

    # Get content being written
    content = tool_input.get('content', '') or tool_input.get('new_string', '')

    if not content:
        sys.exit(0)

    # Check if content uses VBT functions
    vbt_functions = contains_vbt_functions(content)

    if not vbt_functions:
        # No VBT functions, allow
        sys.exit(0)

    # Check for verification markers or context
    if has_verification_markers(content) or check_conversation_context(hook_input):
        sys.exit(0)

    # BLOCK: VBT functions detected without verification markers
    unique_functions = list(set(vbt_functions))[:5]  # Show first 5 unique

    block_message = {
        "decision": "block",
        "reason": (
            "VBT 5-STEP WORKFLOW GUARD: VectorBT Pro code detected without verification.\n\n"
            f"Detected VBT patterns: {', '.join(unique_functions)}\n\n"
            "The VBT 5-Step Workflow is MANDATORY for all VBT implementations:\n"
            "1. SEARCH: mcp__vectorbt-pro__search() for patterns\n"
            "2. VERIFY: resolve_refnames() to confirm methods exist\n"
            "3. FIND: mcp__vectorbt-pro__find() for real-world usage\n"
            "4. TEST: mcp__vectorbt-pro__run_code() minimal example\n"
            "5. IMPLEMENT: Only after steps 1-4 pass\n\n"
            "REQUIRED ACTION:\n"
            "- Complete the 5-step workflow for each VBT function used\n"
            "- Add verification markers to indicate compliance:\n"
            "  # VBT_VERIFIED: Portfolio.from_signals\n"
            "  # VBT_TESTED: Backtest with sample data works\n\n"
            f"Blocked file: {file_path}"
        )
    }
    print(json.dumps(block_message))

    # Exit code 2 = STRICT BLOCK
    sys.exit(2)


if __name__ == '__main__':
    main()

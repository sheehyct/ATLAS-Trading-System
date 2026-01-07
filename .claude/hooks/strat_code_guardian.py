#!/usr/bin/env python3
"""
STRAT Code Guardian Hook (PreToolUse)

Purpose: Prevent modifications to STRAT methodology files without
having consulted the strat-methodology skill first.

Hook Type: PreToolUse
Exit Code: 2 (strict block - stops the tool call AND stops Claude from continuing)

This hook fires BEFORE a tool is executed. It checks if Claude is about to
modify STRAT skill files and blocks if the skill hasn't been referenced.

IMPORTANT: This implements "Strict Block" behavior (exit code 2).
When blocked, Claude MUST stop and ask the user for guidance.
"""

import json
import sys
import os

# Path patterns for STRAT methodology skill files
STRAT_SKILL_PATHS = [
    'strat-methodology/SKILL.md',
    'strat-methodology/PATTERNS.md',
    'strat-methodology/TIMEFRAMES.md',
    'strat-methodology/EXECUTION.md',
    'strat-methodology/OPTIONS.md',
    'strat-methodology/references/',
]

# Tools that modify files
WRITE_TOOLS = ['Write', 'Edit', 'NotebookEdit']

# Environment variable to track skill usage (set by skill invocation)
SKILL_USED_ENV = 'STRAT_SKILL_INVOKED'


def is_strat_file(file_path: str) -> bool:
    """Check if the file path is a STRAT methodology file."""
    if not file_path:
        return False

    # Normalize path separators
    normalized = file_path.replace('\\', '/')

    for pattern in STRAT_SKILL_PATHS:
        if pattern in normalized:
            return True

    return False


def skill_was_invoked() -> bool:
    """
    Check if the strat-methodology skill was invoked in this session.

    NOTE: This is a simplified check. In production, you might want to:
    - Check session context/transcript
    - Use a more robust state tracking mechanism
    - Check for skill tool calls in the conversation

    For now, we rely on the environment variable being set by skill invocation.
    """
    return os.environ.get(SKILL_USED_ENV, '').lower() == 'true'


def check_conversation_context(hook_input: dict) -> bool:
    """
    Check if skill was referenced in the conversation.

    This looks at the session_id or conversation markers to determine
    if the skill has been properly consulted.
    """
    # Check if the Skill tool was called with strat-methodology
    # This requires access to conversation history which hooks don't have directly
    # Instead, we check for markers in the tool input

    tool_input = hook_input.get('tool_input', {})

    # If the file is being modified and contains a clear skill reference comment,
    # we allow it (indicates skill-guided modification)
    content = tool_input.get('content', '') or tool_input.get('new_string', '')

    # Check for markers that indicate skill-guided work
    skill_markers = [
        'strat-methodology',
        'STRAT skill',
        '/strat',
        'Per STRAT methodology',
        'According to STRAT',
    ]

    for marker in skill_markers:
        if marker.lower() in content.lower():
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

    # Only check STRAT methodology files
    if not is_strat_file(file_path):
        sys.exit(0)

    # Check if skill was invoked or context shows skill-guided work
    if skill_was_invoked() or check_conversation_context(hook_input):
        sys.exit(0)

    # BLOCK: Attempting to modify STRAT file without skill consultation
    block_message = {
        "decision": "block",
        "reason": (
            "STRAT METHODOLOGY GUARD: Modification to STRAT skill files blocked.\n\n"
            "You are attempting to modify a STRAT methodology file without having "
            "consulted the strat-methodology skill first.\n\n"
            "REQUIRED ACTION:\n"
            "1. Use /strat-methodology skill to load the correct methodology\n"
            "2. Reference the skill documentation for the changes you want to make\n"
            "3. Ensure your modifications align with validated STRAT rules\n\n"
            "This guard exists because previous sessions introduced methodology errors "
            "when not following the documented patterns.\n\n"
            f"Blocked file: {file_path}"
        )
    }
    print(json.dumps(block_message))

    # Exit code 2 = STRICT BLOCK
    # Claude MUST stop and ask user for guidance
    sys.exit(2)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
STRAT Prompt Validator Hook (UserPromptSubmit)

Purpose: Detect STRAT-related user prompts and advise Claude to use
the strat-methodology skill for accuracy.

Hook Type: UserPromptSubmit
Exit Code: 0 (advisory - message is shown but doesn't block)

This hook fires when the user submits a prompt. It checks for STRAT-related
keywords and outputs an advisory message reminding Claude to use the skill.
"""

import json
import sys
import re

# Keywords that indicate STRAT-related work
STRAT_KEYWORDS = [
    # Bar types
    r'\btype\s*[123]\b',
    r'\b2[UD]\b',
    r'\binside\s*bar\b',
    r'\boutside\s*bar\b',

    # Patterns
    r'\b2-1-2\b',
    r'\b3-1-2\b',
    r'\b2-2\b',
    r'\b3-2\b',
    r'\brev\s*strat\b',

    # Methodology concepts
    r'\bSTRAT\b',
    r'\bstrat\s*method',
    r'\btimeframe\s*continuity\b',
    r'\bTFC\b',
    r'\bFTFC\b',
    r'\b4\s*C\'?s?\b',
    r'\bMOAF\b',
    r'\bmother\s*of\s*all',

    # Entry mechanics
    r'\btrigger\s*level\b',
    r'\bmagnitude\b',
    r'\bentry\s*bar\b',
    r'\bsetup\s*bar\b',

    # Pattern files
    r'PATTERNS\.md',
    r'TIMEFRAMES\.md',
    r'EXECUTION\.md',
    r'OPTIONS\.md',
    r'SKILL\.md',
    r'strat-methodology',
]

def is_strat_related(prompt: str) -> bool:
    """Check if prompt contains STRAT-related content."""
    for pattern in STRAT_KEYWORDS:
        if re.search(pattern, prompt, re.IGNORECASE):
            return True
    return False


def main():
    """Main hook entry point."""
    # Read hook input from stdin
    try:
        hook_input = json.load(sys.stdin)
    except json.JSONDecodeError:
        # If we can't parse input, exit silently
        sys.exit(0)

    # Get the user's prompt
    prompt = hook_input.get('prompt', '')
    if not prompt:
        sys.exit(0)

    # Check if this is STRAT-related
    if is_strat_related(prompt):
        # Output advisory message
        message = {
            "message": "STRAT-related query detected. Remember to use /strat-methodology skill for accurate methodology implementation. The skill contains validated patterns, entry mechanics, TFC scoring, and position management rules."
        }
        print(json.dumps(message))

    # Exit code 0 = advisory (continue with prompt)
    sys.exit(0)


if __name__ == '__main__':
    main()

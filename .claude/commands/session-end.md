---
name: session-end
description: Automate ATLAS development session closure with quality checks
---

# Session End Command

Execute the following steps in order. Pause for user confirmation where noted.

## Step 1: Run Code Simplifier

Run the `code-simplifier` plugin on files modified this session to identify simplification opportunities.

## Step 2: Run Pre-Commit Checks

Execute `/pre-commit` command (or perform these checks inline):

1. Check for changed files: `git status`
2. If trade execution code changed, run silent-failure-hunter review
3. Check HANDOFF.md line count: should be < 1500 lines
4. Check if README.md reflects current system state
5. Check if CLAUDE.md has unnecessary bloat

Report any issues found. If critical issues, STOP and ask user how to proceed.

## Step 3: Update Documentation

### Update .session_startup_prompt.md

Replace contents with template:
```markdown
# Session Startup Prompt - EQUITY-{N+1}

**Previous Session:** EQUITY-{N} ({today's date})
**Current Branch:** `{branch}`
**VPS Status:** {status from this session}
**Plan Mode:** {ON/OFF recommendation}

---

## EQUITY-{N} Accomplishments

{List 3-5 key accomplishments from this session}

---

## EQUITY-{N+1} Priorities

### Priority 1: {highest priority}
{Details}

### Priority 2: {second priority}
{Details}

### Priority 3: {third priority}
{Details}

---

## Reference Documents

- HANDOFF: `docs/HANDOFF.md`
- CLAUDE.md: `docs/CLAUDE.md` (STRICT COMPLIANCE)
- {Any active plan files}
```

### Update docs/HANDOFF.md

Add new session entry at TOP of file:
```markdown
## Session EQUITY-{N}: {Brief Title} (COMPLETE)

**Date:** {today's date}
**Environment:** Claude Code Desktop (Opus 4.5)
**Status:** COMPLETE - {one line summary}

### What Was Accomplished

{Detailed list of accomplishments}

### Files Modified

{List of files changed with brief description}

---
```

If HANDOFF.md exceeds 1500 lines after update:
- STOP and ask user: "HANDOFF.md is {X} lines. Archive older sessions?"
- If yes, move older sessions to `docs/session_archive/`

## Step 4: Store in OpenMemory

Store these facts in OpenMemory:
- Session number and date
- Key accomplishments (2-3 items)
- Any important decisions made
- Blockers or issues encountered
- Tags: ["session-{N}", "ATLAS", "{relevant-feature}"]

## Step 5: Backup OpenMemory

Execute:
```bash
copy "C:\Dev\openmemory\data\atlas_memory.sqlite" "C:\Dev\openmemory\backups\atlas_memory_{YYYY-MM-DD}.sqlite"
```

## Step 6: Prepare Commit

Generate conventional commit message based on changes:
- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation only
- `test:` for test additions
- `refactor:` for code restructuring

Show user:
```
PROPOSED COMMIT
===============
Message: {conventional commit message}

Files:
{git diff --stat output}

Proceed with commit? (y/n)
```

Wait for user confirmation before committing.

## Step 7: Output Session Summary

```
SESSION EQUITY-{N} COMPLETE
===========================

ACCOMPLISHED:
- {task 1}
- {task 2}
- {task 3}

BLOCKERS: {none | list blockers}

NEXT SESSION PRIORITIES:
1. {priority 1}
2. {priority 2}
3. {priority 3}

PLAN MODE RECOMMENDATION: {ON | OFF}
Reason: {why on or off}

REFERENCE PLANS:
- {plan file if multi-session work in progress}

Documentation updated. OpenMemory stored. Ready for commit.
```

## Rules

- Do NOT commit without user confirmation
- Do NOT skip quality checks
- Do NOT archive HANDOFF.md without user confirmation
- Do NOT run test suite (already run in /session-start)
- ALWAYS run code-simplifier plugin first
- Keep summaries concise but complete

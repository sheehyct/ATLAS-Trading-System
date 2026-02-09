---
name: session-start
description: Automate ATLAS development session startup sequence
---

# Session Start Command

Execute the following steps in order:

## Step 1: Read Session Context Files

Read these files and summarize key points:

1. `.session_startup_prompt.md` - Current session priorities
2. `docs/HANDOFF.md` - Recent session history (focus on last 2-3 sessions)
3. `docs/CLAUDE.md` - Development rules (STRICT COMPLIANCE)

## Step 2: Query OpenMemory

Search OpenMemory for recent session context:
- Query: "ATLAS session" (last 5 results)
- Extract: Key decisions, blockers, pending work

## Step 3: Log Session for Resume

Log this session to `docs/SESSION_LOG.md` so it can be easily resumed later:

1. Find the current session ID by running:
   ```bash
   ls -t "$USERPROFILE/.claude/projects/C--Strat-Trading-Bot-vectorbt-workspace/"*.jsonl | head -1 | sed 's/.*\///' | sed 's/\.jsonl//'
   ```
   If that fails (empty $USERPROFILE), try:
   ```bash
   ls -t /c/Users/sheeh/.claude/projects/C--Strat-Trading-Bot-vectorbt-workspace/*.jsonl | head -1 | sed 's/.*\///' | sed 's/\.jsonl//'
   ```
2. Get the session name (EQUITY-{N}) from `.session_startup_prompt.md`
3. Get the current date/time
4. Prepend an entry to `docs/SESSION_LOG.md` (after the `<!-- New sessions -->` comment) in this format:
   ```
   ## EQUITY-{N} - {YYYY-MM-DD HH:MM}
   - **Resume:** `claude --resume {session-id}`
   - **Search:** `claude --resume "EQUITY-{N}"`
   - **Branch:** {current git branch}
   ```

## Step 4: Output Session Brief

Format your output as:

```
SESSION STARTUP COMPLETE
========================

Session: EQUITY-{N} (from .session_startup_prompt.md)
Branch: {current git branch}

TODAY'S PRIORITIES:
1. {from .session_startup_prompt.md}
2. {from .session_startup_prompt.md}
3. {from .session_startup_prompt.md}

RECENT CONTEXT:
- {Key point from HANDOFF.md}
- {Key point from OpenMemory}

REMINDERS:
- strat-methodology skill MANDATORY for any STRAT work
- Accuracy over speed
- VBT 5-step workflow for any VBT implementation

Awaiting previous session summary from user...
```

## Step 5: Wait for User Input

After outputting the brief, wait for the user to paste the previous session summary.

Do NOT start any development work until the user provides direction.

## Rules

- Do NOT skip any steps
- Do NOT summarize CLAUDE.md - just confirm it was read
- Do NOT start work without user direction
- Do NOT run test suite (use /test-focus when needed)

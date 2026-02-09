---
name: pre-commit
description: Run quality checks before committing code
---

# Pre-Commit Quality Checks

Execute all checks and report results. Do NOT auto-fix issues - report them for user decision.

Note: Test suite is NOT run here (already run in /session-start). Use /test-focus if you need to verify tests before committing.

## Check 1: Changed Files Review

```bash
git diff --name-only
git diff --stat
```

For each changed file in `strat/` or `strat/signal_automation/`:
- Flag for methodology review
- If trade execution code: note for silent-failure-hunter

Status: {X files changed, Y need review}

## Check 2: Silent Failure Hunter (if applicable)

If any of these files changed:
- `strat/signal_automation/executor.py`
- `strat/signal_automation/position_monitor.py`
- `strat/signal_automation/daemon.py`
- `strat/signal_automation/entry_monitor.py`

Run code review focusing on:
- Bare except clauses
- Silently swallowed exceptions
- Missing error logging
- Functions that fail without indication

Status: PASS / NEEDS ATTENTION

## Check 3: HANDOFF.md Size

```bash
wc -l docs/HANDOFF.md
```

- If < 1500 lines: PASS
- If >= 1500 lines: NEEDS ARCHIVE

Status: {X lines} - PASS / NEEDS ARCHIVE

## Check 4: README.md Accuracy

Compare README.md claims against actual system:
- Are listed features actually implemented?
- Are version numbers current?
- Are installation instructions accurate?
- Are any deprecated features still listed?

Status: PASS / NEEDS UPDATE (list issues)

## Check 5: CLAUDE.md Health

Check docs/CLAUDE.md for:
- Duplicate rules
- Outdated references
- Excessive length (should be < 300 lines for token efficiency)
- Rules that contradict each other

Status: PASS / NEEDS CLEANUP (list issues)

## Summary Output

```
PRE-COMMIT CHECK RESULTS
========================

[PASS/FAIL] Changed Files: X files, Y flagged for review
[PASS/SKIP] Silent Failure Hunter: {status}
[PASS/WARN] HANDOFF.md: {X lines}
[PASS/WARN] README.md: {status}
[PASS/WARN] CLAUDE.md: {status}

OVERALL: READY TO COMMIT / ISSUES TO ADDRESS

{If issues, list them with recommended actions}
```

## Rules

- Report ALL issues, even minor ones
- Do NOT auto-fix anything
- Do NOT skip checks
- Do NOT run test suite (use /test-focus if needed)
- User decides whether to commit with issues or fix first

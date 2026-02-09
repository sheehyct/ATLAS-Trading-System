---
name: pull-logs
description: Fetch VPS daemon logs and analyze for issues
---

# Pull Logs Command

Fetch logs from the ATLAS VPS daemon and analyze for issues.

## Parameters

- `--since <time>` - How far back to pull (default: "1 hour ago")
- `--grep <pattern>` - Filter for specific pattern (optional)

## Step 1: SSH and Fetch Logs

Connect to VPS and pull daemon logs:

```bash
ssh atlas@178.156.223.251 "sudo journalctl -u atlas-daemon --since '{since_time}' --no-pager"
```

If `--grep` specified:
```bash
ssh atlas@178.156.223.251 "sudo journalctl -u atlas-daemon --since '{since_time}' --no-pager | grep '{pattern}'"
```

## Step 2: Parse Log Entries

Scan logs for these key patterns:

### Entry Events
- Pattern: `ENTRY TRIGGERED` or `EXECUTING ENTRY`
- Extract: Symbol, pattern type, timeframe, entry price, time

### Exit Events
- Pattern: `EXIT TRIGGERED` or `POSITION CLOSED`
- Extract: Symbol, exit reason, P/L, time

### TFC Re-evaluation
- Pattern: `TFC REEVAL`
- Extract: Symbol, original TFC, new TFC, decision (allowed/blocked)

### Pattern Invalidation
- Pattern: `PATTERN INVALIDATED` or `TYPE 3 EVOLUTION`
- Extract: Symbol, original pattern, what happened

### Errors/Warnings
- Pattern: `ERROR` or `WARNING` or `Exception`
- Extract: Full message, timestamp

### Stale Setup Detection
- Pattern: `STALE SETUP` or `setup_stale`
- Extract: Symbol, timeframe, age

## Step 3: Categorize Issues

Group findings into:

1. **Methodology Issues** (route to @strat-validator)
   - Incorrect pattern classification
   - Wrong entry timing
   - TFC scoring errors

2. **Execution Issues** (route to @trade-reviewer)
   - Missed entries
   - Wrong exit reasons
   - P/L discrepancies

3. **Data Issues** (route to @data-auditor)
   - Missing data errors
   - Timezone problems
   - API failures

4. **System Issues**
   - Connection errors
   - Memory/performance issues
   - Unexpected exceptions

## Step 4: Output Report

```
VPS LOG ANALYSIS
================
Time Range: {since_time} to now
Total Lines: {X}

ENTRIES: {count}
{table: symbol | pattern | timeframe | price | time}

EXITS: {count}
{table: symbol | reason | P/L | time}

TFC RE-EVALUATIONS: {count}
- Allowed: {X}
- Blocked: {Y}
{list any blocked with reason}

PATTERN INVALIDATIONS: {count}
{list each with details}

ERRORS/WARNINGS: {count}
{list each with timestamp}

ISSUES REQUIRING ATTENTION:
---------------------------
{Categorized list of issues}

RECOMMENDED FOLLOW-UP:
- {agent recommendation if issues found}
- {specific action items}
```

## Step 5: Agent Routing (Optional)

If significant issues found, offer to route:

```
Issues found that may need deeper analysis:
- {count} methodology issues → @strat-validator
- {count} execution issues → @trade-reviewer
- {count} data issues → @data-auditor

Would you like me to invoke an agent for detailed analysis? (y/n)
```

## Rules

- If SSH fails, report error and suggest checking VPS status
- Do NOT take any action on findings without user direction
- Summarize, don't dump raw logs (unless user requests raw output)
- Flag anything unusual even if not an error

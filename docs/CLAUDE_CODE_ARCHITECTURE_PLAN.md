# Claude Code Architecture Plan - ATLAS Trading System

**Created:** January 17, 2026
**Status:** APPROVED - Pending skill completion
**Implementation Date:** After Monday skill review

---

## Executive Summary

This document defines the Claude Code agent, command, and skill architecture for the ATLAS trading system. The goal is to leverage Claude Code's native capabilities (agents, commands, hooks, skills) to improve development workflow, enforce methodology compliance, and reduce repetitive manual tasks.

**Key Decisions:**
- 6 agents for specialized domains
- 6 commands for workflow automation
- 8 skills with tiered activation (hook-enforced, agent-loaded, explicit)
- Project-level `.claude/` as primary location (version controlled)
- No autonomous market agents (existing daemon.py is superior for that)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERACTION                          │
├─────────────────────────────────────────────────────────────────┤
│  /session-start  /session-end  /pull-logs  /tech-debt           │
│  /pre-commit     /test-focus                                     │
├─────────────────────────────────────────────────────────────────┤
│                          COMMANDS                                │
│         (Orchestrate workflows, invoke agents/skills)            │
├─────────────────────────────────────────────────────────────────┤
│  @strat-validator  @test-runner  @backtest-reviewer             │
│  @data-auditor     @trade-reviewer  @vbt-researcher             │
├─────────────────────────────────────────────────────────────────┤
│                           AGENTS                                 │
│            (Isolated context, specialized analysis)              │
├─────────────────────────────────────────────────────────────────┤
│  strat-methodology (HOOK)    backtesting-validation (HOOK)      │
│  options-trading (AGENT)     position-sizing-risk (EXPLICIT)    │
│  momentum-strategies         mean-reversion-strategies          │
│  execution-quality           portfolio-construction             │
├─────────────────────────────────────────────────────────────────┤
│                           SKILLS                                 │
│              (Knowledge bases, methodology guides)               │
├─────────────────────────────────────────────────────────────────┤
│  strat_code_guardian.py      skill_enforcement.py (NEW)         │
│  strat_prompt_validator.py   vbt_workflow_guardian.py           │
├─────────────────────────────────────────────────────────────────┤
│                           HOOKS                                  │
│              (Enforce compliance, block bad actions)             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agents (6 Total)

### 1. strat-validator

**Purpose:** Validates STRAT methodology compliance in code changes

**Model:** Sonnet (balanced speed/quality)

**Skills Loaded:**
- strat-methodology
- backtesting-validation

**Use Cases:**
- Review PR/changes touching `strat/` directory
- Validate bar classification logic
- Check entry/exit timing implementation
- Verify pattern detection accuracy

**Read-Only:** Yes (analyzes, does not modify)

---

### 2. test-runner

**Purpose:** Runs tests, diagnoses failures (does NOT auto-fix)

**Model:** Haiku (fast, simple task)

**Skills Loaded:** None

**Use Cases:**
- Run full test suite or targeted tests
- Analyze failure root causes
- Categorize failures (code bug vs test bug vs intentional change)
- Provide diagnostic report

**Read-Only:** No (runs bash commands)

**Output Format:**
```
FAILURE: test_stale_hourly_setup_rejected
ROOT CAUSE: [Code change | Test bug | Spec change | Unknown]
HYPOTHESIS: [Specific explanation]
RECOMMENDATION: [Action to take]
```

---

### 3. backtest-reviewer

**Purpose:** Reviews backtest results for statistical validity

**Model:** Sonnet

**Skills Loaded:**
- backtesting-validation

**Use Cases:**
- Review strategy backtest results
- Check for overfitting indicators
- Validate walk-forward methodology
- Assess Monte Carlo results

**Read-Only:** Yes

---

### 4. data-auditor

**Purpose:** Verifies data sources, timezone compliance, no synthetic data

**Model:** Sonnet

**Skills Loaded:** None (uses CLAUDE.md rules)

**Use Cases:**
- Audit data fetching code for compliance
- Check timezone handling (`tz='America/New_York'`)
- Verify no yfinance for equities
- Check weekend/holiday filtering

**Read-Only:** Yes

---

### 5. trade-reviewer

**Purpose:** Audits executed trades against STRAT methodology

**Model:** Sonnet

**Skills Loaded:**
- strat-methodology
- options-trading

**Use Cases:**
- Audit paper/live trades from VPS logs
- Verify entry timing (ON THE BREAK)
- Check stop/target calculations
- Validate TFC scoring at entry

**Read-Only:** Yes

---

### 6. vbt-researcher

**Purpose:** Researches VBT Pro API patterns via MCP

**Model:** Sonnet

**Skills Loaded:** None (uses VBT Pro MCP directly)

**Use Cases:**
- Find VBT Pro implementation patterns
- Resolve method names and signatures
- Discover code examples
- Validate API usage before implementation

**Read-Only:** Yes

---

## Commands (6 Total)

### 1. /session-start

**Purpose:** Automate session startup sequence

**Actions:**
1. Read `.session_startup_prompt.md`
2. Read `docs/HANDOFF.md`
3. Read `docs/CLAUDE.md`
4. Run `uv run pytest tests/ -q --tb=no` (health check)
5. Query OpenMemory for recent context
6. Output session brief:
   - Current session number
   - Test status (pass/fail)
   - Today's priorities
   - Reminders (strat-methodology mandatory, accuracy over speed)

**User Input Required:**
- Previous session summary (pasted after command completes)

**Does NOT:**
- Read previous session summary automatically
- Start any work without user direction

---

### 2. /session-end

**Purpose:** Automate session closure and quality checks

**Actions (in order):**
1. Run `/pre-commit` checks
2. Check CLAUDE.md - flag if bloated/outdated
3. Check HANDOFF.md line count - prompt archive if >1500 lines
4. Check README.md accuracy
5. Update `.session_startup_prompt.md` for next session
6. Store session facts in OpenMemory
7. Backup OpenMemory to dated file
8. Generate conventional commit message
9. Output session summary:
   - Tasks completed
   - Test results
   - Blockers encountered
   - Next priorities
   - Plan mode recommendation
   - Reference plans if multi-session

**User Confirmation Required:**
- Before committing (shows diff + message)
- Before archiving HANDOFF.md

**Does NOT:**
- Auto-commit without confirmation
- Delete files

---

### 3. /pre-commit

**Purpose:** Quality checks before commit

**Actions:**
1. Run full test suite
2. Code review on changed files
3. Run `silent-failure-hunter` on trade execution code
4. Check HANDOFF.md line count
5. Check README.md accuracy
6. Check CLAUDE.md not bloated

**Output:**
- Pass/fail for each check
- Specific issues found
- Recommendation: commit or fix first

**Called By:** `/session-end` (also available standalone)

---

### 4. /pull-logs

**Purpose:** Fetch VPS logs, parse, route to appropriate agent

**Actions:**
1. SSH to VPS (178.156.223.251)
2. Pull daemon logs: `sudo journalctl -u atlas-daemon --since <time>`
3. Parse for patterns:
   - TFC REEVAL events
   - PATTERN INVALIDATED events
   - ENTRY/EXIT events
   - ERROR/WARNING messages
4. Categorize issues
5. Route to agent:
   - Methodology issues → @strat-validator
   - Execution issues → @trade-reviewer
   - Data issues → @data-auditor
6. Output log summary with recommendations

**Parameters:**
- `--since <time>` - How far back (default: "1 hour ago")
- `--grep <pattern>` - Filter for specific pattern

**Example:**
```
/pull-logs --since "4 hours ago" --grep "TFC"
```

---

### 5. /test-focus

**Purpose:** Targeted test run with diagnosis

**Actions:**
1. Run tests for specified area
2. If failures, invoke @test-runner for diagnosis
3. Output results with analysis

**Parameters:**
- `<area>` - Test directory or pattern (e.g., `strat`, `signal_automation`, `regime`)

**Example:**
```
/test-focus strat
/test-focus test_bar_classifier
```

---

### 6. /tech-debt

**Purpose:** Check technical debt status, suggest priorities

**Actions:**
1. Read technical debt plan file
2. Parse current status:
   - Untested modules count
   - God classes count
   - Current phase
   - Recently completed items
3. Suggest next priorities (highest impact first)
4. Output progress summary

**Output:**
- Progress: X/Y items complete
- Current phase status
- Top 3 recommended next items
- Estimated complexity per item

**Does NOT:**
- Automatically fix debt
- Modify plan file

---

## Skills (8 Total)

### Activation Strategy

| Skill | Activation | Mechanism |
|-------|------------|-----------|
| strat-methodology | **Hook-enforced** | `strat_code_guardian.py` blocks until consulted |
| backtesting-validation | **Hook-enforced** | `skill_enforcement.py` (NEW) |
| options-trading | **Agent-loaded** | Loaded when @trade-reviewer invoked |
| position-sizing-risk | **Explicit** | User invokes `/skill position-sizing-risk` |
| momentum-strategies | **Explicit** | User invokes when needed |
| mean-reversion-strategies | **Explicit** | User invokes when needed |
| execution-quality | **Explicit** | User invokes when needed |
| portfolio-construction | **Explicit** | User invokes when needed |

### Skill Descriptions

| Skill | Source Books | Purpose |
|-------|--------------|---------|
| strat-methodology | Rob Smith STRAT | Bar classification, patterns, entry/exit timing |
| backtesting-validation | Davey, Chan | Walk-forward, Monte Carlo, bias detection |
| options-trading | Natenberg, Hull | Greeks, strike selection, premium calculation |
| position-sizing-risk | Carver | ATR-based sizing, portfolio heat, drawdown |
| momentum-strategies | Gray, Clenow, Antonacci | 52W high, dual momentum, trend following |
| mean-reversion-strategies | Connors, Chan | IBS, RSI(2), short-term reversals |
| execution-quality | Kissell, Chan | Slippage, market impact, execution timing |
| portfolio-construction | Grinold & Kahn | Allocation, diversification, rebalancing |

---

## Directory Structure

```
.claude/
├── agents/
│   ├── strat-validator.md
│   ├── test-runner.md
│   ├── backtest-reviewer.md
│   ├── data-auditor.md
│   ├── trade-reviewer.md
│   └── vbt-researcher.md
├── commands/
│   ├── session-start.md
│   ├── session-end.md
│   ├── pre-commit.md
│   ├── pull-logs.md
│   ├── test-focus.md
│   └── tech-debt.md
├── skills/
│   ├── strat-methodology/
│   │   └── SKILL.md
│   ├── backtesting-validation/
│   │   └── SKILL.md
│   ├── options-trading/
│   │   └── SKILL.md
│   ├── position-sizing-risk/
│   │   └── SKILL.md
│   ├── momentum-strategies/
│   │   └── SKILL.md
│   ├── mean-reversion-strategies/
│   │   └── SKILL.md
│   ├── execution-quality/
│   │   └── SKILL.md
│   └── portfolio-construction/
│       └── SKILL.md
├── hooks/
│   ├── strat_code_guardian.py      (existing)
│   ├── strat_prompt_validator.py   (existing)
│   ├── vbt_workflow_guardian.py    (existing)
│   └── skill_enforcement.py        (NEW)
└── settings.local.json
```

---

## Implementation Order

### Phase 1: Skill Migration (After Monday Review)
1. Review all 8 skills for Anthropic best practices
2. Copy from `~/.claude/skills/` to `.claude/skills/`
3. Update any skill references in existing hooks
4. Commit skills to git

### Phase 2: Commands
1. Create `/session-start` (simplest, immediate value)
2. Create `/session-end` (calls /pre-commit)
3. Create `/pre-commit` (standalone checks)
4. Create `/test-focus`
5. Create `/tech-debt`
6. Create `/pull-logs` (requires SSH testing)

### Phase 3: Agents
1. Create `strat-validator` (most critical)
2. Create `test-runner`
3. Create `trade-reviewer`
4. Create `backtest-reviewer`
5. Create `data-auditor`
6. Create `vbt-researcher`

### Phase 4: Hooks
1. Create `skill_enforcement.py` for backtesting-validation
2. Update existing hooks if needed
3. Test hook integration

### Phase 5: Integration Testing
1. Test `/session-start` → work → `/session-end` flow
2. Test `/pull-logs` → agent routing
3. Test hook enforcement with skill loading
4. Verify no conflicts between components

---

## What NOT to Build

Based on research and discussion, these were explicitly excluded:

| Idea | Reason Excluded |
|------|-----------------|
| Autonomous market agents | Claude Code can't run on schedule; existing daemon.py is superior |
| Trade decision algorithm agent | Would duplicate existing code logic; improve code instead |
| VPS/deployment agent | Issues too infrequent to justify dedicated agent |
| Sub-orchestrator pattern | Adds complexity without clear value; parallel agents + files is cleaner |
| Auto-discovery for skills | Proven unreliable; use explicit loading via agents/hooks |

---

## Dependencies

### Required for Implementation
- All 8 skills completed and reviewed
- Existing hooks functional
- VPS SSH access configured
- OpenMemory MCP operational

### MCP Servers Used
- `vectorbt-pro` - VBT Pro documentation/code execution
- `openmemory` - Session memory storage
- `playwright` - Web automation (if needed)
- `ThetaData` - Options data (when terminal running)

---

## Success Metrics

After implementation, these should be true:

1. **Session startup < 30 seconds** - `/session-start` automates manual reading
2. **Session end < 2 minutes** - `/session-end` handles all closure tasks
3. **Zero methodology violations** - Hooks enforce skill consultation
4. **Test failures diagnosed** - @test-runner provides actionable analysis
5. **VPS issues caught early** - `/pull-logs` surfaces problems quickly
6. **Technical debt visible** - `/tech-debt` shows progress and priorities

---

## References

- Claude Code Docs: https://docs.anthropic.com/en/docs/claude-code
- Existing hooks: `.claude/hooks/`
- Technical debt plan: `C:\Users\sheeh\.claude\plans\sharded-foraging-puppy.md`
- CLAUDE.md rules: `docs/CLAUDE.md`

---

**Document Version:** 1.0
**Approved By:** User (January 17, 2026)
**Next Action:** Review skills Monday, then implement Phase 1

# ATLAS Development Rules

> **Token-optimized rules file. Detailed examples in `docs/CLAUDE_REFERENCE.md`.**

## 1. Session Start (MANDATORY)

```
1. Read docs/HANDOFF.md (current state)
2. Read .session_startup_prompt.md (current mission)
3. Verify tests: uv run pytest tests/ -q
```

## 2. MANDATORY Skills (ZERO TOLERANCE)

**MUST invoke these skills before writing related code. No exceptions.**

| Skill | Location | Invoke When |
|-------|----------|-------------|
| `strat-methodology` | `C:/Users/sheeh/.claude/skills/strat-methodology/` | ANY STRAT pattern detection, bar classification, options entry/exit, timeframe analysis |
| `thetadata-api` | `C:/Users/sheeh/.claude/skills/thetadata-api/` | ANY ThetaData API call, debugging 472/500 errors, Greeks endpoints, options data fetching |
| `backtesting-validation` | `C:/Users/sheeh/.claude/skills/backtesting-validation/` | ANY VPS deployment, new strategy implementation, backtest evaluation, signal detection changes |

**Why mandatory:** Previous sessions repeatedly made the same mistakes (wrong endpoints, wrong formats, wrong error handling, timezone bugs, live bar detection errors) that these skills prevent. Skipping skill invocation = bugs that waste hours to debug.

**How to invoke:** Use the Skill tool with skill name (e.g., `Skill: strat-methodology`)

## 3. Communication Standards

- NO emojis or unicode characters (Windows compatibility requirement)
- NO AI attribution in commits or docs
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- Professional tone (third person, declarative)
- Plain ASCII only

## 4. Brutal Honesty Policy

- If you don't know, say "I don't know"
- If guessing, say "I'm guessing"
- If wrong approach, say "This is wrong because..."
- If simpler way exists, suggest it
- If task adds complexity without value, say so

## 5. Data Sources (ZERO TOLERANCE)

| Source | Use Case |
|--------|----------|
| Alpaca | Primary - all equities via `vbt.AlpacaData.pull()` |
| Tiingo | Secondary - via `integrations/tiingo_data_fetcher.py` |
| yfinance | VIX ONLY (`^VIX` not on Alpaca) |

**ALWAYS:** `tz='America/New_York'` on all data fetches

**NEVER:** Synthetic data, mock OHLCV generators, yfinance for equities

## 6. VectorBT Pro Workflow (5 Steps)

Every VBT implementation MUST follow:

```
SEARCH  -> mcp__vectorbt-pro__search(query, asset_names=["examples","api"])
VERIFY  -> mcp__vectorbt-pro__resolve_refnames(["vbt.ClassName"])
FIND    -> mcp__vectorbt-pro__find(refnames, asset_names=["examples"])
TEST    -> mcp__vectorbt-pro__run_code(minimal_example)
IMPLEMENT -> Only after steps 1-4 pass
```

**Skip any step = implementation will fail.**

## 7. Market Data Rules

- Filter weekends: `data.index.dayofweek < 5`
- Filter NYSE holidays: use `pandas_market_calendars`
- Verify no Saturday/Sunday bars before any analysis
- All backtests use `fees=0.001, slippage=0.001` minimum

## 8. Quality Gates

Before claiming ANY functionality works:
- Test it (run the actual code)
- Verify output (check results are correct)
- Show evidence (paste actual output)

## 9. Project Structure

```
validation/     - ATLAS compliance validators (walk-forward, Monte Carlo, bias)
strat/          - STRAT pattern detection and options module
tests/          - Test suite (913 tests)
docs/           - HANDOFF.md, CLAUDE.md, CLAUDE_REFERENCE.md
```

## 10. Key Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run validation tests only
uv run pytest tests/test_validation/ -v

# Run STRAT tests only
uv run pytest tests/test_strat/ -v

# Verify VBT Pro
uv run python -c "import vectorbtpro as vbt; print(vbt.__version__)"
```

## 11. Multi-Session Plans

For implementations spanning 3+ sessions:
- Create plan file: `C:\Users\sheeh\.claude\plans\<name>.md`
- Update `.session_startup_prompt.md` each session
- Store facts in OpenMemory at session end
- Reference: See `docs/CLAUDE_REFERENCE.md` Section 7

## 12. Session End Protocol

```
1. Commit with conventional commit message
2. Update docs/HANDOFF.md (new session entry at TOP)
3. Store session facts in OpenMemory
4. Backup OpenMemory: cp "C:/Dev/openmemory/data/atlas_memory.sqlite" "C:/Dev/openmemory/backups/atlas_memory_YYYY-MM-DD.sqlite"
5. Update .session_startup_prompt.md for next session
```

## 13. STRAT Bar Classification (MANDATORY)

Every directional bar MUST be classified as 2U (bullish) or 2D (bearish). Never use just "2".

| Correct | Incorrect |
|---------|-----------|
| 2D-2U | 2-2 |
| 3-2U, 3-2D | 3-2 |
| 3-2U-2U, 3-2D-2D | 3-2-2 |
| 2U-1-2U, 2D-1-2D | 2-1-2 |
| 3-1-2U, 3-1-2D | 3-1-2 (OK - only exit bar needs direction) |

Bar types: 1=inside, 2U=up, 2D=down, 3=outside

## 14. STRAT Entry Timing (ZERO TOLERANCE)

**Entry happens ON THE BREAK, not at bar close. This is the most common implementation error.**

| Concept | Rule |
|---------|------|
| Entry is LIVE | When price breaks trigger level, enter IMMEDIATELY |
| Forming bar classification | Only "1" if it stays INSIDE prior range - NOT automatic |
| Pre/post market gaps | Bar can OPEN as 2U/2D due to overnight action |
| Pattern completion | Happens the MOMENT price breaks, not at bar close |

**Scanner Implementation:**
- Last bar in data = yesterday's CLOSED bar (NOT today's forming bar)
- Today's forming bar is NOT in the data - it's LIVE
- Create bidirectional setups based on last CLOSED bar
- Entry monitor watches LIVE price for breaks

**Example:** Yesterday = 3 (H=$280, L=$273), Today opens at $271 (below $273)
- Today is ALREADY 2D at open due to gap down
- Pattern 3-2D is COMPLETE at market open
- Entry is IMMEDIATE, not waiting for today to close

**DO NOT:**
- Exclude last bar from bidirectional setup detection (it's the setup bar)
- Wait for forming bar to close before entering
- Assume forming bar is always "1"

See strat-methodology skill `EXECUTION.md` "CRITICAL: Entry Timing" section.

## 15. DO NOT

- Skip HANDOFF.md at session start
- Skip mandatory skills (strat-methodology, thetadata-api) when writing related code
- Skip VBT 5-step workflow
- Use yfinance for non-VIX data
- Generate synthetic/mock OHLCV data
- Create SESSION_XX_*.md files (use HANDOFF.md only)
- Use emojis or special characters anywhere
- Claim code works without testing
- Archive files (DELETE redundant files)
- Create breakout strategies without 2x volume confirmation
- Use unclassified "2" bars in STRAT patterns (must be 2U or 2D)
- Wait for bar close to enter STRAT trades (entry is ON THE BREAK)
- Exclude last closed bar from bidirectional setup detection

## 16. File Tiers

| Tier | Files | When to Read |
|------|-------|--------------|
| 1 | HANDOFF.md, CLAUDE.md | Every session |
| 2 | CLAUDE_REFERENCE.md | When need detailed examples |
| 3 | VectorBT Pro Official Documentation/ | VBT implementation |

## 17. Account Constraints

Schwab Level 1 Options (cash account):
- CAN: Long stock, long calls/puts, cash-secured puts
- CANNOT: Short stock, naked options, spreads

## 18. Plugin Commands

| Plugin | Command/Trigger | When to Use |
|--------|-----------------|-------------|
| `feature-dev` | `/feature-dev <description>` | New features spanning multiple files (7-phase workflow) |
| `code-review` | `/code-review` | Before committing or merging PRs |
| `pr-review-toolkit` | "Review for silent failures" | Trade execution code, error handling |
| `pr-review-toolkit` | "Check test coverage" | After adding new functionality |
| `pyright-lsp` | (automatic) | Python type checking in background |

**Proactive usage:**
- Run `/code-review` before every commit
- Use `/feature-dev` for new strategy implementations (e.g., Quality-Momentum)
- Request silent-failure review on ANY trade execution or order handling code

---

**Version:** 3.4 (December 29, 2025)
**Status:** PRODUCTION - Token-optimized
**Details:** See `docs/CLAUDE_REFERENCE.md` for verbose examples

# ATLAS Development Rules

> **Token-optimized rules file. Detailed examples in `docs/CLAUDE_REFERENCE.md`.**

## 1. Session Start (MANDATORY)

```
1. Read docs/HANDOFF.md (current state)
2. Read .session_startup_prompt.md (current mission)
3. Verify tests: uv run pytest tests/ -q
```

## 2. Communication Standards

- NO emojis or unicode characters (Windows compatibility requirement)
- NO AI attribution in commits or docs
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- Professional tone (third person, declarative)
- Plain ASCII only

## 3. Brutal Honesty Policy

- If you don't know, say "I don't know"
- If guessing, say "I'm guessing"
- If wrong approach, say "This is wrong because..."
- If simpler way exists, suggest it
- If task adds complexity without value, say so

## 4. Data Sources (ZERO TOLERANCE)

| Source | Use Case |
|--------|----------|
| Alpaca | Primary - all equities via `vbt.AlpacaData.pull()` |
| Tiingo | Secondary - via `integrations/tiingo_data_fetcher.py` |
| yfinance | VIX ONLY (`^VIX` not on Alpaca) |

**ALWAYS:** `tz='America/New_York'` on all data fetches

**NEVER:** Synthetic data, mock OHLCV generators, yfinance for equities

## 5. VectorBT Pro Workflow (5 Steps)

Every VBT implementation MUST follow:

```
SEARCH  -> mcp__vectorbt-pro__search(query, asset_names=["examples","api"])
VERIFY  -> mcp__vectorbt-pro__resolve_refnames(["vbt.ClassName"])
FIND    -> mcp__vectorbt-pro__find(refnames, asset_names=["examples"])
TEST    -> mcp__vectorbt-pro__run_code(minimal_example)
IMPLEMENT -> Only after steps 1-4 pass
```

**Skip any step = implementation will fail.**

## 6. Market Data Rules

- Filter weekends: `data.index.dayofweek < 5`
- Filter NYSE holidays: use `pandas_market_calendars`
- Verify no Saturday/Sunday bars before any analysis
- All backtests use `fees=0.001, slippage=0.001` minimum

## 7. Quality Gates

Before claiming ANY functionality works:
- Test it (run the actual code)
- Verify output (check results are correct)
- Show evidence (paste actual output)

## 8. Project Structure

```
validation/     - ATLAS compliance validators (walk-forward, Monte Carlo, bias)
strat/          - STRAT pattern detection and options module
tests/          - Test suite (312+ tests)
docs/           - HANDOFF.md, CLAUDE.md, CLAUDE_REFERENCE.md
```

## 9. Key Commands

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

## 10. Multi-Session Plans

For implementations spanning 3+ sessions:
- Create plan file: `C:\Users\sheeh\.claude\plans\<name>.md`
- Update `.session_startup_prompt.md` each session
- Store facts in OpenMemory at session end
- Reference: See `docs/CLAUDE_REFERENCE.md` Section 7

## 11. Session End Protocol

```
1. Commit with conventional commit message
2. Update docs/HANDOFF.md (new session entry at TOP)
3. Store session facts in OpenMemory
4. Update .session_startup_prompt.md for next session
```

## 12. DO NOT

- Skip HANDOFF.md at session start
- Skip VBT 5-step workflow
- Use yfinance for non-VIX data
- Generate synthetic/mock OHLCV data
- Create SESSION_XX_*.md files (use HANDOFF.md only)
- Use emojis or special characters anywhere
- Claim code works without testing
- Archive files (DELETE redundant files)
- Create breakout strategies without 2x volume confirmation

## 13. File Tiers

| Tier | Files | When to Read |
|------|-------|--------------|
| 1 | HANDOFF.md, CLAUDE.md | Every session |
| 2 | CLAUDE_REFERENCE.md | When need detailed examples |
| 3 | VectorBT Pro Official Documentation/ | VBT implementation |

## 14. Account Constraints

Schwab Level 1 Options (cash account):
- CAN: Long stock, long calls/puts, cash-secured puts
- CANNOT: Short stock, naked options, spreads

---

**Version:** 3.0 (November 27, 2025)
**Status:** PRODUCTION - Token-optimized
**Details:** See `docs/CLAUDE_REFERENCE.md` for verbose examples

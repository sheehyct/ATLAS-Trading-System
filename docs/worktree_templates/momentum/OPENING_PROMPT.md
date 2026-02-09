# MOMENTUM Track Opening Prompt

Copy everything below this line to start your session:

---

Please read the following documents and utilize open memory queries for context of this session's development. We are to maintain our standard professional development workflow. For any STRAT related concepts, development, changes, or additions the use of the strat-methodology skill is mandatory through the entire session. Not just in the initial plan stage. The use of the plugins in CLAUDE.md are also mandatory.

## Required Reading (in order)

1. C:\Strat_Trading_Bot\atlas-strat-momentum-20251229\.session_startup_prompt.md (TRACK-SPECIFIC SCOPE)
2. C:\Users\sheeh\.claude\plans\quiet-floating-clarke.md (FULL PARALLEL PLAN)
3. C:\Strat_Trading_Bot\atlas-strat-momentum-20251229\docs\HANDOFF.md
4. C:\Strat_Trading_Bot\atlas-strat-momentum-20251229\docs\CLAUDE.md **STRICT COMPLIANCE**
5. C:\Strat_Trading_Bot\atlas-strat-momentum-20251229\docs\SYSTEM_ARCHITECTURE\1_ATLAS_OVERVIEW_AND_PROPOSED_STRATEGIES.md
6. C:\Strat_Trading_Bot\atlas-strat-momentum-20251229\strategies\base_strategy.py (READ ONLY - the contract)

## Parallel Development Context

This session is part of a **4-track parallel development effort** using git worktrees.

**THIS SESSION IS: MOMENTUM Track**
**BRANCH: feature/strategies-momentum**
**SCOPE: Quality-Momentum + Semi-Vol Momentum strategies**

## Exclusive Files (ONLY this track modifies)

- strategies/quality_momentum.py
- strategies/semi_vol_momentum.py
- tests/test_strategies/test_quality_momentum.py
- tests/test_strategies/test_semi_vol_momentum.py
- scripts/backtest_quality_momentum.py
- scripts/backtest_semi_vol_momentum.py

## Critical Rules

1. **LOCKED FILES** - DO NOT MODIFY:
   - strategies/base_strategy.py
   - utils/position_sizing.py
   - utils/portfolio_heat.py
   - regime/academic_jump_model.py
2. **VBT 5-STEP WORKFLOW** - MANDATORY for all implementations
3. **HANDOFF** - Create docs/session_handoffs/TRACK_MOMENTUM_SESSION_{N}.md

## Validation Targets

| Strategy | Sharpe | CAGR | Max DD |
|----------|--------|------|--------|
| Quality-Momentum | 1.3-1.7 | 15-22% | < -25% |
| Semi-Vol | 1.4-1.8 | 15-20% | < -25% |

We are to prioritize accuracy over speed.

------

Below is the session end summary from our previous session:

{PASTE PREVIOUS SESSION SUMMARY HERE}

------

## Session End Protocol

- Create/update docs/session_handoffs/TRACK_MOMENTUM_SESSION_{N}.md
- Populate Open Memory with implementation decisions
- Provide summary in chat:
  - Tasks completed
  - Test results (pytest output)
  - Blockers encountered
  - Next priorities
- Recommendation for plan mode (on/off)
- Ensure to include reference plans and context management protocols
- Note any cross-track coordination needs

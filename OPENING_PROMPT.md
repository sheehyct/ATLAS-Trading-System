# AUDIT Track Opening Prompt

Copy everything below this line to start your session:

---

Please read the following documents and utilize open memory queries for context of this session's development. We are to maintain our standard professional development workflow. For any STRAT related concepts, development, changes, or additions the use of the strat-methodology skill is mandatory through the entire session. Not just in the initial plan stage. The use of the plugins in CLAUDE.md are also mandatory.

## Required Reading (in order)

1. C:\Strat_Trading_Bot\atlas-audit-20251229\.session_startup_prompt.md (TRACK-SPECIFIC SCOPE)
2. C:\Users\sheeh\.claude\plans\quiet-floating-clarke.md (FULL PARALLEL PLAN)
3. C:\Strat_Trading_Bot\atlas-audit-20251229\docs\HANDOFF.md
4. C:\Strat_Trading_Bot\atlas-audit-20251229\docs\CLAUDE.md **STRICT COMPLIANCE**
5. C:\Strat_Trading_Bot\atlas-audit-20251229\docs\SYSTEM_ARCHITECTURE\*.md (ALL 9 FILES)

## Parallel Development Context

This session is part of a **4-track parallel development effort** using git worktrees.

**THIS SESSION IS: AUDIT Track**
**BRANCH: feature/codebase-audit**
**SCOPE: Read-only analysis, documentation creation only**

## Deliverables Location

```
docs/audit/
  - MASTER_INDEX.md
  - A_ARCHITECTURE.md
  - B_CODE_QUALITY.md
  - C_TEST_COVERAGE.md
  - D_SECURITY.md
  - E_PERFORMANCE.md
  - F_DOCUMENTATION.md
  - G_TECHNICAL_DEBT.md
  - H_DASHBOARD.md
  - I_ERROR_HANDLING.md
  - J_INTEGRATIONS.md
```

## Critical Rules

1. **READ-ONLY** - Do not modify any code files, only create audit reports
2. **LOCKED FILES** - DO NOT MODIFY:
   - strategies/base_strategy.py
   - utils/position_sizing.py
   - utils/portfolio_heat.py
   - regime/academic_jump_model.py
3. **HANDOFF** - Create docs/session_handoffs/TRACK_AUDIT_SESSION_{N}.md

We are to prioritize accuracy over speed.

------

Below is the session end summary from our previous session:

{PASTE PREVIOUS SESSION SUMMARY HERE}

------

## Session End Protocol

- Create/update docs/session_handoffs/TRACK_AUDIT_SESSION_{N}.md
- Populate Open Memory with key findings
- Provide summary in chat:
  - Domains audited
  - Finding counts by severity (CRITICAL/HIGH/MEDIUM/LOW)
  - Next priorities
- Recommendation for plan mode (on/off)
- Ensure to include reference plans and context management protocols
- Note any cross-track coordination needs

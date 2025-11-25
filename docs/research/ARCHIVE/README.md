# Research Archive - Consolidated Documentation

**Archived:** October 18, 2025
**Reason:** Consolidation to eliminate contradictions and redundancy

---

## What Happened

These 5 research documents were consolidated into **CONSOLIDATED_RESEARCH.md** to resolve 7 major contradictions and eliminate ~60% redundancy.

**Files Archived:**
1. Advanced_Algorithmic_Trading_Systems.md (136 lines)
2. Algorithmic Trading System.md (69 lines)
3. Algorithmic trading systems with asymmetric risk-reward profiles.md (145 lines)
4. CRITICAL_POSITION_SIZING_CLARIFICATION.md (687 lines)
5. STRATEGY_ANALYSIS_AND_DESIGN_SPEC.md (2139 lines)

**Total:** 3,176 lines → 813 lines in CONSOLIDATED_RESEARCH.md

---

## Major Contradictions Resolved

### 1. Volume Confirmation Threshold
**Contradiction:**
- CLAUDE.md (outdated): 1.5x average volume
- STRATEGY_2_IMPLEMENTATION_ADDENDUM.md: 2.0x MANDATORY

**Resolution:** 2.0x MANDATORY (research-backed per Oct 18 article findings)

### 2. Position Sizing Risk Percentage
**Contradiction:**
- Documentation: risk_pct = 0.02 (2%)
- GATE1_RESULTS.md: Produces 40.6% mean (exceeds 10-30% target)

**Resolution:** risk_pct = 0.01 (1%) recommended for ORB strategy

### 3. Sharpe Ratio Targets
**Contradiction:**
- VALIDATION_PROTOCOL.md: >0.8 acceptable
- STRATEGY_ANALYSIS_AND_DESIGN_SPEC.md: 2.396 target
- STRATEGY_2_IMPLEMENTATION_ADDENDUM.md: 2.0 minimum

**Resolution:** Phased targets (Backtest >2.0, Paper >0.8, Live >0.5)

### 4. Risk-Reward Minimum
**Contradiction:**
- Original guidance: 2:1 R:R acceptable
- Expectancy math: Only 0.15% net after 0.35% costs

**Resolution:** 3:1 R:R MINIMUM (net expectancy 0.65%)

### 5. TFC Approach
**Contradiction:**
- STRATEGY_ANALYSIS_AND_DESIGN_SPEC.md: Still discusses TFC confidence scoring
- Medium_Articles_Research_Findings.md: ABANDON TFC scoring

**Resolution:** TFC scoring ABANDONED (6+ parameters = overfitting), classification code KEPT

### 6. Position Sizing Methods
**Contradiction:**
- Confusion about when to use ATR vs Garman-Klass vs Yang-Zhang

**Resolution:**
- ATR: For strategies with ATR stops (ORB - current)
- Garman-Klass: For momentum portfolios (future)
- Yang-Zhang: For regime detection features (future)

### 7. Market Hours Filtering
**Contradiction:**
- Implicit requirement vs explicit implementation guide

**Resolution:** MANDATORY CRITICAL rule with code examples in CONSOLIDATED_RESEARCH.md

---

## Why These Files Were Consolidated

### Advanced_Algorithmic_Trading_Systems.md
- General expectancy framework
- Sharpe ratio analysis
- DUPLICATE of content in other files
- KEY CONTENT: Expectancy formula (preserved in consolidated doc)

### Algorithmic Trading System.md
- High-level algo trading overview
- REDUNDANT with other more detailed docs
- No unique content that wasn't covered elsewhere

### Algorithmic trading systems with asymmetric risk-reward profiles.md
- Asymmetric strategy principles
- Sortino ratio guidance
- MERGED into Part 1 of CONSOLIDATED_RESEARCH.md

### CRITICAL_POSITION_SIZING_CLARIFICATION.md
- 687 lines explaining ATR vs Garman-Klass vs Yang-Zhang
- Created confusion by over-explaining options
- SUPERSEDED by actual implementation (utils/position_sizing.py)
- KEY CONTENT: When to use each method (preserved)

### STRATEGY_ANALYSIS_AND_DESIGN_SPEC.md
- 2139 lines comprehensive strategy guide
- Most content REDUNDANT with other docs
- Still referenced TFC scoring (contradicted current decision)
- KEY CONTENT: ORB analysis (moved to STRATEGY_2_IMPLEMENTATION_ADDENDUM.md)

---

## What Remains in Research Folder (NOT Consolidated)

**Medium_Articles_Research_Findings.md**
- Foundational cross-validated research (1550 lines)
- THE primary reference document
- READ THIS FIRST for philosophical foundation
- NOT consolidated - this is the source of truth

**VALIDATION_PROTOCOL.md**
- Complete 5-phase testing methodology (700 lines)
- Standalone protocol for all strategies
- NOT consolidated - timeless testing standards

**STRATEGY_2_IMPLEMENTATION_ADDENDUM.md**
- ORB-specific implementation corrections (1260 lines)
- 6 critical corrections for Strategy 2
- NOT consolidated - strategy-specific guidance

**algo_trading_diagnostic_framework.md**
- Failure mode taxonomy
- Debugging checklists
- NOT consolidated - diagnostic reference

---

## How to Use Archived Files

**For Historical Research:**
- These files are frozen (never edited)
- Preserve negative results and thought process
- Show evolution of understanding

**For Current Development:**
- DO NOT USE - these contain contradictions
- USE: CONSOLIDATED_RESEARCH.md instead
- CROSS-REFERENCE: Medium_Articles_Research_Findings.md for deep theory

**If You Need Specific Content:**
1. Check CONSOLIDATED_RESEARCH.md first
2. Check Medium_Articles_Research_Findings.md for theory
3. Only then check archived files for historical context

---

## Files Moved to Active Development

**WEEK1_EXECUTIVE_BRIEF.md**
- Week 1 implementation brief
- NOT timeless research - was session-specific
- MOVED TO: docs/active/risk-management-foundation/

**PROJECT_AUDIT_2025-10-16.md**
- Project state snapshot
- NOT timeless research - was status report
- MOVED TO: docs/active/risk-management-foundation/

---

## Consolidation Impact

**Before:**
- 5 files with contradictions
- 3,176 lines total
- 7 major conflicts
- 60%+ redundancy
- Confusion about which guidance to follow

**After:**
- 1 authoritative file (CONSOLIDATED_RESEARCH.md)
- 813 lines (74% reduction)
- 0 contradictions (all resolved with rationale)
- Cross-references to remaining docs
- Clear guidance on what to use when

---

## Next Steps

If new research findings emerge:
1. Add to CONSOLIDATED_RESEARCH.md (don't edit archived files)
2. Update Medium_Articles_Research_Findings.md if foundational
3. Create new research file if topic warrants separate document
4. Document resolution if new finding contradicts consolidated doc

---

**Archive Date:** October 18, 2025
**Consolidated By:** Claude Code (Sonnet 4.5)
**Next Review:** After major research findings or contradictions discovered

# Claude Code Handoff - STRAT Options Strategy Implementation

**Date:** November 6, 2025  
**Current Phase:** Phase 1 - Bar Classification  
**Next Action:** Implement bar classifier in VectorBT Pro

---

## Context Summary

We're building a systematic options trading strategy using:
- **Methodology:** Rob Smith's STRAT (bar pattern recognition)
- **Framework:** VectorBT Pro for backtesting
- **Capital:** $3,000 minimum viable
- **Instruments:** Long calls/puts only (Options Level 2)
- **Edge:** Timeframe continuity filtering (Monthly/Weekly/Daily/4H/1H alignment)

---

## Two Key Documents

### 1. Strategy Handoff (`STRAT_OPTIONS_STRATEGY_HANDOFF.md`)
**Purpose:** Complete strategy context and background  
**Contains:**
- Full STRAT methodology explanation
- Why we're using options ($3k capital efficiency)
- Implementation roadmap (6 phases)
- Risk management framework
- Data requirements
- Critical questions for backtesting

**When to read:** 
- Before starting implementation (understand full context)
- When making strategic decisions
- When questions arise about "why are we doing this?"

### 2. Implementation Guide (`CLAUDE_CODE_IMPLEMENTATION_GUIDE.md`)
**Purpose:** Step-by-step technical implementation  
**Contains:**
- Complete bar classification code
- VectorBT Pro indicator creation
- Validation tests
- Multi-timeframe extension
- Troubleshooting guide
- Success criteria

**When to use:**
- Right now (immediate implementation)
- As reference during coding
- For validation procedures
- When debugging issues

---

## Your Mission (Phase 1)

**Goal:** Build and validate STRAT bar classification indicator

### Tasks:
1. ✅ Read `CLAUDE_CODE_IMPLEMENTATION_GUIDE.md` completely
2. ⬜ Create `strat_classifier.py` file
3. ⬜ Implement bar classification function
4. ⬜ Create VectorBT Pro indicator
5. ⬜ Register indicator with `vbt.IF.register_custom_indicator()`
6. ⬜ Test on SPY daily data
7. ⬜ Validate against manual chart analysis
8. ⬜ Export results to CSV
9. ⬜ Run all edge case tests
10. ⬜ Document results

### Success Criteria:
- Bar classification runs without errors
- 95%+ accuracy vs TradingView manual verification
- All edge cases pass
- Can identify known patterns manually

---

## Quick Reference: Bar Classification

```python
# Core logic (from implementation guide)
def classify_bar(current_high, current_low, prev_high, prev_low):
    broke_high = current_high > prev_high
    broke_low = current_low < prev_low
    
    if broke_high and broke_low:
        return 3  # Outside
    elif broke_high:
        return 2  # 2U
    elif broke_low:
        return -2  # 2D
    else:
        return 1  # Inside
```

**Remember:**
- First bar (index 0) = UNDEFINED (no previous bar)
- Classification is RELATIVE (depends on previous bar)
- Use strict > and < (not >= or <=)

---

## Key Concepts (Quick Reminder)

### STRAT Bar Types:
- **1 (Inside):** Within previous bar range
- **2U (Directional Up):** Broke previous high only
- **2D (Directional Down):** Broke previous low only
- **3 (Outside):** Broke both high and low

### Timeframe Continuity:
- **Full continuity** = All timeframes (Monthly/Weekly/Daily/4H/1H) showing same direction
- **Bullish continuity** = All showing 2U
- **Bearish continuity** = All showing 2D
- **Inside bars on higher timeframes** = Partial continuity = Lower probability

### Why This Matters:
- Timeframe continuity = institutional participation
- Filters out low-probability setups
- Allows aggressive options positioning with defined risk

---

## Development Environment

**Tools:**
- VectorBT Pro 2025.10.15
- Python 3.13.7
- UV package manager
- Alpaca Algo Trader Plus (data source)

**Key Commands:**
```bash
# Install dependencies
pip install vectorbtpro --break-system-packages

# Run implementation
python strat_classifier.py

# Run tests (after creating them)
pytest test_strat_classifier.py
```

---

## What Comes After Phase 1

Once bar classification is validated:

**Phase 2:** Timeframe Continuity Checker
- Check alignment across Monthly/Weekly/Daily
- Return boolean: "Is there full continuity?"

**Phase 3:** Pattern Detection
- Identify 3-1-2, 2-1-2, 2-2, etc.
- Calculate magnitude targets
- Generate entry/exit signals

**Phase 4:** Equity Backtest
- Prove patterns reach magnitude targets
- Calculate hit rates
- Determine average days to magnitude

**Phase 5:** Options Simulation
- Optimize DTE selection
- Optimize strike distance
- Calculate expectancy

**Phase 6:** Production System
- Real-time scanning
- Options chain analysis
- Alert generation

**Don't jump ahead** - each phase builds on the previous. Phase 1 must be bulletproof.

---

## Critical Reminders

### DO:
✅ Follow the implementation guide exactly  
✅ Validate against manual chart analysis  
✅ Test edge cases  
✅ Export results for visual verification  
✅ Document any deviations or issues

### DON'T:
❌ Skip validation steps  
❌ Modify classification logic without testing  
❌ Use bar color for classification  
❌ Look ahead in data  
❌ Proceed to Phase 2 if Phase 1 has issues

---

## Communication Protocol

**If you encounter issues:**
1. Check troubleshooting section in implementation guide
2. Verify data alignment (first bar handling)
3. Export results to CSV for manual inspection
4. Document the exact error and context
5. Report back with specific details

**If you need clarification:**
1. Reference strategy handoff document
2. Check if it's a strategic question (why) vs technical (how)
3. Ask specific questions with context
4. Include code snippets showing the issue

**When complete:**
1. Run all validation tests
2. Export results (CSV + summary stats)
3. Document any discoveries or issues
4. Report success metrics
5. Confirm ready for Phase 2

---

## File Locations

```
/mnt/user-data/outputs/
├── STRAT_OPTIONS_STRATEGY_HANDOFF.md     # Strategy context
├── CLAUDE_CODE_IMPLEMENTATION_GUIDE.md   # Technical guide
└── THIS_FILE.md                          # You are here

Create these in your workspace:
/workspace/ (or wherever you work)
├── strat_classifier.py                   # Main code
├── test_strat_classifier.py             # Tests
└── validation_results/
    ├── spy_classifications.csv          # Export
    └── test_results.txt                 # Results
```

---

## Expected Timeline

**Realistic estimate for Phase 1:**
- Code implementation: 1-2 hours
- Testing & validation: 2-3 hours
- Edge case handling: 1 hour
- Documentation: 30 minutes

**Total Phase 1: 4-6 hours of focused work**

Don't rush - accuracy is critical. Everything else builds on this.

---

## Next Steps

1. **Right now:** Open `CLAUDE_CODE_IMPLEMENTATION_GUIDE.md`
2. **Follow Step 1:** Understand classification logic
3. **Follow Step 2:** Implement in VectorBT Pro
4. **Follow Step 3:** Run validation tests
5. **Report back:** With results and any issues

---

## Questions Before Starting?

Common questions answered in the guides:
- "How does bar classification work?" → Implementation guide, Step 1
- "Why are we using options?" → Strategy handoff, "Options Integration Strategy"
- "What's timeframe continuity?" → Strategy handoff, "STRAT Methodology"
- "What data source?" → Both documents mention Alpaca
- "What if tests fail?" → Implementation guide, "Troubleshooting"

**If your question isn't answered in either document, ask!**

---

## Success Looks Like

At the end of Phase 1, you should have:
- ✅ Working bar classifier that processes any OHLC data
- ✅ Test results showing 95%+ accuracy
- ✅ CSV export matching TradingView
- ✅ All edge cases handled
- ✅ Clean, documented code
- ✅ Confidence to proceed to Phase 2

---

**You got this. Start with the implementation guide and work methodically. Quality over speed.**

Good luck! 🚀

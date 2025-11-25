# Claude Code Migration - Initial Setup Prompt

## PROJECT CONTEXT: STRAT Trading Bot Recovery & Migration

I'm working on a professional algorithmic trading bot that implements Rob Smith's STRAT methodology. The project was previously working with 70%+ signal conversion but got over-engineered and broken. 

**TODAY'S ACHIEVEMENT**: Successfully salvaged all working components from the original project.

**PROJECT LOCATION**: `C:\Strat_Trading_Bot\vectorbt-workspace\`

## IMMEDIATE TASKS:
1. Test the salvaged unified_strat_engine.py with real market data
2. Validate 70%+ signal conversion is restored  
3. Compare unified engine performance vs complex broken system
4. Prepare specialized subagents for different aspects of the trading system

## KEY TECHNICAL CONTEXT:
- **STRAT methodology** focuses on 2-1-2 patterns (Directional → Inside → Breakout)
- **Session 7 breakthrough**: Issue was fractional shares (0.36 × $70 = impossible) + $1 commission eating $25 positions
- **Solution**: Full shares with $200-500 position sizes + proper VectorBT parameters
- **VectorBT Pro 2025.7.27** may have installation issues (TA-Lib dependencies)

## CURRENT DEVELOPMENT ENVIRONMENT:
- **OS**: Windows 11
- **IDE**: VS Code
- **Package Manager**: UV (migrated from conda/virtual env)
- **Backtesting Engine**: VectorBT Pro 2025.7.27
- **Data Source**: Alpaca Markets (3 paper accounts, $5,000 balance each)

## PROJECT STATUS:
✅ **SALVAGE OPERATION COMPLETE**
- All working components recovered to `original_*.py` files
- Session 7 breakthrough insights implemented in `unified_strat_engine.py`
- Clean test suite prepared: `test_unified_vs_complex.py`, `working_breakthrough_test.py`
- Environment configured with UV and Alpaca API credentials

✅ **READY FOR TESTING**
- Unified STRAT engine ready for validation
- Real market data available via Alpaca API
- Performance targets: 70%+ signal conversion, 30%+ win rates

## SPECIALIZED SUBAGENTS DESIGNED:
1. **strat-pattern-expert**: STRAT methodology specialist
2. **data-connection-specialist**: Alpaca API and market data handling  
3. **vectorbt-optimization-expert**: VectorBT Pro parameter tuning
4. **risk-management-specialist**: Position sizing and risk controls

Can you help me execute the unified engine test and validate the salvage operation was successful?

---

## FIRST ACTIONS NEEDED:
```bash
# 1. Run final validation
uv run python final_validation_before_claude_code.py

# 2. Test unified engine
uv run python unified_strat_engine.py

# 3. Compare unified vs complex
uv run python test_unified_vs_complex.py

# 4. Validate breakthrough 
uv run python working_breakthrough_test.py
```

## SUCCESS CRITERIA:
- Signal conversion: 70%+ (from Session 7 breakthrough)
- Win rate: 30%+ (with proper STRAT exits)
- Pattern detection: Multiple 2-1-2, 3-1-2 patterns found
- Position sizing: $200-500 per trade (no fractional shares)
- VectorBT Pro: Functional with correct 2025.7.27 parameters

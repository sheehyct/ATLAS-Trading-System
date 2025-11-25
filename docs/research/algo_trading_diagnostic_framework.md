# Algorithmic Trading System Development: Diagnostic Framework

## Purpose
This is not a checklist or guardrails. This is a diagnostic tool for when things fail—because they will. Use it to categorize failures and identify root causes faster.

---

## Part 1: Failure Mode Categories

### Data/Implementation Failures
**Symptoms:** Code runs but produces nonsensical results, crashes on certain dates, inconsistent behavior across different stocks

**Diagnostic questions:**
- Are you accidentally using future information? (look-ahead bias)
- Does your pattern detection logic match what you see visually on a chart?
- Are you handling edge cases (market open/close, gaps, halts)?
- Is your bar indexing correct? (Off-by-one errors kill backtests)
- Does the data have corporate actions properly adjusted?

**Common culprits:**
- Using `close[i+1]` when you mean `close[i]`
- Pattern detection that "works" in backtests but sees the future
- Data feed inconsistencies (some days missing bars)
- Not accounting for股票 splits, dividends in historical data

---

### Framework Mismatch Failures
**Symptoms:** Fighting with the library, workarounds feel hacky, simple things require complex solutions

**Diagnostic questions:**
- Am I trying to force pattern recognition into an indicator-based framework?
- Is the framework's data structure incompatible with multi-bar pattern state?
- Am I spending more time wrestling with the library than implementing logic?
- Does this feel like I'm translating between two languages constantly?

**When to abandon the framework:**
- If you're storing state in global variables to work around framework limitations
- If pattern detection requires lookahead that the framework prevents
- If you're writing more "glue code" than actual strategy logic

**Alternative:** Raw OHLCV with custom backtesting loop. Less elegant, more control.

---

### Overfitting Failures
**Symptoms:** Amazing backtest results, terrible forward performance OR too many parameters that all need "tuning"

**Diagnostic questions:**
- How many parameters am I optimizing?
- Could a random strategy achieve similar results by chance?
- Am I testing one hypothesis or searching through thousands?
- Does this work on out-of-sample data from a different time period?

**Red flags:**
- Optimizing thresholds for every pattern individually
- "It works great from 2020-2023 but not 2024"
- Win rate above 70% with high profit factor (too good to be true)
- You can't explain WHY the pattern should work, just that it does in backtests

**The hard truth:** If you have to optimize it extensively, you're probably fitting noise.

---

### Edge Erosion Failures
**Symptoms:** Strategy works but barely beats buy-and-hold, or stops working after commissions

**Diagnostic questions:**
- What's the average profit per trade AFTER commissions and slippage?
- How many trades per day/week? (More trades = more friction costs)
- Am I capturing a real behavioral edge or just market movement?
- Would this work with realistic slippage (0.02-0.05% per trade minimum)?

**Reality check for pattern-based systems:**
- If your edge per trade is <0.5%, it's probably getting eaten by costs
- High-frequency patterns need REALLY strong edges to survive friction
- Backtests lie about fills—you don't always get the price you want

---

### Translation Failures (Visual → Code)
**Symptoms:** You see the pattern clearly, but the code misses it or finds false positives

**Diagnostic questions:**
- Can I write down the EXACT rule for every aspect of this pattern?
- What happens at edge cases? (Doji bars, equal highs, gaps)
- Am I using implicit context that I haven't explicitly coded?
- If I showed someone else my rules, could they identify the pattern without me?

**Common implicit assumptions:**
- "Obviously that's not a valid setup" (but why? what rule makes it invalid?)
- Context from previous bars that seems "obvious" but isn't coded
- Visual gestalt patterns that combine multiple discrete rules
- "Clean" vs "messy" patterns (how do you define clean numerically?)

**The test:** Can you detect the pattern on a printed table of OHLC values without looking at a chart? If not, you haven't fully formalized it yet.

---

## Part 2: Edge Validation Framework

### Is This a Real Edge or Noise?

**Noise indicators:**
- Works in backtests, fails in paper trading immediately
- Performance degrades linearly over time
- Works on only one stock/sector
- Requires constant "adjustment" to keep working
- You can't articulate WHY it should work

**Real edge indicators:**
- Based on behavioral/structural market mechanics
- Robust across different time periods (doesn't need reoptimization)
- Graceful degradation (doesn't suddenly stop working)
- Works on out-of-sample data you didn't train on
- You can explain the mechanism (even if you can't prove it)

### The Uncomfortable Question

If this edge is discoverable through automated search over historical data, why hasn't it been arbitraged away already?

**Valid answers:**
- Requires discretionary context that's hard to formalize (but then can you really automate it?)
- Only works at retail scale (edge disappears with larger size)
- It's behavioral and humans don't change quickly (maybe)
- Transaction costs make it unprofitable for institutions but viable for retail

**Invalid answers:**
- "Nobody else has thought of this" (they have)
- "The market is inefficient" (in what specific way?)
- "It works in backtests" (not an answer to why it should persist)

---

## Part 3: When Things Break

### Debugging Priority

1. **Data integrity first:** Before assuming your logic is wrong, verify the data is correct
2. **Pattern detection second:** Print every detection and manually verify a sample
3. **Position management third:** Are entries/exits happening when expected?
4. **Performance metrics last:** Don't optimize what you haven't validated

### The Nuclear Option

If you've been debugging for hours and getting nowhere:

**Start over with the simplest possible version:**
- One pattern
- One stock
- One month of data
- Print every detection
- Manually verify every trade

If the simple version doesn't work, the complex version never will.

---

## Part 4: Strat-Specific Considerations

### Pattern Recognition Implementation

**Multi-bar patterns need state tracking:**
- You can't just check current bar—you need context from previous bars
- State machines or lookback windows required
- Be explicit about when pattern "completes" vs. when it's forming

**Context requirements:**
- Previous range boundaries
- Retracement levels
- "Level of reclaim" needs precise definition
- What constitutes a "higher high" when bars have wicks?

### The Visual-to-Numerical Gap

**You see:** A clean 2-bar reversal with nice rejection wicks

**Code sees:** 
```
Bar 1: O=100.2, H=101.5, L=99.8, C=100.1
Bar 2: O=100.1, H=100.4, L=98.5, C=99.9
Is this a reversal? Need rules for:
- Minimum body size
- Wick-to-body ratio
- Relationship to previous range
- What if Bar 1 closed at 100.15 instead?
```

Every visual judgment needs to become an explicit numerical rule.

---

## Part 5: What to Do When You Don't Know

### Honest Uncertainty

Sometimes you genuinely don't know if something is a bug, a framework limitation, an overfitting issue, or bad data.

**That's okay.**

**What helps:**
- Isolate one variable at a time
- Test on known-good cases first
- Ask "what would prove this is X vs. Y?"
- Accept that some things require experimentation

**What doesn't help:**
- Guessing randomly
- Adding complexity hoping it fixes things
- Blaming external factors without evidence
- Optimizing before validating

---

## Part 6: The Real Filter

### The Only Question That Matters

**"If I ran this with real money tomorrow, would I trust it?"**

If the answer is no, you don't have a system—you have a backtest.

If the answer is "I need to test it more," be specific: test WHAT? Looking for WHAT outcome?

If the answer is "yes, but only with small size," that might actually be honest—and edges at retail scale are valid.

---

## Closing Thought

The goal isn't perfection. The goal isn't even profitability (at first). The goal is **valid testing of a hypothesis.**

If you can't tell whether your system is working or broken, you don't have enough clarity yet. Simplify until you can tell the difference.

Failures are information. Use them.
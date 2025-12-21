# Trade Audit: December 18-19, 2025

**Session:** EQUITY-28
**Status:** IN PROGRESS
**Purpose:** Verify trade entries against STRAT methodology

---

## Terminology Alignment (Confirmed This Session)

### 3-2 Pattern (3-2D or 3-2U)
- **Entry:** ON THE BREAK when forming bar breaks 3 bar's HIGH (CALL/2U) or LOW (PUT/2D)
- **Stop:** 3 bar's opposite extreme (HIGH for PUT, LOW for CALL)
- **Target:** 1.5x measured move
- **Direction:** Last bar determines direction (2D = PUT, 2U = CALL)

### 3-2-2 Pattern (Reversal Only)
- **3-2D-2U:** Entry CALL when forming bar breaks ABOVE 2D bar's HIGH
- **3-2U-2D:** Entry PUT when forming bar breaks BELOW 2U bar's LOW
- **NOT traded:** 3-2D-2D, 3-2U-2U (continuations)

### Position Flip Rule
- If in 3-2D (PUT) and reversal forms (3-2D-2U): EXIT PUT, ENTER CALL
- If in 3-2U (CALL) and reversal forms (3-2U-2D): EXIT CALL, ENTER PUT

---

## Trades to Audit

---

### TRADE 1: QQQ 3-2D-2U CALL (1W)
**Entry:** Dec 18 11:02 EST | **Price:** $5.55 | **Status:** OPEN (+$106)

| Bar Date | Open | High | Low | Close | Classification |
|----------|------|------|-----|-------|----------------|
| Nov 24 | 595.28 | 619.32 | 595.16 | 619.25 | N/A |
| Dec 01 | 613.63 | 628.92 | 612.52 | 625.48 | 2U |
| Dec 08 | 627.21 | 629.21 | 611.36 | 613.62 | **3** |
| Dec 15 | 618.37 | 618.42 | 600.28 | 617.05 | **2D** |

**Issue:** Dec 15 bar was FORMING on Dec 18 at entry time. Used forming bar as if closed.

**Audit Status:** NOT STARTED

---

### TRADE 2: AAPL 3-2D-2U CALL (1D)
**Entry:** Dec 18 11:02 EST | **Price:** $2.97 x2 | **Exit:** Dec 19 15:25 @ $1.48 | **P&L:** -$298 (MAX_LOSS)

| Bar Date | Open | High | Low | Close | Classification |
|----------|------|------|-----|-------|----------------|
| Dec 12 | 277.90 | 279.22 | 276.82 | 278.28 | N/A |
| Dec 15 | 280.15 | 280.15 | 272.84 | 274.11 | **3** |
| Dec 16 | 272.82 | 275.50 | 271.79 | 274.61 | **2D** |
| Dec 17 | 275.01 | 276.16 | 271.64 | 271.84 | **3** |
| Dec 18 | 273.61 | 273.63 | 266.95 | 272.19 | **2D** |
| Dec 19 | 272.14 | 274.60 | 269.90 | 273.67 | **2U** |

**Issue:** Dec 19 2U didn't form until 15:59 EST (1 min before close). Trade entered Dec 18 expecting 2U that hadn't formed.

**User Screenshot Notes (AAPL Daily):**
- 3-2D pattern (Dec 15=3, Dec 16=2D) made 1.8% move - MISSED (should have been PUT)
- 3-2D-2U trade entered before 2U bar actually formed
- 2U bar turned from 1 to 2U at 15:59 EST on Friday 12/19 (one minute before market close)

**Audit Status:** NOT STARTED - Fix already applied in EQUITY-27 (forming bar exclusion)

---

### TRADE 3: AAPL 3-2D PUT (1W)
**Entry:** Dec 18 11:02 EST | **Price:** $2.88 | **Status:** OPEN (-$159, -55%)

| Bar Date | Open | High | Low | Close | Classification |
|----------|------|------|-----|-------|----------------|
| Nov 24 | 270.90 | 280.38 | 270.90 | 278.85 | N/A |
| Dec 01 | 278.01 | 288.62 | 276.14 | 278.78 | 2U |
| Dec 08 | 278.13 | 280.03 | 273.81 | 278.28 | **2D** |
| Dec 15 | 280.15 | 280.15 | 266.95 | 273.67 | **3** |

**Issue:** Dec 15 bar was FORMING on Dec 18. Shows 2U-2D-3 sequence, NOT 3-2D.

**Audit Status:** NOT STARTED

---

### TRADE 4: ACHR 3-2U CALL (1H)
**Entry:** Dec 18 12:17 EST | **Price:** $0.26 | **Exit:** Dec 18 14:55 @ $0.21 | **P&L:** -$25 (STOP)

| Bar Time | Open | High | Low | Close | Classification |
|----------|------|------|-----|-------|----------------|
| 08:00 | 7.71 | 7.84 | 7.64 | 7.77 | N/A |
| 09:00 | 7.78 | 8.10 | 7.77 | 7.94 | 2U |
| 10:00 | 7.95 | 7.99 | 7.76 | 7.93 | 2D |
| 11:00 | 7.94 | 8.01 | 7.87 | 7.97 | 2U |
| **12:00** | 7.97 | 8.00 | 7.82 | 7.97 | **2D** (forming at entry) |
| 13:00 | 7.97 | 8.00 | 7.92 | 7.93 | 2U |
| 14:00 | 7.94 | 7.94 | 7.76 | 7.78 | 2D |

**Issue:** No Type 3 bar visible. Pattern shows 2U-2D-2U-2D chop, not 3-2U.

**Audit Status:** NOT STARTED

---

### TRADE 5: QBTS 3-2U CALL (1H)
**Entry:** Dec 19 09:30 EST | **Price:** $1.12 | **Exit:** Dec 19 09:46 @ $1.38 | **P&L:** +$78 (TARGET)

| Bar Time | Open | High | Low | Close | Classification |
|----------|------|------|-----|-------|----------------|
| 04:00 | 25.07 | 25.44 | 25.00 | 25.44 | N/A |
| 05:00 | 25.43 | 25.43 | 25.31 | 25.31 | 1 |
| 06:00 | 25.32 | 25.35 | 25.20 | 25.27 | 2D |
| 07:00 | 25.25 | 25.29 | 25.05 | 25.06 | 2D |
| 08:00 | 25.23 | 25.44 | 24.95 | 25.26 | **3** |
| **09:00** | 25.19 | 26.10 | 25.05 | 25.70 | **2U** (forming at entry) |
| 10:00 | 25.69 | 25.94 | 25.41 | 25.76 | 1 |
| 11:00 | 25.78 | 26.52 | 25.62 | 25.87 | 2U |

**Issue:** Valid 3-2U pattern visible (08:00=3, 09:00=2U). Entry at 09:30 during forming 2U bar. However, per STRAT 3-2U should be PUT not CALL.

**Audit Status:** NOT STARTED

---

### TRADE 6: GOOGL 3-2U CALL (1H)
**Entry:** Dec 19 09:48 EST | **Price:** $3.65 | **Status:** OPEN (+$125, +34%)

**AUDIT COMPLETE (Session EQUITY-30)**

**Issue 1: Bar Data Source Mismatch**
The original audit used extended hours data (04:00-09:00 bars). Scanner uses market-hours only:

| Bar Time (Market-Aligned) | High | Low | Classification |
|---------------------------|------|-----|----------------|
| Dec 18 14:30 | 302.85 | 301.52 | 2D |
| Dec 18 15:30 | 303.15 | 301.67 | 2U |
| Dec 19 09:30 | 306.19 | 300.97 | 3 (FORMING at 09:48) |

**Issue 2: "Let the Market Breathe" Violation**
- Entry at 09:48 violates 1H 2-bar pattern timing (minimum 10:30 AM)
- At 09:48, there were ZERO closed hourly bars for Dec 19
- The Dec 19 09:30 bar was still forming (ends at 10:30)

**Issue 3: Pattern Validity**
- Last CLOSED bars (Dec 18): 2D-2U sequence, NOT 3-2U
- No closed Type 3 bar existed at entry time
- The forming 09:30 bar (eventually Type 3) was incorrectly used as setup bar

**Root Cause:** Scanner used forming bar as setup bar AND timing filter was bypassed

**Timing Filter Investigation (EQUITY-30):**
- Filter IS implemented in daemon.py and entry_monitor.py
- Filter was deployed before Dec 19 (EQUITY-18)
- Why it failed: UNKNOWN - deferred to next session

**Verdict:** INVALID TRADE - Multiple violations
1. Timing rule violation (entry before 10:30 AM)
2. Forming bar used as setup bar (bug fixed in EQUITY-27)
3. Pattern label incorrect (3-2U vs 2D-2U)

**Audit Status:** COMPLETE

---

### TRADE 7: ACHR 3-2U-2D PUT (1H)
**Entry:** Dec 19 15:48 EST | **Price:** $0.18 | **Status:** OPEN ($0)

| Bar Time | Open | High | Low | Close | Classification |
|----------|------|------|-----|-------|----------------|
| 10:00 | 7.90 | 8.03 | 7.86 | 8.03 | N/A |
| 11:00 | 8.02 | 8.21 | 8.00 | 8.15 | 2U |
| 12:00 | 8.16 | 8.20 | 8.06 | 8.08 | 1 |
| 13:00 | 8.07 | 8.16 | 8.06 | 8.12 | 1 |
| 14:00 | 8.12 | 8.15 | 8.05 | 8.06 | **2D** |
| **15:00** | 8.05 | 8.25 | 8.05 | 8.15 | **2U** (forming at entry) |

**Issue:** Pattern shows 2U-1-1-2D-2U. No Type 3 bar. Not a 3-2U-2D pattern.

**Audit Status:** NOT STARTED

---

## Summary of Initial Observations

| Trade | Claimed Pattern | Actual Pattern at Entry | Issue |
|-------|-----------------|------------------------|-------|
| 1 | 3-2D-2U (1W) | 3-2D (forming) | Forming bar used as closed |
| 2 | 3-2D-2U (1D) | 3-2D (2U not formed) | 2U didn't form until 15:59 |
| 3 | 3-2D (1W) | 2U-2D-3 (forming) | Wrong pattern sequence |
| 4 | 3-2U (1H) | No 3 bar visible | Pattern doesn't exist |
| 5 | 3-2U (1H) | 3-2U (forming) | Valid pattern, but wrong direction per STRAT |
| 6 | 3-2U (1H) | 3-2D at entry | Wrong direction label |
| 7 | 3-2U-2D (1H) | 2U-1-1-2D-2U | No 3 bar, wrong pattern |

---

## Fixes Already Applied

| Session | Fix | Affected Trades |
|---------|-----|-----------------|
| EQUITY-27 | Exclude forming bars from directional setup detection | Trade 2 (AAPL Daily) |

---

## Next Session Tasks

1. Audit each trade in detail with user validation
2. Identify root cause of pattern misclassification
3. Update scanner code if bugs found
4. Update STRAT methodology skill with clarified terminology

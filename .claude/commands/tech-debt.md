---
name: tech-debt
description: Check technical debt status and suggest priorities
---

# Tech Debt Command

Check the current state of technical debt and suggest next priorities.

## Step 1: Read Technical Debt Plan

Read the technical debt plan file:
```
C:\Users\Chris\.claude\plans\sharded-foraging-puppy.md
```

Parse for:
- Current phase (1-4)
- Items per category
- Completion status
- Recent progress

## Step 2: Calculate Current Status

### Categories to Track

1. **Untested Modules**
   - Total modules identified
   - Modules now tested
   - Remaining count

2. **God Classes (>1000 lines)**
   - Total identified
   - Refactored count
   - Remaining count

3. **Missing Base Classes**
   - Total identified
   - Implemented count
   - Remaining count

4. **Pipeline Gaps**
   - Total identified
   - Fixed count
   - Remaining count

### Progress Calculation

```
Overall Progress: {completed}/{total} items ({percentage}%)
Current Phase: {N} - {phase description}
Phase Progress: {completed in phase}/{total in phase}
```

## Step 3: Identify Next Priorities

Prioritization criteria (in order):
1. **Blocking other work** - Items that block feature development
2. **High risk** - Items in critical paths (trade execution, signal processing)
3. **Quick wins** - Items that can be done in < 1 hour
4. **High impact** - Items that improve overall code quality significantly

For each category, identify top items.

## Step 4: Check Recent Progress

Look for:
- Items completed in last 3 sessions
- Items started but not finished
- Items that keep getting deferred

## Step 5: Output Report

```
TECHNICAL DEBT STATUS
=====================

OVERALL PROGRESS
----------------
Total Items: {X}
Completed: {Y} ({percentage}%)
Remaining: {Z}

Current Phase: {N} - {phase name}
Phase Progress: {X}/{Y} items

CATEGORY BREAKDOWN
------------------
Untested Modules:    {completed}/{total} [{████░░░░░░}]
God Classes:         {completed}/{total} [{██░░░░░░░░}]  
Missing Base Classes:{completed}/{total} [{░░░░░░░░░░}]
Pipeline Gaps:       {completed}/{total} [{██████████}]

RECENT ACTIVITY
---------------
Last 3 sessions:
- {session}: {what was completed}
- {session}: {what was completed}
- {session}: {what was completed}

In Progress:
- {item that was started}

TOP 3 PRIORITIES
----------------

1. {Item name}
   Category: {category}
   Why: {reason it's priority}
   Estimated Effort: {time estimate}
   Files: {affected files}

2. {Item name}
   Category: {category}
   Why: {reason it's priority}
   Estimated Effort: {time estimate}
   Files: {affected files}

3. {Item name}
   Category: {category}
   Why: {reason it's priority}
   Estimated Effort: {time estimate}
   Files: {affected files}

QUICK WINS (< 1 hour each)
--------------------------
- {quick win 1}
- {quick win 2}
- {quick win 3}

BLOCKERS
--------
{Items blocking other work, if any}

RECOMMENDATION
--------------
{Suggested focus for this session based on priorities and time available}
```

## Step 6: Update Tracking (Optional)

If user completed items this session, offer:
```
Would you like me to update the tech debt plan with today's progress?
Items to mark complete:
- {item 1}
- {item 2}

(y/n)
```

## Rules

- If plan file not found, report error and suggest creating one
- Do NOT modify plan file without user confirmation
- Prioritize accuracy of status over speed
- If uncertain about item status, note as "needs verification"
- Track trends (improving? stagnating? getting worse?)

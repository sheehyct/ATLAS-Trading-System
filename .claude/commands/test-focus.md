---
name: test-focus
description: Run targeted tests with failure diagnosis
---

# Test Focus Command

Run tests for a specific area and diagnose any failures.

## Parameters

- `<area>` - Required. Test area to focus on.

Valid areas:
- `strat` - All STRAT tests (bar classifier, patterns, options)
- `signal_automation` or `automation` - Signal automation tests
- `regime` - Regime detection tests
- `integrations` - ThetaData, Alpaca integration tests
- `validation` - Walk-forward, Monte Carlo, bias tests
- `crypto` - Crypto pipeline tests
- `dashboard` - Dashboard tests
- `core` - Core module tests (order validator, etc.)
- Or any specific test file name (e.g., `test_bar_classifier`)

## Step 1: Map Area to Test Path

| Area | Test Path |
|------|-----------|
| strat | tests/test_strat/ |
| signal_automation, automation | tests/test_signal_automation/ |
| regime | tests/test_regime/ |
| integrations | tests/test_integrations/ |
| validation | tests/test_validation/ |
| crypto | tests/test_crypto/ |
| dashboard | tests/test_dashboard/ |
| core | tests/test_core/ |
| {filename} | tests/**/{filename}.py |

## Step 2: Run Tests

```bash
uv run pytest {test_path} -v --tb=short
```

Capture full output.

## Step 3: Report Results

If ALL PASS:
```
TEST FOCUS: {area}
==================

Result: ALL PASSED
Tests Run: {X}
Duration: {Y seconds}

Test Summary:
{list test names with PASSED}
```

If ANY FAIL:
```
TEST FOCUS: {area}
==================

Result: {X} PASSED, {Y} FAILED
Duration: {Z seconds}

FAILURES:
---------
{For each failure:}

Test: {test_name}
File: {file_path}:{line_number}

Error:
{error message}

Relevant Code:
{show the assertion or line that failed}
```

## Step 4: Diagnose Failures

For each failed test, analyze:

### Root Cause Categories

1. **Code Bug** - Production code has a defect
   - Symptoms: Test was passing before, code recently changed
   - Evidence: Logic error visible in traceback
   - Recommendation: Fix the production code

2. **Test Bug** - Test itself is incorrect
   - Symptoms: Test expectations don't match spec
   - Evidence: Test logic is wrong, not code logic
   - Recommendation: Fix the test

3. **Spec Change** - Intentional behavior change not reflected in test
   - Symptoms: Code intentionally changed, test needs update
   - Evidence: Recent feature work changed expected behavior
   - Recommendation: Update test to match new spec

4. **Environment Issue** - External factor causing failure
   - Symptoms: Test passes locally, fails in CI, or vice versa
   - Evidence: Missing dependency, timezone issue, file path issue
   - Recommendation: Fix environment setup

5. **Flaky Test** - Intermittent failure
   - Symptoms: Passes sometimes, fails sometimes
   - Evidence: Timing-dependent, order-dependent, or random data
   - Recommendation: Make test deterministic

### Diagnosis Output

```
FAILURE DIAGNOSIS
-----------------

Test: {test_name}
Root Cause: {Code Bug | Test Bug | Spec Change | Environment | Flaky}
Confidence: {High | Medium | Low}

Analysis:
{Explanation of what went wrong}

Hypothesis:
{Specific theory about the cause}

Evidence:
- {Supporting observation 1}
- {Supporting observation 2}

Recommendation:
{Specific action to fix}

Affected Code:
{File and function/method that needs attention}
```

## Step 5: Summary

```
DIAGNOSIS SUMMARY
=================

{X} failures analyzed:
- Code Bugs: {count} → Fix production code
- Test Bugs: {count} → Fix tests
- Spec Changes: {count} → Update tests
- Environment: {count} → Fix setup
- Unknown: {count} → Need investigation

Recommended First Action:
{Most impactful fix to start with}
```

## Rules

- Do NOT auto-fix any code
- Provide diagnosis only - user decides action
- If diagnosis is uncertain, say so
- Show relevant code snippets to help user understand
- If test failure seems related to recent changes, note which commit

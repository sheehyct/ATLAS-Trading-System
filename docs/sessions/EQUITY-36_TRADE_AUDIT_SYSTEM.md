# EQUITY-36: Trade Audit & Version Tracking System

## Commit to Pull
```bash
git fetch origin claude/optimize-trade-auditing-WhsTW
git checkout claude/optimize-trade-auditing-WhsTW
# or merge into your branch:
git merge origin/claude/optimize-trade-auditing-WhsTW
```

**Commit:** `422d0fb` - `feat(audit): add trade-code version tracking system (EQUITY-36)`

---

## Problem Solved

When deploying fixes across sessions, it was impossible to answer:
- "Was this trade before or after the fix?"
- "Which trades used the buggy code?"
- "Has this fix been verified with a real trade?"

This made debugging across equity/crypto systems very difficult since fixes in shared code (like pattern detection) could affect both systems.

---

## What Was Created

### 1. Version Tracker (`utils/version_tracker.py`)

**Purpose:** Automatically captures git commit info to embed in every trade.

**Usage:**
```python
from utils.version_tracker import get_version_info

version = get_version_info()
print(version.version_string)  # "4c94dcc/EQUITY-35"
print(version.trade_metadata)  # Dict to embed in trades
```

**Every trade now includes:**
```json
{
  "code_version": "4c94dcc",
  "code_session": "EQUITY-35",
  "code_branch": "main",
  "code_dirty": false
}
```

### 2. Fix Manifest (`utils/fix_manifest.py`)

**Purpose:** Track every fix with affected components and expected impact.

**Data stored in:** `data/fix_manifest.json`

**Each fix entry includes:**
- Session ID and commit hash
- Deployment timestamp
- Affected components (daemon, position_monitor, etc.)
- Affected systems (equity, crypto, or both)
- Expected impact on trades
- Verification status

### 3. Trade Audit CLI (`scripts/trade_audit.py`)

**Purpose:** Query trades by version, trace fixes, generate audit reports.

**Commands:**
```bash
# Show current code version
python scripts/trade_audit.py version

# List trades with their code versions
python scripts/trade_audit.py list

# Find trades AFTER a fix (should work correctly)
python scripts/trade_audit.py after EQUITY-35

# Find trades BEFORE a fix (potentially buggy)
python scripts/trade_audit.py before EQUITY-35

# Trace a specific trade to see which fixes apply
python scripts/trade_audit.py trace PT_20251229_123456

# Record a new fix
python scripts/trade_audit.py fix-add \
  --session EQUITY-37 \
  --desc "Fix description" \
  --components "position_monitor,daemon" \
  --systems "equity" \
  --impact "How trades should behave now"

# Mark fix as verified after successful trade
python scripts/trade_audit.py fix-verify EQUITY-37 --notes "Verified with trade X"

# Show unverified fixes
python scripts/trade_audit.py unverified

# Full audit report
python scripts/trade_audit.py report
```

### 4. Component Dependencies (`docs/COMPONENT_DEPENDENCIES.md`)

**Purpose:** Document which code affects which trading systems.

**Key mappings:**
| If you change... | It affects... |
|-----------------|---------------|
| `strat/bar_classifier.py` | Equity AND Crypto |
| `strat/pattern_detector.py` | Equity AND Crypto |
| `strat/options_module.py` | Equity only |
| `strat/signal_automation/*` | Equity only |
| `crypto/scanning/*` | Crypto only |

### 5. Updated Trade Records

**Modified files:**
- `strat/paper_trading.py` - Added `code_version`, `code_session`, `code_branch`, `code_dirty` fields to `PaperTrade`
- `strat/trade_execution_log.py` - Added same fields to `TradeExecutionRecord`

The `create_paper_trade()` factory function now **automatically injects version info** into every new trade.

---

## Workflow After This Session

### When Deploying a Fix:
```bash
# After committing your fix
python scripts/trade_audit.py fix-add \
  --session EQUITY-37 \
  --desc "Description of what was fixed" \
  --components "component1,component2" \
  --systems "equity" \
  --impact "Expected behavior after fix"
```

### When Auditing a Trade:
```bash
# See which code version created the trade
python scripts/trade_audit.py trace <trade_id>
```

### After Confirming Fix Works:
```bash
python scripts/trade_audit.py fix-verify EQUITY-37 \
  --notes "Verified with trade PT_20251229_xyz"
```

---

## Files Created/Modified

**New files:**
- `utils/version_tracker.py` (190 lines)
- `utils/fix_manifest.py` (340 lines)
- `scripts/trade_audit.py` (520 lines)
- `docs/COMPONENT_DEPENDENCIES.md`

**Modified files:**
- `strat/paper_trading.py` - Added version tracking fields
- `strat/trade_execution_log.py` - Added version tracking fields

**Runtime data (gitignored):**
- `data/fix_manifest.json` - Stores the fix manifest

---

## Note on Import Structure

The `trade_audit.py` script uses `importlib` to directly import modules, bypassing `utils/__init__.py` which pulls in numpy/pandas. This allows the CLI to run in environments without those dependencies installed.

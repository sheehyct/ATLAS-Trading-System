# ATLAS Component Dependencies

This document maps the dependencies between components to help track which
systems are affected when code changes are made.

## Quick Reference

| If you change... | It affects... |
|-----------------|---------------|
| `strat/bar_classifier.py` | Equity AND Crypto |
| `strat/pattern_detector.py` | Equity AND Crypto |
| `strat/options_module.py` | Equity only |
| `strat/signal_automation/daemon.py` | Equity only |
| `crypto/scanning/daemon.py` | Crypto only |
| `strat/signal_automation/position_monitor.py` | Equity only |
| `crypto/simulation/position_monitor.py` | Crypto only |

## Component Map

### Shared Components (Affect Both Systems)

These files are used by BOTH equity and crypto trading systems:

```
strat/
├── bar_classifier.py         # Bar type classification (Type 1, 2U, 2D, 3)
├── pattern_detector.py       # Pattern matching (3-1-2U, 2-1-2D, etc.)
├── paper_signal_scanner.py   # Signal detection
└── tier1_detector.py         # Pattern type definitions
```

**WARNING**: Changes to these files require testing in BOTH systems!

### Equity-Only Components

These files only affect equity/options trading:

```
strat/
├── options_module.py                    # Strike selection, options pricing
├── greeks.py                            # Delta, theta, gamma, vega
└── signal_automation/
    ├── daemon.py                        # Main equity daemon
    ├── signal_store.py                  # Signal persistence
    ├── entry_monitor.py                 # Entry trigger detection
    ├── position_monitor.py              # Exit detection
    ├── executor.py                      # Order execution
    └── alerters/
        ├── discord_alerter.py           # Discord alerts
        └── logging_alerter.py           # Logging alerts
```

### Crypto-Only Components

These files only affect crypto perpetual futures trading:

```
crypto/
├── scanning/
│   ├── daemon.py                        # Main crypto daemon
│   └── signal_scanner.py                # Pattern detection
├── simulation/
│   ├── paper_trader.py                  # Simulated trading
│   └── position_monitor.py              # Exit detection
├── exchange/
│   ├── coinbase_client.py               # Coinbase integration
│   └── binance_client.py                # Binance integration
└── alerters/
    └── discord_alerter.py               # Discord alerts
```

## Dependency Flow

```
                    ┌──────────────────────┐
                    │  Shared Pattern      │
                    │  Detection           │
                    │  (bar_classifier,    │
                    │   pattern_detector)  │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │ Equity Daemon   │ │ Options Module  │ │ Crypto Daemon   │
    │ (strat/)        │ │ (equity only)   │ │ (crypto/)       │
    └────────┬────────┘ └─────────────────┘ └────────┬────────┘
             │                                        │
             ▼                                        ▼
    ┌─────────────────┐                     ┌─────────────────┐
    │ Equity Position │                     │ Crypto Position │
    │ Monitor         │                     │ Monitor         │
    └────────┬────────┘                     └────────┬────────┘
             │                                        │
             ▼                                        ▼
    ┌─────────────────┐                     ┌─────────────────┐
    │ Alpaca Executor │                     │ Coinbase Client │
    └─────────────────┘                     └─────────────────┘
```

## Trade Fields Affected by Component

When auditing trades, here's which fields each component affects:

### Pattern Detection (Shared)
- `pattern_type`
- `direction`
- `entry_trigger`
- `stop_price`
- `target_price`
- `magnitude_pct`

### Options Module (Equity)
- `strike`
- `expiration`
- `dte_at_entry`
- `delta_at_entry`
- `theta_at_entry`
- `iv_at_entry`

### Position Monitor
- `exit_time`
- `exit_reason`
- `exit_price`
- `pnl_dollars`
- `pnl_percent`

### Signal Store
- `signal_key`
- `status`
- `detected_time`
- `triggered_at`
- `expired_at`

## Audit Workflow

When you fix a bug, use this workflow:

1. **Identify affected components**:
   ```bash
   # Check which files changed
   git diff --name-only HEAD~1 HEAD
   ```

2. **Record the fix**:
   ```bash
   python scripts/trade_audit.py fix-add \
     --desc "Fix EOD exit for 1H trades" \
     --components "equity_daemon,position_monitor" \
     --systems "equity" \
     --impact "1H trades now exit at market close"
   ```

3. **After a trade completes, verify**:
   ```bash
   python scripts/trade_audit.py trace <trade_id>
   ```

4. **Mark fix as verified**:
   ```bash
   python scripts/trade_audit.py fix-verify EQUITY-35 \
     --notes "Verified with trade PT_20251229_123456"
   ```

## Version Tracking

All trades now include version metadata:

```json
{
  "trade_id": "PT_20251229_143000",
  "symbol": "SPY",
  "pattern_type": "3-2U",
  "code_version": "4c94dcc",
  "code_session": "EQUITY-35",
  "code_branch": "main",
  "code_dirty": false
}
```

Use the audit CLI to query:

```bash
# Trades after a specific fix
python scripts/trade_audit.py after EQUITY-35

# Trades before a fix (potentially buggy)
python scripts/trade_audit.py before EQUITY-35

# Full audit report
python scripts/trade_audit.py report
```

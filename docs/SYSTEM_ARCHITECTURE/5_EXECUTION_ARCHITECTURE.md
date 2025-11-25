# ATLAS Trading System - Execution Layer Architecture

## Overview

The execution layer provides the infrastructure for deploying validated strategies to live/paper trading accounts. This layer translates strategy signals into actual broker orders, monitors execution, and maintains audit trails.

**Critical Distinction:** This layer is used ONLY after strategies pass paper trading validation (6+ months, 100+ trades). It is NOT part of the backtesting infrastructure.

**Design Philosophy:**
- Separation of Concerns: Strategy logic, data pipeline, and execution are independent layers
- Audit Trail: Every action logged for compliance and debugging
- Risk Management: Pre-submission validation prevents invalid orders
- Resilience: Retry logic and error handling for network/API failures
- Testability: Mock external APIs in tests, validate with paper trading first

---

## Component Architecture

### 1. AlpacaTradingClient (integrations/)

**Purpose:** Wrapper around alpaca-py SDK for order execution

**Location:** `integrations/alpaca_trading_client.py`

**Responsibilities:**
- Connect to Alpaca Trading API (paper or live account)
- Submit orders (market, limit, stop, trailing-stop)
- Query account info (equity, buying power, positions)
- Track order status (submitted, filled, canceled, rejected)
- Handle API errors and rate limits
- Retry failed API calls with exponential backoff

**NOT Responsible For:**
- Historical data fetching (use `data/alpaca_client.py`)
- Strategy signal generation (use `strategies/`)
- Position sizing calculations (use `utils/position_sizing.py`)
- Risk validation (use `core/order_validator.py`)

**Interface:**
```python
class AlpacaTradingClient:
    def __init__(self, account: str = 'LARGE', logger: Optional[logging.Logger] = None)
    def connect() -> bool
    def get_account() -> dict
    def get_positions() -> List[dict]
    def submit_market_order(symbol: str, qty: int, side: str) -> dict
    def submit_limit_order(symbol: str, qty: int, side: str, limit_price: float) -> dict
    def cancel_order(order_id: str) -> bool
    def get_order_status(order_id: str) -> dict
    def close_position(symbol: str) -> dict
    def is_market_open() -> bool
```

**Error Handling:**
- Network failures: Retry with exponential backoff (1s, 2s, 4s)
- API errors: Log detailed error, raise exception
- Rate limits: Wait and retry (respect 429 headers)
- Invalid symbols: Validate before submission

**Configuration:**
- Credentials: Load from .env (ALPACA_LARGE_KEY, ALPACA_LARGE_SECRET)
- Endpoint: Paper (https://paper-api.alpaca.markets/v2) or live
- Timeout: 30 seconds per API call
- Max retries: 3 attempts

---

### 2. OrderValidator (core/)

**Purpose:** Pre-submission risk checks to prevent invalid orders

**Location:** `core/order_validator.py`

**Responsibilities:**
- Validate buying power sufficient for order
- Check position size within limits (max 15% per position)
- Check total portfolio allocation (max 105% deployed)
- Prevent duplicate orders (same symbol, side, qty)
- Verify market hours (9:30-16:00 ET, NYSE calendar)
- Enforce regime compliance (CRASH regime = 0% allocation)
- Validate symbol format and tradability

**NOT Responsible For:**
- Order submission (use AlpacaTradingClient)
- Signal generation (use strategies/)
- Logging (use ExecutionLogger)

**Interface:**
```python
class OrderValidator:
    def __init__(self, max_position_pct: float = 0.15, max_portfolio_heat: float = 0.08)

    def validate_buying_power(account_info: dict, order_value: float) -> Tuple[bool, str]
    def validate_position_size(order_value: float, portfolio_value: float) -> Tuple[bool, str]
    def validate_total_allocation(positions: List[dict], new_orders: List[dict]) -> Tuple[bool, str]
    def validate_no_duplicate_orders(pending_orders: List[dict], symbol: str, side: str) -> Tuple[bool, str]
    def validate_market_hours(current_time: datetime) -> Tuple[bool, str]
    def validate_regime_compliance(regime: str, allocation_pct: float) -> Tuple[bool, str]
    def validate_symbol(symbol: str) -> Tuple[bool, str]

    def validate_order_batch(orders: List[dict], account_info: dict, regime: str) -> dict
```

**Returns:**
```python
{
    'valid': bool,              # True if all checks pass
    'errors': List[str],        # Critical errors (prevents submission)
    'warnings': List[str]       # Non-critical warnings (logged but allow submission)
}
```

**Validation Gates:**
1. Buying Power: order_value <= account.buying_power
2. Position Size: order_value <= portfolio_value * 0.15 (diversification)
3. Total Allocation: sum(positions + new_orders) <= portfolio_value * 1.05 (over-allocation check)
4. Duplicate Check: No pending order for same symbol+side+qty
5. Market Hours: 9:30 AM - 4:00 PM ET on trading days
6. Regime Compliance: CRASH regime -> 0% allocation, BEAR -> 30%, NEUTRAL -> 70%, BULL -> 100%
7. Symbol Validity: Alpaca-tradable symbol, proper format

---

### 3. ExecutionLogger (utils/)

**Purpose:** Centralized logging for all execution events (audit trail)

**Location:** `utils/execution_logger.py`

**Responsibilities:**
- Log all order submissions (symbol, qty, side, order_type, timestamp)
- Log order fills (fill_price, fill_qty, commission)
- Log order rejections (reason, timestamp)
- Log position updates (open, close, adjust)
- Log errors (API failures, validation failures)
- Maintain CSV audit trail for compliance

**NOT Responsible For:**
- Order submission (use AlpacaTradingClient)
- Order validation (use OrderValidator)
- Strategy logic (use strategies/)

**Interface:**
```python
class ExecutionLogger:
    def __init__(self, log_dir: str = 'logs/')

    def log_order_submission(symbol: str, qty: int, side: str, order_type: str, order_id: str)
    def log_order_fill(order_id: str, fill_price: float, fill_qty: int, commission: float)
    def log_order_rejection(order_id: str, reason: str)
    def log_position_update(symbol: str, action: str, qty: int, price: float)
    def log_error(component: str, error_msg: str, exc_info: Optional[Exception] = None)
    def log_reconciliation(target_positions: dict, actual_positions: dict, discrepancies: List[str])
```

**Log Destinations:**
1. **Console:** INFO level and above (human-readable, real-time monitoring)
2. **File:** logs/execution_{date}.log (all levels, DEBUG to CRITICAL)
3. **CSV:** logs/trades_{date}.csv (trade events only, for analysis)
4. **File:** logs/errors_{date}.log (ERROR and CRITICAL only, for debugging)

**Log Format:**
```
Console/File: [2025-11-18 09:35:42] [INFO] AlpacaTradingClient: Submitting market order: BUY 10 SPY
CSV:          2025-11-18 09:35:42,SPY,BUY,10,450.25,market,abc123,submitted,
```

**CSV Columns:**
- timestamp: YYYY-MM-DD HH:MM:SS
- symbol: Ticker symbol
- action: BUY, SELL, CANCEL, FILL, REJECT
- qty: Share quantity
- price: Fill price (if applicable)
- order_type: market, limit, stop, etc.
- order_id: Alpaca order ID
- status: submitted, filled, canceled, rejected
- error: Error message (if rejection)

**Log Rotation:**
- Daily rotation (new file each day)
- Keep last 90 days
- Archive older logs to logs/archive/

---

## Execution Pipeline Flow

### Signal-to-Order Pipeline

```
Strategy Signal Generation
         |
         v
Regime Detection (ATLAS)
         |
         v
Apply Regime Allocation Percentage
         |
         v
Calculate Target Positions
         |
         v
Generate Rebalancing Orders (close, adjust, open)
         |
         v
Order Validation (OrderValidator)
         |---> [REJECT] --> Log Error --> Exit
         |
         v [PASS]
Submit Orders (AlpacaTradingClient)
         |
         v
Monitor Order Status (poll every 10s for 5 min)
         |---> [TIMEOUT] --> Log Warning --> Continue
         |---> [REJECTED] --> Log Error --> Continue
         |
         v [FILLED]
Update Position Tracking
         |
         v
Reconciliation (compare target vs actual)
         |
         v
Log Summary Report
```

### Rebalancing Workflow (scripts/execute_52w_rebalance.py)

**Step 1: Initialization**
- Parse command-line arguments (--dry-run, --date, --universe, --top-n)
- Check if rebalance date (Feb 1, Aug 1, or --force flag)
- Initialize ExecutionLogger
- Initialize AlpacaTradingClient
- Initialize OrderValidator

**Step 2: Fetch Current State**
- Get account info (equity, buying power)
- Get current positions (symbol, qty, market value)
- Get current regime (ATLAS online_inference)

**Step 3: Generate Signals**
- Run stock scanner (integrations/stock_scanner_bridge.py)
- Select top N stocks by momentum score
- Apply volume filter (if configured)

**Step 4: Apply Regime Allocation**
```python
regime_allocation = {
    'TREND_BULL': 1.00,     # 100% deployed
    'TREND_NEUTRAL': 0.70,  # 70% deployed
    'TREND_BEAR': 0.30,     # 30% deployed
    'CRASH': 0.00           # 0% deployed (cash)
}
```

**Step 5: Calculate Target Positions**
- Determine portfolio value to deploy (equity * regime_allocation)
- Calculate per-stock allocation (equal weight or momentum-weighted)
- Generate target position list: {symbol: target_qty}

**Step 6: Generate Rebalancing Orders**
```python
# Close positions not in new portfolio
close_orders = [position for position in current_positions
                if position.symbol not in target_positions]

# Adjust positions that remain (size changes)
adjust_orders = [position for position in current_positions
                 if position.symbol in target_positions
                 and position.qty != target_positions[position.symbol]]

# Open new positions
open_orders = [symbol for symbol in target_positions
               if symbol not in current_positions]
```

**Step 7: Validate Orders**
```python
validation_result = order_validator.validate_order_batch(
    orders=all_orders,
    account_info=account,
    regime=current_regime
)

if not validation_result['valid']:
    logger.error(f"Validation failed: {validation_result['errors']}")
    exit(1)
```

**Step 8: Submit Orders (if not dry-run)**
```python
for order in all_orders:
    try:
        result = trading_client.submit_market_order(
            symbol=order.symbol,
            qty=order.qty,
            side=order.side
        )
        logger.log_order_submission(order.symbol, order.qty, order.side, 'market', result.id)
    except Exception as e:
        logger.log_error('OrderSubmission', f"Failed to submit {order}: {e}")
```

**Step 9: Monitor Fills**
```python
timeout = 300  # 5 minutes
start_time = time.time()

while time.time() - start_time < timeout:
    unfilled_orders = [order for order in submitted_orders
                       if trading_client.get_order_status(order.id).status != 'filled']

    if not unfilled_orders:
        break  # All orders filled

    time.sleep(10)  # Poll every 10 seconds
```

**Step 10: Reconciliation**
```python
final_positions = trading_client.get_positions()

discrepancies = []
for symbol, target_qty in target_positions.items():
    actual_qty = next((p.qty for p in final_positions if p.symbol == symbol), 0)
    if abs(actual_qty - target_qty) > 1:  # Allow 1 share tolerance
        discrepancies.append(f"{symbol}: target {target_qty}, actual {actual_qty}")

if discrepancies:
    logger.log_reconciliation(target_positions, final_positions, discrepancies)
```

**Step 11: Summary Report**
```python
summary = {
    'total_orders': len(submitted_orders),
    'filled_orders': len([o for o in submitted_orders if o.status == 'filled']),
    'rejected_orders': len([o for o in submitted_orders if o.status == 'rejected']),
    'total_slippage': sum([o.fill_price - o.expected_price for o in filled_orders]),
    'final_equity': account.equity,
    'regime': current_regime,
    'allocation_pct': regime_allocation[current_regime]
}

logger.info(f"Rebalance complete: {summary}")
```

---

## Error Handling and Recovery

### Error Categories

**1. Network Errors (Transient)**
- Symptoms: Connection timeout, DNS failure, SSL error
- Handling: Retry with exponential backoff (1s, 2s, 4s)
- Max retries: 3 attempts
- Logging: WARNING level

**2. API Errors (Permanent)**
- Symptoms: 400 Bad Request, 403 Forbidden, 404 Not Found
- Handling: Log error, raise exception (do not retry)
- Max retries: 0 (immediate failure)
- Logging: ERROR level

**3. Rate Limit Errors (Temporary)**
- Symptoms: 429 Too Many Requests
- Handling: Wait for retry-after header duration, then retry
- Max retries: 3 attempts
- Logging: WARNING level

**4. Validation Errors (Pre-submission)**
- Symptoms: Insufficient buying power, position size exceeded
- Handling: Log error, skip order (do not submit)
- Max retries: 0 (validation failure is permanent)
- Logging: ERROR level

**5. Order Rejection Errors (Post-submission)**
- Symptoms: Insufficient funds, invalid symbol, market closed
- Handling: Log rejection, continue with remaining orders
- Max retries: 0 (broker rejected, cannot override)
- Logging: ERROR level

### Recovery Procedures

**Scenario 1: Partial Fill Failure**
- Problem: Some orders filled, others timed out
- Recovery:
  1. Log partial fills
  2. Calculate actual vs target positions
  3. Generate new orders for remaining positions
  4. Re-validate and re-submit
  5. Document manual intervention required if still fails

**Scenario 2: Network Failure Mid-Execution**
- Problem: Lost connection during order submission
- Recovery:
  1. Reconnect to Alpaca API
  2. Query account positions and pending orders
  3. Reconcile submitted vs filled orders
  4. Cancel orphaned pending orders
  5. Generate new orders for remaining positions

**Scenario 3: Order Rejection Due to Insufficient Funds**
- Problem: Buying power decreased between validation and submission
- Recovery:
  1. Log rejection error
  2. Re-fetch account info
  3. Re-calculate position sizes with updated buying power
  4. Re-validate orders
  5. Re-submit with adjusted quantities

**Scenario 4: CRASH Regime During Rebalance**
- Problem: Regime transitions to CRASH mid-execution
- Recovery:
  1. Cancel all pending BUY orders
  2. Allow SELL orders to complete
  3. Move to 100% cash (0% allocation)
  4. Log regime transition event
  5. Wait for next rebalance date

---

## Logging and Monitoring Standards

### Logging Levels

**DEBUG:** Detailed information for debugging
- API request/response payloads
- Calculation intermediates
- Retry attempt details

**INFO:** Normal operation events
- Order submissions
- Order fills
- Position updates
- Account balance changes

**WARNING:** Unusual but recoverable events
- Retrying API call after failure
- Order partially filled
- Slippage exceeded threshold
- Minor reconciliation discrepancies

**ERROR:** Errors that prevent completion
- Order validation failures
- API errors after max retries
- Order rejections
- Reconciliation failures

**CRITICAL:** System failures requiring immediate attention
- Cannot connect to Alpaca API
- Cannot load credentials
- Cannot write to log files
- Repeated order submission failures

### Monitoring Dashboards

**Portfolio Status Dashboard (view_portfolio_status.py)**

Display:
```
ATLAS Trading System - Portfolio Status
========================================
Date: 2025-11-18 14:32:05 ET
Market: OPEN (closes in 1h 28m)

ACCOUNT (LARGE - $10,000)
  Equity: $10,245.67
  Buying Power: $2,150.23
  Cash: $2,150.23

CURRENT REGIME
  TREND_BULL (100% allocation)
  Days until next rebalance: 74 (Feb 1, 2026)

POSITIONS (8 holdings)
  Symbol    Qty    Value     Cost Basis    P&L        P/L %    % Portfolio
  ------    ---    -----     ----------    ---        -----    -----------
  AAPL      8      $1,520    $1,480       +$40       +2.7%     14.8%
  MSFT      4      $1,680    $1,640       +$40       +2.4%     16.4%
  NVDA      15     $1,650    $1,600       +$50       +3.1%     16.1%
  ...

PERFORMANCE SINCE LAST REBALANCE (Aug 1, 2025)
  Total Return: +2.46%
  SPY Return: +1.82%
  Outperformance: +0.64%
  Sharpe Ratio: 0.95
  Max Drawdown: -3.21%

LAST REBALANCE
  Date: 2025-08-01
  Orders: 18 (10 filled, 0 rejected)
  Regime: TREND_NEUTRAL (70% allocation)
```

**Execution Log Monitoring**

Watch logs/execution_{today}.log in real-time:
```bash
tail -f logs/execution_2025-11-18.log
```

Key events to monitor:
- Order submissions (count should match expected)
- Order fills (all should fill within 5 minutes)
- Validation errors (should be zero for normal operation)
- API errors (transient errors acceptable, permanent errors need investigation)

**Trade History Analysis**

Query logs/trades_{date}.csv:
```python
import pandas as pd

trades = pd.read_csv('logs/trades_2025-11-18.csv')

# Calculate slippage
trades['slippage'] = trades['fill_price'] - trades['expected_price']
avg_slippage = trades['slippage'].mean()

# Analyze rejections
rejections = trades[trades['status'] == 'rejected']
print(f"Rejection rate: {len(rejections) / len(trades):.1%}")
print(f"Rejection reasons: {rejections['error'].value_counts()}")
```

---

## Testing Strategy

### Unit Tests (tests/test_integrations/, tests/test_execution/)

**Mock External APIs:**
- Mock alpaca-py TradingClient (do NOT hit live API in tests)
- Mock AlpacaTradingClient responses
- Test all public methods
- Test error scenarios (network failures, API errors, rate limits)

**Test Coverage Goals:**
- AlpacaTradingClient: 90%+
- OrderValidator: 95%+ (critical risk checks)
- ExecutionLogger: 85%+
- execute_52w_rebalance.py: 80%+ (integration test)

### Integration Tests (tests/test_execution/test_end_to_end.py)

**Dry-Run Historical Rebalances:**
- Test dates: 2024-02-01 (BULL), 2024-08-01 (NEUTRAL), 2020-03-15 (CRASH)
- Verify: Signal generation, regime allocation, order generation, validation
- No API calls: Use cached data and mock trading client

### Paper Trading Validation (Manual, 30+ Days)

**First Test Trade:**
- Configuration: Top 3 stocks, $300 total ($100 each)
- Purpose: Validate execution pipeline with minimal capital
- Verify: Orders submit, fill within 5 min, positions match target

**First Real Rebalance:**
- Configuration: Full 10-stock portfolio, $1,000 per position
- Date: Next semi-annual rebalance (Feb 1 or Aug 1)
- Monitor: Order quality, fill prices, slippage, reconciliation

**Validation Period:**
- Duration: 30+ days minimum (6 months preferred)
- Frequency: Daily monitoring via view_portfolio_status.py
- Metrics: Compare actual vs backtest performance (Sharpe, CAGR, MaxDD)

---

## Deployment Checklist

### Pre-Deployment (Session 42-44)

- [ ] Architecture documentation complete (this file)
- [ ] AlpacaTradingClient implemented and tested
- [ ] OrderValidator implemented and tested
- [ ] ExecutionLogger implemented and tested
- [ ] execute_52w_rebalance.py implemented
- [ ] Unit tests passing (90%+ coverage)
- [ ] Integration tests passing (dry-run historical rebalances)
- [ ] First test trade successful (small position sizes)
- [ ] Scheduler configured (semi-annual rebalances)
- [ ] Monitoring dashboard operational

### Paper Trading (Sessions 45+)

- [ ] Paper account connected (LARGE account, $10,000)
- [ ] First real rebalance executed successfully
- [ ] Daily monitoring active (30+ days)
- [ ] Performance within expectations (Sharpe >= 0.8, CAGR >= 10%)
- [ ] No infrastructure failures or errors
- [ ] Reconciliation passes daily (positions match target)
- [ ] Slippage analysis acceptable (<0.5% average)

### Pre-Live Deployment (Sessions 50+)

- [ ] 6 months paper trading complete (100+ trades)
- [ ] Performance validated (within 30% of backtest)
- [ ] All tests passing (no regressions)
- [ ] Live account credentials configured (.env updated)
- [ ] Risk limits reviewed and approved
- [ ] Backup procedures documented
- [ ] Emergency stop mechanism tested

---

## Paper vs Live Execution Distinction

### Paper Trading

**Purpose:** Validate execution infrastructure without real capital risk

**Configuration:**
- Endpoint: https://paper-api.alpaca.markets/v2
- Credentials: ALPACA_LARGE_KEY (paper account)
- Initial Capital: $10,000 (simulated)
- Orders: Real API calls but simulated fills
- Fill simulation: Alpaca paper trading engine

**Limitations:**
- Fill simulation may not match live market (no real liquidity)
- Slippage estimates may be optimistic
- No real commission costs
- Market impact not simulated

**Validation Period:** 6 months minimum, 100+ trades

### Live Trading

**Purpose:** Deploy validated strategies with real capital

**Configuration:**
- Endpoint: https://api.alpaca.markets/v2
- Credentials: ALPACA_LIVE_KEY (live account)
- Initial Capital: User-determined
- Orders: Real API calls, real market fills
- Fill execution: Real broker execution

**Requirements:**
- Paper trading validation complete (6+ months)
- Performance within expectations (Sharpe >= 0.8, CAGR >= 10%)
- User approval and understanding of risks
- Adequate capital for strategy ($10,000+ for 52W momentum)

**Transition Procedure:**
1. Update .env credentials (paper -> live)
2. Run dry-run test with live account (verify connectivity)
3. Execute first live rebalance with small position sizes (25% normal)
4. Monitor closely for 30 days
5. Scale up to full position sizes if successful
6. Continue daily monitoring indefinitely

---

## Summary

The execution layer provides the critical infrastructure to deploy validated strategies to live/paper trading accounts. Key components:

1. **AlpacaTradingClient:** Wrapper for broker API (order submission, position tracking)
2. **OrderValidator:** Pre-submission risk checks (buying power, position limits, regime compliance)
3. **ExecutionLogger:** Audit trail for all execution events (compliance, debugging)

**Design Principles:**
- Separation of concerns (strategy, data, execution are independent)
- Audit trail (every action logged)
- Risk management (validation before submission)
- Resilience (retry logic, error handling)
- Testability (mock APIs, dry-run mode)

**Deployment Path:**
- Build infrastructure (Sessions 42-44)
- Validate with paper trading (30+ days minimum)
- Deploy to live (only after validation passes)

This architecture enables safe, auditable, and reliable strategy deployment from backtested signals to real market execution.

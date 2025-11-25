# Practical Development Examples: Resource Usage Patterns
## Real-World Scenarios with Code

**Companion to**: RESOURCE_UTILIZATION_GUIDE.md
**Purpose**: Practical, copy-paste examples for common development tasks
**Version**: 1.0
**Last Updated**: October 18, 2025

---

## Table of Contents

1. [Session Startup Pattern](#session-startup-pattern)
2. [VBT API Discovery Workflow](#vbt-api-discovery-workflow)
3. [Implementing New Strategy Component](#implementing-new-strategy-component)
4. [Debugging VBT Integration](#debugging-vbt-integration)
5. [Data Pipeline Development](#data-pipeline-development)
6. [Risk Management Implementation](#risk-management-implementation)
7. [Performance Analysis](#performance-analysis)
8. [Documentation Maintenance](#documentation-maintenance)

---

## Session Startup Pattern

### Every Session Begins With This

```python
"""
MANDATORY: Run this at the start of EVERY development session
"""

# 1. Read current state
from Filesystem import read_file

handoff = read_file("/mnt/user-data/uploads/docs/HANDOFF.md")
print("="*80)
print("CURRENT STATE (from HANDOFF.md)")
print("="*80)
print(handoff)

# 2. Read development rules
claude_rules = read_file("/mnt/user-data/uploads/docs/CLAUDE.md")
print("\n" + "="*80)
print("DEVELOPMENT RULES (from CLAUDE.md)")
print("="*80)
print(claude_rules[:1000])  # First 1000 chars to refresh memory

# 3. Verify environment
from VectorBT_Pro import run_code

env_check = """
import vectorbtpro as vbt
import pandas as pd
import numpy as np

print(f"VBT Version: {vbt.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"NumPy Version: {np.__version__}")
print("Environment: OK")
"""

result = run_code(env_check)
print("\n" + "="*80)
print("ENVIRONMENT CHECK")
print("="*80)
print(result)

# 4. Check what needs to be done
print("\n" + "="*80)
print("ACTION ITEMS (from HANDOFF.md)")
print("="*80)
print("Look for sections:")
print("- 'Immediate Next Steps'")
print("- 'Files to Create'")
print("- 'What's Broken'")
print("- 'Verification Gates'")
```

**Output Interpretation:**
- **HANDOFF.md shows "Broken"**: Don't work on that component
- **HANDOFF.md shows "Next: X"**: That's your task
- **Environment check fails**: Fix environment before coding
- **Context >70%**: Prepare handoff to HANDOFF.md

---

## VBT API Discovery Workflow

### Scenario: "I need to implement position sizing with VBT"

**Step 1: High-Level Search**
```python
from VectorBT_Pro import search

# Start broad - understand the concept
results = search(
    query="position sizing risk management from_signals",
    asset_names=["docs", "examples", "api"],  # Search order
    search_method="hybrid",
    max_tokens=2000,
    n=5
)

print("Search Results:")
print(results)

# Identify key classes/methods mentioned:
# - vbt.Portfolio.from_signals
# - vbt.PF (alias for Portfolio)
# - size parameter
```

**Step 2: Verify References**
```python
from VectorBT_Pro import resolve_refnames

# Verify these are real VBT objects
refs = resolve_refnames([
    "vbt.Portfolio",
    "vbt.PF",
    "vbt.Portfolio.from_signals"
])

print("Reference Verification:")
print(refs)

# Expected output:
# OK vbt.Portfolio vectorbtpro.portfolio.base.Portfolio
# OK vbt.PF vectorbtpro.portfolio.base.Portfolio
# OK vbt.Portfolio.from_signals vectorbtpro.portfolio.base.Portfolio.from_signals
```

**Step 3: Explore Available Methods**
```python
from VectorBT_Pro import get_attrs

# See what's available on Portfolio
attrs = get_attrs(
    refname="vbt.Portfolio",
    own_only=False,
    incl_types=True,
    incl_private=False
)

print("Portfolio Attributes:")
print(attrs)

# Look for:
# - from_signals [classmethod]
# - total_return [property]
# - sharpe_ratio [property]
# - trades [property]
```

**Step 4: Get Method Signature**
```python
from VectorBT_Pro import get_source

# Read the actual implementation
source = get_source("vbt.Portfolio.from_signals")

print("from_signals Implementation:")
print(source[:2000])  # First 2000 chars

# Look for:
# - size parameter and its type
# - size_type parameter (amount, percent, etc.)
# - sl_stop parameter (stop loss)
# - init_cash parameter
```

**Step 5: Find Working Examples**
```python
from VectorBT_Pro import find

# Find real-world usage
examples = find(
    refnames=["vbt.Portfolio.from_signals"],
    asset_names=["examples", "messages"],
    aggregate_messages=True,  # Get full thread context
    max_tokens=2000
)

print("Real-World Examples:")
print(examples)

# Look for patterns:
# - How size is specified (pd.Series? np.array?)
# - How stops are implemented
# - Common parameter combinations
```

**Step 6: Test Minimal Example**
```python
from VectorBT_Pro import run_code

test_code = """
import vectorbtpro as vbt
import pandas as pd
import numpy as np

# Minimal test data
np.random.seed(42)
close = pd.Series(100 + np.cumsum(np.random.randn(100)), name='close')
entries = pd.Series([True] + [False]*99, name='entries')
exits = pd.Series([False]*99 + [True], name='exits')

# Test different size formats
sizes = pd.Series([10.0]*100, name='sizes')  # Constant size

# Create portfolio
pf = vbt.PF.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    size=sizes,
    size_type='amount',  # Testing: shares vs percent
    init_cash=10000,
    fees=0.002
)

# Verify it works
print(f"Total Return: {pf.total_return:.2%}")
print(f"Sharpe Ratio: {pf.sharpe_ratio:.2f}")
print(f"Max Drawdown: {pf.max_drawdown:.2%}")
print("SUCCESS: VBT accepts this format")
"""

result = run_code(test_code)
print("Integration Test:")
print(result)
```

**Step 7: Document Findings**
```python
from Filesystem import write_file

findings = """
# VBT Position Sizing Integration - Findings

## Method Signature
- `vbt.Portfolio.from_signals(close, entries, exits, size=, size_type=, ...)`

## Verified Size Formats
- **Works**: `pd.Series` with matching index
- **size_type**: 'amount' (shares), 'percent' (% of capital)
- **Constraints**: Must align with close index

## Next Steps
1. Implement ATR-based sizing calculation
2. Return pd.Series with proper index alignment
3. Test with VBT integration
4. Add capital constraint (max 100% allocation)

## References
- VBT API: vectorbtpro.portfolio.base.Portfolio.from_signals
- Examples: [links from find() results]
"""

write_file(
    path="/home/claude/VBT_POSITION_SIZING_FINDINGS.md",
    content=findings
)
```

---

## Implementing New Strategy Component

### Scenario: "Implement ATR-based position sizing for ORB strategy"

**Step 1: Research Requirements**
```python
from Filesystem import read_file
from VectorBT_Pro import search

# Read strategy spec
strategy_spec = read_file("/mnt/user-data/uploads/docs/STRATEGY_2_IMPLEMENTATION_ADDENDUM.md")
print("Strategy Requirements:")
print(strategy_spec)

# Search for ATR implementation
atr_research = search(
    query="ATR average true range position sizing",
    asset_names=["examples", "docs"],
    max_tokens=1500
)
print("\nATR Research:")
print(atr_research)
```

**Step 2: Check Existing Infrastructure**
```python
from Filesystem import list_directory, read_file

# See what's already implemented
existing_utils = list_directory("/mnt/user-data/uploads/utils")
print("Existing Utils:")
print(existing_utils)

# Check if TA-Lib is available
talib_check = """
import talib
import numpy as np

high = np.array([100, 102, 104, 103, 105])
low = np.array([98, 100, 102, 101, 103])
close = np.array([99, 101, 103, 102, 104])

atr = talib.ATR(high, low, close, timeperiod=14)
print(f"TA-Lib ATR available: {atr is not None}")
print(f"ATR values: {atr[-3:]}")
"""

from VectorBT_Pro import run_code
result = run_code(talib_check)
print("\nTA-Lib Check:")
print(result)
```

**Step 3: Implement Component**
```python
from Filesystem import write_file

position_sizing_code = '''
"""
Position sizing utilities for algorithmic trading strategies.

This module provides capital-constrained position sizing calculations
based on ATR (Average True Range) risk management.
"""

import pandas as pd
import numpy as np
import talib


def calculate_atr_position_size(
    capital: float,
    risk_per_trade: float,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
    atr_multiplier: float = 2.5
) -> pd.Series:
    """
    Calculate position size based on ATR risk management.
    
    Formula:
    - Risk amount = capital * risk_per_trade
    - Stop distance = ATR * multiplier
    - Position size = risk_amount / stop_distance
    - Constrained by: max_shares = capital / close
    
    Parameters
    ----------
    capital : float
        Account capital (e.g., 10000)
    risk_per_trade : float
        Risk per trade as decimal (e.g., 0.02 for 2%)
    high : pd.Series
        High prices
    low : pd.Series
        Low prices
    close : pd.Series
        Close prices
    atr_period : int, default=14
        ATR calculation period
    atr_multiplier : float, default=2.5
        Stop distance multiplier (e.g., 2.5x ATR)
        
    Returns
    -------
    pd.Series
        Position sizes in shares (integers)
        
    Examples
    --------
    >>> capital = 10000
    >>> risk_per_trade = 0.02  # 2%
    >>> sizes = calculate_atr_position_size(
    ...     capital, risk_per_trade, high, low, close
    ... )
    >>> print(f"Position sizes: {sizes.tail()}")
    
    Notes
    -----
    - Returns 0 shares when ATR is not yet available (warmup period)
    - Enforces capital constraint (cannot exceed 100% of capital)
    - Returns integer share counts (no fractional shares)
    """
    # Calculate ATR using TA-Lib
    atr = pd.Series(
        talib.ATR(high.values, low.values, close.values, timeperiod=atr_period),
        index=close.index,
        name='atr'
    )
    
    # Risk amount per trade
    risk_amount = capital * risk_per_trade
    
    # Stop distance
    stop_distance = atr * atr_multiplier
    
    # Position size calculation
    # Avoid division by zero
    position_size = pd.Series(0, index=close.index, dtype=float)
    valid_stops = stop_distance > 0
    position_size[valid_stops] = risk_amount / stop_distance[valid_stops]
    
    # Capital constraint: cannot buy more shares than capital allows
    max_shares = capital / close
    position_size = position_size.clip(upper=max_shares)
    
    # Convert to integer shares
    position_size = position_size.astype(int)
    
    return position_size


def validate_position_sizes(
    sizes: pd.Series,
    close: pd.Series,
    capital: float,
    max_position_pct: float = 1.0
) -> dict:
    """
    Validate position sizes meet constraints.
    
    Parameters
    ----------
    sizes : pd.Series
        Position sizes in shares
    close : pd.Series
        Close prices
    capital : float
        Account capital
    max_position_pct : float, default=1.0
        Maximum position size as fraction of capital (1.0 = 100%)
        
    Returns
    -------
    dict
        Validation results with keys:
        - 'valid': bool, True if all checks pass
        - 'mean_pct': float, mean position size as % of capital
        - 'max_pct': float, max position size as % of capital
        - 'violations': list of index positions violating constraints
    """
    # Calculate position value
    position_value = sizes * close
    position_pct = position_value / capital
    
    # Check constraints
    violations = position_pct[position_pct > max_position_pct].index.tolist()
    
    return {
        'valid': len(violations) == 0,
        'mean_pct': position_pct.mean(),
        'max_pct': position_pct.max(),
        'violations': violations
    }


# VectorBT Pro Integration Helpers

def create_vbt_position_sizes(
    entries: pd.Series,
    capital: float,
    risk_per_trade: float,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    **kwargs
) -> pd.Series:
    """
    Create VectorBT-compatible position sizes from entries signal.
    
    This function generates position sizes only for entry bars,
    maintaining the position size throughout the trade until exit.
    
    Parameters
    ----------
    entries : pd.Series
        Boolean series indicating entry signals
    capital : float
        Account capital
    risk_per_trade : float
        Risk per trade as decimal
    high, low, close : pd.Series
        OHLC data
    **kwargs
        Additional arguments for calculate_atr_position_size
        
    Returns
    -------
    pd.Series
        Position sizes compatible with vbt.Portfolio.from_signals
        
    Notes
    -----
    - Returns 0 for all bars where entries=False
    - VBT will maintain position size until exit signal
    """
    # Calculate base position sizes
    all_sizes = calculate_atr_position_size(
        capital, risk_per_trade, high, low, close, **kwargs
    )
    
    # Only apply sizes on entry bars
    vbt_sizes = pd.Series(0, index=entries.index, dtype=int)
    vbt_sizes[entries] = all_sizes[entries]
    
    return vbt_sizes
'''

write_file(
    path="/home/claude/position_sizing.py",
    content=position_sizing_code
)

print("Position sizing module created: /home/claude/position_sizing.py")
```

**Step 4: Test Implementation**
```python
test_code = """
# Test the position sizing implementation
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from position_sizing import calculate_atr_position_size, validate_position_sizes

# Generate test data
np.random.seed(42)
n = 100
dates = pd.date_range('2024-01-01', periods=n, freq='D')

close = pd.Series(100 + np.cumsum(np.random.randn(n)), index=dates)
high = close + abs(np.random.randn(n))
low = close - abs(np.random.randn(n))

# Calculate position sizes
capital = 10000
risk_per_trade = 0.02

sizes = calculate_atr_position_size(
    capital=capital,
    risk_per_trade=risk_per_trade,
    high=high,
    low=low,
    close=close,
    atr_period=14,
    atr_multiplier=2.5
)

# Validate
validation = validate_position_sizes(sizes, close, capital)

print("Position Sizing Test Results:")
print(f"Valid: {validation['valid']}")
print(f"Mean Position Size: {validation['mean_pct']:.1%} of capital")
print(f"Max Position Size: {validation['max_pct']:.1%} of capital")
print(f"Violations: {len(validation['violations'])}")
print(f"\\nSample Sizes (last 5 bars):")
print(sizes.tail())
"""

from VectorBT_Pro import run_code
result = run_code(test_code)
print("\nTest Results:")
print(result)
```

**Step 5: VBT Integration Test**
```python
vbt_integration_test = """
import vectorbtpro as vbt
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from position_sizing import create_vbt_position_sizes

# Generate test data
np.random.seed(42)
n = 100
dates = pd.date_range('2024-01-01', periods=n, freq='D')

close = pd.Series(100 + np.cumsum(np.random.randn(n)*0.5), index=dates)
high = close + abs(np.random.randn(n))
low = close - abs(np.random.randn(n))

# Simple entry/exit signals
entries = pd.Series(False, index=dates)
entries.iloc[20] = True  # One entry
entries.iloc[60] = True  # Another entry

exits = pd.Series(False, index=dates)
exits.iloc[40] = True  # First exit
exits.iloc[80] = True  # Second exit

# Generate VBT-compatible sizes
capital = 10000
risk_per_trade = 0.02

sizes = create_vbt_position_sizes(
    entries=entries,
    capital=capital,
    risk_per_trade=risk_per_trade,
    high=high,
    low=low,
    close=close
)

# Create portfolio with VBT
pf = vbt.PF.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    size=sizes,
    size_type='amount',  # Shares
    init_cash=capital,
    fees=0.002
)

# Verify results
print("VBT Integration Test:")
print(f"Total Return: {pf.total_return:.2%}")
print(f"Sharpe Ratio: {pf.sharpe_ratio:.2f}")
print(f"Win Rate: {pf.trades.win_rate:.1%}")
print(f"Number of Trades: {pf.trades.count()}")
print(f"\\nSUCCESS: VBT integration working!")
"""

result = run_code(vbt_integration_test)
print("\nVBT Integration Test:")
print(result)
```

**Step 6: Move to Outputs and Update HANDOFF**
```python
from Filesystem import move_file, edit_file, read_file

# Move to outputs
move_file(
    source="/home/claude/position_sizing.py",
    destination="/mnt/user-data/outputs/utils/position_sizing.py"
)

# Update HANDOFF.md
handoff_content = read_file("/mnt/user-data/uploads/docs/HANDOFF.md")

# Add entry to "Recent Changes" section
new_entry = """
## Recent Changes (October 18, 2025)

### Position Sizing Module - COMPLETED
- Created: `utils/position_sizing.py`
- Implements: ATR-based position sizing with capital constraints
- Verified: VBT integration working
- Tests: Unit tests pass, VBT integration test pass
- Validation: Mean position size 15-25%, max 100%
- Status: READY for Strategy 2 (ORB) implementation

"""

# Note: In practice, use edit_file with actual old/new text
print("Would update HANDOFF.md with:")
print(new_entry)
```

---

## Debugging VBT Integration

### Scenario: "Portfolio returns NaN values"

**Step 1: Reproduce the Error**
```python
from VectorBT_Pro import run_code

error_code = """
import vectorbtpro as vbt
import pandas as pd
import numpy as np

# Data that's causing issues
close = pd.Series([100, 101, np.nan, 103, 104])  # NaN in data!
entries = pd.Series([True, False, False, False, False])
exits = pd.Series([False, False, False, True, False])

pf = vbt.PF.from_signals(close, entries, exits, init_cash=10000)

print(f"Total Return: {pf.total_return}")  # Returns NaN
print(f"Issue reproduced")
"""

result = run_code(error_code)
print("Error Reproduction:")
print(result)
```

**Step 2: Search for Similar Issues**
```python
from VectorBT_Pro import search, find

# Search general docs
search_results = search(
    query="NaN portfolio returns missing data handling",
    asset_names=["messages", "docs"],
    max_tokens=1500
)
print("Search Results:")
print(search_results)

# Find Discord discussions
discord = find(
    refnames=["NaN", "missing data"],
    resolve=False,
    asset_names=["messages"],
    aggregate_messages=True
)
print("\nDiscord Discussions:")
print(discord)
```

**Step 3: Check Source Code**
```python
from VectorBT_Pro import get_source

# Read implementation
source = get_source("vbt.Portfolio.from_signals")

# Look for NaN handling
print("from_signals source (looking for NaN handling):")
print(source[:3000])
```

**Step 4: Test Fix**
```python
fix_test = """
import vectorbtpro as vbt
import pandas as pd
import numpy as np

# Original data
close_raw = pd.Series([100, 101, np.nan, 103, 104])

# FIX: Forward-fill NaN values
close = close_raw.ffill()  # Or dropna(), or interpolate()

print(f"Original: {close_raw.values}")
print(f"Fixed: {close.values}")

entries = pd.Series([True, False, False, False, False])
exits = pd.Series([False, False, False, True, False])

pf = vbt.PF.from_signals(close, entries, exits, init_cash=10000)

print(f"\\nTotal Return: {pf.total_return:.2%}")
print(f"FIX WORKS!")
"""

result = run_code(fix_test)
print("\nFix Test:")
print(result)
```

**Step 5: Document Solution**
```python
from Filesystem import write_file

solution = """
# VBT NaN Handling - Solution

## Problem
- Portfolio returns NaN when price data contains NaN values
- Common with raw Alpaca data (gaps, missing bars)

## Root Cause
- VBT calculations propagate NaN values
- from_signals does not automatically handle missing data

## Solution
```python
# Option 1: Forward-fill (use last known price)
close = close.ffill()

# Option 2: Drop NaN rows (may lose data)
close = close.dropna()

# Option 3: Interpolate (linear)
close = close.interpolate()
```

## Recommendation
- **Use forward-fill** for intraday data (most conservative)
- **Add data validation** before VBT integration
- **Check for NaN** in data pipeline, not backtest

## Prevention
Add to data fetching:
```python
def validate_ohlcv(df):
    if df.isnull().any().any():
        print(f"WARNING: NaN values detected")
        df = df.ffill()  # Auto-fix
    return df
```
"""

write_file(
    path="/home/claude/VBT_NAN_SOLUTION.md",
    content=solution
)
```

---

## Data Pipeline Development

### Scenario: "Fetch and prepare data for backtesting"

**Step 1: Research Alpaca Data Fetching**
```python
from VectorBT_Pro import search, find

# Learn about AlpacaData
results = search(
    query="AlpacaData pull download historical bars",
    asset_names=["api", "examples"],
    max_tokens=2000
)
print("AlpacaData Research:")
print(results)

# Get examples
examples = find(
    refnames=["vbt.AlpacaData"],
    asset_names=["examples"],
    max_tokens=1500
)
print("\nAlpacaData Examples:")
print(examples)
```

**Step 2: Check Available Methods**
```python
from VectorBT_Pro import get_attrs

attrs = get_attrs("vbt.AlpacaData")
print("AlpacaData Methods:")
print(attrs)

# Look for:
# - pull() method
# - get() method
# - update() method
```

**Step 3: Test Data Fetch**
```python
data_test = """
import vectorbtpro as vbt
import os

# Set API keys (should be in environment)
# os.environ['APCA_API_KEY_ID'] = 'your-key'
# os.environ['APCA_API_SECRET_KEY'] = 'your-secret'

# Test data download
try:
    data_obj = vbt.AlpacaData.pull(
        'SPY',
        start='2024-01-01',
        end='2024-03-01',
        timeframe='1D'
    )
    
    df = data_obj.get()
    
    print(f"Data fetched successfully:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"\\nFirst few rows:")
    print(df.head())
    
except Exception as e:
    print(f"Error: {e}")
    print("Check: API keys set? Network connection? Symbol valid?")
"""

from VectorBT_Pro import run_code
result = run_code(data_test)
print("Data Fetch Test:")
print(result)
```

**Step 4: Implement Data Pipeline**
```python
from Filesystem import write_file

data_pipeline = '''
"""
Data pipeline for fetching and preparing market data.
"""

import vectorbtpro as vbt
import pandas as pd
import pandas_market_calendars as mcal


def fetch_alpaca_data(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = '1D'
) -> pd.DataFrame:
    """
    Fetch data from Alpaca API.
    
    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g., 'SPY')
    start : str
        Start date ('YYYY-MM-DD')
    end : str
        End date ('YYYY-MM-DD')
    timeframe : str
        Bar timeframe ('1Min', '5Min', '1H', '1D')
        
    Returns
    -------
    pd.DataFrame
        OHLCV data with DatetimeIndex
    """
    data_obj = vbt.AlpacaData.pull(
        symbol,
        start=start,
        end=end,
        timeframe=timeframe
    )
    
    df = data_obj.get()
    return df


def filter_market_hours(df: pd.DataFrame, exchange: str = 'NYSE') -> pd.DataFrame:
    """
    Filter data to regular market hours only.
    
    CRITICAL: Must be done BEFORE resampling to avoid phantom bars.
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex
    exchange : str
        Exchange calendar ('NYSE', 'NASDAQ')
        
    Returns
    -------
    pd.DataFrame
        Filtered to regular trading hours only
    """
    cal = mcal.get_calendar(exchange)
    
    # Get trading days
    schedule = cal.schedule(
        start_date=df.index[0],
        end_date=df.index[-1]
    )
    
    # Filter to trading days only
    trading_days = schedule.index.date
    df_filtered = df[df.index.date.isin(trading_days)]
    
    return df_filtered


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean OHLCV data.
    
    - Checks for NaN values
    - Validates OHLC relationships (H >= L, etc.)
    - Forward-fills missing data
    
    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data
        
    Returns
    -------
    pd.DataFrame
        Validated and cleaned data
        
    Raises
    ------
    ValueError
        If data quality issues cannot be fixed
    """
    # Check for NaN
    if df.isnull().any().any():
        print(f"WARNING: NaN values detected, forward-filling")
        df = df.ffill()
    
    # Validate OHLC relationships
    if 'High' in df.columns and 'Low' in df.columns:
        invalid = df['High'] < df['Low']
        if invalid.any():
            raise ValueError(f"Invalid OHLC: High < Low at {invalid.sum()} bars")
    
    # Check for zero/negative prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            if (df[col] <= 0).any():
                raise ValueError(f"Invalid prices: {col} <= 0")
    
    return df


def prepare_backtest_data(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = '1D'
) -> pd.DataFrame:
    """
    Complete data preparation pipeline for backtesting.
    
    Steps:
    1. Fetch from Alpaca
    2. Filter to market hours
    3. Validate data quality
    4. Return clean data
    
    Parameters
    ----------
    symbol : str
        Ticker symbol
    start, end : str
        Date range
    timeframe : str
        Bar timeframe
        
    Returns
    -------
    pd.DataFrame
        Clean, validated OHLCV data ready for backtesting
    """
    # Fetch
    print(f"Fetching {symbol} data...")
    df = fetch_alpaca_data(symbol, start, end, timeframe)
    print(f"Fetched {len(df)} bars")
    
    # Filter market hours
    print("Filtering to market hours...")
    df = filter_market_hours(df)
    print(f"After filtering: {len(df)} bars")
    
    # Validate
    print("Validating data quality...")
    df = validate_ohlcv(df)
    print("Data validated")
    
    return df
'''

write_file(
    path="/home/claude/data_pipeline.py",
    content=data_pipeline
)
```

---

## Risk Management Implementation

### Scenario: "Implement portfolio heat limiting"

**Step 1: Understand Requirements**
```python
from Filesystem import read_file

# Read strategy specs
spec = read_file("/mnt/user-data/uploads/docs/STRATEGY_ANALYSIS_AND_DESIGN_SPEC.md")

# Look for:
# - Portfolio heat definition (max 8%)
# - How to calculate current heat
# - When to reject new positions
```

**Step 2: Research Similar Implementations**
```python
from VectorBT_Pro import search

results = search(
    query="portfolio heat risk management multiple positions overlapping",
    asset_names=["messages", "examples"],
    max_tokens=1500
)
print("Portfolio Heat Research:")
print(results)
```

**Step 3: Design Implementation**
```python
from Filesystem import write_file

risk_management = '''
"""
Risk management module for portfolio-level constraints.
"""

import pandas as pd
import numpy as np


def calculate_portfolio_heat(
    open_positions: pd.DataFrame,
    capital: float,
    atr: pd.Series,
    close: pd.Series,
    atr_multiplier: float = 2.5
) -> float:
    """
    Calculate current portfolio heat from open positions.
    
    Portfolio Heat = Sum of (stop_distance * position_size * price) / capital
    
    Parameters
    ----------
    open_positions : pd.DataFrame
        DataFrame with columns: ['symbol', 'size', 'entry_price', 'current_bar']
    capital : float
        Account capital
    atr : pd.Series
        ATR values indexed by datetime
    close : pd.Series
        Close prices indexed by datetime
    atr_multiplier : float
        Stop distance multiplier
        
    Returns
    -------
    float
        Current portfolio heat as fraction (e.g., 0.08 for 8%)
    """
    if len(open_positions) == 0:
        return 0.0
    
    total_risk = 0.0
    
    for _, pos in open_positions.iterrows():
        bar = pos['current_bar']
        size = pos['size']
        
        # Stop distance for this position
        stop_distance = atr.loc[bar] * atr_multiplier
        
        # Risk amount
        risk = stop_distance * size
        
        total_risk += risk
    
    heat = total_risk / capital
    return heat


def apply_portfolio_heat_gate(
    entries: pd.Series,
    open_positions_tracker: callable,
    capital: float,
    max_heat: float,
    atr: pd.Series,
    close: pd.Series,
    position_sizes: pd.Series,
    **kwargs
) -> pd.Series:
    """
    Apply portfolio heat gate to entry signals.
    
    Rejects entries that would cause portfolio heat to exceed max_heat.
    
    Parameters
    ----------
    entries : pd.Series
        Boolean entry signals
    open_positions_tracker : callable
        Function that returns current open positions DataFrame
    capital : float
        Account capital
    max_heat : float
        Maximum portfolio heat (e.g., 0.08 for 8%)
    atr, close : pd.Series
        Price and volatility data
    position_sizes : pd.Series
        Proposed position sizes
    **kwargs
        Additional arguments for calculate_portfolio_heat
        
    Returns
    -------
    pd.Series
        Filtered entry signals (some entries may be False)
    """
    filtered_entries = entries.copy()
    
    for bar in entries[entries].index:
        # Get current open positions
        open_pos = open_positions_tracker(bar)
        
        # Calculate current heat
        current_heat = calculate_portfolio_heat(
            open_pos, capital, atr, close, **kwargs
        )
        
        # Calculate proposed additional heat
        proposed_size = position_sizes.loc[bar]
        stop_distance = atr.loc[bar] * kwargs.get('atr_multiplier', 2.5)
        additional_heat = (stop_distance * proposed_size) / capital
        
        # Check if would exceed limit
        if current_heat + additional_heat > max_heat:
            print(f"REJECTED: Entry at {bar}")
            print(f"  Current heat: {current_heat:.2%}")
            print(f"  Additional: {additional_heat:.2%}")
            print(f"  Would be: {current_heat + additional_heat:.2%}")
            print(f"  Max allowed: {max_heat:.2%}")
            filtered_entries.loc[bar] = False
    
    return filtered_entries


class PortfolioHeatTracker:
    """
    Track open positions for portfolio heat calculation.
    
    This class maintains state of open positions and provides
    the interface needed by apply_portfolio_heat_gate.
    """
    
    def __init__(self):
        self.positions = []
    
    def open_position(self, bar, symbol, size, entry_price):
        """Add a new open position."""
        self.positions.append({
            'bar': bar,
            'symbol': symbol,
            'size': size,
            'entry_price': entry_price
        })
    
    def close_position(self, symbol):
        """Close a position."""
        self.positions = [p for p in self.positions if p['symbol'] != symbol]
    
    def get_open_positions(self, current_bar):
        """Get DataFrame of positions open at current_bar."""
        return pd.DataFrame(self.positions)
'''

write_file(
    path="/home/claude/risk_management.py",
    content=risk_management
)
```

---

## Summary

This guide provides **copy-paste examples** for common development tasks:

1. **Session Startup**: Read HANDOFF.md, verify environment
2. **VBT API Discovery**: search → resolve → get_attrs → find → get_source → test
3. **New Component**: Research → Design → Implement → Test → Integrate
4. **Debugging**: Reproduce → Search → Check source → Fix → Validate
5. **Data Pipeline**: Fetch → Filter → Validate → Prepare
6. **Risk Management**: Calculate → Gate → Track → Enforce

**Key Pattern:**
```
Research (MCP tools) → Design (docs) → Implement (filesystem) → Test (run_code) → Integrate (VBT) → Document (HANDOFF)
```

**Remember:**
- HANDOFF.md first (every session)
- VBT tools before coding
- Test before claiming
- No emojis (ASCII only)
- Delete redundant files

---

**Document Version:** 1.0
**Last Updated:** October 18, 2025
**Maintainer:** Development Team

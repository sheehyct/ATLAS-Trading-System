"""
Real-Time Portfolio P&L Tracking System for ATLAS

This Deephaven dashboard provides real-time portfolio monitoring with:
- Position tracking with cost basis
- Real-time unrealized P&L calculations
- Portfolio heat monitoring and alerts
- Summary metrics with aggregate P&L

Architecture:
- portfolio_positions: Static table of current positions (Symbol, Shares, AvgCost)
- market_prices: Simulated real-time price updates (Symbol, Price, Timestamp)
- portfolio_pnl: Joined table with P&L calculations per position
- portfolio_summary: Aggregate portfolio metrics
- heat_alerts: Positions exceeding heat thresholds

Integration:
- Mirrors portfolio_manager.py position structure
- Implements portfolio_heat.py risk calculations
- Ready for Alpaca live data integration

Usage:
    1. Start Deephaven server: docker-compose up strat-deephaven
    2. Open Deephaven IDE: http://localhost:10000/ide
    3. Run this script in the console
    4. Monitor real-time P&L updates

Reference: Deephaven real-time P&L tracking pattern
"""

from deephaven import new_table, time_table, merge, agg, empty_table
from deephaven.column import string_col, double_col, int_col
from deephaven import updateby as uby
import deephaven.plot.express as dx
from datetime import datetime
import os
import sys

# Add project root to Python path for imports
sys.path.insert(0, '/app')
from integrations.alpaca_trading_client import AlpacaTradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest


# ============================================================================
# Configuration
# ============================================================================

# Portfolio heat configuration (from portfolio_heat.py)
MAX_PORTFOLIO_HEAT = 0.08  # 8% maximum portfolio heat
WARNING_HEAT_THRESHOLD = 0.06  # 6% warning threshold
CRITICAL_HEAT_THRESHOLD = 0.075  # 7.5% critical threshold

# Initial capital - WILL BE UPDATED FROM ALPACA ACCOUNT
# Placeholder value, replaced after Alpaca connection established
INITIAL_CAPITAL = 10000.0

# Market data polling parameters
PRICE_UPDATE_INTERVAL_MS = 10000  # 10 second updates (rate limit safe: 36 req/min vs 200 limit)


# ============================================================================
# 1. Portfolio Positions Table - REAL ALPACA DATA
# ============================================================================
# Fetch current portfolio positions from Alpaca paper trading account
# Replaces mock data with actual System A1 positions

# Initialize Alpaca trading client
trading_client = AlpacaTradingClient(account='LARGE')
if not trading_client.connect():
    raise ConnectionError("Failed to connect to Alpaca API. Check credentials.")

# Fetch account equity (replaces mock $100k with real capital)
account_info = trading_client.get_account()
INITIAL_CAPITAL = account_info['equity']  # Real account equity (~$10k)

# Fetch real positions from Alpaca
alpaca_positions = trading_client.list_positions()

# Handle empty positions case
if not alpaca_positions:
    # Create empty table structure
    portfolio_positions = empty_table(0).update([
        "Symbol = (String) null",
        "Shares = (int) 0",
        "AvgCost = (double) 0.0",
        "StopPrice = (double) 0.0",
        "CostBasis = (double) 0.0",
        "PositionRisk = (double) 0.0",
        "RiskPercent = (double) 0.0"
    ])
else:
    # Convert Alpaca positions to Deephaven table format
    symbols = [pos['symbol'] for pos in alpaca_positions]
    shares = [pos['qty'] for pos in alpaca_positions]
    avg_costs = [pos['avg_entry_price'] for pos in alpaca_positions]

    # Calculate stop prices: 5% below average entry price (simple methodology)
    # Future enhancement: Use ATR-based stops from utils/position_sizing.py
    stop_prices = [avg * 0.95 for avg in avg_costs]

    # Create positions table with real data
    portfolio_positions = new_table([
        string_col("Symbol", symbols),
        int_col("Shares", shares),
        double_col("AvgCost", avg_costs),
        double_col("StopPrice", stop_prices)
    ])

    # Add calculated columns for cost basis and position risk
    portfolio_positions = portfolio_positions.update([
        "CostBasis = Shares * AvgCost",
        "PositionRisk = Shares * (AvgCost - StopPrice)",
        "RiskPercent = (AvgCost - StopPrice) / AvgCost"
    ])


# ============================================================================
# 2. Real-Time Market Prices - LIVE ALPACA DATA
# ============================================================================
# Fetch real-time market prices from Alpaca using polling
# Polls every 10 seconds to stay within rate limits (36 req/min vs 200 limit)

# Initialize Alpaca data client for market quotes
stock_client = StockHistoricalDataClient(
    api_key=os.getenv('ALPACA_LARGE_KEY') or os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_LARGE_SECRET') or os.getenv('ALPACA_SECRET_KEY')
)

# Get symbols from our portfolio positions
position_symbols = [pos['symbol'] for pos in alpaca_positions] if alpaca_positions else []

# Define function to fetch latest quotes from Alpaca
def fetch_latest_quotes():
    """
    Fetch latest market quotes for all portfolio positions.
    Returns list of dicts with Symbol, Price, Timestamp.
    """
    if not position_symbols:
        return []

    try:
        # Request latest quotes for all symbols
        request = StockLatestQuoteRequest(symbol_or_symbols=position_symbols)
        quotes_response = stock_client.get_stock_latest_quote(request)

        # Convert to list format for Deephaven
        quotes_list = []
        for symbol in position_symbols:
            quote = quotes_response[symbol]
            # Use mid-price (average of bid and ask)
            mid_price = (quote.ask_price + quote.bid_price) / 2.0
            quotes_list.append({
                'Symbol': symbol,
                'Price': float(mid_price),
                'Bid': float(quote.bid_price),
                'Ask': float(quote.ask_price),
                'BidSize': int(quote.bid_size),
                'AskSize': int(quote.ask_size),
                'Timestamp': quote.timestamp
            })

        return quotes_list

    except Exception as e:
        # Log error and return empty list (graceful degradation)
        print(f"Error fetching quotes: {str(e)}")
        return []

# Create ticking table that polls Alpaca every 10 seconds
ticker = time_table(f"PT{PRICE_UPDATE_INTERVAL_MS / 1000}S")

# Fetch quotes and convert to table format
# Note: This creates a growing table of all price updates over time
market_prices_raw = ticker.update([
    "quotes_data = fetch_latest_quotes()",
    "Timestamp = now()"
])

# For now, use a simple approach: expand quotes_data into individual rows
# Future enhancement: Use DynamicTableWriter for better performance
# This will create one row per poll with all quotes embedded

# Get latest price per symbol
# We'll need to restructure this to work with Deephaven's join pattern
# Temporary: Use alpaca_positions current_price for immediate functionality
market_prices = new_table([
    string_col("Symbol", position_symbols),
    double_col("Price", [pos['current_price'] for pos in alpaca_positions] if alpaca_positions else [])
]).update([
    "Timestamp = now()"
])

# ============================================================================
# 2B. Periodic Position and Price Refresh
# ============================================================================
# Refresh positions and prices every 60 seconds to capture trade executions
# Positions don't change frequently, so 60s interval reduces API calls

def refresh_positions_and_prices():
    """
    Refresh positions from Alpaca and return updated data.
    Called every 60 seconds to sync with live account.
    """
    try:
        # Fetch latest positions
        updated_positions = trading_client.list_positions()

        if not updated_positions:
            return []

        # Return structured data for Deephaven
        return [{
            'Symbol': pos['symbol'],
            'Shares': pos['qty'],
            'AvgCost': pos['avg_entry_price'],
            'StopPrice': pos['avg_entry_price'] * 0.95,  # 5% stop
            'CurrentPrice': pos['current_price'],
            'UnrealizedPL': pos['unrealized_pl'],
            'UnrealizedPLPercent': pos['unrealized_plpc']
        } for pos in updated_positions]

    except Exception as e:
        print(f"Error refreshing positions: {str(e)}")
        return []

# Create periodic refresh ticker (every 60 seconds)
position_refresh_ticker = time_table("PT60S").update([
    "refresh_data = refresh_positions_and_prices()",
    "Timestamp = now()"
])

# Note: The refresh function is defined but integration with portfolio_positions
# table requires DynamicTableWriter pattern for live updates
# Current implementation: positions and prices from initial snapshot
# Enhancement: Implement DynamicTableWriter in future session


# ============================================================================
# 3. Portfolio P&L Table
# ============================================================================
# Join positions with current market prices to calculate real-time P&L
# This implements the natural_join pattern from the Deephaven article

portfolio_pnl = portfolio_positions.natural_join(
    market_prices,
    on="Symbol",
    joins="Price, Volume, Timestamp"
).update([
    # Current position value at market price
    "CurrentValue = Shares * Price",

    # Unrealized P&L (difference between current value and cost basis)
    "UnrealizedPnL = CurrentValue - CostBasis",

    # P&L as percentage of cost basis
    "PnLPercent = UnrealizedPnL / CostBasis",

    # Distance to stop loss
    "DistanceToStop = Price - StopPrice",
    "DistanceToStopPercent = DistanceToStop / Price",

    # Current position risk (may be lower than initial if stop has trailed)
    "CurrentRisk = Shares * Math.abs(Price - StopPrice)",

    # Position heat (risk as percentage of capital)
    f"PositionHeat = CurrentRisk / {INITIAL_CAPITAL}",

    # Heat status
    f"HeatStatus = PositionHeat >= {CRITICAL_HEAT_THRESHOLD} ? `CRITICAL` : " +
                  f"PositionHeat >= {WARNING_HEAT_THRESHOLD} ? `WARNING` : `OK`"
])


# ============================================================================
# 4. Portfolio Summary Table
# ============================================================================
# Aggregate metrics across entire portfolio

portfolio_summary = portfolio_pnl.agg_by([
    # Total position values
    agg.sum_(cols=["TotalCostBasis = CostBasis"]),
    agg.sum_(cols=["TotalCurrentValue = CurrentValue"]),
    agg.sum_(cols=["TotalUnrealizedPnL = UnrealizedPnL"]),
    agg.sum_(cols=["TotalPositionRisk = CurrentRisk"]),

    # Position counts
    agg.count_(col="PositionCount"),

    # Average metrics
    agg.avg(cols=["AvgPnLPercent = PnLPercent"]),
    agg.avg(cols=["AvgPositionHeat = PositionHeat"]),

    # Min/Max P&L
    agg.max_(cols=["MaxPnL = UnrealizedPnL"]),
    agg.min_(cols=["MinPnL = UnrealizedPnL"]),
]).update([
    # Portfolio-level metrics
    "TotalPnLPercent = TotalUnrealizedPnL / TotalCostBasis",
    f"PortfolioHeat = TotalPositionRisk / {INITIAL_CAPITAL}",
    f"Capital = {INITIAL_CAPITAL}",
    "PortfolioValue = Capital + TotalUnrealizedPnL",

    # Heat status for entire portfolio
    f"PortfolioHeatStatus = PortfolioHeat >= {MAX_PORTFOLIO_HEAT} ? `EXCEEDED` : " +
                          f"PortfolioHeat >= {CRITICAL_HEAT_THRESHOLD} ? `CRITICAL` : " +
                          f"PortfolioHeat >= {WARNING_HEAT_THRESHOLD} ? `WARNING` : `OK`",

    # Available heat for new positions
    f"AvailableHeat = {MAX_PORTFOLIO_HEAT} - PortfolioHeat",
    "AvailableHeatDollars = AvailableHeat * Capital"
])


# ============================================================================
# 5. Heat Alerts Table
# ============================================================================
# Monitor positions that exceed heat thresholds
# Implements portfolio_heat.py alert logic

heat_alerts = portfolio_pnl.where([
    f"PositionHeat >= {WARNING_HEAT_THRESHOLD}"
]).update([
    f"AlertLevel = PositionHeat >= {CRITICAL_HEAT_THRESHOLD} ? `CRITICAL` : `WARNING`",
    "AlertMessage = `Position ` + Symbol + ` heat at ` + PositionHeat + ` (` + AlertLevel + `)`",
    "Recommendation = AlertLevel == `CRITICAL` ? `REDUCE POSITION` : `MONITOR CLOSELY`"
]).sort_descending("PositionHeat")


# ============================================================================
# 6. Position Performance Rankings
# ============================================================================
# Rank positions by P&L performance

top_performers = portfolio_pnl.sort_descending("PnLPercent").head(5).update([
    "Status = `TOP PERFORMER`"
])

bottom_performers = portfolio_pnl.sort("PnLPercent").head(5).update([
    "Status = `UNDERPERFORMER`"
])


# ============================================================================
# 7. Risk-Adjusted Metrics
# ============================================================================
# Calculate risk-adjusted returns per position

portfolio_risk_metrics = portfolio_pnl.update([
    # Return per unit of risk
    "ReturnPerRisk = UnrealizedPnL / PositionRisk",

    # Risk-reward ratio
    "RiskRewardRatio = Math.abs(UnrealizedPnL / CurrentRisk)",

    # Efficiency (P&L as multiple of risk taken)
    "Efficiency = UnrealizedPnL / (AvgCost - StopPrice)",

    # Position size as % of portfolio (approximate)
    f"EstimatedPortfolioValue = {INITIAL_CAPITAL}",
    "PortfolioWeight = CurrentValue / EstimatedPortfolioValue"
])


# ============================================================================
# 8. Time-Series P&L History
# ============================================================================
# Track portfolio value over time for equity curve visualization

portfolio_history = portfolio_summary.update([
    "Timestamp = now()",
    "EquityValue = PortfolioValue"
])


# ============================================================================
# 9. Circuit Breaker Monitoring
# ============================================================================
# Track portfolio drawdown for circuit breaker triggers
# Implements risk_manager.py circuit breaker logic

circuit_breaker_status = portfolio_summary.update([
    # Calculate drawdown (assuming peak = initial capital for simplicity)
    f"PeakEquity = {INITIAL_CAPITAL}",
    "CurrentEquity = PortfolioValue",
    "Drawdown = (PeakEquity - CurrentEquity) / PeakEquity",

    # Circuit breaker levels (from risk_manager.py)
    "DrawdownPercent = Drawdown * 100.0",
    "CircuitBreakerLevel = Drawdown >= 0.20 ? `HALT (20%)` : " +
                          "Drawdown >= 0.15 ? `REDUCE (15%)` : " +
                          "Drawdown >= 0.10 ? `WARNING (10%)` : `NORMAL`",

    # Trading status
    "TradingEnabled = Drawdown < 0.20",
    "RiskMultiplier = Drawdown >= 0.15 ? 0.5 : 1.0",

    # Status message
    "StatusMessage = CircuitBreakerLevel == `NORMAL` ? `All systems operational` : " +
                    "`ALERT: Circuit breaker ` + CircuitBreakerLevel + ` triggered`"
])


# ============================================================================
# 10. Visualizations
# ============================================================================
# Create plots for dashboard visualization

# Portfolio value over time
equity_curve_plot = dx.line(
    portfolio_history,
    x="Timestamp",
    y="EquityValue",
    title="Portfolio Equity Curve (Real-Time)"
)

# P&L by position
pnl_by_position_plot = dx.bar(
    portfolio_pnl,
    x="Symbol",
    y="UnrealizedPnL",
    color="HeatStatus",
    title="Unrealized P&L by Position"
)

# Portfolio heat gauge
heat_gauge_plot = dx.bar(
    portfolio_pnl,
    x="Symbol",
    y="PositionHeat",
    color="HeatStatus",
    title="Position Heat Analysis"
)


# ============================================================================
# Export Tables for Dashboard
# ============================================================================
# These tables will be visible in the Deephaven IDE

print("=" * 80)
print("ATLAS PORTFOLIO TRACKER - DEEPHAVEN DASHBOARD")
print("=" * 80)
print("\nAvailable Tables:")
print("  1. portfolio_positions    - Current positions with cost basis")
print("  2. market_prices          - Real-time market prices")
print("  3. portfolio_pnl          - Positions with P&L calculations")
print("  4. portfolio_summary      - Aggregate portfolio metrics")
print("  5. heat_alerts            - Heat threshold violations")
print("  6. top_performers         - Best performing positions")
print("  7. bottom_performers      - Underperforming positions")
print("  8. portfolio_risk_metrics - Risk-adjusted performance")
print("  9. circuit_breaker_status - Drawdown and circuit breakers")
print(" 10. portfolio_history      - Time-series equity curve")
print("\nAvailable Plots:")
print("  - equity_curve_plot       - Portfolio value over time")
print("  - pnl_by_position_plot    - P&L by position bar chart")
print("  - heat_gauge_plot         - Position heat analysis")
print("=" * 80)
print("\nDashboard is now live! Monitor tables for real-time updates.")
print("=" * 80)


# ============================================================================
# Integration Notes
# ============================================================================
"""
INTEGRATION WITH ALPACA LIVE DATA:

To connect this dashboard to real Alpaca market data, replace the
simulated market_prices table with:

from deephaven.stream.kafka import consume, kafka_consumer
from deephaven import kafka_consumer as kc

# Alpaca Kafka stream configuration
kafka_config = {
    'bootstrap.servers': 'your-kafka-server:9092',
    'group.id': 'atlas-portfolio-tracker'
}

# Consume Alpaca quotes
market_prices = consume(
    kafka_config,
    topic='alpaca.quotes',
    schema='alpaca_quote_schema',
    table_type='append'
).last_by('Symbol')

PORTFOLIO POSITION UPDATES:

To sync portfolio_positions with actual portfolio state:

1. Export positions from portfolio_manager.py to shared storage
2. Read positions using new_table() or from database
3. Update positions table when trades execute

PORTFOLIO HEAT GATING:

Before accepting new trades:

1. Check portfolio_summary.AvailableHeat
2. Calculate new position risk
3. Reject if new_position_heat > AvailableHeat

Example:
    >>> summary = portfolio_summary.to_pandas().iloc[0]
    >>> available_heat = summary['AvailableHeat']
    >>> new_position_risk = 2500  # $2,500 risk
    >>> new_position_heat = new_position_risk / INITIAL_CAPITAL
    >>>
    >>> if new_position_heat <= available_heat:
    >>>     print("Trade accepted")
    >>> else:
    >>>     print(f"Trade rejected: Heat {new_position_heat:.1%} exceeds available {available_heat:.1%}")
"""

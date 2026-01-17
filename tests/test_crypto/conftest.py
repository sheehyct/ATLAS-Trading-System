"""
Shared fixtures for crypto module tests.

Session EQUITY-68: Provides mock objects for CoinbaseClient, PaperTrader,
SimulatedTrade, and CryptoSignalContext.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from crypto.simulation.paper_trader import SimulatedTrade, PaperTrader


@pytest.fixture
def mock_trade():
    """
    Create a SimulatedTrade with EQUITY-67 pattern invalidation fields.

    Default values represent a BUY position with Type 2U entry.
    """
    return SimulatedTrade(
        trade_id="TEST-001",
        symbol="BTC-USD",
        side="BUY",
        quantity=0.01,
        entry_price=50000.0,
        entry_time=datetime.utcnow(),
        stop_price=49000.0,
        target_price=52000.0,
        timeframe="1H",
        pattern_type="2-1-2U",
        entry_bar_type="2U",
        entry_bar_high=50500.0,  # Setup bar bounds
        entry_bar_low=49500.0,
        intrabar_high=50000.0,   # Initialized to entry price
        intrabar_low=50000.0,
    )


@pytest.fixture
def mock_trade_2d():
    """Create a SimulatedTrade with Type 2D entry."""
    return SimulatedTrade(
        trade_id="TEST-002",
        symbol="BTC-USD",
        side="SELL",
        quantity=0.01,
        entry_price=50000.0,
        entry_time=datetime.utcnow(),
        stop_price=51000.0,
        target_price=48000.0,
        timeframe="1H",
        pattern_type="3-2D",
        entry_bar_type="2D",
        entry_bar_high=50500.0,
        entry_bar_low=49500.0,
        intrabar_high=50000.0,
        intrabar_low=50000.0,
    )


@pytest.fixture
def mock_client():
    """
    Create a mock CoinbaseClient for price fetching.

    Default returns $50,000 for BTC-USD.
    """
    from crypto.exchange.coinbase_client import CoinbaseClient

    client = Mock(spec=CoinbaseClient)
    client.get_current_price.return_value = 50000.0
    return client


@pytest.fixture
def mock_paper_trader():
    """
    Create a mock PaperTrader with empty open trades.
    """
    trader = Mock(spec=PaperTrader)
    trader.account = Mock()
    trader.account.open_trades = []
    trader.account.closed_trades = []
    trader.close_trade = Mock(return_value=None)
    return trader


@pytest.fixture
def mock_paper_trader_with_trade(mock_trade):
    """Create a mock PaperTrader with one open trade."""
    trader = Mock(spec=PaperTrader)
    trader.account = Mock()
    trader.account.open_trades = [mock_trade]
    trader.account.closed_trades = []

    def close_trade_side_effect(trade_id, exit_price):
        """Simulate closing a trade."""
        for trade in trader.account.open_trades:
            if trade.trade_id == trade_id:
                trade.close(exit_price)
                trader.account.open_trades.remove(trade)
                trader.account.closed_trades.append(trade)
                return trade
        return None

    trader.close_trade = Mock(side_effect=close_trade_side_effect)
    return trader


@pytest.fixture
def position_monitor(mock_client, mock_paper_trader):
    """
    Create a CryptoPositionMonitor with mocked dependencies.
    """
    from crypto.simulation.position_monitor import CryptoPositionMonitor

    return CryptoPositionMonitor(
        client=mock_client,
        paper_trader=mock_paper_trader,
    )


@pytest.fixture
def position_monitor_with_trade(mock_client, mock_paper_trader_with_trade):
    """
    Create a CryptoPositionMonitor with one open trade.
    """
    from crypto.simulation.position_monitor import CryptoPositionMonitor

    return CryptoPositionMonitor(
        client=mock_client,
        paper_trader=mock_paper_trader_with_trade,
    )

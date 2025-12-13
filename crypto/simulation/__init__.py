"""
Paper trading simulation module.

Provides simulated trading functionality since Coinbase doesn't offer
native paper trading. Tracks mock positions, orders, and P&L.
"""

from crypto.simulation.paper_trader import PaperTrader, SimulatedTrade
from crypto.simulation.position_monitor import CryptoPositionMonitor, ExitSignal

__all__ = [
    "PaperTrader",
    "SimulatedTrade",
    "CryptoPositionMonitor",
    "ExitSignal",
]

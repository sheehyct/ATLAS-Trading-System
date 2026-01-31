"""
EQUITY-94: CryptoStatArbExecutor - Extracted from CryptoSignalDaemon

Handles all StatArb signal checking and trade execution:
- STRAT priority conflict resolution (skip symbols in active STRAT trades)
- StatArb signal generation from z-score crossovers
- Entry execution (open long + short legs)
- Exit execution (close both legs on mean reversion)

Extracted as part of Phase 6.4 coordinator extraction.
"""

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Protocol, Set

from crypto import config
from crypto.config import get_max_leverage_for_symbol
from crypto.trading.fees import calculate_round_trip_fee, calculate_num_contracts

logger = logging.getLogger(__name__)

# StatArb imports - graceful fallback
try:
    from crypto.statarb.signal_generator import (
        StatArbConfig,
        StatArbSignal,
        StatArbSignalGenerator,
        StatArbSignalType,
    )
    STATARB_AVAILABLE = True
except ImportError:
    StatArbConfig = None  # type: ignore
    StatArbSignal = None  # type: ignore
    StatArbSignalGenerator = None  # type: ignore
    StatArbSignalType = None  # type: ignore
    STATARB_AVAILABLE = False


class PriceClient(Protocol):
    """Protocol for price fetching."""
    def get_current_price(self, symbol: str) -> Optional[float]: ...


class PaperTraderProtocol(Protocol):
    """Protocol for paper trading operations."""
    def get_available_balance(self) -> float: ...
    def get_trades_by_strategy(self, strategy: str) -> list: ...
    def open_trade(self, **kwargs) -> Any: ...
    def close_trade(self, trade_id: str, exit_price: float) -> None: ...


class AlerterProtocol(Protocol):
    """Protocol for alert sending."""
    def _send_message(self, message: str) -> None: ...


class CryptoStatArbExecutor:
    """
    StatArb signal checking and trade execution for crypto daemon.

    STRAT has priority - StatArb will not trade symbols that are
    currently in active STRAT positions.
    """

    def __init__(
        self,
        client: PriceClient,
        statarb_pairs: List[tuple],
        statarb_config: Optional[Any] = None,
        paper_balance: float = 10000.0,
        paper_trader: Optional[PaperTraderProtocol] = None,
        get_current_time_et: Optional[Callable[[], datetime]] = None,
        on_execution: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize StatArb executor.

        Args:
            client: Price fetching client
            statarb_pairs: List of (symbol1, symbol2) pairs
            statarb_config: StatArbConfig instance
            paper_balance: Initial paper trading balance
            paper_trader: Paper trader for execution
            get_current_time_et: Callback for current ET time
            on_execution: Callback to increment execution count
            on_error: Callback to increment error count
        """
        self._client = client
        self._pairs = statarb_pairs
        self._paper_trader = paper_trader
        self._get_current_time_et = get_current_time_et
        self._on_execution = on_execution or (lambda: None)
        self._on_error = on_error or (lambda: None)
        self._strat_active_symbols: Set[str] = set()
        self._alerter: Optional[AlerterProtocol] = None
        self._alert_on_entry: bool = True
        self._alert_on_exit: bool = True

        # Initialize generator
        self.generator: Optional[StatArbSignalGenerator] = None
        if STATARB_AVAILABLE and statarb_pairs:
            self._init_generator(statarb_config, paper_balance, paper_trader)

    def _init_generator(
        self,
        statarb_config: Optional[Any],
        paper_balance: float,
        paper_trader: Optional[PaperTraderProtocol],
    ) -> None:
        """Initialize the StatArb signal generator."""
        try:
            if statarb_config is None:
                statarb_config = StatArbConfig()

            account_value = paper_balance
            if paper_trader:
                account_value = paper_trader.get_available_balance()

            self.generator = StatArbSignalGenerator(
                pairs=self._pairs,
                config=statarb_config,
                account_value=account_value,
            )

            pairs_str = ", ".join(f"{p[0]}/{p[1]}" for p in self._pairs)
            logger.info(f"StatArb signal generator initialized (pairs: {pairs_str})")

        except Exception as e:
            logger.error(f"Failed to initialize StatArb generator: {e}")
            self.generator = None

    def set_alerter(
        self,
        alerter: Optional[AlerterProtocol],
        alert_on_entry: bool = True,
        alert_on_exit: bool = True,
    ) -> None:
        """Set alerter for StatArb notifications."""
        self._alerter = alerter
        self._alert_on_entry = alert_on_entry
        self._alert_on_exit = alert_on_exit

    def update_strat_active_symbols(self) -> None:
        """Update set of symbols currently in STRAT trades."""
        if self._paper_trader is None:
            self._strat_active_symbols = set()
            return

        strat_trades = self._paper_trader.get_trades_by_strategy("strat")
        open_trades = [t for t in strat_trades if t.exit_time is None]
        self._strat_active_symbols = {t.symbol for t in open_trades}

    def check_and_execute(self) -> None:
        """
        Check for StatArb signals and execute if no STRAT conflict.

        STRAT has priority - StatArb will not trade symbols that are
        currently in active STRAT positions.
        """
        if self.generator is None or self._paper_trader is None:
            return

        # Update STRAT active symbols before checking
        self.update_strat_active_symbols()

        # Fetch current prices for statarb pairs
        prices: Dict[str, float] = {}
        for sym1, sym2 in self._pairs:
            try:
                p1 = self._client.get_current_price(sym1)
                p2 = self._client.get_current_price(sym2)
                if p1 and p1 > 0:
                    prices[sym1] = p1
                if p2 and p2 > 0:
                    prices[sym2] = p2
            except Exception as e:
                logger.warning(f"Failed to get prices for {sym1}/{sym2}: {e}")

        if not prices:
            return

        # Update account value for position sizing
        self.generator.update_account_value(
            self._paper_trader.get_available_balance()
        )

        # Check for signals
        try:
            signals = self.generator.check_for_signals(prices)
        except Exception as e:
            logger.error(f"StatArb signal check error: {e}")
            self._on_error()
            return

        for signal in signals:
            # STRAT priority: skip if any leg is in active STRAT trade
            if signal.long_symbol in self._strat_active_symbols:
                logger.info(
                    f"STATARB SKIPPED: {signal.long_symbol} in active STRAT trade"
                )
                continue
            if signal.short_symbol in self._strat_active_symbols:
                logger.info(
                    f"STATARB SKIPPED: {signal.short_symbol} in active STRAT trade"
                )
                continue

            self._execute(signal)

    def _execute(self, signal: "StatArbSignal") -> None:
        """Execute a StatArb signal (entry or exit)."""
        if self._paper_trader is None:
            return

        if signal.signal_type == StatArbSignalType.EXIT:
            self._execute_exit(signal)
        else:
            self._execute_entry(signal)

    def _execute_entry(self, signal: "StatArbSignal") -> None:
        """Execute StatArb entry (opens two positions)."""
        if self._paper_trader is None:
            return

        # Get current prices
        try:
            long_price = self._client.get_current_price(signal.long_symbol)
            short_price = self._client.get_current_price(signal.short_symbol)
            if not long_price or not short_price:
                logger.warning("StatArb entry skipped: could not get prices")
                return
        except Exception as e:
            logger.warning(f"StatArb entry skipped: {e}")
            return

        # Get current leverage tier
        if self._get_current_time_et:
            now_et = self._get_current_time_et()
            max_leverage = get_max_leverage_for_symbol(signal.long_symbol, now_et)
        else:
            max_leverage = 4.0

        # Calculate quantities from notional values
        long_qty = signal.long_notional / long_price if long_price > 0 else 0
        short_qty = signal.short_notional / short_price if short_price > 0 else 0

        if long_qty <= 0 or short_qty <= 0:
            logger.warning("StatArb entry skipped: invalid quantities")
            return

        # Fee profitability check (Session EQUITY-99)
        if config.FEE_PROFITABILITY_FILTER_ENABLED:
            try:
                # Calculate contracts and fees for each leg
                long_contracts = calculate_num_contracts(
                    signal.long_notional, long_price, signal.long_symbol
                )
                short_contracts = calculate_num_contracts(
                    signal.short_notional, short_price, signal.short_symbol
                )

                total_rt_fees = (
                    calculate_round_trip_fee(signal.long_notional, long_contracts)
                    + calculate_round_trip_fee(signal.short_notional, short_contracts)
                )

                # StatArb target is mean reversion - estimate 2% profit expectation
                total_notional = signal.long_notional + signal.short_notional
                expected_profit = total_notional * 0.02

                if expected_profit > 0:
                    fee_pct = total_rt_fees / expected_profit
                    if fee_pct > config.MAX_FEE_PCT_OF_TARGET:
                        logger.info(
                            f"STATARB SKIPPED (FEES): Fees ${total_rt_fees:.2f} = "
                            f"{fee_pct:.1%} of expected profit "
                            f"(max: {config.MAX_FEE_PCT_OF_TARGET:.1%})"
                        )
                        return
                    logger.debug(
                        f"StatArb fee check passed: fees {fee_pct:.1%} of expected profit"
                    )
            except Exception as e:
                logger.warning(f"StatArb fee check failed: {e}")
                # Continue with entry on fee check failure

        signal_type_str = signal.signal_type.value

        logger.info(
            f"STATARB ENTRY: {signal_type_str} Z={signal.zscore:.2f} | "
            f"LONG {signal.long_symbol} qty={long_qty:.6f} @ ${long_price:,.2f} | "
            f"SHORT {signal.short_symbol} qty={short_qty:.6f} @ ${short_price:,.2f}"
        )

        # Open long leg
        try:
            long_trade = self._paper_trader.open_trade(
                symbol=signal.long_symbol,
                side="BUY",
                quantity=long_qty,
                entry_price=long_price,
                leverage=max_leverage,
                pattern_type=f"statarb_{signal_type_str}",
                strategy="statarb",
            )
            if long_trade:
                self._on_execution()
                logger.info(
                    f"  LONG leg opened: {long_trade.trade_id} {signal.long_symbol}"
                )
        except Exception as e:
            logger.error(f"StatArb long leg failed: {e}")

        # Open short leg
        try:
            short_trade = self._paper_trader.open_trade(
                symbol=signal.short_symbol,
                side="SELL",
                quantity=short_qty,
                entry_price=short_price,
                leverage=max_leverage,
                pattern_type=f"statarb_{signal_type_str}",
                strategy="statarb",
            )
            if short_trade:
                self._on_execution()
                logger.info(
                    f"  SHORT leg opened: {short_trade.trade_id} {signal.short_symbol}"
                )
        except Exception as e:
            logger.error(f"StatArb short leg failed: {e}")

        # Send Discord alert
        if self._alerter and self._alert_on_entry:
            try:
                self._alerter._send_message(
                    f"**STATARB ENTRY** | Z={signal.zscore:.2f}\n"
                    f"LONG {signal.long_symbol} @ ${long_price:,.2f}\n"
                    f"SHORT {signal.short_symbol} @ ${short_price:,.2f}"
                )
            except Exception as alert_err:
                logger.warning(f"Failed to send statarb entry alert: {alert_err}")

    def _execute_exit(self, signal: "StatArbSignal") -> None:
        """Execute StatArb exit (closes both legs)."""
        if self._paper_trader is None:
            return

        statarb_trades = self._paper_trader.get_trades_by_strategy("statarb")
        open_trades = [t for t in statarb_trades if t.exit_time is None]

        closed_count = 0
        for trade in open_trades:
            if trade.symbol in (signal.long_symbol, signal.short_symbol):
                try:
                    current_price = self._client.get_current_price(trade.symbol)
                    if current_price and current_price > 0:
                        trade.exit_reason = "statarb_zscore_reversion"
                        self._paper_trader.close_trade(
                            trade.trade_id,
                            exit_price=current_price,
                        )
                        closed_count += 1
                        logger.info(
                            f"STATARB EXIT: {trade.symbol} @ ${current_price:,.2f} "
                            f"(Z={signal.zscore:.2f}, held {signal.bars_held} bars)"
                        )
                except Exception as e:
                    logger.error(f"StatArb exit failed for {trade.symbol}: {e}")

        if closed_count > 0 and self._alerter and self._alert_on_exit:
            try:
                self._alerter._send_message(
                    f"**STATARB EXIT** | Z={signal.zscore:.2f} | "
                    f"Closed {closed_count} position(s)"
                )
            except Exception as alert_err:
                logger.warning(f"Failed to send statarb exit alert: {alert_err}")

    def get_status(self) -> Dict[str, Any]:
        """Get StatArb status for inclusion in daemon status."""
        if self.generator is None:
            return {}
        return self.generator.get_status()

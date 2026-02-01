"""
Coinbase CFM Database Persistence Layer.

Stores historical fills and P/L data in PostgreSQL to prevent data loss
when Coinbase API limits access to older fills (typically 90 days).

Tables:
- cfm_fills: Raw fill data from Coinbase API
- cfm_realized_pnl: Calculated realized P/L records
- cfm_daily_snapshots: Daily cumulative P/L snapshots

Usage:
    from dashboard.data_loaders.cfm_database import CFMDatabase

    db = CFMDatabase()
    if db.is_connected():
        db.save_fills(fills)
        historical = db.get_all_fills()
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Try to import SQLAlchemy
try:
    from sqlalchemy import (
        create_engine,
        Column,
        String,
        Float,
        DateTime,
        Integer,
        Boolean,
        Text,
        Date,
        UniqueConstraint,
        Index,
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.dialects.postgresql import JSONB
    _SQLALCHEMY_AVAILABLE = True
except ImportError:
    _SQLALCHEMY_AVAILABLE = False
    logger.warning("SQLAlchemy not available - database persistence disabled")

Base = declarative_base() if _SQLALCHEMY_AVAILABLE else None


# =============================================================================
# DATABASE MODELS
# =============================================================================

if _SQLALCHEMY_AVAILABLE:

    class CFMFill(Base):
        """Raw fill data from Coinbase API."""
        __tablename__ = 'cfm_fills'

        id = Column(Integer, primary_key=True, autoincrement=True)
        fill_id = Column(String(100), unique=True, nullable=False, index=True)
        order_id = Column(String(100), nullable=False)
        product_id = Column(String(50), nullable=False, index=True)
        base_symbol = Column(String(10), nullable=False, index=True)
        side = Column(String(10), nullable=False)  # BUY or SELL
        price = Column(Float, nullable=False)
        size = Column(Float, nullable=False)
        fee = Column(Float, nullable=False, default=0)
        trade_time = Column(DateTime(timezone=True), nullable=False, index=True)
        is_maker = Column(Boolean, default=False)
        trade_type = Column(String(50))
        raw_json = Column(JSONB)  # Store full API response for audit
        created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

        __table_args__ = (
            Index('idx_fills_product_time', 'product_id', 'trade_time'),
        )

    class CFMRealizedPL(Base):
        """Calculated realized P/L records."""
        __tablename__ = 'cfm_realized_pnl'

        id = Column(Integer, primary_key=True, autoincrement=True)
        product_id = Column(String(50), nullable=False, index=True)
        base_symbol = Column(String(10), nullable=False, index=True)
        open_fill_id = Column(String(100), nullable=False)
        close_fill_id = Column(String(100), nullable=False)
        quantity = Column(Float, nullable=False)
        entry_price = Column(Float, nullable=False)
        exit_price = Column(Float, nullable=False)
        entry_fee = Column(Float, default=0)
        exit_fee = Column(Float, default=0)
        gross_pnl = Column(Float, nullable=False)
        net_pnl = Column(Float, nullable=False)
        pnl_percent = Column(Float)
        hold_duration_seconds = Column(Integer)
        entry_time = Column(DateTime(timezone=True), nullable=False)
        exit_time = Column(DateTime(timezone=True), nullable=False, index=True)
        side = Column(String(10), nullable=False)  # Original position direction
        product_type = Column(String(20))  # crypto_perp or commodity_future
        created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

        __table_args__ = (
            UniqueConstraint('open_fill_id', 'close_fill_id', 'quantity', name='uq_realized_pnl'),
            Index('idx_realized_exit_time', 'exit_time'),
        )

    class CFMDailySnapshot(Base):
        """Daily cumulative P/L snapshots."""
        __tablename__ = 'cfm_daily_snapshots'

        id = Column(Integer, primary_key=True, autoincrement=True)
        snapshot_date = Column(Date, unique=True, nullable=False, index=True)
        daily_realized_pnl = Column(Float, default=0)
        cumulative_realized_pnl = Column(Float, default=0)
        daily_trade_count = Column(Integer, default=0)
        total_trade_count = Column(Integer, default=0)
        total_fees = Column(Float, default=0)
        # Per-product breakdown stored as JSON
        pnl_by_product = Column(JSONB)
        created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
        updated_at = Column(DateTime(timezone=True), onupdate=lambda: datetime.now(timezone.utc))


# =============================================================================
# DATABASE CLASS
# =============================================================================

class CFMDatabase:
    """
    Database interface for CFM P/L persistence.

    Connects to PostgreSQL via DATABASE_URL environment variable.
    Falls back gracefully if database is not configured.
    """

    def __init__(self):
        """Initialize database connection."""
        self._engine = None
        self._Session = None
        self._connected = False
        self.init_error: Optional[str] = None

        if not _SQLALCHEMY_AVAILABLE:
            self.init_error = "SQLAlchemy not installed"
            return

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            self.init_error = "DATABASE_URL not configured"
            logger.info("CFMDatabase: DATABASE_URL not set - persistence disabled")
            return

        try:
            # Handle Railway's postgres:// vs postgresql:// URL format
            if database_url.startswith("postgres://"):
                database_url = database_url.replace("postgres://", "postgresql://", 1)

            self._engine = create_engine(database_url, pool_pre_ping=True)
            self._Session = sessionmaker(bind=self._engine)

            # Create tables if they don't exist
            Base.metadata.create_all(self._engine)

            self._connected = True
            logger.info("CFMDatabase connected to PostgreSQL")

        except Exception as e:
            self.init_error = str(e)
            logger.error(f"CFMDatabase connection failed: {e}")

    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected

    def _get_session(self):
        """Get a new database session."""
        if not self._connected:
            return None
        return self._Session()

    # =========================================================================
    # FILL OPERATIONS
    # =========================================================================

    def save_fills(self, fills: List[Dict[str, Any]]) -> int:
        """
        Save fills to database (upsert - skip existing).

        Args:
            fills: List of fill dicts from Coinbase API

        Returns:
            Number of new fills saved
        """
        if not self._connected:
            return 0

        session = self._get_session()
        saved_count = 0

        try:
            for fill in fills:
                fill_id = fill.get("entry_id") or fill.get("trade_id")
                if not fill_id:
                    continue

                # Check if already exists
                existing = session.query(CFMFill).filter_by(fill_id=fill_id).first()
                if existing:
                    continue

                # Parse timestamp
                timestamp_str = fill.get("trade_time") or fill.get("sequence_timestamp")
                if isinstance(timestamp_str, str):
                    try:
                        trade_time = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                    except ValueError:
                        trade_time = datetime.now(timezone.utc)
                else:
                    trade_time = datetime.now(timezone.utc)

                # Parse fee
                commission = fill.get("commission", "0")
                fee = float(commission) if commission else 0.0

                # Extract base symbol
                product_id = fill.get("product_id", "")
                base_symbol = product_id.split("-")[0].upper() if product_id else ""

                db_fill = CFMFill(
                    fill_id=fill_id,
                    order_id=fill.get("order_id", ""),
                    product_id=product_id,
                    base_symbol=base_symbol,
                    side=fill.get("side", "").upper(),
                    price=float(fill.get("price", 0)),
                    size=float(fill.get("size", 0)),
                    fee=fee,
                    trade_time=trade_time,
                    is_maker=fill.get("liquidity_indicator", "").upper() == "MAKER",
                    trade_type=fill.get("trade_type", ""),
                    raw_json=fill,
                )
                session.add(db_fill)
                saved_count += 1

            session.commit()
            if saved_count > 0:
                logger.info(f"Saved {saved_count} new fills to database")

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving fills: {e}")
        finally:
            session.close()

        return saved_count

    def get_all_fills(self) -> List[Dict[str, Any]]:
        """
        Get all fills from database.

        Returns:
            List of fill dicts in Coinbase API format
        """
        if not self._connected:
            return []

        session = self._get_session()
        try:
            fills = session.query(CFMFill).order_by(CFMFill.trade_time).all()
            return [self._fill_to_dict(f) for f in fills]
        except Exception as e:
            logger.error(f"Error getting fills: {e}")
            return []
        finally:
            session.close()

    def get_fills_since(self, since: datetime) -> List[Dict[str, Any]]:
        """Get fills since a specific date."""
        if not self._connected:
            return []

        session = self._get_session()
        try:
            fills = session.query(CFMFill).filter(
                CFMFill.trade_time >= since
            ).order_by(CFMFill.trade_time).all()
            return [self._fill_to_dict(f) for f in fills]
        except Exception as e:
            logger.error(f"Error getting fills since {since}: {e}")
            return []
        finally:
            session.close()

    def get_latest_fill_time(self) -> Optional[datetime]:
        """Get timestamp of most recent fill in database."""
        if not self._connected:
            return None

        session = self._get_session()
        try:
            latest = session.query(CFMFill).order_by(CFMFill.trade_time.desc()).first()
            return latest.trade_time if latest else None
        except Exception as e:
            logger.error(f"Error getting latest fill time: {e}")
            return None
        finally:
            session.close()

    def _fill_to_dict(self, fill: 'CFMFill') -> Dict[str, Any]:
        """Convert database fill to API-compatible dict."""
        return {
            "entry_id": fill.fill_id,
            "trade_id": fill.fill_id,
            "order_id": fill.order_id,
            "product_id": fill.product_id,
            "side": fill.side,
            "price": str(fill.price),
            "size": str(fill.size),
            "commission": str(fill.fee),
            "trade_time": fill.trade_time.isoformat() if fill.trade_time else None,
            "liquidity_indicator": "MAKER" if fill.is_maker else "TAKER",
            "trade_type": fill.trade_type,
        }

    # =========================================================================
    # REALIZED P/L OPERATIONS
    # =========================================================================

    def save_realized_pnl(self, pnl_records: List[Dict[str, Any]]) -> int:
        """
        Save realized P/L records to database.

        Args:
            pnl_records: List of P/L dicts from calculator

        Returns:
            Number of new records saved
        """
        if not self._connected:
            return 0

        session = self._get_session()
        saved_count = 0

        try:
            for record in pnl_records:
                # Check if already exists
                existing = session.query(CFMRealizedPL).filter_by(
                    open_fill_id=record.get("open_fill_id"),
                    close_fill_id=record.get("close_fill_id"),
                    quantity=record.get("quantity"),
                ).first()
                if existing:
                    continue

                # Parse times
                entry_time = record.get("entry_time")
                exit_time = record.get("exit_time")
                if isinstance(entry_time, str):
                    entry_time = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))

                # Calculate hold duration in seconds
                hold_duration = record.get("hold_duration")
                if hasattr(hold_duration, "total_seconds"):
                    hold_seconds = int(hold_duration.total_seconds())
                elif isinstance(hold_duration, (int, float)):
                    hold_seconds = int(hold_duration)
                else:
                    hold_seconds = 0

                db_record = CFMRealizedPL(
                    product_id=record.get("product_id", ""),
                    base_symbol=record.get("base_symbol", ""),
                    open_fill_id=record.get("open_fill_id", ""),
                    close_fill_id=record.get("close_fill_id", ""),
                    quantity=record.get("quantity", 0),
                    entry_price=record.get("entry_price", 0),
                    exit_price=record.get("exit_price", 0),
                    entry_fee=record.get("entry_fee", 0),
                    exit_fee=record.get("exit_fee", 0),
                    gross_pnl=record.get("gross_pnl", 0),
                    net_pnl=record.get("net_pnl", 0),
                    pnl_percent=record.get("pnl_percent", 0),
                    hold_duration_seconds=hold_seconds,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    side=record.get("side", ""),
                    product_type=record.get("product_type", ""),
                )
                session.add(db_record)
                saved_count += 1

            session.commit()
            if saved_count > 0:
                logger.info(f"Saved {saved_count} realized P/L records to database")

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving realized P/L: {e}")
        finally:
            session.close()

        return saved_count

    def get_all_realized_pnl(self) -> List[Dict[str, Any]]:
        """Get all realized P/L records from database."""
        if not self._connected:
            return []

        session = self._get_session()
        try:
            records = session.query(CFMRealizedPL).order_by(CFMRealizedPL.exit_time).all()
            return [self._pnl_to_dict(r) for r in records]
        except Exception as e:
            logger.error(f"Error getting realized P/L: {e}")
            return []
        finally:
            session.close()

    def _pnl_to_dict(self, record: 'CFMRealizedPL') -> Dict[str, Any]:
        """Convert database P/L record to dict."""
        return {
            "product_id": record.product_id,
            "base_symbol": record.base_symbol,
            "open_fill_id": record.open_fill_id,
            "close_fill_id": record.close_fill_id,
            "quantity": record.quantity,
            "entry_price": record.entry_price,
            "exit_price": record.exit_price,
            "entry_fee": record.entry_fee,
            "exit_fee": record.exit_fee,
            "gross_pnl": record.gross_pnl,
            "net_pnl": record.net_pnl,
            "pnl_percent": record.pnl_percent,
            "hold_duration_seconds": record.hold_duration_seconds,
            "entry_time": record.entry_time.isoformat() if record.entry_time else None,
            "exit_time": record.exit_time.isoformat() if record.exit_time else None,
            "side": record.side,
            "product_type": record.product_type,
        }

    # =========================================================================
    # DAILY SNAPSHOT OPERATIONS
    # =========================================================================

    def save_daily_snapshot(
        self,
        date: datetime,
        daily_pnl: float,
        cumulative_pnl: float,
        daily_trades: int,
        total_trades: int,
        total_fees: float,
        pnl_by_product: Dict[str, float],
    ) -> bool:
        """Save or update daily snapshot."""
        if not self._connected:
            return False

        session = self._get_session()
        try:
            snapshot_date = date.date() if isinstance(date, datetime) else date

            existing = session.query(CFMDailySnapshot).filter_by(
                snapshot_date=snapshot_date
            ).first()

            if existing:
                existing.daily_realized_pnl = daily_pnl
                existing.cumulative_realized_pnl = cumulative_pnl
                existing.daily_trade_count = daily_trades
                existing.total_trade_count = total_trades
                existing.total_fees = total_fees
                existing.pnl_by_product = pnl_by_product
            else:
                snapshot = CFMDailySnapshot(
                    snapshot_date=snapshot_date,
                    daily_realized_pnl=daily_pnl,
                    cumulative_realized_pnl=cumulative_pnl,
                    daily_trade_count=daily_trades,
                    total_trade_count=total_trades,
                    total_fees=total_fees,
                    pnl_by_product=pnl_by_product,
                )
                session.add(snapshot)

            session.commit()
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Error saving daily snapshot: {e}")
            return False
        finally:
            session.close()

    def get_daily_snapshots(self) -> List[Dict[str, Any]]:
        """Get all daily snapshots."""
        if not self._connected:
            return []

        session = self._get_session()
        try:
            snapshots = session.query(CFMDailySnapshot).order_by(
                CFMDailySnapshot.snapshot_date
            ).all()
            return [{
                "date": s.snapshot_date.isoformat(),
                "daily_pnl": s.daily_realized_pnl,
                "cumulative_pnl": s.cumulative_realized_pnl,
                "daily_trades": s.daily_trade_count,
                "total_trades": s.total_trade_count,
                "total_fees": s.total_fees,
                "pnl_by_product": s.pnl_by_product or {},
            } for s in snapshots]
        except Exception as e:
            logger.error(f"Error getting daily snapshots: {e}")
            return []
        finally:
            session.close()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_fill_count(self) -> int:
        """Get total number of fills in database."""
        if not self._connected:
            return 0

        session = self._get_session()
        try:
            return session.query(CFMFill).count()
        except Exception as e:
            logger.error(f"Error getting fill count: {e}")
            return 0
        finally:
            session.close()

    def get_status(self) -> Dict[str, Any]:
        """Get database status."""
        return {
            "connected": self._connected,
            "error": self.init_error,
            "fill_count": self.get_fill_count() if self._connected else 0,
            "sqlalchemy_available": _SQLALCHEMY_AVAILABLE,
        }

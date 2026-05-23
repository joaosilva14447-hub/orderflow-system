"""Abstract base for all data providers."""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from core.models import Trade, Bar, OrderBook


class BaseProvider(ABC):
    """
    All providers must implement this interface.
    Processing engine is completely decoupled from data source.
    """

    @abstractmethod
    async def stream_trades(self, symbol: str) -> AsyncGenerator[Trade, None]:
        """Real-time trade stream with aggressor side."""
        ...

    @abstractmethod
    async def stream_orderbook(self, symbol: str, depth: int = 20) -> AsyncGenerator[OrderBook, None]:
        """Real-time order book stream."""
        ...

    @abstractmethod
    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> list[Bar]:
        """Historical OHLCV bars."""
        ...

    @abstractmethod
    def normalize_symbol(self, symbol: str) -> str:
        """Convert universal symbol (BTC/USDT) to exchange format."""
        ...

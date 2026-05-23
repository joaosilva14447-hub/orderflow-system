from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"


class Market(str, Enum):
    CRYPTO = "crypto"
    STOCK = "stock"
    FUTURES = "futures"
    FOREX = "forex"
    COMMODITY = "commodity"


@dataclass
class Trade:
    timestamp: float       # Unix ms
    price: float
    volume: float
    side: Side
    symbol: str
    market: Market
    exchange: str = ""

    @property
    def signed_volume(self) -> float:
        if self.side == Side.BUY:
            return self.volume
        if self.side == Side.SELL:
            return -self.volume
        return 0.0


@dataclass
class Bar:
    timestamp: float       # Unix ms (open time)
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    market: Market
    timeframe: str = "1m"
    buy_volume: float = 0.0
    sell_volume: float = 0.0

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2


@dataclass
class OrderBookLevel:
    timestamp: float
    price: float
    size: float
    side: Side
    symbol: str
    market: Market


@dataclass
class OrderBook:
    timestamp: float
    symbol: str
    market: Market
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

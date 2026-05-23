"""
Order Book Heatmap engine.
- Local (Bybit): real L2 order book data via WebSocket snapshot
- Fallback: simulate liquidity clusters from price action (OHLCV)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from core.models import Bar, OrderBook


@dataclass
class LiquidityLevel:
    price:     float
    bid_size:  float = 0.0
    ask_size:  float = 0.0

    @property
    def total(self) -> float:
        return self.bid_size + self.ask_size

    @property
    def imbalance(self) -> float:
        """Positive = bid heavy, negative = ask heavy."""
        if self.total == 0:
            return 0.0
        return (self.bid_size - self.ask_size) / self.total


@dataclass
class HeatmapSnapshot:
    timestamp: float
    levels:    list[LiquidityLevel] = field(default_factory=list)
    mid_price: Optional[float]      = None

    @property
    def max_size(self) -> float:
        return max((lv.total for lv in self.levels), default=1.0)

    def bid_wall(self, min_size_pct: float = 0.05) -> list[LiquidityLevel]:
        threshold = self.max_size * min_size_pct
        return [lv for lv in self.levels if lv.bid_size >= threshold]

    def ask_wall(self, min_size_pct: float = 0.05) -> list[LiquidityLevel]:
        threshold = self.max_size * min_size_pct
        return [lv for lv in self.levels if lv.ask_size >= threshold]


# ── From real order book ──────────────────────────────────────────────────────

def heatmap_from_orderbook(book: OrderBook, tick_size: float) -> HeatmapSnapshot:
    """Build heatmap snapshot from a real L2 order book."""
    levels: dict[float, LiquidityLevel] = {}

    def _bucket(price: float) -> float:
        return round(round(price / tick_size) * tick_size, 10)

    for bid in book.bids:
        pk = _bucket(bid.price)
        if pk not in levels:
            levels[pk] = LiquidityLevel(pk)
        levels[pk].bid_size += bid.size

    for ask in book.asks:
        pk = _bucket(ask.price)
        if pk not in levels:
            levels[pk] = LiquidityLevel(pk)
        levels[pk].ask_size += ask.size

    sorted_levels = sorted(levels.values(), key=lambda x: x.price, reverse=True)
    return HeatmapSnapshot(
        timestamp = book.timestamp,
        levels    = sorted_levels,
        mid_price = book.mid_price,
    )


# ── Approximation from OHLCV ──────────────────────────────────────────────────

def heatmap_from_bars(bars: list[Bar], tick_size: float, window: int = 50) -> HeatmapSnapshot:
    """
    Approximate liquidity clusters from price action.
    High-volume price levels act as proxy for order concentration.
    Not as accurate as real L2 — directional guidance only.
    """
    recent = bars[-window:]
    levels: dict[float, LiquidityLevel] = {}

    def _bucket(p: float) -> float:
        return round(round(p / tick_size) * tick_size, 10)

    for b in recent:
        vol = b.volume

        # Approximate: support around lows (bid concentration)
        low_pk  = _bucket(b.low)
        high_pk = _bucket(b.high)

        if low_pk not in levels:  levels[low_pk]  = LiquidityLevel(low_pk)
        if high_pk not in levels: levels[high_pk] = LiquidityLevel(high_pk)

        # Bid liquidity clusters near swing lows
        levels[low_pk].bid_size  += b.sell_volume * 0.6
        # Ask liquidity clusters near swing highs
        levels[high_pk].ask_size += b.buy_volume  * 0.6

        # Volume at close price (absorbed liquidity / POC proxy)
        close_pk = _bucket(b.close)
        if close_pk not in levels: levels[close_pk] = LiquidityLevel(close_pk)
        levels[close_pk].bid_size += b.buy_volume  * 0.4
        levels[close_pk].ask_size += b.sell_volume * 0.4

    sorted_levels = sorted(levels.values(), key=lambda x: x.price, reverse=True)
    mid = recent[-1].close if recent else None

    return HeatmapSnapshot(
        timestamp = recent[-1].timestamp if recent else 0,
        levels    = sorted_levels,
        mid_price = mid,
    )


# ── Historical heatmap (for time-based visualization) ────────────────────────

def rolling_heatmap(bars: list[Bar], tick_size: float, window: int = 20) -> list[HeatmapSnapshot]:
    """Build a series of heatmap snapshots (sliding window) for time-based viz."""
    snaps = []
    for i in range(window, len(bars) + 1):
        snap = heatmap_from_bars(bars[max(0, i - window):i], tick_size, window)
        snaps.append(snap)
    return snaps

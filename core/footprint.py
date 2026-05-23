"""
Footprint chart engine — buy/sell volume per price level per bar.
Works with approximated OHLCV data or real tick data (Bybit/Binance local).
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from core.models import Bar, Trade, Side


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FootprintLevel:
    price: float
    buy:   float = 0.0
    sell:  float = 0.0

    @property
    def delta(self) -> float:
        return self.buy - self.sell

    @property
    def total(self) -> float:
        return self.buy + self.sell

    @property
    def buy_ratio(self) -> float:
        return self.buy / self.total if self.total > 0 else 0.5

    def is_bid_imbalance(self, threshold: float = 3.0) -> bool:
        """Buyers dominate — potential support / continuation up."""
        return self.sell > 0 and (self.buy / self.sell) >= threshold

    def is_ask_imbalance(self, threshold: float = 3.0) -> bool:
        """Sellers dominate — potential resistance / continuation down."""
        return self.buy > 0 and (self.sell / self.buy) >= threshold


@dataclass
class FootprintBar:
    bar:       Bar
    tick_size: float
    levels:    dict = field(default_factory=dict)   # float → FootprintLevel

    @property
    def poc(self) -> Optional[float]:
        if not self.levels:
            return None
        return max(self.levels, key=lambda p: self.levels[p].total)

    @property
    def delta(self) -> float:
        return sum(lv.delta for lv in self.levels.values())

    @property
    def cum_buy(self) -> float:
        return sum(lv.buy for lv in self.levels.values())

    @property
    def cum_sell(self) -> float:
        return sum(lv.sell for lv in self.levels.values())

    def get_imbalances(self, threshold: float = 3.0) -> list[tuple[float, str]]:
        result = []
        for price, lv in self.levels.items():
            if lv.is_bid_imbalance(threshold):
                result.append((price, "bid"))
            elif lv.is_ask_imbalance(threshold):
                result.append((price, "ask"))
        return result


# ── Builder from OHLCV (approximation) ───────────────────────────────────────

def approximate_footprint(bar: Bar, tick_size: float) -> FootprintBar:
    """
    Distribute OHLCV volume across price levels.
    Uses a triangular distribution peaking at close (most trading near close).
    Buy/sell split per level based on price position relative to open.
    NOTE: This is an approximation — real footprint requires tick data.
    """
    fp = FootprintBar(bar=bar, tick_size=tick_size)

    lo = round(bar.low  / tick_size) * tick_size
    hi = round(bar.high / tick_size) * tick_size
    n_levels = max(int(round((hi - lo) / tick_size)) + 1, 1)

    if n_levels == 1:
        fp.levels[lo] = FootprintLevel(lo, bar.buy_volume, bar.sell_volume)
        return fp

    prices = np.linspace(lo, hi, n_levels)

    # Volume weight: triangular, peak at close
    close_idx = np.argmin(np.abs(prices - bar.close))
    raw_w = np.array([1.0 / (abs(i - close_idx) + 1) for i in range(n_levels)], dtype=float)
    raw_w /= raw_w.sum()

    total_buy  = bar.buy_volume
    total_sell = bar.sell_volume
    mid = (bar.open + bar.close) / 2

    for price, w in zip(prices, raw_w):
        vol = bar.volume * w
        # Levels above midpoint → more buy pressure; below → more sell
        if bar.close >= bar.open:             # bullish bar
            buy_ratio = 0.65 if price >= mid else 0.35
        else:                                  # bearish bar
            buy_ratio = 0.35 if price >= mid else 0.65

        buy  = vol * buy_ratio
        sell = vol * (1 - buy_ratio)
        pk   = round(price, 10)
        fp.levels[pk] = FootprintLevel(pk, buy, sell)

    # Normalise totals to match bar buy/sell volumes
    actual_buy  = sum(lv.buy  for lv in fp.levels.values())
    actual_sell = sum(lv.sell for lv in fp.levels.values())
    scale_b = total_buy  / actual_buy  if actual_buy  > 0 else 1.0
    scale_s = total_sell / actual_sell if actual_sell > 0 else 1.0
    for lv in fp.levels.values():
        lv.buy  *= scale_b
        lv.sell *= scale_s

    return fp


# ── Builder from real tick data ───────────────────────────────────────────────

def footprint_from_trades(
    trades:    list[Trade],
    bar:       Bar,
    tick_size: float,
) -> FootprintBar:
    """Real footprint — requires tick data with aggressor side (Bybit/Binance local)."""
    fp = FootprintBar(bar=bar, tick_size=tick_size)
    for t in trades:
        pk = round(round(t.price / tick_size) * tick_size, 10)
        if pk not in fp.levels:
            fp.levels[pk] = FootprintLevel(pk)
        if t.side == Side.BUY:
            fp.levels[pk].buy  += t.volume
        elif t.side == Side.SELL:
            fp.levels[pk].sell += t.volume
    return fp


# ── Batch builder ─────────────────────────────────────────────────────────────

def build_footprints(bars: list[Bar], tick_size: float) -> list[FootprintBar]:
    return [approximate_footprint(b, tick_size) for b in bars]

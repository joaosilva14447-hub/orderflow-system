"""
Core processing engine — fully market-agnostic.
All methods accept the unified Trade/Bar/OrderBook models.
"""

from collections import defaultdict, deque
from typing import Optional
import numpy as np

from core.models import Trade, Bar, OrderBook, Side


class CVDEngine:
    """Cumulative Volume Delta — real-time and historical."""

    def __init__(self, window: Optional[int] = None):
        self._cvd: float = 0.0
        self._history: deque = deque(maxlen=window)

    def update(self, trade: Trade) -> float:
        self._cvd += trade.signed_volume
        self._history.append(self._cvd)
        return self._cvd

    def update_bar(self, bar: Bar) -> float:
        self._cvd += bar.delta
        self._history.append(self._cvd)
        return self._cvd

    @property
    def value(self) -> float:
        return self._cvd

    @property
    def history(self) -> list[float]:
        return list(self._history)

    def reset(self) -> None:
        self._cvd = 0.0
        self._history.clear()


class VolumeProfileEngine:
    """
    Volume Profile (VPVR) — price levels with buy/sell volume.
    tick_size: minimum price granularity for bucketing.
    """

    def __init__(self, tick_size: float = 1.0):
        self.tick_size = tick_size
        self._profile: dict[float, dict] = defaultdict(
            lambda: {"total": 0.0, "buy": 0.0, "sell": 0.0, "delta": 0.0}
        )

    def _bucket(self, price: float) -> float:
        return round(round(price / self.tick_size) * self.tick_size, 10)

    def update(self, trade: Trade) -> None:
        level = self._bucket(trade.price)
        self._profile[level]["total"] += trade.volume
        if trade.side == Side.BUY:
            self._profile[level]["buy"] += trade.volume
        elif trade.side == Side.SELL:
            self._profile[level]["sell"] += trade.volume
        self._profile[level]["delta"] = (
            self._profile[level]["buy"] - self._profile[level]["sell"]
        )

    def update_bar(self, bar: Bar, distribute: bool = True) -> None:
        level = self._bucket(bar.mid)
        self._profile[level]["total"] += bar.volume
        self._profile[level]["buy"] += bar.buy_volume
        self._profile[level]["sell"] += bar.sell_volume
        self._profile[level]["delta"] += bar.delta

    @property
    def poc(self) -> Optional[float]:
        if not self._profile:
            return None
        return max(self._profile, key=lambda p: self._profile[p]["total"])

    def value_area(self, pct: float = 0.70) -> tuple[float, float]:
        if not self._profile:
            return (0.0, 0.0)
        total = sum(v["total"] for v in self._profile.values())
        target = total * pct
        poc_price = self.poc
        levels = sorted(self._profile.keys())
        poc_idx = levels.index(poc_price)
        accumulated = self._profile[poc_price]["total"]
        lo, hi = poc_idx, poc_idx
        while accumulated < target:
            expand_up = hi + 1 < len(levels)
            expand_dn = lo - 1 >= 0
            if not expand_up and not expand_dn:
                break
            vol_up = self._profile[levels[hi + 1]]["total"] if expand_up else 0
            vol_dn = self._profile[levels[lo - 1]]["total"] if expand_dn else 0
            if vol_up >= vol_dn and expand_up:
                hi += 1
                accumulated += vol_up
            elif expand_dn:
                lo -= 1
                accumulated += vol_dn
            else:
                hi += 1
                accumulated += vol_up
        return (levels[lo], levels[hi])

    def to_dict(self) -> dict:
        return dict(self._profile)

    def reset(self) -> None:
        self._profile.clear()


class DeltaEngine:
    """Per-bar and rolling delta analysis."""

    def __init__(self, window: int = 20):
        self._deltas: deque = deque(maxlen=window)

    def update(self, bar: Bar) -> float:
        self._deltas.append(bar.delta)
        return bar.delta

    @property
    def cumulative(self) -> float:
        return sum(self._deltas)

    @property
    def mean(self) -> float:
        if not self._deltas:
            return 0.0
        return np.mean(list(self._deltas))

    @property
    def divergence(self) -> bool:
        """Price making HH but delta making LH — bearish divergence, vice versa."""
        return False  # Extended in FootprintEngine

    @property
    def history(self) -> list[float]:
        return list(self._deltas)


class ImbalanceEngine:
    """
    Detects bid/ask imbalances in the order book.
    threshold: minimum ratio to flag as imbalance (e.g. 3.0 = 3:1).
    """

    def __init__(self, threshold: float = 3.0, levels: int = 5):
        self.threshold = threshold
        self.levels = levels

    def detect(self, book: OrderBook) -> dict:
        bids = book.bids[: self.levels]
        asks = book.asks[: self.levels]

        bid_vol = sum(l.size for l in bids)
        ask_vol = sum(l.size for l in asks)

        ratio = bid_vol / ask_vol if ask_vol > 0 else float("inf")
        inverse = ask_vol / bid_vol if bid_vol > 0 else float("inf")

        return {
            "bid_volume": bid_vol,
            "ask_volume": ask_vol,
            "ratio": ratio,
            "bid_imbalance": ratio >= self.threshold,
            "ask_imbalance": inverse >= self.threshold,
            "net": bid_vol - ask_vol,
        }


class OrderFlowEngine:
    """Unified engine — wires CVD, VolumeProfile, Delta, Imbalance together."""

    def __init__(self, tick_size: float = 1.0, cvd_window: int = 500):
        self.cvd = CVDEngine(window=cvd_window)
        self.profile = VolumeProfileEngine(tick_size=tick_size)
        self.delta = DeltaEngine()
        self.imbalance = ImbalanceEngine()

    def on_trade(self, trade: Trade) -> dict:
        cvd = self.cvd.update(trade)
        self.profile.update(trade)
        return {"cvd": cvd, "poc": self.profile.poc}

    def on_bar(self, bar: Bar) -> dict:
        cvd = self.cvd.update_bar(bar)
        self.profile.update_bar(bar)
        delta = self.delta.update(bar)
        return {
            "cvd": cvd,
            "delta": delta,
            "poc": self.profile.poc,
            "value_area": self.profile.value_area(),
            "delta_mean": self.delta.mean,
        }

    def on_book(self, book: OrderBook) -> dict:
        return self.imbalance.detect(book)

    def reset_profile(self) -> None:
        self.profile.reset()
        self.cvd.reset()

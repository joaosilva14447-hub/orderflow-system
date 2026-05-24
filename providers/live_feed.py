"""
Live trade accumulator — WebSocket feed from Bybit.

Architecture:
  • Runs a WebSocket in a background daemon thread (one per symbol/timeframe).
  • Accumulates trades into OHLCV bars with EXACT buy/sell volumes — no approximation.
  • Thread-safe: dashboard reads bars from any thread via get_bars() / get_open_bar().
  • Hybrid mode: seed with REST historical bars, then extend with live WebSocket bars.

Usage (Streamlit):
    @st.cache_resource
    def get_live_feed(symbol, timeframe):
        feed = LiveFeed(symbol, timeframe)
        feed.seed(historical_bars)   # optional: pre-fill with REST data
        feed.start()
        return feed

    feed   = get_live_feed("BTC/USDT", "1m")
    bars   = feed.get_bars(n=200)   # completed bars + current open bar
    status = feed.status()           # dict for UI display
"""

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.models import Bar, Market, Side

# ── Timeframe helpers ─────────────────────────────────────────────────────────

_TF_MS: dict[str, int] = {
    "1m":  60_000,       "3m":  180_000,    "5m":  300_000,
    "15m": 900_000,      "30m": 1_800_000,  "1h":  3_600_000,
    "4h":  14_400_000,   "1d":  86_400_000,
}


def _tf_to_ms(timeframe: str) -> int:
    return _TF_MS.get(timeframe, 60_000)


def _bar_start(ts_ms: float, tf_ms: int) -> int:
    """Round a timestamp down to the nearest bar boundary."""
    return int(ts_ms // tf_ms) * tf_ms


# ── Internal bar builder ──────────────────────────────────────────────────────

@dataclass
class _BarAccum:
    """Mutable bar accumulator — updated trade by trade."""
    ts:         int
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float = 0.0
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    trade_count: int  = 0

    def update(self, price: float, volume: float, is_buy: bool) -> None:
        self.high   = max(self.high,  price)
        self.low    = min(self.low,   price)
        self.close  = price
        self.volume      += volume
        self.buy_volume  += volume if is_buy else 0.0
        self.sell_volume += volume if not is_buy else 0.0
        self.trade_count += 1

    def to_bar(self, symbol: str, timeframe: str) -> Bar:
        return Bar(
            timestamp   = float(self.ts),
            open        = self.open,
            high        = self.high,
            low         = self.low,
            close       = self.close,
            volume      = self.volume,
            symbol      = symbol,
            market      = Market.CRYPTO,
            timeframe   = timeframe,
            buy_volume  = self.buy_volume,
            sell_volume = self.sell_volume,
        )


# ── LiveFeed ──────────────────────────────────────────────────────────────────

class LiveFeed:
    """
    Thread-safe live trade accumulator.

    Completed bars (from WebSocket) are stored in self._live_bars.
    The current open bar is in self._open.
    Historical (seed) bars are in self._hist_bars.

    get_bars() merges hist + live_completed + open_bar, deduplicating by timestamp.
    """

    def __init__(
        self,
        symbol:    str,
        timeframe: str  = "1m",
        max_live_bars: int = 500,
    ):
        self.symbol    = symbol
        self.timeframe = timeframe
        self.tf_ms     = _tf_to_ms(timeframe)

        self._hist_bars:  list[Bar]          = []
        self._live_bars:  deque[Bar]         = deque(maxlen=max_live_bars)
        self._open:       Optional[_BarAccum] = None
        self._lock        = threading.RLock()

        # Connection state
        self._thread:      Optional[threading.Thread] = None
        self._stop         = threading.Event()
        self.connected:    bool  = False
        self.trade_count:  int   = 0
        self.last_trade_ts: float = 0.0
        self.error_msg:    str   = ""
        self.started_at:   float = 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    def seed(self, bars: list[Bar]) -> None:
        """Pre-fill with historical REST bars (call before start())."""
        with self._lock:
            self._hist_bars = list(bars)

    def start(self) -> None:
        """Start background WebSocket thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.started_at = time.time()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"livefeed-{self.symbol.replace('/', '')}-{self.timeframe}",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to exit."""
        self._stop.set()

    def get_bars(self, n: int = 500) -> list[Bar]:
        """
        Return the merged bar list: historical + live-completed + open bar.
        All unique by timestamp (live takes priority over historical for same ts).
        """
        with self._lock:
            hist  = list(self._hist_bars)
            live  = list(self._live_bars)
            open_ = self._open.to_bar(self.symbol, self.timeframe) if (
                self._open and self._open.volume > 0
            ) else None

        # Merge: build dict by timestamp (live overwrites hist)
        merged: dict[int, Bar] = {}
        for b in hist:
            merged[int(b.timestamp)] = b
        for b in live:
            merged[int(b.timestamp)] = b
        if open_:
            merged[int(open_.timestamp)] = open_

        bars = sorted(merged.values(), key=lambda b: b.timestamp)
        return bars[-n:]

    def get_open_bar(self) -> Optional[Bar]:
        """Return the current (incomplete) bar, or None if not yet started."""
        with self._lock:
            if self._open and self._open.volume > 0:
                return self._open.to_bar(self.symbol, self.timeframe)
            return None

    def status(self) -> dict:
        """Snapshot for UI display."""
        with self._lock:
            live_bars = len(self._live_bars)
            hist_bars = len(self._hist_bars)
            open_tc   = self._open.trade_count if self._open else 0

        elapsed = time.time() - self.last_trade_ts if self.last_trade_ts else None
        return {
            "connected":      self.connected,
            "symbol":         self.symbol,
            "timeframe":      self.timeframe,
            "trade_count":    self.trade_count,
            "last_trade_ago": round(elapsed, 1) if elapsed else None,
            "hist_bars":      hist_bars,
            "live_bars":      live_bars,
            "open_trades":    open_tc,
            "error":          self.error_msg,
        }

    # ── Background thread ─────────────────────────────────────────────────────

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while not self._stop.is_set():
            try:
                loop.run_until_complete(self._stream())
            except Exception as e:
                self.error_msg = str(e)
                self.connected = False
            if not self._stop.is_set():
                time.sleep(3)   # back-off before reconnect
        loop.close()

    async def _stream(self) -> None:
        # Import here to avoid circular deps at module level
        from providers.bybit import BybitProvider
        prov = BybitProvider()
        self.connected  = True
        self.error_msg  = ""
        try:
            async for trade in prov.stream_trades(self.symbol):
                if self._stop.is_set():
                    break
                self._on_trade(trade)
        finally:
            self.connected = False

    # ── Trade handler (called from background thread) ─────────────────────────

    def _on_trade(self, trade) -> None:
        ts       = trade.timestamp
        bar_ts   = _bar_start(ts, self.tf_ms)
        is_buy   = (trade.side == Side.BUY)

        with self._lock:
            self.trade_count   += 1
            self.last_trade_ts  = time.time()

            if self._open is None:
                # First trade ever
                self._open = _BarAccum(
                    ts=bar_ts, open=trade.price, high=trade.price,
                    low=trade.price, close=trade.price,
                )
                self._open.update(trade.price, trade.volume, is_buy)
                return

            if bar_ts > self._open.ts:
                # Current bar closed → push to completed, start new
                self._live_bars.append(
                    self._open.to_bar(self.symbol, self.timeframe)
                )
                self._open = _BarAccum(
                    ts=bar_ts, open=trade.price, high=trade.price,
                    low=trade.price, close=trade.price,
                )

            self._open.update(trade.price, trade.volume, is_buy)

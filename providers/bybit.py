"""
Bybit provider — globally accessible, no geo-restrictions.
Supports Spot, Linear Perpetuals, Inverse Futures.
No API key required for public market data.

Delta quality levels (best → approximated):
  WebSocket (LiveFeed)  → exact aggressor side per trade  ✅✅✅
  REST recent-trade     → real side for bars in last 1000 trades  ✅✅
  close-position approx → (close-low)/(high-low) weighting  ✅
  flat 50/50 split      → old approach, now replaced  ❌
"""

import json
import time
from typing import AsyncGenerator, Optional

import certifi
import requests
import aiohttp
import websockets

from core.ssl_fix import apply as _ssl_fix; _ssl_fix()
from core.models import Bar, Market, OrderBook, OrderBookLevel, Side, Trade
from providers.base import BaseProvider

REST_BASE = "https://api.bybit.com/v5/market"
WS_BASE   = "wss://stream.bybit.com/v5/public"

TF_MAP = {
    "1m": "1",  "3m": "3",   "5m": "5",  "15m": "15",
    "30m": "30","1h": "60",  "4h": "240","1d": "D",
}

TF_MS = {
    "1m": 60_000,       "3m": 180_000,    "5m": 300_000,
    "15m": 900_000,     "30m": 1_800_000, "1h": 3_600_000,
    "4h": 14_400_000,   "1d": 86_400_000,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _category(symbol: str) -> str:
    sym = symbol.replace("/", "").upper()
    if sym.endswith("USDT") or sym.endswith("USDC"):
        return "linear"   # perpetual futures (most liquid)
    if sym.endswith("USD"):
        return "inverse"
    return "spot"


def _buy_vol_approx(high: float, low: float, close: float, volume: float) -> float:
    """
    Close-position approximation for taker buy volume.
    If close is near the high → strong buy pressure → buy_vol ≈ volume.
    If close is near the low  → strong sell pressure → buy_vol ≈ 0.
    Far more accurate than a flat 50/50 split.
    """
    rng = high - low
    if rng < 1e-10:
        return volume * 0.5
    return volume * (close - low) / rng


def _parse_klines(raw: list, symbol: str, timeframe: str) -> list[Bar]:
    """
    Bybit kline format: [startTime, open, high, low, close, volume, turnover]
    Data arrives newest-first → reversed for chronological order.
    Buy/sell split: close-position approximation (replaces old 50/50).
    """
    bars = []
    for k in reversed(raw):
        if not isinstance(k, list) or len(k) < 6:
            continue
        o   = float(k[1])
        h   = float(k[2])
        lo  = float(k[3])
        c   = float(k[4])
        vol = float(k[5])
        bv  = _buy_vol_approx(h, lo, c, vol)
        bars.append(Bar(
            timestamp   = float(k[0]),
            open        = o,
            high        = h,
            low         = lo,
            close       = c,
            volume      = vol,
            symbol      = symbol,
            market      = Market.CRYPTO,
            timeframe   = timeframe,
            buy_volume  = bv,
            sell_volume = vol - bv,
        ))
    return bars


def _overlay_trades(bars: list[Bar], trades: list[Trade], tf_ms: int) -> list[Bar]:
    """
    Replace close-position approximation with real aggressor-side volumes
    for bars that have sufficient trade coverage (≥ 50% of bar volume).
    """
    if not trades or not bars:
        return bars

    # Accumulate buy/sell per bar-start timestamp
    bar_data: dict[int, dict] = {}
    for t in trades:
        bs = int(t.timestamp // tf_ms) * tf_ms
        if bs not in bar_data:
            bar_data[bs] = {"buy": 0.0, "sell": 0.0}
        if t.side == Side.BUY:
            bar_data[bs]["buy"]  += t.volume
        else:
            bar_data[bs]["sell"] += t.volume

    # Build lookup by timestamp
    bar_map = {int(b.timestamp): i for i, b in enumerate(bars)}

    result = list(bars)
    for bs, data in bar_data.items():
        idx = bar_map.get(bs)
        if idx is None:
            continue
        b         = bars[idx]
        trade_vol = data["buy"] + data["sell"]
        if trade_vol <= 0:
            continue
        # Only replace if trades cover ≥ 50% of the bar's known volume
        coverage = trade_vol / b.volume if b.volume > 0 else 0.0
        if coverage >= 0.5:
            # Scale to match the kline volume exactly
            scale = b.volume / trade_vol
            result[idx] = Bar(
                timestamp   = b.timestamp,
                open        = b.open,
                high        = b.high,
                low         = b.low,
                close       = b.close,
                volume      = b.volume,
                symbol      = b.symbol,
                market      = b.market,
                timeframe   = b.timeframe,
                buy_volume  = data["buy"]  * scale,
                sell_volume = data["sell"] * scale,
            )
    return result


# ── Provider ──────────────────────────────────────────────────────────────────

class BybitProvider(BaseProvider):

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").upper()

    # ── Recent trades (real side data) ────────────────────────────────────────

    def _fetch_recent_trades_sync(self, symbol: str, limit: int = 1000) -> list[Trade]:
        """
        Fetch up to 1000 recent executed trades with real aggressor side.
        Used to overlay accurate buy/sell on the most recent kline bars.
        """
        sym      = self.normalize_symbol(symbol)
        category = _category(symbol)
        params   = {"category": category, "symbol": sym, "limit": min(limit, 1000)}
        try:
            resp = requests.get(
                f"{REST_BASE}/recent-trade", params=params,
                timeout=8, verify=certifi.where(),
            )
            data = resp.json()
            if data.get("retCode", -1) != 0:
                return []
            trades = []
            for t in data["result"]["list"]:
                trades.append(Trade(
                    timestamp = float(t["time"]),
                    price     = float(t["price"]),
                    volume    = float(t["size"]),
                    side      = Side.BUY if t["side"] == "Buy" else Side.SELL,
                    symbol    = symbol,
                    market    = Market.CRYPTO,
                    exchange  = "bybit",
                ))
            return trades
        except Exception:
            return []

    # ── Sync REST ─────────────────────────────────────────────────────────────

    def fetch_bars_sync(
        self,
        symbol:    str,
        timeframe: str,
        limit:     int = 500,
        start:     Optional[float] = None,
        end:       Optional[float] = None,
    ) -> list[Bar]:
        """Kline bars with close-position approximated buy/sell."""
        sym      = self.normalize_symbol(symbol)
        tf       = TF_MAP.get(timeframe, "60")
        category = _category(symbol)
        params   = {
            "category": category, "symbol": sym,
            "interval": tf, "limit": min(limit, 1000),
        }
        if start:
            params["start"] = int(start)
        if end:
            params["end"] = int(end)

        resp = requests.get(
            f"{REST_BASE}/kline", params=params,
            timeout=10, verify=certifi.where(),
        )
        data = resp.json()
        if data.get("retCode", -1) != 0:
            raise ValueError(f"Bybit API error: {data.get('retMsg', data)}")

        return _parse_klines(data["result"]["list"], symbol, timeframe)

    def fetch_bars_paginated(
        self,
        symbol:    str,
        timeframe: str,
        limit:     int = 1000,
    ) -> list[Bar]:
        """
        Fetch up to `limit` bars by paginating the Bybit kline API.
        The standard endpoint caps at 200 per call; this loops backwards in time.
        """
        tf_ms    = TF_MS.get(timeframe, 60_000)
        all_bars: list[Bar] = []
        end_ts:   Optional[float] = None  # ms — walk backwards

        while len(all_bars) < limit:
            n     = min(200, limit - len(all_bars))
            batch = self.fetch_bars_sync(symbol, timeframe, limit=n, end=end_ts)
            if not batch:
                break
            # batch is ascending (oldest first after _parse_klines reversal)
            all_bars = batch + all_bars          # prepend older bars
            if len(batch) < n:
                break                            # no more history available
            # Next page ends just before the oldest bar in this batch
            end_ts = batch[0].timestamp - tf_ms

        return all_bars[-limit:]

    def fetch_bars_enriched(
        self,
        symbol:    str,
        timeframe: str,
        limit:     int = 500,
    ) -> tuple[list[Bar], int]:
        """
        Best-effort delta quality:
          • All bars  → close-position approximation
          • Recent bars (covered by last 1000 trades) → real aggressor-side volumes

        Returns (bars, n_enriched) where n_enriched = bars with real trade data.
        """
        bars   = self.fetch_bars_sync(symbol, timeframe, limit)
        tf_ms  = TF_MS.get(timeframe, 60_000)
        trades = self._fetch_recent_trades_sync(symbol, limit=1000)

        before = [b.buy_volume for b in bars]
        bars   = _overlay_trades(bars, trades, tf_ms)
        after  = [b.buy_volume for b in bars]

        n_enriched = sum(1 for a, b in zip(before, after) if abs(a - b) > 0.001)
        return bars, n_enriched

    # ── Async REST ────────────────────────────────────────────────────────────

    async def fetch_bars(
        self,
        symbol:    str,
        timeframe: str,
        limit:     int = 500,
        start:     Optional[float] = None,
        end:       Optional[float] = None,
    ) -> list[Bar]:
        sym      = self.normalize_symbol(symbol)
        tf       = TF_MAP.get(timeframe, "60")
        category = _category(symbol)
        params   = {
            "category": category, "symbol": sym,
            "interval": tf, "limit": min(limit, 1000),
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{REST_BASE}/kline", params=params) as resp:
                data = await resp.json()

        if data.get("retCode", -1) != 0:
            raise ValueError(f"Bybit API error: {data.get('retMsg', data)}")

        return _parse_klines(data["result"]["list"], symbol, timeframe)

    # ── WebSocket streams ─────────────────────────────────────────────────────

    async def stream_trades(self, symbol: str) -> AsyncGenerator[Trade, None]:
        sym      = self.normalize_symbol(symbol)
        category = _category(symbol)
        url      = f"{WS_BASE}/{category}"

        async with websockets.connect(url, ping_interval=20) as ws:
            await ws.send(json.dumps({
                "op": "subscribe",
                "args": [f"publicTrade.{sym}"],
            }))
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("topic", "").startswith("publicTrade"):
                    for t in msg.get("data", []):
                        yield Trade(
                            timestamp = float(t["T"]),
                            price     = float(t["p"]),
                            volume    = float(t["v"]),
                            side      = Side.BUY if t["S"] == "Buy" else Side.SELL,
                            symbol    = symbol,
                            market    = Market.CRYPTO,
                            exchange  = "bybit",
                        )

    async def stream_orderbook(self, symbol: str, depth: int = 20) -> AsyncGenerator[OrderBook, None]:
        sym      = self.normalize_symbol(symbol)
        category = _category(symbol)
        url      = f"{WS_BASE}/{category}"

        async with websockets.connect(url, ping_interval=20) as ws:
            await ws.send(json.dumps({
                "op": "subscribe",
                "args": [f"orderbook.{depth}.{sym}"],
            }))
            async for raw in ws:
                msg = json.loads(raw)
                if msg.get("topic", "").startswith("orderbook"):
                    d  = msg.get("data", {})
                    ts = float(msg.get("ts", time.time() * 1000))
                    bids = [
                        OrderBookLevel(ts, float(p), float(s), Side.BUY,  symbol, Market.CRYPTO)
                        for p, s in d.get("b", [])
                    ]
                    asks = [
                        OrderBookLevel(ts, float(p), float(s), Side.SELL, symbol, Market.CRYPTO)
                        for p, s in d.get("a", [])
                    ]
                    bids.sort(key=lambda x: -x.price)
                    asks.sort(key=lambda x:  x.price)
                    yield OrderBook(
                        timestamp = ts,
                        symbol    = symbol,
                        market    = Market.CRYPTO,
                        bids      = bids,
                        asks      = asks,
                    )

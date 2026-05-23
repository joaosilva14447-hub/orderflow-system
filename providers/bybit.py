"""
Bybit provider — globally accessible, no geo-restrictions.
Supports Spot, Linear Perpetuals, Inverse Futures.
No API key required for public market data.
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

# Symbol → category mapping heuristic
def _category(symbol: str) -> str:
    sym = symbol.replace("/", "").upper()
    if sym.endswith("USDT") or sym.endswith("USDC"):
        return "linear"   # perpetual futures (most liquid)
    if sym.endswith("USD"):
        return "inverse"
    return "spot"


def _parse_klines(raw: list, symbol: str, timeframe: str) -> list[Bar]:
    """
    Bybit returns: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
    Data is newest-first → reverse.
    """
    bars = []
    for k in reversed(raw):
        if not isinstance(k, list) or len(k) < 6:
            continue
        vol = float(k[5])
        bars.append(Bar(
            timestamp   = float(k[0]),
            open        = float(k[1]),
            high        = float(k[2]),
            low         = float(k[3]),
            close       = float(k[4]),
            volume      = vol,
            symbol      = symbol,
            market      = Market.CRYPTO,
            timeframe   = timeframe,
            buy_volume  = vol * 0.5,   # Bybit klines have no taker split — approximated
            sell_volume = vol * 0.5,
        ))
    return bars


class BybitProvider(BaseProvider):

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").upper()

    # ── Sync ──────────────────────────────────────────────────────────────
    def fetch_bars_sync(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> list[Bar]:
        sym      = self.normalize_symbol(symbol)
        tf       = TF_MAP.get(timeframe, "5")
        category = _category(symbol)
        params   = {"category": category, "symbol": sym, "interval": tf, "limit": min(limit, 1000)}
        if start:
            params["start"] = int(start)
        if end:
            params["end"] = int(end)

        resp = requests.get(f"{REST_BASE}/kline", params=params, timeout=10, verify=certifi.where())
        data = resp.json()

        if data.get("retCode", -1) != 0:
            raise ValueError(f"Bybit API error: {data.get('retMsg', data)}")

        raw = data["result"]["list"]
        return _parse_klines(raw, symbol, timeframe)

    # ── Async ─────────────────────────────────────────────────────────────
    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> list[Bar]:
        sym      = self.normalize_symbol(symbol)
        tf       = TF_MAP.get(timeframe, "5")
        category = _category(symbol)
        params   = {"category": category, "symbol": sym, "interval": tf, "limit": min(limit, 1000)}

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{REST_BASE}/kline", params=params) as resp:
                data = await resp.json()

        if data.get("retCode", -1) != 0:
            raise ValueError(f"Bybit API error: {data.get('retMsg', data)}")

        return _parse_klines(data["result"]["list"], symbol, timeframe)

    async def stream_trades(self, symbol: str) -> AsyncGenerator[Trade, None]:
        sym      = self.normalize_symbol(symbol)
        category = _category(symbol)
        url      = f"{WS_BASE}/{category}"

        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({
                "op": "subscribe",
                "args": [f"publicTrade.{sym}"]
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

        async with websockets.connect(url) as ws:
            await ws.send(json.dumps({
                "op": "subscribe",
                "args": [f"orderbook.{depth}.{sym}"]
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

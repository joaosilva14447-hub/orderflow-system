"""
Binance provider — real tick data with true aggressor side.
Uses public WebSocket API (no API key required for trades + orderbook).
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

WS_BASE       = "wss://stream.binance.com:9443/ws"
REST_ENDPOINTS = [
    "https://api.binance.com/api/v3",   # global
    "https://api.binance.us/api/v3",    # US fallback
]

TF_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d",
}


def _parse_klines(data: list, symbol: str, timeframe: str) -> list[Bar]:
    """Parse raw Binance kline list into Bar objects."""
    bars = []
    for k in data:
        if not isinstance(k, list) or len(k) < 10:
            continue
        vol      = float(k[5])
        taker    = float(k[9])
        bars.append(Bar(
            timestamp  = float(k[0]),
            open       = float(k[1]),
            high       = float(k[2]),
            low        = float(k[3]),
            close      = float(k[4]),
            volume     = vol,
            symbol     = symbol,
            market     = Market.CRYPTO,
            timeframe  = timeframe,
            buy_volume = taker,
            sell_volume= vol - taker,
        ))
    return bars


class BinanceProvider(BaseProvider):

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").upper()

    # ── Sync (used by dashboard) ───────────────────────────────────────────
    def fetch_bars_sync(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> list[Bar]:
        sym    = self.normalize_symbol(symbol)
        tf     = TF_MAP.get(timeframe, "1m")
        params = {"symbol": sym, "interval": tf, "limit": limit}
        if start:
            params["startTime"] = int(start)
        if end:
            params["endTime"] = int(end)

        last_error = "No endpoints reachable"
        for base in REST_ENDPOINTS:
            try:
                resp = requests.get(f"{base}/klines", params=params, timeout=10, verify=certifi.where())
                data = resp.json()
                if isinstance(data, list):
                    return _parse_klines(data, symbol, timeframe)
                last_error = data.get("msg", str(data))
            except Exception as e:
                last_error = str(e)

        raise ValueError(f"Binance API error: {last_error}")

    # ── Async (used by live streaming) ────────────────────────────────────
    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> list[Bar]:
        sym    = self.normalize_symbol(symbol)
        tf     = TF_MAP.get(timeframe, "1m")
        params = {"symbol": sym, "interval": tf, "limit": limit}
        if start:
            params["startTime"] = int(start)
        if end:
            params["endTime"] = int(end)

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{REST_BASE}/klines", params=params) as resp:
                data = await resp.json()

        if isinstance(data, dict):
            raise ValueError(f"Binance API error: {data.get('msg', data)}")

        return _parse_klines(data, symbol, timeframe)

    async def stream_trades(self, symbol: str) -> AsyncGenerator[Trade, None]:
        sym = self.normalize_symbol(symbol).lower()
        url = f"{WS_BASE}/{sym}@aggTrade"
        async with websockets.connect(url) as ws:
            async for raw in ws:
                msg = json.loads(raw)
                yield Trade(
                    timestamp = float(msg["T"]),
                    price     = float(msg["p"]),
                    volume    = float(msg["q"]),
                    side      = Side.SELL if msg["m"] else Side.BUY,
                    symbol    = symbol,
                    market    = Market.CRYPTO,
                    exchange  = "binance",
                )

    async def stream_orderbook(self, symbol: str, depth: int = 20) -> AsyncGenerator[OrderBook, None]:
        sym = self.normalize_symbol(symbol).lower()
        url = f"{WS_BASE}/{sym}@depth{depth}@100ms"
        async with websockets.connect(url) as ws:
            async for raw in ws:
                msg = json.loads(raw)
                ts  = time.time() * 1000
                bids = [
                    OrderBookLevel(ts, float(p), float(s), Side.BUY,  symbol, Market.CRYPTO)
                    for p, s in msg.get("b", [])
                ]
                asks = [
                    OrderBookLevel(ts, float(p), float(s), Side.SELL, symbol, Market.CRYPTO)
                    for p, s in msg.get("a", [])
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

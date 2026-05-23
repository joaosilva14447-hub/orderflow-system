"""
Binance provider — real tick data with true aggressor side.
Uses public WebSocket API (no API key required for trades + orderbook).
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Optional

import aiohttp
import websockets

from core.models import Bar, Market, OrderBook, OrderBookLevel, Side, Trade
from providers.base import BaseProvider

WS_BASE = "wss://stream.binance.com:9443/ws"
REST_BASE = "https://api.binance.com/api/v3"

TF_MAP = {
    "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d",
}


class BinanceProvider(BaseProvider):

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").upper()

    async def stream_trades(self, symbol: str) -> AsyncGenerator[Trade, None]:
        sym = self.normalize_symbol(symbol).lower()
        url = f"{WS_BASE}/{sym}@aggTrade"
        async with websockets.connect(url) as ws:
            async for raw in ws:
                msg = json.loads(raw)
                yield Trade(
                    timestamp=float(msg["T"]),
                    price=float(msg["p"]),
                    volume=float(msg["q"]),
                    side=Side.SELL if msg["m"] else Side.BUY,  # m=True → market maker = sell
                    symbol=symbol,
                    market=Market.CRYPTO,
                    exchange="binance",
                )

    async def stream_orderbook(self, symbol: str, depth: int = 20) -> AsyncGenerator[OrderBook, None]:
        sym = self.normalize_symbol(symbol).lower()
        url = f"{WS_BASE}/{sym}@depth{depth}@100ms"
        async with websockets.connect(url) as ws:
            async for raw in ws:
                msg = json.loads(raw)
                ts = time.time() * 1000
                bids = [
                    OrderBookLevel(ts, float(p), float(s), Side.BUY, symbol, Market.CRYPTO)
                    for p, s in msg.get("b", [])
                ]
                asks = [
                    OrderBookLevel(ts, float(p), float(s), Side.SELL, symbol, Market.CRYPTO)
                    for p, s in msg.get("a", [])
                ]
                bids.sort(key=lambda x: -x.price)
                asks.sort(key=lambda x: x.price)
                yield OrderBook(
                    timestamp=ts,
                    symbol=symbol,
                    market=Market.CRYPTO,
                    bids=bids,
                    asks=asks,
                )

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> list[Bar]:
        sym = self.normalize_symbol(symbol)
        tf = TF_MAP.get(timeframe, "1m")
        params = {"symbol": sym, "interval": tf, "limit": limit}
        if start:
            params["startTime"] = int(start)
        if end:
            params["endTime"] = int(end)

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{REST_BASE}/klines", params=params) as resp:
                data = await resp.json()

        bars = []
        for k in data:
            vol = float(k[5])
            taker_buy = float(k[9])
            buy_vol = taker_buy
            sell_vol = vol - taker_buy
            bars.append(Bar(
                timestamp=float(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=vol,
                symbol=symbol,
                market=Market.CRYPTO,
                timeframe=timeframe,
                buy_volume=buy_vol,
                sell_volume=sell_vol,
            ))
        return bars

"""
Alpaca provider — US Stocks & Crypto (free tier).
Requires: ALPACA_API_KEY + ALPACA_SECRET_KEY in config.
Real aggressor side available via tape conditions.
"""

import asyncio
import json
import os
import time
from typing import AsyncGenerator, Optional

import aiohttp
import websockets

from core.models import Bar, Market, OrderBook, OrderBookLevel, Side, Trade
from providers.base import BaseProvider

WS_STOCKS = "wss://stream.data.alpaca.markets/v2/iex"
WS_CRYPTO = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
REST_BASE  = "https://data.alpaca.markets/v2"

# Tape conditions that indicate a buy aggressor
BUY_CONDITIONS = {"@", "F", "I", "M", "Q", "R", "W"}


def _infer_side(conditions: list[str]) -> Side:
    for c in conditions:
        if c in BUY_CONDITIONS:
            return Side.BUY
    return Side.UNKNOWN


class AlpacaProvider(BaseProvider):

    def __init__(self, api_key: str = "", secret_key: str = ""):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY", "")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "")
        self._headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.replace("/", "").upper()

    async def _auth(self, ws) -> None:
        await ws.send(json.dumps({
            "action": "auth",
            "key": self.api_key,
            "secret": self.secret_key,
        }))
        await ws.recv()

    async def stream_trades(self, symbol: str) -> AsyncGenerator[Trade, None]:
        is_crypto = "/" in symbol
        url = WS_CRYPTO if is_crypto else WS_STOCKS
        sym = symbol if is_crypto else self.normalize_symbol(symbol)
        mkt = Market.CRYPTO if is_crypto else Market.STOCK

        async with websockets.connect(url) as ws:
            await ws.recv()  # connected msg
            await self._auth(ws)
            await ws.send(json.dumps({"action": "subscribe", "trades": [sym]}))
            await ws.recv()

            async for raw in ws:
                msgs = json.loads(raw)
                for msg in msgs:
                    if msg.get("T") != "t":
                        continue
                    conditions = msg.get("c", [])
                    side = _infer_side(conditions)
                    yield Trade(
                        timestamp=float(
                            msg.get("t", "").replace("Z", "+00:00")
                            if isinstance(msg.get("t"), str)
                            else time.time() * 1000
                        ),
                        price=float(msg["p"]),
                        volume=float(msg["s"]),
                        side=side,
                        symbol=symbol,
                        market=mkt,
                        exchange="alpaca",
                    )

    async def stream_orderbook(self, symbol: str, depth: int = 20) -> AsyncGenerator[OrderBook, None]:
        # Alpaca free tier does not provide L2 order book streaming
        # Yields empty books — caller should handle gracefully
        while True:
            yield OrderBook(
                timestamp=time.time() * 1000,
                symbol=symbol,
                market=Market.STOCK,
                bids=[],
                asks=[],
            )
            await asyncio.sleep(1)

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> list[Bar]:
        is_crypto = "/" in symbol
        sym = symbol if is_crypto else self.normalize_symbol(symbol)
        mkt = Market.CRYPTO if is_crypto else Market.STOCK
        endpoint = "crypto/us/bars" if is_crypto else f"stocks/{sym}/bars"

        tf_map = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour", "1d": "1Day"}
        tf = tf_map.get(timeframe, "1Min")

        params = {"timeframe": tf, "limit": limit, "feed": "iex"}
        if is_crypto:
            params["symbols"] = sym

        async with aiohttp.ClientSession(headers=self._headers) as session:
            async with session.get(f"{REST_BASE}/{endpoint}", params=params) as resp:
                data = await resp.json()

        raw_bars = data.get("bars", {})
        if isinstance(raw_bars, dict):
            raw_bars = raw_bars.get(sym, [])

        bars = []
        for b in raw_bars:
            vol = float(b.get("v", 0))
            bars.append(Bar(
                timestamp=float(b.get("t", 0)),
                open=float(b["o"]),
                high=float(b["h"]),
                low=float(b["l"]),
                close=float(b["c"]),
                volume=vol,
                symbol=symbol,
                market=mkt,
                timeframe=timeframe,
                buy_volume=vol * 0.5,
                sell_volume=vol * 0.5,
            ))
        return bars

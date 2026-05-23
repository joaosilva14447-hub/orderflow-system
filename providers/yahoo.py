"""
Yahoo Finance provider — OHLCV fallback (no tick data).
Used for stocks, ETFs, commodities, forex when no premium feed available.
Aggressor side approximated via price action (not reliable for orderflow).
"""

import time
from typing import AsyncGenerator, Optional

import aiohttp

from core.models import Bar, Market, OrderBook, Side, Trade
from providers.base import BaseProvider

BASE = "https://query1.finance.yahoo.com/v8/finance/chart"

TF_MAP = {
    "1m": "1m", "2m": "2m", "5m": "5m", "15m": "15m",
    "30m": "30m", "1h": "1h", "1d": "1d", "1wk": "1wk",
}

_MARKET_MAP = {
    "=F": Market.FUTURES,
    "=X": Market.FOREX,
    "^": Market.STOCK,
}


def _detect_market(symbol: str) -> Market:
    if "=F" in symbol:
        return Market.FUTURES
    if "=X" in symbol:
        return Market.FOREX
    if symbol.startswith("^"):
        return Market.STOCK
    return Market.STOCK


def _approx_side(open_: float, close: float) -> Side:
    """Rough approximation — not suitable for real orderflow analysis."""
    if close > open_:
        return Side.BUY
    if close < open_:
        return Side.SELL
    return Side.UNKNOWN


class YahooProvider(BaseProvider):

    def normalize_symbol(self, symbol: str) -> str:
        return symbol.upper()

    async def stream_trades(self, symbol: str) -> AsyncGenerator[Trade, None]:
        raise NotImplementedError("Yahoo Finance does not provide real-time tick data.")

    async def stream_orderbook(self, symbol: str, depth: int = 20) -> AsyncGenerator[OrderBook, None]:
        raise NotImplementedError("Yahoo Finance does not provide order book data.")

    async def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> list[Bar]:
        sym = self.normalize_symbol(symbol)
        tf = TF_MAP.get(timeframe, "1d")
        mkt = _detect_market(sym)

        params = {"interval": tf, "range": "1y"}
        if start:
            params["period1"] = int(start / 1000)
            params["period2"] = int((end or time.time() * 1000) / 1000)
            del params["range"]

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{BASE}/{sym}",
                params=params,
                headers={"User-Agent": "Mozilla/5.0"},
            ) as resp:
                data = await resp.json()

        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        ohlcv = result["indicators"]["quote"][0]

        bars = []
        for i, ts in enumerate(timestamps):
            o = ohlcv["open"][i]
            h = ohlcv["high"][i]
            l = ohlcv["low"][i]
            c = ohlcv["close"][i]
            v = ohlcv["volume"][i] or 0

            if o is None or c is None:
                continue

            side = _approx_side(o, c)
            buy_vol = v if side == Side.BUY else 0.0
            sell_vol = v if side == Side.SELL else 0.0

            bars.append(Bar(
                timestamp=float(ts) * 1000,
                open=o, high=h, low=l, close=c,
                volume=float(v),
                symbol=symbol,
                market=mkt,
                timeframe=timeframe,
                buy_volume=buy_vol,
                sell_volume=sell_vol,
            ))

        return bars[-limit:]

"""
Yahoo Finance provider via direct REST API — bypasses yfinance library.
Works on any server. No auth, no geo-restrictions, no rate-limit issues.
Covers: Crypto (BTC-USD), Stocks (SPY), Futures (GC=F, CL=F, ES=F), Forex.
"""

import time
import datetime
from typing import AsyncGenerator, Optional

from core.ssl_fix import apply as _ssl_fix; _ssl_fix()
import requests
import pandas as pd

from core.models import Bar, Market, OrderBook, Side, Trade
from providers.base import BaseProvider

BASE = "https://query1.finance.yahoo.com/v8/finance/chart"
FALLBACK = "https://query2.finance.yahoo.com/v8/finance/chart"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://finance.yahoo.com/",
}

SYMBOL_MAP = {
    "BTC/USDT": "BTC-USD", "ETH/USDT": "ETH-USD",
    "SOL/USDT": "SOL-USD", "BNB/USDT": "BNB-USD",
    "BTC/USD":  "BTC-USD", "ETH/USD":  "ETH-USD",
}

TF_MAP = {
    "1m": "1m",  "2m": "2m",  "5m": "5m",  "15m": "15m",
    "30m": "30m","1h": "1h",  "4h": "4h",  "1d": "1d",
}

PERIOD_MAP = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d",
    "30m": "60d","1h": "2y", "4h": "2y",  "1d": "10y",
}


def _yf_sym(symbol: str) -> str:
    return SYMBOL_MAP.get(symbol, symbol)

def _detect_market(symbol: str) -> Market:
    s = symbol.upper()
    if "=F" in s: return Market.FUTURES
    if "=X" in s: return Market.FOREX
    if "-USD" in s or "/USD" in s or "/USDT" in s: return Market.CRYPTO
    return Market.STOCK

def _approx_buy_sell(o: float, c: float, vol: float):
    if c > o:
        buy = vol * min(0.5 + (c - o) / (o + 1e-9) * 10, 0.8)
    elif c < o:
        buy = vol * max(0.5 - (o - c) / (o + 1e-9) * 10, 0.2)
    else:
        buy = vol * 0.5
    return buy, vol - buy

def _fetch_raw(sym: str, interval: str, period: str = None,
               start: int = None, end: int = None) -> dict:
    """Fetch from Yahoo Finance with fallback URL and retries."""
    params = {"interval": interval}
    if start and end:
        params["period1"] = start
        params["period2"] = end
    else:
        params["range"] = period or "1y"

    last_err = None
    for base in (BASE, FALLBACK):
        for attempt in range(3):
            try:
                r = requests.get(
                    f"{base}/{sym}", params=params,
                    headers=HEADERS, timeout=15
                )
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                result = data.get("chart", {}).get("result")
                if result:
                    return result[0]
                err = data.get("chart", {}).get("error", {})
                last_err = err.get("description", "No data returned") if err else "Empty result"
            except Exception as exc:
                last_err = str(exc)
                time.sleep(1)

    raise ValueError(f"Yahoo Finance error for {sym}: {last_err}")


def _parse_result(result: dict, symbol: str, timeframe: str, limit: int) -> list[Bar]:
    market = _detect_market(symbol)
    timestamps = result.get("timestamp", [])
    quote = result["indicators"]["quote"][0]
    opens   = quote.get("open",   [])
    highs   = quote.get("high",   [])
    lows    = quote.get("low",    [])
    closes  = quote.get("close",  [])
    volumes = quote.get("volume", [])

    bars = []
    for i in range(len(timestamps)):
        o = opens[i]
        h = highs[i]
        l = lows[i]
        c = closes[i]
        v = volumes[i]
        if o is None or c is None:
            continue
        v = float(v or 0)
        buy, sell = _approx_buy_sell(float(o), float(c), v)
        bars.append(Bar(
            timestamp   = float(timestamps[i]) * 1000,
            open        = float(o),
            high        = float(h) if h else float(o),
            low         = float(l) if l else float(o),
            close       = float(c),
            volume      = v,
            symbol      = symbol,
            market      = market,
            timeframe   = timeframe,
            buy_volume  = buy,
            sell_volume = sell,
        ))

    return bars[-limit:]


class YFinanceProvider(BaseProvider):

    def normalize_symbol(self, symbol: str) -> str:
        return _yf_sym(symbol)

    def fetch_bars_sync(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
        start: Optional[float] = None,
        end: Optional[float] = None,
    ) -> list[Bar]:
        sym      = _yf_sym(symbol)
        interval = TF_MAP.get(timeframe, "1d")
        period   = PERIOD_MAP.get(timeframe, "1y")

        if start:
            result = _fetch_raw(sym, interval,
                                start=int(start / 1000),
                                end=int((end or time.time() * 1000) / 1000))
        else:
            result = _fetch_raw(sym, interval, period=period)

        return _parse_result(result, symbol, timeframe, limit)

    async def fetch_bars(self, symbol, timeframe, limit=500, start=None, end=None):
        return self.fetch_bars_sync(symbol, timeframe, limit, start, end)

    async def stream_trades(self, symbol: str) -> AsyncGenerator[Trade, None]:
        raise NotImplementedError("Yahoo Finance does not support real-time streaming.")

    async def stream_orderbook(self, symbol: str, depth: int = 20) -> AsyncGenerator[OrderBook, None]:
        raise NotImplementedError("Yahoo Finance does not support order book data.")

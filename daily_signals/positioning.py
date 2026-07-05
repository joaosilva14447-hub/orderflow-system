"""Positioning / Dry Powder pillar — a from-scratch 0-100 fuel & sentiment score.

Three orthogonal, free-to-source indicators (no API key):
  1. Stablecoin momentum — total USD-pegged supply 30d growth (DefiLlama).
     Growing dry powder = fuel accumulating = bullish.
  2. SSR — BTC market cap / stablecoin supply (DefiLlama + Coin Metrics).
     Low SSR = lots of buying power vs price = bullish (inverted).
  3. Funding — BTC perp funding rate (Binance). Low/negative = crowded shorts /
     capitulated leverage = contrarian bullish (inverted).

Each -> rolling-percentile 0-100 oriented so HIGH = loaded (bullish), LOW =
greedy/empty. Equal weights, no tuned thresholds.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

LLAMA_URL = "https://stablecoins.llama.fi/stablecoincharts/all"
CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
_UA = "Mozilla/5.0 (compatible; PositioningBot/1.0; +daily_signals)"

PCT_WINDOW = 1461   # ~4y
PCT_MIN = 200


def _apply_ssl_fix() -> None:
    if sys.platform != "win32":
        return
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


_apply_ssl_fix()


@dataclass
class PositioningResult:
    score: float                 # 0-100 (high = loaded/bullish)
    zone: str                    # empty | neutral | loaded
    components: dict[str, float]
    asof: str

    @property
    def is_loaded(self) -> bool:
        return self.zone == "loaded"


def _get(url: str, **kw) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.get(url, headers={"User-Agent": _UA}, timeout=40, **kw)
            r.raise_for_status()
            return r
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise last_exc if last_exc else RuntimeError(f"no response: {url}")


def _fetch_stablecoins() -> pd.Series:
    data = _get(LLAMA_URL).json()
    dts, vals = [], []
    for row in data:
        try:
            dts.append(datetime.fromtimestamp(int(row["date"]), tz=timezone.utc).date())
            vals.append(float(row["totalCirculating"]["peggedUSD"]
                             if isinstance(row.get("totalCirculating"), dict)
                             else row["peggedUSD"]))
        except (KeyError, TypeError, ValueError):
            continue
    s = pd.Series(vals, index=pd.to_datetime(dts)).sort_index()
    return s[~s.index.duplicated(keep="last")]


def _fetch_btc_mcap() -> pd.Series:
    r = _get(CM_URL, params={"assets": "btc", "metrics": "CapMrktCurUSD",
                             "frequency": "1d", "page_size": 10000,
                             "start_time": "2015-01-01"})
    rows = r.json().get("data", [])
    dts = pd.to_datetime([x["time"] for x in rows]).tz_localize(None).normalize()
    vals = [float(x["CapMrktCurUSD"]) for x in rows]
    return pd.Series(vals, index=dts).sort_index()


def _fetch_funding() -> pd.Series:
    """Paginate Binance funding history (~5y) -> daily mean funding rate."""
    start = int((datetime.now(timezone.utc) - timedelta(days=1900)).timestamp() * 1000)
    rows: list[dict] = []
    for _ in range(12):
        data = _get(BINANCE_FUNDING, params={"symbol": "BTCUSDT",
                    "startTime": start, "limit": 1000}).json()
        if not data:
            break
        rows.extend(data)
        last = data[-1]["fundingTime"]
        if len(data) < 1000:
            break
        start = last + 1
    if not rows:
        raise RuntimeError("funding: no data")
    dts = pd.to_datetime([r["fundingTime"] for r in rows], unit="ms").normalize()
    rates = [float(r["fundingRate"]) for r in rows]
    s = pd.Series(rates, index=dts)
    return s.groupby(s.index).mean().sort_index()  # daily mean (3 funds/day)


def _rolling_pct(s: pd.Series) -> pd.Series:
    def pct_of_last(a: np.ndarray) -> float:
        return 100.0 * float((a <= a[-1]).sum()) / len(a)
    return s.rolling(PCT_WINDOW, min_periods=PCT_MIN).apply(pct_of_last, raw=True)


def compute() -> PositioningResult:
    sc = _fetch_stablecoins()
    mcap = _fetch_btc_mcap()
    funding = _fetch_funding()

    idx = pd.date_range(min(sc.index.min(), mcap.index.min()),
                        max(sc.index.max(), mcap.index.max()), freq="D")
    sc = sc.reindex(idx).ffill()
    mcap = mcap.reindex(idx).ffill()
    funding = funding.reindex(idx).ffill()

    raw = pd.DataFrame(index=idx)
    raw["sc_mom"] = sc.pct_change(30)          # + growth = bullish
    raw["ssr"] = -(mcap / sc)                   # low SSR = bullish (inverted)
    raw["funding"] = -funding                   # low funding = bullish (inverted)

    pct = pd.DataFrame(index=idx)
    for col in ("sc_mom", "ssr", "funding"):
        pct[col] = _rolling_pct(raw[col])
    pct = pct.dropna()
    if pct.empty:
        raise RuntimeError("positioning: not enough history")

    latest = pct.iloc[-1]
    components = {k: round(float(v), 1) for k, v in latest.items()}
    score = round(float(latest.mean()), 1)
    zone = "empty" if score < 35 else "neutral" if score < 65 else "loaded"
    result = PositioningResult(score=score, zone=zone, components=components,
                               asof=pct.index[-1].date().isoformat())
    print(f"[positioning] score={score} zone={zone} "
          f"components={components} asof={result.asof}")
    return result


if __name__ == "__main__":
    compute()

"""SDCA Valuation pillar — a from-scratch 0-100 cycle-valuation score.

Four orthogonal, free-to-source indicators (Coin Metrics Community API, no key):
  1. MVRV Z-Score      — cost-basis extremes (market cap vs realized cap)
  2. Mayer Multiple     — price vs its 200d mean (trend mean-reversion)
  3. Puell Multiple     — miner issuance USD vs its 365d mean (supply side)
  4. Metcalfe ratio     — market cap per active address (network utility)

Each raw indicator is turned into a rolling-percentile 0-100 (window ~4y, past
data only -> no lookahead, decay-resistant). Equal weights, no tuned thresholds:
this is deliberately anti-overfit. High score = expensive, low = cheap.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import requests

CM_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
FREE_METRICS = ["CapMVRVCur", "CapMrktCurUSD", "PriceUSD", "IssTotUSD", "AdrActCnt"]

# Rolling normalization window (~4 years of daily data) and warm-up minimum.
PCT_WINDOW = 1461
PCT_MIN = 365


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
class ValuationResult:
    score: float                 # composite 0-100 (high = expensive)
    zone: str                    # deep_value | cheap | fair | expensive
    components: dict[str, float]  # per-indicator percentile 0-100
    asof: str                    # ISO date of the latest data point

    @property
    def is_cheap(self) -> bool:
        """Gate helper: is valuation in an accumulation zone?"""
        return self.zone in ("deep_value", "cheap")


def fetch_metrics(start: str = "2011-01-01") -> pd.DataFrame:
    """Fetch the free metric set into a date-indexed DataFrame (retries + paging)."""
    frames: dict[str, pd.Series] = {}
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            rows: list[dict] = []
            params = {
                "assets": "btc", "metrics": ",".join(FREE_METRICS),
                "frequency": "1d", "page_size": 10000, "start_time": start,
            }
            url = CM_URL
            while url:
                r = requests.get(url, params=params, timeout=40)
                r.raise_for_status()
                payload = r.json()
                rows.extend(payload.get("data", []))
                url = payload.get("next_page_url")
                params = None  # next_page_url already carries the query
            if len(rows) < 500:
                raise ValueError(f"too few rows: {len(rows)}")

            df = pd.DataFrame(rows)
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None).dt.normalize()
            df = df.set_index("time").sort_index()
            for m in FREE_METRICS:
                frames[m] = pd.to_numeric(df.get(m), errors="coerce")
            out = pd.DataFrame(frames).dropna(how="all")
            print(f"[valuation] fetched {len(out)} rows, "
                  f"{out.index.min().date()}..{out.index.max().date()}")
            return out
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"[valuation] fetch attempt {attempt + 1} failed: {exc}")
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise last_exc if last_exc else RuntimeError("Coin Metrics no response")


def _rolling_pct(s: pd.Series, window: int = PCT_WINDOW,
                 min_periods: int = PCT_MIN) -> pd.Series:
    """Percentile rank (0-100) of each point within its trailing window.

    Uses only past+current data (the window ends at the current point), so there
    is no lookahead. NaN until `min_periods` observations exist.
    """
    def pct_of_last(a: np.ndarray) -> float:
        return 100.0 * float((a <= a[-1]).sum()) / len(a)

    return s.rolling(window, min_periods=min_periods).apply(pct_of_last, raw=True)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 4 raw indicators, then normalize each to a rolling percentile."""
    out = pd.DataFrame(index=df.index)

    # 1) MVRV Z-Score: realized cap derived as market cap / MVRV.
    mvrv = df["CapMVRVCur"]
    realized = df["CapMrktCurUSD"] / mvrv
    excess = df["CapMrktCurUSD"] - realized
    mcap_std = df["CapMrktCurUSD"].rolling(PCT_WINDOW, min_periods=PCT_MIN).std()
    out["mvrv_z"] = excess / mcap_std

    # 2) Mayer Multiple: price / 200d SMA.
    out["mayer"] = df["PriceUSD"] / df["PriceUSD"].rolling(200, min_periods=100).mean()

    # 3) Puell Multiple: daily issuance USD / 365d SMA of it.
    iss = df["IssTotUSD"]
    out["puell"] = iss / iss.rolling(365, min_periods=180).mean()

    # 4) Metcalfe ratio: market cap per active address (7d-smoothed addresses).
    adr = df["AdrActCnt"].rolling(7, min_periods=3).mean()
    out["metcalfe"] = df["CapMrktCurUSD"] / adr

    # Normalize each to a rolling percentile (high = expensive).
    pct = pd.DataFrame(index=df.index)
    for col in ("mvrv_z", "mayer", "puell", "metcalfe"):
        pct[col] = _rolling_pct(out[col])
    return pct


def _zone(score: float) -> str:
    if score < 20:
        return "deep_value"
    if score < 40:
        return "cheap"
    if score < 75:
        return "fair"
    return "expensive"


def compute() -> ValuationResult:
    df = fetch_metrics()
    pct = compute_indicators(df).dropna()
    if pct.empty:
        raise RuntimeError("valuation: not enough history to normalize")

    latest = pct.iloc[-1]
    components = {k: round(float(v), 1) for k, v in latest.items()}
    score = round(float(latest.mean()), 1)  # equal weights
    result = ValuationResult(
        score=score, zone=_zone(score), components=components,
        asof=pct.index[-1].date().isoformat(),
    )
    print(f"[valuation] score={score} zone={result.zone} "
          f"components={components} asof={result.asof}")
    return result


if __name__ == "__main__":
    compute()

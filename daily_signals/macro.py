"""Macro Liquidity pillar — a from-scratch 0-100 risk-regime score.

Three orthogonal, free-to-source indicators (FRED, no API key — CSV endpoint):
  1. Net Liquidity momentum — Fed balance sheet minus RRP minus TGA, rate of
     change (expanding liquidity = tailwind for risk assets)
  2. Dollar momentum       — broad USD index (a rising dollar is a headwind)
  3. Credit stress          — US High-Yield OAS (tight spreads = risk-on)

Each is turned into a rolling-percentile 0-100 oriented so HIGH = risk-on
tailwind, LOW = risk-off headwind. Equal weights, no tuned thresholds.
This pillar answers *when* (regime), not *how cheap*.
"""

from __future__ import annotations

import io
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
_UA = "Mozilla/5.0 (compatible; MacroBot/1.0; +daily_signals)"

# FRED series ids.
S_WALCL = "WALCL"          # Fed total assets, millions USD (weekly)
S_RRP = "RRPONTSYD"        # Overnight reverse repo, billions USD (daily)
S_TGA = "WTREGEN"          # Treasury General Account, billions USD (weekly)
S_DXY = "DTWEXBGS"         # Broad USD index (daily)
S_OAS = "BAMLH0A0HYM2"     # US HY OAS, percent (daily)

MOM_DAYS = 90              # momentum lookback (calendar days, on ffilled daily)
PCT_WINDOW = 1461          # ~4y rolling normalization
PCT_MIN = 252


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
class MacroResult:
    score: float                 # composite 0-100 (high = risk-on tailwind)
    zone: str                    # risk_off | neutral | risk_on
    components: dict[str, float]  # per-indicator percentile 0-100
    asof: str

    @property
    def is_tailwind(self) -> bool:
        return self.zone == "risk_on"


def _fetch_fred(sid: str) -> pd.Series:
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            r = requests.get(FRED_CSV.format(sid=sid),
                             headers={"User-Agent": _UA}, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"]).dt.normalize()
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            s = df.dropna().set_index("date")["value"]
            if len(s) < 100:
                raise ValueError(f"{sid}: too few rows ({len(s)})")
            return s
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"[macro] fetch {sid} attempt {attempt + 1} failed: {exc}")
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise last_exc if last_exc else RuntimeError(f"FRED {sid} no response")


def _rolling_pct(s: pd.Series) -> pd.Series:
    def pct_of_last(a: np.ndarray) -> float:
        return 100.0 * float((a <= a[-1]).sum()) / len(a)
    return s.rolling(PCT_WINDOW, min_periods=PCT_MIN).apply(pct_of_last, raw=True)


def fetch_frame() -> pd.DataFrame:
    """Fetch all series and forward-fill onto a common daily index."""
    series = {
        "walcl": _fetch_fred(S_WALCL) / 1000.0,  # millions -> billions
        "rrp": _fetch_fred(S_RRP),               # billions
        "tga": _fetch_fred(S_TGA),               # billions
        "dxy": _fetch_fred(S_DXY),
        "oas": _fetch_fred(S_OAS),
    }
    idx = pd.date_range(
        min(s.index.min() for s in series.values()),
        max(s.index.max() for s in series.values()),
        freq="D",
    )
    df = pd.DataFrame({k: v.reindex(idx).ffill() for k, v in series.items()})
    print(f"[macro] frame {df.index.min().date()}..{df.index.max().date()}")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    net_liq = df["walcl"] - df["rrp"] - df["tga"]

    raw = pd.DataFrame(index=df.index)
    raw["liq_mom"] = net_liq.diff(MOM_DAYS)          # + = expanding (bullish)
    raw["usd_mom"] = -df["dxy"].diff(MOM_DAYS)       # + = weakening USD (bullish)
    raw["credit"] = -df["oas"]                        # + = tighter spreads (bullish)

    pct = pd.DataFrame(index=df.index)
    for col in ("liq_mom", "usd_mom", "credit"):
        pct[col] = _rolling_pct(raw[col])            # high = risk-on
    return pct


def _zone(score: float) -> str:
    if score < 35:
        return "risk_off"
    if score < 65:
        return "neutral"
    return "risk_on"


def compute() -> MacroResult:
    df = fetch_frame()
    pct = compute_indicators(df).dropna()
    if pct.empty:
        raise RuntimeError("macro: not enough history to normalize")

    latest = pct.iloc[-1]
    components = {k: round(float(v), 1) for k, v in latest.items()}
    score = round(float(latest.mean()), 1)
    result = MacroResult(score=score, zone=_zone(score), components=components,
                         asof=pct.index[-1].date().isoformat())
    print(f"[macro] score={score} zone={result.zone} "
          f"components={components} asof={result.asof}")
    return result


if __name__ == "__main__":
    compute()

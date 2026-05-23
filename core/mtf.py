"""
Multi-Timeframe CVD — confluence across timeframes.
Uses the same YFinanceProvider to fetch each TF independently.
"""

from dataclasses import dataclass
from core.engine import CVDEngine
from core.models import Bar


TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

TF_LABELS = {
    "1m": "1M", "5m": "5M", "15m": "15M",
    "30m": "30M", "1h": "1H", "4h": "4H", "1d": "1D",
}


@dataclass
class MTFSnapshot:
    timeframe:  str
    cvd:        float
    last_delta: float
    direction:  str          # "bullish" | "bearish" | "neutral"
    bars_used:  int


def _direction(cvd: float, delta: float) -> str:
    if cvd > 0 and delta > 0:
        return "bullish"
    if cvd < 0 and delta < 0:
        return "bearish"
    if cvd > 0 or delta > 0:
        return "bullish"
    if cvd < 0 or delta < 0:
        return "bearish"
    return "neutral"


def calculate_mtf_cvd(
    symbol:     str,
    timeframes: list[str] = None,
    limit:      int = 100,
) -> list[MTFSnapshot]:
    """
    Fetch bars for each timeframe and compute CVD.
    Returns list of MTFSnapshot sorted from lowest to highest TF.
    """
    from providers.yfinance_provider import YFinanceProvider
    provider   = YFinanceProvider()
    tfs        = timeframes or TIMEFRAMES
    snapshots  = []

    for tf in tfs:
        try:
            bars = provider.fetch_bars_sync(symbol, tf, limit)
            if not bars:
                continue
            engine = CVDEngine()
            for b in bars:
                engine.update_bar(b)
            cvd   = engine.value
            delta = bars[-1].delta if bars else 0.0
            snapshots.append(MTFSnapshot(
                timeframe  = tf,
                cvd        = cvd,
                last_delta = delta,
                direction  = _direction(cvd, delta),
                bars_used  = len(bars),
            ))
        except Exception:
            continue

    return snapshots


def confluence_score(snapshots: list[MTFSnapshot]) -> dict:
    """
    Confluence score: how many TFs agree on direction.
    Returns: {'bullish': N, 'bearish': N, 'score': float -1..1}
    """
    if not snapshots:
        return {"bullish": 0, "bearish": 0, "score": 0.0}

    bull = sum(1 for s in snapshots if s.direction == "bullish")
    bear = sum(1 for s in snapshots if s.direction == "bearish")
    total = len(snapshots)
    score = (bull - bear) / total   # -1 = fully bearish, +1 = fully bullish

    return {"bullish": bull, "bearish": bear, "total": total, "score": round(score, 2)}

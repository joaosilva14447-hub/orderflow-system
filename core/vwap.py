"""
VWAP + Standard Deviation bands.
Anchored VWAP from any reference bar.
"""

import numpy as np
from core.models import Bar


def calculate_vwap(bars: list[Bar]) -> list[float]:
    """Rolling VWAP — resets each day."""
    vwaps, cum_pv, cum_vol = [], 0.0, 0.0
    prev_date = None

    for b in bars:
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(b.timestamp / 1000, tz=timezone.utc)
        date = dt.date()

        if date != prev_date:          # Daily reset
            cum_pv = cum_vol = 0.0
            prev_date = date

        typical = (b.high + b.low + b.close) / 3
        cum_pv  += typical * b.volume
        cum_vol += b.volume
        vwaps.append(cum_pv / cum_vol if cum_vol > 0 else typical)

    return vwaps


def calculate_vwap_bands(
    bars: list[Bar],
    vwaps: list[float],
    n_bands: int = 3,
) -> dict[str, list[float]]:
    """
    VWAP Standard Deviation bands (1σ, 2σ, 3σ).
    Returns dict: {'+1': [...], '-1': [...], '+2': [...], '-2': [...], ...}
    """
    bands: dict[str, list[float]] = {f"{s}{i}": [] for i in range(1, n_bands + 1) for s in ("+", "-")}
    cum_pv = cum_vol = cum_pv2 = 0.0
    prev_date = None

    for idx, b in enumerate(bars):
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(b.timestamp / 1000, tz=timezone.utc)
        date = dt.date()

        if date != prev_date:
            cum_pv = cum_vol = cum_pv2 = 0.0
            prev_date = date

        typical    = (b.high + b.low + b.close) / 3
        cum_pv    += typical * b.volume
        cum_vol   += b.volume
        cum_pv2   += typical ** 2 * b.volume

        vwap = vwaps[idx]
        variance = (cum_pv2 / cum_vol - vwap ** 2) if cum_vol > 0 else 0.0
        sd = max(variance, 0.0) ** 0.5

        for i in range(1, n_bands + 1):
            bands[f"+{i}"].append(vwap + i * sd)
            bands[f"-{i}"].append(vwap - i * sd)

    return bands


def anchored_vwap(bars: list[Bar], anchor_idx: int = 0) -> list[float]:
    """VWAP anchored from a specific bar (e.g. swing low/high)."""
    result = [None] * anchor_idx
    cum_pv = cum_vol = 0.0
    for b in bars[anchor_idx:]:
        typical  = (b.high + b.low + b.close) / 3
        cum_pv  += typical * b.volume
        cum_vol += b.volume
        result.append(cum_pv / cum_vol if cum_vol > 0 else typical)
    return result

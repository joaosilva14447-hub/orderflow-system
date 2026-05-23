"""
Session analysis — London / New York / Asia.
Calculates POC, Value Area, VWAP per session.
All times in UTC.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, time as dtime
from typing import Optional

from core.models import Bar
from core.engine import VolumeProfileEngine


SESSION_HOURS = {
    "Asia":   (dtime(0,  0), dtime(8,  0)),   # 00:00–08:00 UTC
    "London": (dtime(8,  0), dtime(13, 0)),   # 08:00–13:00 UTC
    "NY":     (dtime(13, 0), dtime(21, 0)),   # 13:00–21:00 UTC
}

SESSION_COLORS = {
    "Asia":   "rgba(100, 180, 255, 0.12)",
    "London": "rgba(255, 200,  80, 0.10)",
    "NY":     "rgba(100, 220, 140, 0.10)",
}

SESSION_LINE_COLORS = {
    "Asia":   "#64b4ff",
    "London": "#ffc850",
    "NY":     "#64dc8c",
}


@dataclass
class SessionResult:
    name:       str
    start_ts:   float          # ms
    end_ts:     float          # ms
    poc:        Optional[float]
    va_high:    Optional[float]
    va_low:     Optional[float]
    vwap:       float
    high:       float
    low:        float
    total_vol:  float
    bars:       list = field(default_factory=list, repr=False)


def _session_of(dt: datetime) -> Optional[str]:
    t = dt.time().replace(second=0, microsecond=0)
    for name, (start, end) in SESSION_HOURS.items():
        if start <= t < end:
            return name
    return None


def calculate_sessions(bars: list[Bar], tick_size: float = 10.0) -> list[SessionResult]:
    """
    Group bars by session and calculate orderflow metrics per session.
    Returns list of completed sessions (most recent last).
    """
    # Group bars into sessions
    sessions: dict[tuple, dict] = {}

    for b in bars:
        dt   = datetime.fromtimestamp(b.timestamp / 1000, tz=timezone.utc)
        date = dt.date()
        sname = _session_of(dt)
        if not sname:
            continue

        key = (date, sname)
        if key not in sessions:
            sessions[key] = {"name": sname, "bars": [], "date": date}
        sessions[key]["bars"].append(b)

    results = []
    for (date, sname), data in sorted(sessions.items()):
        sbars = data["bars"]
        if not sbars:
            continue

        engine = VolumeProfileEngine(tick_size=tick_size)
        cum_pv = cum_vol = 0.0
        high = max(b.high for b in sbars)
        low  = min(b.low  for b in sbars)

        for b in sbars:
            engine.update_bar(b)
            tp     = (b.high + b.low + b.close) / 3
            cum_pv += tp * b.volume
            cum_vol += b.volume

        poc   = engine.poc
        va_lo, va_hi = engine.value_area(0.70)
        vwap  = cum_pv / cum_vol if cum_vol > 0 else sbars[-1].close

        results.append(SessionResult(
            name      = sname,
            start_ts  = sbars[0].timestamp,
            end_ts    = sbars[-1].timestamp,
            poc       = poc,
            va_high   = va_hi if va_hi else None,
            va_low    = va_lo if va_lo else None,
            vwap      = vwap,
            high      = high,
            low       = low,
            total_vol = cum_vol,
            bars      = sbars,
        ))

    return results


def get_current_sessions(sessions: list[SessionResult]) -> list[SessionResult]:
    """Return sessions from the most recent day only."""
    if not sessions:
        return []
    last_ts = max(s.end_ts for s in sessions)
    last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).date()
    return [s for s in sessions if
            datetime.fromtimestamp(s.start_ts / 1000, tz=timezone.utc).date() == last_dt]

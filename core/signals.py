"""
Signal Engine — orderflow-based setups.
All signals computed bar-by-bar using only past data (zero lookahead bias).

Signals implemented:
  1. CVD_DIV_BULL / CVD_DIV_BEAR   — price/CVD divergence
  2. VWAP_CROSS_BULL / BEAR        — VWAP crossover + delta confirmation
  3. POC_RECLAIM_BULL / BEAR       — price reclaims rolling POC
  4. VA_BREAKOUT_BULL / BEAR       — Value Area breakout with volume
  5. VWAP_BOUNCE_BULL / BEAR       — VWAP pullback in trend
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from core.models import Bar
from core.engine import VolumeProfileEngine, CVDEngine
from core.vwap   import calculate_vwap


# ── Signal dataclass ──────────────────────────────────────────────────────────

@dataclass
class Signal:
    bar_idx:    int
    timestamp:  float
    symbol:     str
    signal_type: str
    direction:  str          # 'long' | 'short'
    price:      float        # entry at bar close
    atr:        float        # ATR at signal time
    stop_loss:  float
    take_profit: float
    rr_ratio:   float
    confidence: float        # 0.0 – 1.0
    metadata:   dict = field(default_factory=dict)

    @property
    def risk(self) -> float:
        return abs(self.price - self.stop_loss)


# ── ATR ───────────────────────────────────────────────────────────────────────

def _atr(bars: list[Bar], period: int = 14) -> list[float]:
    trs = [bars[0].high - bars[0].low]
    for i in range(1, len(bars)):
        tr = max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low  - bars[i - 1].close),
        )
        trs.append(tr)
    atrs = [None] * period
    window = list(trs[:period])
    atrs[period - 1] = sum(window) / period
    for i in range(period, len(trs)):
        atrs.append((atrs[-1] * (period - 1) + trs[i]) / period)
    return atrs


def _rolling_cvd(bars: list[Bar], window: int = 50) -> list[float]:
    """Rolling CVD resets every `window` bars."""
    result = []
    for i, b in enumerate(bars):
        start = max(0, i - window + 1)
        cum = sum(bars[j].delta for j in range(start, i + 1))
        result.append(cum)
    return result


def _rolling_poc(bars: list[Bar], window: int = 100, tick_size: float = 10.0) -> list[Optional[float]]:
    pocs = []
    for i, b in enumerate(bars):
        start = max(0, i - window + 1)
        eng = VolumeProfileEngine(tick_size=tick_size)
        for j in range(start, i + 1):
            eng.update_bar(bars[j])
        pocs.append(eng.poc)
    return pocs


def _rolling_va(bars: list[Bar], window: int = 100, tick_size: float = 10.0,
                va_pct: float = 0.70) -> tuple[list, list]:
    va_highs, va_lows = [], []
    for i in range(len(bars)):
        start = max(0, i - window + 1)
        eng = VolumeProfileEngine(tick_size=tick_size)
        for j in range(start, i + 1):
            eng.update_bar(bars[j])
        lo, hi = eng.value_area(va_pct)
        va_highs.append(hi)
        va_lows.append(lo)
    return va_highs, va_lows


def _avg_volume(bars: list[Bar], i: int, window: int = 20) -> float:
    start = max(0, i - window)
    vols  = [bars[j].volume for j in range(start, i)]
    return np.mean(vols) if vols else bars[i].volume


def _sma(bars: list[Bar], period: int = 50) -> list[Optional[float]]:
    """Simple moving average of close prices — None for first (period-1) bars."""
    result: list[Optional[float]] = [None] * (period - 1)
    window_sum = sum(bars[j].close for j in range(period - 1))
    for i in range(period - 1, len(bars)):
        window_sum += bars[i].close
        result.append(window_sum / period)
        window_sum -= bars[i - period + 1].close
    return result


# ── Signal detection ──────────────────────────────────────────────────────────

class SignalEngine:
    """
    Detects orderflow-based signals on a list of Bar objects.
    All indicators computed bar-by-bar — zero lookahead bias.
    """

    def __init__(
        self,
        tick_size:       float = 10.0,
        atr_period:      int   = 14,
        sl_atr_mult:     float = 1.5,
        rr_ratio:        float = 2.0,
        vp_window:       int   = 100,
        cvd_window:      int   = 50,
        div_lookback:    int   = 5,
        vol_mult:        float = 1.3,
        cooldown_bars:   int   = 10,   # min bars between same-type signals
        min_confidence:  float = 0.60, # discard signals below this threshold
        trend_period:    int   = 50,   # SMA period for trend alignment filter
        enabled_signals: Optional[list] = None,  # None = all; list = whitelist
    ):
        self.tick_size       = tick_size
        self.atr_period      = atr_period
        self.cooldown_bars   = cooldown_bars
        self.min_confidence  = min_confidence
        self.trend_period    = trend_period
        self.enabled_signals = enabled_signals
        self.sl_atr_mult  = sl_atr_mult
        self.rr_ratio     = rr_ratio
        self.vp_window    = vp_window
        self.cvd_window   = cvd_window
        self.div_lookback = div_lookback
        self.vol_mult     = vol_mult

    def _should_emit(self, stype: str) -> bool:
        """Return True if this signal type is enabled (or no whitelist is set)."""
        return self.enabled_signals is None or stype in self.enabled_signals

    def _make_signal(self, bar: Bar, idx: int, stype: str,
                     direction: str, atr: float, confidence: float,
                     metadata: dict = None) -> Signal:
        price = bar.close
        risk  = atr * self.sl_atr_mult
        if direction == "long":
            sl = price - risk
            tp = price + risk * self.rr_ratio
        else:
            sl = price + risk
            tp = price - risk * self.rr_ratio
        return Signal(
            bar_idx     = idx,
            timestamp   = bar.timestamp,
            symbol      = bar.symbol,
            signal_type = stype,
            direction   = direction,
            price       = price,
            atr         = atr,
            stop_loss   = sl,
            take_profit = tp,
            rr_ratio    = self.rr_ratio,
            confidence  = confidence,
            metadata    = metadata or {},
        )

    def detect(self, bars: list[Bar]) -> list[Signal]:
        """
        Run all signal detectors on the bar list.

        Post-processing (applied after raw detection):
          • min_confidence  — drop any signal below threshold
          • cooldown_bars   — same signal type cannot fire again within N bars
          • one per bar     — if multiple pass filters, keep highest-confidence only
        """
        if len(bars) < max(self.atr_period, self.vp_window) + 10:
            return []

        atrs    = _atr(bars, self.atr_period)
        cvds    = _rolling_cvd(bars, self.cvd_window)
        vwaps   = calculate_vwap(bars)
        pocs    = _rolling_poc(bars, self.vp_window, self.tick_size)
        va_hi, va_lo = _rolling_va(bars, self.vp_window, self.tick_size)
        smas    = _sma(bars, self.trend_period)

        # ── Raw candidate collection ──────────────────────────────────────────
        candidates: list[Signal] = []
        lb = self.div_lookback

        for i in range(self.vp_window + lb, len(bars)):
            b    = bars[i]
            atr  = atrs[i]
            if atr is None or atr == 0:
                continue

            cvd_now  = cvds[i]
            cvd_prev = cvds[i - lb]
            poc      = pocs[i]
            vwap     = vwaps[i]
            sma      = smas[i]          # None if SMA not yet warm
            avg_vol  = _avg_volume(bars, i)

            # ── 1. CVD Divergence (reversal signal) ───────────────────────────
            # Bear divergence: price HH but CVD LH → reversal down expected.
            #   Trend alignment: price above SMA confirms we're in an uptrend
            #   that is now weakening — ideal setup for a reversal short.
            # Bull divergence: price LL but CVD HL → reversal up expected.
            #   Trend alignment: price below SMA confirms downtrend weakness.
            # ATR guard: require a meaningful price move (≥ 0.5×ATR) to avoid
            #   false divergences on flat/choppy bars.
            price_hh = b.close > bars[i - lb].close
            price_ll = b.close < bars[i - lb].close
            cvd_lh   = cvd_now < cvd_prev
            cvd_hl   = cvd_now > cvd_prev

            trend_bear_ok  = sma is None or b.close > sma  # in uptrend → reversal short valid
            trend_bull_ok  = sma is None or b.close < sma  # in downtrend → reversal long valid
            price_move_ok  = abs(b.close - bars[i - lb].close) > 0.5 * atr  # meaningful swing

            if (price_hh and cvd_lh and trend_bear_ok and price_move_ok
                    and self._should_emit("CVD_DIV_BEAR")):
                candidates.append(self._make_signal(b, i, "CVD_DIV_BEAR", "short", atr,
                    confidence=0.70,
                    metadata={"price_chg": b.close - bars[i-lb].close,
                              "cvd_chg": cvd_now - cvd_prev}))

            elif (price_ll and cvd_hl and trend_bull_ok and price_move_ok
                    and self._should_emit("CVD_DIV_BULL")):
                candidates.append(self._make_signal(b, i, "CVD_DIV_BULL", "long", atr,
                    confidence=0.70,
                    metadata={"price_chg": b.close - bars[i-lb].close,
                              "cvd_chg": cvd_now - cvd_prev}))

            # ── 2. VWAP Cross (trend-following) ──────────────────────────────
            # Only take crosses that align with the SMA trend direction.
            if vwap and i > 0 and vwaps[i-1]:
                prev_close = bars[i-1].close
                prev_vwap  = vwaps[i-1]

                if (prev_close < prev_vwap and b.close > vwap and b.delta > 0
                        and (sma is None or b.close > sma)
                        and self._should_emit("VWAP_CROSS_BULL")):
                    candidates.append(self._make_signal(b, i, "VWAP_CROSS_BULL", "long", atr,
                        confidence=0.65,
                        metadata={"vwap": vwap, "delta": b.delta}))

                elif (prev_close > prev_vwap and b.close < vwap and b.delta < 0
                        and (sma is None or b.close < sma)
                        and self._should_emit("VWAP_CROSS_BEAR")):
                    candidates.append(self._make_signal(b, i, "VWAP_CROSS_BEAR", "short", atr,
                        confidence=0.65,
                        metadata={"vwap": vwap, "delta": b.delta}))

            # ── 3. POC Reclaim (trend-continuation) ──────────────────────────
            # Price reclaiming a key level in the direction of the larger trend.
            if poc and i >= 3:
                n_below = sum(1 for j in range(i-3, i) if bars[j].close < poc)
                n_above = sum(1 for j in range(i-3, i) if bars[j].close > poc)

                if (n_below >= 2 and b.close > poc and cvd_now > 0
                        and (sma is None or b.close >= sma * 0.99)
                        and self._should_emit("POC_RECLAIM_BULL")):
                    candidates.append(self._make_signal(b, i, "POC_RECLAIM_BULL", "long", atr,
                        confidence=0.72,
                        metadata={"poc": poc, "cvd": cvd_now}))

                elif (n_above >= 2 and b.close < poc and cvd_now < 0
                        and (sma is None or b.close <= sma * 1.01)
                        and self._should_emit("POC_RECLAIM_BEAR")):
                    candidates.append(self._make_signal(b, i, "POC_RECLAIM_BEAR", "short", atr,
                        confidence=0.72,
                        metadata={"poc": poc, "cvd": cvd_now}))

            # ── 4. Value Area Breakout (momentum/trend) ───────────────────────
            # Breakout must align with SMA trend for confirmation.
            # Confidence raised to 0.78 (best performer — wins one-per-bar competition).
            vah = va_hi[i]
            val = va_lo[i]
            high_vol = b.volume >= avg_vol * self.vol_mult

            if vah and i > 0:
                prev_vah = va_hi[i-1]
                if (prev_vah and bars[i-1].close < prev_vah and b.close > vah and high_vol
                        and (sma is None or b.close > sma)
                        and self._should_emit("VA_BREAKOUT_BULL")):
                    candidates.append(self._make_signal(b, i, "VA_BREAKOUT_BULL", "long", atr,
                        confidence=0.78,
                        metadata={"va_high": vah, "vol_ratio": b.volume / avg_vol}))

            if val and i > 0:
                prev_val = va_lo[i-1]
                if (prev_val and bars[i-1].close > prev_val and b.close < val and high_vol
                        and (sma is None or b.close < sma)
                        and self._should_emit("VA_BREAKOUT_BEAR")):
                    candidates.append(self._make_signal(b, i, "VA_BREAKOUT_BEAR", "short", atr,
                        confidence=0.78,
                        metadata={"va_low": val, "vol_ratio": b.volume / avg_vol}))

            # ── 5. VWAP Bounce (trend-following pullback) ─────────────────────
            # Pullback to VWAP in the direction of the SMA trend.
            # Confidence lowered to 0.58 (below default min_confidence=0.60)
            # so this type is effectively off unless the user explicitly lowers
            # the confidence slider or manually enables it.
            if vwap and i >= 2:
                touched_vwap = bars[i-1].low <= vwaps[i-1] <= bars[i-1].high
                uptrend      = cvd_now > 0 and b.close > bars[i-2].close
                downtrend    = cvd_now < 0 and b.close < bars[i-2].close

                if (touched_vwap and uptrend and b.close > vwap
                        and (sma is None or b.close > sma)
                        and self._should_emit("VWAP_BOUNCE_BULL")):
                    candidates.append(self._make_signal(b, i, "VWAP_BOUNCE_BULL", "long", atr,
                        confidence=0.58,
                        metadata={"vwap": vwap, "cvd": cvd_now}))

                elif (touched_vwap and downtrend and b.close < vwap
                        and (sma is None or b.close < sma)
                        and self._should_emit("VWAP_BOUNCE_BEAR")):
                    candidates.append(self._make_signal(b, i, "VWAP_BOUNCE_BEAR", "short", atr,
                        confidence=0.58,
                        metadata={"vwap": vwap, "cvd": cvd_now}))

        # ── Post-processing: confidence → cooldown → one-per-bar ─────────────
        # Group by bar index (preserves chronological order)
        by_bar: dict[int, list[Signal]] = {}
        for sig in candidates:
            by_bar.setdefault(sig.bar_idx, []).append(sig)

        signals:  list[Signal] = []
        last_bar: dict[str, int] = {}   # signal_type → bar_idx of last emission

        for bar_idx in sorted(by_bar):
            bar_sigs = by_bar[bar_idx]

            # 1. Drop below min_confidence
            bar_sigs = [s for s in bar_sigs if s.confidence >= self.min_confidence]
            if not bar_sigs:
                continue

            # 2. Cooldown filter — per signal type, independently
            eligible: list[Signal] = []
            for s in bar_sigs:
                last = last_bar.get(s.signal_type, -(self.cooldown_bars + 1))
                if bar_idx - last > self.cooldown_bars:
                    eligible.append(s)
            if not eligible:
                continue

            # 3. One signal per bar — highest confidence wins
            best = max(eligible, key=lambda s: s.confidence)
            signals.append(best)
            last_bar[best.signal_type] = bar_idx

        return signals

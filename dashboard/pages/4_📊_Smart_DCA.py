"""
📊 Smart DCA Compass
=====================
Composite Extreme Score (0–100) para Long-Term DCA.
Score alto (>70) → oversold → acumular.
Score baixo (<30) → overbought → distribuir.
Sinais apenas em zonas extremas — ignorar tudo no meio.
"""
from __future__ import annotations

import sys, os
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
from datetime import datetime

st.set_page_config(
    page_title="Smart DCA Compass",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# (lo, hi, fill_rgba, label, label_color, action, hex_color)
_DCA_ZONES = [
    (  0, 20, "rgba(189,40,40,.88)",   "EXTREME OVERBOUGHT", "rgba(255,110,100,.95)", "VENDER MÁXIMO",   "#bd2828"),
    ( 20, 35, "rgba(190,90,20,.82)",   "OVERBOUGHT",         "rgba(240,145,55,.95)",  "REDUZIR POSIÇÃO", "#be5a14"),
    ( 35, 55, "rgba(55,55,60,.78)",    "NEUTRO",             "rgba(175,175,180,.90)", "AGUARDAR",        "#505058"),
    ( 55, 70, "rgba(14,110,60,.82)",   "OVERSOLD",           "rgba(50,200,120,.95)",  "COMPRA PARCIAL",  "#0e6e3c"),
    ( 70, 85, "rgba(10,140,65,.85)",   "STRONG OVERSOLD",    "rgba(40,220,110,.95)",  "COMPRA FORTE",    "#0a8c41"),
    ( 85,100, "rgba(5,165,70,.90)",    "EXTREME OVERSOLD",   "rgba(30,245,105,.95)",  "DCA MÁXIMO",      "#05a546"),
]

_DCA_ACTIONS = {
    "EXTREME OVERBOUGHT": ("🔴 VENDER MÁXIMO",    "rgba(189,40,40,.15)",  "#bd2828",
                           "Mercado em euforia extrema. Zona de distribuição — liquidar posição gradualmente."),
    "OVERBOUGHT":         ("🟠 REDUZIR POSIÇÃO",  "rgba(190,90,20,.12)",  "#be5a14",
                           "Sobrecomprado. Realizar lucros parciais. Não adicionar à posição."),
    "NEUTRO":             ("⚪ AGUARDAR",          "rgba(55,55,60,.12)",   "#888890",
                           "Mercado em equilíbrio. Sem acção — aguardar por extremos."),
    "OVERSOLD":           ("🟢 COMPRA PARCIAL",   "rgba(14,110,60,.15)",  "#26a69a",
                           "1ª janela de acumulação. Iniciar 1ª tranche DCA (25–33%)."),
    "STRONG OVERSOLD":    ("🟢 COMPRA FORTE",     "rgba(10,140,65,.18)",  "#0a8c41",
                           "Oversold confirmado. Adicionar 2ª tranche DCA (50–66% da alocação)."),
    "EXTREME OVERSOLD":   ("🟢 DCA MÁXIMO",       "rgba(5,165,70,.20)",   "#05a546",
                           "Capitulação extrema. Zona histórica de acumulação máxima — activar todas as tranches."),
}

_ASSETS_PRESET = {
    "₿  BTC-USD":  "BTC-USD",
    "Ξ  ETH-USD":  "ETH-USD",
    "◎  SOL-USD":  "SOL-USD",
    "📈  SPY":      "SPY",
    "📊  QQQ":      "QQQ",
    "🔷  NVDA":     "NVDA",
    "🍎  AAPL":     "AAPL",
}

_PERIOD_MAP   = {"1 Ano": "1y", "2 Anos": "2y", "3 Anos": "3y", "5 Anos": "5y"}
_SUB_WEIGHTS  = {"RSI(14)": 0.20, "RSI(2)": 0.10, "Z-score": 0.30, "BB %B": 0.20, "Vol Climax": 0.20}

# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _divider() -> None:
    st.markdown(
        "<div style='height:1px;background:linear-gradient(90deg,"
        "transparent 0%,rgba(38,166,154,.18) 20%,rgba(255,255,255,.06) 50%,"
        "rgba(38,166,154,.18) 80%,transparent 100%);margin:20px 0 8px'></div>",
        unsafe_allow_html=True,
    )

def _section_header(title: str, subtitle: str = "") -> None:
    sub = (f"<div style='font-size:.78em;color:rgba(255,255,255,.40);margin-top:2px'>"
           f"{subtitle}</div>") if subtitle else ""
    st.markdown(
        f"<div style='border-left:3px solid rgba(38,166,154,.65);"
        f"padding:4px 0 4px 12px;margin:14px 0 10px'>"
        f"<span style='font-size:.94em;font-weight:600;color:rgba(255,255,255,.90)'>"
        f"{title}</span>{sub}</div>",
        unsafe_allow_html=True,
    )

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"

def _zone_for_score(score: float) -> dict:
    for lo, hi, col, label, lcol, action, hexc in _DCA_ZONES:
        if lo <= score < hi or (hi == 100 and score == 100):
            return {"bg": col, "label": label, "lcol": lcol, "action": action, "hex": hexc}
    return {"bg": "rgba(55,55,60,.78)", "label": "NEUTRO",
            "lcol": "rgba(175,175,180,.90)", "action": "AGUARDAR", "hex": "#505058"}

# ─────────────────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).clip(0, 100)

def _bollinger(prices: pd.Series, period: int = 20, n_std: float = 2.0):
    ma    = prices.rolling(period, min_periods=period // 2).mean()
    sigma = prices.rolling(period, min_periods=period // 2).std()
    upper = ma + n_std * sigma
    lower = ma - n_std * sigma
    band  = (upper - lower).replace(0, np.nan)
    pct_b = (prices - lower) / band
    width = band / ma.replace(0, np.nan)
    return pct_b, width, ma, upper, lower

def _zscore(prices: pd.Series, period: int = 252) -> pd.Series:
    ma  = prices.rolling(period, min_periods=period // 3).mean()
    std = prices.rolling(period, min_periods=period // 3).std()
    return (prices - ma) / std.replace(0, np.nan)

def _volume_climax(prices: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    ret     = prices.pct_change()
    avg_vol = volume.rolling(period, min_periods=5).mean().replace(0, np.nan)
    ratio   = volume / avg_vol
    direct  = np.where(ret < 0, 1.0, -1.0)
    return pd.Series(direct * ratio, index=prices.index)

# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score 0–100: alto = oversold (comprar), baixo = overbought (vender).
    """
    close  = df["Close"]
    volume = df.get("Volume", None)

    rsi14       = _rsi(close, 14)
    rsi14_score = (100 - rsi14).clip(0, 100)

    rsi2        = _rsi(close, 2)
    rsi2_score  = (100 - rsi2).clip(0, 100)

    z           = _zscore(close, 252)
    z_score     = (50 - z * 16.67).clip(0, 100)

    pct_b, bb_width, bb_ma, bb_up, bb_lo = _bollinger(close, 20, 2.0)
    bb_score    = (100 - pct_b * 100).clip(0, 100)

    if volume is not None and volume.sum() > 0:
        vc        = _volume_climax(close, volume)
        vol_score = (50 + vc * 10).clip(0, 100)
    else:
        vol_score = pd.Series(50.0, index=close.index)

    W = _SUB_WEIGHTS
    composite = (
        rsi14_score * W["RSI(14)"] +
        rsi2_score  * W["RSI(2)"]  +
        z_score     * W["Z-score"] +
        bb_score    * W["BB %B"]   +
        vol_score   * W["Vol Climax"]
    ).clip(0, 100)

    return pd.DataFrame({
        "close":       close,
        "volume":      volume if volume is not None else pd.Series(np.nan, index=close.index),
        "score":       composite,
        "rsi14":       rsi14,       "rsi14_score": rsi14_score,
        "rsi2":        rsi2,        "rsi2_score":  rsi2_score,
        "z":           z,           "z_score":     z_score,
        "pct_b":       pct_b,       "bb_score":    bb_score,
        "bb_width":    bb_width,
        "bb_ma":       bb_ma,       "bb_up":       bb_up,    "bb_lo": bb_lo,
        "vol_score":   vol_score,
    }).dropna(subset=["score"])

# ─────────────────────────────────────────────────────────────────────────────
# PERCENTILE RANK
# ─────────────────────────────────────────────────────────────────────────────

def _percentile_rank(current: float, series: pd.Series) -> float:
    valid = series.dropna()
    return float((valid <= current).mean() * 100)

# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL RETURNS FROM ZONE
# ─────────────────────────────────────────────────────────────────────────────

def _zone_returns(scores_df: pd.DataFrame, zone_lo: float, zone_hi: float,
                  horizons: list[int] = None) -> dict:
    """
    Find entries into zone and compute forward price returns.
    Returns stats for each horizon.
    """
    if horizons is None:
        horizons = [90, 180, 365]

    prices = scores_df["close"]
    scores = scores_df["score"]
    n      = len(scores_df)

    # Entry = first bar to enter zone after being outside for 5+ bars
    entries: list[int] = []
    last_out = 0
    inside   = False
    for i in range(n):
        val = scores.iloc[i]
        in_zone = zone_lo <= val <= zone_hi or (zone_hi == 100 and val == 100)
        if in_zone and not inside and i - last_out >= 5:
            entries.append(i)
            inside = True
        if not in_zone and inside:
            inside   = False
            last_out = i

    result: dict = {"n": len(entries)}
    max_h  = max(horizons)

    for h in horizons:
        rets = []
        for e in entries:
            future = e + h
            if future >= n:
                continue
            p0 = float(prices.iloc[e])
            p1 = float(prices.iloc[future])
            if p0 > 0:
                rets.append((p1 / p0 - 1) * 100)
        if rets:
            a = np.array(rets)
            result[h] = {
                "avg":     float(a.mean()),
                "pct_pos": float((a > 0).mean() * 100),
                "best":    float(a.max()),
                "worst":   float(a.min()),
                "n":       len(a),
            }

    return result

# ─────────────────────────────────────────────────────────────────────────────
# SPAGHETTI PATHS
# ─────────────────────────────────────────────────────────────────────────────

def _spaghetti_paths(scores_df: pd.DataFrame, zone_lo: float, zone_hi: float,
                     max_days: int = 180) -> list[list[float]]:
    """
    Forward price return paths (%) from each historical zone entry.
    Paths normalised to 0 at entry.
    Excludes the last max_days rows (no forward data).
    """
    prices = scores_df["close"]
    scores = scores_df["score"]
    n      = len(scores_df)
    cutoff = n - max_days - 1

    paths: list[list[float]] = []
    inside   = False
    last_out = 0

    for i in range(min(n, cutoff)):
        val     = scores.iloc[i]
        in_zone = zone_lo <= val <= zone_hi or (zone_hi == 100 and val == 100)

        if in_zone and not inside and i - last_out >= 5:
            p0   = float(prices.iloc[i])
            if p0 <= 0:
                continue
            path = []
            for d in range(max_days + 1):
                if i + d < n:
                    path.append((float(prices.iloc[i + d]) / p0 - 1) * 100)
            if len(path) == max_days + 1:
                paths.append(path)
            inside = True
        if not in_zone and inside:
            inside   = False
            last_out = i

    return paths

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load_data(ticker: str, period: str) -> pd.DataFrame:
    try:
        raw = yf.download(ticker, period=period, interval="1d",
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index).normalize()
        return raw.dropna(how="all")
    except Exception:
        return pd.DataFrame()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("📊 Smart DCA Compass")
st.sidebar.caption("Long-Term Composite Extreme Score · Sinais apenas em zonas extremas.")

_asset_choice = st.sidebar.selectbox(
    "Ativo", list(_ASSETS_PRESET.keys()) + ["✏️ Personalizado"], index=0)

_ticker = (st.sidebar.text_input("Ticker Yahoo Finance", "BTC-USD").strip().upper()
           if _asset_choice == "✏️ Personalizado"
           else _ASSETS_PRESET[_asset_choice])

_period_label = st.sidebar.selectbox("Histórico", list(_PERIOD_MAP.keys()), index=1)
_period       = _PERIOD_MAP[_period_label]
_chart_days   = min(int(_period_label.split()[0]) * 252, 1260)

st.sidebar.markdown("---")
_show_bb = st.sidebar.checkbox("Bollinger Bands", value=True)
_show_ma = st.sidebar.checkbox("MA50 / MA200",    value=True)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Score composto**\n\n"
    "| Indicador | Peso |\n|---|---|\n"
    "| RSI(14) | 20% |\n| RSI(2) | 10% |\n"
    "| Z-score (1A) | 30% |\n| BB %B | 20% |\n| Vol Climax | 20% |"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD + COMPUTE
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner(f"⏳ A carregar {_ticker}..."):
    _raw = _load_data(_ticker, _period)

if _raw.empty:
    st.error(f"Sem dados para **{_ticker}**. Verifica o ticker.")
    st.stop()

_scores_df  = _compute_scores(_raw)
if _scores_df.empty:
    st.error("Dados insuficientes. Tenta período mais longo.")
    st.stop()

_cur        = _scores_df.iloc[-1]
_cur_score  = float(_cur["score"])
_zone       = _zone_for_score(_cur_score)
_z_label    = _zone["label"]
_z_color    = _zone["hex"]
_z_action   = _zone["action"]
_pct_rank   = _percentile_rank(_cur_score, _scores_df["score"])

# Zone bounds for current zone
_cur_zone_lo, _cur_zone_hi = next(
    (lo, hi) for lo, hi, *_ in _DCA_ZONES if _z_label in _
    ) if any(_z_label in rest for lo, hi, *rest in _DCA_ZONES) else (35, 55)
# Simpler lookup:
for _zlo, _zhi, _zbg, _zlbl, _zlcol, _zact, _zhex in _DCA_ZONES:
    if _zlbl == _z_label:
        _cur_zone_lo, _cur_zone_hi = _zlo, _zhi
        break

_ret_stats  = _zone_returns(_scores_df, _cur_zone_lo, _cur_zone_hi)
_spag_paths = _spaghetti_paths(_scores_df, _cur_zone_lo, _cur_zone_hi)

# Score delta vs 10d ago
_score_delta = float(_cur["score"] - _scores_df["score"].iloc[-11]) \
    if len(_scores_df) >= 11 else None

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(f"### 📊 Smart DCA Compass — **{_ticker}**")
st.caption(
    f"{_scores_df.index[-1].strftime('%Y-%m-%d')}  ·  "
    f"Histórico: {_period_label}  ·  "
    f"Score: **{_cur_score:.1f}**  ·  "
    f"Zona: **{_z_label}**  ·  "
    f"Percentile: **{_pct_rank:.0f}th**"
)

# ─────────────────────────────────────────────────────────────────────────────
# ACTION BANNER
# ─────────────────────────────────────────────────────────────────────────────

_act_info  = _DCA_ACTIONS.get(_z_label, _DCA_ACTIONS["NEUTRO"])
_act_label, _act_bg, _act_border, _act_desc = _act_info

st.markdown(
    f"<div style='background:linear-gradient(135deg,{_act_bg} 0%,"
    f"rgba(14,14,22,.95) 100%);border:1px solid {_act_border}55;"
    f"border-radius:14px;padding:16px 24px;margin-bottom:18px'>"
    f"<div style='font-size:1.45em;font-weight:900;color:{_act_border};"
    f"letter-spacing:2px'>{_act_label}</div>"
    f"<div style='font-size:.83em;color:rgba(255,255,255,.55);margin-top:5px'>"
    f"{_act_desc}</div></div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1 — Score Circle (B) | Action Details | Return Stats
# ─────────────────────────────────────────────────────────────────────────────

col_sc, col_ad, col_rs = st.columns([1, 1, 1], gap="medium")

# ── Score Circle ─────────────────────────────────────────────────────────────
with col_sc:
    _pct_rank_suffix = "th" if _pct_rank not in (11,12,13) and _pct_rank % 10 in (1,2,3) else "th"
    if _pct_rank % 10 == 1 and _pct_rank != 11:  _pct_rank_suffix = "st"
    elif _pct_rank % 10 == 2 and _pct_rank != 12: _pct_rank_suffix = "nd"
    elif _pct_rank % 10 == 3 and _pct_rank != 13: _pct_rank_suffix = "rd"

    _delta_str = ""
    if _score_delta is not None:
        _da = "↑" if _score_delta >= 1 else ("↓" if _score_delta <= -1 else "→")
        _dc = _z_color if _score_delta >= 0 else "#ef5350"
        _delta_str = (f"<div style='font-size:.78em;color:{_dc};margin-top:6px'>"
                      f"{_da} {_score_delta:+.1f} pts (10d)</div>")

    st.markdown(
        f"<div style='display:flex;flex-direction:column;align-items:center;"
        f"padding:18px 0 10px;background:#0e0e1c;border-radius:16px;"
        f"border:1px solid rgba(255,255,255,.06)'>"
        f"<div style='font-size:9px;letter-spacing:3px;color:rgba(255,255,255,.28);"
        f"margin-bottom:14px'>COMPOSITE SCORE</div>"
        f"<div style='width:148px;height:148px;border-radius:50%;"
        f"border:2px solid {_z_color}55;"
        f"background:radial-gradient(circle at 35% 35%,{_z_color}22,{_z_color}06 70%);"
        f"display:flex;flex-direction:column;align-items:center;justify-content:center;"
        f"box-shadow:0 0 38px {_z_color}28,inset 0 0 28px {_z_color}0a'>"
        f"<div style='font-size:50px;font-weight:900;color:{_z_color};"
        f"font-family:Arial Black,Arial;line-height:1'>{_cur_score:.0f}</div>"
        f"<div style='font-size:9px;color:{_z_color}66;letter-spacing:2px;"
        f"margin-top:2px'>/ 100</div>"
        f"</div>"
        f"<div style='margin-top:14px;background:{_z_color}22;border:1px solid {_z_color}55;"
        f"border-radius:20px;padding:5px 16px;font-size:9px;font-weight:700;"
        f"color:{_z_color};letter-spacing:1.5px'>{_z_label}</div>"
        f"<div style='margin-top:8px;font-size:.80em;color:rgba(255,255,255,.40)'>"
        f"Percentile: <b style='color:{_z_color}'>{_pct_rank:.0f}{_pct_rank_suffix}</b> (2A)</div>"
        f"{_delta_str}"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Action Details ────────────────────────────────────────────────────────────
with col_ad:
    # DCA Tranche status
    _tranches = [
        ("1ª Tranche — 33%", _cur_zone_lo >= 55, f"Score >{_cur_zone_lo} ativo"),
        ("2ª Tranche — 33%", _cur_zone_lo >= 70, f"Score >{_cur_zone_lo} por 3+ dias"),
        ("3ª Tranche — 33%", _cur_zone_lo >= 85, f"Score >{_cur_zone_lo} por 7+ dias"),
    ]
    rows_t = ""
    for name, active, cond in _tranches:
        bg   = f"{_z_color}22" if active else "rgba(40,40,50,.50)"
        bord = f"{_z_color}50" if active else "rgba(255,255,255,.08)"
        ico  = "✅" if active else "⏳"
        col_t = _z_color if active else "rgba(255,255,255,.35)"
        rows_t += (
            f"<div style='background:{bg};border:1px solid {bord};"
            f"border-radius:10px;padding:10px 14px;margin-bottom:8px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<span style='font-size:.82em;font-weight:700;color:{col_t}'>{ico} {name}</span>"
            f"</div>"
            f"<div style='font-size:.72em;color:rgba(255,255,255,.38);margin-top:3px'>{cond}</div>"
            f"</div>"
        )
    st.markdown(
        f"<div style='background:#0e0e1c;border-radius:16px;padding:18px 16px;"
        f"border:1px solid rgba(255,255,255,.06);height:100%'>"
        f"<div style='font-size:9px;letter-spacing:3px;color:rgba(255,255,255,.28);"
        f"margin-bottom:14px'>DCA LADDER — TRANCHES</div>"
        f"{rows_t}"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Historical Return Stats ───────────────────────────────────────────────────
with col_rs:
    _n_inst = _ret_stats.get("n", 0)
    rows_r  = ""
    for h, label_h in [(90, "90 dias"), (180, "180 dias"), (365, "1 ano")]:
        s = _ret_stats.get(h, {})
        if not s:
            rows_r += (
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:7px 0;border-bottom:1px solid rgba(255,255,255,.04)'>"
                f"<span style='font-size:.80em;color:rgba(255,255,255,.38)'>Retorno {label_h}</span>"
                f"<span style='font-size:.80em;color:rgba(255,255,255,.28)'>—</span></div>"
            )
            continue
        avg_c  = _z_color if s["avg"] >= 0 else "#ef5350"
        ppos_c = _z_color if s["pct_pos"] >= 55 else "#ef5350"
        rows_r += (
            f"<div style='padding:7px 0;border-bottom:1px solid rgba(255,255,255,.04)'>"
            f"<div style='display:flex;justify-content:space-between'>"
            f"<span style='font-size:.80em;color:rgba(255,255,255,.38)'>Retorno {label_h}</span>"
            f"<span style='font-size:.82em;font-weight:700;color:{avg_c}'>{s['avg']:+.1f}%</span>"
            f"</div>"
            f"<div style='font-size:.70em;color:rgba(255,255,255,.28);margin-top:1px'>"
            f"{s['pct_pos']:.0f}% positivo · n={s['n']}</div>"
            f"</div>"
        )

    st.markdown(
        f"<div style='background:#0e0e1c;border-radius:16px;padding:18px 16px;"
        f"border:1px solid rgba(255,255,255,.06);height:100%'>"
        f"<div style='font-size:9px;letter-spacing:3px;color:rgba(255,255,255,.28);"
        f"margin-bottom:14px'>RETORNOS HISTÓRICOS — ZONA {_z_label}</div>"
        f"{rows_r}"
        f"<div style='margin-top:10px;font-size:.70em;color:rgba(255,255,255,.22)'>"
        f"{_n_inst} ocorrências históricas · Resultados passados não garantem futuros.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# THERMOMETER
# ─────────────────────────────────────────────────────────────────────────────

_divider()

_thermo_fig = go.Figure()

_zone_segs = [
    (0,  20,  "rgba(189,40,40,.80)",  "VENDA MÁX"),
    (20, 35,  "rgba(190,90,20,.70)",  "OVERBOUGHT"),
    (35, 55,  "rgba(60,60,70,.60)",   "NEUTRO"),
    (55, 70,  "rgba(14,110,60,.70)",  "OVERSOLD"),
    (70, 85,  "rgba(10,140,65,.75)",  "COMPRA FORTE"),
    (85, 100, "rgba(5,165,70,.85)",   "DCA MÁX"),
]

for lo, hi, col, lbl in _zone_segs:
    # Highlight active zone slightly brighter
    _is_active = lo <= _cur_score < hi or (hi == 100 and _cur_score == 100)
    _bar_alpha  = "1.00" if _is_active else "0.55"
    _bar_col    = col.replace(col[col.rfind(",")+1:col.rfind(")")], _bar_alpha)

    _thermo_fig.add_shape(
        type="rect", x0=lo, x1=hi, y0=0.0, y1=1.0,
        fillcolor=_bar_col, line_width=0,
    )
    # Label INSIDE the bar
    _thermo_fig.add_annotation(
        x=(lo + hi) / 2, y=0.50,
        text=f"<b>{lbl}</b>",
        showarrow=False,
        font=dict(
            size=11,
            color="rgba(255,255,255,.92)" if _is_active else "rgba(255,255,255,.60)",
            family="monospace",
        ),
        xanchor="center", yanchor="middle",
    )
    # Range label at bottom
    _thermo_fig.add_annotation(
        x=(lo + hi) / 2, y=-0.35,
        text=f"{lo}–{hi}",
        showarrow=False,
        font=dict(size=8, color="rgba(255,255,255,.28)", family="monospace"),
        xanchor="center",
    )

# Needle (white vertical line + score bubble)
_thermo_fig.add_shape(
    type="line",
    x0=_cur_score, x1=_cur_score, y0=-0.10, y1=1.10,
    line=dict(color="white", width=3),
)
_thermo_fig.add_annotation(
    x=_cur_score, y=1.55,
    text=f"<b>{_cur_score:.1f}</b>",
    showarrow=False,
    font=dict(size=14, color="white", family="Arial Black, Arial"),
    bgcolor="rgba(255,255,255,.15)",
    bordercolor="rgba(255,255,255,.40)",
    borderwidth=1,
    borderpad=4,
)

_thermo_fig.update_layout(
    height=95,
    margin=dict(l=10, r=10, t=40, b=22),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(range=[0, 100], visible=False),
    yaxis=dict(range=[-0.5, 1.8], visible=False),
    showlegend=False,
)
st.plotly_chart(_thermo_fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PRICE CHART (Concept C) + DISTRIBUTION HISTOGRAM
# ─────────────────────────────────────────────────────────────────────────────

_divider()
_section_header(
    "Histórico de Preço + DCA Score Oscillator",
    "Preço com zonas extremas marcadas · Oscillator com zonas coloridas + EMA suavizada"
)

_plot_df    = _scores_df.tail(_chart_days)
_score_ema  = _plot_df["score"].ewm(span=14, min_periods=5).mean()

col_price, col_dist = st.columns([3, 1], gap="medium")

# ── Main chart: price (top) + score oscillator (bottom) ──────────────────────
with col_price:
    fig_price = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.03,
    )

    # ── ROW 1: Price candlestick ─────────────────────────────────────────────
    # Zone shading on price chart (highlight extreme periods)
    _in_extreme = False; _ext_start = _plot_df.index[0]; _ext_buy = True
    _price_shapes = []

    for _date, _row in _plot_df.iterrows():
        _s = float(_row["score"])
        _is_buy_zone  = _s >= 70
        _is_sell_zone = _s <= 30

        if not _in_extreme and (_is_buy_zone or _is_sell_zone):
            _in_extreme = True
            _ext_start  = _date
            _ext_buy    = _is_buy_zone
        if _in_extreme and not (_is_buy_zone if _ext_buy else _is_sell_zone):
            _price_shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=str(_ext_start)[:10], x1=str(_date)[:10],
                y0=0, y1=1,
                fillcolor="rgba(5,165,70,0.10)" if _ext_buy else "rgba(189,40,40,0.09)",
                line_width=0,
            ))
            _in_extreme = False
    if _in_extreme:
        _price_shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=str(_ext_start)[:10], x1=str(_plot_df.index[-1])[:10],
            y0=0, y1=1,
            fillcolor="rgba(5,165,70,0.10)" if _ext_buy else "rgba(189,40,40,0.09)",
            line_width=0,
        ))

    if "High" in _raw.columns and "Open" in _raw.columns:
        _ohlc = _raw.loc[_plot_df.index]
        fig_price.add_trace(go.Candlestick(
            x=_ohlc.index,
            open=_ohlc["Open"], high=_ohlc["High"],
            low=_ohlc["Low"],   close=_ohlc["Close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            increasing_fillcolor="rgba(38,166,154,0.27)",
            decreasing_fillcolor="rgba(239,83,80,0.27)",
            name=_ticker, showlegend=False,
        ), row=1, col=1)
    else:
        fig_price.add_trace(go.Scatter(
            x=_plot_df.index, y=_plot_df["close"],
            mode="lines", line=dict(color="rgba(200,200,210,.85)", width=1.6),
            name=_ticker, showlegend=False,
        ), row=1, col=1)

    if _show_bb:
        for _cn, _cc, _ds, _nm in [
            ("bb_up", "rgba(100,150,255,.35)", "dot",  "BB Upper"),
            ("bb_ma", "rgba(200,180,60,.35)",  "dash", "BB MA20"),
            ("bb_lo", "rgba(100,150,255,.35)", "dot",  "BB Lower"),
        ]:
            if _cn in _plot_df.columns:
                fig_price.add_trace(go.Scatter(
                    x=_plot_df.index, y=_plot_df[_cn],
                    mode="lines", line=dict(color=_cc, width=0.8, dash=_ds),
                    name=_nm, showlegend=True,
                ), row=1, col=1)

    if _show_ma:
        for _pm, _cm, _nm in [(50,"rgba(255,165,0,.65)","MA50"),(200,"rgba(160,90,255,.65)","MA200")]:
            if len(_raw) >= _pm:
                _ma_s = _raw["Close"].rolling(_pm).mean().loc[_plot_df.index]
                fig_price.add_trace(go.Scatter(
                    x=_plot_df.index, y=_ma_s,
                    mode="lines", line=dict(color=_cm, width=1.2),
                    name=_nm, showlegend=True,
                ), row=1, col=1)

    # ── ROW 2: Score Oscillator (TradingView style) ───────────────────────────
    # Colored zone bands as background
    _osc_zone_cfg = [
        (85, 100, "rgba(5,165,70,0.22)",   "DCA MÁX",       "rgba(5,165,70,.55)"),
        (70,  85, "rgba(10,140,65,0.14)",  "COMPRA FORTE",  "rgba(10,140,65,.40)"),
        (55,  70, "rgba(14,110,60,0.08)",  "OVERSOLD",      "rgba(14,110,60,.30)"),
        (35,  55, "rgba(55,55,60,0.06)",   "NEUTRO",        "rgba(120,120,130,.25)"),
        (20,  35, "rgba(190,90,20,0.10)",  "OVERBOUGHT",    "rgba(190,90,20,.35)"),
        ( 0,  20, "rgba(189,40,40,0.18)",  "VENDA MÁX",     "rgba(189,40,40,.55)"),
    ]

    for _oz_lo, _oz_hi, _oz_fill, _oz_lbl, _oz_line in _osc_zone_cfg:
        fig_price.add_hrect(
            y0=_oz_lo, y1=_oz_hi, fillcolor=_oz_fill, line_width=0,
            row=2, col=1,
        )
        # Zone boundary line
        fig_price.add_hline(
            y=_oz_lo, line_dash="dot",
            line_color=_oz_line, line_width=0.8,
            row=2, col=1,
        )
        # Zone label on right side
        fig_price.add_annotation(
            xref="paper", x=1.01, y=(_oz_lo + _oz_hi) / 2,
            yref="y2",
            text=f"<b>{_oz_lbl}</b>",
            showarrow=False,
            font=dict(size=8, color=_oz_line, family="monospace"),
            xanchor="left", yanchor="middle",
        )

    _sv  = _plot_df["score"].values
    _ev  = _score_ema.values
    _idx = _plot_df.index

    # ── Change 2: Gradient fill — score vs neutral 50 ────────────────────────
    # Above 50 → green (stronger as score rises)
    _above_50 = np.where(_sv >= 50, _sv, 50.0)
    fig_price.add_trace(go.Scatter(
        x=_idx, y=np.full(len(_sv), 50.0),
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig_price.add_trace(go.Scatter(
        x=_idx, y=_above_50,
        mode="none", fill="tonexty",
        fillcolor="rgba(5,165,70,0.13)",
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    # Below 50 → red (stronger as score falls)
    _below_50 = np.where(_sv <= 50, _sv, 50.0)
    fig_price.add_trace(go.Scatter(
        x=_idx, y=np.full(len(_sv), 50.0),
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig_price.add_trace(go.Scatter(
        x=_idx, y=_below_50,
        mode="none", fill="tonexty",
        fillcolor="rgba(189,40,40,0.11)",
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    # ── Change 1: MACD-style fill — score vs EMA ─────────────────────────────
    # Score above EMA → bullish momentum → green fill
    _above_ema = np.where(_sv >= _ev, _sv, _ev)
    fig_price.add_trace(go.Scatter(
        x=_idx, y=_ev,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig_price.add_trace(go.Scatter(
        x=_idx, y=_above_ema,
        mode="none", fill="tonexty",
        fillcolor="rgba(5,165,70,0.28)",
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    # Score below EMA → bearish momentum → red fill
    _below_ema = np.where(_sv <= _ev, _sv, _ev)
    fig_price.add_trace(go.Scatter(
        x=_idx, y=_ev,
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig_price.add_trace(go.Scatter(
        x=_idx, y=_below_ema,
        mode="none", fill="tonexty",
        fillcolor="rgba(189,40,40,0.25)",
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    # ── Change 5: Glow effect — extreme zones only ────────────────────────────
    _glow_buy  = np.where(_sv > 70, _sv, np.nan)
    _glow_sell = np.where(_sv < 30, _sv, np.nan)
    for _gw, _ga_buy, _ga_sell in [
        (14, 0.04, 0.04),
        ( 8, 0.10, 0.09),
        ( 4, 0.22, 0.20),
    ]:
        fig_price.add_trace(go.Scatter(
            x=_idx, y=_glow_buy, mode="lines",
            line=dict(color=f"rgba(5,165,70,{_ga_buy})", width=_gw),
            showlegend=False, hoverinfo="skip", connectgaps=False,
        ), row=2, col=1)
        fig_price.add_trace(go.Scatter(
            x=_idx, y=_glow_sell, mode="lines",
            line=dict(color=f"rgba(189,40,40,{_ga_sell})", width=_gw),
            showlegend=False, hoverinfo="skip", connectgaps=False,
        ), row=2, col=1)

    # ── EMA (smooth reference — blue) ────────────────────────────────────────
    fig_price.add_trace(go.Scatter(
        x=_idx, y=_score_ema,
        mode="lines", name="Score EMA(14)",
        line=dict(color="rgba(100,140,255,0.70)", width=1.6),
        hovertemplate="%{x|%Y-%m-%d}<br>EMA: <b>%{y:.1f}</b><extra></extra>",
    ), row=2, col=1)

    # ── Raw score (main line) ─────────────────────────────────────────────────
    _score_color_line = _hex_to_rgba(_z_color, 0.92)
    fig_price.add_trace(go.Scatter(
        x=_idx, y=_sv,
        mode="lines", name="DCA Score",
        line=dict(color=_score_color_line, width=2.2),
        hovertemplate="%{x|%Y-%m-%d}<br>Score: <b>%{y:.1f}</b><extra></extra>",
    ), row=2, col=1)

    # ── Current score marker ──────────────────────────────────────────────────
    fig_price.add_trace(go.Scatter(
        x=[_idx[-1]], y=[_cur_score],
        mode="markers",
        marker=dict(color=_z_color, size=10, line=dict(color="white", width=1.8)),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    fig_price.update_layout(
        template="plotly_dark",
        height=560,
        margin=dict(l=0, r=90, t=6, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        shapes=_price_shapes,
        legend=dict(
            orientation="h", y=1.04, x=0,
            font=dict(size=9, color="rgba(255,255,255,.45)"),
        ),
    )
    fig_price.update_xaxes(
        showgrid=False, rangeslider_visible=False,
        row=1, col=1,
    )
    fig_price.update_xaxes(
        showgrid=False, color="rgba(255,255,255,.30)",
        row=2, col=1,
    )
    fig_price.update_yaxes(
        showgrid=True, gridcolor="rgba(255,255,255,.04)",
        row=1, col=1,
    )
    fig_price.update_yaxes(
        range=[0, 100],
        showgrid=False,
        tickvals=[20, 35, 55, 70, 85],
        tickfont=dict(size=8, color="rgba(255,255,255,.25)"),
        row=2, col=1,
    )
    st.plotly_chart(fig_price, use_container_width=True)

# ── Distribution Histogram ────────────────────────────────────────────────────
with col_dist:
    _hist_scores = _scores_df["score"].dropna().values
    _bins        = np.arange(0, 105, 5)
    _counts, _   = np.histogram(_hist_scores, bins=_bins)
    _bin_mids    = (_bins[:-1] + _bins[1:]) / 2

    def _hist_color(v):
        if v < 20:   return "rgba(189,40,40,.78)"
        if v < 35:   return "rgba(190,90,20,.68)"
        if v < 55:   return "rgba(60,60,70,.60)"
        if v < 70:   return "rgba(14,110,60,.68)"
        if v < 85:   return "rgba(10,140,65,.74)"
        return "rgba(5,165,70,.82)"

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(
        x=_counts, y=_bin_mids,
        orientation="h", width=4.5,
        marker=dict(color=[_hist_color(v) for v in _bin_mids], line_width=0),
        hovertemplate="Score %{y:.0f}<br>%{x} dias<extra></extra>",
        showlegend=False,
    ))

    # Current score marker
    fig_dist.add_shape(
        type="line", x0=0, x1=max(_counts) * 1.15,
        y0=_cur_score, y1=_cur_score,
        line=dict(color="white", width=1.8, dash="dot"),
    )
    fig_dist.add_annotation(
        x=max(_counts) * 0.5, y=_cur_score + 3,
        text=f"<b>← {_cur_score:.0f}</b>",
        showarrow=False, font=dict(size=10, color="white"),
    )

    fig_dist.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=0, r=10, t=6, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Dias", showgrid=True, gridcolor="rgba(255,255,255,.04)",
                   color="rgba(255,255,255,.30)"),
        yaxis=dict(range=[0, 100], title="Score",
                   showgrid=True, gridcolor="rgba(255,255,255,.04)",
                   color="rgba(255,255,255,.30)",
                   tickvals=[0, 20, 35, 55, 70, 85, 100]),
        showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR TABLE
# ─────────────────────────────────────────────────────────────────────────────

_divider()
_section_header(
    "Decomposição do Score — Indicadores",
    "Valor raw de cada indicador · score 0–100 · contribuição para o composite"
)

_rsi14_v  = float(_cur["rsi14"])
_rsi2_v   = float(_cur["rsi2"])
_z_v      = float(_cur["z"])
_pct_b_v  = float(_cur["pct_b"])
_vol_v    = float(_cur["vol_score"])

_ind_rows = [
    ("RSI(14)",    f"{_rsi14_v:.1f}",          float(_cur["rsi14_score"]), 0.20,
     "< 30 oversold · > 70 overbought"),
    ("RSI(2)",     f"{_rsi2_v:.1f}",            float(_cur["rsi2_score"]),  0.10,
     "< 10 extremo oversold · > 90 extremo overbought"),
    ("Z-score",    f"{_z_v:+.2f}σ",             float(_cur["z_score"]),     0.30,
     "Desvios-padrão abaixo da média 1A · < −2 = barato"),
    ("BB %B",      f"{_pct_b_v:+.3f}",          float(_cur["bb_score"]),    0.20,
     "< 0 = abaixo lower band · > 1 = acima upper band"),
    ("Vol Climax", f"{_vol_v:.1f} (score)",      _vol_v,                     0.20,
     "Volume alto em queda = capitulação · em subida = blowoff"),
]

def _score_bar_html(score_val, width_px=80):
    bc = (_z_color if score_val >= 70 else
          "#26a69a" if score_val >= 55 else
          "#ef5350" if score_val <= 20 else
          "#be5a14" if score_val <= 35 else
          "rgba(140,140,150,.60)")
    pct = int(score_val)
    return (f"<div style='display:flex;align-items:center;gap:8px'>"
            f"<div style='width:{width_px}px;height:5px;border-radius:3px;"
            f"background:rgba(255,255,255,.08)'>"
            f"<div style='width:{pct}%;height:100%;border-radius:3px;"
            f"background:{bc}'></div></div>"
            f"<span style='font-size:.78em;font-weight:700;color:{bc}'>{score_val:.1f}</span>"
            f"</div>")

_total_contrib = sum(s * w for _, _, s, w, _ in _ind_rows)

rows_ind = ""
for name, raw_val, score_val, weight, desc in _ind_rows:
    contrib    = score_val * weight
    contrib_c  = (_z_color if contrib >= 14 else
                  "#26a69a" if contrib >= 8 else
                  "#ef5350" if contrib <= 4 else
                  "rgba(180,180,180,.70)")
    rows_ind += (
        f"<tr style='border-bottom:1px solid rgba(255,255,255,.04)'>"
        f"<td style='padding:9px 14px'>"
        f"<span style='font-size:.82em;font-weight:700;color:rgba(255,255,255,.80);"
        f"font-family:monospace'>{name}</span>"
        f"<div style='font-size:.68em;color:rgba(255,255,255,.28);margin-top:2px'>{desc}</div>"
        f"</td>"
        f"<td style='padding:9px 14px;font-size:.82em;font-weight:700;"
        f"color:rgba(255,255,255,.70);font-family:monospace'>{raw_val}</td>"
        f"<td style='padding:9px 14px'>{_score_bar_html(score_val)}</td>"
        f"<td style='padding:9px 14px;font-size:.80em;color:rgba(255,255,255,.40)'>"
        f"{int(weight*100)}%</td>"
        f"<td style='padding:9px 14px;font-size:.84em;font-weight:700;color:{contrib_c}'>"
        f"{contrib:+.1f} pts</td>"
        f"</tr>"
    )

# Total row
rows_ind += (
    f"<tr style='background:rgba(255,255,255,.03)'>"
    f"<td style='padding:10px 14px;font-size:.84em;font-weight:700;"
    f"color:rgba(255,255,255,.60);letter-spacing:1px'>COMPOSITE</td>"
    f"<td style='padding:10px 14px'></td>"
    f"<td style='padding:10px 14px'>{_score_bar_html(_cur_score, 100)}</td>"
    f"<td style='padding:10px 14px;font-size:.80em;color:rgba(255,255,255,.40)'>100%</td>"
    f"<td style='padding:10px 14px;font-size:.88em;font-weight:900;color:{_z_color}'>"
    f"{_cur_score:.1f} pts</td>"
    f"</tr>"
)

st.markdown(
    f"<div style='background:#0e0e1c;border-radius:12px;overflow:hidden;"
    f"border:1px solid rgba(255,255,255,.07)'>"
    f"<table style='width:100%;border-collapse:collapse'>"
    f"<thead><tr style='background:rgba(255,255,255,.04)'>"
    f"<th style='padding:8px 14px;text-align:left;font-size:.62em;letter-spacing:1.8px;"
    f"color:rgba(255,255,255,.28)'>INDICADOR</th>"
    f"<th style='padding:8px 14px;text-align:left;font-size:.62em;letter-spacing:1.8px;"
    f"color:rgba(255,255,255,.28)'>VALOR</th>"
    f"<th style='padding:8px 14px;text-align:left;font-size:.62em;letter-spacing:1.8px;"
    f"color:rgba(255,255,255,.28)'>SCORE (0–100)</th>"
    f"<th style='padding:8px 14px;text-align:left;font-size:.62em;letter-spacing:1.8px;"
    f"color:rgba(255,255,255,.28)'>PESO</th>"
    f"<th style='padding:8px 14px;text-align:left;font-size:.62em;letter-spacing:1.8px;"
    f"color:rgba(255,255,255,.28)'>CONTRIBUIÇÃO</th>"
    f"</tr></thead><tbody>{rows_ind}</tbody></table></div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# SPAGHETTI PATHS
# ─────────────────────────────────────────────────────────────────────────────

_divider()
_section_header(
    "Spaghetti Paths — Retorno após entrar nesta zona",
    f"Cada linha = 1 ocorrência histórica em {_z_label} · Linha branca = média · "
    f"Eixo X = dias após sinal · {len(_spag_paths)} ocorrências encontradas"
)

if len(_spag_paths) >= 2:
    _spag_days = list(range(len(_spag_paths[0])))
    _avg_path  = [
        np.mean([p[d] for p in _spag_paths]) for d in _spag_days
    ]

    fig_spag = go.Figure()

    # Individual paths
    _n_spag = len(_spag_paths)
    for _pi, _path in enumerate(_spag_paths):
        _alpha = max(0.12, 0.30 - _pi * 0.02)
        _col_spag = (f"rgba(5,165,70,{_alpha:.2f})"  if _cur_score >= 55 else
                     f"rgba(189,40,40,{_alpha:.2f})")
        fig_spag.add_trace(go.Scatter(
            x=_spag_days, y=_path,
            mode="lines", line=dict(color=_col_spag, width=1.1),
            showlegend=False,
            hovertemplate=f"Ocorrência {_pi+1}<br>Dia %{{x}}: %{{y:+.1f}}%<extra></extra>",
        ))

    # Average path
    fig_spag.add_trace(go.Scatter(
        x=_spag_days, y=_avg_path,
        mode="lines", name="Média histórica",
        line=dict(color="rgba(255,255,255,.90)", width=2.5),
        hovertemplate="Média<br>Dia %{x}: %{y:+.1f}%<extra></extra>",
    ))

    # Zero line
    fig_spag.add_hline(y=0, line_dash="dot",
                        line_color="rgba(255,255,255,.25)", line_width=1.0)

    # Horizon markers
    for _hd, _hl in [(90,"90d"), (180,"180d")]:
        if _hd <= len(_spag_days):
            fig_spag.add_vline(x=_hd, line_dash="dot",
                                line_color="rgba(255,255,255,.20)", line_width=1)
            fig_spag.add_annotation(
                x=_hd, y=max(_avg_path) * 0.9 if _avg_path else 20,
                text=f"<b>{_hl}</b>", showarrow=False,
                font=dict(size=9, color="rgba(255,255,255,.40)"),
            )

    fig_spag.update_layout(
        template="plotly_dark", height=280,
        margin=dict(l=0, r=10, t=6, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Dias após sinal", showgrid=True,
                   gridcolor="rgba(255,255,255,.04)",
                   color="rgba(255,255,255,.35)"),
        yaxis=dict(title="Retorno %", showgrid=True,
                   gridcolor="rgba(255,255,255,.04)",
                   color="rgba(255,255,255,.35)", ticksuffix="%"),
        legend=dict(orientation="h", y=1.06, x=0,
                    font=dict(size=9, color="rgba(255,255,255,.50)")),
    )
    st.plotly_chart(fig_spag, use_container_width=True)
else:
    st.caption(
        f"Dados insuficientes para spaghetti paths na zona **{_z_label}** "
        f"({len(_spag_paths)} ocorrências). Aumenta o histórico para ver caminhos futuros."
    )

# ─────────────────────────────────────────────────────────────────────────────
# READING GUIDE
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("📖 Como ler o Smart DCA Compass"):
    st.markdown(f"""
**Sistema Long-Term DCA — apenas actuar em extremos**

Este indicador foi desenhado para investidores de longo prazo que querem acumular em zonas de capitulação e reduzir em zonas de euforia. Os sinais no meio (Neutro 35–55) são ignorados intencionalmente.

**Score (0–100)**

| Score | Zona | Ação |
|---|---|---|
| 85–100 | EXTREME OVERSOLD | DCA Máximo — activar todas as tranches |
| 70–85 | STRONG OVERSOLD | Compra Forte — 2ª/3ª tranche |
| 55–70 | OVERSOLD | Compra Parcial — 1ª tranche |
| 35–55 | NEUTRO | Aguardar — sem acção |
| 20–35 | OVERBOUGHT | Reduzir — realizar lucros |
| 0–20 | EXTREME OVERBOUGHT | Vender Máximo — sair ou liquidar |

**Percentile Rank**
Mostra em que percentil da história recente o score se encontra.
Score no percentil 7 = o ativo está mais barato que 93% dos dias históricos → zona rara de acumulação.

**Spaghetti Paths**
Todas as vezes que o score entrou na zona atual, o que aconteceu ao preço nos 90/180 dias seguintes.
Cada linha fina = 1 ocorrência. Linha branca = retorno médio histórico.

**Indicator Table**
| Indicador | Oversold | Overbought |
|---|---|---|
| RSI(14) | < 30 | > 70 |
| RSI(2) | < 10 | > 90 |
| Z-score | < −2σ | > +2σ |
| BB %B | < 0 | > 1 |
| Vol Climax | Alto volume em quedas | Alto volume em subidas |
    """)

"""
🌐 Macro Regime Compass
========================
5 macro indicators → single regime score  (0 to 100)
Score scale: 0 = extreme risk-off · 50 = neutral · 100 = extreme risk-on
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
from datetime import datetime
import streamlit as st
import requests
import yfinance as yf
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Macro Regime Compass",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _rolling_zscore(s: pd.Series, window: int) -> pd.Series:
    m = s.rolling(window, min_periods=max(3, window // 2)).mean()
    d = s.rolling(window, min_periods=max(3, window // 2)).std().replace(0, np.nan)
    return (s - m) / d

def _pct_rank_series(s: pd.Series, window: int = 252) -> pd.Series:
    return (
        s.rolling(window, min_periods=max(10, window // 4))
         .apply(lambda x: float((x[:-1] < x[-1]).mean() * 100), raw=True)
    )

def _safe_last(s: pd.Series, default: float = 0.0) -> float:
    c = s.dropna()
    return float(c.iloc[-1]) if not c.empty else default

def _yf_close(ticker: str, period: str = "2y") -> pd.Series:
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return pd.Series(dtype=float)
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.squeeze()
        close.index = pd.to_datetime(close.index).normalize()
        return pd.to_numeric(close, errors="coerce").dropna()
    except Exception:
        return pd.Series(dtype=float)

def _hex_rgba(hex_c: str, alpha: float) -> str:
    """Convert '#rrggbb' hex colour + alpha float → 'rgba(r,g,b,a)' for Plotly."""
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"

def _divider() -> None:
    """Gradient divider — replaces plain st.markdown('---')."""
    st.markdown(
        "<div style='height:1px;"
        "background:linear-gradient(90deg,"
        "transparent 0%,rgba(38,166,154,.18) 20%,"
        "rgba(255,255,255,.06) 50%,"
        "rgba(38,166,154,.18) 80%,transparent 100%);"
        "margin:18px 0 6px'></div>",
        unsafe_allow_html=True,
    )

def _mini_spark_svg(values: np.ndarray, color: str,
                    w: int = 64, h: int = 20) -> str:
    """Inline SVG sparkline from an array of values. Returns empty string on failure."""
    vals = np.array(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) < 2:
        return ""
    mn, mx = float(vals.min()), float(vals.max())
    rng = (mx - mn) or 1.0
    pts = []
    for i, v in enumerate(vals):
        x = round(i / (len(vals) - 1) * w, 1)
        y = round(h - (v - mn) / rng * (h - 4) - 2, 1)
        pts.append(f"{x},{y}")
    pts_str = " ".join(pts)
    return (
        f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' "
        f"style='vertical-align:middle;overflow:visible'>"
        f"<polyline points='{pts_str}' fill='none' "
        f"stroke='{color}' stroke-width='1.5' "
        f"stroke-linecap='round' stroke-linejoin='round' opacity='.82'/>"
        f"</svg>"
    )

def _section_header(title: str, subtitle: str = "") -> None:
    sub = (f"<div style='font-size:.79em;color:rgba(255,255,255,.45);"
           f"margin-top:2px'>{subtitle}</div>") if subtitle else ""
    st.markdown(
        f"<div style='border-left:3px solid rgba(38,166,154,.65);"
        f"padding:4px 0 4px 12px;margin:16px 0 8px'>"
        f"<span style='font-size:.95em;font-weight:600;"
        f"color:rgba(255,255,255,.92)'>{title}</span>{sub}</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SEMICIRCULAR ARC GAUGE
# ─────────────────────────────────────────────────────────────────────────────

def _build_arc_gauge(display_score: float, color: str, label: str) -> go.Figure:
    """
    Custom semicircular gauge built entirely from Plotly Scatter traces + shapes.
    Arc sweeps left (score=0, angle=π) → top (score=50, angle=π/2) → right (score=100, angle=0).
    Uses scaleanchor='x' so arcs appear perfectly circular.
    """
    R_OUT    = 1.00   # outer arc radius
    R_IN     = 0.58   # inner arc radius  (track width = 0.42)
    R_NEEDLE = 0.82   # needle tip
    R_HUB    = 0.055  # centre hub

    def s2a(score: float) -> float:
        """Score [0,100] → angle [π, 0] radians."""
        return np.pi * (1.0 - score / 100.0)

    def annular_arc(s0: float, s1: float,
                    r_in: float, r_out: float, n: int = 90):
        a_s  = s2a(s0)
        a_e  = s2a(s1)
        fwd  = np.linspace(a_s, a_e, n)
        rev  = np.linspace(a_e, a_s, n)
        x = np.r_[r_out * np.cos(fwd), r_in * np.cos(rev), r_out * np.cos(a_s)]
        y = np.r_[r_out * np.sin(fwd), r_in * np.sin(rev), r_out * np.sin(a_s)]
        return x, y

    # (fill_rgba, label, label_color)
    _zones = [
        (  0.0,  30.0, "rgba(155, 25, 25, 0.95)",  "RISK-OFF", "rgba(225, 85, 85,.95)"),
        ( 30.0,  42.5, "rgba(165, 82, 16, 0.90)",  "MILD OFF", "rgba(215,125, 45,.95)"),
        ( 42.5,  57.5, "rgba( 58, 58, 58, 0.88)",  "NEUTRAL",  "rgba(185,185,185,.90)"),
        ( 57.5,  70.0, "rgba( 18,124, 68, 0.90)",  "MILD ON",  "rgba( 55,200,125,.95)"),
        ( 70.0, 100.0, "rgba( 15,152, 70, 0.95)",  "RISK-ON",  "rgba( 45,225,110,.95)"),
    ]

    fig = go.Figure()

    # 1 ── Dark background track (full semicircle)
    xb, yb = annular_arc(0, 100, R_IN * 0.97, R_OUT * 1.02, n=200)
    fig.add_trace(go.Scatter(x=xb, y=yb, fill="toself",
                             fillcolor="rgba(14,14,22,.96)",
                             line=dict(color="rgba(255,255,255,.04)", width=.5),
                             showlegend=False, hoverinfo="skip"))

    # 2 ── Coloured zone arcs
    for s0, s1, fill, _, _ in _zones:
        xz, yz = annular_arc(s0, s1, R_IN, R_OUT)
        fig.add_trace(go.Scatter(x=xz, y=yz, fill="toself",
                                 fillcolor=fill,
                                 line=dict(color="rgba(8,8,8,.55)", width=1.0),
                                 showlegend=False, hoverinfo="skip"))

    # 2b ── Dim inactive zones so the active zone stands out
    for s0, s1, _, _, _ in _zones:
        if not (s0 <= display_score <= s1):
            xd, yd = annular_arc(s0, s1, R_IN, R_OUT)
            fig.add_trace(go.Scatter(x=xd, y=yd, fill="toself",
                                     fillcolor="rgba(0,0,0,.20)",
                                     line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))

    # 3 ── Active zone: white brightness overlay + glowing border
    active = next((z for z in _zones if z[0] <= display_score <= z[1]), _zones[-1])
    xab, yab = annular_arc(active[0], active[1], R_IN, R_OUT)
    fig.add_trace(go.Scatter(x=xab, y=yab, fill="toself",
                             fillcolor="rgba(255,255,255,.12)",
                             line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    xa, ya = annular_arc(active[0], active[1], R_IN - .015, R_OUT + .03)
    fig.add_trace(go.Scatter(x=xa, y=ya, fill="toself",
                             fillcolor="rgba(0,0,0,0)",
                             line=dict(color=color, width=3.0),
                             showlegend=False, hoverinfo="skip"))

    # 4 ── Scale tick marks (at 0, 25, 50, 75, 100)
    for tick in [0, 25, 50, 75, 100]:
        a_t = s2a(tick)
        fig.add_trace(go.Scatter(
            x=[R_OUT * np.cos(a_t), 1.10 * np.cos(a_t)],
            y=[R_OUT * np.sin(a_t), 1.10 * np.sin(a_t)],
            mode="lines", line=dict(color="rgba(255,255,255,.60)", width=1.8),
            showlegend=False, hoverinfo="skip"))
        fig.add_annotation(
            x=1.19 * np.cos(a_t), y=1.19 * np.sin(a_t),
            text=f"<b>{tick}</b>", showarrow=False,
            font=dict(size=12, color="rgba(255,255,255,.85)"), align="center")

    # 5 ── Zone labels — each coloured to match its zone
    for s0, s1, _, zlbl, zlbl_col in _zones:
        a_mid = s2a((s0 + s1) / 2)
        fig.add_annotation(
            x=1.42 * np.cos(a_mid), y=1.42 * np.sin(a_mid),
            text=f"<b>{zlbl}</b>", showarrow=False,
            font=dict(size=13, color=zlbl_col, family="monospace"),
            align="center")

    # 6 ── Tapered needle — diamond shape (wide at hub, sharp tip, small tail)
    a_n = s2a(display_score)
    _bw = 0.030   # half-width at pivot
    _bk = 0.08    # tail length (hidden under hub)

    _ndl_x = [
        R_NEEDLE * np.cos(a_n),            # ① tip
        _bw * np.cos(a_n + np.pi / 2),     # ② left side at hub
        _bk * np.cos(a_n + np.pi),         # ③ tail (hidden by hub)
        _bw * np.cos(a_n - np.pi / 2),     # ④ right side at hub
        R_NEEDLE * np.cos(a_n),            # close
    ]
    _ndl_y = [
        R_NEEDLE * np.sin(a_n),
        _bw * np.sin(a_n + np.pi / 2),
        _bk * np.sin(a_n + np.pi),
        _bw * np.sin(a_n - np.pi / 2),
        R_NEEDLE * np.sin(a_n),
    ]
    fig.add_trace(go.Scatter(
        x=_ndl_x, y=_ndl_y, fill="toself",
        fillcolor="rgba(245,245,250,.97)",
        line=dict(color="rgba(200,200,220,.55)", width=0.8),
        showlegend=False, hoverinfo="skip"))

    # 7 ── Centre hub: outer glow ring + inner regime-coloured disc
    th_h = np.linspace(0, 2 * np.pi, 60)
    fig.add_trace(go.Scatter(                        # glow ring
        x=(R_HUB + 0.028) * np.cos(th_h),
        y=(R_HUB + 0.028) * np.sin(th_h),
        fill="toself", fillcolor="rgba(255,255,255,.10)",
        line=dict(color="rgba(255,255,255,0)", width=0),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(                        # inner disc
        x=R_HUB * np.cos(th_h), y=R_HUB * np.sin(th_h),
        fill="toself", fillcolor=color,
        line=dict(color="rgba(255,255,255,.90)", width=2.0),
        showlegend=False, hoverinfo="skip"))

    # 8 ── Score number centred below the baseline (pushed down to clear needle base)
    fig.add_annotation(x=0, y=-0.26,
                       text=f"<b>{display_score:.0f}</b>", showarrow=False,
                       font=dict(size=62, color=color,
                                 family="Arial Black, Arial, sans-serif"))

    fig.update_layout(
        xaxis=dict(range=[-1.55, 1.55], visible=False,
                   showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.62, 1.58], visible=False,
                   showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=450,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING  (cached 1× per day)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400 * 2, show_spinner=False)
def _load_yahoo(cache_date: str) -> dict[str, pd.Series]:   # noqa: ARG001
    tickers = {"VIX": "^VIX", "DXY": "DX-Y.NYB", "Gold": "GC=F",
               "US10Y": "^TNX", "US3M": "^IRX", "SP500": "^GSPC"}
    return {name: _yf_close(tkr) for name, tkr in tickers.items()}

@st.cache_data(ttl=86400 * 2, show_spinner=False)
def _load_fear_greed(cache_date: str, limit: int = 500) -> pd.DataFrame:  # noqa: ARG001
    try:
        url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
        r   = requests.get(url, timeout=12)
        raw = r.json()["data"]
        idx  = pd.to_datetime([int(d["timestamp"]) for d in raw], unit="s").normalize()
        vals = [int(d["value"]) for d in raw]
        lbls = [d["value_classification"] for d in raw]
        return pd.DataFrame({"value": vals, "label": lbls}, index=idx).sort_index()
    except Exception:
        return pd.DataFrame(columns=["value", "label"])

@st.cache_data(ttl=86400 * 2, show_spinner=False)
def _load_fred_spread(cache_date: str) -> pd.Series:   # noqa: ARG001
    """ICE BofA US High Yield OAS — tries FRED first, falls back to Yahoo HYG.
    High spread / low HYG = credit stress = risk-off  (risk_sign = -1).
    """
    from io import StringIO

    # ── Attempt 1: FRED BAMLH0A0HYM2 (real OAS in %) ──────────────────────
    try:
        _headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
            ),
            "Accept": "text/html,text/csv,*/*",
        }
        url = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
               "?id=BAMLH0A0HYM2")
        r = requests.get(url, timeout=15, headers=_headers)
        if r.status_code == 200 and "DATE" in r.text:
            df = pd.read_csv(StringIO(r.text),
                             parse_dates=[0], index_col=0)
            df.index = pd.to_datetime(df.index).normalize()
            s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
            s = s[s > 0]
            if len(s) >= 30:
                return s
    except Exception:
        pass

    # ── Attempt 2: Yahoo Finance — HYG ETF as credit proxy ─────────────────
    # HYG price falling = spreads widening = risk-off
    # We invert: spread_proxy = 200 - HYG  so that higher value = more stress
    # The z-score + risk_sign=-1 then correctly signals risk-off
    try:
        hyg = _yf_close("HYG", period="5y")
        if not hyg.empty:
            return (200.0 - hyg).rename("CreditSpread_proxy")
    except Exception:
        pass

    return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# REGIME COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

_INDICATOR_META = {
    "VIX":          {"label": "Volatility (VIX)",    "unit": "pts", "risk_sign": -1, "icon": "😱"},
    "DXY":          {"label": "US Dollar (DXY)",      "unit": "pts", "risk_sign": -1, "icon": "💵"},
    "Gold":         {"label": "Gold (Safe Haven)",    "unit": "USD", "risk_sign": -1, "icon": "🥇"},
    "YieldCurve":   {"label": "Yield Curve 10Y-3M",  "unit": "%",   "risk_sign": +1, "icon": "📈"},
    "FearGreed":    {"label": "Fear & Greed",         "unit": "/100","risk_sign": +1, "icon": "🎰"},
    "CreditSpread": {"label": "HY Credit Spread",     "unit": "idx", "risk_sign": -1, "icon": "📉"},
}

def _compute(ydata, fg, z_window, pct_window=252):
    indicators: dict = {}

    for key, tkr_key, extra in [
        ("VIX",          "VIX",          None),
        ("DXY",          "DXY",          None),
        ("Gold",         "Gold",         "pct5"),
        ("CreditSpread", "CreditSpread", None),
    ]:
        raw = ydata.get(tkr_key, pd.Series())
        if raw.empty:
            continue
        s  = raw.pct_change(5) * 100 if extra == "pct5" else raw
        z  = _rolling_zscore(s, z_window)
        pr = _pct_rank_series(raw, pct_window)
        indicators[key] = dict(series=raw, zseries=z, prseries=pr,
                                current=_safe_last(raw), z=_safe_last(z),
                                pct=_safe_last(pr, 50))

    us10 = ydata.get("US10Y", pd.Series())
    us3m = ydata.get("US3M",  pd.Series())
    if not us10.empty and not us3m.empty:
        spread = us10.subtract(us3m, fill_value=np.nan).dropna()
        z  = _rolling_zscore(spread, z_window)
        pr = _pct_rank_series(spread, pct_window)
        indicators["YieldCurve"] = dict(series=spread, zseries=z, prseries=pr,
                                         current=_safe_last(spread), z=_safe_last(z),
                                         pct=_safe_last(pr, 50))

    if not fg.empty:
        s  = fg["value"].astype(float)
        z  = _rolling_zscore(s, z_window)
        pr = _pct_rank_series(s, min(pct_window, len(s)))
        indicators["FearGreed"] = dict(series=s, zseries=z, prseries=pr,
                                        current=_safe_last(s, 50), z=_safe_last(z),
                                        pct=_safe_last(pr, 50))

    n = len(indicators)
    score_parts: dict = {}
    if n:
        w = 100.0 / (n * 3.0)
        for k, v in indicators.items():
            sign = _INDICATOR_META[k]["risk_sign"]
            score_parts[k] = round(sign * float(np.clip(v["z"], -3, 3)) * w, 1)

    total = float(np.clip(sum(score_parts.values()), -100, 100))

    z_frames = {}
    for k, v in indicators.items():
        zs = v["zseries"].copy()
        zs.index = pd.to_datetime(zs.index).normalize()
        z_frames[k] = zs

    if z_frames:
        hdf  = pd.DataFrame(z_frames).sort_index()
        hist = pd.Series(0.0, index=hdf.index)
        hw   = 100.0 / (len(z_frames) * 3.0)
        for k, col in hdf.items():
            hist += _INDICATOR_META[k]["risk_sign"] * col.clip(-3, 3) * hw
        hist = hist.clip(-100, 100)
    else:
        hist = pd.Series(dtype=float)

    return dict(indicators=indicators, score_parts=score_parts,
                total_score=total, hist_score=hist)


# ─────────────────────────────────────────────────────────────────────────────
# DAILY REFRESH TIMER  (anchored to 00:00 UTC — independent of when page was loaded)
# ─────────────────────────────────────────────────────────────────────────────

_now_utc       = datetime.utcnow()
_today_key     = _now_utc.strftime('%Y-%m-%d')        # cache key changes at midnight UTC
_secs_into_day = (_now_utc.hour * 3600
                  + _now_utc.minute * 60
                  + _now_utc.second)
_remaining_s   = max(0.0, float(86400 - _secs_into_day))  # true time until 00:00 UTC
_TTL           = 86400                                    # used for ring geometry only


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("🌐 Macro Regime")
st.sidebar.caption("Global macro risk — no volume, no curve fitting.")

_lookback_map = {"1 Month":30,"3 Months":90,"6 Months":180,"1 Year":365,"2 Years":730}
lookback_opt  = st.sidebar.selectbox("Chart History", list(_lookback_map), index=2)
lookback_days = _lookback_map[lookback_opt]

z_window = st.sidebar.slider("Z-Score Window (days)", 10, 60, 20,
    help="Rolling window for z-score normalisation.")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

_auto_refresh_on = st.sidebar.checkbox(
    "⚡ Auto-Refresh", value=True,
    help="Automatically reloads the page when the daily data cache expires."
)

# ── Live circular countdown ────────────────────────────────────────────────
_rm      = int(_remaining_s)
# Human-readable format: "23h 45m" when hours remain, "45:30" in last hour
_rm_str  = (f"{_rm//3600}h {(_rm%3600)//60}m" if _rm >= 3600
            else f"{_rm//60}:{_rm%60:02d}")
_circ    = 94.25                                        # circumference = 2π × r15
_off0    = f"{_circ * (1.0 - _remaining_s / _TTL):.2f}"
_auto_js = "true" if _auto_refresh_on else "false"

_countdown_html = f"""<style>
body{{margin:0;padding:0;background:transparent}}
.w{{display:flex;align-items:center;gap:10px;padding:6px 0 0 2px}}
.lb{{font-size:9px;letter-spacing:1.8px;color:rgba(255,255,255,.30);
     font-family:sans-serif;margin-bottom:2px}}
#t{{font-size:16px;font-weight:700;font-family:monospace;
    color:rgba(38,166,154,.95);transition:color .5s}}
</style>
<div class='w'>
  <svg width='38' height='38' viewBox='0 0 38 38'>
    <circle cx='19' cy='19' r='15' fill='none'
            stroke='rgba(255,255,255,.07)' stroke-width='3'/>
    <circle id='ring' cx='19' cy='19' r='15' fill='none'
            stroke='rgba(38,166,154,.82)' stroke-width='3'
            stroke-dasharray='{_circ:.2f} {_circ:.2f}'
            stroke-dashoffset='{_off0}'
            stroke-linecap='round' transform='rotate(-90 19 19)'/>
  </svg>
  <div>
    <div class='lb'>REFRESH IN</div>
    <div id='t'>{_rm_str}</div>
  </div>
</div>
<script>
(function(){{
  let r={_rm},total={_TTL};
  const ring=document.getElementById('ring'),tel=document.getElementById('t');
  const C={_circ:.2f},auto={_auto_js};
  function tick(){{
    if(r<=0){{
      tel.textContent='↻ now';
      tel.style.color='rgba(255,220,50,.95)';
      ring.style.stroke='rgba(255,220,50,.80)';
      if(auto)setTimeout(()=>window.parent.location.reload(),800);
      return;
    }}
    tel.textContent=r>=3600?Math.floor(r/3600)+'h '+Math.floor((r%3600)/60)+'m':Math.floor(r/60)+':'+String(Math.floor(r%60)).padStart(2,'0');
    ring.style.strokeDashoffset=C*(1-r/total);
    if(r/total>.50){{
      tel.style.color='rgba(38,166,154,.95)';
      ring.style.stroke='rgba(38,166,154,.82)';
    }}else if(r/total>.20){{
      tel.style.color='rgba(255,167,38,.95)';
      ring.style.stroke='rgba(255,167,38,.82)';
    }}else{{
      tel.style.color='rgba(239,83,80,.95)';
      ring.style.stroke='rgba(239,83,80,.82)';
    }}
    r--;setTimeout(tick,1000);
  }}
  tick();
}})();
</script>"""

with st.sidebar:
    components.html(_countdown_html, height=68)

st.sidebar.markdown("---")
st.sidebar.caption("**Data sources:**\n- Yahoo Finance (VIX, DXY, Gold, Yields)\n- Alternative.me (Fear & Greed)\n- FRED BAMLH0A0HYM2 (Credit Spread — scored only when FRED available)")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & COMPUTE
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Loading macro data..."):
    ydata = _load_yahoo(_today_key)
    fg_df = _load_fear_greed(_today_key)
    _cs   = _load_fred_spread(_today_key)

# Distinguish real FRED OAS from HYG proxy:
# Real HY OAS is always 3–20 % → mean < 50
# HYG proxy (200 - price) is ~115–130 → mean > 50
_cs_is_real_oas = (not _cs.empty and
                   float(_cs.dropna().mean()) < 50)

if _cs_is_real_oas:
    # ✅ Real FRED data — include in score & historical calculation
    ydata["CreditSpread"] = _cs
# If proxy or empty: don't inject → score stays based on 5 reliable indicators

result = _compute(ydata, fg_df, z_window)
ind    = result["indicators"]
total  = result["total_score"]
parts  = result["score_parts"]
hist   = result["hist_score"]

display_score = round((total + 100) / 2, 1)   # -100→0, 0→50, +100→100

# Guard: warn if no indicators loaded (score would silently show 50/NEUTRAL)
if not ind:
    st.error("⚠️ No indicator data could be loaded. Check your internet connection "
             "and click **🔄 Refresh Data** in the sidebar.", icon="🚨")

# ── Regime label ──────────────────────────────────────────────────────────────
if   display_score >= 70.0: label, color, icon = "RISK-ON",       "#1a9b5c", "🟢"
elif display_score >= 57.5: label, color, icon = "MILD RISK-ON",  "#2d9a50", "🟡"
elif display_score >= 42.5: label, color, icon = "NEUTRAL",       "#505050", "⚪"
elif display_score >= 30.0: label, color, icon = "MILD RISK-OFF", "#8a5520", "🟠"
else:                       label, color, icon = "RISK-OFF",      "#9b2020", "🔴"

_descs = {
    "RISK-ON":       "Strong risk appetite — favorable macro backdrop for long positions.",
    "MILD RISK-ON":  "Positive macro bias — normal risk-taking environment.",
    "NEUTRAL":       "Mixed signals — no strong directional edge from macro.",
    "MILD RISK-OFF": "Macro headwinds building — reduce size, tighten stops.",
    "RISK-OFF":      "Defensive environment — capital preservation over aggressive longs.",
}

_card_order = ["VIX", "DXY", "Gold", "YieldCurve", "FearGreed"]

# ── Score momentum: delta vs 1 day and 7 days ago ─────────────────────────────
_hist_disp_all = (hist + 100) / 2
_hda           = _hist_disp_all.dropna()

def _delta_style(d):
    """(formatted_str, css_color, arrow) for a score delta value."""
    if d is None:
        return "—", "rgba(255,255,255,.28)", "─"
    if d > 0.4:
        return f"+{d:.1f}", "#26a69a", "↑"
    if d < -0.4:
        return f"{d:.1f}", "#ef5350", "↓"
    return f"{d:+.1f}", "rgba(255,255,255,.42)", "→"

# iloc[-2] = yesterday, iloc[-6] = ~5 trading days (1 week), iloc[-22] = ~1 month
_d1  = round(display_score - float(_hda.iloc[-2]),  1) if len(_hda) >= 2  else None
_d7  = round(display_score - float(_hda.iloc[-6]),  1) if len(_hda) >= 6  else None   # ~5td ≈ 1wk
_d30 = round(display_score - float(_hda.iloc[-22]), 1) if len(_hda) >= 22 else None   # ~1 month
_d1_val,  _d1_col,  _d1_arrow  = _delta_style(_d1)
_d7_val,  _d7_col,  _d7_arrow  = _delta_style(_d7)
_d30_val, _d30_col, _d30_arrow = _delta_style(_d30)

# ── Regime duration — how long have we been in the current regime? ────────────
def _regime_of(s: float) -> str:
    if   s >= 70.0: return "RISK-ON"
    elif s >= 57.5: return "MILD RISK-ON"
    elif s >= 42.5: return "NEUTRAL"
    elif s >= 30.0: return "MILD RISK-OFF"
    else:           return "RISK-OFF"

def _to_naive(ts: pd.Timestamp) -> pd.Timestamp:
    """Strip timezone safely — works for both tz-aware and tz-naive timestamps."""
    ts = pd.Timestamp(ts)
    return ts.tz_convert(None) if ts.tzinfo is not None else ts

_regime_duration_days: int = 0
if len(_hda) >= 2:
    _reg_series  = _hda.apply(_regime_of)
    _changes     = (_reg_series != _reg_series.shift(1))
    _change_idx  = _changes[_changes].index
    if len(_change_idx) > 0:
        _regime_start = _change_idx[-1]
        _regime_duration_days = (
            pd.Timestamp.now() - _to_naive(_regime_start)
        ).days
    else:
        _regime_duration_days = (
            pd.Timestamp.now() - _to_naive(_hda.index[0])
        ).days

_dur_str = (f"{_regime_duration_days}d"
            if _regime_duration_days < 60
            else f"{_regime_duration_days // 30}mo {_regime_duration_days % 30}d")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 🌐 Macro Regime Compass")
st.caption(
    f"Data: {_today_key} UTC  ·  Next refresh: 00:00 UTC  ·  "
    f"Z-window: {z_window}d  ·  History: {lookback_opt}"
)

# ── GLOWING REGIME BANNER ─────────────────────────────────────────────────────
st.markdown(
    f"<div style='"
    f"background:linear-gradient(135deg,{color}ee 0%,{color}99 60%,{color}bb 100%);"
    f"padding:28px 36px 20px;border-radius:16px;text-align:center;margin-bottom:6px;"
    f"box-shadow:0 0 40px {color}55,0 0 80px {color}22,"
    f"inset 0 1px 0 rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.10)'>"

    f"<div style='font-size:2.6em;font-weight:900;color:white;letter-spacing:4px;"
    f"text-shadow:0 0 25px rgba(255,255,255,.80),0 0 50px {color};margin-bottom:4px'>"
    f"{icon} {label}</div>"

    f"<div style='font-size:1.12em;color:rgba(255,255,255,.88);margin-bottom:8px'>"
    f"Regime Score: <b style='font-size:1.35em'>{display_score:.0f}</b> / 100"
    f"<span style='margin-left:16px;font-size:.60em;color:{_d1_col};"
    f"font-weight:700;letter-spacing:.5px'>{_d1_arrow} {_d1_val} 1d</span>"
    f"<span style='margin-left:10px;font-size:.60em;color:{_d7_col};"
    f"font-weight:700;letter-spacing:.5px'>{_d7_arrow} {_d7_val} 1wk</span>"
    f"</div>"

    f"<div style='font-size:.84em;color:rgba(255,255,255,.60);margin-bottom:16px;"
    f"font-style:italic'>{_descs[label]}</div>"

    f"<div style='max-width:380px;margin:0 auto'>"
    f"<div style='height:7px;background:rgba(0,0,0,.35);border-radius:4px;overflow:hidden'>"
    f"<div style='height:7px;width:{display_score}%;background:rgba(255,255,255,.88);"
    f"border-radius:4px;box-shadow:0 0 10px rgba(255,255,255,.70)'></div></div>"
    f"<div style='display:flex;justify-content:space-between;"
    f"color:rgba(255,255,255,.48);font-size:.68em;margin-top:3px'>"
    f"<span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>"
    f"</div></div></div>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# HERO SECTION: ARC GAUGE (left) + STATS PANEL (right)
# ─────────────────────────────────────────────────────────────────────────────

col_gauge, col_stats = st.columns([3, 2], gap="medium")

with col_gauge:
    st.plotly_chart(_build_arc_gauge(display_score, color, label),
                    use_container_width=True)

with col_stats:
    n_bull = sum(1 for k, v in ind.items()
                 if v["z"] * _INDICATOR_META[k]["risk_sign"] > 0)

    signal_rows = "".join([
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04)'>"
        f"<span style='color:rgba(255,255,255,.68);font-size:.82em'>"
        f"{_INDICATOR_META[k]['icon']} {_INDICATOR_META[k]['label']}</span>"
        f"<span style='color:"
        f"{'#26a69a' if ind[k]['z']*_INDICATOR_META[k]['risk_sign']>0 else '#ef5350'};"
        f"font-weight:700;font-size:.80em;letter-spacing:1px'>"
        f"{'▲ ON' if ind[k]['z']*_INDICATOR_META[k]['risk_sign']>0 else '▼ OFF'}"
        f"</span></div>"
        for k in _card_order if k in ind
    ])

    st.markdown(
        f"<div style='padding:6px 0'>"

        # Regime badge + duration
        f"<div style='background:#12121e;border-radius:10px;padding:16px 18px;"
        f"margin-bottom:10px;border-left:4px solid {color};"
        f"box-shadow:-4px 0 20px {color}33'>"
        f"<div style='display:flex;justify-content:space-between;align-items:flex-start'>"
        f"<div>"
        f"<div style='color:rgba(255,255,255,.40);font-size:.68em;"
        f"letter-spacing:2px;margin-bottom:5px'>CURRENT REGIME</div>"
        f"<div style='color:{color};font-size:1.4em;font-weight:900;"
        f"letter-spacing:2px'>{icon} {label}</div>"
        f"</div>"
        f"<div style='text-align:right'>"
        f"<div style='font-size:.60em;letter-spacing:1.5px;"
        f"color:rgba(255,255,255,.28);margin-bottom:3px'>ACTIVE FOR</div>"
        f"<div style='font-size:1.1em;font-weight:700;color:rgba(255,255,255,.70)'>"
        f"{_dur_str}</div>"
        f"</div>"
        f"</div></div>"

        # Score + momentum
        f"<div style='background:#12121e;border-radius:10px;padding:16px 18px;"
        f"margin-bottom:10px;border:1px solid rgba(255,255,255,.06)'>"
        f"<div style='color:rgba(255,255,255,.40);font-size:.68em;"
        f"letter-spacing:2px;margin-bottom:3px'>COMPOSITE SCORE</div>"
        f"<div style='display:flex;align-items:baseline;gap:8px;margin-bottom:12px'>"
        f"<span style='color:white;font-size:2.8em;font-weight:900;line-height:1.05'>"
        f"{display_score:.0f}</span>"
        f"<span style='font-size:.75em;color:rgba(255,255,255,.28);font-weight:400'>/ 100</span>"
        f"</div>"
        f"<div style='display:flex;border-top:1px solid rgba(255,255,255,.07);"
        f"padding-top:10px;gap:0'>"
        f"<div style='flex:1;text-align:center;"
        f"border-right:1px solid rgba(255,255,255,.07)'>"
        f"<div style='font-size:.62em;letter-spacing:1.5px;"
        f"color:rgba(255,255,255,.28);margin-bottom:4px'>1 DAY</div>"
        f"<div style='font-size:1.05em;font-weight:700;color:{_d1_col}'>"
        f"{_d1_arrow} {_d1_val}</div></div>"
        f"<div style='flex:1;text-align:center;"
        f"border-right:1px solid rgba(255,255,255,.07)'>"
        f"<div style='font-size:.62em;letter-spacing:1.5px;"
        f"color:rgba(255,255,255,.28);margin-bottom:4px'>1 WEEK</div>"
        f"<div style='font-size:1.05em;font-weight:700;color:{_d7_col}'>"
        f"{_d7_arrow} {_d7_val}</div></div>"
        f"<div style='flex:1;text-align:center'>"
        f"<div style='font-size:.62em;letter-spacing:1.5px;"
        f"color:rgba(255,255,255,.28);margin-bottom:4px'>1 MONTH</div>"
        f"<div style='font-size:1.05em;font-weight:700;color:{_d30_col}'>"
        f"{_d30_arrow} {_d30_val}</div></div>"
        f"</div></div>"

        # Signal breakdown
        f"<div style='background:#12121e;border-radius:10px;padding:16px 18px;"
        f"margin-bottom:10px;border:1px solid rgba(255,255,255,.06)'>"
        f"<div style='color:rgba(255,255,255,.40);font-size:.68em;"
        f"letter-spacing:2px;margin-bottom:8px'>"
        f"SIGNAL BREAKDOWN  ·  {n_bull}/{len(ind)} BULLISH</div>"
        f"{signal_rows}</div>"

        # Strategy note
        f"<div style='background:{color}14;border-radius:10px;padding:14px 16px;"
        f"border:1px solid {color}38'>"
        f"<div style='color:rgba(255,255,255,.55);font-size:.80em;"
        f"line-height:1.5'>{_descs[label]}</div></div>"

        f"</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 90-DAY REGIME COLOUR RIBBON
# ─────────────────────────────────────────────────────────────────────────────

_divider()
_section_header("90-Day Regime Timeline",
                "Each bar = one trading day · hover for score · colour = regime zone")

if not hist.empty:
    _ribbon_s = (hist + 100) / 2
    _cut90    = pd.Timestamp.now() - pd.Timedelta(days=90)
    _ribbon   = _ribbon_s[_ribbon_s.index >= _cut90].dropna()

    def _rc(s: float) -> str:
        if   s >= 70:   return "rgba(15,152,70,.92)"
        elif s >= 57.5: return "rgba(18,128,72,.85)"
        elif s >= 42.5: return "rgba(62,62,62,.82)"
        elif s >= 30:   return "rgba(165,82,16,.85)"
        else:           return "rgba(155,25,25,.92)"

    if not _ribbon.empty:
        fig_rib = go.Figure(go.Bar(
            x=_ribbon.index, y=[1.0] * len(_ribbon),
            marker_color=[_rc(float(v)) for v in _ribbon.values],
            marker_line_width=0, showlegend=False,
            customdata=_ribbon.values,
            hovertemplate="%{x|%b %d, %Y}<br>Score: <b>%{customdata:.1f}</b><extra></extra>",
        ))
        fig_rib.update_layout(
            template="plotly_dark", height=52,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(visible=False, range=[0, 1.05]),
            bargap=0.04,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rib, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT CARDS  (with mini z-score bars)
# ─────────────────────────────────────────────────────────────────────────────

_divider()
_section_header("Component Breakdown",
                "Individual indicator readings · bar shows risk-on strength (50% = neutral)")

_cols = st.columns(len(_card_order))
for col, key in zip(_cols, _card_order):
    meta    = _INDICATOR_META[key]
    v_ind   = ind.get(key)
    contrib = parts.get(key, 0.0)

    if v_ind is None:
        col.markdown(
            f"<div style='background:#1a1a2e;padding:12px;border-radius:8px;"
            f"border-left:3px solid #444;opacity:.5'>"
            f"<b>{meta['icon']} {meta['label']}</b><br>— No data —</div>",
            unsafe_allow_html=True)
        continue

    z, pct, cur = v_ind["z"], v_ind["pct"], v_ind["current"]

    if contrib > 6:   card_bg, border_c = "#0d2b1a", "#26a69a"
    elif contrib < -6: card_bg, border_c = "#2b0d0d", "#ef5350"
    else:              card_bg, border_c = "#12121e", "#444"

    arrow = "▲" if z > .35 else "▼" if z < -.35 else "─"
    sign  = meta["risk_sign"]
    z_col = ("#26a69a" if z < -.5 and sign == -1 else
             "#ef5350" if z >  .5 and sign == -1 else
             "#26a69a" if z >  .5 and sign == +1 else
             "#ef5350" if z < -.5 and sign == +1 else "white")

    z_contrib = float(z) * sign
    bar_pct   = max(2.0, min(100.0, 50.0 + np.clip(z_contrib, -3, 3) / 3.0 * 50.0))
    bar_col   = "#26a69a" if z_contrib >= 0 else "#ef5350"

    col.markdown(
        f"<div style='background:{card_bg};padding:14px;border-radius:10px;"
        f"border-left:3px solid {border_c};min-height:178px;"
        f"box-shadow:0 2px 12px rgba(0,0,0,.40)'>"

        f"<div style='font-size:.90em;font-weight:600;color:white;margin-bottom:6px'>"
        f"{meta['icon']} {meta['label']}</div>"

        f"<div style='display:flex;align-items:center;justify-content:space-between;"
        f"margin-bottom:2px'>"
        f"<div style='font-size:1.45em;font-weight:bold;color:white'>"
        f"{cur:,.2f}<span style='font-size:.50em;color:#777;margin-left:4px'>"
        f"{meta['unit']}</span></div>"
        f"<div style='opacity:.75'>"
        f"{_mini_spark_svg(v_ind['zseries'].dropna().iloc[-30:].values, z_col)}"
        f"</div></div>"

        # mini z-score bar
        f"<div style='height:4px;border-radius:2px;margin:6px 0 4px;background:#1a1a1a'>"
        f"<div style='height:4px;border-radius:2px;width:{bar_pct:.0f}%;"
        f"background:{bar_col};opacity:.80'></div></div>"

        f"<div style='font-size:.82em;margin-bottom:3px'>"
        f"Z: <span style='color:{z_col};font-weight:bold'>{z:+.2f}σ {arrow}</span></div>"

        f"<div style='font-size:.80em;color:#888;margin-bottom:2px'>"
        f"1Y Pct: <b style='color:#ccc'>{pct:.0f}th</b></div>"

        f"<div style='font-size:.80em;color:#888'>"
        f"Contrib: <b style='color:#ccc'>{contrib:+.1f} pts</b></div>"

        f"</div>",
        unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW: RADAR CHART  +  HISTORICAL SCORE
# ─────────────────────────────────────────────────────────────────────────────

_divider()
col_radar, col_hist = st.columns([1, 2], gap="medium")

# ── Radar chart ───────────────────────────────────────────────────────────────
with col_radar:
    _section_header("Signal Radar", "Outer edge = extreme · 0.5 ring = neutral")

    _rk = [k for k in _card_order if k in ind]

    if not _rk:
        st.caption("No indicator data available for radar chart.")
    else:
        _rl   = [_INDICATOR_META[k]["label"] for k in _rk]
        _rv   = [(np.clip(ind[k]["z"] * _INDICATOR_META[k]["risk_sign"], -3, 3) + 3) / 6
                 for k in _rk]
        _rl_c = _rl + [_rl[0]]
        _rv_c = _rv + [_rv[0]]

        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(        # neutral reference ring
            r=[0.5] * len(_rl_c), theta=_rl_c, mode="lines",
            line=dict(color="rgba(255,255,255,.18)", width=1, dash="dot"),
            showlegend=False))
        fig_rad.add_trace(go.Scatterpolar(        # current reading
            r=_rv_c, theta=_rl_c, fill="toself",
            fillcolor=_hex_rgba(color, 0.16),
            line=dict(color=color, width=2.5),
            mode="lines+markers",
            marker=dict(size=9, color=color, line=dict(color="white", width=1.5)),
            showlegend=False,
            hovertemplate="<b>%{theta}</b><br>Strength: %{r:.2f}<extra></extra>"))
        fig_rad.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1],
                                tickvals=[0, .25, .5, .75, 1],
                                ticktext=["", "", "Neutral", "", ""],
                                gridcolor="rgba(255,255,255,.07)",
                                linecolor="rgba(255,255,255,.12)",
                                tickfont=dict(size=8, color="rgba(255,255,255,.38)")),
                angularaxis=dict(gridcolor="rgba(255,255,255,.07)",
                                 linecolor="rgba(255,255,255,.12)",
                                 tickfont=dict(size=10, color="rgba(255,255,255,.72)")),
                bgcolor="rgba(0,0,0,.45)"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=55, r=55, t=30, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_rad, use_container_width=True)

# ── Historical score ──────────────────────────────────────────────────────────
with col_hist:
    _section_header("Regime Score — Historical", f"Past {lookback_opt}")

    if not hist.empty:
        cutoff    = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
        hist_plot = hist[hist.index >= cutoff].dropna()
        hist_disp = (hist_plot + 100) / 2

        if not hist_disp.empty:
            # ── Adaptive Y range: focus on where data lives (±12 pts padding)
            _y_min = max(-2,  float(hist_disp.min()) - 12)
            _y_max = min(102, float(hist_disp.max()) + 12)

            fig_hist = go.Figure()

            # 1 ── Zone bands (slightly more opaque — give visual context)
            for y0, y1, zclr in [
                (  0,  30,  "rgba(155,25,25,.07)"),
                ( 30,  42.5,"rgba(165,82,16,.05)"),
                ( 42.5, 57.5,"rgba(62,62,62,.04)"),
                ( 57.5, 70, "rgba(18,124,68,.05)"),
                ( 70,  100, "rgba(15,152,70,.07)"),
            ]:
                fig_hist.add_hrect(y0=y0, y1=y1, fillcolor=zclr, line_width=0)

            # 2 ── Dual fill anchored at 50: green above, red below (more visible)
            _hx = hist_disp.index
            _hy = hist_disp.values
            _neut = np.full(len(hist_disp), 50.0)

            fig_hist.add_trace(go.Scatter(
                x=_hx, y=_neut, mode="none",
                showlegend=False, hoverinfo="skip"))
            fig_hist.add_trace(go.Scatter(
                x=_hx, y=np.maximum(_hy, 50.0), mode="none",
                fill="tonexty", fillcolor="rgba(38,166,154,.28)",
                showlegend=False, hoverinfo="skip"))

            fig_hist.add_trace(go.Scatter(
                x=_hx, y=_neut, mode="none",
                showlegend=False, hoverinfo="skip"))
            fig_hist.add_trace(go.Scatter(
                x=_hx, y=np.minimum(_hy, 50.0), mode="none",
                fill="tonexty", fillcolor="rgba(239,83,80,.28)",
                showlegend=False, hoverinfo="skip"))

            # 3 ── Line glow: wide faint trace drawn BEFORE main line
            fig_hist.add_trace(go.Scatter(
                x=_hx, y=_hy, mode="lines",
                line=dict(color="rgba(210,210,210,.09)", width=12),
                showlegend=False, hoverinfo="skip"))

            # 4 ── Main score line
            fig_hist.add_trace(go.Scatter(
                x=_hx, y=_hy, mode="lines",
                line=dict(color="rgba(222,222,222,.96)", width=2.2),
                showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>Score: <b>%{y:.1f}</b><extra></extra>"))

            # 5 ── Current value: outer glow ring + solid dot + label
            _cv, _cd = float(hist_disp.values[-1]), hist_disp.index[-1]
            fig_hist.add_trace(go.Scatter(
                x=[_cd], y=[_cv], mode="markers",
                marker=dict(size=24, color=_hex_rgba(color, 0.18), symbol="circle"),
                showlegend=False, hoverinfo="skip"))
            fig_hist.add_trace(go.Scatter(
                x=[_cd], y=[_cv], mode="markers+text",
                marker=dict(size=12, color=color, symbol="circle",
                            line=dict(color="white", width=1.8)),
                text=[f"  {_cv:.0f}"], textposition="middle right",
                textfont=dict(size=12, color=color, family="Arial Black, Arial"),
                showlegend=False, hoverinfo="skip"))

            # 6 ── Reference lines + labels
            #      NEUTRAL = dash + bright  |  RISK-ON/OFF = dot + dimmer
            #      Labels via xref="paper" → no overlap with y-axis
            for yval, txt, rclr, lw, ld in [
                (70, "RISK-ON",  "rgba(38,166,154,.82)",  1.2, "dot"),
                (50, "NEUTRAL",  "rgba(255,255,255,.58)",  1.8, "dash"),
                (30, "RISK-OFF", "rgba(239,83,80,.82)",   1.2, "dot"),
            ]:
                fig_hist.add_hline(y=yval, line_dash=ld,
                                   line_color=rclr, line_width=lw)
                if _y_min - 2 <= yval <= _y_max + 2:   # only label if visible
                    fig_hist.add_annotation(
                        xref="paper", x=0.01, y=yval,
                        text=f" {txt}",
                        font=dict(size=9, color=rclr, family="monospace"),
                        showarrow=False, xanchor="left", yanchor="bottom",
                        bgcolor="rgba(10,10,18,.80)", borderpad=2)

            # 7 ── Regime change markers — vertical dotted lines at transitions
            _reg_hist  = hist_disp.apply(_regime_of)
            _reg_chg   = _reg_hist != _reg_hist.shift(1)
            _reg_dates = _reg_hist[_reg_chg].index[1:]  # skip first (initial state)
            _rclr_map  = {
                "RISK-ON":       "rgba(15,152,70,.55)",
                "MILD RISK-ON":  "rgba(18,124,68,.48)",
                "NEUTRAL":       "rgba(150,150,150,.42)",
                "MILD RISK-OFF": "rgba(165,82,16,.48)",
                "RISK-OFF":      "rgba(155,25,25,.55)",
            }
            for _rcd in _reg_dates:
                _new_reg  = _reg_hist.loc[_rcd]
                _vclr     = _rclr_map.get(_new_reg, "rgba(150,150,150,.42)")
                _lbl      = _new_reg.replace("MILD RISK-", "M.R-")
                fig_hist.add_vline(x=_rcd, line_dash="dot",
                                   line_color=_vclr, line_width=1.0)
                fig_hist.add_annotation(
                    x=_rcd, y=_y_max * 0.97,
                    text=_lbl, showarrow=False,
                    font=dict(size=7, color=_vclr, family="monospace"),
                    xanchor="left", yanchor="top", textangle=-90,
                )

            fig_hist.update_layout(
                template="plotly_dark", height=380,
                margin=dict(l=0, r=65, t=6, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(range=[_y_min, _y_max], showgrid=True,
                           gridcolor="rgba(255,255,255,.06)",
                           tickfont=dict(size=10, color="rgba(255,255,255,.50)")),
                showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW: YIELD CURVE  +  FEAR & GREED
# ─────────────────────────────────────────────────────────────────────────────

_divider()
col_yc, col_fg = st.columns(2, gap="medium")

cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)

# ── Yield curve ───────────────────────────────────────────────────────────────
with col_yc:
    _section_header("Yield Curve", "10Y − 3M spread · red zone = inversion")
    if "YieldCurve" in ind:
        sp = ind["YieldCurve"]["series"].copy()
        sp.index = pd.to_datetime(sp.index).normalize()
        sp = sp[sp.index >= cutoff].dropna()
        if not sp.empty:
            fig_yc = go.Figure()
            fig_yc.add_hrect(y0=min(sp.min()-.2, -.1), y1=0,
                              fillcolor="rgba(239,83,80,.07)", line_width=0)
            fig_yc.add_hline(y=0, line_dash="dash",
                              line_color="rgba(255,255,255,.55)", line_width=1.3,
                              annotation_text="Inversion",
                              annotation_font=dict(color="rgba(255,255,255,.50)", size=9),
                              annotation_position="bottom right")
            fig_yc.add_trace(go.Scatter(
                x=sp.index, y=sp.clip(lower=0).values, mode="lines",
                line=dict(color="rgba(38,166,154,.9)", width=1.5),
                fill="tozeroy", fillcolor="rgba(38,166,154,.13)",
                showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>Spread: <b>%{y:.2f}%</b><extra></extra>"))
            fig_yc.add_trace(go.Scatter(
                x=sp.index, y=sp.clip(upper=0).values, mode="lines",
                line=dict(color="rgba(239,83,80,.9)", width=1.5),
                fill="tozeroy", fillcolor="rgba(239,83,80,.18)",
                showlegend=False))
            fig_yc.update_layout(template="plotly_dark", height=260,
                margin=dict(l=0,r=0,t=6,b=0),
                yaxis=dict(title="Spread (%)", showgrid=True,
                           gridcolor="rgba(255,255,255,.04)"),
                showlegend=False)
            st.plotly_chart(fig_yc, use_container_width=True)
            cur_sp = _safe_last(sp)
            st.caption(f"Current: **{cur_sp:+.2f}%**  ·  "
                       f"{'🔴 INVERTED' if cur_sp < 0 else '🟢 Normal'}")

# ── Fear & Greed ──────────────────────────────────────────────────────────────
with col_fg:
    _section_header("Fear & Greed Index", "Alternative.me · 0 = Extreme Fear · 100 = Extreme Greed")
    if not fg_df.empty:
        fg_plot = fg_df.copy()
        fg_plot.index = pd.to_datetime(fg_plot.index).normalize()
        fg_plot = fg_plot[fg_plot.index >= cutoff]
        if not fg_plot.empty:
            def _fgc(v):
                return ("rgba(239,83,80,.90)" if v<=25 else
                        "rgba(255,112,67,.85)" if v<=45 else
                        "rgba(120,120,120,.80)" if v<=55 else
                        "rgba(38,166,154,.85)" if v<=75 else
                        "rgba(0,230,118,.90)")
            fig_fg = go.Figure()
            fig_fg.add_hrect(y0=75, y1=100, fillcolor="rgba(0,230,118,.06)", line_width=0)
            fig_fg.add_hrect(y0=0,  y1=25,  fillcolor="rgba(239,83,80,.06)", line_width=0)
            fig_fg.add_hline(y=50, line_dash="dot",
                              line_color="rgba(255,255,255,.25)", line_width=1)
            fig_fg.add_trace(go.Bar(
                x=fg_plot.index, y=fg_plot["value"].values,
                marker_color=[_fgc(int(v)) for v in fg_plot["value"]],
                marker_line_width=0, showlegend=False,
                hovertemplate="%{x|%Y-%m-%d}<br>F&G: <b>%{y}</b><extra></extra>"))
            fig_fg.update_layout(template="plotly_dark", height=260,
                margin=dict(l=0,r=0,t=6,b=0),
                yaxis=dict(range=[0,100], title="Score", showgrid=True,
                           gridcolor="rgba(255,255,255,.04)"),
                showlegend=False)
            st.plotly_chart(fig_fg, use_container_width=True)
            fg_cur = int(_safe_last(fg_plot["value"].astype(float), 50))
            fg_lbl = str(fg_plot["label"].iloc[-1]) if not fg_plot.empty else "—"
            st.caption(f"Current: **{fg_cur}** — {fg_lbl}")

# ─────────────────────────────────────────────────────────────────────────────
# SCORE DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

_divider()
_section_header("Score Decomposition",
                "Each bar = contribution to the 0–100 score · baseline 50 = neutral · "
                "final bar = composite score")

_ks = [k for k in _card_order if k in parts]
_vs = [parts[k] for k in _ks]
_ls = [_INDICATOR_META[k]["label"] for k in _ks]

_idx    = sorted(range(len(_ks)), key=lambda i: abs(_vs[i]), reverse=True)
_vs2    = [_vs[i] for i in _idx]
_ls2    = [_ls[i] for i in _idx]
# Convert to 0–100 scale: internal range is -100/+100, display is 0-100
# Each contribution divides by 2 (same linear transform as display_score)
_vs2_d  = [v / 2.0 for v in _vs2]

fig_dec = go.Figure(go.Waterfall(
    orientation="v",
    measure=["relative"] * len(_vs2_d) + ["total"],
    x=_ls2 + ["COMPOSITE"],
    y=_vs2_d + [None],
    base=50,        # ← waterfall starts at 50 (neutral), ends at display_score
    connector=dict(mode="between",
                   line=dict(color="rgba(255,255,255,.10)", width=1, dash="dot")),
    increasing=dict(marker=dict(color="rgba(38,166,154,.85)", line=dict(width=0))),
    decreasing=dict(marker=dict(color="rgba(239,83,80,.85)", line=dict(width=0))),
    totals=dict(marker=dict(color=color,
                            line=dict(color="rgba(255,255,255,.55)", width=1.5))),
    text=[f"{v:+.1f}" for v in _vs2_d] + [f"  {display_score:.0f}"],
    textposition="outside",
    textfont=dict(size=10, color="rgba(255,255,255,.80)"),
    hovertemplate="<b>%{x}</b><br>Contribution: %{y:+.2f} pts (0–100 scale)<extra></extra>",
))
# Neutral reference line at 50
fig_dec.add_hline(y=50, line_color="rgba(255,255,255,.28)", line_width=1.4,
                  line_dash="dash",
                  annotation_text=" Neutral (50)",
                  annotation_font=dict(color="rgba(255,255,255,.38)", size=9),
                  annotation_position="top left")
fig_dec.update_layout(
    template="plotly_dark", height=260,
    margin=dict(l=0, r=0, t=28, b=0),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(title="Score (0–100)",
               range=[max(0,  display_score - max(abs(v) for v in _vs2_d) - 8),
                      min(100, display_score + max(abs(v) for v in _vs2_d) + 8)]
               if _vs2_d else [20, 85],
               showgrid=True,
               gridcolor="rgba(255,255,255,.04)", zeroline=False),
    showlegend=False)
st.plotly_chart(fig_dec, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# READING GUIDE
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("📖 How to read the Macro Regime Compass"):
    st.markdown(f"""
**Score** runs from **0** (extreme risk-off) to **100** (extreme risk-on) · **50** = neutral
Current: **{display_score:.0f} / 100**

| Score | Regime | Implication |
|---|---|---|
| 70–100 | 🟢 Risk-On | Strong macro tailwind — full size on long setups |
| 57.5–70 | 🟡 Mild Risk-On | Positive bias — normal sizing |
| 42.5–57.5 | ⚪ Neutral | No macro edge — be selective |
| 30–42.5 | 🟠 Mild Risk-Off | Headwinds — reduce size, widen stops |
| 0–30 | 🔴 Risk-Off | Defensive — capital preservation |

| Indicator | Risk-On | Risk-Off |
|---|---|---|
| **VIX** | Low/falling (<15) | High/rising (>25) |
| **DXY** | Weak/falling | Strong/rising |
| **Gold** | Stable/falling | Sharply rising |
| **Yield Curve** | Positive (>0%) | Inverted (<0%) |
| **Fear & Greed** | Greed (>60) | Fear (<40) |

**Anti-overfitting:** Rolling z-scores only · equal weights · zero fitted parameters.
    """)


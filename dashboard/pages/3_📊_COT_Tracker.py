"""
📊 COT Positioning Tracker
===========================
CFTC Commitment of Traders — Managed Money net positioning.
COT Score 0–100:
  0  = Extreme Short  → historically bullish (contrarian buy signal)
  50 = Neutral        → no clear signal
  100 = Extreme Long  → historically bearish (contrarian sell signal)
"""
from __future__ import annotations

import sys, os
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import io
import zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import streamlit as st
import requests
import yfinance as yf

st.set_page_config(
    page_title="COT Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# MARKET DEFINITIONS
# Priority order: Crypto → Stocks → Commodities
# ─────────────────────────────────────────────────────────────────────────────

_MARKETS: dict[str, dict] = {
    # ── Crypto (CFTC Disaggregated — CFTC classifies BTC as a commodity)
    "Bitcoin":   {"search": "BITCOIN",           "report": "disagg", "price_ticker": "BTC-USD", "price_label": "BTC Price (USD)",      "icon": "₿",  "color": "#f7931a", "cat": "crypto",      "unit": "contracts"},
    # ── Stocks (CFTC TFF report — Leveraged Funds = Managed Money) ──────────
    "S&P 500":   {"search": "E-MINI S&P STOCKS", "report": "tff",    "price_ticker": "^GSPC",   "price_label": "S&P 500 Index",        "icon": "📈", "color": "#26a69a", "cat": "stocks",      "unit": "contracts"},
    "Nasdaq":    {"search": "NASDAQ-100",         "report": "tff",    "price_ticker": "^NDX",    "price_label": "Nasdaq 100 Index",     "icon": "💻", "color": "#1a7bc4", "cat": "stocks",      "unit": "contracts"},
    # ── Commodities (CFTC Disaggregated report) ──────────────────────────────
    "Gold":      {"search": "GOLD - COMMODITY",   "report": "disagg", "price_ticker": "GC=F",    "price_label": "Gold Price (USD/oz)",  "icon": "🥇", "color": "#ffd700", "cat": "commodities", "unit": "contracts"},
    "Silver":    {"search": "SILVER - COMMODITY", "report": "disagg", "price_ticker": "SI=F",    "price_label": "Silver Price (USD/oz)","icon": "🥈", "color": "#c0c0c0", "cat": "commodities", "unit": "contracts"},
    "Crude Oil": {"search": "CRUDE OIL, LIGHT",   "report": "disagg", "price_ticker": "CL=F",    "price_label": "WTI Crude (USD/bbl)",  "icon": "🛢️", "color": "#cd853f", "cat": "commodities", "unit": "contracts"},
    "Copper":    {"search": "COPPER-",            "report": "disagg", "price_ticker": "HG=F",    "price_label": "Copper (USD/lb)",      "icon": "🔶", "color": "#b87333", "cat": "commodities", "unit": "contracts"},
}

_CATS = {
    "crypto":      {"label": "₿ Crypto",       "markets": ["Bitcoin"]},
    "stocks":      {"label": "📈 Stocks",       "markets": ["S&P 500", "Nasdaq"]},
    "commodities": {"label": "🥇 Commodities",  "markets": ["Gold", "Silver", "Crude Oil", "Copper"]},
}

_COT_WINDOW = 156   # 3 years of weekly data → rolling min-max window

# COT zone definitions (0→100: extreme short → extreme long)
_COT_ZONES = [
    (  0, 20, "rgba(38,166,154,.88)",  "EXT. SHORT", "Managed Money at historic short extreme — contrarian buy signal"),
    ( 20, 40, "rgba(18,124,68,.78)",   "SHORT",      "Below-average MM positioning — mild bearish bias"),
    ( 40, 60, "rgba(62,62,62,.72)",    "NEUTRAL",    "No positioning extreme — no directional signal"),
    ( 60, 80, "rgba(165,82,16,.78)",   "LONG",       "Above-average MM positioning — mild bullish exhaustion risk"),
    ( 80,100, "rgba(155,25,25,.88)",   "EXT. LONG",  "Managed Money at historic long extreme — contrarian sell signal"),
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _hex_rgba(hex_c: str, alpha: float = 1.0) -> str:
    """Convert '#rrggbb' + alpha → 'rgba(r,g,b,a)' for Plotly."""
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"

def _divider() -> None:
    """Gradient divider replacing plain st.markdown('---')."""
    st.markdown(
        "<div style='height:1px;"
        "background:linear-gradient(90deg,"
        "transparent 0%,rgba(38,166,154,.18) 20%,"
        "rgba(255,255,255,.06) 50%,"
        "rgba(38,166,154,.18) 80%,transparent 100%);"
        "margin:18px 0 6px'></div>",
        unsafe_allow_html=True,
    )

def _mini_sparkline_svg(values: np.ndarray, color: str,
                        w: int = 80, h: int = 24) -> str:
    """Inline SVG sparkline for COT score trend in the summary table."""
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
    pts_str  = " ".join(pts)
    last_cx, last_cy = pts[-1].split(",")   # pre-compute to avoid backslash in f-string
    return (
        f"<svg width='{w}' height='{h}' viewBox='0 0 {w} {h}' "
        f"style='vertical-align:middle;overflow:visible'>"
        f"<polyline points='{pts_str}' fill='none' "
        f"stroke='{color}' stroke-width='1.8' "
        f"stroke-linecap='round' stroke-linejoin='round' opacity='.85'/>"
        f"<circle cx='{last_cx}' cy='{last_cy}' "
        f"r='2.5' fill='{color}' opacity='.90'/>"
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

def _cot_score_series(net: pd.Series) -> pd.Series:
    """Rolling percentile rank of MM net position → COT Score 0–100.

    Why percentile rank instead of min-max:
    ▸ Resistant to single outliers (e.g. COVID spike doesn't compress all other values)
    ▸ 50 always = true historical median (genuine neutral)
    ▸ 0/100 = genuine historical extremes, stable across different markets

    Dynamic window (Melhoria 2):
    ▸ Target: _COT_WINDOW (156 weeks = 3 years)
    ▸ For markets with less history (e.g. Bitcoin ~2018), adapts to 80% of available data
    ▸ Minimum: 10 weeks (below this the score is meaningless)
    """
    n        = len(net)
    actual_w = max(10, min(_COT_WINDOW, int(n * 0.80)))
    min_p    = max(10, actual_w // 4)

    def _pct_rank(x: np.ndarray) -> float:
        """Percentile rank of the last element in the window."""
        if len(x) < 2:
            return 50.0
        return float((x[:-1] < x[-1]).mean() * 100)

    return (
        net.rolling(actual_w, min_periods=min_p)
           .apply(_pct_rank, raw=True)
           .clip(0, 100)
    )

def _zone_for(score: float) -> dict:
    """Return zone info. Boundaries belong to the HIGHER zone (strict upper bound)."""
    if   score <  20: return {"color": _COT_ZONES[0][2], "label": _COT_ZONES[0][3], "desc": _COT_ZONES[0][4]}
    elif score <  40: return {"color": _COT_ZONES[1][2], "label": _COT_ZONES[1][3], "desc": _COT_ZONES[1][4]}
    elif score <  60: return {"color": _COT_ZONES[2][2], "label": _COT_ZONES[2][3], "desc": _COT_ZONES[2][4]}
    elif score <  80: return {"color": _COT_ZONES[3][2], "label": _COT_ZONES[3][3], "desc": _COT_ZONES[3][4]}
    else:             return {"color": _COT_ZONES[4][2], "label": _COT_ZONES[4][3], "desc": _COT_ZONES[4][4]}

def _delta_fmt(d: float | None) -> tuple[str, str, str]:
    if d is None:
        return "—", "rgba(255,255,255,.28)", "─"
    if d > 500:    return f"+{d:,.0f}", "#26a69a", "↑"
    if d < -500:   return f"{d:,.0f}",  "#ef5350", "↓"
    return f"{d:+,.0f}", "rgba(255,255,255,.42)", "→"


# ─────────────────────────────────────────────────────────────────────────────
# COT GAUGE (semicircular — green left / red right)
# ─────────────────────────────────────────────────────────────────────────────

def _build_cot_gauge(score: float, color: str, label: str) -> go.Figure:
    R_OUT = 1.00; R_IN = 0.58; R_NEEDLE = 0.82; R_HUB = 0.055

    def s2a(s: float) -> float:
        return np.pi * (1.0 - s / 100.0)

    def arc(s0, s1, ri, ro, n=90):
        a0, a1 = s2a(s0), s2a(s1)
        fw = np.linspace(a0, a1, n); rv = np.linspace(a1, a0, n)
        x  = np.r_[ro*np.cos(fw), ri*np.cos(rv), ro*np.cos(a0)]
        y  = np.r_[ro*np.sin(fw), ri*np.sin(rv), ro*np.sin(a0)]
        return x, y

    fig = go.Figure()

    # Background track
    xb, yb = arc(0, 100, R_IN*.97, R_OUT*1.02, 200)
    fig.add_trace(go.Scatter(x=xb, y=yb, fill="toself",
        fillcolor="rgba(14,14,22,.96)",
        line=dict(color="rgba(255,255,255,.04)", width=.5),
        showlegend=False, hoverinfo="skip"))

    # Zone arcs
    zone_fills = [
        (  0, 20, "rgba(38,166,154,.88)"),
        ( 20, 40, "rgba(18,124,68,.78)"),
        ( 40, 60, "rgba(62,62,62,.72)"),
        ( 60, 80, "rgba(165,82,16,.78)"),
        ( 80,100, "rgba(155,25,25,.88)"),
    ]
    for s0, s1, fc in zone_fills:
        xz, yz = arc(s0, s1, R_IN, R_OUT)
        fig.add_trace(go.Scatter(x=xz, y=yz, fill="toself",
            fillcolor=fc, line=dict(color="rgba(8,8,8,.55)", width=1.0),
            showlegend=False, hoverinfo="skip"))

    # Dim inactive zones; brighten active
    for s0, s1, _ in zone_fills:
        if not (s0 <= score <= s1):
            xd, yd = arc(s0, s1, R_IN, R_OUT)
            fig.add_trace(go.Scatter(x=xd, y=yd, fill="toself",
                fillcolor="rgba(0,0,0,.22)", line=dict(width=0),
                showlegend=False, hoverinfo="skip"))
    active_zone = next((z for z in zone_fills if z[0] <= score <= z[1]), zone_fills[2])
    xab, yab = arc(active_zone[0], active_zone[1], R_IN, R_OUT)
    fig.add_trace(go.Scatter(x=xab, y=yab, fill="toself",
        fillcolor="rgba(255,255,255,.12)", line=dict(width=0),
        showlegend=False, hoverinfo="skip"))
    xa, ya = arc(active_zone[0], active_zone[1], R_IN-.015, R_OUT+.03)
    fig.add_trace(go.Scatter(x=xa, y=ya, fill="toself",
        fillcolor="rgba(0,0,0,0)", line=dict(color=color, width=3.0),
        showlegend=False, hoverinfo="skip"))

    # Tick marks + numbers
    for tick in [0, 25, 50, 75, 100]:
        a_t = s2a(tick)
        fig.add_trace(go.Scatter(
            x=[R_OUT*np.cos(a_t), 1.10*np.cos(a_t)],
            y=[R_OUT*np.sin(a_t), 1.10*np.sin(a_t)],
            mode="lines", line=dict(color="rgba(255,255,255,.60)", width=1.8),
            showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=1.19*np.cos(a_t), y=1.19*np.sin(a_t),
            text=f"<b>{tick}</b>", showarrow=False,
            font=dict(size=11, color="rgba(255,255,255,.85)"), align="center")

    # Zone labels (color-coded)
    zone_labels = [
        (  0, 20, "EXT.\nSHORT", "rgba(80,220,160,.95)"),
        ( 20, 40, "SHORT",       "rgba(50,190,110,.90)"),
        ( 40, 60, "NEUTRAL",     "rgba(175,175,175,.88)"),
        ( 60, 80, "LONG",        "rgba(215,125,45,.90)"),
        ( 80,100, "EXT.\nLONG",  "rgba(225,85,85,.95)"),
    ]
    for s0, s1, zlbl, zcol in zone_labels:
        a_mid = s2a((s0+s1)/2)
        fig.add_annotation(x=1.40*np.cos(a_mid), y=1.40*np.sin(a_mid),
            text=f"<b>{zlbl}</b>", showarrow=False,
            font=dict(size=10, color=zcol, family="monospace"), align="center")

    # Needle (tapered diamond)
    a_n = s2a(score)
    bw  = 0.030; bk = 0.08
    ndl_x = [R_NEEDLE*np.cos(a_n), bw*np.cos(a_n+np.pi/2),
              bk*np.cos(a_n+np.pi), bw*np.cos(a_n-np.pi/2), R_NEEDLE*np.cos(a_n)]
    ndl_y = [R_NEEDLE*np.sin(a_n), bw*np.sin(a_n+np.pi/2),
              bk*np.sin(a_n+np.pi), bw*np.sin(a_n-np.pi/2), R_NEEDLE*np.sin(a_n)]
    fig.add_trace(go.Scatter(x=ndl_x, y=ndl_y, fill="toself",
        fillcolor="rgba(245,245,250,.97)",
        line=dict(color="rgba(200,200,220,.55)", width=0.8),
        showlegend=False, hoverinfo="skip"))

    # Hub
    th = np.linspace(0, 2*np.pi, 60)
    fig.add_trace(go.Scatter(
        x=(R_HUB+.028)*np.cos(th), y=(R_HUB+.028)*np.sin(th),
        fill="toself", fillcolor="rgba(255,255,255,.10)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=R_HUB*np.cos(th), y=R_HUB*np.sin(th),
        fill="toself", fillcolor=color,
        line=dict(color="rgba(255,255,255,.90)", width=2.0),
        showlegend=False, hoverinfo="skip"))

    # Score number
    fig.add_annotation(x=0, y=-0.26,
        text=f"<b>{score:.0f}</b>", showarrow=False,
        font=dict(size=58, color=color, family="Arial Black, Arial, sans-serif"))

    fig.update_layout(
        xaxis=dict(range=[-1.55, 1.55], visible=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.62, 1.58], visible=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

_CFTC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/zip,*/*",
}

def _download_cot_years(url_pattern_hist: str, url_pattern_new: str,
                        cache_date: str) -> pd.DataFrame:
    """Generic CFTC ZIP downloader for the last 3 years."""
    current_year = int(cache_date[:4])
    dfs = []
    for year in range(current_year - 2, current_year + 1):
        for url in [url_pattern_hist.format(year), url_pattern_new.format(year)]:
            try:
                r = requests.get(url, timeout=45, headers=_CFTC_HEADERS)
                if r.status_code != 200:
                    continue
                z    = zipfile.ZipFile(io.BytesIO(r.content))
                name = z.namelist()[0]
                df   = pd.read_csv(z.open(name), low_memory=False,
                                   encoding="latin-1", on_bad_lines="skip")
                dfs.append(df)
                break
            except Exception:
                continue
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


@st.cache_data(ttl=86400*2, show_spinner=False)
def _load_cot_raw(cache_date: str) -> pd.DataFrame:   # noqa: ARG001
    """CFTC Disaggregated report — commodities + Bitcoin. MM = Managed Money."""
    return _download_cot_years(
        "https://www.cftc.gov/files/dea/history/fut_disagg_txt_{}.zip",
        "https://www.cftc.gov/files/dea/newcot/fut_disagg_txt_{}.zip",
        cache_date,
    )


@st.cache_data(ttl=86400*2, show_spinner=False)
def _load_cot_fin(cache_date: str) -> pd.DataFrame:   # noqa: ARG001
    """CFTC TFF report — financial futures (stock indices, currencies).
    Leveraged Funds = equivalent of Managed Money for financial instruments.
    """
    return _download_cot_years(
        "https://www.cftc.gov/files/dea/history/fut_fin_txt_{}.zip",
        "https://www.cftc.gov/files/dea/newcot/fut_fin_txt_{}.zip",
        cache_date,
    )


def _extract_market(df: pd.DataFrame, search: str,
                    col_prefix: str = "M_Money") -> pd.DataFrame | None:
    """Filter raw COT dataframe for one market; return weekly series.

    col_prefix:
      "M_Money"   → Disaggregated report  (commodities, Bitcoin)
      "Lev_Money" → TFF report  (stock indices — Leveraged Funds ≈ Managed Money)
    """
    if df.empty:
        return None

    # ── find columns ──────────────────────────────────────────────────────
    name_col = next((c for c in df.columns
                     if "Market_and_Exchange" in c or "market" in c.lower()), None)
    date_col = next((c for c in df.columns if "YYYY-MM-DD" in c), None)
    if name_col is None or date_col is None:
        return None

    # ── precise column detection ──────────────────────────────────────────
    # Use "_Long_All" / "_Short_All" as substrings to avoid matching
    # subcategory columns like _Long_Old_All or _Long_Other_All
    long_col  = next((c for c in df.columns
                      if col_prefix in c and "_Long_All"  in c), None)
    short_col = next((c for c in df.columns
                      if col_prefix in c and "_Short_All" in c
                      and "Spread" not in c), None)
    oi_col    = next((c for c in df.columns if "Open_Interest_All" in c), None)

    if long_col is None or short_col is None:
        return None

    # ── filter ────────────────────────────────────────────────────────────
    mask = df[name_col].astype(str).str.upper().str.contains(
        search.upper(), na=False, regex=False
    )
    mdf = df[mask].copy()
    if mdf.empty:
        return None

    mdf[date_col] = pd.to_datetime(mdf[date_col], errors="coerce")
    mdf = (mdf.dropna(subset=[date_col])
              .sort_values(date_col)
              .set_index(date_col))
    mdf.index = mdf.index.normalize()

    # ── deduplicate dates (year files overlap at boundaries) ──────────────
    mdf = mdf[~mdf.index.duplicated(keep="last")]

    # ── compute ───────────────────────────────────────────────────────────
    mdf["mm_long"]  = pd.to_numeric(mdf[long_col],  errors="coerce")
    mdf["mm_short"] = pd.to_numeric(mdf[short_col], errors="coerce")
    # Drop rows where both long and short are 0 (data artifact)
    mdf = mdf[(mdf["mm_long"] != 0) | (mdf["mm_short"] != 0)]
    mdf["mm_net"]   = mdf["mm_long"] - mdf["mm_short"]

    if oi_col:
        mdf["oi"] = pd.to_numeric(mdf[oi_col], errors="coerce")

    mdf["cot_score"] = _cot_score_series(mdf["mm_net"])
    mdf["mm_net_wow"] = mdf["mm_net"].diff()   # week-over-week change

    keep = ["mm_long", "mm_short", "mm_net", "mm_net_wow", "cot_score"] + \
           (["oi"] if oi_col else [])
    return mdf[keep].dropna(subset=["mm_net", "cot_score"])


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("📊 COT Tracker")
st.sidebar.caption("CFTC Commitment of Traders — Managed Money positioning.")

_lookback_map = {"6 Months": 26, "1 Year": 52, "2 Years": 104, "3 Years": 156}
_lookback_opt = st.sidebar.selectbox("Chart History", list(_lookback_map), index=1)
_lookback_wks = _lookback_map[_lookback_opt]

st.sidebar.markdown("---")

_now_utc   = datetime.utcnow()
_today_key = _now_utc.strftime("%Y-%m-%d")

if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Data source:**\n- CFTC.gov — Disaggregated Futures Report\n"
    "- Published every Friday (Tuesday positioning)\n"
    "- Free, no API key required"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Loading CFTC data (Disaggregated + Financial Futures)..."):
    _raw_disagg = _load_cot_raw(_today_key)   # commodities + Bitcoin
    _raw_fin    = _load_cot_fin(_today_key)   # stock indices

_data: dict[str, pd.DataFrame | None] = {}
for name, meta in _MARKETS.items():
    raw        = _raw_fin    if meta["report"] == "tff" else _raw_disagg
    col_prefix = "Lev_Money" if meta["report"] == "tff" else "M_Money"
    mdf = _extract_market(raw, meta["search"], col_prefix)

    # Bitcoin: try multiple search strings across both reports
    if mdf is None and name == "Bitcoin":
        for _src, _pref, _q in [
            (_raw_disagg, "M_Money",   "BITCOIN"),
            (_raw_fin,    "Lev_Money", "BITCOIN"),
            (_raw_disagg, "M_Money",   "BIT"),
        ]:
            mdf = _extract_market(_src, _q, _pref)
            if mdf is not None:
                break

    # S&P 500: try multiple name variants in TFF
    if mdf is None and name == "S&P 500":
        for alt in ["S&P 500 STOCK", "E-MINI S&P 500", "E-MINI S&P", "S&P 500", "S&P"]:
            mdf = _extract_market(raw, alt, col_prefix)
            if mdf is not None:
                break

    _data[name] = mdf

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 📊 COT Positioning Tracker")
st.caption(
    f"Source: CFTC Disaggregated + TFF Reports  ·  "
    f"Data as of: {_today_key}  ·  "
    f"Window: {_lookback_opt}"
)

# ── Data status — shows immediately which markets loaded ──────────────────
_loaded   = [n for n, d in _data.items() if d is not None and not d.empty]
_missing  = [n for n, d in _data.items() if d is None or d.empty]
if _missing:
    with st.expander(f"⚠️ Data status — {len(_missing)} market(s) not loaded", expanded=True):
        c1, c2 = st.columns(2)
        c1.markdown("**✅ Loaded:**\n" + "\n".join(f"- {n}" for n in _loaded))
        c2.markdown("**❌ Not found:**\n" + "\n".join(f"- {n}" for n in _missing))
        st.caption(
            "For missing markets, open '🔍 Debug: market names found in CFTC files' "
            "at the bottom of the page to see the exact market names in the downloaded files."
        )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────

def _tab_badge(cat: str) -> str:
    """Emoji badge for the most extreme signal in a category."""
    scores = [float(_data[m].iloc[-1]["cot_score"])
              for m in _CATS[cat]["markets"]
              if _data.get(m) is not None and not _data[m].empty]
    if not scores:
        return "⚫"
    extreme = max(scores, key=lambda s: abs(s - 50))
    z = _zone_for(extreme)
    return {"EXT. SHORT": "🟢", "SHORT": "🟡", "NEUTRAL": "⚪",
            "LONG": "🟠", "EXT. LONG": "🔴"}.get(z["label"], "⚪")

tab_crypto, tab_stocks, tab_comm = st.tabs([
    f"₿  Bitcoin & Crypto  {_tab_badge('crypto')}",
    f"📈  Stocks  {_tab_badge('stocks')}",
    f"🥇  Commodities  {_tab_badge('commodities')}",
])


def _render_market(name: str, mdf: pd.DataFrame | None) -> None:
    """Render gauge + metrics + chart for one market."""
    meta  = _MARKETS[name]
    color = meta["color"]
    icon  = meta["icon"]

    if mdf is None or mdf.empty:
        st.markdown(
            f"<div style='background:rgba(14,14,22,.60);border-radius:10px;"
            f"padding:28px 24px;border:1px dashed rgba(255,255,255,.12);"
            f"text-align:center'>"
            f"<div style='font-size:2.2em;margin-bottom:8px;opacity:.30'>{icon}</div>"
            f"<div style='font-size:.90em;font-weight:600;"
            f"color:rgba(255,255,255,.45)'>{name}</div>"
            f"<div style='font-size:.76em;color:rgba(255,255,255,.25);margin-top:5px'>"
            f"CFTC data not found in this report period</div>"
            f"<div style='font-size:.72em;color:rgba(38,166,154,.40);margin-top:3px'>"
            f"→ Click 🔄 Refresh Data in sidebar</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    # ── current values ────────────────────────────────────────────────────
    latest      = mdf.iloc[-1]
    score       = float(latest["cot_score"])
    net         = int(latest["mm_net"])
    mm_long     = int(latest["mm_long"])
    mm_short    = int(latest["mm_short"])
    wow         = latest["mm_net_wow"]
    wow_val, wow_col, wow_arrow = _delta_fmt(wow if pd.notna(wow) else None)
    net_col = "#26a69a" if net >= 0 else "#ef5350"

    zone = _zone_for(score)
    z_color, z_label, z_desc = zone["color"], zone["label"], zone["desc"]

    # ── layout ────────────────────────────────────────────────────────────
    col_g, col_s = st.columns([3, 2], gap="medium")

    with col_g:
        st.plotly_chart(_build_cot_gauge(score, _hex_rgba(color), z_label),
                        use_container_width=True)

    with col_s:
        # Signal banner
        st.markdown(
            f"<div style='background:linear-gradient(135deg,{z_color} 0%,"
            f"rgba(14,14,22,.95) 100%);border-radius:12px;padding:16px 18px;"
            f"margin-bottom:10px;border:1px solid rgba(255,255,255,.08)'>"
            f"<div style='font-size:.65em;letter-spacing:2px;"
            f"color:rgba(255,255,255,.35);margin-bottom:4px'>POSITIONING SIGNAL</div>"
            f"<div style='font-size:1.3em;font-weight:900;color:white;"
            f"letter-spacing:2px'>{icon} {z_label}</div>"
            f"<div style='font-size:.78em;color:rgba(255,255,255,.55);"
            f"margin-top:6px;line-height:1.4'>{z_desc}</div></div>",
            unsafe_allow_html=True,
        )

        # Metrics grid
        st.markdown(
            f"<div style='background:#12121e;border-radius:12px;padding:16px 18px;"
            f"border:1px solid rgba(255,255,255,.06)'>"

            f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;"
            f"margin-bottom:12px'>"

            f"<div><div style='font-size:.62em;letter-spacing:1.5px;"
            f"color:rgba(255,255,255,.30);margin-bottom:3px'>MM LONG</div>"
            f"<div style='font-size:1.05em;font-weight:700;color:#26a69a'>"
            f"{mm_long:,}</div></div>"

            f"<div><div style='font-size:.62em;letter-spacing:1.5px;"
            f"color:rgba(255,255,255,.30);margin-bottom:3px'>MM SHORT</div>"
            f"<div style='font-size:1.05em;font-weight:700;color:#ef5350'>"
            f"{mm_short:,}</div></div>"

            f"</div>"
            f"<div style='border-top:1px solid rgba(255,255,255,.07);"
            f"padding-top:12px;display:grid;grid-template-columns:1fr 1fr;gap:12px'>"

            f"<div><div style='font-size:.62em;letter-spacing:1.5px;"
            f"color:rgba(255,255,255,.30);margin-bottom:3px'>NET POSITION</div>"
            f"<div style='font-size:1.05em;font-weight:700;color:{net_col}'>"
            f"{net:+,}</div></div>"

            f"<div><div style='font-size:.62em;letter-spacing:1.5px;"
            f"color:rgba(255,255,255,.30);margin-bottom:3px'>WoW CHANGE</div>"
            f"<div style='font-size:1.05em;font-weight:700;color:{wow_col}'>"
            f"{wow_arrow} {wow_val}</div></div>"

            f"</div></div>",
            unsafe_allow_html=True,
        )

    # ── Historical chart — professional bar + EMA design ─────────────────
    st.markdown("")
    _section_header(f"{icon} {name} — Net MM Position", _lookback_opt)

    hist = mdf.tail(_lookback_wks)
    if not hist.empty:
        # Fetch price data for the panel below (done here so it shares the download)
        price_ticker = meta.get("price_ticker", "")
        _price_s: pd.Series | None = None
        if price_ticker:
            try:
                _pd = yf.download(price_ticker,
                                   period=f"{max(_lookback_wks // 52 + 1, 1)}y",
                                   interval="1wk", progress=False, auto_adjust=True)
                if not _pd.empty:
                    _cl = _pd["Close"]
                    if isinstance(_cl, pd.DataFrame):
                        _cl = _cl.iloc[:, 0]
                    _cl = _cl.squeeze()
                    _cl.index = pd.to_datetime(_cl.index).normalize()
                    _price_s = _cl.reindex(hist.index, method="ffill").dropna()
            except Exception:
                _price_s = None

        _net = hist["mm_net"].values
        _ema = (pd.Series(_net, index=hist.index)
                .ewm(span=4, min_periods=1).mean())   # 4-week EMA trend

        fig = go.Figure()

        # 1 ── Weekly bars — green = net long, red = net short
        _bar_cols = [
            "rgba(38,166,154,.72)"  if v >= 0 else
            "rgba(239,83,80,.72)"
            for v in _net
        ]
        fig.add_trace(go.Bar(
            x=hist.index, y=_net,
            marker_color=_bar_cols,
            marker_line_width=0,
            name="Weekly Net",
            showlegend=True,
            hovertemplate="%{x|%b %d, %Y}<br>Net MM: <b>%{y:+,.0f}</b> contracts<extra></extra>",
        ))

        # 2 ── EMA trend line — the hero signal (4-week exponential MA)
        fig.add_trace(go.Scatter(
            x=hist.index, y=_ema.values, mode="lines",
            name="4-wk EMA",
            line=dict(color="rgba(230,230,230,.92)", width=2.2),
            showlegend=True,
            hovertemplate="%{x|%b %d, %Y}<br>EMA: <b>%{y:+,.0f}</b><extra></extra>",
        ))

        # 3 ── Zero reference line
        fig.add_hline(y=0, line_color="rgba(255,255,255,.35)", line_width=1.2)

        # 4 ── Current EMA dot + label
        _ema_last, _ema_x = float(_ema.iloc[-1]), hist.index[-1]
        fig.add_trace(go.Scatter(
            x=[_ema_x], y=[_ema_last], mode="markers+text",
            marker=dict(size=10, color=color,
                        line=dict(color="white", width=1.5)),
            text=[f"  {_ema_last:+,.0f}"], textposition="middle right",
            textfont=dict(size=10, color=color),
            showlegend=False, hoverinfo="skip"))

        fig.update_layout(
            template="plotly_dark",
            height=300,
            margin=dict(l=0, r=10, t=6, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            bargap=0.15,
            yaxis=dict(title="Net Contracts", showgrid=True,
                       gridcolor="rgba(255,255,255,.05)"),
            legend=dict(orientation="h", y=1.06, x=0,
                        font=dict(size=10, color="rgba(255,255,255,.55)")),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Dedicated price panel below positioning chart ──────────────────
        if _price_s is not None and len(_price_s) > 2:
            fig_px = go.Figure()
            # Glow
            fig_px.add_trace(go.Scatter(
                x=_price_s.index, y=_price_s.values, mode="lines",
                line=dict(color=_hex_rgba(color, 0.08), width=10),
                showlegend=False, hoverinfo="skip"))
            # Price line
            fig_px.add_trace(go.Scatter(
                x=_price_s.index, y=_price_s.values, mode="lines",
                name=price_ticker,
                line=dict(color=_hex_rgba(color, 0.85), width=1.8),
                hovertemplate="%{x|%Y-%m-%d}<br>%{meta}: <b>%{y:,.2f}</b><extra></extra>",
                meta=price_ticker,
            ))
            # Current price dot + label
            _px_last, _py_last = _price_s.index[-1], float(_price_s.iloc[-1])
            fig_px.add_trace(go.Scatter(
                x=[_px_last], y=[_py_last], mode="markers+text",
                marker=dict(size=8, color=color,
                            line=dict(color="white", width=1.5)),
                text=[f"  {_py_last:,.0f}"], textposition="middle right",
                textfont=dict(size=10, color=color),
                showlegend=False, hoverinfo="skip"))
            fig_px.update_layout(
                template="plotly_dark",
                height=110,
                margin=dict(l=0, r=10, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(
                    title=meta.get("price_label", price_ticker),
                    showgrid=True,
                    gridcolor="rgba(255,255,255,.04)",
                    tickfont=dict(size=9, color=_hex_rgba(color, 0.50)),
                ),
                xaxis=dict(showgrid=False, zeroline=False),
                showlegend=False,
            )
            st.plotly_chart(fig_px, use_container_width=True)


# ─── TAB: CRYPTO ─────────────────────────────────────────────────────────────
with tab_crypto:
    _btc_mdf = _data.get("Bitcoin")

    # ── Bitcoin hero banner (only when data is available) ─────────────────
    if _btc_mdf is not None and not _btc_mdf.empty:
        # Current BTC price (daily from yfinance)
        _btc_price_html = ""
        try:
            _bp = yf.download("BTC-USD", period="5d", interval="1d",
                              progress=False, auto_adjust=True)
            if not _bp.empty:
                _bcl = _bp["Close"]
                if isinstance(_bcl, pd.DataFrame):
                    _bcl = _bcl.iloc[:, 0]
                _bcl = _bcl.squeeze().dropna()
                _btc_cur  = float(_bcl.iloc[-1])
                _btc_prev = float(_bcl.iloc[-2]) if len(_bcl) > 1 else _btc_cur
                _btc_chg  = (_btc_cur - _btc_prev) / _btc_prev * 100
                _bcc = "#26a69a" if _btc_chg >= 0 else "#ef5350"
                _bca = "↑" if _btc_chg >= 0 else "↓"
                _btc_price_html = (
                    f"<div style='text-align:center;border-left:1px solid rgba(255,255,255,.07);"
                    f"padding-left:32px'>"
                    f"<div style='font-size:.60em;letter-spacing:1.5px;"
                    f"color:rgba(255,255,255,.30);margin-bottom:3px'>BTC PRICE</div>"
                    f"<div style='font-size:1.8em;font-weight:700;color:white'>"
                    f"${_btc_cur:,.0f}</div>"
                    f"<div style='font-size:.88em;color:{_bcc}'>{_bca} {abs(_btc_chg):.2f}% 24h</div>"
                    f"</div>"
                )
        except Exception:
            pass

        _btc_score = float(_btc_mdf["cot_score"].iloc[-1])
        _btc_net   = int(_btc_mdf["mm_net"].iloc[-1])
        _btc_zone  = _zone_for(_btc_score)
        _bzone_col, _bzone_lbl, _bzone_desc = (
            _btc_zone["color"], _btc_zone["label"], _btc_zone["desc"]
        )

        st.markdown(
            f"<div style='background:linear-gradient(135deg,"
            f"rgba(247,147,26,.12) 0%,rgba(14,14,22,.98) 60%);"
            f"border:1px solid rgba(247,147,26,.22);border-radius:16px;"
            f"padding:24px 32px;margin-bottom:16px;"
            f"display:flex;align-items:center;gap:36px'>"

            f"<div style='font-size:3.8em;line-height:1'>₿</div>"

            f"<div style='flex:1'>"
            f"<div style='font-size:.60em;letter-spacing:2px;"
            f"color:rgba(255,255,255,.30);margin-bottom:4px'>CME BITCOIN FUTURES — CFTC COT</div>"
            f"<div style='font-size:1.6em;font-weight:900;color:#f7931a;"
            f"letter-spacing:1px'>Bitcoin</div>"
            f"<div style='font-size:.80em;color:rgba(255,255,255,.45);margin-top:4px'>"
            f"{_bzone_desc}</div>"
            f"</div>"

            f"{_btc_price_html}"

            f"<div style='text-align:center;border-left:1px solid rgba(255,255,255,.07);"
            f"padding-left:32px'>"
            f"<div style='font-size:.60em;letter-spacing:1.5px;"
            f"color:rgba(255,255,255,.30);margin-bottom:6px'>COT SIGNAL</div>"
            f"<div style='background:{_bzone_col};padding:8px 20px;border-radius:8px;"
            f"font-size:1.15em;font-weight:900;letter-spacing:2px;color:white;"
            f"margin-bottom:6px'>{_bzone_lbl}</div>"
            f"<div style='font-size:.78em;color:rgba(255,255,255,.42)'>"
            f"Score {_btc_score:.0f}/100  ·  Net {_btc_net:+,} contracts</div>"
            f"</div>"

            f"</div>",
            unsafe_allow_html=True,
        )

    _render_market("Bitcoin", _btc_mdf)

# ─── TAB: STOCKS ─────────────────────────────────────────────────────────────
with tab_stocks:
    for name in _CATS["stocks"]["markets"]:
        _render_market(name, _data[name])
        _divider()

# ─── TAB: COMMODITIES ────────────────────────────────────────────────────────
with tab_comm:
    for name in _CATS["commodities"]["markets"]:
        _render_market(name, _data[name])
        _divider()

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE  (all markets)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
_section_header("All Markets — COT Score Snapshot",
                "Sorted by positioning extreme (most extreme first)")

_rows = []
for name, mdf in _data.items():
    meta = _MARKETS[name]
    if mdf is None or mdf.empty:
        continue
    latest   = mdf.iloc[-1]
    sc       = float(latest["cot_score"])
    # Sparkline: last 12 weeks of COT score
    _spark12 = mdf["cot_score"].dropna().iloc[-12:].values
    net    = int(latest["mm_net"])
    wow    = latest["mm_net_wow"]
    zone   = _zone_for(sc)
    wow_val, wow_col, wow_arrow = _delta_fmt(wow if pd.notna(wow) else None)
    _rows.append({
        "market":    name,
        "icon":      meta["icon"],
        "color":     meta["color"],
        "score":     sc,
        "spark":     _spark12,
        "net":       net,
        "net_col":   "#26a69a" if net >= 0 else "#ef5350",
        "wow_val":   wow_val,
        "wow_col":   wow_col,
        "wow_arrow": wow_arrow,
        "z_label":   zone["label"],
        "z_color":   zone["color"],
        "cat":       meta["cat"],
    })

# Sort by distance from neutral (50) → most extreme first
_rows.sort(key=lambda r: abs(r["score"] - 50), reverse=True)

if _rows:
    # Build HTML table
    rows_html = ""
    for r in _rows:
        # Bidirectional bar centered at 50 (neutral)
        # score < 50: bar goes left  from center → starts at score%, width = (50-score)%
        # score >= 50: bar goes right from center → starts at 50%,    width = (score-50)%
        if r["score"] >= 50:
            bar_left  = 50.0
            bar_right = min(r["score"] - 50, 50)
        else:
            bar_left  = r["score"]
            bar_right = 50 - r["score"]
        rows_html += (
            f"<tr style='border-bottom:1px solid rgba(255,255,255,.05)'>"
            f"<td style='padding:10px 14px;font-size:.90em'>"
            f"{r['icon']} <b style='color:white'>{r['market']}</b></td>"
            f"<td style='padding:10px 14px;text-align:center'>"
            f"<span style='background:{r['z_color']};padding:2px 8px;"
            f"border-radius:4px;font-size:.72em;font-weight:700;"
            f"letter-spacing:1px;color:white'>{r['z_label']}</span></td>"
            f"<td style='padding:10px 14px'>"
            f"<div style='display:flex;align-items:center;gap:6px'>"
            f"<div style='width:120px;height:6px;background:rgba(255,255,255,.08);"
            f"border-radius:3px;overflow:hidden;position:relative'>"
            f"<div style='position:absolute;left:{bar_left:.0f}%;width:{bar_right:.0f}%;"
            f"height:100%;background:{r['color']};opacity:.80;border-radius:3px'></div>"
            f"</div>"
            f"<span style='font-size:.85em;font-weight:700;color:{r['color']}'>"
            f"{r['score']:.0f}</span>"
            f"<span style='margin-left:8px'>{_mini_sparkline_svg(r['spark'], r['color'])}</span>"
            f"</div></td>"
            f"<td style='padding:10px 14px;font-size:.85em;color:{r['net_col']}'>"
            f"{r['net']:+,}</td>"
            f"<td style='padding:10px 14px;font-size:.85em;color:{r['wow_col']}'>"
            f"{r['wow_arrow']} {r['wow_val']}</td>"
            f"<td style='padding:10px 14px;font-size:.75em;"
            f"color:rgba(255,255,255,.35);text-transform:capitalize'>{r['cat']}</td>"
            f"</tr>"
        )

    st.markdown(
        f"<div style='background:#0e0e1a;border-radius:12px;overflow:hidden;"
        f"border:1px solid rgba(255,255,255,.07)'>"
        f"<table style='width:100%;border-collapse:collapse;"
        f"color:rgba(255,255,255,.75)'>"
        f"<thead><tr style='background:rgba(255,255,255,.04)'>"
        f"<th style='padding:10px 14px;text-align:left;font-size:.68em;"
        f"letter-spacing:1.8px;color:rgba(255,255,255,.35)'>MARKET</th>"
        f"<th style='padding:10px 14px;text-align:center;font-size:.68em;"
        f"letter-spacing:1.8px;color:rgba(255,255,255,.35)'>SIGNAL</th>"
        f"<th style='padding:10px 14px;font-size:.68em;"
        f"letter-spacing:1.8px;color:rgba(255,255,255,.35)'>COT SCORE</th>"
        f"<th style='padding:10px 14px;font-size:.68em;"
        f"letter-spacing:1.8px;color:rgba(255,255,255,.35)'>NET POS.</th>"
        f"<th style='padding:10px 14px;font-size:.68em;"
        f"letter-spacing:1.8px;color:rgba(255,255,255,.35)'>WoW CHANGE</th>"
        f"<th style='padding:10px 14px;font-size:.68em;"
        f"letter-spacing:1.8px;color:rgba(255,255,255,.35)'>CATEGORY</th>"
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody></table></div>",
        unsafe_allow_html=True,
    )
else:
    st.info("No COT data available. Check your connection to CFTC.gov and click Refresh Data.")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC EXPANDER — shows market names inside the downloaded CFTC files
# Useful for finding the exact name of Bitcoin / S&P 500 in the TFF report
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("🔍 Debug: market names found in CFTC files"):
    _kw = ["BITCOIN", "BIT", "S&P", "NASDAQ", "GOLD", "SILVER", "CRUDE",
           "COPPER", "NATURAL GAS", "EURO", "DOLLAR", "TREASURY", "NOTE"]

    def _show_markets(raw: pd.DataFrame, label: str) -> None:
        if raw.empty:
            st.caption(f"**{label}** — file not downloaded")
            return
        nc = next((c for c in raw.columns if "Market_and_Exchange" in c), None)
        if nc is None:
            st.caption(f"**{label}** — market name column not found")
            return
        all_names = raw[nc].dropna().unique().tolist()
        filtered  = sorted({n for n in all_names
                            if any(k in str(n).upper() for k in _kw)})
        st.caption(f"**{label}** — {len(all_names)} total markets · "
                   f"{len(filtered)} matching keywords")
        st.code("\n".join(filtered) if filtered else "(none found)", language="")

    _show_markets(_raw_disagg, "Disaggregated (commodities)")
    _show_markets(_raw_fin,    "TFF (financial futures)")


# ─────────────────────────────────────────────────────────────────────────────
# READING GUIDE
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("📖 How to read the COT Tracker"):
    st.markdown("""
**What is COT?**
The CFTC publishes the Commitment of Traders report every Friday, showing positioning as of the previous Tuesday.
We track **Managed Money** (hedge funds, CTAs, large speculators) — the group whose extremes tend to mark turning points.

**COT Score (0–100)** — rolling 3-year min-max normalisation of net MM position:

| Score | Signal | Interpretation |
|---|---|---|
| 80–100 | 🔴 EXT. LONG | MM historically overbought — contrarian **SELL** signal |
| 60–80  | 🟠 LONG | Bullish bias — watch for exhaustion |
| 40–60  | ⚪ NEUTRAL | No positioning extreme |
| 20–40  | 🟡 SHORT | Bearish bias — watch for reversal |
| 0–20   | 🟢 EXT. SHORT | MM historically oversold — contrarian **BUY** signal |

**Key rule:** COT is a *contrarian* indicator. When everyone is positioned one way, the move is usually already priced in.

**Data frequency:** Weekly (Tuesday positioning, published Friday).
**Source:** CFTC.gov — Disaggregated Futures-Only Report. Free, no API key.
    """)

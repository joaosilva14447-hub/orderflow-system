"""
🌊 Breadth Thrust Detector
===========================
Market internals — detects capitulation, thrust and exhaustion signals.

Equities : S&P 500 real components (~500 stocks) via Yahoo Finance.
Crypto   : Top-100 coins via CoinGecko  +  Top-30 historical via Yahoo.

Zweig Breadth Thrust: breadth moves from <40 % to >61.5 % in ≤10 days → bullish.
Top Exhaustion      : breadth >70 % for 15+ days then drops below 60 % → caution.
"""
from __future__ import annotations

import sys, os
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import io, zipfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import streamlit as st
import requests
import yfinance as yf

st.set_page_config(
    page_title="Breadth Thrust Detector",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# SPDR Sector ETFs — "abbr" used in heatmap cards (avoids truncation)
_SECTORS: dict[str, dict] = {
    "XLK":  {"name": "Technology",       "abbr": "Tech",     "icon": "💻", "color": "#1a7bc4"},
    "XLF":  {"name": "Financials",       "abbr": "Finance",  "icon": "🏦", "color": "#26a69a"},
    "XLV":  {"name": "Healthcare",       "abbr": "Health",   "icon": "🏥", "color": "#2d9a50"},
    "XLY":  {"name": "Consumer Disc.",   "abbr": "Cons.D",   "icon": "🛍️", "color": "#f7931a"},
    "XLI":  {"name": "Industrials",      "abbr": "Indust",   "icon": "🏭", "color": "#8b6914"},
    "XLC":  {"name": "Comm. Services",   "abbr": "Comm",     "icon": "📡", "color": "#9b2020"},
    "XLE":  {"name": "Energy",           "abbr": "Energy",   "icon": "⚡", "color": "#cd853f"},
    "XLB":  {"name": "Materials",        "abbr": "Matrl",    "icon": "🔩", "color": "#b87333"},
    "XLRE": {"name": "Real Estate",      "abbr": "R.Estate", "icon": "🏠", "color": "#8a5520"},
    "XLU":  {"name": "Utilities",        "abbr": "Utils",    "icon": "💡", "color": "#505050"},
    "XLP":  {"name": "Cons. Staples",    "abbr": "Staples",  "icon": "🛒", "color": "#3a7a3a"},
}

# GICS Sector → SPDR ETF mapping (for real sector breadth computation)
_GICS_TO_ETF: dict[str, str] = {
    "Information Technology":  "XLK",
    "Financials":              "XLF",
    "Health Care":             "XLV",
    "Consumer Discretionary":  "XLY",
    "Industrials":             "XLI",
    "Communication Services":  "XLC",
    "Energy":                  "XLE",
    "Materials":               "XLB",
    "Real Estate":             "XLRE",
    "Utilities":               "XLU",
    "Consumer Staples":        "XLP",
}

# Top-30 crypto tickers available on Yahoo Finance (for historical MA breadth)
_CRYPTO_YF: list[str] = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD",
    "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "MATIC-USD",
    "LTC-USD", "LINK-USD", "BCH-USD", "XLM-USD", "ATOM-USD",
    "NEAR-USD", "ALGO-USD", "VET-USD", "FIL-USD", "ICP-USD",
    "HBAR-USD", "APT-USD", "ARB-USD", "OP-USD", "INJ-USD",
    "TIA-USD", "SUI-USD", "APE-USD", "SAND-USD", "MANA-USD",
]

# Breadth score zones (same visual language as other pages)
# Each tuple: (lo, hi, fill_color, label, label_color, faint_bg_color)
_BREADTH_ZONES = [
    (  0, 20, "rgba(155,25,25,.88)",  "EXTREME WEAK",   "rgba(225,85,85,.95)",  "rgba(155,25,25,.05)"),
    ( 20, 40, "rgba(165,82,16,.82)",  "WEAK",           "rgba(215,125,45,.95)", "rgba(165,82,16,.04)"),
    ( 40, 60, "rgba(58,58,58,.78)",   "NEUTRAL",        "rgba(185,185,185,.90)","rgba(58,58,58,.03)"),
    ( 60, 80, "rgba(18,124,68,.82)",  "HEALTHY",        "rgba(55,200,125,.95)", "rgba(18,124,68,.04)"),
    ( 80,100, "rgba(15,152,70,.88)",  "EXTREME STRONG", "rgba(45,225,110,.95)", "rgba(15,152,70,.05)"),
]

# Stablecoins + wrapped tokens — excluded from crypto breadth calculation
_STABLECOINS: set[str] = {
    "usdt", "usdc", "dai", "busd", "tusd", "frax", "usdp", "gusd",
    "usdd", "pyusd", "lusd", "crvusd", "gho", "fdusd", "usde", "susd",
    "eurc", "wbtc", "weth", "steth", "cbeth", "reth",   # wrapped / liquid staking
    "xaut", "paxg",                                       # gold tokens
}

# Thrust / Exhaustion detection thresholds
_THRUST_LOW       = 40.0    # breadth must dip below this
_THRUST_HIGH      = 61.5    # then surge above this within window
_THRUST_WINDOW    = 10      # trading days
_EXHAUST_HIGH     = 70.0    # breadth sustained above this
_EXHAUST_DURATION = 15      # consecutive days above _EXHAUST_HIGH
_EXHAUST_DROP     = 60.0    # then falls below this

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _hex_rgba(hex_c: str, alpha: float = 1.0) -> str:
    h = hex_c.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"

def _divider() -> None:
    st.markdown(
        "<div style='height:1px;"
        "background:linear-gradient(90deg,"
        "transparent 0%,rgba(38,166,154,.18) 20%,"
        "rgba(255,255,255,.06) 50%,"
        "rgba(38,166,154,.18) 80%,transparent 100%);"
        "margin:18px 0 6px'></div>",
        unsafe_allow_html=True,
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

def _zone_for_breadth(score: float) -> dict:
    for lo, hi, col, label, lcol, *_ in _BREADTH_ZONES:   # *_ handles extra faint_bg field
        if lo <= score < hi or (hi == 100 and score == 100):
            return {"bg": col, "label": label, "lcol": lcol}
    return {"bg": "rgba(58,58,58,.78)", "label": "NEUTRAL", "lcol": "rgba(185,185,185,.90)"}

# ─────────────────────────────────────────────────────────────────────────────
# BREADTH GAUGE  (reuses same arc design as other pages)
# ─────────────────────────────────────────────────────────────────────────────

def _build_breadth_gauge(score: float, color: str, label: str) -> go.Figure:
    R_OUT = 1.00; R_IN = 0.58; R_NEEDLE = 0.82; R_HUB = 0.055

    def s2a(s): return np.pi * (1.0 - s / 100.0)

    def arc(s0, s1, ri, ro, n=90):
        a0, a1 = s2a(s0), s2a(s1)
        fw = np.linspace(a0, a1, n); rv = np.linspace(a1, a0, n)
        x = np.r_[ro*np.cos(fw), ri*np.cos(rv), ro*np.cos(a0)]
        y = np.r_[ro*np.sin(fw), ri*np.sin(rv), ro*np.sin(a0)]
        return x, y

    zone_fills = [(z[0], z[1], z[2]) for z in _BREADTH_ZONES]
    fig = go.Figure()

    # Background
    xb, yb = arc(0, 100, R_IN*.97, R_OUT*1.02, 200)
    fig.add_trace(go.Scatter(x=xb, y=yb, fill="toself",
        fillcolor="rgba(14,14,22,.96)",
        line=dict(color="rgba(255,255,255,.04)", width=.5),
        showlegend=False, hoverinfo="skip"))

    # Zones
    for s0, s1, fc in zone_fills:
        xz, yz = arc(s0, s1, R_IN, R_OUT)
        fig.add_trace(go.Scatter(x=xz, y=yz, fill="toself", fillcolor=fc,
            line=dict(color="rgba(8,8,8,.55)", width=1.0),
            showlegend=False, hoverinfo="skip"))

    # Dim inactive, brighten active
    for s0, s1, _ in zone_fills:
        if not (s0 <= score < s1) and not (s1 == 100 and score == 100):
            xd, yd = arc(s0, s1, R_IN, R_OUT)
            fig.add_trace(go.Scatter(x=xd, y=yd, fill="toself",
                fillcolor="rgba(0,0,0,.22)", line=dict(width=0),
                showlegend=False, hoverinfo="skip"))
    act = next((z for z in zone_fills if z[0] <= score < z[1]), zone_fills[-1])
    xab, yab = arc(act[0], act[1], R_IN, R_OUT)
    fig.add_trace(go.Scatter(x=xab, y=yab, fill="toself",
        fillcolor="rgba(255,255,255,.12)", line=dict(width=0),
        showlegend=False, hoverinfo="skip"))
    xa, ya = arc(act[0], act[1], R_IN-.015, R_OUT+.03)
    fig.add_trace(go.Scatter(x=xa, y=ya, fill="toself",
        fillcolor="rgba(0,0,0,0)", line=dict(color=color, width=3.0),
        showlegend=False, hoverinfo="skip"))

    # Ticks + zone labels
    for tick in [0, 25, 50, 75, 100]:
        a_t = s2a(tick)
        fig.add_trace(go.Scatter(
            x=[R_OUT*np.cos(a_t), 1.10*np.cos(a_t)],
            y=[R_OUT*np.sin(a_t), 1.10*np.sin(a_t)],
            mode="lines", line=dict(color="rgba(255,255,255,.60)", width=1.8),
            showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=1.19*np.cos(a_t), y=1.19*np.sin(a_t),
            text=f"<b>{tick}</b>", showarrow=False,
            font=dict(size=11, color="rgba(255,255,255,.85)"))

    for s0, s1, _, zlbl, zcol, *__ in _BREADTH_ZONES:
        a_mid = s2a((s0+s1)/2)
        fig.add_annotation(x=1.40*np.cos(a_mid), y=1.40*np.sin(a_mid),
            text=f"<b>{zlbl}</b>", showarrow=False,
            font=dict(size=9, color=zcol, family="monospace"), align="center")

    # Needle
    a_n = s2a(score); bw = 0.030; bk = 0.08
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
    fig.add_trace(go.Scatter(x=(R_HUB+.028)*np.cos(th), y=(R_HUB+.028)*np.sin(th),
        fill="toself", fillcolor="rgba(255,255,255,.10)", line=dict(width=0),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=R_HUB*np.cos(th), y=R_HUB*np.sin(th),
        fill="toself", fillcolor=color,
        line=dict(color="rgba(255,255,255,.90)", width=2.0),
        showlegend=False, hoverinfo="skip"))

    # Score number
    fig.add_annotation(x=0, y=-0.26, text=f"<b>{score:.0f}</b>", showarrow=False,
        font=dict(size=58, color=color, family="Arial Black, Arial, sans-serif"))

    fig.update_layout(
        xaxis=dict(range=[-1.55, 1.55], visible=False, showgrid=False, zeroline=False),
        yaxis=dict(range=[-0.62, 1.58], visible=False, showgrid=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=400,
        margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400*2, show_spinner=False)
def _get_sp500_tickers(cache_date: str) -> list[str]:   # noqa: ARG001
    """Fetch current S&P 500 constituents from Wikipedia via requests."""
    try:
        from io import StringIO as _SIO
        _hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=_hdrs, timeout=20,
        )
        r.raise_for_status()
        tables = pd.read_html(_SIO(r.text))
        tickers = tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 400:           # sanity check — must have near-full list
            return sorted(tickers)
    except Exception:
        pass
    # ── Fallback at correct indentation — reached when Wikipedia fails OR
    # sanity check fails (< 400 tickers).  Uses a representative ~100 large caps.
    return [
        "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","BRK-B","AVGO",
        "JPM","LLY","V","UNH","XOM","MA","HD","PG","COST","JNJ","ABBV","MRK","CRM",
        "BAC","NFLX","AMD","CVX","TMO","ORCL","WMT","KO","ACN","MCD","ABT","DHR",
        "LIN","TXN","NEE","BMY","PM","ISRG","AMGN","UNP","SCHW","INTU","QCOM","SPGI",
        "GE","AMAT","RTX","CAT","GS","HON","SYK","BLK","MDLZ","AXP","T","ADI","C",
        "BKNG","MS","DE","GILD","TJX","MMC","VRTX","PLD","CI","REGN","MU","LRCX",
        "ZTS","ELV","CB","MCO","ETN","AON","KLAC","CME","SO","DUK","WM","ITW","APH",
        "SHW","HCA","MCK","NOC","GD","EMR","COP","EOG","PSX","PH","EW","HLT","MAR",
    ]


@st.cache_data(ttl=86400*2, show_spinner=False)
def _load_sp500_prices(cache_date: str) -> pd.DataFrame:   # noqa: ARG001
    """Download 1-year daily close prices for S&P 500 components (batch)."""
    tickers = _get_sp500_tickers(cache_date)
    raw = yf.download(
        tickers, period="1y", interval="1d",
        auto_adjust=True, progress=False, threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
    # Keep columns with at least 70 % valid data
    close = close.dropna(axis=1, thresh=int(len(close) * 0.70))
    close.index = pd.to_datetime(close.index).normalize()
    return close


@st.cache_data(ttl=86400*2, show_spinner=False)
def _get_sp500_sectors(cache_date: str) -> pd.DataFrame:   # noqa: ARG001
    """Fetch S&P 500 ticker → GICS sector mapping from Wikipedia."""
    try:
        from io import StringIO as _SIO
        _hdrs = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=_hdrs, timeout=20,
        )
        r.raise_for_status()
        table = pd.read_html(_SIO(r.text))[0]
        table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
        return table[["Symbol", "GICS Sector"]].rename(
            columns={"Symbol": "ticker", "GICS Sector": "sector"})
    except Exception:
        return pd.DataFrame(columns=["ticker", "sector"])


def _compute_sector_breadth(prices: pd.DataFrame,
                             sector_df: pd.DataFrame) -> dict[str, float | None]:
    """Compute real % above MA50 for each SPDR sector using actual constituents."""
    if prices.empty or sector_df.empty:
        return {}
    ma50 = prices.rolling(50, min_periods=35).mean()
    above = (prices > ma50).where(ma50.notna())   # exclude stocks without MA50
    result: dict = {}
    for gics, etf in _GICS_TO_ETF.items():
        tickers = sector_df[sector_df["sector"] == gics]["ticker"].tolist()
        cols    = [t for t in tickers if t in above.columns]
        if cols:
            result[etf] = float(above[cols].iloc[-1].mean() * 100)
        else:
            result[etf] = None
    return result


@st.cache_data(ttl=86400*2, show_spinner=False)
def _load_sector_prices(cache_date: str) -> pd.DataFrame:   # noqa: ARG001
    """Download 5-year daily prices for sector ETFs + SPY.
    5-year history needed for statistically meaningful historical validation.
    """
    tickers = list(_SECTORS.keys()) + ["SPY"]
    raw = yf.download(tickers, period="5y", interval="1d",
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw.copy()
    close.index = pd.to_datetime(close.index).normalize()
    return close.dropna(how="all")


@st.cache_data(ttl=3600, show_spinner=False)   # crypto refreshes hourly
def _load_crypto_cg(cache_date: str) -> pd.DataFrame:   # noqa: ARG001
    """Top-100 coins snapshot from CoinGecko free API."""
    try:
        url = (
            "https://api.coingecko.com/api/v3/coins/markets"
            "?vs_currency=usd&order=market_cap_desc&per_page=100&page=1"
            "&sparkline=false"
            "&price_change_percentage=24h,7d,14d,30d"
        )
        r = requests.get(url, timeout=20,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        for col in ["price_change_percentage_24h",
                    "price_change_percentage_7d_in_currency",
                    "price_change_percentage_14d_in_currency",
                    "price_change_percentage_30d_in_currency"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400*2, show_spinner=False)
def _load_crypto_hist(cache_date: str) -> pd.DataFrame:   # noqa: ARG001
    """Historical weekly close for top-30 crypto (for MA breadth)."""
    raw = yf.download(
        _CRYPTO_YF, period="2y", interval="1d",
        auto_adjust=True, progress=False, threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw.copy()
    close.index = pd.to_datetime(close.index).normalize()
    return close.dropna(axis=1, thresh=int(len(close) * 0.60))


# ─────────────────────────────────────────────────────────────────────────────
# BREADTH COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_equity_breadth(prices: pd.DataFrame) -> dict:
    """Compute full breadth metrics from price matrix."""
    if prices.empty or len(prices) < 20:
        return {}

    ma20  = prices.rolling(20,  min_periods=15).mean()
    ma50  = prices.rolling(50,  min_periods=35).mean()
    ma200 = prices.rolling(200, min_periods=150).mean()

    p     = prices.iloc[-1]
    prev5 = prices.iloc[-6] if len(prices) >= 6 else prices.iloc[0]
    prev20= prices.iloc[-21] if len(prices) >= 21 else prices.iloc[0]

    pct_ma20   = (p > ma20.iloc[-1]).mean()   * 100
    pct_ma50   = (p > ma50.iloc[-1]).mean()   * 100
    pct_ma200  = (p > ma200.iloc[-1]).mean()  * 100
    pct_pos5d  = (p > prev5).mean()           * 100
    pct_pos20d = (p > prev20).mean()          * 100

    high52_ts = prices.rolling(252, min_periods=200).max()
    low52_ts  = prices.rolling(252, min_periods=200).min()
    high52    = high52_ts.iloc[-1]
    low52     = low52_ts.iloc[-1]
    pct_52h   = ((p / high52) >= 0.98).mean() * 100
    pct_52l   = ((p / low52)  <= 1.02).mean() * 100

    # ── New Highs / New Lows Ratio ─────────────────────────────────────────
    # NH/(NH+NL): >60 = bullish, <40 = bearish, extremes mark reversals
    nh_ts = ((prices / high52_ts) >= 0.98).sum(axis=1)
    nl_ts = ((prices / low52_ts)  <= 1.02).sum(axis=1)
    _total_hnl = nh_ts + nl_ts
    nh_nl_hist = (nh_ts / _total_hnl.replace(0, np.nan) * 100).where(
        _total_hnl >= 5)   # only meaningful when ≥5 stocks near extremes
    nh_nl_ratio = float(nh_ts.iloc[-1] / max(_total_hnl.iloc[-1], 1) * 100)

    # ── Fix #3: Historical series using valid-count mask ────────────────────
    # (prices > ma50) returns False when ma50 is NaN → inflates "below MA50"
    # Fix: use .where() so stocks with no MA50 yet are excluded from the mean
    above_ma50_masked = (prices > ma50).where(ma50.notna())   # NaN = excluded
    hist_ma50_raw     = above_ma50_masked.mean(axis=1) * 100  # skipna=True by default

    # Only keep dates where at least 30% of stocks have valid MA50 data
    valid_count = above_ma50_masked.notna().sum(axis=1)
    min_valid   = max(10, int(len(prices.columns) * 0.30))
    hist_ma50   = hist_ma50_raw[valid_count >= min_valid].dropna()

    # Advance/Decline ratio + cumulative A/D Line
    daily_ret    = prices.pct_change(1)
    adv_ratio    = (daily_ret > 0).mean(axis=1) * 100
    thrust_ratio = adv_ratio.ewm(span=10, min_periods=5).mean()
    _adv_cnt = (daily_ret > 0).sum(axis=1)
    _dec_cnt = (daily_ret < 0).sum(axis=1)
    ad_line  = (_adv_cnt - _dec_cnt).cumsum().reindex(hist_ma50.index)

    # Composite breadth score (now includes NH/NL ratio)
    breadth_score = (
        pct_ma20    * 0.18 +
        pct_ma50    * 0.27 +
        pct_ma200   * 0.22 +
        pct_pos20d  * 0.13 +
        nh_nl_ratio * 0.12 +
        (100 - pct_52l) * 0.08
    )

    # ── Fix #4: Breadth momentum — Δ vs 10 trading days ago ─────────────────
    breadth_delta = None
    if len(hist_ma50) >= 11:
        breadth_delta = float(hist_ma50.iloc[-1] - hist_ma50.iloc[-11])

    return {
        "score":          float(np.clip(breadth_score, 0, 100)),
        "pct_ma20":       float(pct_ma20),
        "pct_ma50":       float(pct_ma50),
        "pct_ma200":      float(pct_ma200),
        "pct_pos5d":      float(pct_pos5d),
        "pct_pos20d":     float(pct_pos20d),
        "pct_52h":        float(pct_52h),
        "pct_52l":        float(pct_52l),
        "nh_nl_ratio":    nh_nl_ratio,
        "nh_nl_hist":     nh_nl_hist.dropna(),
        "hist_ma50":      hist_ma50,
        "thrust_ratio":   thrust_ratio.dropna(),
        "breadth_delta":  breadth_delta,
        "ad_line":        ad_line.dropna(),
        "n_stocks":       len(prices.columns),
    }


def _compute_crypto_breadth(cg_df: pd.DataFrame,
                             hist_prices: pd.DataFrame) -> dict:
    """Breadth metrics for crypto."""
    result: dict = {}

    if not cg_df.empty:
        # ── Fix #2: Exclude stablecoins + wrapped tokens ──────────────────
        df_real = cg_df[~cg_df["symbol"].isin(_STABLECOINS)].copy()
        result["n_coins"]      = len(cg_df)
        result["n_real_coins"] = len(df_real)
        result["coins"]        = cg_df          # keep full set for treemap

        p24  = df_real["price_change_percentage_24h"]
        p7   = df_real["price_change_percentage_7d_in_currency"]
        p30  = df_real["price_change_percentage_30d_in_currency"]

        result["pct_pos24h"]  = float((p24  > 0).mean() * 100)
        result["pct_pos7d"]   = float((p7   > 0).mean() * 100)
        result["pct_pos30d"]  = float((p30  > 0).mean() * 100)

        # Altcoin breadth (exclude BTC, ETH, stablecoins)
        alts = df_real[~df_real["symbol"].isin(["btc", "eth"])]
        if not alts.empty:
            result["alt_pct_7d"] = float(
                (alts["price_change_percentage_7d_in_currency"] > 0).mean() * 100)
        else:
            result["alt_pct_7d"] = 0.0

        # Composite crypto breadth score (stablecoin-free)
        result["score"] = float(np.clip(
            result["pct_pos24h"] * 0.20 +
            result["pct_pos7d"]  * 0.50 +
            result["pct_pos30d"] * 0.30,
            0, 100,
        ))

    # Historical MA breadth from yfinance
    if not hist_prices.empty and len(hist_prices) >= 20:
        ma20h = hist_prices.rolling(20,  min_periods=15).mean()
        ma50h = hist_prices.rolling(50,  min_periods=35).mean()
        ph    = hist_prices.iloc[-1]

        # Use .where() to exclude coins without enough data for MA50
        # (same fix as equity breadth — avoids deflating early readings)
        above_ma50h = (hist_prices > ma50h).where(ma50h.notna())
        result["hist_pct_ma50"] = above_ma50h.mean(axis=1) * 100
        result["pct_ma20_hist"] = float((ph > ma20h.iloc[-1]).mean() * 100)
        result["pct_ma50_hist"] = float((ph > ma50h.iloc[-1]).mean() * 100)
        result["n_hist_coins"]  = len(hist_prices.columns)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# THRUST / EXHAUSTION DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_events(hist_ma50: pd.Series) -> pd.DataFrame:
    """Detect Zweig Breadth Thrust and Top Exhaustion events."""
    if len(hist_ma50) < _THRUST_WINDOW + 1:
        return pd.DataFrame(columns=["date", "type", "breadth"])

    events = []
    above_high_streak = 0

    for i in range(_THRUST_WINDOW, len(hist_ma50)):
        val = float(hist_ma50.iloc[i])

        # ── Bottom Thrust ──────────────────────────────────────────────────
        window = hist_ma50.iloc[i - _THRUST_WINDOW : i + 1]
        if window.min() < _THRUST_LOW and val > _THRUST_HIGH:
            # Only flag if previous event was not also a thrust within 5 days
            if not events or (events[-1]["type"] != "BOTTOM_THRUST" or
                              (hist_ma50.index[i] - events[-1]["date"]).days > 5):
                events.append({
                    "date":    hist_ma50.index[i],
                    "type":    "BOTTOM_THRUST",
                    "breadth": val,
                })

        # ── Top Exhaustion ─────────────────────────────────────────────────
        if val > _EXHAUST_HIGH:
            above_high_streak += 1
        else:
            if (above_high_streak >= _EXHAUST_DURATION and
                    val < _EXHAUST_DROP):
                if not events or (events[-1]["type"] != "TOP_EXHAUST" or
                                  (hist_ma50.index[i] - events[-1]["date"]).days > 5):
                    events.append({
                        "date":    hist_ma50.index[i],
                        "type":    "TOP_EXHAUST",
                        "breadth": val,
                    })
            above_high_streak = 0

    return pd.DataFrame(events) if events else pd.DataFrame(
        columns=["date", "type", "breadth"])


def _historical_validation(hist_ma50: pd.Series,
                            spy_prices: pd.Series,
                            current_score: float,
                            band: float = 7.0) -> dict:
    """
    Find all past dates where breadth was within ±band of current_score.
    Compute SPY forward returns at 10, 20, 30 trading days.
    Returns statistics for the 'what happened next?' card.
    """
    if len(hist_ma50) < 40 or spy_prices.empty:
        return {}

    similar = hist_ma50[
        (hist_ma50 >= current_score - band) &
        (hist_ma50 <= current_score + band)
    ].index

    # Remove the last 30 trading days (no forward data yet)
    cutoff = hist_ma50.index[-31] if len(hist_ma50) > 31 else hist_ma50.index[0]
    similar = similar[similar <= cutoff]

    if len(similar) < 3:
        return {"n": 0}

    fwd: dict = {10: [], 20: [], 30: []}
    spy_idx = spy_prices.index

    for date in similar:
        # Find closest SPY date
        loc = spy_idx.searchsorted(date)
        if loc >= len(spy_idx):
            continue
        p0 = float(spy_prices.iloc[loc])
        if p0 <= 0:
            continue
        for days, lst in fwd.items():
            future_loc = loc + days
            if future_loc < len(spy_idx):
                lst.append((float(spy_prices.iloc[future_loc]) / p0 - 1) * 100)

    def _stats(lst: list) -> dict:
        if not lst:
            return {}
        a = np.array(lst)
        return {
            "avg":     float(a.mean()),
            "med":     float(np.median(a)),
            "pct_pos": float((a > 0).mean() * 100),
            "best":    float(a.max()),
            "worst":   float(a.min()),
            "n":       len(a),
        }

    return {
        "n":    len(similar),
        "band": band,
        "10d":  _stats(fwd[10]),
        "20d":  _stats(fwd[20]),
        "30d":  _stats(fwd[30]),
    }


def _event_spy_returns(event_date: pd.Timestamp,
                       spy_prices: pd.Series) -> dict:
    """SPY forward returns (+10d / +20d / +30d trading days) from event_date."""
    if spy_prices.empty:
        return {}
    spy_prices = spy_prices.dropna()
    spy_idx = spy_prices.index
    loc = int(spy_idx.searchsorted(pd.Timestamp(event_date)))
    if loc >= len(spy_idx):
        return {}
    p0 = float(spy_prices.iloc[loc])
    if p0 <= 0:
        return {}
    result = {}
    for days, key in [(10, "+10d"), (20, "+20d"), (30, "+30d")]:
        fl = loc + days
        if fl < len(spy_idx):
            result[key] = (float(spy_prices.iloc[fl]) / p0 - 1) * 100
    return result


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

_now_utc   = datetime.utcnow()
_today_key = _now_utc.strftime("%Y-%m-%d")

st.sidebar.title("🌊 Breadth Tracker")
st.sidebar.caption("Market internals — S&P 500 components + Top-100 crypto.")

_lookback_map = {"3 Months": 63, "6 Months": 126, "1 Year": 252}
_lookback_opt = st.sidebar.selectbox("Chart History", list(_lookback_map), index=2)
_lookback_d   = _lookback_map[_lookback_opt]

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.caption(
    "**Data sources:**\n"
    "- Wikipedia (S&P 500 constituents)\n"
    "- Yahoo Finance (stock + crypto prices)\n"
    "- CoinGecko API (top-100 crypto snapshot)"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("⏳ Loading S&P 500 breadth data (~500 components · first load 2-3 min)..."):
    _sp500_prices  = _load_sp500_prices(_today_key)
    _sector_prices = _load_sector_prices(_today_key)
    _sp500_sectors = _get_sp500_sectors(_today_key)   # GICS sector mapping

with st.spinner("Loading crypto data..."):
    _cg_df        = _load_crypto_cg(_today_key)
    _crypto_hist  = _load_crypto_hist(_today_key)

_eq_breadth      = _compute_equity_breadth(_sp500_prices)
_cry_breadth     = _compute_crypto_breadth(_cg_df, _crypto_hist)
_sector_breadth  = _compute_sector_breadth(_sp500_prices, _sp500_sectors)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("### 🌊 Breadth Thrust Detector")
st.caption(
    f"Data: {_today_key} UTC  ·  "
    f"S&P 500: {_eq_breadth.get('n_stocks', '—')} stocks  ·  "
    f"Crypto: {_cry_breadth.get('n_real_coins', _cry_breadth.get('n_coins', '—'))} real coins (stablecoins excluded)  ·  "
    f"History: {_lookback_opt}"
)

# ─────────────────────────────────────────────────────────────────────────────
# TABS  (badge emoji = zone color)
# ─────────────────────────────────────────────────────────────────────────────

_ZONE_BADGE = {
    "EXTREME WEAK":   "🔴",
    "WEAK":           "🟠",
    "NEUTRAL":        "⚪",
    "HEALTHY":        "🟢",
    "EXTREME STRONG": "🟢",
}
_eq_zone   = _zone_for_breadth(_eq_breadth.get("score",  50))
_cry_zone  = _zone_for_breadth(_cry_breadth.get("score", 50))
_eq_badge  = _ZONE_BADGE.get(_eq_zone["label"],  "⚪")
_cry_badge = _ZONE_BADGE.get(_cry_zone["label"], "⚪")

tab_eq, tab_cry = st.tabs([
    f"📈  Equities  {_eq_badge}  ({_eq_breadth.get('score', 0):.0f})",
    f"₿  Crypto  {_cry_badge}  ({_cry_breadth.get('score', 0):.0f})",
])

# ═══════════════════════════════════════════════════════════════════════════
# EQUITIES TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_eq:
    if not _eq_breadth:
        st.error("S&P 500 price data unavailable. Click 🔄 Refresh Data.")
    else:
        eq_score  = _eq_breadth["score"]
        zone_info = _zone_for_breadth(eq_score)   # single source of truth
        z_bg      = zone_info["bg"]
        z_lbl     = zone_info["label"]
        z_lcol    = zone_info["lcol"]

        # ── Hex color for gauge (use zone's base color) ───────────────────
        _eq_colors = {
            "EXTREME WEAK":   "#9b1919",
            "WEAK":           "#a55214",
            "NEUTRAL":        "#505050",
            "HEALTHY":        "#18a050",  # noqa: E241
            "EXTREME STRONG": "#0f9846",
        }
        gauge_color = _eq_colors.get(z_lbl, "#505050")

        # ── Thrust / Exhaustion detection ─────────────────────────────────
        _events_df = pd.DataFrame()
        if "hist_ma50" in _eq_breadth and len(_eq_breadth["hist_ma50"]) > 20:
            _events_df = _detect_events(_eq_breadth["hist_ma50"])

        # Most recent event
        _last_event: dict = {}
        if not _events_df.empty:
            _last_event = _events_df.sort_values("date").iloc[-1].to_dict()

        # ── Signal banner ─────────────────────────────────────────────────
        _is_thrust  = (_last_event.get("type") == "BOTTOM_THRUST" and
                       (pd.Timestamp.now() - pd.Timestamp(_last_event["date"])).days <= 15)
        _is_exhaust = (_last_event.get("type") == "TOP_EXHAUST" and
                       (pd.Timestamp.now() - pd.Timestamp(_last_event["date"])).days <= 15)

        if _is_thrust:
            _days_ago = (pd.Timestamp.now() - pd.Timestamp(_last_event["date"])).days
            st.markdown(
                f"<div style='background:linear-gradient(135deg,rgba(15,152,70,.25) 0%,"
                f"rgba(14,14,22,.95) 100%);border:1px solid rgba(15,152,70,.50);"
                f"border-radius:14px;padding:18px 24px;margin-bottom:16px'>"
                f"<div style='font-size:1.6em;font-weight:900;color:#26a69a;"
                f"letter-spacing:2px'>🚀 BREADTH THRUST DETECTED</div>"
                f"<div style='font-size:.85em;color:rgba(255,255,255,.60);margin-top:4px'>"
                f"Detected {_days_ago}d ago · Breadth surged from extreme weakness to "
                f"{_last_event['breadth']:.1f}% above MA50 in ≤10 days — historically "
                f"one of the strongest bull signals.</div></div>",
                unsafe_allow_html=True,
            )
        elif _is_exhaust:
            _days_ago = (pd.Timestamp.now() - pd.Timestamp(_last_event["date"])).days
            st.markdown(
                f"<div style='background:linear-gradient(135deg,rgba(155,25,25,.25) 0%,"
                f"rgba(14,14,22,.95) 100%);border:1px solid rgba(155,25,25,.50);"
                f"border-radius:14px;padding:18px 24px;margin-bottom:16px'>"
                f"<div style='font-size:1.6em;font-weight:900;color:#ef5350;"
                f"letter-spacing:2px'>⚠️ BREADTH EXHAUSTION DETECTED</div>"
                f"<div style='font-size:.85em;color:rgba(255,255,255,.60);margin-top:4px'>"
                f"Detected {_days_ago}d ago · Breadth sustained above 70% for 15+ days "
                f"then fell below 60% — breadth expansion is over, caution warranted.</div></div>",
                unsafe_allow_html=True,
            )

        # ── Thrust setup "in development" tracker ─────────────────────────
        _hist_for_setup = _eq_breadth.get("hist_ma50", pd.Series(dtype=float))
        if (not _is_thrust and not _is_exhaust and len(_hist_for_setup) >= 2):
            _recent_window = _hist_for_setup.tail(_THRUST_WINDOW)
            _cur_brd = float(_hist_for_setup.iloc[-1])
            if float(_recent_window.min()) < _THRUST_LOW and _cur_brd < _THRUST_HIGH:
                _below_idx = _recent_window[_recent_window < _THRUST_LOW]
                _setup_start = _below_idx.index[0]
                _elapsed_days = (_recent_window.index[-1] - _setup_start).days
                _days_left = max(0, _THRUST_WINDOW - _elapsed_days)
                _prog_pct  = min(100.0, (_cur_brd / _THRUST_HIGH) * 100.0)
                _prog_w    = f"{_prog_pct:.0f}%"
                st.markdown(
                    f"<div style='background:linear-gradient(135deg,rgba(38,166,154,.18) 0%,"
                    f"rgba(14,14,22,.95) 100%);border:1px solid rgba(38,166,154,.40);"
                    f"border-radius:14px;padding:18px 24px;margin-bottom:16px'>"
                    f"<div style='font-size:1.0em;font-weight:800;color:#26a69a;"
                    f"letter-spacing:2px;margin-bottom:8px'>⏳ THRUST SETUP IN DEVELOPMENT</div>"
                    f"<div style='font-size:.84em;color:rgba(255,255,255,.60);margin-bottom:10px'>"
                    f"Breadth touched &lt;{_THRUST_LOW:.0f}% within last {_THRUST_WINDOW} days · "
                    f"Target: reach {_THRUST_HIGH}% within "
                    f"<b style='color:rgba(255,255,255,.85)'>{_days_left}d</b> remaining · "
                    f"Current breadth: <b style='color:#26a69a'>{_cur_brd:.1f}%</b></div>"
                    f"<div style='background:rgba(255,255,255,.08);border-radius:4px;"
                    f"height:8px;overflow:hidden'>"
                    f"<div style='background:linear-gradient(90deg,#26a69a,#0f9846);"
                    f"height:100%;width:{_prog_w};border-radius:4px'></div></div>"
                    f"<div style='display:flex;justify-content:space-between;margin-top:4px'>"
                    f"<span style='font-size:.68em;color:rgba(255,255,255,.30)'>0%</span>"
                    f"<span style='font-size:.68em;color:rgba(255,255,255,.30)'>"
                    f"{_THRUST_HIGH}% target</span></div></div>",
                    unsafe_allow_html=True,
                )

        # ── Market Regime + Price/Breadth Divergence ──────────────────────
        _spy_r = (_sector_prices["SPY"]
                  if "SPY" in _sector_prices.columns
                  else pd.Series(dtype=float))

        _reg_lbl = "UNKNOWN"; _reg_col = "#505050"; _reg_bg = "rgba(50,50,50,.12)"
        _reg_desc = "Insufficient price history"; _spy_gap_str = ""

        if len(_spy_r) >= 150:
            _spy_now  = float(_spy_r.iloc[-1])
            _spy_ma2  = float(_spy_r.rolling(200, min_periods=150).mean().iloc[-1])
            _above200 = _spy_now > _spy_ma2
            _spy_gap  = (_spy_now / _spy_ma2 - 1) * 100
            _spy_gap_str = f"SPY {_spy_gap:+.1f}% vs MA200"
            if _above200 and eq_score >= 60:
                _reg_lbl  = "BULL CONFIRMED"
                _reg_col  = "#0f9846"
                _reg_bg   = "rgba(15,152,70,.15)"
                _reg_desc = "Broad participation · Price above MA200 · High conviction long bias"
            elif _above200 and eq_score >= 40:
                _reg_lbl  = "BULL WARNING"
                _reg_col  = "#c8b400"
                _reg_bg   = "rgba(200,180,0,.12)"
                _reg_desc = "Narrowing leadership · Price above MA200 · Watch for breadth decay"
            elif not _above200 and eq_score >= 55:
                _reg_lbl  = "BEAR RALLY"
                _reg_col  = "#e07820"
                _reg_bg   = "rgba(224,120,32,.13)"
                _reg_desc = "Counter-trend bounce · Price below MA200 · Treat strength with caution"
            elif not _above200 and eq_score >= 30:
                _reg_lbl  = "BEAR TRANSITION"
                _reg_col  = "#c03820"
                _reg_bg   = "rgba(192,56,32,.16)"
                _reg_desc = "Deteriorating internals · Price below MA200 · Reduce risk exposure"
            else:
                _reg_lbl  = "BEAR CONFIRMED"
                _reg_col  = "#9b1919"
                _reg_bg   = "rgba(155,25,25,.20)"
                _reg_desc = "Breadth collapsed · Price below MA200 · Capital preservation mode"

        _dv_lbl  = "NO DIVERGENCE"
        _dv_col  = "rgba(100,100,100,.80)"
        _dv_desc = "Price and breadth moving in sync (20-day window)"
        _dv_icon = "≡"
        _hist_d  = _eq_breadth.get("hist_ma50", pd.Series(dtype=float))
        if len(_spy_r) >= 21 and len(_hist_d) >= 21:
            _spy_r20 = float((_spy_r.iloc[-1] / _spy_r.iloc[-21] - 1) * 100)
            _brd_20  = float(_hist_d.iloc[-1] - _hist_d.iloc[-21])
            if _spy_r20 > 2.5 and _brd_20 < -4.0:
                _dv_lbl  = "BEARISH DIVERGENCE"
                _dv_col  = "#ef5350"
                _dv_icon = "⚠"
                _dv_desc = (f"SPY {_spy_r20:+.1f}% but breadth {_brd_20:+.1f}pp (20d) — "
                            f"gains built on narrowing participation")
            elif _spy_r20 < -2.5 and _brd_20 > 4.0:
                _dv_lbl  = "BULLISH DIVERGENCE"
                _dv_col  = "#26a69a"
                _dv_icon = "↗"
                _dv_desc = (f"SPY {_spy_r20:+.1f}% but breadth {_brd_20:+.1f}pp (20d) — "
                            f"internals recovering ahead of price")

        # 5-state spectrum chips
        _all_regimes = [
            ("BEAR CONFIRMED",  "#9b1919"),
            ("BEAR TRANSITION", "#c03820"),
            ("BEAR RALLY",      "#e07820"),
            ("BULL WARNING",    "#c8b400"),
            ("BULL CONFIRMED",  "#0f9846"),
        ]
        _regime_chips = ""
        for _rn, _rc in _all_regimes:
            _is_cur = (_rn == _reg_lbl)
            _c_bg   = _hex_rgba(_rc, 0.28) if _is_cur else "rgba(255,255,255,.04)"
            _c_col  = _rc if _is_cur else "rgba(255,255,255,.22)"
            _c_brd  = f"1px solid {_rc}" if _is_cur else "1px solid rgba(255,255,255,.07)"
            _c_wt   = "800" if _is_cur else "400"
            _c_glow = f";box-shadow:0 0 12px {_hex_rgba(_rc, 0.40)}" if _is_cur else ""
            _regime_chips += (
                f"<span style='background:{_c_bg};border:{_c_brd};border-radius:20px;"
                f"padding:4px 12px;font-size:.70em;font-weight:{_c_wt};"
                f"color:{_c_col};letter-spacing:.8px{_c_glow};white-space:nowrap'>"
                f"{_rn}</span>"
            )

        _spy_gap_sep = "  ·  " if _spy_gap_str else ""
        _dv_bg_card  = (_hex_rgba("#ef5350", 0.12) if _dv_lbl == "BEARISH DIVERGENCE" else
                        _hex_rgba("#26a69a", 0.10) if _dv_lbl == "BULLISH DIVERGENCE" else
                        "rgba(255,255,255,.03)")
        st.markdown(
            f"<div style='background:{_reg_bg};border-radius:14px;padding:16px 22px;"
            f"border-left:4px solid {_reg_col};margin-bottom:12px'>"
            f"<div style='display:flex;justify-content:space-between;"
            f"align-items:flex-start;gap:16px;flex-wrap:wrap'>"
            f"<div>"
            f"<div style='font-size:.58em;letter-spacing:2.2px;"
            f"color:rgba(255,255,255,.28);margin-bottom:4px'>MARKET REGIME</div>"
            f"<div style='font-size:1.30em;font-weight:900;color:{_reg_col};"
            f"letter-spacing:1.8px'>{_reg_lbl}</div>"
            f"<div style='font-size:.76em;color:rgba(255,255,255,.48);margin-top:5px'>"
            f"{_reg_desc}</div>"
            f"<div style='font-size:.64em;color:rgba(255,255,255,.24);margin-top:5px'>"
            f"{_spy_gap_str}{_spy_gap_sep}Breadth {eq_score:.0f}"
            f"</div></div>"
            f"<div style='background:{_dv_bg_card};border:1px solid {_dv_col};"
            f"border-radius:10px;padding:10px 16px;min-width:190px;flex-shrink:0'>"
            f"<div style='font-size:.56em;letter-spacing:2px;"
            f"color:rgba(255,255,255,.25);margin-bottom:3px'>PRICE / BREADTH (20d)</div>"
            f"<div style='font-size:.92em;font-weight:800;color:{_dv_col}'>"
            f"{_dv_icon} {_dv_lbl}</div>"
            f"<div style='font-size:.70em;color:rgba(255,255,255,.38);margin-top:4px'>"
            f"{_dv_desc}</div>"
            f"</div></div>"
            f"<div style='display:flex;gap:6px;margin-top:12px;flex-wrap:wrap'>"
            f"{_regime_chips}"
            f"</div></div>",
            unsafe_allow_html=True,
        )

        # ── Gauge + Metrics ───────────────────────────────────────────────
        col_g, col_m = st.columns([3, 2], gap="medium")

        with col_g:
            st.plotly_chart(_build_breadth_gauge(eq_score, gauge_color, z_lbl),
                            use_container_width=True)

        with col_m:
            # Status card
            st.markdown(
                f"<div style='background:#12121e;border-radius:12px;padding:16px 18px;"
                f"margin-bottom:10px;border-left:4px solid {gauge_color};"
                f"box-shadow:-4px 0 20px {gauge_color}44'>"
                f"<div style='font-size:.65em;letter-spacing:2px;"
                f"color:rgba(255,255,255,.35);margin-bottom:5px'>BREADTH STATUS</div>"
                f"<div style='color:{z_lcol};font-size:1.4em;font-weight:900;"
                f"letter-spacing:2px'>{z_lbl}</div>"
                f"<div style='font-size:.78em;color:rgba(255,255,255,.40);margin-top:4px'>"
                f"{_eq_breadth['n_stocks']} S&P 500 components</div></div>",
                unsafe_allow_html=True,
            )

            # Breadth momentum (Δ vs 10 trading days ago)
            _bdelta = _eq_breadth.get("breadth_delta")
            if _bdelta is not None:
                _bd_col   = "#26a69a" if _bdelta >= 0 else "#ef5350"
                _bd_arrow = "↑" if _bdelta >= 1 else ("↓" if _bdelta <= -1 else "→")
                _bd_str   = f"{_bd_arrow} {_bdelta:+.1f}pp  (10d)"
            else:
                _bd_col = "rgba(255,255,255,.35)"; _bd_str = "—"

            st.markdown(
                f"<div style='background:#12121e;border-radius:12px;padding:12px 16px;"
                f"margin-bottom:10px;border:1px solid rgba(255,255,255,.06)'>"
                f"<div style='font-size:.62em;letter-spacing:1.5px;"
                f"color:rgba(255,255,255,.30);margin-bottom:3px'>BREADTH MOMENTUM</div>"
                f"<div style='font-size:1.2em;font-weight:700;color:{_bd_col}'>{_bd_str}</div>"
                f"<div style='font-size:.70em;color:rgba(255,255,255,.30);margin-top:2px'>"
                f"Change in % stocks above MA50 over last 2 weeks</div></div>",
                unsafe_allow_html=True,
            )

            # Metrics grid
            _nh_nl = _eq_breadth.get("nh_nl_ratio", 0)
            _nh_nl_col = ("#26a69a" if _nh_nl >= 55 else
                          "#ef5350"  if _nh_nl <= 35 else
                          "rgba(255,255,255,.85)")
            _metrics = [
                ("% Above MA20",   f"{_eq_breadth['pct_ma20']:.1f}%"),
                ("% Above MA50",   f"{_eq_breadth['pct_ma50']:.1f}%"),
                ("% Above MA200",  f"{_eq_breadth['pct_ma200']:.1f}%"),
                ("% Positive 5d",  f"{_eq_breadth['pct_pos5d']:.1f}%"),
                ("% Positive 20d", f"{_eq_breadth['pct_pos20d']:.1f}%"),
                ("52W Highs",      f"{_eq_breadth['pct_52h']:.1f}%"),
                ("52W Lows",       f"{_eq_breadth['pct_52l']:.1f}%"),
                ("NH/(NH+NL)",     f"{_nh_nl:.1f}%"),
            ]
            rows_html = "".join([
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04)'>"
                f"<span style='font-size:.80em;color:rgba(255,255,255,.45)'>{k}</span>"
                f"<span style='font-size:.80em;font-weight:700;color:rgba(255,255,255,.85)'>{v}</span>"
                f"</div>"
                for k, v in _metrics
            ])
            st.markdown(
                f"<div style='background:#12121e;border-radius:12px;padding:14px 16px;"
                f"border:1px solid rgba(255,255,255,.06)'>{rows_html}</div>",
                unsafe_allow_html=True,
            )

        # ── Score Breakdown ───────────────────────────────────────────────
        _divider()
        _section_header("Score Breakdown",
                        "Weighted sub-components of the composite breadth score")
        _sb_components = sorted([
            ("% Above MA20",  _eq_breadth["pct_ma20"],         18),
            ("% Above MA50",  _eq_breadth["pct_ma50"],         27),
            ("% Above MA200", _eq_breadth["pct_ma200"],        22),
            ("% Pos 20d",     _eq_breadth["pct_pos20d"],       13),
            ("NH / (NH+NL)",  _eq_breadth["nh_nl_ratio"],      12),
            ("52W Safety",    100.0 - _eq_breadth["pct_52l"],   8),
        ], key=lambda x: x[1])
        def _sb_color(v: float) -> str:
            if v >= 75: return "#26a69a"
            if v >= 60: return "#66bb6a"
            if v >= 45: return "#c8b400"
            if v >= 30: return "#ff7043"
            return "#ef5350"

        _sb_rows = ""
        _sb_total = 0.0
        for _sb_lbl, _sb_val, _sb_wt in _sb_components:
            _sbc     = _sb_color(_sb_val)
            _contrib = _sb_val * _sb_wt / 100.0
            _sb_total += _contrib
            _sb_rows += (
                f"<div style='display:flex;align-items:center;gap:10px;"
                f"padding:7px 0;border-bottom:1px solid rgba(255,255,255,.03)'>"
                f"<div style='width:110px;font-size:.76em;color:rgba(255,255,255,.55);"
                f"flex-shrink:0'>{_sb_lbl}</div>"
                f"<div style='flex:1;position:relative;background:rgba(239,83,80,.10);"
                f"border-radius:3px;height:8px;overflow:hidden'>"
                f"<div style='position:absolute;left:0;top:0;height:100%;width:{_sb_val:.1f}%;"
                f"border-radius:3px;"
                f"background:linear-gradient(90deg,rgba(239,83,80,.45),{_sbc})'></div>"
                f"</div>"
                f"<div style='width:38px;text-align:right;font-size:.82em;"
                f"font-weight:700;color:{_sbc};flex-shrink:0'>{_sb_val:.0f}%</div>"
                f"<div style='width:64px;text-align:right;font-size:.72em;flex-shrink:0'>"
                f"<span style='color:{_sbc};opacity:.75'>+{_contrib:.1f}pts</span>"
                f"<span style='color:rgba(255,255,255,.18)'> ×{_sb_wt}</span>"
                f"</div>"
                f"</div>"
            )
        _tot_col = _sb_color(_sb_total)
        _sb_rows += (
            f"<div style='display:flex;justify-content:flex-end;align-items:center;"
            f"padding:8px 0 2px;gap:8px;border-top:1px solid rgba(255,255,255,.07)'>"
            f"<span style='font-size:.64em;letter-spacing:1.8px;"
            f"color:rgba(255,255,255,.28)'>COMPOSITE SCORE</span>"
            f"<span style='font-size:1.12em;font-weight:900;color:{_tot_col}'>"
            f"{_sb_total:.1f}</span>"
            f"</div>"
        )
        st.markdown(
            f"<div style='background:#12121e;border-radius:10px;padding:12px 16px;"
            f"border:1px solid rgba(255,255,255,.06)'>{_sb_rows}</div>",
            unsafe_allow_html=True,
        )

        # ── Sector Heatmap ────────────────────────────────────────────────
        _divider()
        _has_real_breadth = any(v is not None for v in _sector_breadth.values())
        _section_header("Sector Breadth Heatmap",
                        "Real breadth = % of constituent stocks above MA50  ·  "
                        "ETF performance vs SPY" if _has_real_breadth else
                        "Each sector ETF vs MA20 / MA50 / SPY relative performance")

        if not _sector_prices.empty:
            _spy = _sector_prices.get("SPY", pd.Series(dtype=float))
            _spy_ret20 = float(_spy.pct_change(20).iloc[-1] * 100) if len(_spy) > 20 else 0.0

            _sorted_sectors = sorted(
                _SECTORS.items(),
                key=lambda x: (_sector_breadth.get(x[0]) or -1),
                reverse=True,
            )
            cols = st.columns(len(_sorted_sectors))
            for i, (tkr, meta) in enumerate(_sorted_sectors):
                if tkr not in _sector_prices.columns:
                    continue
                s = _sector_prices[tkr].dropna()
                if len(s) < 50:
                    continue

                cur    = float(s.iloc[-1])
                ma20v  = float(s.rolling(20).mean().iloc[-1])
                ma50v  = float(s.rolling(50).mean().iloc[-1])
                ret5   = float((s.iloc[-1] / s.iloc[-6] - 1) * 100) if len(s) > 5  else 0.0
                ret20  = float((s.iloc[-1] / s.iloc[-21] - 1) * 100) if len(s) > 20 else 0.0
                vs_spy = ret20 - _spy_ret20

                above_ma20  = cur > ma20v
                above_ma50  = cur > ma50v

                # Real sector breadth from constituent stocks
                real_brd  = _sector_breadth.get(tkr)
                brd_str   = f"{real_brd:.0f}%" if real_brd is not None else "—"
                brd_col   = ("#26a69a" if (real_brd or 0) >= 60 else
                             "#ef5350" if (real_brd or 0) <= 35 else
                             "rgba(185,185,185,.80)")

                # Card colour driven by real breadth when available, else by ETF MAs
                if real_brd is not None:
                    c_bg     = ("#0d2b1a" if real_brd >= 60 else
                                "#1a1a0d" if real_brd >= 45 else "#2b0d0d")
                    c_border = brd_col
                else:
                    c_bg     = "#0d2b1a" if above_ma50 else ("#1a1a0d" if above_ma20 else "#2b0d0d")
                    c_border = "#26a69a" if above_ma50 else ("#a0a020" if above_ma20 else "#ef5350")

                vs_col   = "#26a69a" if vs_spy >= 0 else "#ef5350"
                ret5_col = "#26a69a" if ret5   >= 0 else "#ef5350"

                cols[i].markdown(
                    f"<div style='background:{c_bg};border-radius:8px;padding:10px 8px;"
                    f"border-left:3px solid {c_border};text-align:center'>"
                    f"<div style='font-size:1.2em'>{meta['icon']}</div>"
                    f"<div style='font-size:.70em;font-weight:600;color:white;"
                    f"margin:2px 0'>{meta['abbr']}</div>"
                    # Real breadth (prominent) or ETF MA status
                    f"<div style='font-size:.88em;font-weight:700;color:{brd_col}'>"
                    f"{brd_str} MA50</div>"
                    f"<div style='font-size:.70em;color:{ret5_col}'>"
                    f"{ret5:+.1f}% 5d</div>"
                    f"<div style='font-size:.65em;color:{vs_col}'>"
                    f"vs SPY: {vs_spy:+.1f}%</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── Historical Breadth Chart ──────────────────────────────────────
        _divider()
        _section_header("Breadth Oscillator — % S&P 500 Above MA50",
                        "MA10 coloured by zone · >70% = Overbought (TE territory) · <30% = Oversold (BT territory) · dots = BT/TE confirmed events")

        if "hist_ma50" in _eq_breadth:
            _hma50 = _eq_breadth["hist_ma50"].tail(252)

            if len(_hma50) > 5:
                # A/D Line prep
                _ad_s    = _eq_breadth.get("ad_line", pd.Series(dtype=float))
                _ad_plot = pd.Series(dtype=float)
                if len(_ad_s) > 5:
                    _ad_plot = _ad_s[_ad_s.index >= _hma50.index[0]].tail(_lookback_d).dropna()
                _has_ad = len(_ad_plot) > 5

                # ── BREADTH OSCILLATOR ────────────────────────────────────
                # MA10 (fast) + MA20 (slow). Fills grow INTO OB/OS extremes.
                # OB >70%: teal fill pushes upward  →  TE territory
                # OS <30%: red fill pushes downward  →  BT territory
                _ma10  = _hma50.rolling(10, min_periods=5).mean()
                _ma20  = _hma50.rolling(20, min_periods=10).mean()
                _ma10c = _ma10.dropna()
                _ma20c = _ma20.dropna()
                _m10   = _ma10c.values
                _m10x  = _ma10c.index

                def _bzone(v):
                    if v >= 70: return "OB"
                    if v >= 55: return "HEALTHY"
                    if v >= 45: return "NEUTRAL"
                    if v >= 30: return "WEAK"
                    return "OS"

                _zlc = {
                    "OS":      "#ef5350",
                    "WEAK":    "#e07820",
                    "NEUTRAL": "#a0a0b8",
                    "HEALTHY": "#66bb6a",
                    "OB":      "#26a69a",
                }

                fig_bh = make_subplots(rows=1, cols=1)
                _r1 = dict(row=1, col=1)

                # Zone bands — static horizontal fills per regime
                for _y0, _y1, _fc in [
                    (70,  102, "rgba(38,166,154,.16)"),
                    (55,  70,  "rgba(102,187,106,.09)"),
                    (45,  55,  "rgba(160,160,180,.04)"),
                    (30,  45,  "rgba(230,130,30,.09)"),
                    (0,   30,  "rgba(210,35,35,.16)"),
                ]:
                    fig_bh.add_hrect(y0=_y0, y1=_y1, line_width=0,
                                     fillcolor=_fc, **_r1)

                # ── 7. EVENT MARKERS — BT / TE vertical lines only ────────
                if not _events_df.empty:
                    for _, ev in _events_df[
                            _events_df["date"] >= _hma50.index[0]].iterrows():
                        _et   = ev["type"] == "BOTTOM_THRUST"
                        _elc  = "#26a69a" if _et else "#ef5350"
                        _etxt = "BT"      if _et else "TE"
                        _ed   = pd.Timestamp(ev["date"])
                        fig_bh.add_vline(
                            x=_ed.value,
                            line_color=_elc,
                            line_width=1.2,
                            line_dash="dot",
                            **_r1,
                        )
                        fig_bh.add_annotation(
                            x=_ed, yref="y", y=97,
                            text=f"<b>{_etxt}</b>",
                            font=dict(size=8, color=_elc, family="monospace"),
                            showarrow=False, xanchor="center", yanchor="top",
                            bgcolor="rgba(6,6,14,.88)", borderpad=2,
                        )

                # ── 8. REFERENCE LINES ────────────────────────────────────
                for _yv, _ytxt, _yc, _ld, _lw in [
                    (70,   "OVERBOUGHT  70%", "rgba(38,166,154,.90)",  "solid", 1.6),
                    (61.5, "THRUST  61.5%",   "rgba(38,166,154,.40)",  "dash",  1.0),
                    (50,   "",                "rgba(255,255,255,.16)", "dot",   1.0),
                    (40.0, "SETUP   40.0%",   "rgba(210, 35, 35,.40)", "dash",  1.0),
                    (30,   "OVERSOLD    30%", "rgba(210, 35, 35,.90)", "solid", 1.6),
                ]:
                    fig_bh.add_hline(y=_yv, line_dash=_ld,
                                     line_color=_yc, line_width=_lw, **_r1)
                    if _ytxt:
                        fig_bh.add_annotation(
                            xref="paper", yref="y", x=1.0, y=_yv,
                            text=f"  {_ytxt}",
                            font=dict(size=8, color=_yc, family="monospace"),
                            showarrow=False, xanchor="left", yanchor="middle",
                            bgcolor="rgba(6,6,14,.86)", borderpad=2,
                        )


                # MA10 line — neutral, zone bands carry the color meaning
                fig_bh.add_trace(go.Scatter(
                    x=_m10x, y=_m10,
                    mode="lines", name="MA10",
                    line=dict(color="rgba(210,215,235,.90)", width=2.2),
                    hovertemplate="%{x|%b %d, %Y}  ·  MA10 <b>%{y:.1f}%</b><extra></extra>",
                ), **_r1)

                # ── 12. ENDPOINT DOT + VALUE LABEL ────────────────────────
                _cur_ma  = float(_ma10c.iloc[-1])
                _cur_col = _zlc[_bzone(_cur_ma)]
                fig_bh.add_trace(go.Scatter(
                    x=[_m10x[-1]], y=[_cur_ma], mode="markers",
                    marker=dict(size=12, color=_cur_col,
                                line=dict(color="rgba(0,0,0,.55)", width=2)),
                    showlegend=False, hoverinfo="skip",
                ), **_r1)
                fig_bh.add_annotation(
                    x=_m10x[-1], yref="y", y=_cur_ma,
                    text=f"  {_cur_ma:.1f}%",
                    font=dict(size=11, color=_cur_col, family="monospace"),
                    showarrow=False, xanchor="left", yanchor="middle",
                    bgcolor="rgba(6,6,14,.92)", borderpad=3,
                )

                # ── LAYOUT ────────────────────────────────────────────────
                fig_bh.update_layout(
                    template="plotly_dark",
                    height=620,
                    margin=dict(l=52, r=138, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(
                        range=[-2, 102],
                        showgrid=True,
                        gridcolor="rgba(255,255,255,.04)",
                        tickvals=[0, 30, 50, 70, 100],
                        ticktext=["0%", "30%", "50%", "70%", "100%"],
                        tickfont=dict(size=9, color="rgba(255,255,255,.36)"),
                        zeroline=False,
                    ),
                    legend=dict(
                        orientation="h", y=1.05, x=0,
                        font=dict(size=11, color="rgba(255,255,255,.68)"),
                        bgcolor="rgba(0,0,0,0)",
                    ),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_bh, use_container_width=True)

            # Thrust history table
            if not _events_df.empty:
                _divider()
                _section_header("Thrust & Exhaustion Events",
                                "All detected events in the current data window")
                _ev_display = _events_df.sort_values("date", ascending=False).head(10)
                _spy_ev = (_sector_prices["SPY"]
                           if "SPY" in _sector_prices.columns
                           else pd.Series(dtype=float))

                def _spy_chip(pct, label):
                    if pct is None:
                        return (f"<div style='text-align:center;min-width:56px'>"
                                f"<div style='font-size:.60em;color:rgba(255,255,255,.25);"
                                f"margin-bottom:3px'>{label}</div>"
                                f"<div style='font-size:.78em;color:rgba(255,255,255,.20)'>—</div>"
                                f"</div>")
                    _c  = "#26a69a" if pct >= 0 else "#ef5350"
                    _bg = _hex_rgba("#26a69a", 0.16) if pct >= 0 else _hex_rgba("#ef5350", 0.14)
                    _bd = _hex_rgba("#26a69a", 0.35) if pct >= 0 else _hex_rgba("#ef5350", 0.35)
                    return (f"<div style='text-align:center;min-width:56px'>"
                            f"<div style='font-size:.60em;color:rgba(255,255,255,.28);"
                            f"margin-bottom:4px'>{label}</div>"
                            f"<div style='background:{_bg};border:1px solid {_bd};"
                            f"border-radius:6px;padding:3px 8px;font-size:.86em;"
                            f"font-weight:700;color:{_c}'>{pct:+.1f}%</div>"
                            f"</div>")

                rows_ev = ""
                for _, r in _ev_display.iterrows():
                    _is_bt    = r["type"] == "BOTTOM_THRUST"
                    _ev_col   = "#26a69a"   if _is_bt else "#ef5350"
                    _ev_bg    = _hex_rgba("#26a69a", 0.14) if _is_bt else _hex_rgba("#ef5350", 0.12)
                    _ev_brd   = _hex_rgba("#26a69a", 0.45) if _is_bt else _hex_rgba("#ef5350", 0.45)
                    _ev_ico   = "🚀" if _is_bt else "⚠️"
                    _ev_lbl   = "BOTTOM THRUST" if _is_bt else "TOP EXHAUST"
                    _ev_date  = pd.Timestamp(r["date"])
                    _days_ago = (pd.Timestamp.now() - _ev_date).days
                    _since    = (f"{_days_ago}d ago"        if _days_ago < 30  else
                                 f"{_days_ago // 30}mo ago" if _days_ago < 365 else
                                 f"{_days_ago // 365}y ago")
                    _fwd = _event_spy_returns(_ev_date, _spy_ev)
                    rows_ev += (
                        f"<div style='display:flex;align-items:center;gap:14px;"
                        f"padding:12px 18px;border-left:3px solid {_ev_col};"
                        f"border-bottom:1px solid rgba(255,255,255,.04)'>"
                        # Date column
                        f"<div style='flex-shrink:0;min-width:96px'>"
                        f"<div style='font-size:.88em;font-weight:600;"
                        f"color:rgba(255,255,255,.88)'>{_ev_date.strftime('%Y-%m-%d')}</div>"
                        f"<div style='font-size:.66em;color:rgba(255,255,255,.28);"
                        f"margin-top:2px'>{_since}</div>"
                        f"</div>"
                        # Event badge
                        f"<div style='flex-shrink:0'>"
                        f"<div style='background:{_ev_bg};border:1px solid {_ev_brd};"
                        f"border-radius:8px;padding:5px 13px;display:inline-flex;"
                        f"align-items:center;gap:6px'>"
                        f"<span style='font-size:1.0em'>{_ev_ico}</span>"
                        f"<span style='font-size:.78em;font-weight:800;color:{_ev_col};"
                        f"letter-spacing:1px'>{_ev_lbl}</span>"
                        f"</div></div>"
                        # Breadth reading
                        f"<div style='flex-shrink:0;min-width:100px'>"
                        f"<div style='font-size:.60em;color:rgba(255,255,255,.28);"
                        f"letter-spacing:1.5px;margin-bottom:2px'>BREADTH</div>"
                        f"<div style='font-size:.90em;font-weight:700;color:{_ev_col}'>"
                        f"{r['breadth']:.1f}%</div>"
                        f"</div>"
                        # SPY forward return chips
                        f"<div style='flex:1;display:flex;justify-content:flex-end;gap:10px'>"
                        + _spy_chip(_fwd.get("+10d"), "SPY +10d")
                        + _spy_chip(_fwd.get("+20d"), "SPY +20d")
                        + _spy_chip(_fwd.get("+30d"), "SPY +30d")
                        + f"</div></div>"
                    )

                # Table header
                _ev_header = (
                    f"<div style='display:flex;align-items:center;gap:14px;"
                    f"padding:8px 18px;background:rgba(255,255,255,.025);"
                    f"border-bottom:1px solid rgba(255,255,255,.06)'>"
                    f"<div style='flex-shrink:0;min-width:96px;font-size:.58em;"
                    f"letter-spacing:2px;color:rgba(255,255,255,.28)'>DATE</div>"
                    f"<div style='flex-shrink:0;min-width:152px;font-size:.58em;"
                    f"letter-spacing:2px;color:rgba(255,255,255,.28)'>EVENT</div>"
                    f"<div style='flex-shrink:0;min-width:100px;font-size:.58em;"
                    f"letter-spacing:2px;color:rgba(255,255,255,.28)'>BREADTH</div>"
                    f"<div style='flex:1;text-align:right;font-size:.58em;"
                    f"letter-spacing:2px;color:rgba(255,255,255,.28)'>SPY FORWARD RETURNS</div>"
                    f"</div>"
                )
                st.markdown(
                    f"<div style='background:#0c0c18;border-radius:12px;overflow:hidden;"
                    f"border:1px solid rgba(255,255,255,.08)'>"
                    f"{_ev_header}{rows_ev}</div>",
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════
# CRYPTO TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab_cry:
    if not _cry_breadth:
        st.error("Crypto data unavailable. Click 🔄 Refresh Data.")
    else:
        cry_score = _cry_breadth.get("score", 50.0)
        cry_zone  = _zone_for_breadth(cry_score)
        cry_lcol  = cry_zone["lcol"]
        cry_lbl   = cry_zone["label"]
        _cry_colors = {
            "EXTREME WEAK":   "#9b1919",
            "WEAK":           "#a55214",
            "NEUTRAL":        "#505050",
            "HEALTHY":        "#18a050",
            "EXTREME STRONG": "#0f9846",
        }
        cry_gauge_color = _cry_colors.get(cry_lbl, "#f7931a")

        # ── Gauge + Metrics ───────────────────────────────────────────────
        col_cg, col_cm = st.columns([3, 2], gap="medium")

        with col_cg:
            st.plotly_chart(_build_breadth_gauge(cry_score, cry_gauge_color, cry_lbl),
                            use_container_width=True)

        with col_cm:
            n_coins = _cry_breadth.get("n_coins", 0)
            n_hist  = _cry_breadth.get("n_hist_coins", 0)

            st.markdown(
                f"<div style='background:#12121e;border-radius:12px;padding:16px 18px;"
                f"margin-bottom:10px;border-left:4px solid {cry_gauge_color};"
                f"box-shadow:-4px 0 20px {cry_gauge_color}44'>"
                f"<div style='font-size:.65em;letter-spacing:2px;"
                f"color:rgba(255,255,255,.35);margin-bottom:5px'>CRYPTO BREADTH</div>"
                f"<div style='color:{cry_lcol};font-size:1.4em;font-weight:900;"
                f"letter-spacing:2px'>{cry_lbl}</div>"
                f"<div style='font-size:.78em;color:rgba(255,255,255,.40);margin-top:4px'>"
                f"Top {n_coins} coins · {n_hist} with MA history</div></div>",
                unsafe_allow_html=True,
            )

            _cry_metrics = [
                ("% Positive 24h",   f"{_cry_breadth.get('pct_pos24h', 0):.1f}%"),
                ("% Positive 7d",    f"{_cry_breadth.get('pct_pos7d',  0):.1f}%"),
                ("% Positive 30d",   f"{_cry_breadth.get('pct_pos30d', 0):.1f}%"),
                ("Altcoins Pos 7d",  f"{_cry_breadth.get('alt_pct_7d', 0):.1f}%"),
                ("% Above MA20 (30)", f"{_cry_breadth.get('pct_ma20_hist', 0):.1f}%"),
                ("% Above MA50 (30)", f"{_cry_breadth.get('pct_ma50_hist', 0):.1f}%"),
            ]
            rows_cry = "".join([
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04)'>"
                f"<span style='font-size:.80em;color:rgba(255,255,255,.45)'>{k}</span>"
                f"<span style='font-size:.80em;font-weight:700;color:rgba(255,255,255,.85)'>{v}</span>"
                f"</div>"
                for k, v in _cry_metrics
            ])
            st.markdown(
                f"<div style='background:#12121e;border-radius:12px;padding:14px 16px;"
                f"border:1px solid rgba(255,255,255,.06)'>{rows_cry}</div>",
                unsafe_allow_html=True,
            )

        # ── Coin Treemap with timeframe toggle ───────────────────────────
        _divider()
        _n_real = _cry_breadth.get("n_real_coins", "?")

        # Timeframe selector
        _tf_col, _ = st.columns([1, 3])
        _tf_map = {"24h": "price_change_percentage_24h",
                   "7d":  "price_change_percentage_7d_in_currency",
                   "30d": "price_change_percentage_30d_in_currency"}
        _tf_sel = _tf_col.radio("Timeframe", list(_tf_map.keys()),
                                horizontal=True, index=1,
                                label_visibility="collapsed")
        _tf_col_name = _tf_map[_tf_sel]

        _section_header(f"Top Coins — {_tf_sel} Performance",
                        f"Stablecoins & wrapped tokens excluded · {_n_real} real assets · "
                        f"Size = market cap · colour = {_tf_sel} return")

        if "coins" in _cry_breadth and not _cry_breadth["coins"].empty:
            _cdf = _cry_breadth["coins"].dropna(
                subset=[_tf_col_name, "market_cap"])
            # Remove stablecoins from treemap entirely
            _cdf = _cdf[~_cdf["symbol"].isin(_STABLECOINS)]

            _chg7   = _cdf[_tf_col_name].values
            _labels = _cdf["symbol"].str.upper().tolist()
            _mktcap = _cdf["market_cap"].values
            _price  = _cdf["current_price"].values

            fig_tree = go.Figure(go.Treemap(
                labels=_labels,
                parents=[""] * len(_cdf),
                values=_mktcap.tolist(),
                customdata=np.column_stack([_chg7, _price]),
                marker=dict(
                    colors=_chg7.tolist(),
                    colorscale=[
                        [0.00, "rgba(155,25,25,.90)"],
                        [0.30, "rgba(120,40,40,.80)"],
                        [0.50, "rgba(50,50,50,.80)"],
                        [0.70, "rgba(18,100,60,.80)"],
                        [1.00, "rgba(15,152,70,.90)"],
                    ],
                    cmid=0,
                    showscale=True,
                    colorbar=dict(
                        title=dict(
                            text=f"{_tf_sel} %",
                            font=dict(size=10, color="rgba(255,255,255,.55)"),
                        ),
                        thickness=12,
                        tickfont=dict(size=9, color="rgba(255,255,255,.55)"),
                    ),
                ),
                textfont=dict(size=11, color="white"),
                hovertemplate=(
                    f"<b>%{{label}}</b><br>"
                    f"{_tf_sel}: <b>%{{customdata[0]:+.1f}}%</b><br>"
                    f"Price: $%{{customdata[1]:,.4f}}<extra></extra>"
                ),
            ))
            fig_tree.update_layout(
                template="plotly_dark", height=420,
                margin=dict(l=0, r=0, t=6, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_tree, use_container_width=True)

        # ── Historical Crypto Breadth ─────────────────────────────────────
        if "hist_pct_ma50" in _cry_breadth:
            _divider()
            _section_header("Crypto Breadth — % Top-30 Above MA50",
                            f"Based on {n_hist} coins with ≥60% data coverage")

            _cry_hist_ma50 = _cry_breadth["hist_pct_ma50"].tail(_lookback_d)
            _cry_events    = _detect_events(_cry_hist_ma50)

            if len(_cry_hist_ma50) > 5:
                # MA10 smoothing
                _c_ma10  = _cry_hist_ma50.rolling(10, min_periods=5).mean()
                _c_ma10c = _c_ma10.dropna()
                _c_m10   = _c_ma10c.values
                _c_m10x  = _c_ma10c.index

                # 3 colours only: green (>50%), neutral (=50%), red (<50%)
                def _czone(v):
                    if v > 52: return "BULL"
                    if v < 48: return "BEAR"
                    return "NEUTRAL"

                _czlc = {
                    "BULL":    "#66bb6a",
                    "NEUTRAL": "#a0a0a0",
                    "BEAR":    "#ef5350",
                }

                fig_cbh = go.Figure()

                # Zone bands — static horizontal fills per regime
                for _cy0, _cy1, _cfc in [
                    (70,  102, "rgba(38,166,154,.16)"),
                    (55,  70,  "rgba(102,187,106,.09)"),
                    (45,  55,  "rgba(160,160,180,.04)"),
                    (30,  45,  "rgba(230,130,30,.09)"),
                    (0,   30,  "rgba(210,35,35,.16)"),
                ]:
                    fig_cbh.add_hrect(y0=_cy0, y1=_cy1, line_width=0,
                                      fillcolor=_cfc)

                # Event markers — dotted vlines + BT/TE labels
                for _, ev in _cry_events.iterrows():
                    _is_t  = ev["type"] == "BOTTOM_THRUST"
                    _elc   = "#26a69a" if _is_t else "#ef5350"
                    _etxt  = "BT"      if _is_t else "TE"
                    _ed    = pd.Timestamp(ev["date"])
                    fig_cbh.add_vline(x=_ed.value,
                                      line_dash="dot", line_color=_elc, line_width=1.2)
                    fig_cbh.add_annotation(
                        x=_ed, yref="y", y=97,
                        text=f"<b>{_etxt}</b>",
                        font=dict(size=8, color=_elc, family="monospace"),
                        showarrow=False, xanchor="center", yanchor="top",
                        bgcolor="rgba(6,6,14,.88)", borderpad=2,
                    )

                # Reference lines — same scheme as equities
                for _yv, _ytxt, _yc, _ld, _lw in [
                    (70,   "OVERBOUGHT  70%", "rgba(38,166,154,.90)",  "solid", 1.6),
                    (61.5, "61.5%",           "rgba(38,166,154,.40)",  "dash",  1.0),
                    (50,   "",                "rgba(255,255,255,.16)", "dot",   1.0),
                    (40.0, "40.0%",           "rgba(210, 35, 35,.40)", "dash",  1.0),
                    (30,   "OVERSOLD    30%", "rgba(210, 35, 35,.90)", "solid", 1.6),
                ]:
                    fig_cbh.add_hline(y=_yv, line_dash=_ld,
                                      line_color=_yc, line_width=_lw)
                    if _ytxt:
                        fig_cbh.add_annotation(
                            xref="paper", yref="y", x=1.0, y=_yv,
                            text=f"  {_ytxt}",
                            font=dict(size=8, color=_yc, family="monospace"),
                            showarrow=False, xanchor="left", yanchor="middle",
                            bgcolor="rgba(6,6,14,.86)", borderpad=2,
                        )

                # MA10 line — neutral, zone bands carry the color meaning
                fig_cbh.add_trace(go.Scatter(
                    x=_c_m10x, y=_c_m10,
                    mode="lines", name="MA10 Breadth",
                    line=dict(color="rgba(210,215,235,.90)", width=2.2),
                    hovertemplate="%{x|%Y-%m-%d}  ·  MA10 <b>%{y:.1f}%</b><extra></extra>",
                ))

                # Endpoint dot + right-margin label
                _cur_cry     = float(_c_ma10c.iloc[-1])
                _cur_cry_col = _czlc[_czone(_cur_cry)]
                fig_cbh.add_trace(go.Scatter(
                    x=[_c_m10x[-1]], y=[_cur_cry], mode="markers",
                    marker=dict(size=12, color=_cur_cry_col,
                                line=dict(color="rgba(0,0,0,.55)", width=2)),
                    showlegend=False, hoverinfo="skip",
                ))
                fig_cbh.add_annotation(
                    x=_c_m10x[-1], yref="y", y=_cur_cry,
                    text=f"  {_cur_cry:.1f}%",
                    font=dict(size=11, color=_cur_cry_col, family="monospace"),
                    showarrow=False, xanchor="left", yanchor="middle",
                    bgcolor="rgba(6,6,14,.92)", borderpad=3,
                )

                fig_cbh.update_layout(
                    template="plotly_dark", height=440,
                    margin=dict(l=52, r=138, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    yaxis=dict(
                        range=[-2, 102],
                        showgrid=True,
                        gridcolor="rgba(255,255,255,.04)",
                        tickvals=[0, 30, 50, 70, 100],
                        ticktext=["0%", "30%", "50%", "70%", "100%"],
                        tickfont=dict(size=9, color="rgba(255,255,255,.36)"),
                        zeroline=False,
                    ),
                    hovermode="x unified",
                    showlegend=False,
                )
                st.plotly_chart(fig_cbh, use_container_width=True)

            # Crypto events table
            if not _cry_events.empty:
                _divider()
                _section_header("Crypto Thrust & Exhaustion Events",
                                "All detected events in the crypto breadth window")
                _cry_ev_disp = _cry_events.sort_values("date", ascending=False).head(8)
                _cry_rows_ev = ""
                for _, r in _cry_ev_disp.iterrows():
                    _is_bt   = r["type"] == "BOTTOM_THRUST"
                    _ev_bg   = "rgba(38,166,154,.30)" if _is_bt else "rgba(155,25,25,.30)"
                    _ev_col  = "#26a69a"               if _is_bt else "#ef5350"
                    _ev_txt  = "🚀 BOTTOM THRUST"       if _is_bt else "⚠️ TOP EXHAUST"
                    _cry_rows_ev += (
                        f"<tr style='border-bottom:1px solid rgba(255,255,255,.05)'>"
                        f"<td style='padding:8px 14px;font-size:.85em;"
                        f"color:rgba(255,255,255,.75)'>"
                        f"{pd.Timestamp(r['date']).strftime('%Y-%m-%d')}</td>"
                        f"<td style='padding:8px 14px'>"
                        f"<span style='background:{_ev_bg};padding:2px 8px;"
                        f"border-radius:4px;font-size:.75em;font-weight:700;"
                        f"color:{_ev_col}'>{_ev_txt}</span></td>"
                        f"<td style='padding:8px 14px;font-size:.85em;"
                        f"color:rgba(255,255,255,.55)'>"
                        f"{r['breadth']:.1f}% above MA50</td>"
                        f"</tr>"
                    )
                st.markdown(
                    f"<div style='background:#0e0e1a;border-radius:10px;overflow:hidden;"
                    f"border:1px solid rgba(255,255,255,.07)'>"
                    f"<table style='width:100%;border-collapse:collapse;"
                    f"color:rgba(255,255,255,.75)'>"
                    f"<thead><tr style='background:rgba(255,255,255,.04)'>"
                    f"<th style='padding:8px 14px;text-align:left;font-size:.65em;"
                    f"letter-spacing:1.8px;color:rgba(255,255,255,.35)'>DATE</th>"
                    f"<th style='padding:8px 14px;text-align:left;font-size:.65em;"
                    f"letter-spacing:1.8px;color:rgba(255,255,255,.35)'>EVENT</th>"
                    f"<th style='padding:8px 14px;text-align:left;font-size:.65em;"
                    f"letter-spacing:1.8px;color:rgba(255,255,255,.35)'>"
                    f"BREADTH AT EVENT</th>"
                    f"</tr></thead><tbody>{_cry_rows_ev}</tbody></table></div>",
                    unsafe_allow_html=True,
                )

# ─────────────────────────────────────────────────────────────────────────────
# READING GUIDE
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("📖 How to read the Breadth Thrust Detector"):
    st.markdown("""
**What is Market Breadth?**
Breadth measures how many individual stocks/coins are participating in a market move.
A rising index driven by only a few large caps = weak breadth = fragile rally.
A rising index with >70% of stocks above their MA50 = strong breadth = healthy trend.

**Breadth Score (0–100)**

| Score | Status | Interpretation |
|---|---|---|
| 80–100 | 🟢 EXTREME STRONG | Thrust confirmed OR potential exhaustion — watch for slowdown |
| 60–80  | 🟢 HEALTHY | Broad participation — trend is real |
| 40–60  | ⚪ NEUTRAL | Mixed signals |
| 20–40  | 🔴 WEAK | Narrow leadership, risk-off internals |
| 0–20   | 🔴 EXTREME WEAK | Capitulation zone — watch for Thrust |

**🚀 Breadth Thrust (Zweig Method)**
When breadth moves from **<40%** to **>61.5%** within 10 trading days.
Historically, this signal has preceded some of the strongest bull market rallies.
It means the market went from extreme fear to broad buying very quickly — institutions are loading.

**⚠️ Top Exhaustion**
When breadth stays **above 70%** for 15+ consecutive days and then **drops below 60%**.
This doesn't mean crash — it means the broad participation is fading and the market is living on fewer and fewer leaders.

**Data:**
- **Equities**: ~500 S&P 500 real components from Yahoo Finance.
- **Crypto**: Top-100 from CoinGecko (snapshot) + Top-30 historical from Yahoo Finance.
    """)

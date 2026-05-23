"""
Streamlit dashboard — real-time orderflow visualization.
Run: streamlit run dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import time
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st
import yaml

from core.engine import OrderFlowEngine
from core.models import Market
from providers.binance import BinanceProvider
from providers.yahoo import YahooProvider

st.set_page_config(
    page_title="OrderFlow System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("config/settings.yaml") as f:
    CFG = yaml.safe_load(f)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("OrderFlow System")

provider_name = st.sidebar.selectbox("Provider", ["Binance", "Yahoo"])
timeframe = st.sidebar.selectbox(
    "Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
    index=1,
)

if provider_name == "Binance":
    symbols = CFG["watchlist"]["crypto"]
else:
    symbols = CFG["watchlist"]["stocks"] + CFG["watchlist"]["futures"]

symbol = st.sidebar.selectbox("Symbol", symbols)
bar_limit = st.sidebar.slider("Bars", 50, 500, CFG["dashboard"]["bar_limit"])
tick_size = st.sidebar.number_input("Tick Size", value=0.1, min_value=0.0001, format="%.4f")
va_pct = st.sidebar.slider("Value Area %", 0.5, 0.9, 0.70)

auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
if auto_refresh:
    refresh_ms = st.sidebar.slider("Refresh (ms)", 500, 10000, 2000)


# ── Data fetch ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_bars(provider_name, symbol, timeframe, limit):
    if provider_name == "Binance":
        provider = BinanceProvider()
        return provider.fetch_bars_sync(symbol, timeframe, limit)

    provider = YahooProvider()
    loop = asyncio.new_event_loop()
    try:
        bars = loop.run_until_complete(provider.fetch_bars(symbol, timeframe, limit))
    finally:
        loop.close()
    return bars


def build_engine(bars, tick_size, va_pct):
    engine = OrderFlowEngine(tick_size=tick_size)
    engine.profile._value_area_pct = va_pct

    results = []
    for bar in bars:
        result = engine.on_bar(bar)
        results.append(result)

    return engine, results


# ── Main ──────────────────────────────────────────────────────────────────────
bars = fetch_bars(provider_name, symbol, timeframe, bar_limit)

if not bars:
    st.error("No data returned. Check symbol and provider.")
    st.stop()

engine, results = build_engine(bars, tick_size, va_pct)

timestamps = [datetime.fromtimestamp(b.timestamp / 1000) for b in bars]
closes = [b.close for b in bars]
volumes = [b.volume for b in bars]
buy_vols = [b.buy_volume for b in bars]
sell_vols = [b.sell_volume for b in bars]
deltas = [b.delta for b in bars]
cvds = [r["cvd"] for r in results]

poc = engine.profile.poc
va_lo, va_hi = engine.profile.value_area(va_pct)
profile_data = engine.profile.to_dict()

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Last Price", f"{closes[-1]:,.2f}")
col2.metric("POC", f"{poc:,.2f}" if poc else "—")
col3.metric("Value Area", f"{va_lo:,.2f} – {va_hi:,.2f}" if va_lo else "—")
col4.metric("CVD", f"{cvds[-1]:+,.2f}" if cvds else "—")

st.markdown("---")

# ── Price + CVD chart ─────────────────────────────────────────────────────────
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=timestamps,
    open=[b.open for b in bars],
    high=[b.high for b in bars],
    low=[b.low for b in bars],
    close=closes,
    name="Price",
    increasing_line_color="#26a69a",
    decreasing_line_color="#ef5350",
))

if poc:
    fig.add_hline(y=poc, line_dash="dash", line_color="yellow",
                  annotation_text="POC", annotation_position="left")
if va_lo and va_hi:
    fig.add_hrect(y0=va_lo, y1=va_hi, fillcolor="rgba(100,100,255,0.08)",
                  line_width=0, annotation_text="VA 70%")

fig.update_layout(
    title=f"{symbol} — {timeframe}",
    xaxis_rangeslider_visible=False,
    template="plotly_dark",
    height=420,
    margin=dict(l=0, r=0, t=40, b=0),
)
st.plotly_chart(fig, use_container_width=True)

# ── CVD chart ─────────────────────────────────────────────────────────────────
fig_cvd = go.Figure()
cvd_colors = ["#26a69a" if c >= 0 else "#ef5350" for c in cvds]
fig_cvd.add_trace(go.Bar(x=timestamps, y=cvds, marker_color=cvd_colors, name="CVD"))
fig_cvd.update_layout(
    title="Cumulative Volume Delta (CVD)",
    template="plotly_dark",
    height=200,
    margin=dict(l=0, r=0, t=40, b=0),
)
st.plotly_chart(fig_cvd, use_container_width=True)

# ── Volume Delta bar ──────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=timestamps, y=buy_vols, name="Buy Vol", marker_color="#26a69a"))
    fig_vol.add_trace(go.Bar(x=timestamps, y=[-v for v in sell_vols], name="Sell Vol", marker_color="#ef5350"))
    fig_vol.update_layout(
        barmode="relative",
        title="Buy / Sell Volume",
        template="plotly_dark",
        height=250,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

with col_b:
    if profile_data:
        prices_vp = sorted(profile_data.keys())
        total_vp = [profile_data[p]["total"] for p in prices_vp]
        delta_vp = [profile_data[p]["delta"] for p in prices_vp]
        colors_vp = ["#26a69a" if d >= 0 else "#ef5350" for d in delta_vp]

        fig_vp = go.Figure()
        fig_vp.add_trace(go.Bar(
            x=total_vp, y=prices_vp,
            orientation="h",
            marker_color=colors_vp,
            name="Volume Profile",
        ))
        if poc:
            fig_vp.add_hline(y=poc, line_dash="dash", line_color="yellow")
        fig_vp.update_layout(
            title="Volume Profile",
            template="plotly_dark",
            height=250,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_vp, use_container_width=True)

if auto_refresh:
    time.sleep(refresh_ms / 1000)
    st.rerun()

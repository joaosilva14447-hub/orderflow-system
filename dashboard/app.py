"""
OrderFlow System — Streamlit Dashboard
Reliable on any server: yfinance (cloud/local) + Bybit/Binance (local only).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import plotly.graph_objects as go
import streamlit as st
import yaml

from core.engine import OrderFlowEngine
from core.footprint import build_footprints
from dashboard.footprint_chart import render_footprint, render_footprint_summary
from providers.yfinance_provider import YFinanceProvider

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "OrderFlow System",
    page_icon   = "📊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.yaml")) as f:
    CFG = yaml.safe_load(f)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("OrderFlow System")

MARKETS = {
    "Crypto":    CFG["watchlist"]["crypto"],
    "Stocks":    CFG["watchlist"]["stocks"],
    "Futures":   CFG["watchlist"]["futures"],
}

market_choice = st.sidebar.selectbox("Market", list(MARKETS.keys()))
symbol        = st.sidebar.selectbox("Symbol", MARKETS[market_choice])
timeframe     = st.sidebar.selectbox(
    "Timeframe", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=1
)
bar_limit = st.sidebar.slider("Bars", 50, 500, 200)
tick_size = st.sidebar.number_input("Tick Size", value=10.0, min_value=0.0001, format="%.4f")
va_pct    = st.sidebar.slider("Value Area %", 0.5, 0.9, 0.70)

st.sidebar.markdown("---")
st.sidebar.markdown("**Footprint Settings**")
fp_bars      = st.sidebar.slider("Footprint Bars", 5, 50, 20)
fp_imb_thresh= st.sidebar.slider("Imbalance Ratio", 1.5, 5.0, 3.0, step=0.5)
fp_show_nums = st.sidebar.checkbox("Show Volume Numbers", value=True)

auto_ref  = st.sidebar.checkbox("Auto Refresh (60s)", value=False)

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner="Loading market data…")
def load(symbol: str, timeframe: str, limit: int) -> list:
    return YFinanceProvider().fetch_bars_sync(symbol, timeframe, limit)

try:
    bars = load(symbol, timeframe, bar_limit)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

if not bars:
    st.warning("No data returned. Try a different symbol or timeframe.")
    st.stop()

# ── Engine ────────────────────────────────────────────────────────────────────
engine = OrderFlowEngine(tick_size=tick_size)
results = [engine.on_bar(b) for b in bars]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_main, tab_fp = st.tabs(["📊 OrderFlow", "🔬 Footprint"])

timestamps  = [datetime.fromtimestamp(b.timestamp / 1000) for b in bars]
closes      = [b.close      for b in bars]
buy_vols    = [b.buy_volume  for b in bars]
sell_vols   = [b.sell_volume for b in bars]
deltas      = [b.delta       for b in bars]
cvds        = [r["cvd"]      for r in results]

poc         = engine.profile.poc
va_lo, va_hi= engine.profile.value_area(va_pct)

# ── Tab: OrderFlow ────────────────────────────────────────────────────────────
with tab_main:
 c1, c2, c3, c4, c5 = st.columns(5)
 c1.metric("Last Price",  f"{closes[-1]:,.2f}")
 c2.metric("POC",         f"{poc:,.2f}"          if poc   else "—")
 c3.metric("VA High",     f"{va_hi:,.2f}"         if va_hi else "—")
 c4.metric("VA Low",      f"{va_lo:,.2f}"         if va_lo else "—")
 c5.metric("CVD",         f"{cvds[-1]:+,.2f}"     if cvds  else "—")

 st.markdown("---")

 # Candlestick + POC/VA
 fig = go.Figure()
 fig.add_trace(go.Candlestick(
     x=timestamps, open=[b.open for b in bars], high=[b.high for b in bars],
     low=[b.low for b in bars], close=closes, name="Price",
     increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
 ))
 if poc:
     fig.add_hline(y=poc, line_dash="dash", line_color="yellow",
                   annotation_text="POC", annotation_position="left")
 if va_lo and va_hi:
     fig.add_hrect(y0=va_lo, y1=va_hi, fillcolor="rgba(100,100,255,0.07)",
                   line_width=0, annotation_text=f"VA {int(va_pct*100)}%")
 fig.update_layout(title=f"{symbol}  ·  {timeframe}", xaxis_rangeslider_visible=False,
                   template="plotly_dark", height=430, margin=dict(l=0,r=0,t=40,b=0))
 st.plotly_chart(fig, use_container_width=True)

 # CVD
 fig_cvd = go.Figure()
 fig_cvd.add_trace(go.Bar(x=timestamps, y=cvds,
     marker_color=["#26a69a" if c >= 0 else "#ef5350" for c in cvds], name="CVD"))
 fig_cvd.update_layout(title="Cumulative Volume Delta (CVD)", template="plotly_dark",
                        height=200, margin=dict(l=0,r=0,t=40,b=0))
 st.plotly_chart(fig_cvd, use_container_width=True)

 # Buy/Sell + Volume Profile
 col_a, col_b = st.columns(2)
 with col_a:
     fig_vol = go.Figure()
     fig_vol.add_trace(go.Bar(x=timestamps, y=buy_vols,  name="Buy Vol",  marker_color="#26a69a"))
     fig_vol.add_trace(go.Bar(x=timestamps, y=[-v for v in sell_vols], name="Sell Vol", marker_color="#ef5350"))
     fig_vol.update_layout(barmode="relative", title="Buy / Sell Volume",
                            template="plotly_dark", height=260, margin=dict(l=0,r=0,t=40,b=0))
     st.plotly_chart(fig_vol, use_container_width=True)
 with col_b:
     profile = engine.profile.to_dict()
     if profile:
         prices_vp = sorted(profile.keys())
         fig_vp = go.Figure()
         fig_vp.add_trace(go.Bar(
             x=[profile[p]["total"] for p in prices_vp], y=prices_vp, orientation="h",
             marker_color=["#26a69a" if profile[p]["delta"] >= 0 else "#ef5350" for p in prices_vp],
             name="Volume Profile",
         ))
         if poc:   fig_vp.add_hline(y=poc, line_dash="dash", line_color="yellow")
         if va_lo: fig_vp.add_hrect(y0=va_lo, y1=va_hi, fillcolor="rgba(100,100,255,0.07)", line_width=0)
         fig_vp.update_layout(title="Volume Profile", template="plotly_dark",
                               height=260, margin=dict(l=0,r=0,t=40,b=0))
         st.plotly_chart(fig_vp, use_container_width=True)

 # Bar Delta
 fig_delta = go.Figure()
 fig_delta.add_trace(go.Bar(x=timestamps, y=deltas,
     marker_color=["#26a69a" if d >= 0 else "#ef5350" for d in deltas], name="Delta"))
 fig_delta.update_layout(title="Bar Delta (Buy Vol − Sell Vol)", template="plotly_dark",
                          height=180, margin=dict(l=0,r=0,t=40,b=0))
 st.plotly_chart(fig_delta, use_container_width=True)

# ── Tab: Footprint ────────────────────────────────────────────────────────────
with tab_fp:
    st.markdown(f"### Footprint Chart · {symbol} · {timeframe}")
    st.caption(
        "⚠️ Buy/Sell split approximated from OHLCV (Yahoo Finance has no tick data). "
        "For real footprint, connect Bybit locally."
    )

    with st.spinner("Building footprint…"):
        footprints = build_footprints(bars, tick_size)

    if footprints:
        # KPIs from last footprint bar
        last_fp = footprints[-1]
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Last Bar Delta",  f"{last_fp.delta:+,.0f}")
        fc2.metric("Last Bar Buy",    f"{last_fp.cum_buy:,.0f}")
        fc3.metric("Last Bar Sell",   f"{last_fp.cum_sell:,.0f}")
        fc4.metric("Bar POC",         f"{last_fp.poc:,.2f}" if last_fp.poc else "—")

        st.markdown("---")

        fig_fp = render_footprint(
            footprints,
            imb_thresh = fp_imb_thresh,
            show_nums  = fp_show_nums,
            max_bars   = fp_bars,
        )
        st.plotly_chart(fig_fp, use_container_width=True)

        st.markdown("---")
        fig_summary = render_footprint_summary(footprints)
        st.plotly_chart(fig_summary, use_container_width=True)

if auto_ref:
    import time; time.sleep(60); st.rerun()

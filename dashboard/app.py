"""
OrderFlow System — Streamlit Dashboard v2
Tier 2: VWAP+SD, Session POC, Multi-TF CVD, Heatmap
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

from core.engine    import OrderFlowEngine
from core.vwap      import calculate_vwap, calculate_vwap_bands
from core.sessions  import calculate_sessions, get_current_sessions, SESSION_LINE_COLORS, SESSION_COLORS
from core.footprint import build_footprints
from core.heatmap   import heatmap_from_bars
from core.mtf       import calculate_mtf_cvd, confluence_score
from dashboard.footprint_chart import render_footprint, render_footprint_summary
from providers.yfinance_provider import YFinanceProvider

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="OrderFlow System", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")

cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.yaml")
with open(cfg_path) as f:
    CFG = yaml.safe_load(f)

MARKETS = {
    "Crypto":  CFG["watchlist"]["crypto"],
    "Stocks":  CFG["watchlist"]["stocks"],
    "Futures": CFG["watchlist"]["futures"],
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("OrderFlow System")

market_choice = st.sidebar.selectbox("Market", list(MARKETS.keys()))
symbol        = st.sidebar.selectbox("Symbol", MARKETS[market_choice])
timeframe     = st.sidebar.selectbox("Timeframe",
    ["1m","5m","15m","30m","1h","4h","1d"], index=2)
bar_limit     = st.sidebar.slider("Bars", 50, 500, 200)
tick_size     = st.sidebar.number_input("Tick Size", value=10.0, min_value=0.0001, format="%.4f")
va_pct        = st.sidebar.slider("Value Area %", 0.5, 0.9, 0.70)

st.sidebar.markdown("---")
st.sidebar.markdown("**VWAP**")
show_vwap  = st.sidebar.checkbox("Show VWAP", value=True)
show_bands = st.sidebar.checkbox("Show SD Bands", value=True)
n_bands    = st.sidebar.slider("SD Bands", 1, 3, 2)

st.sidebar.markdown("---")
st.sidebar.markdown("**Sessions**")
show_sessions = st.sidebar.checkbox("Show Session POC/VA", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("**Footprint**")
fp_bars       = st.sidebar.slider("Footprint Bars", 5, 40, 15)
fp_imb_thresh = st.sidebar.slider("Imbalance Ratio", 1.5, 5.0, 3.0, step=0.5)
fp_show_nums  = st.sidebar.checkbox("Show Volume Numbers", value=True)

auto_ref = st.sidebar.checkbox("Auto Refresh (60s)", value=False)

# ── Data fetch ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner="Loading market data…")
def load(symbol, timeframe, limit):
    return YFinanceProvider().fetch_bars_sync(symbol, timeframe, limit)

try:
    bars = load(symbol, timeframe, bar_limit)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

if not bars:
    st.warning("No data returned.")
    st.stop()

# ── Compute all layers ────────────────────────────────────────────────────────
engine  = OrderFlowEngine(tick_size=tick_size)
results = [engine.on_bar(b) for b in bars]

timestamps = [datetime.fromtimestamp(b.timestamp / 1000) for b in bars]
closes     = [b.close for b in bars]
cvds       = [r["cvd"] for r in results]
deltas     = [b.delta  for b in bars]
buy_vols   = [b.buy_volume  for b in bars]
sell_vols  = [b.sell_volume for b in bars]

poc       = engine.profile.poc
va_lo, va_hi = engine.profile.value_area(va_pct)

vwaps     = calculate_vwap(bars)
vwap_bands= calculate_vwap_bands(bars, vwaps, n_bands) if show_bands else {}
sessions  = calculate_sessions(bars, tick_size)
cur_sess  = get_current_sessions(sessions)
heatmap   = heatmap_from_bars(bars, tick_size)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_main, tab_fp, tab_heatmap, tab_mtf = st.tabs([
    "📊 OrderFlow", "🔬 Footprint", "🌡️ Heatmap", "📐 Multi-TF CVD"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OrderFlow
# ════════════════════════════════════════════════════════════════════════════════
with tab_main:

    # KPIs
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Last Price", f"{closes[-1]:,.2f}")
    c2.metric("POC",        f"{poc:,.2f}"       if poc   else "—")
    c3.metric("VA High",    f"{va_hi:,.2f}"      if va_hi else "—")
    c4.metric("VA Low",     f"{va_lo:,.2f}"      if va_lo else "—")
    c5.metric("CVD",        f"{cvds[-1]:+,.0f}"  if cvds  else "—")
    c6.metric("VWAP",       f"{vwaps[-1]:,.2f}"  if vwaps else "—")

    st.markdown("---")

    # ── Candlestick + VWAP + Sessions ────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=timestamps, open=[b.open for b in bars],
        high=[b.high for b in bars], low=[b.low for b in bars],
        close=closes, name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))

    # Volume Profile levels
    if poc:
        fig.add_hline(y=poc, line_dash="dash", line_color="yellow",
                      annotation_text="POC", annotation_position="left")
    if va_lo and va_hi:
        fig.add_hrect(y0=va_lo, y1=va_hi, fillcolor="rgba(100,100,255,0.07)",
                      line_width=0, annotation_text=f"VA {int(va_pct*100)}%")

    # VWAP
    if show_vwap and vwaps:
        fig.add_trace(go.Scatter(
            x=timestamps, y=vwaps, name="VWAP",
            line=dict(color="rgba(255,255,255,0.9)", width=1.5, dash="solid"),
        ))

    # SD Bands
    if show_bands and vwap_bands:
        band_colors = {1: "rgba(100,200,255,0.6)", 2: "rgba(100,200,255,0.4)", 3: "rgba(100,200,255,0.25)"}
        for i in range(1, n_bands + 1):
            col = band_colors.get(i, "rgba(100,200,255,0.3)")
            fig.add_trace(go.Scatter(
                x=timestamps, y=vwap_bands[f"+{i}"], name=f"+{i}σ",
                line=dict(color=col, width=1, dash="dot"), showlegend=(i == 1),
            ))
            fig.add_trace(go.Scatter(
                x=timestamps, y=vwap_bands[f"-{i}"], name=f"-{i}σ",
                line=dict(color=col, width=1, dash="dot"),
                fill="tonexty" if i > 1 else None,
                fillcolor=f"rgba(100,200,255,{0.03 * i})",
                showlegend=False,
            ))

    # Session POC/VA lines
    if show_sessions and cur_sess:
        for sess in cur_sess:
            col  = SESSION_LINE_COLORS.get(sess.name, "white")
            ts_s = datetime.fromtimestamp(sess.start_ts / 1000)
            ts_e = datetime.fromtimestamp(sess.end_ts   / 1000)

            if sess.poc:
                fig.add_shape(type="line", x0=ts_s, x1=ts_e, y0=sess.poc, y1=sess.poc,
                              line=dict(color=col, width=1.5, dash="dash"),
                              xref="x", yref="y")
                fig.add_annotation(x=ts_e, y=sess.poc,
                                   text=f"{sess.name} POC {sess.poc:,.0f}",
                                   font=dict(color=col, size=10),
                                   showarrow=False, xanchor="left")
            if sess.va_high and sess.va_low:
                fig.add_hrect(y0=sess.va_low, y1=sess.va_high,
                              fillcolor=SESSION_COLORS.get(sess.name, "rgba(255,255,255,0.05)"),
                              line_width=0)

    fig.update_layout(
        title=f"{symbol}  ·  {timeframe}",
        xaxis_rangeslider_visible=False,
        template="plotly_dark", height=480,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── CVD ───────────────────────────────────────────────────────────────────
    fig_cvd = go.Figure()
    fig_cvd.add_trace(go.Bar(x=timestamps, y=cvds,
        marker_color=["#26a69a" if c >= 0 else "#ef5350" for c in cvds], name="CVD"))
    fig_cvd.add_hline(y=0, line_color="gray", line_width=0.8)
    fig_cvd.update_layout(title="Cumulative Volume Delta", template="plotly_dark",
                           height=180, margin=dict(l=0,r=0,t=40,b=0))
    st.plotly_chart(fig_cvd, use_container_width=True)

    # ── Buy/Sell + Volume Profile ─────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=timestamps, y=buy_vols,  name="Buy",  marker_color="#26a69a"))
        fig_vol.add_trace(go.Bar(x=timestamps, y=[-v for v in sell_vols], name="Sell", marker_color="#ef5350"))
        fig_vol.update_layout(barmode="relative", title="Buy / Sell Volume",
                               template="plotly_dark", height=240, margin=dict(l=0,r=0,t=40,b=0))
        st.plotly_chart(fig_vol, use_container_width=True)
    with col_b:
        profile = engine.profile.to_dict()
        if profile:
            pvp = sorted(profile.keys())
            fig_vp = go.Figure()
            fig_vp.add_trace(go.Bar(
                x=[profile[p]["total"] for p in pvp], y=pvp, orientation="h",
                marker_color=["#26a69a" if profile[p]["delta"] >= 0 else "#ef5350" for p in pvp],
            ))
            if poc:   fig_vp.add_hline(y=poc, line_dash="dash", line_color="yellow")
            if va_lo: fig_vp.add_hrect(y0=va_lo, y1=va_hi, fillcolor="rgba(100,100,255,0.07)", line_width=0)
            fig_vp.update_layout(title="Volume Profile", template="plotly_dark",
                                  height=240, margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(fig_vp, use_container_width=True)

    # ── Session Table ─────────────────────────────────────────────────────────
    if cur_sess:
        st.markdown("---")
        st.markdown("**Session Analysis — Today**")
        scols = st.columns(len(cur_sess))
        for i, sess in enumerate(cur_sess):
            with scols[i]:
                color = SESSION_LINE_COLORS.get(sess.name, "white")
                st.markdown(f"<span style='color:{color}'>**{sess.name}**</span>", unsafe_allow_html=True)
                st.metric("POC",    f"{sess.poc:,.2f}"     if sess.poc     else "—")
                st.metric("VA H",   f"{sess.va_high:,.2f}" if sess.va_high else "—")
                st.metric("VA L",   f"{sess.va_low:,.2f}"  if sess.va_low  else "—")
                st.metric("VWAP",   f"{sess.vwap:,.2f}")
                st.metric("Range",  f"{sess.high - sess.low:,.2f}")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Footprint
# ════════════════════════════════════════════════════════════════════════════════
with tab_fp:
    st.caption("⚠️ Buy/Sell approximated from OHLCV — real footprint requires Bybit local connection.")

    footprints = build_footprints(bars, tick_size)
    if footprints:
        last_fp = footprints[-1]
        fc1, fc2, fc3, fc4 = st.columns(4)
        fc1.metric("Bar Delta",  f"{last_fp.delta:+,.0f}")
        fc2.metric("Bar Buy",    f"{last_fp.cum_buy:,.0f}")
        fc3.metric("Bar Sell",   f"{last_fp.cum_sell:,.0f}")
        fc4.metric("Bar POC",    f"{last_fp.poc:,.2f}" if last_fp.poc else "—")
        st.markdown("---")
        st.plotly_chart(render_footprint(footprints, fp_imb_thresh, fp_show_nums, fp_bars),
                        use_container_width=True)
        st.plotly_chart(render_footprint_summary(footprints), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Heatmap
# ════════════════════════════════════════════════════════════════════════════════
with tab_heatmap:
    st.caption("Liquidity clusters approximated from OHLCV. Green = bid concentration, Red = ask concentration.")

    lvls = sorted(heatmap.levels, key=lambda x: x.price)
    prices_hm  = [lv.price    for lv in lvls]
    bids_hm    = [lv.bid_size for lv in lvls]
    asks_hm    = [-lv.ask_size for lv in lvls]
    imb_colors = [
        "#26a69a" if lv.imbalance > 0.2 else
        "#ef5350" if lv.imbalance < -0.2 else
        "#888888"
        for lv in lvls
    ]

    fig_hm = go.Figure()
    fig_hm.add_trace(go.Bar(x=bids_hm,  y=prices_hm, orientation="h",
                             name="Bid Liquidity",  marker_color="#26a69a"))
    fig_hm.add_trace(go.Bar(x=asks_hm,  y=prices_hm, orientation="h",
                             name="Ask Liquidity",  marker_color="#ef5350"))
    if heatmap.mid_price:
        fig_hm.add_hline(y=heatmap.mid_price, line_color="white", line_dash="dot",
                          annotation_text=f"Mid {heatmap.mid_price:,.2f}")
    if poc:
        fig_hm.add_hline(y=poc, line_color="yellow", line_dash="dash",
                          annotation_text=f"POC {poc:,.2f}")
    fig_hm.update_layout(
        barmode="relative", title=f"Liquidity Heatmap — {symbol}",
        template="plotly_dark", height=600, margin=dict(l=0,r=0,t=40,b=0),
        xaxis_title="Volume", yaxis_title="Price",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Bid/Ask walls table
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("**Top Bid Walls (Support)**")
        bid_walls = sorted(heatmap.bid_wall(0.04), key=lambda x: -x.bid_size)[:8]
        for lv in bid_walls:
            st.write(f"`{lv.price:,.2f}` — {lv.bid_size:,.0f}")
    with col_w2:
        st.markdown("**Top Ask Walls (Resistance)**")
        ask_walls = sorted(heatmap.ask_wall(0.04), key=lambda x: -x.ask_size)[:8]
        for lv in ask_walls:
            st.write(f"`{lv.price:,.2f}` — {lv.ask_size:,.0f}")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Multi-TF CVD
# ════════════════════════════════════════════════════════════════════════════════
with tab_mtf:
    st.markdown(f"### Multi-Timeframe CVD — {symbol}")
    st.caption("Fetches CVD for each timeframe independently. Confluence = alignment across TFs.")

    with st.spinner("Fetching multi-timeframe data…"):
        snapshots = calculate_mtf_cvd(symbol, ["5m","15m","1h","4h","1d"], limit=100)
        conf = confluence_score(snapshots)

    # Confluence score bar
    score = conf["score"]
    score_color = "#26a69a" if score > 0.2 else "#ef5350" if score < -0.2 else "#888888"
    score_label = "BULLISH" if score > 0.2 else "BEARISH" if score < -0.2 else "NEUTRAL"
    st.markdown(
        f"<div style='background:{score_color};padding:12px;border-radius:8px;text-align:center;"
        f"font-size:1.3em;font-weight:bold;color:white'>"
        f"MTF Confluence: {score_label} ({conf['bullish']}↑ / {conf['bearish']}↓ of {conf['total']} TFs)"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    if snapshots:
        # CVD per TF bar chart
        tfs    = [s.timeframe  for s in snapshots]
        cvd_v  = [s.cvd        for s in snapshots]
        dirs   = [s.direction  for s in snapshots]
        colors = ["#26a69a" if d == "bullish" else "#ef5350" for d in dirs]

        fig_mtf = go.Figure()
        fig_mtf.add_trace(go.Bar(x=tfs, y=cvd_v, marker_color=colors, name="CVD per TF"))
        fig_mtf.add_hline(y=0, line_color="gray", line_width=1)
        fig_mtf.update_layout(
            title="CVD by Timeframe (positive = net buying pressure)",
            template="plotly_dark", height=320,
            margin=dict(l=0,r=0,t=40,b=0),
        )
        st.plotly_chart(fig_mtf, use_container_width=True)

        # Detail table
        st.markdown("**Timeframe Detail**")
        cols_mtf = st.columns(len(snapshots))
        for i, snap in enumerate(snapshots):
            with cols_mtf[i]:
                arrow = "🟢" if snap.direction == "bullish" else "🔴" if snap.direction == "bearish" else "⚪"
                st.metric(snap.timeframe, f"{snap.cvd:+,.0f}", delta=snap.direction)
                st.caption(f"{arrow} {snap.bars_used} bars")

if auto_ref:
    import time; time.sleep(60); st.rerun()

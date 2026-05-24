"""
OrderFlow System — Streamlit Dashboard v4
Tier 1: Signal Engine + Backtesting (Sharpe, Sortino, Omega, Kelly, Walk-Forward)
Tier 2: VWAP+SD, Session POC, Multi-TF CVD, Heatmap
Tier 3: Bybit live WebSocket feed — real buy/sell volumes, zero approximation
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from datetime import datetime
import pandas as pd
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
from core.signals   import SignalEngine
from core.backtest  import BacktestEngine, walk_forward
from dashboard.footprint_chart import render_footprint, render_footprint_summary
from dashboard.backtest_page   import (render_equity_curve, render_metrics_table,
                                        render_trades_scatter, render_signal_breakdown,
                                        render_walk_forward, render_monthly_pnl,
                                        render_rolling_winrate, render_pnl_distribution,
                                        render_r_multiple_chart, trade_streak_stats)
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

# ── Data source ───────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Source**")

_is_crypto = (market_choice == "Crypto")
_provider_opts = (
    ["Yahoo Finance", "Bybit — REST (enriched)", "Bybit — Live WebSocket"]
    if _is_crypto else ["Yahoo Finance"]
)
provider_choice = st.sidebar.radio(
    "Provider",
    _provider_opts,
    index=0,
    help=(
        "**Yahoo Finance**: works everywhere, buy/sell approximated.\n\n"
        "**Bybit REST**: local only — real data, close-position delta + recent-trade overlay.\n\n"
        "**Bybit Live**: local only — exact buy/sell from WebSocket trades in real-time."
    ),
)

_bybit_live = (provider_choice == "Bybit — Live WebSocket")
_bybit_rest = (provider_choice == "Bybit — REST (enriched)")
_use_bybit  = _bybit_live or _bybit_rest

if _use_bybit:
    st.sidebar.caption("⚠️ Bybit requires local network — geo-blocked on Streamlit Cloud.")

live_refresh_s = 5
if _bybit_live:
    live_refresh_s = st.sidebar.select_slider(
        "Live refresh (seconds)", options=[3, 5, 10, 15, 30], value=5
    )

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

st.sidebar.markdown("---")
st.sidebar.markdown("**Backtest**")
bt_capital    = st.sidebar.number_input("Capital (€)", value=10000, step=1000)
bt_risk       = st.sidebar.slider("Risk per Trade %", 0.5, 5.0, 1.0, step=0.5) / 100
bt_rr         = st.sidebar.slider("R:R Ratio", 1.0, 4.0, 2.0, step=0.5)
bt_sl_mult    = st.sidebar.slider("SL ATR Multiplier", 1.0, 3.0, 1.5, step=0.25)
bt_is_pct     = st.sidebar.slider("Walk-Forward IS %", 0.5, 0.85, 0.70, step=0.05)
bt_history    = st.sidebar.slider(
    "Backtest History (bars)", 500, 3000, 2000, step=100,
    help="Bars used for historical backtest only — separate from chart. "
         "2000 bars 1h ≈ 83 days ≈ ~120–180 trades."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Signal Filters**")
_ALL_SIGNALS = [
    "VA_BREAKOUT_BULL", "VA_BREAKOUT_BEAR",
    "POC_RECLAIM_BULL", "POC_RECLAIM_BEAR",
    "CVD_DIV_BULL",     "CVD_DIV_BEAR",
    "VWAP_CROSS_BULL",  "VWAP_CROSS_BEAR",
    "VWAP_BOUNCE_BULL", "VWAP_BOUNCE_BEAR",
]
# Default: disable the known underperformers (CVD_DIV_BEAR 25% WR, VWAP_BOUNCE negatives)
_DEFAULT_SIGNALS = [
    "VA_BREAKOUT_BULL", "VA_BREAKOUT_BEAR",
    "POC_RECLAIM_BULL", "POC_RECLAIM_BEAR",
    "CVD_DIV_BULL",
    "VWAP_CROSS_BULL",  "VWAP_CROSS_BEAR",
]
enabled_signals = st.sidebar.multiselect(
    "Active Signal Types",
    _ALL_SIGNALS,
    default=_DEFAULT_SIGNALS,
    help=(
        "Enable/disable individual signal types.\n\n"
        "**VA_BREAKOUT**: best performer (80%/67% WR) — always keep.\n"
        "**CVD_DIV_BEAR**: 25% WR with approximated data — disabled by default.\n"
        "**VWAP_BOUNCE**: negative avg PnL — disabled by default."
    ),
)
bt_min_conf = st.sidebar.slider(
    "Min Signal Confidence", 0.50, 0.90, 0.60, step=0.05,
    help="Signals below this threshold are discarded before backtest."
)

if not _bybit_live:
    auto_ref = st.sidebar.checkbox("Auto Refresh (60s)", value=False)
else:
    auto_ref = False   # live mode manages its own refresh cycle

# ── Live feed (Bybit WebSocket, one instance per symbol+timeframe) ────────────
@st.cache_resource
def _get_live_feed(symbol: str, timeframe: str):
    """One LiveFeed per (symbol, timeframe), shared across Streamlit reruns."""
    from providers.live_feed import LiveFeed
    feed = LiveFeed(symbol, timeframe)
    return feed   # start() called after seeding


# ── Data fetch ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60, show_spinner="Fetching Yahoo Finance data…")
def _load_yahoo(symbol, timeframe, limit):
    return YFinanceProvider().fetch_bars_sync(symbol, timeframe, limit), 0

@st.cache_data(ttl=30, show_spinner="Fetching Bybit enriched data…")
def _load_bybit_rest(symbol, timeframe, limit):
    from providers.bybit import BybitProvider
    bars, n_enriched = BybitProvider().fetch_bars_enriched(symbol, timeframe, limit)
    return bars, n_enriched

def _load_bybit_live(symbol, timeframe, limit):
    """Load historical REST bars + start live feed, merge both."""
    from providers.bybit import BybitProvider
    hist_bars, n_enriched = BybitProvider().fetch_bars_enriched(symbol, timeframe, limit)

    feed = _get_live_feed(symbol, timeframe)
    if not feed._thread or not feed._thread.is_alive():
        feed.seed(hist_bars)
        feed.start()

    bars = feed.get_bars(n=limit)
    if not bars:
        bars = hist_bars
    return bars, n_enriched

# Dispatch based on provider choice
n_enriched = 0
try:
    if _bybit_live:
        bars, n_enriched = _load_bybit_live(symbol, timeframe, bar_limit)
    elif _bybit_rest:
        bars, n_enriched = _load_bybit_rest(symbol, timeframe, bar_limit)
    else:
        bars, n_enriched = _load_yahoo(symbol, timeframe, bar_limit)
except Exception as e:
    st.error(f"Data error ({provider_choice}): {e}")
    if _use_bybit:
        st.info("Tip: Bybit only works locally. On Streamlit Cloud, switch to Yahoo Finance.")
    st.stop()

if not bars:
    st.warning("No data returned.")
    st.stop()

# ── Data quality badge ────────────────────────────────────────────────────────
if _bybit_live:
    feed_status = _get_live_feed(symbol, timeframe).status()
    if feed_status["connected"]:
        ago = feed_status.get("last_trade_ago")
        ago_txt = f"{ago}s ago" if ago is not None else "—"
        st.sidebar.success(
            f"🔴 **LIVE** · {feed_status['trade_count']:,} trades\n\n"
            f"Last: {ago_txt} · {feed_status['live_bars']} live bars"
        )
    else:
        err = feed_status.get("error", "")
        st.sidebar.warning(f"⚡ Connecting…{' — ' + err if err else ''}")
elif _bybit_rest:
    quality_label = (
        f"✅ {n_enriched} bars with real delta" if n_enriched > 0
        else "📊 Close-position approximation"
    )
    st.sidebar.info(quality_label)
else:
    st.sidebar.caption("📊 Delta approximated from OHLCV")

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

# Pre-compute signals — reuse already-loaded bars so no extra fetch
@st.cache_data(ttl=120, show_spinner=False)
def _compute_signals_cached(symbol, timeframe, limit, tick_size, rr, sl_mult, provider,
                             enabled_signals_tuple=None, min_conf=0.60):
    """Cached signal computation keyed by provider + active signal types."""
    # Convert hashable tuple back to list (or None = all signals)
    _enabled = list(enabled_signals_tuple) if enabled_signals_tuple else None
    if provider == "Bybit — REST (enriched)":
        from providers.bybit import BybitProvider
        b, _ = BybitProvider().fetch_bars_enriched(symbol, timeframe, limit)
    elif provider == "Bybit — Live WebSocket":
        # For signals, use the seeded+live merged bars (already computed above)
        return None, None   # handled outside cache
    else:
        b = YFinanceProvider().fetch_bars_sync(symbol, timeframe, limit)
    eng = SignalEngine(
        tick_size       = tick_size,
        rr_ratio        = rr,
        sl_atr_mult     = sl_mult,
        enabled_signals = _enabled,
        min_confidence  = min_conf,
    )
    return b, eng.detect(b)

# Convert list → tuple (hashable) for cache key; None means all enabled
_enabled_tuple = tuple(enabled_signals) if enabled_signals != _ALL_SIGNALS else None

if _bybit_live:
    # Use the already-merged bars — no extra fetch needed
    sig_bars = bars
    eng_sig  = SignalEngine(
        tick_size       = tick_size,
        rr_ratio        = bt_rr,
        sl_atr_mult     = bt_sl_mult,
        enabled_signals = list(_enabled_tuple) if _enabled_tuple else None,
        min_confidence  = bt_min_conf,
    )
    signals  = eng_sig.detect(sig_bars)
else:
    _cached = _compute_signals_cached(
        symbol, timeframe, bar_limit, tick_size, bt_rr, bt_sl_mult, provider_choice,
        enabled_signals_tuple=_enabled_tuple,
        min_conf=bt_min_conf,
    )
    sig_bars, signals = _cached if _cached[0] is not None else (bars, [])

# ── Historical backtest data (larger dataset — separate from chart) ────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_bt_history(symbol, timeframe, limit, provider):
    """
    Fetch a large historical bar set for the backtest.
    Separate from the chart data so the chart stays fast.
    Yahoo Finance returns up to 2 years of 1h data (~17,000 bars).
    """
    if "Bybit" in provider:
        from providers.bybit import BybitProvider
        try:
            return BybitProvider().fetch_bars_paginated(symbol, timeframe, limit)
        except Exception:
            pass   # fall back to Yahoo Finance
    return YFinanceProvider().fetch_bars_sync(symbol, timeframe, limit)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_main, tab_fp, tab_heatmap, tab_mtf, tab_bt = st.tabs([
    "📊 OrderFlow", "🔬 Footprint", "🌡️ Heatmap", "📐 Multi-TF CVD", "⚡ Backtest"
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

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Backtest
# ════════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.markdown(f"### Historical Backtest + Walk-Forward — {symbol} · {timeframe}")

    # ── Load large historical dataset (separate from chart) ───────────────────
    with st.spinner(f"Loading {bt_history} bars of history… (first load may take ~15s)"):
        bt_bars = load_bt_history(symbol, timeframe, bt_history, provider_choice)

    if not bt_bars:
        st.warning("No historical data returned.")
        st.stop()

    # ── Run signals on historical data ────────────────────────────────────────
    with st.spinner("Detecting signals on historical data…"):
        _bt_eng  = SignalEngine(
            tick_size       = tick_size,
            rr_ratio        = bt_rr,
            sl_atr_mult     = bt_sl_mult,
            enabled_signals = list(_enabled_tuple) if _enabled_tuple else None,
            min_confidence  = bt_min_conf,
        )
        bt_sigs  = _bt_eng.detect(bt_bars)

    # Date range of the backtest data
    _dt_start = datetime.fromtimestamp(bt_bars[0].timestamp  / 1000).strftime("%Y-%m-%d")
    _dt_end   = datetime.fromtimestamp(bt_bars[-1].timestamp / 1000).strftime("%Y-%m-%d")

    # Context banner
    _est_trades = max(1, int(len(bt_sigs) * 0.4))   # ~40% of signals → actual trades
    st.info(
        f"📅 **{_dt_start} → {_dt_end}** &nbsp;·&nbsp; "
        f"**{len(bt_bars):,} bars** &nbsp;·&nbsp; "
        f"**{len(bt_sigs)} signals** → ~**{_est_trades} estimated trades**"
    )

    if len(bt_sigs) < 10:
        st.warning("Too few signals. Try a higher Backtest History, or switch to 1h/4h timeframe.")
        st.stop()

    # ── Walk-Forward validation ───────────────────────────────────────────────
    with st.spinner("Running walk-forward validation…"):
        wf = walk_forward(
            bt_bars, bt_sigs,
            is_pct     = bt_is_pct,
            engine_cfg = dict(
                initial_capital   = bt_capital,
                risk_per_trade    = bt_risk,
                max_bars_in_trade = 20,
            ),
        )

    # Full backtest on ALL historical data (not split)
    _full_bt = BacktestEngine(
        initial_capital   = bt_capital,
        risk_per_trade    = bt_risk,
        max_bars_in_trade = 20,
    ).run(bt_bars, bt_sigs)

    n_total = _full_bt.report.get("n_trades", 0)

    # ── Edge verdict ──────────────────────────────────────────────────────────
    edge_color = "#26a69a" if wf.has_edge else "#ef5350"
    edge_label = "✅ EDGE CONFIRMED" if wf.has_edge else "❌ NO EDGE — DO NOT TRADE LIVE"
    st.markdown(
        f"<div style='background:{edge_color};padding:14px;border-radius:8px;"
        f"text-align:center;font-size:1.4em;font-weight:bold;color:white'>"
        f"{edge_label} &nbsp;|&nbsp; OOS/IS Sharpe degradation: {wf.degradation:.2f}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Full-history KPIs ─────────────────────────────────────────────────────
    st.markdown(f"#### Full Historical Backtest — {n_total} trades ({_dt_start} → {_dt_end})")
    _r = _full_bt.report
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Trades",        str(_r.get("n_trades", 0)))
    k2.metric("Win Rate",      f"{_r.get('win_rate', 0)*100:.1f}%")
    k3.metric("Profit Factor", f"{_r.get('profit_factor', 0):.2f}")
    k4.metric("Sharpe",        f"{_r.get('sharpe', 0):.2f}")
    k5.metric("Max DD",        f"{_r.get('max_drawdown_pct', 0):.1f}%")
    k6.metric("Total PnL",     f"€{_r.get('total_pnl', 0):,.0f}")

    # Streak stats
    _streaks = trade_streak_stats(_full_bt)
    if _streaks:
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Max Win Streak",  str(_streaks["max_win_streak"]))
        s2.metric("Max Loss Streak", str(_streaks["max_loss_streak"]))
        s3.metric("Avg Win Streak",  str(_streaks["avg_win_streak"]))
        s4.metric("Avg Loss Streak", str(_streaks["avg_loss_streak"]))
        cur = _streaks["current_streak"]
        s5.metric("Current Streak",
                  f"{'🟢 +' if cur > 0 else '🔴 '}{abs(cur)} {'wins' if cur > 0 else 'losses'}")

    st.markdown("---")

    # ── Full equity curve ─────────────────────────────────────────────────────
    st.plotly_chart(render_equity_curve(_full_bt, f"Full Equity Curve — {n_total} trades"),
                    use_container_width=True)

    # ── Monthly PnL + R-multiple ──────────────────────────────────────────────
    col_m, col_r = st.columns(2)
    with col_m:
        st.plotly_chart(render_monthly_pnl(_full_bt), use_container_width=True)
    with col_r:
        st.plotly_chart(render_r_multiple_chart(_full_bt), use_container_width=True)

    # ── Rolling metrics + distribution ────────────────────────────────────────
    col_rw, col_dist = st.columns(2)
    with col_rw:
        st.plotly_chart(render_rolling_winrate(_full_bt, window=20),
                        use_container_width=True)
    with col_dist:
        st.plotly_chart(render_pnl_distribution(_full_bt), use_container_width=True)

    st.markdown("---")

    # ── Walk-Forward IS vs OOS ────────────────────────────────────────────────
    st.markdown("#### Walk-Forward Validation (Out-of-Sample Test)")
    col_is, col_oos = st.columns(2)

    with col_is:
        st.markdown(f"#### In-Sample ({int(bt_is_pct*100)}%)")
        is_r = wf.is_result.report
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades",   str(is_r.get("n_trades", 0)))
        m2.metric("Win Rate", f"{is_r.get('win_rate', 0)*100:.1f}%")
        m3.metric("Sharpe",   f"{is_r.get('sharpe', 0):.2f}")
        m4.metric("PnL",      f"€{is_r.get('total_pnl', 0):,.0f}")
        st.plotly_chart(render_equity_curve(wf.is_result, "IS Equity"),
                        use_container_width=True)

    with col_oos:
        st.markdown(f"#### Out-of-Sample ({int((1-bt_is_pct)*100)}%)")
        oos_r = wf.oos_result.report
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Trades",   str(oos_r.get("n_trades", 0)))
        m2.metric("Win Rate", f"{oos_r.get('win_rate', 0)*100:.1f}%")
        m3.metric("Sharpe",   f"{oos_r.get('sharpe', 0):.2f}")
        m4.metric("PnL",      f"€{oos_r.get('total_pnl', 0):,.0f}")
        st.plotly_chart(render_equity_curve(wf.oos_result, "OOS Equity"),
                        use_container_width=True)

    st.markdown("---")

    # ── Full metrics table (OOS) ──────────────────────────────────────────────
    st.markdown("#### Full Performance Report — Out-of-Sample")
    col_met, col_eq = st.columns([1, 1.5])
    with col_met:
        st.plotly_chart(render_metrics_table(wf.oos_result.report),
                        use_container_width=True)
    with col_eq:
        st.plotly_chart(render_walk_forward(wf), use_container_width=True)
        st.markdown("##### Kelly Position Sizing")
        k_full = wf.oos_result.report.get("kelly_full", 0)
        k_half = wf.oos_result.report.get("kelly_half", 0)
        kc1, kc2 = st.columns(2)
        kc1.metric("Full Kelly", f"{k_full*100:.1f}% of capital",
                    help="Maximum theoretical bet size — aggressive")
        kc2.metric("Half Kelly", f"{k_half*100:.1f}% of capital",
                    help="Recommended: half Kelly reduces ruin probability")

    st.markdown("---")

    # ── Signal type breakdown ─────────────────────────────────────────────────
    st.markdown("#### Performance by Signal Type — Full History")
    st.plotly_chart(render_signal_breakdown(_full_bt), use_container_width=True)

    # ── Trade scatter + log ───────────────────────────────────────────────────
    st.markdown("#### All Trades — Full History")
    st.plotly_chart(render_trades_scatter(_full_bt, bt_bars), use_container_width=True)

    if _full_bt.trades:
        st.markdown("#### Trade Log")
        rows = []
        for t in _full_bt.trades:
            rows.append({
                "Date":    datetime.fromtimestamp(t.signal.timestamp/1000).strftime("%Y-%m-%d %H:%M"),
                "Signal":  t.signal.signal_type,
                "Dir":     t.signal.direction.upper(),
                "Entry":   f"{t.entry_price:,.2f}",
                "Exit":    f"{t.exit_price:,.2f}",
                "Reason":  t.exit_reason.upper(),
                "PnL (€)": f"{t.pnl:+,.2f}",
                "R":       f"{t.r_multiple:+.2f}",
                "Conf":    f"{t.signal.confidence:.0%}",
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, height=350)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if _bybit_live:
    # Live mode: short refresh cycle so the open bar updates in near-real-time
    time.sleep(live_refresh_s)
    st.rerun()
elif auto_ref:
    time.sleep(60)
    st.rerun()

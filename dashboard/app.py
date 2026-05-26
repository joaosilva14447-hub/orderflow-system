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
from core.mtf       import calculate_mtf_cvd, confluence_score as mtf_confluence_score
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
st.sidebar.title("📊 OrderFlow System")

# ── Mercado ───────────────────────────────────────────────────────────────────
market_choice = st.sidebar.selectbox("Mercado", list(MARKETS.keys()))
symbol        = st.sidebar.selectbox("Símbolo", MARKETS[market_choice])
timeframe     = st.sidebar.selectbox("Timeframe",
    ["1m","5m","15m","30m","1h","4h","1d"], index=2)
bar_limit     = st.sidebar.slider("Barras no gráfico", 50, 500, 200)

# ── Fonte de dados ────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
_is_crypto = (market_choice == "Crypto")
_provider_opts = (
    ["Yahoo Finance", "Bybit — REST (real delta)", "Bybit — Live WebSocket"]
    if _is_crypto else ["Yahoo Finance"]
)
provider_choice = st.sidebar.radio(
    "📡 Fonte de dados",
    _provider_opts,
    index=0,
    help=(
        "**Yahoo Finance** — funciona em qualquer lugar, delta aproximado.\n\n"
        "**Bybit REST** — delta real (close-position + trades recentes). Só local.\n\n"
        "**Bybit Live** — delta exacto via WebSocket em tempo real. Só local."
    ),
)

_bybit_live = (provider_choice == "Bybit — Live WebSocket")
_bybit_rest = (provider_choice == "Bybit — REST (real delta)")
_use_bybit  = _bybit_live or _bybit_rest

if _use_bybit:
    st.sidebar.caption("⚠️ Bybit só funciona localmente.")

live_refresh_s = 5
if _bybit_live:
    live_refresh_s = st.sidebar.select_slider(
        "Refresh (segundos)", options=[3, 5, 10, 15, 30], value=5
    )

# ── Indicadores (expander) ────────────────────────────────────────────────────
with st.sidebar.expander("📈 Indicadores", expanded=True):
    tick_size  = st.number_input("Tick Size", value=10.0, min_value=0.0001, format="%.4f")
    va_pct     = st.slider("Value Area %", 0.5, 0.9, 0.70, key="va_pct")
    st.markdown("**VWAP**")
    show_vwap  = st.checkbox("Mostrar VWAP", value=True)
    show_bands = st.checkbox("Mostrar Bandas SD", value=True)
    n_bands    = st.slider("Nº Bandas SD", 1, 3, 2)
    st.markdown("**Sessões**")
    show_sessions = st.checkbox("Mostrar POC/VA das sessões", value=True)

# ── Confluência (expander) ─────────────────────────────────────────────────────
with st.sidebar.expander("🎯 Confluência", expanded=True):
    conf_min_score = st.slider(
        "Score mínimo para alerta", 3, 6, 4,
        help="Número mínimo de factores alinhados para considerar confluência forte."
    )
    conf_cvd_bars = st.slider(
        "Janela CVD (barras)", 5, 30, 10,
        help="Nº de barras para calcular a tendência do CVD."
    )

# ── Avançado (expander fechado) ───────────────────────────────────────────────
with st.sidebar.expander("⚙️ Avançado", expanded=False):
    fp_bars       = st.slider("Footprint Bars", 5, 40, 15)
    fp_imb_thresh = st.slider("Imbalance Ratio", 1.5, 5.0, 3.0, step=0.5)
    fp_show_nums  = st.checkbox("Mostrar números de volume", value=True)
    st.markdown("---")
    if not _bybit_live:
        auto_ref = st.checkbox("Auto Refresh (60s)", value=False)
    else:
        auto_ref = False

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

# ── Função de confluência ─────────────────────────────────────────────────────

def _calc_confluence(bars, closes, cvds, vwaps, poc, va_lo, va_hi, cur_sess, conf_cvd_bars):
    """
    Calcula score de confluência para o último bar.
    Retorna (bull_factors, bear_factors) como listas de strings.
    Cada factor representa um indicador alinhado numa direcção.
    """
    if not bars:
        return [], []

    price = closes[-1]
    bull, bear = [], []

    # 1. VWAP
    if vwaps and vwaps[-1]:
        vwap = vwaps[-1]
        diff_pct = (price - vwap) / vwap * 100
        if diff_pct > 0.10:
            bull.append(f"Preço acima do VWAP (+{diff_pct:.2f}%)")
        elif diff_pct < -0.10:
            bear.append(f"Preço abaixo do VWAP ({diff_pct:.2f}%)")
        if abs(diff_pct) < 0.15:
            label = "🔵 Preço junto ao VWAP — zona de decisão"
            bull.append(label); bear.append(label)

    # 2. POC
    if poc:
        diff_pct = (price - poc) / poc * 100
        if diff_pct > 0.15:
            bull.append(f"Preço acima do POC (+{diff_pct:.2f}%)")
        elif diff_pct < -0.15:
            bear.append(f"Preço abaixo do POC ({diff_pct:.2f}%)")

    # 3. Value Area
    if va_hi and va_lo:
        mid_va = (va_hi + va_lo) / 2
        if price > va_hi:
            bull.append(f"Breakout acima da VA High ({va_hi:,.2f})")
        elif price < va_lo:
            bear.append(f"Breakdown abaixo da VA Low ({va_lo:,.2f})")
        elif price > mid_va:
            bull.append("Preço na metade superior da Value Area")
        else:
            bear.append("Preço na metade inferior da Value Area")

    # 4. CVD direcção (últimas N barras)
    n = conf_cvd_bars
    if cvds and len(cvds) >= n:
        delta_cvd = cvds[-1] - cvds[-n]
        if delta_cvd > 0:
            bull.append(f"CVD crescente (+{delta_cvd:+,.0f} em {n} barras)")
        else:
            bear.append(f"CVD decrescente ({delta_cvd:+,.0f} em {n} barras)")

    # 5. CVD nível absoluto
    if cvds:
        if cvds[-1] > 0:
            bull.append(f"CVD positivo acumulado ({cvds[-1]:+,.0f})")
        else:
            bear.append(f"CVD negativo acumulado ({cvds[-1]:+,.0f})")

    # 6. Delta da última barra
    if bars:
        last_delta = bars[-1].delta
        if last_delta > 0:
            bull.append(f"Delta positivo na última barra (+{last_delta:+,.0f})")
        else:
            bear.append(f"Delta negativo na última barra ({last_delta:+,.0f})")

    # 7. POC da sessão
    if cur_sess:
        for sess in cur_sess:
            if sess.poc:
                diff_pct = (price - sess.poc) / sess.poc * 100
                if diff_pct > 0.1:
                    bull.append(f"Acima do POC da sessão {sess.name} (+{diff_pct:.2f}%)")
                elif diff_pct < -0.1:
                    bear.append(f"Abaixo do POC da sessão {sess.name} ({diff_pct:.2f}%)")

    return bull, bear


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_main, tab_conf, tab_fp, tab_heatmap, tab_mtf = st.tabs([
    "📊 Orderflow", "🎯 Confluência", "🔬 Footprint", "🌡️ Heatmap", "📐 Multi-TF CVD"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OrderFlow
# ════════════════════════════════════════════════════════════════════════════════
with tab_main:

    # ── Score de confluência (calculado antes dos KPIs) ───────────────────────
    _bull_f, _bear_f = _calc_confluence(
        bars, closes, cvds, vwaps, poc, va_lo, va_hi, cur_sess, conf_cvd_bars
    )
    _score_bull = len(_bull_f)
    _score_bear = len(_bear_f)
    _max_score  = max(_score_bull + _score_bear, 1)

    # ── Linha 1: Preço e níveis ───────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Preço",   f"{closes[-1]:,.2f}")
    c2.metric("📍 VWAP",    f"{vwaps[-1]:,.2f}"  if vwaps else "—",
              delta=f"{(closes[-1]-vwaps[-1])/vwaps[-1]*100:+.2f}%" if vwaps and vwaps[-1] else None)
    c3.metric("🎯 POC",     f"{poc:,.2f}"         if poc   else "—",
              delta=f"{(closes[-1]-poc)/poc*100:+.2f}%" if poc else None)
    c4.metric("🔼 VA High", f"{va_hi:,.2f}"       if va_hi else "—")
    c5.metric("🔽 VA Low",  f"{va_lo:,.2f}"       if va_lo else "—")

    # ── Linha 2: Orderflow e confluência ─────────────────────────────────────
    d1, d2, d3, d4, d5 = st.columns(5)
    _last_delta = bars[-1].delta if bars else 0
    _buy_pct    = bars[-1].buy_volume / bars[-1].volume * 100 if bars and bars[-1].volume else 50
    d1.metric("📊 CVD",       f"{cvds[-1]:+,.0f}"  if cvds  else "—")
    d2.metric("⚡ Delta",     f"{_last_delta:+,.0f}")
    d3.metric("🟢 Buy Vol %", f"{_buy_pct:.1f}%",
              delta=f"{_buy_pct-50:+.1f}pp")

    # Confluência card
    if _score_bull >= conf_min_score:
        _conf_color = "#1a6b3c"; _conf_icon = "🟢"; _conf_txt = "BULLISH"
    elif _score_bear >= conf_min_score:
        _conf_color = "#7b1a1a"; _conf_icon = "🔴"; _conf_txt = "BEARISH"
    else:
        _conf_color = "#333333"; _conf_icon = "⚪"; _conf_txt = "NEUTRO"

    d4.metric(f"{_conf_icon} Confluência Bull", f"{_score_bull}/{_score_bull+_score_bear}")
    d5.metric(f"{_conf_icon} Confluência Bear", f"{_score_bear}/{_score_bull+_score_bear}")

    # Banner de confluência
    st.markdown(
        f"<div style='background:{_conf_color};padding:8px 16px;border-radius:6px;"
        f"text-align:center;font-weight:bold;color:white;margin-bottom:8px'>"
        f"{_conf_icon} {_conf_txt} — {_score_bull} factores bull · {_score_bear} factores bear"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Candlestick + VWAP + Sessions + Volume Profile overlay ──────────────
    # Volume Profile: calcular cores antes de criar o subplot
    _profile  = engine.profile.to_dict()
    _pvp      = sorted(_profile.keys()) if _profile else []
    _vol_vals = [_profile[p]["total"] for p in _pvp]
    _vp_w     = tick_size * 0.92   # largura de cada barra VP (em unidades de preço)

    _vp_colors = []
    for p in _pvp:
        d      = _profile[p]["delta"]
        is_poc = poc and abs(p - poc) < tick_size * 0.5
        in_va  = va_lo and va_hi and va_lo <= p <= va_hi
        if is_poc:
            _vp_colors.append("rgba(255,215,0,0.95)")          # POC — dourado
        elif in_va:
            _vp_colors.append("rgba(38,166,154,0.70)" if d >= 0
                               else "rgba(239,83,80,0.70)")     # VA — saturado
        else:
            _vp_colors.append("rgba(38,166,154,0.28)" if d >= 0
                               else "rgba(239,83,80,0.28)")     # Fora da VA — transparente

    # Subplot: candlestick (82%) | VP (18%), eixo Y partilhado
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.82, 0.18],
        shared_yaxes=True,
        horizontal_spacing=0.004,
    )

    # ── Candlestick ──────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=timestamps, open=[b.open for b in bars],
        high=[b.high for b in bars], low=[b.low for b in bars],
        close=closes, name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # ── Volume Profile (col 2) ───────────────────────────────────────────────
    if _pvp:
        fig.add_trace(go.Bar(
            x=_vol_vals, y=_pvp,
            orientation="h",
            marker_color=_vp_colors,
            marker_line_width=0,
            width=_vp_w,
            name="VP",
            showlegend=False,
            hovertemplate="Preço: %{y:,.2f}<br>Volume: %{x:,.0f}<extra>VP</extra>",
        ), row=1, col=2)

    # ── POC e VA (col 1 — linha e zona) ──────────────────────────────────────
    if poc:
        fig.add_hline(y=poc, line_dash="dash",
                      line_color="rgba(255,215,0,0.85)", line_width=1,
                      annotation_text="POC", annotation_position="left",
                      row=1, col=1)
    if va_lo and va_hi:
        fig.add_hrect(y0=va_lo, y1=va_hi,
                      fillcolor="rgba(100,100,255,0.06)", line_width=0,
                      row=1, col=1)

    # ── VWAP ─────────────────────────────────────────────────────────────────
    if show_vwap and vwaps:
        fig.add_trace(go.Scatter(
            x=timestamps, y=vwaps, name="VWAP",
            line=dict(color="rgba(255,255,255,0.9)", width=1.5),
        ), row=1, col=1)

    # ── SD Bands ─────────────────────────────────────────────────────────────
    if show_bands and vwap_bands:
        _band_colors = {1: "rgba(100,200,255,0.6)", 2: "rgba(100,200,255,0.4)", 3: "rgba(100,200,255,0.25)"}
        for i in range(1, n_bands + 1):
            _bc = _band_colors.get(i, "rgba(100,200,255,0.3)")
            fig.add_trace(go.Scatter(
                x=timestamps, y=vwap_bands[f"+{i}"], name=f"+{i}σ",
                line=dict(color=_bc, width=1, dash="dot"), showlegend=(i == 1),
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=timestamps, y=vwap_bands[f"-{i}"], name=f"-{i}σ",
                line=dict(color=_bc, width=1, dash="dot"),
                fill="tonexty" if i > 1 else None,
                fillcolor=f"rgba(100,200,255,{0.03*i})",
                showlegend=False,
            ), row=1, col=1)

    # ── Sessões ───────────────────────────────────────────────────────────────
    if show_sessions and cur_sess:
        for sess in cur_sess:
            _sc  = SESSION_LINE_COLORS.get(sess.name, "white")
            ts_s = datetime.fromtimestamp(sess.start_ts / 1000)
            ts_e = datetime.fromtimestamp(sess.end_ts   / 1000)
            if sess.poc:
                fig.add_shape(type="line",
                              x0=ts_s, x1=ts_e, y0=sess.poc, y1=sess.poc,
                              line=dict(color=_sc, width=1.5, dash="dash"),
                              row=1, col=1)
                fig.add_annotation(x=ts_e, y=sess.poc,
                                   text=f"{sess.name} POC {sess.poc:,.0f}",
                                   font=dict(color=_sc, size=10),
                                   showarrow=False, xanchor="left",
                                   row=1, col=1)
            if sess.va_high and sess.va_low:
                fig.add_hrect(y0=sess.va_low, y1=sess.va_high,
                              fillcolor=SESSION_COLORS.get(sess.name, "rgba(255,255,255,0.05)"),
                              line_width=0, row=1, col=1)

    fig.update_layout(
        title=f"{symbol}  ·  {timeframe}",
        template="plotly_dark", height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis =dict(rangeslider_visible=False),
        xaxis2=dict(showticklabels=False, showgrid=False, fixedrange=True,
                    title_text="Volume"),
        yaxis =dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
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

    # ── Buy / Sell Volume (largura total — VP já está no gráfico principal) ──
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=timestamps, y=buy_vols,
                             name="Buy",  marker_color="#26a69a"))
    fig_vol.add_trace(go.Bar(x=timestamps, y=[-v for v in sell_vols],
                             name="Sell", marker_color="#ef5350"))
    fig_vol.update_layout(
        barmode="relative", title="Buy / Sell Volume",
        template="plotly_dark", height=200,
        margin=dict(l=0, r=0, t=36, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # ── Nota de sessões (detalhes no tab Confluência) ─────────────────────────
    if cur_sess:
        st.markdown("---")
        _sess_parts = []
        for _s in cur_sess:
            if _s.poc:
                _col = SESSION_LINE_COLORS.get(_s.name, "white")
                _sess_parts.append(
                    f"<span style='color:{_col}'><b>{_s.name}</b> POC {_s.poc:,.0f}</span>"
                )
        _sess_names = " &nbsp;·&nbsp; ".join(_sess_parts)
        st.markdown(
            f"<div style='font-size:0.85em;color:#bbb;padding:6px 0'>"
            f"🕐 Sessões activas: {_sess_names}"
            f"&nbsp;— detalhes completos no tab <b>🎯 Confluência</b></div>",
            unsafe_allow_html=True,
        )

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
        conf = mtf_confluence_score(snapshots)

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
# TAB 2 — Confluência
# ════════════════════════════════════════════════════════════════════════════════
with tab_conf:
    st.markdown(f"### 🎯 Análise de Confluência — {symbol} · {timeframe}")
    st.caption("Quantos indicadores apontam na mesma direcção? Quanto mais factores alinhados, maior a confiança no setup.")

    # ── Score principal ───────────────────────────────────────────────────────
    _bull_fc, _bear_fc = _calc_confluence(
        bars, closes, cvds, vwaps, poc, va_lo, va_hi, cur_sess, conf_cvd_bars
    )
    _sb = len(_bull_fc)
    _se = len(_bear_fc)
    _total = _sb + _se

    # Barra de confluência visual
    _pct_bull = int(_sb / _total * 100) if _total > 0 else 50
    _pct_bear = 100 - _pct_bull

    if _sb >= conf_min_score and _sb > _se:
        _verdict_color = "#1a6b3c"; _verdict_icon = "🟢"; _verdict = "CONFLUÊNCIA BULLISH"
    elif _se >= conf_min_score and _se > _sb:
        _verdict_color = "#7b1a1a"; _verdict_icon = "🔴"; _verdict = "CONFLUÊNCIA BEARISH"
    elif _sb >= conf_min_score or _se >= conf_min_score:
        _verdict_color = "#6b5a1a"; _verdict_icon = "🟡"; _verdict = "CONFLUÊNCIA MODERADA"
    else:
        _verdict_color = "#2a2a2a"; _verdict_icon = "⚪"; _verdict = "SEM CONFLUÊNCIA CLARA"

    st.markdown(
        f"<div style='background:{_verdict_color};padding:18px 24px;border-radius:10px;"
        f"text-align:center;font-size:1.5em;font-weight:bold;color:white;margin-bottom:16px'>"
        f"{_verdict_icon} {_verdict}<br>"
        f"<span style='font-size:0.65em;font-weight:normal'>"
        f"{_sb} factores bullish · {_se} factores bearish · Score mínimo: {conf_min_score}"
        f"</span></div>",
        unsafe_allow_html=True,
    )

    # Barra de força visual
    st.markdown(
        f"<div style='display:flex;height:20px;border-radius:4px;overflow:hidden;margin-bottom:16px'>"
        f"<div style='width:{_pct_bull}%;background:#26a69a'></div>"
        f"<div style='width:{_pct_bear}%;background:#ef5350'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Factores lado a lado ──────────────────────────────────────────────────
    col_bull, col_bear = st.columns(2)

    with col_bull:
        st.markdown(f"#### 🟢 Factores Bullish ({_sb})")
        if _bull_fc:
            for f in _bull_fc:
                bg = "#0d2b1a" if not f.startswith("🔵") else "#1a2b40"
                st.markdown(
                    f"<div style='background:{bg};padding:8px 12px;border-radius:6px;"
                    f"border-left:3px solid #26a69a;margin-bottom:6px;font-size:0.9em'>"
                    f"✅ {f}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Nenhum factor bullish activo.")

    with col_bear:
        st.markdown(f"#### 🔴 Factores Bearish ({_se})")
        if _bear_fc:
            for f in _bear_fc:
                bg = "#2b0d0d" if not f.startswith("🔵") else "#1a2b40"
                st.markdown(
                    f"<div style='background:{bg};padding:8px 12px;border-radius:6px;"
                    f"border-left:3px solid #ef5350;margin-bottom:6px;font-size:0.9em'>"
                    f"❌ {f}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Nenhum factor bearish activo.")

    st.markdown("---")

    # ── Níveis-chave actuais ──────────────────────────────────────────────────
    st.markdown("#### 📍 Níveis-Chave Actuais")
    nk1, nk2, nk3, nk4, nk5 = st.columns(5)
    _price = closes[-1]
    nk1.metric("Preço Actual", f"{_price:,.2f}")
    nk2.metric("VWAP",         f"{vwaps[-1]:,.2f}" if vwaps else "—",
               delta=f"{(_price-vwaps[-1])/vwaps[-1]*100:+.2f}%" if vwaps and vwaps[-1] else None)
    nk3.metric("POC",          f"{poc:,.2f}" if poc else "—",
               delta=f"{(_price-poc)/poc*100:+.2f}%" if poc else None)
    nk4.metric("VA High",      f"{va_hi:,.2f}" if va_hi else "—",
               delta=f"{(_price-va_hi)/va_hi*100:+.2f}%" if va_hi else None)
    nk5.metric("VA Low",       f"{va_lo:,.2f}" if va_lo else "—",
               delta=f"{(_price-va_lo)/va_lo*100:+.2f}%" if va_lo else None)

    # ── Sessões ───────────────────────────────────────────────────────────────
    if cur_sess:
        st.markdown("---")
        st.markdown("#### 🕐 Análise de Sessões")
        scols = st.columns(len(cur_sess))
        for i, sess in enumerate(cur_sess):
            with scols[i]:
                color = SESSION_LINE_COLORS.get(sess.name, "white")
                st.markdown(
                    f"<div style='border-left:4px solid {color};padding-left:10px'>"
                    f"<strong style='color:{color}'>{sess.name}</strong></div>",
                    unsafe_allow_html=True,
                )
                st.metric("POC",   f"{sess.poc:,.2f}"     if sess.poc     else "—",
                           delta=f"{(_price-sess.poc)/sess.poc*100:+.2f}%" if sess.poc else None)
                st.metric("VA H",  f"{sess.va_high:,.2f}" if sess.va_high else "—")
                st.metric("VA L",  f"{sess.va_low:,.2f}"  if sess.va_low  else "—")
                st.metric("VWAP",  f"{sess.vwap:,.2f}")
                st.metric("Range", f"{sess.high - sess.low:,.2f}")

    # ── Multi-TF CVD (confluência cross-timeframe) ────────────────────────────
    st.markdown("---")
    st.markdown("#### 📐 CVD Multi-Timeframe")
    with st.spinner("A carregar CVD multi-timeframe…"):
        _snapshots = calculate_mtf_cvd(symbol, ["5m","15m","1h","4h","1d"], limit=100)
        _conf_mtf  = mtf_confluence_score(_snapshots)

    _mtf_score  = _conf_mtf["score"]
    _mtf_color  = "#1a6b3c" if _mtf_score > 0.2 else "#7b1a1a" if _mtf_score < -0.2 else "#333333"
    _mtf_label  = "BULLISH" if _mtf_score > 0.2 else "BEARISH" if _mtf_score < -0.2 else "NEUTRO"

    st.markdown(
        f"<div style='background:{_mtf_color};padding:10px;border-radius:6px;"
        f"text-align:center;font-weight:bold;color:white'>"
        f"MTF CVD: {_mtf_label} — {_conf_mtf['bullish']}↑ / {_conf_mtf['bearish']}↓ de {_conf_mtf['total']} TFs"
        f"</div>",
        unsafe_allow_html=True,
    )

    if _snapshots:
        _mtf_cols = st.columns(len(_snapshots))
        for i, snap in enumerate(_snapshots):
            with _mtf_cols[i]:
                arrow = "🟢" if snap.direction == "bullish" else "🔴" if snap.direction == "bearish" else "⚪"
                st.metric(snap.timeframe, f"{snap.cvd:+,.0f}", delta=snap.direction)
                st.caption(f"{arrow} {snap.bars_used} barras")

    # ── Guia de leitura ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 Como usar esta análise de confluência"):
        st.markdown("""
**Score mínimo** (configurável na sidebar, padrão = 4):
- **≥ 4 factores alinhados** → confluência forte — setup de alta qualidade
- **3 factores** → confluência moderada — usar com precaução
- **≤ 2 factores** → sem confluência — aguardar melhores condições

**Factores incluídos (7 no total):**
1. **Preço vs VWAP** — acima/abaixo do equilíbrio do dia
2. **Preço vs POC** — acima/abaixo do nível de maior volume
3. **Posição na Value Area** — bull/bear dentro da VA, breakout/breakdown fora
4. **CVD direcção** — o fluxo de ordens está a crescer ou a decrescer?
5. **CVD nível** — acumulação positiva ou negativa geral?
6. **Delta da última barra** — quem dominou a última vela?
7. **POC da sessão** — como estamos em relação ao POC da sessão actual?

**Regra prática:** Só consideres entrar quando ≥ 4 factores apontam na mesma direcção E o preço está perto de um nível-chave (VWAP, POC, VA High/Low).
        """)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if _bybit_live:
    # Live mode: short refresh cycle so the open bar updates in near-real-time
    time.sleep(live_refresh_s)
    st.rerun()
elif auto_ref:
    time.sleep(60)
    st.rerun()

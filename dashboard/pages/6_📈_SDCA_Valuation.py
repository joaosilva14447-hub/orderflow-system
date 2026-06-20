"""
📈 SDCA Valuation Oscillator
============================
Oscilador proprietário de valorização de longo prazo (0–100) para SDCA.
Sinaliza extremos de ciclo (oversold ↔ overbought) — ignora ruído intermédio.
Score alto = caro (realizar); score baixo = barato (acumular).
Anti-overfit: pesos iguais + normalização por percentil com decaimento (2 anos).
"""
from __future__ import annotations

import math
import sys
import os
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from dashboard import valuation_engine as ve

st.set_page_config(page_title="SDCA Valuation Oscillator", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

OVERSOLD_MAX = 20    # < 20 → sobrevendido (acumular)
OVERBOUGHT_MIN = 80  # > 80 → sobrecomprado (realizar)

# Paleta TradingView: teal + coral + linha branca
C_GREEN = "#26A69A"   # teal (sobrevendido / acumular)
C_RED = "#EF5350"     # coral (sobrecomprado / realizar)
C_LINE = "#FFFFFF"    # linha branca
C_NEUTRAL = "#78909C"
C_GREEN_SOFT = "#4DB6AC"
C_RED_SOFT = "#EF9A9A"


@st.cache_data(ttl=3600)
def load_series():
    """Preço (yfinance→CoinGecko→demo) + MVRV on-chain (opcional)."""
    source = "Dados diários reais"
    try:
        dates, prices = ve.fetch_btc_history()
        dates, prices = ve.resample_weekly(dates, prices)
        ppy = 52
    except Exception as exc:  # noqa: BLE001
        dates, prices = ve._demo_series()
        ppy = 12
        source = f"Demo (sem rede: {exc})"
    extra, onchain = {}, False
    try:
        md, mv = ve.fetch_onchain_mvrv()
        extra["mvrv"] = ve.align_series(dates, md, mv)
        onchain = True
    except Exception:  # noqa: BLE001
        onchain = False
    val = ve.compute_series(dates, prices, periods_per_year=ppy, extra_raw=extra)
    return dates, prices, val, source, onchain


def _last_valid(arr: np.ndarray) -> int:
    idx = np.where(~np.isnan(arr))[0]
    return int(idx[-1]) if len(idx) else len(arr) - 1


def zone_color(s: float) -> str:
    if s < OVERSOLD_MAX:
        return C_GREEN
    if s < 35:
        return C_GREEN_SOFT
    if s < 65:
        return C_NEUTRAL
    if s < OVERBOUGHT_MIN:
        return "#FB923C"
    return C_RED


def card_html(label: str, value: str, accent: str) -> str:
    return (f'<div style="background:rgba(255,255,255,0.04);border-left:4px solid '
            f'{accent};border-radius:10px;padding:13px 16px;height:100%;">'
            f'<div style="font-size:11px;color:#9aa0a6;text-transform:uppercase;'
            f'letter-spacing:.6px;">{label}</div>'
            f'<div style="font-size:22px;font-weight:600;color:#eaecef;margin-top:5px;'
            f'line-height:1.15;">{value}</div></div>')


# ── Gauge SVG personalizado (arco premium + ponteiro + número grande) ─────────
def _polar(cx, cy, r, ang):
    a = math.radians(ang - 90)
    return cx + r * math.cos(a), cy + r * math.sin(a)


def _arc(cx, cy, r, v0, v1):
    a0, a1 = -90 + 1.8 * v0, -90 + 1.8 * v1
    sx, sy = _polar(cx, cy, r, a1)
    ex, ey = _polar(cx, cy, r, a0)
    large = 1 if (a1 - a0) > 180 else 0
    return f"M {sx:.1f} {sy:.1f} A {r} {r} 0 {large} 0 {ex:.1f} {ey:.1f}"


def gauge_svg(score: float, accent: str) -> str:
    cx, cy, r, sw = 170, 150, 118, 24
    s = max(0.0, min(100.0, score))
    tx, ty = _polar(cx, cy, r - 18, -90 + 1.8 * s)
    svg = (
        f'<svg viewBox="0 0 340 232" width="100%" style="max-width:380px;'
        f'font-family:Inter,Arial,sans-serif;">'
        f'<path d="{_arc(cx, cy, r, 0, 100)}" stroke="#262c38" stroke-width="{sw}" '
        f'fill="none" stroke-linecap="round"/>'
        f'<path d="{_arc(cx, cy, r, 0, OVERSOLD_MAX)}" stroke="{C_GREEN}" '
        f'stroke-width="{sw}" fill="none" stroke-linecap="round"/>'
        f'<path d="{_arc(cx, cy, r, OVERBOUGHT_MIN, 100)}" stroke="{C_RED}" '
        f'stroke-width="{sw}" fill="none" stroke-linecap="round"/>'
        f'<line x1="{cx}" y1="{cy}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="#ffffff" '
        f'stroke-width="4" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="9" fill="#ffffff"/>'
        f'<circle cx="{cx}" cy="{cy}" r="4.5" fill="#0E1117"/>'
        f'<text x="{cx}" y="{cy + 54}" text-anchor="middle" font-size="52" '
        f'font-weight="800" fill="{accent}">{round(s)}</text>'
        f'<text x="{cx}" y="{cy + 76}" text-anchor="middle" font-size="12" '
        f'letter-spacing="2" fill="#7d8595">DE 100</text>'
        f'</svg>')
    return (f"<style>body{{margin:0;background:#0E1117;}}</style>"
            f"<div style='display:flex;justify-content:center;align-items:center;'>{svg}</div>")


dates, prices, val, source, onchain = load_series()
i = _last_valid(val.composite)
score = float(val.composite[i])
label, action = ve.zone_for(score)
conv = float(val.conviction[i]) if not np.isnan(val.conviction[i]) else 0.0
conv_label = "Alta" if conv >= 0.6 else ("Média" if conv >= 0.3 else "Baixa")
accent = zone_color(score)
conv_accent = "#3498DB" if conv >= 0.6 else ("#95A5A6" if conv >= 0.3 else "#7F8C8D")
dot = "🟢" if score < 35 else ("🟠" if score >= 65 else "⚪")

st.title("📈 SDCA Valuation Oscillator")
oc = "MVRV on-chain ✓" if onchain else "MVRV on-chain indisponível"
st.caption(f"Valorização de longo prazo do BTC (0–100) para SDCA — extremos de "
           f"ciclo. {source} · {oc}")

st.markdown(
    f'<div style="background:{accent}22;border-left:5px solid {accent};'
    f'border-radius:10px;padding:12px 18px;font-size:16px;color:#eaecef;'
    f'margin-bottom:14px;">{dot} &nbsp;<b style="color:{accent};">{action}</b> '
    f'— zona <b>{label}</b> · score <b>{score:.0f}/100</b> · convicção '
    f'<b>{conv_label}</b> ({conv * 100:.0f}%)</div>', unsafe_allow_html=True)

gcol, ccol = st.columns([1, 2])
with gcol:
    components.html(gauge_svg(score, accent), height=240)
ccol.markdown(
    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;height:230px;">'
    + card_html("Zona", label, accent)
    + card_html("Ação SDCA", action, accent)
    + card_html("Convicção", f"{conv_label} ({conv * 100:.0f}%)", conv_accent)
    + card_html("Fonte de dados", "Preço + MVRV ✓" if onchain else "Preço", C_LINE)
    + '</div>', unsafe_allow_html=True)

st.divider()
st.subheader("Histórico do ciclo")
fig = go.Figure()
fig.add_shape(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=OVERBOUGHT_MIN,
              y1=100, fillcolor=C_RED, opacity=0.18, line_width=0, layer="below")
fig.add_shape(type="rect", xref="paper", x0=0, x1=1, yref="y", y0=0,
              y1=OVERSOLD_MAX, fillcolor=C_GREEN, opacity=0.18, line_width=0, layer="below")
fig.add_hline(y=OVERBOUGHT_MIN, line=dict(color=C_RED, width=1.3, dash="dash"))
fig.add_hline(y=OVERSOLD_MAX, line=dict(color=C_GREEN, width=1.3, dash="dash"))
fig.add_annotation(xref="paper", x=0.012, y=95, yref="y", xanchor="left",
                   text="SOBRECOMPRADO — realizar", showarrow=False,
                   font=dict(color=C_RED_SOFT, size=12))
fig.add_annotation(xref="paper", x=0.012, y=5, yref="y", xanchor="left",
                   text="SOBREVENDIDO — acumular", showarrow=False,
                   font=dict(color=C_GREEN_SOFT, size=12))
fig.add_trace(go.Scatter(x=dates, y=val.composite, mode="lines", showlegend=False,
                         line=dict(color=C_LINE, width=2.8),
                         fill="tozeroy", fillcolor="rgba(255,255,255,0.04)"))
fig.add_trace(go.Scatter(x=[dates[i]], y=[score], mode="markers", showlegend=False,
                         marker=dict(color="#FFA726", size=10, line=dict(color="white", width=1.5))))
fig.add_annotation(x=dates[i], y=score, text=f"<b>{score:.0f}</b>", showarrow=False,
                   xanchor="left", xshift=9, font=dict(color="#0E1117", size=13),
                   bgcolor="#FFA726", borderpad=3)
for hd, _r in ve.HALVINGS:
    if dates[0] <= hd <= dates[-1]:
        fig.add_vline(x=hd, line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot"))
fig.update_yaxes(title_text="Valorização (0–100)", range=[0, 100],
                 gridcolor="rgba(148,163,184,0.10)", zeroline=False)
fig.update_xaxes(gridcolor="rgba(148,163,184,0.06)")
fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                  hovermode="x unified", template="plotly_dark",
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True, theme=None)

st.divider()
st.subheader("Decomposição — o que está a puxar o score (hoje)")
names = {
    "trend_deviation": "Desvio lei de potência",
    "long_ma_ratio": "Rácio MA longa",
    "drawdown": "Drawdown",
    "momentum": "Momentum",
    "issuance_value": "Valor de emissão",
    "ma_spread": "Spread MA (topo)",
    "mvrv": "MVRV (on-chain)",
}
pairs = [(names[k], round(float(val.primitives_pct[k][i]) * 100)) for k in names
         if k in val.primitives_pct and not np.isnan(val.primitives_pct[k][i])]
pairs.sort(key=lambda kv: kv[1], reverse=True)


def _bar_color(v: float) -> str:
    if v < 35:
        return C_GREEN
    if v <= 65:
        return C_NEUTRAL
    return C_RED


bar = go.Figure(go.Bar(
    x=[p[0] for p in pairs], y=[p[1] for p in pairs],
    marker_color=[_bar_color(p[1]) for p in pairs],
    text=[p[1] for p in pairs], textposition="outside"))
bar.add_hline(y=50, line=dict(color="rgba(200,200,200,0.4)", width=1, dash="dash"))
bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                  yaxis=dict(title="Percentil (0–100)", range=[0, 105],
                            gridcolor="rgba(148,163,184,0.10)"),
                  template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(bar, use_container_width=True, theme=None)
st.caption("Verde = baixo percentil (barato) · vermelho = alto (caro). "
           "Linha a 50 = mediana histórica.")
st.caption("Pesos iguais + normalização com decaimento (2 anos) + suavização — "
           "anti-overfit. NÃO é aconselhamento financeiro.")

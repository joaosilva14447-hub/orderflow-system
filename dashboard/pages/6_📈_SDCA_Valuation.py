"""
📈 SDCA Valuation Oscillator
============================
Oscilador proprietário de valorização de longo prazo (0–100) para SDCA.
Sinaliza extremos de ciclo (oversold ↔ overbought) — ignora ruído intermédio.
Score alto = caro (realizar); score baixo = barato (acumular).
Anti-overfit: pesos iguais + normalização por percentil com decaimento (2 anos).
"""
from __future__ import annotations

import sys
import os
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dashboard import valuation_engine as ve

st.set_page_config(
    page_title="SDCA Valuation Oscillator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

OVERSOLD_MAX = 20    # < 20 → sobrevendido (acumular)
OVERBOUGHT_MIN = 80  # > 80 → sobrecomprado (realizar)


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
        return "#27AE60"
    if s < 35:
        return "#58D68D"
    if s < 65:
        return "#95A5A6"
    if s < OVERBOUGHT_MIN:
        return "#E59866"
    return "#CB4335"


def card_html(label: str, value: str, accent: str) -> str:
    return (f'<div style="background:rgba(255,255,255,0.04);border-left:4px solid '
            f'{accent};border-radius:10px;padding:13px 16px;height:100%;">'
            f'<div style="font-size:11px;color:#9aa0a6;text-transform:uppercase;'
            f'letter-spacing:.6px;">{label}</div>'
            f'<div style="font-size:22px;font-weight:600;color:#eaecef;margin-top:5px;'
            f'line-height:1.15;">{value}</div></div>')


dates, prices, val, source, onchain = load_series()
i = _last_valid(val.composite)
score = float(val.composite[i])
label, action = ve.zone_for(score)
conv = float(val.conviction[i]) if not np.isnan(val.conviction[i]) else 0.0
conv_label = "Alta" if conv >= 0.6 else ("Média" if conv >= 0.3 else "Baixa")
accent = zone_color(score)
conv_accent = "#3498DB" if conv >= 0.6 else ("#95A5A6" if conv >= 0.3 else "#7F8C8D")
dot = "🟢" if score < 35 else ("🔴" if score >= OVERBOUGHT_MIN else
                               ("🟠" if score >= 65 else "⚪"))

st.title("📈 SDCA Valuation Oscillator")
oc = "MVRV on-chain ✓" if onchain else "MVRV on-chain indisponível"
st.caption(f"Valorização de longo prazo do BTC (0–100) para SDCA — extremos de "
           f"ciclo. {source} · {oc}")

# ── Faixa de sinal (verdito num relance) ─────────────────────────────────────
st.markdown(
    f'<div style="background:{accent}22;border-left:5px solid {accent};'
    f'border-radius:10px;padding:12px 18px;font-size:16px;color:#eaecef;'
    f'margin-bottom:14px;">{dot} &nbsp;<b style="color:{accent};">{action}</b> '
    f'— zona <b>{label}</b> · score <b>{score:.0f}/100</b> · convicção '
    f'<b>{conv_label}</b> ({conv * 100:.0f}%)</div>',
    unsafe_allow_html=True)

# ── Leitura atual: gauge + cartões ───────────────────────────────────────────
gcol, ccol = st.columns([1, 2])
gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=round(score),
    number={"suffix": "/100", "font": {"size": 44, "color": accent}},
    gauge={
        "axis": {"range": [0, 100], "tickvals": [0, 50, 100], "tickcolor": "#9aa0a6",
                 "tickwidth": 1},
        "bar": {"color": "rgba(0,0,0,0)", "thickness": 0},
        "bgcolor": "rgba(0,0,0,0)",
        "borderwidth": 0,
        "steps": [
            {"range": [0, OVERSOLD_MAX], "color": "rgba(39,174,96,0.65)"},
            {"range": [OVERSOLD_MAX, OVERBOUGHT_MIN], "color": "rgba(120,125,140,0.18)"},
            {"range": [OVERBOUGHT_MIN, 100], "color": "rgba(203,67,53,0.65)"},
        ],
        "threshold": {"line": {"color": "#FFFFFF", "width": 4}, "thickness": 0.9,
                      "value": round(score)},
    },
))
gauge.update_layout(height=250, margin=dict(l=30, r=30, t=25, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", font={"color": "#cfd2d6"})
gcol.plotly_chart(gauge, use_container_width=True)

ccol.markdown(
    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;height:230px;">'
    + card_html("Zona", label, accent)
    + card_html("Ação SDCA", action, accent)
    + card_html("Convicção", f"{conv_label} ({conv * 100:.0f}%)", conv_accent)
    + card_html("Fonte de dados", "Preço + MVRV ✓" if onchain else "Preço", "#5D4FB0")
    + '</div>', unsafe_allow_html=True)

st.divider()

# ── Histórico do ciclo ───────────────────────────────────────────────────────
st.subheader("Histórico do ciclo")
fig = go.Figure()
fig.add_shape(type="rect", xref="paper", x0=0, x1=1, yref="y",
              y0=OVERBOUGHT_MIN, y1=100, fillcolor="#A93226", opacity=0.33,
              line_width=0, layer="below")
fig.add_shape(type="rect", xref="paper", x0=0, x1=1, yref="y",
              y0=0, y1=OVERSOLD_MAX, fillcolor="#1E8449", opacity=0.33,
              line_width=0, layer="below")
fig.add_hline(y=OVERBOUGHT_MIN, line=dict(color="#CB4335", width=1.3, dash="dash"))
fig.add_hline(y=OVERSOLD_MAX, line=dict(color="#27AE60", width=1.3, dash="dash"))
fig.add_annotation(xref="paper", x=0.012, y=95, yref="y", xanchor="left",
                   text="SOBRECOMPRADO — realizar", showarrow=False,
                   font=dict(color="#F1948A", size=12))
fig.add_annotation(xref="paper", x=0.012, y=5, yref="y", xanchor="left",
                   text="SOBREVENDIDO — acumular", showarrow=False,
                   font=dict(color="#7DCEA0", size=12))
fig.add_trace(go.Scatter(x=dates, y=val.composite, mode="lines", showlegend=False,
                         line=dict(color="#38BDF8", width=2.8),
                         fill="tozeroy", fillcolor="rgba(56,189,248,0.08)"))
fig.add_trace(go.Scatter(x=[dates[i]], y=[score], mode="markers", showlegend=False,
                         marker=dict(color="#38BDF8", size=11, line=dict(color="white", width=1.5))))
fig.add_annotation(x=dates[i], y=score, text=f"<b>{score:.0f}</b>", showarrow=False,
                   xanchor="left", xshift=9, font=dict(color="#ffffff", size=14),
                   bgcolor="#38BDF8", borderpad=3)
for hd, _r in ve.HALVINGS:
    if dates[0] <= hd <= dates[-1]:
        fig.add_vline(x=hd, line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot"))
fig.update_yaxes(title_text="Valorização (0–100)", range=[0, 100])
fig.update_layout(height=480, margin=dict(l=10, r=10, t=10, b=10),
                  hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Decomposição ─────────────────────────────────────────────────────────────
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
        return "#27AE60"
    if v <= 65:
        return "#95A5A6"
    return "#CB4335"


bar = go.Figure(go.Bar(
    x=[p[0] for p in pairs], y=[p[1] for p in pairs],
    marker_color=[_bar_color(p[1]) for p in pairs],
    text=[p[1] for p in pairs], textposition="outside"))
bar.add_hline(y=50, line=dict(color="rgba(200,200,200,0.4)", width=1, dash="dash"))
bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                  yaxis=dict(title="Percentil (0–100)", range=[0, 105]),
                  template="plotly_white")
st.plotly_chart(bar, use_container_width=True)
st.caption("Verde = baixo percentil (barato) · vermelho = alto (caro). "
           "Linha a 50 = mediana histórica.")

st.caption("Pesos iguais + normalização com decaimento (2 anos) + suavização — "
           "anti-overfit. NÃO é aconselhamento financeiro.")

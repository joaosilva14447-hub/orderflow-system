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
from plotly.subplots import make_subplots
import streamlit as st

from dashboard import valuation_engine as ve

st.set_page_config(
    page_title="SDCA Valuation Oscillator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

OVERSOLD_MAX = 35    # < 35 → sobrevendido (acumular)
OVERBOUGHT_MIN = 65  # > 65 → sobrecomprado (realizar)


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


dates, prices, val, source, onchain = load_series()
i = _last_valid(val.composite)
score = float(val.composite[i])
label, action = ve.zone_for(score)
conv = float(val.conviction[i]) if not np.isnan(val.conviction[i]) else 0.0
conv_label = "Alta" if conv >= 0.6 else ("Média" if conv >= 0.3 else "Baixa")

st.title("📈 SDCA Valuation Oscillator")
oc = "MVRV on-chain ✓" if onchain else "MVRV on-chain indisponível"
st.caption(f"Valorização de longo prazo do BTC (0–100) para SDCA — extremos de "
           f"ciclo. {source} · {oc}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Score", f"{score:.0f}/100")
c2.metric("Zona", label)
c3.metric("Ação SDCA", action)
c4.metric("Convicção", f"{conv_label} ({conv * 100:.0f}%)")

fig = make_subplots(specs=[[{"secondary_y": True}]])
# 2 zonas claras: verde em baixo (sobrevendido), vermelho em cima (sobrecomprado)
fig.add_shape(type="rect", xref="paper", x0=0, x1=1, yref="y",
              y0=OVERBOUGHT_MIN, y1=100, fillcolor="#A93226", opacity=0.33,
              line_width=0, layer="below")
fig.add_shape(type="rect", xref="paper", x0=0, x1=1, yref="y",
              y0=0, y1=OVERSOLD_MAX, fillcolor="#1E8449", opacity=0.33,
              line_width=0, layer="below")
fig.add_hline(y=OVERBOUGHT_MIN, line=dict(color="#CB4335", width=1.3, dash="dash"),
              secondary_y=False)
fig.add_hline(y=OVERSOLD_MAX, line=dict(color="#27AE60", width=1.3, dash="dash"),
              secondary_y=False)
fig.add_annotation(xref="paper", x=0.012, y=93, yref="y", xanchor="left",
                   text="SOBRECOMPRADO — realizar", showarrow=False,
                   font=dict(color="#F1948A", size=12))
fig.add_annotation(xref="paper", x=0.012, y=7, yref="y", xanchor="left",
                   text="SOBREVENDIDO — acumular", showarrow=False,
                   font=dict(color="#7DCEA0", size=12))

fig.add_trace(go.Scatter(x=dates, y=val.composite, name="Valorização (0–100)",
                         line=dict(color="#7F77DD", width=2.5)), secondary_y=False)
fig.add_trace(go.Scatter(x=dates, y=prices, name="Preço BTC (log)",
                         line=dict(color="#BA7517", width=1.3, dash="dash")), secondary_y=True)
fig.add_trace(go.Scatter(x=[dates[i]], y=[score], name="Agora", mode="markers",
                         marker=dict(color="#26215C", size=11, line=dict(color="white", width=1)),
                         showlegend=False), secondary_y=False)
for hd, _r in ve.HALVINGS:
    if dates[0] <= hd <= dates[-1]:
        fig.add_vline(x=hd, line=dict(color="rgba(128,128,128,0.5)", width=1, dash="dot"))

fig.update_yaxes(title_text="Valorização (0–100)", range=[0, 100], secondary_y=False)
fig.update_yaxes(title_text="Preço BTC (log, USD)", type="log", secondary_y=True,
                 showgrid=False, tickvals=[100, 1000, 10000, 100000],
                 ticktext=["100", "1k", "10k", "100k"], minor=dict(ticks="", showgrid=False))
fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10),
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
                  hovermode="x unified", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Decomposição — percentil de cada primitiva (hoje)")
names = {
    "trend_deviation": "Desvio lei de potência",
    "long_ma_ratio": "Rácio MA longa",
    "drawdown": "Drawdown",
    "momentum": "Momentum",
    "issuance_value": "Valor de emissão",
    "ma_spread": "Spread MA (topo)",
    "mvrv": "MVRV (on-chain)",
}
rows = {names[k]: round(float(val.primitives_pct[k][i]) * 100) for k in names
        if k in val.primitives_pct and not np.isnan(val.primitives_pct[k][i])}
bar = go.Figure(go.Bar(x=list(rows.keys()), y=list(rows.values()), marker_color="#7F77DD"))
bar.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                  yaxis=dict(title="Percentil (0–100)", range=[0, 100]),
                  template="plotly_white")
st.plotly_chart(bar, use_container_width=True)

st.caption("Pesos iguais + normalização com decaimento (2 anos) + suavização — "
           "anti-overfit. NÃO é aconselhamento financeiro.")

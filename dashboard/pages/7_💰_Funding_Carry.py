"""
💰 Funding Carry Monitor
========================
Monitoriza a funding rate dos perpetuals (BTC/ETH/SOL) e mostra o YIELD
anualizado da estratégia delta-neutra (long spot + short perp = cash-and-carry).

NÃO executa nada — é só um monitor que te diz QUANDO o carry vale a pena.
Dados grátis: OKX (principal) → Bybit (fallback). Sem chave.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Funding Carry Monitor", page_icon="💰",
                   layout="wide", initial_sidebar_state="expanded")

# Paleta (igual ao resto do dashboard)
C_GREEN = "#26A69A"
C_RED = "#EF5350"
C_NEUTRAL = "#78909C"
C_GREEN_SOFT = "#4DB6AC"
C_LINE = "#FFFFFF"

ASSETS = {
    "BTC": {"okx": "BTC-USDT-SWAP", "bybit": "BTCUSDT"},
    "ETH": {"okx": "ETH-USDT-SWAP", "bybit": "ETHUSDT"},
    "SOL": {"okx": "SOL-USDT-SWAP", "bybit": "SOLUSDT"},
}
ANN = 3 * 365          # funding de 8h → períodos por ano
HIST_N = 90            # ~30 dias de histórico (3 períodos/dia)


def _pack(rate: float, hist: list, venue: str) -> dict:
    ann = rate * ANN * 100.0
    vals = [v for _, v in hist]
    # Carry recente ROBUSTO: mediana dos últimos 7 dias (21 períodos de 8h).
    # Resolve o ruído do print instantâneo (F1) e é robusto a spikes (F2).
    med7 = (float(np.median(vals[-21:])) if len(vals) >= 21
            else (float(np.median(vals)) if vals else ann))
    # Ganho acumulado da posição delta-neutra na janela = verdade REALIZADA.
    cum = float(np.sum([v / ANN for v in vals])) if vals else 0.0
    return {"ok": True, "venue": venue, "per8h": rate * 100.0,
            "ann": ann, "med7": med7, "cum": cum, "hist": hist}


def _demo(asset: str) -> dict:
    base = {"BTC": 0.00010, "ETH": 0.00008, "SOL": 0.00013}[asset]
    now = datetime.utcnow()
    rng = np.random.default_rng(abs(hash(asset)) % 2**32)
    hist = []
    for k in range(HIST_N):
        t = now - timedelta(hours=8 * (HIST_N - k))
        r = max(-0.0005, base + rng.normal(0, 0.00006))
        hist.append((t, r * ANN * 100.0))
    d = _pack(base, hist, "Demo (sem rede)")
    d["ok"] = False
    return d


@st.cache_data(ttl=900)
def fetch_funding(asset: str) -> dict:
    cfg = ASSETS[asset]
    # 1) OKX
    try:
        cur = requests.get("https://www.okx.com/api/v5/public/funding-rate",
                           params={"instId": cfg["okx"]}, timeout=12).json()
        rate = float(cur["data"][0]["fundingRate"])
        h = requests.get("https://www.okx.com/api/v5/public/funding-rate-history",
                         params={"instId": cfg["okx"], "limit": HIST_N}, timeout=12).json()
        hist = [(datetime.utcfromtimestamp(int(r["fundingTime"]) / 1000),
                 float(r["fundingRate"]) * ANN * 100.0) for r in h["data"]][::-1]
        if hist:
            return _pack(rate, hist, "OKX")
    except Exception:  # noqa: BLE001
        pass
    # 2) Bybit
    try:
        cur = requests.get("https://api.bybit.com/v5/market/tickers",
                           params={"category": "linear", "symbol": cfg["bybit"]}, timeout=12).json()
        rate = float(cur["result"]["list"][0]["fundingRate"])
        h = requests.get("https://api.bybit.com/v5/market/funding/history",
                         params={"category": "linear", "symbol": cfg["bybit"], "limit": HIST_N},
                         timeout=12).json()
        hist = [(datetime.utcfromtimestamp(int(r["fundingRateTimestamp"]) / 1000),
                 float(r["fundingRate"]) * ANN * 100.0) for r in h["result"]["list"]][::-1]
        if hist:
            return _pack(rate, hist, "Bybit")
    except Exception:  # noqa: BLE001
        pass
    # 3) Demo
    return _demo(asset)


def verdict(ann: float) -> tuple[str, str]:
    if ann < 0:
        return "Negativa — sem carry", C_RED
    if ann < 5:
        return "Fraco", C_NEUTRAL
    if ann < 15:
        return "Razoável", C_GREEN_SOFT
    if ann < 30:
        return "Bom", C_GREEN
    return "Excelente (euforia)", C_GREEN


st.title("💰 Funding Carry Monitor")
st.caption("Yield anualizado da estratégia delta-neutra (long spot + short perp). "
           "Não executa nada — diz-te QUANDO o carry vale a pena. Fonte: OKX → Bybit.")

data = {a: fetch_funding(a) for a in ASSETS}
live = [a for a, d in data.items() if d["ok"]]
src = "ao vivo" if live else "Demo (sem rede — vê o link no Streamlit Cloud)"
best = max(data, key=lambda a: data[a]["med7"])
bmed = data[best]["med7"]
bverd, bcol = verdict(bmed)
st.markdown(
    f'<div style="background:{bcol}22;border-left:5px solid {bcol};border-radius:10px;'
    f'padding:12px 18px;font-size:16px;color:#eaecef;margin-bottom:14px;">'
    f'🎯 Melhor carry recente: <b style="color:{bcol};">{best}</b> a '
    f'<b>{bmed:.1f}%/ano</b> (mediana 7d) — {bverd} &nbsp;'
    f'<span style="font-size:12px;color:#7d8595;">dados {src}</span></div>',
    unsafe_allow_html=True)

cols = st.columns(3)
for col, asset in zip(cols, ASSETS):
    d = data[asset]
    vlabel, vcol = verdict(d["med7"])      # veredito pelo carry recente robusto
    with col:
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.04);border-left:4px solid {vcol};'
            f'border-radius:10px;padding:16px;">'
            f'<div style="font-size:13px;color:#9aa0a6;letter-spacing:.5px;">{asset} · {d["venue"]}</div>'
            f'<div style="font-size:34px;font-weight:700;color:{vcol};line-height:1.1;'
            f'margin:4px 0;">{d["med7"]:+.1f}%<span style="font-size:14px;color:#7d8595;">/ano</span></div>'
            f'<div style="font-size:11px;color:#7d8595;margin-bottom:6px;">carry recente · mediana 7d</div>'
            f'<div style="font-size:13px;color:#cfd3da;">agora: <b>{d["ann"]:+.1f}%/ano</b> '
            f'· 8h {d["per8h"]:+.4f}%</div>'
            f'<div style="font-size:13px;color:#cfd3da;">ganho acum. 30d (realizado): '
            f'<b style="color:{C_GREEN if d["cum"] >= 0 else C_RED};">{d["cum"]:+.2f}%</b></div>'
            f'<div style="margin-top:8px;font-size:13px;font-weight:600;color:{vcol};">{vlabel}</div>'
            f'</div>', unsafe_allow_html=True)

st.divider()
st.subheader("Histórico da funding (anualizada)")
fig = go.Figure()
palette = {"BTC": C_LINE, "ETH": C_GREEN, "SOL": C_RED}
for asset in ASSETS:
    h = data[asset]["hist"]
    if h:
        fig.add_trace(go.Scatter(x=[t for t, _ in h], y=[v for _, v in h],
                                 mode="lines", name=asset,
                                 line=dict(color=palette[asset], width=2)))
fig.add_hline(y=0, line=dict(color="rgba(148,163,184,0.4)", width=1, dash="dash"))
fig.add_hline(y=15, line=dict(color=C_GREEN_SOFT, width=1, dash="dot"))
fig.add_annotation(xref="paper", x=0.01, y=15, yref="y", xanchor="left",
                   text="carry interessante (>15%/ano)", showarrow=False,
                   font=dict(color=C_GREEN_SOFT, size=11))
fig.update_yaxes(title_text="Funding anualizada (%)", gridcolor="rgba(148,163,184,0.10)")
fig.update_xaxes(gridcolor="rgba(148,163,184,0.06)")
fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                  template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                  plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, use_container_width=True, theme=None)

st.subheader("Ganho acumulado — quanto terias ganho (delta-neutro)")
figc = go.Figure()
for asset in ASSETS:
    h = data[asset]["hist"]
    if h:
        cum = np.cumsum([v / ANN for _, v in h])
        figc.add_trace(go.Scatter(x=[t for t, _ in h], y=cum, mode="lines", name=asset,
                                  line=dict(color=palette[asset], width=2.2)))
figc.add_hline(y=0, line=dict(color="rgba(148,163,184,0.4)", width=1, dash="dash"))
figc.update_yaxes(title_text="Funding acumulada (%)", gridcolor="rgba(148,163,184,0.10)")
figc.update_xaxes(gridcolor="rgba(148,163,184,0.06)")
figc.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                   template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.1))
st.plotly_chart(figc, use_container_width=True, theme=None)
st.caption("Soma da funding recebida ao longo da janela, mantendo a posição delta-neutra "
           "(antes de fees). É o 'dinheiro real' que a estratégia teria gerado.")

with st.expander("ℹ️ Como funciona o cash-and-carry (e os riscos)"):
    st.markdown(
        "- **A estratégia:** comprar 1 BTC **spot** + vender 1 BTC em **perpetual** "
        "→ exposição ao preço = 0 (neutro). Recebes a **funding** que os longs pagam.\n"
        "- **O yield:** funding por 8h × 3 × 365. Positiva no agregado histórico "
        "porque, em bull, o retalho amontoa-se em longs alavancados e paga para os manter.\n"
        "- **Riscos:** funding pode virar **negativa** (desmontar); **liquidação** do "
        "short se o preço dispara (usar baixa alavancagem + spot como colateral); "
        "**contraparte** da exchange; **fees** têm de ser < funding.\n"
        "- **Não é passivo** — exige capital e gestão ativa.")

st.caption("Sinergia: quando o valuation diz 'euforia/caro' (mau para acumular), a "
           "funding está no máximo → roda o capital realizado para carry delta-neutro. "
           "NÃO é aconselhamento financeiro.")

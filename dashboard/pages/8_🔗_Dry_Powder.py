"""
🔗 Dry Powder & Liquidez On-chain
=================================
Agregados ON-CHAIN limpos e GRÁTIS (sem DIY whale-tracking ruidoso):
  • Oferta total de stablecoins + variação 30/90d — "dry powder" (combustível).
  • Repartição USDT vs USDC.
  • TVL total de DeFi — apetite por risco.

Fonte: DefiLlama (grátis, sem chave). Edge de CONTEXTO/regime, não de timing.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Dry Powder & Liquidez On-chain", page_icon="🔗",
                   layout="wide", initial_sidebar_state="expanded")

C_GREEN = "#26A69A"
C_RED = "#EF5350"
C_NEUTRAL = "#78909C"
C_GREEN_SOFT = "#4DB6AC"
C_LINE = "#FFFFFF"


def _usd(row: dict):
    tot = row.get("totalCirculatingUSD") or row.get("totalCirculating") or {}
    v = tot.get("peggedUSD") if isinstance(tot, dict) else None
    return float(v) if isinstance(v, (int, float)) else None


def _pct(series: list, days: int) -> float | None:
    if len(series) < days + 1:
        return None
    now_v, past_v = series[-1][1], series[-(days + 1)][1]
    return (now_v / past_v - 1.0) * 100.0 if past_v else None


def _demo_series(base: float, n: int = 400, drift: float = 0.0006) -> list:
    now = datetime.utcnow()
    rng = np.random.default_rng(int(base) % 2**32)
    out, v = [], base * 0.8
    for k in range(n):
        v *= (1 + drift + rng.normal(0, 0.004))
        out.append((now - timedelta(days=(n - k)), v))
    return out


@st.cache_data(ttl=3600)
def load_data():
    live = True
    # Stablecoins — série histórica
    try:
        r = requests.get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=15).json()
        stables = [(datetime.utcfromtimestamp(int(x["date"])), _usd(x)) for x in r]
        stables = [(d, v) for d, v in stables if v]
        if len(stables) < 120:
            raise ValueError("série curta")
    except Exception:  # noqa: BLE001
        stables, live = _demo_series(2.4e11), False
    # Repartição USDT/USDC
    split = {}
    try:
        r = requests.get("https://stablecoins.llama.fi/stablecoins?includePrices=false",
                         timeout=15).json()
        for a in r.get("peggedAssets", []):
            sym = a.get("symbol", "")
            if sym in ("USDT", "USDC"):
                c = a.get("circulating", {})
                if isinstance(c.get("peggedUSD"), (int, float)):
                    split[sym] = float(c["peggedUSD"])
    except Exception:  # noqa: BLE001
        split = {}
    # TVL DeFi
    try:
        r = requests.get("https://api.llama.fi/v2/historicalChainTvl", timeout=15).json()
        tvl = [(datetime.utcfromtimestamp(int(x["date"])), float(x["tvl"])) for x in r]
        if len(tvl) < 120:
            raise ValueError("série curta")
    except Exception:  # noqa: BLE001
        tvl, live = _demo_series(1.0e11), False
    return stables, split, tvl, live


def _b(v: float) -> str:
    return f"${v / 1e9:,.1f}B"


def card(label: str, value: str, accent: str, sub: str = "") -> str:
    s = f'<div style="font-size:12px;color:#cfd3da;margin-top:4px;">{sub}</div>' if sub else ""
    return (f'<div style="background:rgba(255,255,255,0.04);border-left:4px solid {accent};'
            f'border-radius:10px;padding:14px 16px;height:100%;">'
            f'<div style="font-size:11px;color:#9aa0a6;text-transform:uppercase;'
            f'letter-spacing:.6px;">{label}</div>'
            f'<div style="font-size:23px;font-weight:700;color:#eaecef;margin-top:4px;'
            f'line-height:1.15;">{value}</div>{s}</div>')


stables, split, tvl, live = load_data()
chg30 = _pct(stables, 30)
chg90 = _pct(stables, 90)
tvl30 = _pct(tvl, 30)

if chg30 is None:
    regime, rcol = "Dados insuficientes", C_NEUTRAL
elif chg30 > 2:
    regime, rcol = "Expansão — liquidez a entrar", C_GREEN
elif chg30 < -2:
    regime, rcol = "Contração — liquidez a sair", C_RED
else:
    regime, rcol = "Estável", C_NEUTRAL

st.title("🔗 Dry Powder & Liquidez On-chain")
st.caption("Combustível do mercado: oferta de stablecoins + TVL DeFi. Agregados "
           "limpos e grátis (DefiLlama). Indicador de regime/contexto, não de timing.")

src = "ao vivo" if live else "Demo (sem rede)"
st.markdown(
    f'<div style="background:{rcol}22;border-left:5px solid {rcol};border-radius:10px;'
    f'padding:12px 18px;font-size:16px;color:#eaecef;margin-bottom:14px;">'
    f'🛢️ Regime de liquidez: <b style="color:{rcol};">{regime}</b> &nbsp;'
    f'<span style="font-size:12px;color:#7d8595;">dados {src}</span></div>',
    unsafe_allow_html=True)

c30 = C_GREEN if (chg30 or 0) > 0 else C_RED
c90 = C_GREEN if (chg90 or 0) > 0 else C_RED
ct = C_GREEN if (tvl30 or 0) > 0 else C_RED
usdt_share = ""
if split.get("USDT") and split.get("USDC"):
    tot = split["USDT"] + split["USDC"]
    usdt_share = f'USDT {split["USDT"] / tot * 100:.0f}% · USDC {split["USDC"] / tot * 100:.0f}%'
st.markdown(
    '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">'
    + card("Stablecoins (total)", _b(stables[-1][1]), C_LINE, usdt_share)
    + card("Variação 30d", f"{chg30:+.1f}%" if chg30 is not None else "n/d", c30, "dry powder")
    + card("Variação 90d", f"{chg90:+.1f}%" if chg90 is not None else "n/d", c90, "tendência")
    + card("TVL DeFi", _b(tvl[-1][1]), ct,
           f"{tvl30:+.1f}% em 30d" if tvl30 is not None else "")
    + '</div>', unsafe_allow_html=True)

st.divider()
st.subheader("Oferta de stablecoins (dry powder)")
yr = stables[-400:]
figs = go.Figure()
figs.add_trace(go.Scatter(x=[d for d, _ in yr], y=[v / 1e9 for _, v in yr], mode="lines",
                          line=dict(color=C_LINE, width=2.4), fill="tozeroy",
                          fillcolor="rgba(255,255,255,0.04)", showlegend=False))
figs.update_yaxes(title_text="Stablecoins (B USD)", gridcolor="rgba(148,163,184,0.10)")
figs.update_xaxes(gridcolor="rgba(148,163,184,0.06)")
figs.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                   template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(figs, use_container_width=True, theme=None)

st.subheader("TVL total de DeFi (apetite por risco)")
yt = tvl[-400:]
figt = go.Figure()
figt.add_trace(go.Scatter(x=[d for d, _ in yt], y=[v / 1e9 for _, v in yt], mode="lines",
                          line=dict(color=C_GREEN, width=2.4), fill="tozeroy",
                          fillcolor="rgba(38,166,154,0.08)", showlegend=False))
figt.update_yaxes(title_text="TVL DeFi (B USD)", gridcolor="rgba(148,163,184,0.10)")
figt.update_xaxes(gridcolor="rgba(148,163,184,0.06)")
figt.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                   template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(figt, use_container_width=True, theme=None)

st.caption("Honesto: edge de contexto/regime (combustível), não de timing. Stablecoins a "
           "expandir = capital pronto a entrar; a contrair = capital a sair. "
           "NÃO é aconselhamento financeiro.")

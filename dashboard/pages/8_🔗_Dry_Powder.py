"""
🔗 Dry Powder & Liquidez On-chain
=================================
Agregados ON-CHAIN limpos e GRÁTIS (sem DIY whale-tracking ruidoso):
  • Oferta total de stablecoins + variação 30/90d — "dry powder" (combustível).
  • SSR (Stablecoin Supply Ratio) = BTC mcap ÷ stablecoins — dry powder vs preço.
  • TVL total de DeFi — apetite por risco. Divergência stables ↔ DeFi.

Fonte: DefiLlama + yfinance (grátis, sem chave). Edge de CONTEXTO/regime, não timing.
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
SUPPLY = 19_900_000  # BTC em circulação (aprox.; o SSR relativo é dominado pelo preço)


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
    try:
        r = requests.get("https://stablecoins.llama.fi/stablecoincharts/all", timeout=15).json()
        stables = [(datetime.utcfromtimestamp(int(x["date"])), _usd(x)) for x in r]
        stables = [(d, v) for d, v in stables if v]
        if len(stables) < 120:
            raise ValueError("série curta")
    except Exception:  # noqa: BLE001
        stables, live = _demo_series(2.4e11), False
    split = {}
    try:
        r = requests.get("https://stablecoins.llama.fi/stablecoins?includePrices=false",
                         timeout=15).json()
        for a in r.get("peggedAssets", []):
            if a.get("symbol") in ("USDT", "USDC"):
                c = a.get("circulating", {})
                if isinstance(c.get("peggedUSD"), (int, float)):
                    split[a["symbol"]] = float(c["peggedUSD"])
    except Exception:  # noqa: BLE001
        split = {}
    try:
        r = requests.get("https://api.llama.fi/v2/historicalChainTvl", timeout=15).json()
        tvl = [(datetime.utcfromtimestamp(int(x["date"])), float(x["tvl"])) for x in r]
        if len(tvl) < 120:
            raise ValueError("série curta")
    except Exception:  # noqa: BLE001
        tvl, live = _demo_series(1.0e11), False
    # BTC (para o SSR) — preço diário via yfinance; supply ~constante.
    btc = {}
    try:
        import yfinance as yf
        df = yf.download("BTC-USD", period="2y", interval="1d", progress=False, auto_adjust=True)
        close = df["Close"]
        if hasattr(close, "columns"):
            close = close.iloc[:, 0]
        btc = {d.date(): float(v) for d, v in close.items() if v == v}
    except Exception:  # noqa: BLE001
        btc = {}
    return stables, split, tvl, btc, live


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


def classify_regime(chg30, chg90, cur, hi90):
    """Regime guiado pela TENDÊNCIA de 90d (estrutural), não por um wobble de 30d."""
    near_high = bool(hi90) and cur >= 0.97 * hi90
    if chg90 is None:
        return "Dados insuficientes", C_NEUTRAL
    if chg90 > 3:
        return "Expansão — liquidez a entrar", C_GREEN
    if chg90 < -3:
        return "Contração — liquidez a sair", C_RED
    if near_high:
        return "Consolidação no topo — combustível elevado", C_GREEN_SOFT
    if (chg30 or 0) < -2:
        return "Pullback ligeiro", C_NEUTRAL
    return "Estável", C_NEUTRAL


def divergence(st90, tvl90):
    if st90 is None or tvl90 is None:
        return None, C_NEUTRAL
    if st90 > -3 and tvl90 < -8:
        return ("Capital parado em stables mas a sair de DeFi — risk-off interno, "
                "dry powder a acumular (cautela).", C_NEUTRAL)
    if st90 > 3 and tvl90 > 3:
        return ("Risk-on: capital a entrar e a ser aplicado em DeFi.", C_GREEN)
    if st90 < -3 and tvl90 < -3:
        return ("Risk-off geral: capital a sair de stables e de DeFi.", C_RED)
    if st90 < -3 and tvl90 > 3:
        return ("Capital a sair de stables para DeFi (a ser aplicado).", C_GREEN_SOFT)
    return ("Sem divergência clara entre stables e DeFi.", C_NEUTRAL)


stables, split, tvl, btc, live = load_data()
chg30, chg90 = _pct(stables, 30), _pct(stables, 90)
tvl30, tvl90 = _pct(tvl, 30), _pct(tvl, 90)
hi90 = max(v for _, v in stables[-90:]) if len(stables) >= 90 else None
regime, rcol = classify_regime(chg30, chg90, stables[-1][1], hi90)

# SSR — Stablecoin Supply Ratio (BTC mcap ÷ stablecoins). Baixo = muito dry powder.
ssr_series = []
for d, sv in stables[-400:]:
    p = btc.get(d.date())
    if p and sv:
        ssr_series.append((d, (p * SUPPLY) / sv))
ssr_cur = ssr_series[-1][1] if ssr_series else None
ssr_pct = None
if ssr_series:
    vals = [v for _, v in ssr_series]
    ssr_pct = sum(1 for v in vals if v <= ssr_cur) / len(vals) * 100
if ssr_pct is None:
    ssr_txt, ssr_col = "n/d", C_NEUTRAL
elif ssr_pct <= 33:
    ssr_txt, ssr_col = "baixo — muito dry powder", C_GREEN
elif ssr_pct >= 67:
    ssr_txt, ssr_col = "alto — pouco dry powder", C_RED
else:
    ssr_txt, ssr_col = "médio", C_NEUTRAL

st.title("🔗 Dry Powder & Liquidez On-chain")
st.caption("Combustível do mercado: stablecoins, SSR (vs preço) e TVL DeFi. Agregados "
           "limpos e grátis. Indicador de regime/contexto, não de timing.")

src = "ao vivo" if live else "Demo (sem rede)"
st.markdown(
    f'<div style="background:{rcol}22;border-left:5px solid {rcol};border-radius:10px;'
    f'padding:12px 18px;font-size:16px;color:#eaecef;margin-bottom:10px;">'
    f'🛢️ Regime de liquidez: <b style="color:{rcol};">{regime}</b> &nbsp;'
    f'<span style="font-size:12px;color:#7d8595;">dados {src}</span></div>',
    unsafe_allow_html=True)

dv_txt, dv_col = divergence(chg90, tvl90)
if dv_txt:
    st.markdown(
        f'<div style="background:{dv_col}18;border-left:4px solid {dv_col};border-radius:10px;'
        f'padding:10px 16px;font-size:14px;color:#cfd3da;margin-bottom:14px;">'
        f'🔀 <b>Stables ↔ DeFi:</b> {dv_txt}</div>', unsafe_allow_html=True)

c30 = C_GREEN if (chg30 or 0) > 0 else C_RED
c90 = C_GREEN if (chg90 or 0) > 0 else C_RED
ct = C_GREEN if (tvl30 or 0) > 0 else C_RED
usdt_share = ""
if split.get("USDT") and split.get("USDC"):
    tot = split["USDT"] + split["USDC"]
    usdt_share = f'USDT {split["USDT"] / tot * 100:.0f}% · USDC {split["USDC"] / tot * 100:.0f}%'
st.markdown(
    '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:10px;">'
    + card("Stablecoins (total)", _b(stables[-1][1]), C_LINE, usdt_share)
    + card("Variação 30d", f"{chg30:+.1f}%" if chg30 is not None else "n/d", c30, "dry powder")
    + card("Variação 90d", f"{chg90:+.1f}%" if chg90 is not None else "n/d", c90, "tendência")
    + card("SSR", f"{ssr_cur:.1f}" if ssr_cur is not None else "n/d", ssr_col,
           f"percentil {ssr_pct:.0f} · {ssr_txt}" if ssr_pct is not None else "")
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
figs.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                   template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(figs, use_container_width=True, theme=None)

if ssr_series:
    st.subheader("SSR — dry powder face ao preço (mais baixo = mais combustível)")
    figr = go.Figure()
    figr.add_trace(go.Scatter(x=[d for d, _ in ssr_series], y=[v for _, v in ssr_series],
                              mode="lines", line=dict(color=C_GREEN_SOFT, width=2.2),
                              showlegend=False))
    figr.update_yaxes(title_text="SSR (BTC mcap ÷ stables)", gridcolor="rgba(148,163,184,0.10)")
    figr.update_xaxes(gridcolor="rgba(148,163,184,0.06)")
    figr.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                       template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figr, use_container_width=True, theme=None)

st.subheader("TVL total de DeFi (apetite por risco)")
yt = tvl[-400:]
figt = go.Figure()
figt.add_trace(go.Scatter(x=[d for d, _ in yt], y=[v / 1e9 for _, v in yt], mode="lines",
                          line=dict(color=C_GREEN, width=2.4), fill="tozeroy",
                          fillcolor="rgba(38,166,154,0.08)", showlegend=False))
figt.update_yaxes(title_text="TVL DeFi (B USD)", gridcolor="rgba(148,163,184,0.10)")
figt.update_xaxes(gridcolor="rgba(148,163,184,0.06)")
figt.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                   template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                   plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(figt, use_container_width=True, theme=None)

st.caption("SSR baixo = muitos stablecoins face ao preço do BTC = poder de compra latente. "
           "Regime guiado pela tendência de 90d (não por oscilações de 30d). "
           "NÃO é aconselhamento financeiro.")

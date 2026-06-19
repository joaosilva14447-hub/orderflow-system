"""
valuation_engine.py — SDCA Valuation Engine (v1, PROPRIETÁRIO e auto-contido)

Objetivo: um oscilador 0–100 de valorização de LONGO PRAZO que sinaliza zonas
de ciclo (oversold ↔ overbought) para SDCA. Pensado para apanhar extremos de
ciclo (fundo pós-COVID, topo 2021, fundo 2022), ignorando ruído intermédio.

NADA é importado pré-cozinhado. Tudo é nosso, em 4 camadas:
  0. Dados crus ....... preço diário (CoinGecko) + supply/emissão determinística.
  1. Primitivas ....... 5 sinais que NÓS definimos (matemática nossa).
  2. Normalização ..... percentil em janela EXPANSÍVEL (adaptativo, sem lookahead;
                        vence os "diminishing returns" de cada ciclo).
  3. Composto ......... média ponderada (os nossos pesos) → score 0–100 + zona.

Orientação: TODAS as primitivas apontam no mesmo sentido — valor alto = "caro"
(overbought). Logo o composto: 0 = deep value (acumular), 100 = euforia (realizar).

NOTA: este ficheiro é a v1 para REVISÃO. Ainda não está ligado ao Discord.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np

log = logging.getLogger("valuation")

# ─────────────────────────────────────────────────────────────────────────────
# Camada 0 — supply/emissão determinística do BTC (sem qualquer fonte externa)
# ─────────────────────────────────────────────────────────────────────────────
GENESIS = date(2009, 1, 3)
BLOCKS_PER_DAY = 144  # ~10 min/bloco
# (data de início da época, recompensa por bloco em BTC)
HALVINGS = [
    (date(2009, 1, 3), 50.0),
    (date(2012, 11, 28), 25.0),
    (date(2016, 7, 9), 12.5),
    (date(2020, 5, 11), 6.25),
    (date(2024, 4, 20), 3.125),
    (date(2028, 4, 1), 1.5625),  # estimativa do próximo (aproximada)
]


def _reward_at(d: date) -> float:
    r = HALVINGS[0][1]
    for start, reward in HALVINGS:
        if d >= start:
            r = reward
    return r


def daily_issuance_btc(d: date) -> float:
    """Nova emissão de BTC nesse dia (recompensa × blocos/dia). Determinística."""
    return _reward_at(d) * BLOCKS_PER_DAY


# ─────────────────────────────────────────────────────────────────────────────
# Camada 0 — preço (CoinGecko, grátis, sem chave). Resample para semanal.
# ─────────────────────────────────────────────────────────────────────────────
def fetch_btc_history() -> tuple[list[date], np.ndarray]:
    """
    Histórico diário máximo do BTC (CoinGecko). AUTO-CONTIDO: usa só requests
    (com pequenas re-tentativas), para o módulo ser portável a qualquer projeto.
    Sem 'interval' (parâmetro pago): para days=max a CoinGecko já devolve diário.
    """
    import time
    import requests

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "max"}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; SDCA-Valuation/1.0)"}
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            rows = resp.json()["prices"]  # [[ms, price], ...]
            dates = [datetime.fromtimestamp(ms / 1000, tz=timezone.utc).date()
                     for ms, _ in rows]
            prices = np.array([p for _, p in rows], dtype=float)
            return dates, prices
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    raise last_exc if last_exc else RuntimeError("CoinGecko sem resposta")


def resample_weekly(dates: list[date], prices: np.ndarray) -> tuple[list[date], np.ndarray]:
    """Fica com o último preço de cada semana (reduz ruído + custo)."""
    buckets: dict[tuple[int, int], tuple[date, float]] = {}
    for d, p in zip(dates, prices):
        key = d.isocalendar()[:2]  # (ano ISO, semana ISO)
        buckets[key] = (d, p)      # último da semana vence
    items = [buckets[k] for k in sorted(buckets)]
    return [d for d, _ in items], np.array([p for _, p in items], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Camada 1 — as NOSSAS primitivas (todas: alto = caro/overbought)
# ─────────────────────────────────────────────────────────────────────────────
MIN_HISTORY = 12  # nº mínimo de pontos antes de produzir leitura


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = np.mean(x[lo:i + 1])
    return out


def prim_trend_deviation(dates: list[date], prices: np.ndarray) -> np.ndarray:
    """
    Desvio face à NOSSA lei de potência: ajustamos log(preço) ~ log(idade) em
    janela expansível e medimos o resíduo. Acima da tendência = caro.
    """
    logp = np.log(prices)
    age = np.array([max((d - GENESIS).days, 1) for d in dates], dtype=float)
    t = np.log(age)
    out = np.full(len(prices), np.nan)
    for i in range(MIN_HISTORY, len(prices)):
        a, b = np.polyfit(t[:i + 1], logp[:i + 1], 1)
        out[i] = logp[i] - (a * t[i] + b)
    return out


def prim_long_ma_ratio(prices: np.ndarray, w: int) -> np.ndarray:
    """Preço / média longa (~1 ano). Muito acima da média = caro."""
    return prices / _rolling_mean(prices, w)


def prim_drawdown(prices: np.ndarray) -> np.ndarray:
    """Preço / máximo histórico até à data − 1. Perto de 0 (perto do ATH) = caro."""
    run_max = np.maximum.accumulate(prices)
    return prices / run_max - 1.0


def prim_momentum(prices: np.ndarray, w: int) -> np.ndarray:
    """Variação a ~1 ano (rate-of-change). Subida forte = caro/esticado."""
    out = np.full(len(prices), np.nan)
    for i in range(w, len(prices)):
        if prices[i - w] > 0:
            out[i] = prices[i] / prices[i - w] - 1.0
    return out


def prim_issuance_value(dates: list[date], prices: np.ndarray, w: int) -> np.ndarray:
    """
    Valor da emissão diária (emissão × preço) face à sua média a ~1 ano
    (estilo "receita de mineração"). Muito acima da média = ciclo esticado.
    Usa só preço + supply determinístico — totalmente nosso.
    """
    iss_usd = np.array([daily_issuance_btc(d) for d in dates]) * prices
    return iss_usd / _rolling_mean(iss_usd, w)


def prim_ma_spread(prices: np.ndarray, fast: int, slow: int) -> np.ndarray:
    """
    Detetor de TOPO: distância entre uma média curta e uma longa
    (MA_rápida / MA_lenta − 1). Quando a rápida dispara acima da lenta, o preço
    está esticado/parabólico (blow-off). Geral (não calibrado a nenhum evento).
    """
    return _rolling_mean(prices, fast) / _rolling_mean(prices, slow) - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Camada 2 — normalização: percentil em janela expansível (0..1)
# ─────────────────────────────────────────────────────────────────────────────
def pct_rank_expanding(x: np.ndarray) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        window = x[:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) >= MIN_HISTORY and not np.isnan(x[i]):
            out[i] = float(np.mean(valid <= x[i]))
    return out


def pct_rank_decayed(x: np.ndarray, halflife: float) -> np.ndarray:
    """
    Percentil em janela expansível mas com DECAIMENTO temporal: a história
    recente pesa mais (peso = 0.5 ** (idade / halflife)). Isto vence os
    'diminishing returns' (a mania de 2013/2017 não esmaga os ciclos seguintes)
    sem afinar nada aos dados — há um único hiperparâmetro (half-life), fixado
    num valor redondo e justificado (~2 anos), NÃO otimizado ao backtest.
    """
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        if np.isnan(x[i]):
            continue
        idx = np.arange(i + 1)
        w = 0.5 ** ((i - idx) / halflife)
        hist = x[:i + 1]
        valid = ~np.isnan(hist)
        if int(valid.sum()) >= MIN_HISTORY:
            num = float(np.sum(w[valid] * (hist[valid] <= x[i])))
            den = float(np.sum(w[valid]))
            out[i] = num / den if den else np.nan
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Camada 3 — composto (os NOSSOS pesos)
# ─────────────────────────────────────────────────────────────────────────────
# Pesos IGUAIS de propósito (1/6 cada) — anti-overfit: não otimizamos pesos
# aos poucos ciclos que temos. Cada primitiva contribui o mesmo.
WEIGHTS = {
    "trend_deviation": 1 / 6,
    "long_ma_ratio": 1 / 6,
    "drawdown": 1 / 6,
    "momentum": 1 / 6,
    "issuance_value": 1 / 6,
    "ma_spread": 1 / 6,
}

ZONES = [
    (15, "🟢🟢 Deep value", "Acumular agressivo"),
    (35, "🟢 Value", "Acumular"),
    (65, "😐 Fair", "DCA neutro"),
    (85, "🟠 Elevated", "Reduzir ritmo"),
    (101, "🔴 Euphoria", "Realizar / parar de comprar"),
]


def zone_for(score: float) -> tuple[str, str]:
    for upper, label, action in ZONES:
        if score < upper:
            return label, action
    return ZONES[-1][1], ZONES[-1][2]


@dataclass
class Valuation:
    dates: list[date]
    composite: np.ndarray                  # 0..100 (NaN no aquecimento)
    primitives_pct: dict[str, np.ndarray]  # percentis 0..1 por primitiva
    conviction: np.ndarray                 # 0..1 (concordância das primitivas)


def compute_series(dates: list[date], prices: np.ndarray,
                   periods_per_year: int, halflife_years: float = 2.0) -> Valuation:
    ppy = periods_per_year
    fast = max(2, round(ppy / 3))   # ~4 meses
    slow = max(fast + 1, ppy)       # ~1 ano
    raw = {
        "trend_deviation": prim_trend_deviation(dates, prices),
        "long_ma_ratio": prim_long_ma_ratio(prices, ppy),
        "drawdown": prim_drawdown(prices),
        "momentum": prim_momentum(prices, ppy),
        "issuance_value": prim_issuance_value(dates, prices, ppy),
        "ma_spread": prim_ma_spread(prices, fast, slow),
    }
    halflife = halflife_years * ppy
    pct = {k: pct_rank_decayed(v, halflife) for k, v in raw.items()}

    composite = np.full(len(prices), np.nan)
    conviction = np.full(len(prices), np.nan)
    for i in range(len(prices)):
        vals = [pct[k][i] for k in WEIGHTS if not np.isnan(pct[k][i])]
        if not vals:
            continue
        # Pesos iguais → média simples dos percentis disponíveis.
        composite[i] = 100.0 * float(np.mean(vals))
        # Convicção: quão alinhadas estão as primitivas (baixa dispersão = alta).
        conviction[i] = max(0.0, 1.0 - 2.0 * float(np.std(vals)))
    return Valuation(dates=dates, composite=composite,
                     primitives_pct=pct, conviction=conviction)


# ─────────────────────────────────────────────────────────────────────────────
# Backtest / demonstração (dados aproximados embebidos — ver nota no fundo)
# ─────────────────────────────────────────────────────────────────────────────
# Âncoras mensais APROXIMADAS de fecho do BTC (USD), só para validar a LÓGICA
# offline. O backtest definitivo corre com dados DIÁRIOS reais da CoinGecko.
DEMO_PRICES: dict[str, float] = {
    "2016-01": 380, "2016-04": 430, "2016-07": 650, "2016-10": 700, "2016-12": 960,
    "2017-03": 1080, "2017-06": 2500, "2017-09": 4340, "2017-11": 9900, "2017-12": 13800,
    "2018-03": 6900, "2018-06": 6400, "2018-09": 6600, "2018-12": 3700,
    "2019-03": 4100, "2019-06": 10800, "2019-09": 8300, "2019-12": 7200,
    "2020-02": 8600, "2020-03": 6400, "2020-06": 9100, "2020-09": 10800, "2020-12": 29000,
    "2021-02": 45000, "2021-04": 57800, "2021-06": 35000, "2021-08": 47000,
    "2021-10": 61000, "2021-11": 57000, "2021-12": 46200,
    "2022-02": 43000, "2022-04": 37600, "2022-06": 19900, "2022-08": 20000,
    "2022-10": 20500, "2022-11": 17000, "2022-12": 16500,
    "2023-02": 23100, "2023-06": 30500, "2023-10": 34500, "2023-12": 42300,
    "2024-02": 61000, "2024-03": 71300, "2024-06": 62700, "2024-09": 63300,
    "2024-11": 76000, "2024-12": 93000,
    "2025-03": 84000, "2025-06": 70000, "2025-09": 72000, "2025-12": 68000,
    "2026-03": 64000, "2026-06": 66000,
}

# Zonas que queremos que o sistema acerte (a nossa "verdade").
LABELED = [
    ("2020-03", "Fundo pós-COVID", "deve dar BAIXO (oversold)"),
    ("2021-11", "Topo 2021", "deve dar ALTO (overbought)"),
    ("2022-11", "Fundo 2022 (FTX)", "deve dar BAIXO (oversold)"),
    ("2024-12", "Topo/ATH 2024", "deve dar ALTO (overbought)"),
    ("2026-06", "Atual", "leitura de hoje"),
]


def _demo_series() -> tuple[list[date], np.ndarray]:
    keys = sorted(DEMO_PRICES)
    dates = [date(int(k[:4]), int(k[5:7]), 28) for k in keys]
    prices = np.array([DEMO_PRICES[k] for k in keys], dtype=float)
    return dates, prices


def run_demo() -> None:
    dates, prices = _demo_series()
    val = compute_series(dates, prices, periods_per_year=12)  # demo é mensal
    ym = [f"{d.year}-{d.month:02d}" for d in dates]

    print("=" * 72)
    print("SDCA VALUATION ENGINE v1 — backtest (âncoras mensais aproximadas)")
    print("Pesos:", ", ".join(f"{k} {int(w*100)}%" for k, w in WEIGHTS.items()))
    print("=" * 72)
    for key, nome, esperado in LABELED:
        if key not in ym:
            continue
        i = ym.index(key)
        score = val.composite[i]
        if np.isnan(score):
            print(f"{key}  {nome:<22} score=n/d (sem histórico)")
            continue
        label, action = zone_for(score)
        print(f"\n{key}  {nome}  —  preço ~${prices[i]:,.0f}")
        print(f"   SCORE = {score:5.1f}/100  →  {label}  ({action})")
        print(f"   esperado: {esperado}")
        det = " · ".join(f"{k.split('_')[0]}={val.primitives_pct[k][i]*100:.0f}"
                         for k in WEIGHTS if not np.isnan(val.primitives_pct[k][i]))
        print(f"   percentis: {det}")


def render_chart(save_path: str = "valuation_chart.png", show: bool = False) -> str:
    """
    Desenha o oscilador (0–100) com bandas de zona + o preço (log) e marca os
    extremos de ciclo. Grava um PNG (para GitHub/Discord) e/ou mostra (PyCharm).
    Requer matplotlib (instalar localmente: pip install matplotlib).
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")  # backend sem ecrã (para gravar PNG)
    import matplotlib.pyplot as plt

    dates, prices = _demo_series()
    val = compute_series(dates, prices, periods_per_year=12)
    x = list(range(len(dates)))
    ym = [f"{d.year}-{d.month:02d}" for d in dates]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    for lo, hi, col, al in [(0, 15, "#639922", 0.20), (15, 35, "#97C459", 0.13),
                            (35, 65, "#888780", 0.07), (65, 85, "#EF9F27", 0.16),
                            (85, 100, "#E24B4A", 0.20)]:
        ax1.axhspan(lo, hi, color=col, alpha=al, lw=0)
    ax1.plot(x, val.composite, color="#534AB7", lw=2.3, label="Valorização (0–100)")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Valorização (0–100)")

    ax2 = ax1.twinx()
    ax2.plot(x, prices, color="#BA7517", lw=1.4, ls="--", label="Preço BTC (log)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Preço BTC (log, USD)")

    for i, txt in {19: "Fundo COVID", 24: "Blow-off 2021", 35: "Fundo FTX",
                   46: "Topo 2024", 52: "Hoje"}.items():
        c = val.composite[i]
        if not np.isnan(c):
            ax1.scatter([i], [c], color="#26215C", s=28, zorder=5)
            ax1.annotate(txt, (i, c), textcoords="offset points", xytext=(0, 9),
                         ha="center", fontsize=8, color="#26215C")

    ax1.set_xticks(x[::4])
    ax1.set_xticklabels(ym[::4], rotation=45, ha="right", fontsize=8)
    ax1.set_title("SDCA Valuation Oscillator v1 — backtest (dados demo aproximados)")
    h1, la1 = ax1.get_legend_handles_labels()
    h2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, la1 + la2, loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    import sys
    if "chart" in sys.argv:
        path = render_chart(show=("--show" in sys.argv))
        print(f"Gráfico gravado em {path}")
    else:
        run_demo()

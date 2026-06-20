"""
Auditoria anti-overfit do motor de valorização (testes de regressão).

Estes testes travam, no futuro, qualquer alteração que introduza:
  • lookahead / curve-fitting (o passado tem de ser imune ao futuro),
  • dependência de um único hiperparâmetro (half-life),
  • dependência de uma única primitiva (leave-one-out),
  • inversão de orientação (alto = caro),
  • valores fora dos limites.

Funciona em qualquer um dos repos: descobre valuation_engine.py quer esteja
na raiz (btc-bottom-score) quer em dashboard/ (orderflow-system).
"""
import os
import sys
from datetime import date, timedelta

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.dirname(_here),
              os.path.join(os.path.dirname(_here), "dashboard")):
    if os.path.exists(os.path.join(_cand, "valuation_engine.py")):
        sys.path.insert(0, _cand)
        break

import valuation_engine as ve  # noqa: E402

# Âncoras conhecidas do demo (verdade de ciclo).
_DATES, _PRICES = ve._demo_series()
_YM = [f"{d.year}-{d.month:02d}" for d in _DATES]
_IDX = {k: _YM.index(k) for k in ("2020-03", "2022-11", "2024-12")}
_BOTTOMS = ("2020-03", "2022-11")   # devem dar BAIXO
_TOPS = ("2024-12",)                # deve dar ALTO


def _composite(halflife=2.0, ppy=12, dates=None, prices=None):
    dates = _DATES if dates is None else dates
    prices = _PRICES if prices is None else prices
    return ve.compute_series(dates, prices, periods_per_year=ppy,
                             halflife_years=halflife).composite


# ── A. CAUSALIDADE — o teste decisivo anti-lookahead ─────────────────────────
def test_sem_lookahead_o_passado_e_imune_ao_futuro():
    for ppy in (12, 52):
        full = _composite(ppy=ppy)
        for T in range(20, len(_PRICES)):
            cut = ve.compute_series(_DATES[:T], _PRICES[:T],
                                    periods_per_year=ppy).composite
            both = ~np.isnan(full[:T]) & ~np.isnan(cut)
            assert np.allclose(full[:T][both], cut[both], atol=1e-9), (
                f"lookahead detetado em ppy={ppy}, T={T}")


# ── B. ROBUSTEZ AO HALF-LIFE — não está afinado a um valor mágico ────────────
def test_robusto_ao_halflife():
    for hl in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        c = _composite(halflife=hl)
        for k in _BOTTOMS:
            assert c[_IDX[k]] < 50, f"bottom {k} alto com half-life={hl}"
        for k in _TOPS:
            assert c[_IDX[k]] > 50, f"top {k} baixo com half-life={hl}"


# ── C. LEAVE-ONE-OUT — nenhuma primitiva sozinha decide ──────────────────────
def test_leave_one_out():
    raw_all = {
        "trend_deviation": ve.prim_trend_deviation(_DATES, _PRICES),
        "long_ma_ratio": ve.prim_long_ma_ratio(_PRICES, 12),
        "drawdown": ve.prim_drawdown(_PRICES),
        "momentum": ve.prim_momentum(_PRICES, 12),
        "issuance_value": ve.prim_issuance_value(_DATES, _PRICES, 12),
        "ma_spread": ve.prim_ma_spread(_PRICES, 4, 12),
    }
    for drop in list(raw_all):
        raw = {k: v for k, v in raw_all.items() if k != drop}
        pct = {k: ve.pct_rank_decayed(v, 2.0 * 12) for k, v in raw.items()}
        comp = np.full(len(_PRICES), np.nan)
        for i in range(len(_PRICES)):
            vals = [pct[k][i] for k in raw if not np.isnan(pct[k][i])]
            if vals:
                comp[i] = 100.0 * np.mean(vals)
        for k in _BOTTOMS:
            assert comp[_IDX[k]] < 50, f"sem {drop}: bottom {k} subiu"
        for k in _TOPS:
            assert comp[_IDX[k]] > 50, f"sem {drop}: top {k} desceu"


# ── D. DETERMINISMO ──────────────────────────────────────────────────────────
def test_determinismo():
    assert np.allclose(_composite(), _composite(), equal_nan=True)


# ── E. ORIENTAÇÃO — alto = caro ──────────────────────────────────────────────
def test_orientacao():
    n = 300
    up = np.exp(np.linspace(np.log(100), np.log(10000), n))
    down = np.concatenate([up[:n // 2],
                           up[n // 2] * np.exp(-np.linspace(0, 2, n - n // 2))])
    ds = [date(2015, 1, 1) + timedelta(weeks=k) for k in range(n)]
    cu = ve.compute_series(ds, up, periods_per_year=52).composite
    cd = ve.compute_series(ds, down, periods_per_year=52).composite
    assert np.nanmean(cu[-10:]) > 70, "subida longa devia dar score alto"
    assert np.nanmean(cd[-10:]) < 40, "queda forte devia dar score baixo"


# ── F. LIMITES E ESTABILIDADE ────────────────────────────────────────────────
def test_limites():
    v = ve.compute_series(_DATES, _PRICES, periods_per_year=12)
    comp = v.composite[~np.isnan(v.composite)]
    assert comp.min() >= 0 and comp.max() <= 100
    conv = v.conviction[~np.isnan(v.conviction)]
    assert conv.min() >= 0 and conv.max() <= 1
    assert isinstance(ve.zone_for(float("nan"))[0], str)  # não rebenta


# ── G. MONOTONIA DO PERCENTIL ────────────────────────────────────────────────
def test_percentil_monotono():
    x = np.arange(1.0, 14.0)
    lo = ve.pct_rank_decayed(x.copy(), 24.0)[-1]
    x2 = x.copy()
    x2[-1] = 100.0
    hi = ve.pct_rank_decayed(x2, 24.0)[-1]
    assert hi >= lo

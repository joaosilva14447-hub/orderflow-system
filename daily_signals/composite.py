"""Composite layer — blends the 4 pillars into one Accumulation Score + a
named setup + alignment/divergence flags.

Orientation: every pillar is mapped so HIGH = favours accumulation.
  - Valuation:    cheap is bullish        -> contribution = 100 - val.score
  - Doom:         panic is bullish (contrarian) -> contribution = doom_score
  - Macro:        risk-on tailwind bullish -> contribution = macro.score
  - Positioning:  loaded is bullish        -> contribution = pos.score

The blended number is for at-a-glance only; the named setup and the flags are
what carry the real, non-averaged signal (so conflicts aren't hidden).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from valuation import ValuationResult
from macro import MacroResult
from positioning import PositioningResult


@dataclass
class CompositeResult:
    score: float                       # 0-100 (high = favours accumulation)
    label: str                         # human label for the score band
    setup: str                         # named, rules-based setup
    flags: list[str] = field(default_factory=list)  # alignment/divergence notes
    n_pillars: int = 0


def _label(score: float) -> str:
    if score >= 70:
        return "🟢 Acumulação forte"
    if score >= 57:
        return "🟢 Acumular"
    if score >= 43:
        return "⚪ Neutro"
    if score >= 30:
        return "🟡 Cauteloso"
    return "🔴 Evitar (caro)"


def _named_setup(val_zone: str | None, doom_zone: str | None,
                 macro_zone: str | None, pos_zone: str | None) -> str:
    cheap = val_zone in ("deep_value", "cheap")
    panic = doom_zone in ("high", "extreme")
    tailwind = macro_zone == "risk_on"
    loaded = pos_zone == "loaded"

    # Priority order: strongest, most specific setup first.
    if panic and cheap:
        return "🚨 CAPITULAÇÃO + VALOR — pânico narrativo com preço em acumulação"
    if cheap and (tailwind or loaded):
        return "🟢 ACUMULAR — barato com vento a favor / combustível"
    if cheap:
        return "🟡 BARATO, SEM GATILHO — valor presente, falta confirmação"
    if val_zone == "expensive":
        return "🔴 CARO — fora de zona de acumulação"
    return "⚪ NEUTRO — sem setup claro"


def _flags(val: ValuationResult | None, doom_zone: str | None,
           macro: MacroResult | None, pos: PositioningResult | None) -> list[str]:
    flags: list[str] = []
    vz = val.zone if val else None
    mz = macro.zone if macro else None
    pz = pos.zone if pos else None

    # Divergences worth surfacing (a blended average would hide these).
    if vz in ("deep_value", "cheap") and mz == "risk_off":
        flags.append("⚠️ Divergência: valuation barato mas macro hostil "
                     "(possível transição de regime — o barato pode ficar mais barato).")
    if doom_zone in ("high", "extreme") and vz in ("fair", "expensive"):
        flags.append("⚠️ Divergência: pânico noticioso sem valor "
                     "(típico do início de um bear, não do fundo).")
    if vz in ("deep_value", "cheap") and pz == "loaded":
        flags.append("✅ Alinhamento: valor + dry powder carregado.")
    if val and val.zone == "deep_value" and doom_zone in ("high", "extreme"):
        flags.append("✅ Alinhamento forte: deep value + capitulação.")
    return flags


def compute(val: ValuationResult | None, doom_score: float, doom_zone: str,
            doom_provisional: bool, macro: MacroResult | None,
            pos: PositioningResult | None) -> CompositeResult:
    contributions: list[float] = []
    if val is not None:
        contributions.append(100.0 - val.score)
    contributions.append(doom_score)  # always available (provisional early on)
    if macro is not None:
        contributions.append(macro.score)
    if pos is not None:
        contributions.append(pos.score)

    score = round(sum(contributions) / len(contributions), 1)
    setup = _named_setup(val.zone if val else None, doom_zone,
                         macro.zone if macro else None,
                         pos.zone if pos else None)
    flags = _flags(val, doom_zone, macro, pos)
    if doom_provisional:
        flags.append("ℹ️ Doom ainda provisório (a acumular histórico) — "
                     "peso do pânico no score é preliminar.")

    return CompositeResult(score=score, label=_label(score), setup=setup,
                           flags=flags, n_pillars=len(contributions))

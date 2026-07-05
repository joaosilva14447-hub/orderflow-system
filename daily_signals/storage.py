"""Persist the daily snapshot to a versioned CSV + per-day headline archive.

The CSV is the backtestable time series. The JSON archive keeps the raw
headlines so you build your own searchable news history from day one.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

from doom_index import DoomResult
from valuation import ValuationResult
from macro import MacroResult
from positioning import PositioningResult
from composite import CompositeResult

HISTORY_DIR = Path(__file__).parent / "history"
CSV_FILE = HISTORY_DIR / "doom_index.csv"
HEADLINES_DIR = HISTORY_DIR / "headlines"

CSV_FIELDS = [
    "date",
    "comp_score", "setup",
    "doom_score", "provisional", "doom_raw",
    "n_headlines", "n_severe", "n_systemic", "zone",
    "val_score", "val_zone", "val_mvrv", "val_mayer", "val_puell", "val_metcalfe",
    "macro_score", "macro_zone", "macro_liq", "macro_usd", "macro_credit",
    "pos_score", "pos_zone", "pos_scmom", "pos_ssr", "pos_funding",
]

# Percentile-based zones (computed over trailing history).
ZONE_CALM, ZONE_ELEVATED, ZONE_HIGH, ZONE_EXTREME = (
    "calm", "elevated", "high", "extreme",
)


def _zone(score: float, provisional: bool) -> str:
    # During warm-up the score is a provisional absolute value; don't make
    # regime claims (calm/extreme) we can't back with history yet.
    if provisional:
        return "warming_up"
    if score >= 95:
        return ZONE_EXTREME
    if score >= 80:
        return ZONE_HIGH
    if score >= 50:
        return ZONE_ELEVATED
    return ZONE_CALM


def _provisional_score(doom_raw: float, total_weight: float) -> float:
    """Absolute 0-100 used only until enough history exists for a percentile.

    Average weighted severity (0-3) mapped to 0-100. Sits low by design —
    it's a placeholder, clearly flagged as provisional, not a regime signal.
    """
    if total_weight <= 0:
        return 0.0
    avg_severity = doom_raw / total_weight  # 0..3
    return round(min(100.0, 100.0 * avg_severity / 3.0), 1)


def _read_history() -> list[dict]:
    if not CSV_FILE.exists():
        return []
    with open(CSV_FILE, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _percentile(value: float, history_values: list[float], min_n: int = 30) -> float | None:
    """Percentile rank of `value` within prior history. None until enough data."""
    if len(history_values) < min_n:
        return None
    below = sum(1 for v in history_values if v <= value)
    return round(100.0 * below / len(history_values), 1)


def save_snapshot(result: DoomResult, valuation: ValuationResult | None = None,
                  macro: MacroResult | None = None,
                  positioning: PositioningResult | None = None,
                  composite: CompositeResult | None = None,
                  snapshot_date: date | None = None) -> dict:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    HEADLINES_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_date = snapshot_date or date.today()
    iso = snapshot_date.isoformat()

    history = _read_history()
    # Exclude any existing row for today (idempotent re-runs).
    prior = [r for r in history if r["date"] != iso]
    prior_raws = [float(r["doom_raw"]) for r in prior]

    pct = _percentile(result.doom_raw, prior_raws)
    provisional = pct is None
    if provisional:
        total_weight = sum(s.headline.weight for s in result.scored)
        doom_score = _provisional_score(result.doom_raw, total_weight)
    else:
        doom_score = pct  # percentile is already a regime-aware 0-100
    zone = _zone(doom_score, provisional)

    vc = valuation.components if valuation else {}
    mc = macro.components if macro else {}
    pc = positioning.components if positioning else {}
    row = {
        "date": iso,
        "comp_score": composite.score if composite else "",
        "setup": composite.setup if composite else "",
        "doom_score": doom_score,
        "provisional": int(provisional),
        "doom_raw": result.doom_raw,
        "n_headlines": result.n_headlines,
        "n_severe": result.n_severe,
        "n_systemic": result.n_systemic,
        "zone": zone,
        "val_score": valuation.score if valuation else "",
        "val_zone": valuation.zone if valuation else "",
        "val_mvrv": vc.get("mvrv_z", ""),
        "val_mayer": vc.get("mayer", ""),
        "val_puell": vc.get("puell", ""),
        "val_metcalfe": vc.get("metcalfe", ""),
        "macro_score": macro.score if macro else "",
        "macro_zone": macro.zone if macro else "",
        "macro_liq": mc.get("liq_mom", ""),
        "macro_usd": mc.get("usd_mom", ""),
        "macro_credit": mc.get("credit", ""),
        "pos_score": positioning.score if positioning else "",
        "pos_zone": positioning.zone if positioning else "",
        "pos_scmom": pc.get("sc_mom", ""),
        "pos_ssr": pc.get("ssr", ""),
        "pos_funding": pc.get("funding", ""),
    }

    # Rewrite CSV (prior rows + today) sorted by date.
    all_rows = sorted(prior + [row], key=lambda r: r["date"])
    with open(CSV_FILE, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    # Archive today's raw headlines.
    with open(HEADLINES_DIR / f"{iso}.json", "w", encoding="utf-8") as fh:
        json.dump(
            [{"severity": s.severity, "category": s.category, **s.headline.as_dict()}
             for s in result.scored],
            fh, ensure_ascii=False, indent=2,
        )

    prev_zone = prior[-1]["zone"] if prior else None
    prev_val_zone = prior[-1].get("val_zone") if prior else None
    return {**row, "provisional": provisional,
            "prev_zone": prev_zone, "prev_val_zone": prev_val_zone}


if __name__ == "__main__":
    print(f"CSV: {CSV_FILE}")
    print(f"rows: {len(_read_history())}")

"""Send the daily Doom Index to Discord via webhook."""

from __future__ import annotations

import os

import requests

from doom_index import DoomResult
from valuation import ValuationResult
from macro import MacroResult
from positioning import PositioningResult
from composite import CompositeResult

_ZONE_COLOR = {
    "calm": 0x2ECC71,       # green
    "elevated": 0xF1C40F,   # yellow
    "high": 0xE67E22,       # orange
    "extreme": 0xE74C3C,    # red
    "warming_up": 0x95A5A6,  # grey
}
_ZONE_LABEL = {
    "calm": "🟢 Calmo",
    "elevated": "🟡 Elevado",
    "high": "🟠 Alto",
    "extreme": "🔴 Extremo (capitulação narrativa)",
    "warming_up": "⚪ A acumular histórico",
}
_VAL_LABEL = {
    "deep_value": "🟢 Deep Value", "cheap": "🟢 Cheap",
    "fair": "🟡 Fair", "expensive": "🔴 Expensive",
}
_MACRO_LABEL = {
    "risk_off": "🔴 Risk-off", "neutral": "🟡 Neutral", "risk_on": "🟢 Risk-on",
}
_POS_LABEL = {
    "empty": "🔴 Vazio", "neutral": "🟡 Neutro", "loaded": "🟢 Carregado",
}


def _webhook_url() -> str | None:
    return os.environ.get("DISCORD_WEBHOOK_URL")


def _bar(score: float) -> str:
    """A tiny 10-slot text gauge for at-a-glance reading."""
    filled = max(0, min(10, round(score / 10)))
    return "█" * filled + "░" * (10 - filled)


def _embed(snapshot: dict, result: DoomResult,
           valuation: ValuationResult | None, macro: MacroResult | None,
           positioning: PositioningResult | None,
           composite: CompositeResult | None, gate: bool) -> dict:
    zone = snapshot["zone"]
    doom_score = snapshot["doom_score"]
    provisional = snapshot.get("provisional", False)

    lines: list[str] = []

    # ── Composite header ──────────────────────────────────────────────
    if composite is not None:
        lines.append(f"# {composite.score:.0f}/100  ·  {composite.label}")
        lines.append(f"`{_bar(composite.score)}`")
        lines.append("")

    # ── Pillars ───────────────────────────────────────────────────────
    lines.append("**Pilares**")
    doom_tag = " _(provisório)_" if provisional else ""
    lines.append(f"🌊 Doom (capitulação): **{doom_score:.0f}** "
                 f"— {_ZONE_LABEL.get(zone, zone)}{doom_tag}")
    if valuation is not None:
        lines.append(f"📈 Valuation: **{valuation.score:.0f}** "
                     f"— {_VAL_LABEL.get(valuation.zone, valuation.zone)}")
    if macro is not None:
        lines.append(f"🌐 Macro: **{macro.score:.0f}** "
                     f"— {_MACRO_LABEL.get(macro.zone, macro.zone)}")
    if positioning is not None:
        lines.append(f"🔗 Posicionamento: **{positioning.score:.0f}** "
                     f"— {_POS_LABEL.get(positioning.zone, positioning.zone)}")

    # ── Gate + flags ──────────────────────────────────────────────────
    if gate:
        lines.append("\n🚨 **GATE: capitulação narrativa COM valuation barato** "
                     "— alinhamento de pânico + preço em acumulação.")
    if composite is not None and composite.flags:
        lines.append("")
        lines.extend(composite.flags)

    # ── Top negative headlines ────────────────────────────────────────
    top = result.top_negative[:4]
    if top:
        lines.append("\n**Manchetes negativas:**")
        for s in top:
            title = s.headline.title[:120]
            link = s.headline.link
            lines.append(f"• [{title}]({link})" if link else f"• {title}")

    color = 0xC0392B if gate else _ZONE_COLOR.get(zone, 0x95A5A6)
    return {
        "title": "📊 Market State — Bottom Radar",
        "description": "\n".join(lines),
        "color": color,
        "footer": {"text": f"{snapshot['date']} • daily_signals"},
    }


def send(snapshot: dict, result: DoomResult,
         valuation: ValuationResult | None = None,
         macro: MacroResult | None = None,
         positioning: PositioningResult | None = None,
         composite: CompositeResult | None = None, gate: bool = False) -> bool:
    """Post the daily embed. Returns True on success."""
    url = _webhook_url()
    if not url:
        print("[discord] DISCORD_WEBHOOK_URL not set; skipping send")
        return False
    try:
        embed = _embed(snapshot, result, valuation, macro, positioning,
                       composite, gate)
        resp = requests.post(url, json={"embeds": [embed]}, timeout=15)
        resp.raise_for_status()
        print(f"[discord] sent (status {resp.status_code})")
        return True
    except Exception as exc:
        print(f"[discord] WARN send failed: {exc}")
        return False

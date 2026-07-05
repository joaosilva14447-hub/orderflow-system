"""Daily orchestrator: 4 pillars -> composite -> store -> Discord.

Pillars: Doom (news), Valuation (Coin Metrics), Macro (FRED), Positioning
(DefiLlama + Binance). Entry point for the GitHub Actions cron. Idempotent per
day. Each on-chain/macro pillar is fetched in a guarded block: if one source
hiccups, the run still completes with whatever pillars succeeded.

Notification policy (anti-fatigue): send on a strong setup/gate, when the Doom
OR Valuation zone changes, when Doom is high/extreme, or on Sunday (weekly
digest). Override with FORCE_SEND=1.
"""

from __future__ import annotations

import os
import sys
from datetime import date

# Ensure emoji-laden logs never crash on a non-UTF-8 console (e.g. Windows cp1252).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

import news_feed
import doom_index
import valuation as valuation_mod
import macro as macro_mod
import positioning as positioning_mod
import composite as composite_mod
import storage
import discord_notify


def _guarded(name: str, fn):
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        print(f"[run] WARN {name} failed, continuing without it: {exc}")
        return None


def _should_send(snapshot: dict, gate: bool) -> bool:
    if os.environ.get("FORCE_SEND") == "1":
        return True
    if date.today().weekday() == 6:  # Sunday weekly digest
        return True
    if gate:
        return True
    zone, prev_zone = snapshot["zone"], snapshot.get("prev_zone")
    if zone in ("high", "extreme"):
        return True
    if prev_zone is not None and zone != prev_zone:
        return True
    val_zone, prev_val = snapshot.get("val_zone"), snapshot.get("prev_val_zone")
    return bool(prev_val) and val_zone != prev_val


def main() -> None:
    # Pillar 1 — Doom (news).
    headlines = news_feed.fetch_headlines(hours=24)
    doom = doom_index.score(headlines)

    # Pillars 2-4 — guarded (a failed source must not sink the whole run).
    valuation = _guarded("valuation", valuation_mod.compute)
    macro = _guarded("macro", macro_mod.compute)
    positioning = _guarded("positioning", positioning_mod.compute)

    # Persist first (need the resolved doom score/zone for the composite).
    snapshot = storage.save_snapshot(doom, valuation, macro, positioning)

    # Composite layer.
    comp = composite_mod.compute(
        val=valuation, doom_score=snapshot["doom_score"], doom_zone=snapshot["zone"],
        doom_provisional=snapshot.get("provisional", False),
        macro=macro, pos=positioning,
    )
    # Re-persist with the composite fields filled in.
    snapshot = storage.save_snapshot(doom, valuation, macro, positioning, comp)

    # Capitulation gate: narrative panic WITH cheap valuation.
    gate = bool(valuation and valuation.is_cheap
                and snapshot["zone"] in ("high", "extreme"))

    print(f"[run] comp={comp.score} setup={comp.setup} "
          f"doom={snapshot['zone']} val={snapshot.get('val_zone')} "
          f"macro={snapshot.get('macro_zone')} pos={snapshot.get('pos_zone')} gate={gate}")

    if _should_send(snapshot, gate):
        discord_notify.send(snapshot, doom, valuation=valuation, macro=macro,
                            positioning=positioning, composite=comp, gate=gate)
    else:
        print("[run] nothing changed and not digest day; Discord send skipped")


if __name__ == "__main__":
    main()

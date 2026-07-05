"""Score headlines for negativity/severity and aggregate into a daily Doom raw score.

Primary path: Claude Haiku (batched, one API call). Falls back to a keyword
heuristic if no API key is set, so the pipeline never fully breaks.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from news_feed import Headline

MODEL = "claude-haiku-4-5-20251001"
BATCH_SIZE = 60  # headlines per API call

# Severity scale:
#   0 = neutral / positive
#   1 = mildly negative
#   2 = clearly negative (big drawdown, single-entity failure)
#   3 = systemic / contagion (insolvency, ban, hack with spillover)
_SYSTEM = (
    "You are a crypto market sentiment classifier. For each numbered headline, "
    "return its fear/severity for a crypto investor on this scale: "
    "0 neutral or positive; 1 mildly negative; 2 clearly negative (large price "
    "drop, single-company failure, lawsuit); 3 systemic/contagion (exchange "
    "insolvency, hack with spillover, outright ban, sovereign/regulatory shock). "
    "Also tag a category: price, hack, insolvency, regulation, macro, other. "
    "Respond ONLY with a JSON array; one object per headline in the same order, "
    'shape: {"i": <index>, "severity": <0-3>, "category": "<cat>"}. No prose.'
)

# Tier 3 = genuinely systemic/contagion events only (kept tight to avoid the
# classic false positive where a minor "ban" headline looks catastrophic).
_NEG_KEYWORDS = {
    3: ["insolven", "bankrupt", "contagion", "collapse", "exploit",
        "hacked", "stolen", "drained", "funds lost", "depeg", "halt withdrawal",
        "freeze withdrawal", "default on", "seized", "meltdown", "wiped out",
        "bank run", "liquidity crisis"],
    2: ["crash", "plunge", "lawsuit", "sec charges", "sec sues", "liquidat",
        "sell-off", "selloff", "slump", "tumble", "probe", "fraud", "delist",
        "trading ban", "outright ban", "bans crypto", "halt trading", "outflow"],
    1: ["fall", "drop", "decline", "fear", "warning", "risk", "concern",
        "slide", "weak", "dip", "pressure", "uncertain", "caution"],
}

# If a headline is clearly positive and has no tier-3 term, damp it down one
# level — cuts false positives like "Bitcoin drops fear as ETF inflows surge".
_POS_KEYWORDS = [
    "surge", "rally", "soar", "record high", "all-time high", "etf approv",
    "approved", "adopt", "jumps", "gains", "rebound", "recover", "inflow",
    "bullish", "upgrade", "milestone",
]


@dataclass
class ScoredHeadline:
    headline: Headline
    severity: int
    category: str


@dataclass
class DoomResult:
    scored: list[ScoredHeadline]
    doom_raw: float          # weighted sum of severities
    n_headlines: int
    n_severe: int            # count of severity >= 2
    n_systemic: int          # count of severity == 3

    @property
    def top_negative(self) -> list[ScoredHeadline]:
        return sorted(
            [s for s in self.scored if s.severity >= 2],
            key=lambda s: (s.severity, s.headline.weight),
            reverse=True,
        )


def _keyword_severity(text: str) -> int:
    t = text.lower()
    sev = 0
    for level in (3, 2, 1):
        if any(k in t for k in _NEG_KEYWORDS[level]):
            sev = level
            break
    # Positive-tone dampener (never touches genuine systemic tier-3 events).
    if sev in (1, 2) and any(p in t for p in _POS_KEYWORDS):
        sev -= 1
    return sev


def _score_keywords(headlines: list[Headline]) -> list[ScoredHeadline]:
    print("[doom_index] scoring with keyword fallback")
    out = []
    for h in headlines:
        sev = _keyword_severity(f"{h.title} {h.summary}")
        out.append(ScoredHeadline(h, sev, "other"))
    return out


def _score_llm(headlines: list[Headline], api_key: str) -> list[ScoredHeadline]:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    results: list[ScoredHeadline] = []

    for start in range(0, len(headlines), BATCH_SIZE):
        batch = headlines[start : start + BATCH_SIZE]
        numbered = "\n".join(f"{i}. {h.title}" for i, h in enumerate(batch))
        try:
            msg = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=[{"type": "text", "text": _SYSTEM,
                         "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": numbered}],
            )
            raw = msg.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
            parsed = json.loads(raw)
            by_i = {int(o["i"]): o for o in parsed}
            for i, h in enumerate(batch):
                o = by_i.get(i, {})
                sev = max(0, min(3, int(o.get("severity", 0))))
                results.append(ScoredHeadline(h, sev, str(o.get("category", "other"))))
        except Exception as exc:
            print(f"[doom_index] WARN LLM batch failed ({exc}); keyword fallback")
            results.extend(_score_keywords(batch))

    return results


def score(headlines: list[Headline]) -> DoomResult:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not headlines:
        return DoomResult([], 0.0, 0, 0, 0)

    scored = _score_llm(headlines, api_key) if api_key else _score_keywords(headlines)

    doom_raw = sum(s.severity * s.headline.weight for s in scored)
    n_severe = sum(1 for s in scored if s.severity >= 2)
    n_systemic = sum(1 for s in scored if s.severity == 3)
    print(f"[doom_index] doom_raw={doom_raw:.1f} severe={n_severe} systemic={n_systemic}")
    return DoomResult(scored, round(doom_raw, 2), len(scored), n_severe, n_systemic)

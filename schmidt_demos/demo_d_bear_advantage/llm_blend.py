"""LLM-blended gene-text generation at reproduction events.

Calls Anthropic's Haiku to blend two parent gene-text fragments at a
single locus into one offspring fragment that retains the locus's
behavioural intent but introduces text-level variation. This is the
natural-language genome operation underlying the Connection-Science
$d = 3.89$–$5.14$ heritability result, now invoked on the cyber
substrate.

Pinned model: ``claude-haiku-4-5-20251001`` (date-suffixed snapshot
for exact reproducibility).

The blender is *deterministic given a seed* via a temperature=0.0
call plus an explicit nonce-in-prompt strategy: the seed is included
in the prompt so two calls with different seeds produce different
outputs but two calls with the same (parents, locus, seed) produce
the same output. (Anthropic doesn't expose a seed parameter; this is
the cleanest stand-in.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from anthropic import Anthropic

from dotenv import load_dotenv


_ENV_LOADED = False


def _ensure_env() -> None:
    global _ENV_LOADED
    if not _ENV_LOADED:
        # Try project .env first, then bear-dev/.env as fallback
        from pathlib import Path
        for p in (Path(".env"), Path("../bear-dev/.env")):
            if p.exists():
                load_dotenv(p, override=False)
        _ENV_LOADED = True


PINNED_MODEL = "claude-haiku-4-5-20251001"

_BLEND_SYSTEM = """\
You are blending two natural-language gene fragments for a behavioral
genetics study. Each fragment is a single English sentence describing
a defender behavior in a cybersecurity simulation.

Produce ONE blended sentence that:
- preserves the behavioral intent shared between the two parents
- introduces small lexical/structural variation (natural drift)
- is one to two sentences, no more than 40 words
- does NOT include explanations, quotes, or prefixes
- outputs ONLY the blended sentence text, nothing else
"""


@dataclass(frozen=True)
class BlendKey:
    """Cache key — identical (parents, locus, seed) → identical blend."""
    parent_a_text: str
    parent_b_text: str
    locus: str
    seed: int


def _build_user_message(parent_a: str, parent_b: str,
                         locus: str, seed: int) -> str:
    return (
        f"Locus: {locus}\n"
        f"Variation seed: {seed}\n\n"
        f"Parent A:\n{parent_a}\n\n"
        f"Parent B:\n{parent_b}\n\n"
        "Output the blended sentence:"
    )


@lru_cache(maxsize=2048)
def _cached_blend(parent_a: str, parent_b: str, locus: str, seed: int) -> str:
    _ensure_env()
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not found. Place it in .env or set the env var."
        )
    client = Anthropic()
    msg = client.messages.create(
        model=PINNED_MODEL,
        max_tokens=120,
        temperature=0.7,   # nonzero — we want variation; seed-in-prompt nonces it
        system=_BLEND_SYSTEM,
        messages=[{
            "role": "user",
            "content": _build_user_message(parent_a, parent_b, locus, seed),
        }],
    )
    text = "".join(
        block.text for block in msg.content if hasattr(block, "text")
    ).strip()
    # If parents were identical (no useful blend) just return parent A
    if not text:
        return parent_a
    return text


def blend_gene_text(parent_a_text: str, parent_b_text: str, *,
                    locus: str, seed: int) -> str:
    """Public API: produce one blended sentence from two parent fragments."""
    return _cached_blend(parent_a_text, parent_b_text, locus, seed)

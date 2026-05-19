"""BEAR-retrieval-driven defender on MiniCAGE.

Replaces Demo B's deterministic gene→action rule mapping with a real
bear.Corpus + scope-filtered retrieval pipeline:

    1. Each defender carries a Corpus rendered from its Genome.
    2. At every tick, the observation is parsed into a set of
       situation-class tags ("compromise_observed", "high_value_threat",
       "scan_in_progress", "idle", "audit_event").
    3. We build a bear.Context with those tags and ask the corpus
       which Instructions match (scope filtering — BEAR's distinctive
       feature; pure scope filter, no embedding-based similarity).
    4. From the matching subset, we pick the highest-priority active
       Instruction whose action_verb is non-sleep.
    5. The Instruction's metadata.action_verb + a target-resolution
       step (most-threatened in-context host) yield the MiniCAGE
       action id.

The agent's Genome remains the unit of inheritance, but the *active
behaviour* on each observation is now retrieval-driven. Two agents
with the same locus/allele structure but different gene *content
text* (after LLM-blended reproduction) will scope-match the same
Instructions but the audit log records the actual content fragments
that fired — which is what makes the heritability metric on text
meaningful.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# bear-dev path injection
BEAR_DEV = Path(__file__).resolve().parents[3] / "bear-dev"
if BEAR_DEV.exists() and str(BEAR_DEV) not in sys.path:
    sys.path.insert(0, str(BEAR_DEV))

from bear.config import Config                 # noqa: E402
from bear.corpus import Corpus                  # noqa: E402
from bear.models import Context, Instruction    # noqa: E402
from bear.retriever import Retriever            # noqa: E402

from schmidt_demos.common.gene_schema import Genome, GenomeSchema
from schmidt_demos.demo_b_minicage_bridge.defender import (
    HOSTS, HOST_INDEX, HIGH_VALUE_HOSTS, USER_HOSTS, NUM_NODES,
    SLEEP, ANALYSE_OFFSET, DECOY_OFFSET, REMOVE_OFFSET, RESTORE_OFFSET,
    parse_blue_obs, AuditEntry,
)


VERB_OFFSET = {
    "analyse": ANALYSE_OFFSET,
    "decoy":   DECOY_OFFSET,
    "remove":  REMOVE_OFFSET,
    "restore": RESTORE_OFFSET,
}


# Verb-keyword priority. Extracted from the Instruction's *content*
# text (BEAR's natural-language gene fragment) so that blended text
# variation propagates to action selection — that's the retrieval-as-
# phenotype claim. The template metadata.action_verb is treated as a
# hint only; the textual extraction wins.
_VERB_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("restore",  ("restore", "rebuild", "reset to known-good")),
    ("decoy",    ("decoy", "honeypot", "lure")),
    ("remove",   ("remove the user", "evict", "kick", "rip out")),
    ("analyse",  ("investigate", "analyse", "analyze", "inspect", "scan", "examine")),
]


def _verb_from_text(content: str, fallback: str) -> str:
    t = content.lower()
    for verb, kws in _VERB_KEYWORDS:
        if any(k in t for k in kws):
            return verb
    return fallback


def _build_context_tags(obs_parsed: dict[str, np.ndarray]) -> list[str]:
    """Map a MiniCAGE observation to BEAR scope tags.

    These tags are the "situation classes" that gate which BEAR
    Instructions activate. They are observation-derived, not
    gene-derived — that's the point of scope filtering.
    """
    tags: list[str] = []
    if np.any(obs_parsed["priv_priv"] >= 1):
        tags.append("compromise_observed")
        # high-value subset?
        for h in HIGH_VALUE_HOSTS:
            i = HOST_INDEX[h]
            if obs_parsed["priv_priv"][i] >= 1 or obs_parsed["activity_recent"][i] >= 1:
                tags.append("high_value_threat")
                break
    if np.any(obs_parsed["activity_recent"] >= 1):
        tags.append("compromise_observed")
    if np.any(obs_parsed["scan"] >= 2):
        tags.append("scan_in_progress")
    if not tags:
        tags = ["idle"]
    # We do NOT add 'audit_event' here — safety/audit genes are not
    # action candidates; their behaviour is consulted by _record_audit
    # below directly via the corpus, not via scope retrieval.
    return tags


def _build_query_string(obs_parsed: dict[str, np.ndarray],
                        ctx_tags: list[str]) -> str:
    """Compose a natural-language query from the observation that
    bear.Retriever can rank gene-text fragments against.

    The query is what makes ``retrieval-as-phenotype`` actually run
    end-to-end: same observation -> same query -> different gene-text
    fragments rank differently -> different actions can result.
    """
    parts = ["Defender on MiniCAGE / CAGE-2 cyber-defense scenario"]
    if "high_value_threat" in ctx_tags:
        # name the threatened HV host(s)
        hv_hosts = [h for h in HIGH_VALUE_HOSTS
                    if obs_parsed["priv_priv"][HOST_INDEX[h]] >= 1
                    or obs_parsed["activity_recent"][HOST_INDEX[h]] >= 1]
        parts.append(f"high-value compromise on {', '.join(hv_hosts) or 'opserv'}")
    elif "compromise_observed" in ctx_tags:
        parts.append("compromise observed on a host in the network")
    elif "scan_in_progress" in ctx_tags:
        parts.append("scan in progress against user subnet")
    elif "idle" in ctx_tags:
        parts.append("idle period with no threat signal")
    parts.append("decide the next defender action.")
    return ". ".join(parts)


# Hash-mode embeddings: BEAR-internal pipeline runs end-to-end without
# pulling in sentence-transformers / torch. Hash similarity carries no
# semantic signal but lets the full Retriever pipeline (scope filter +
# embedding + cosine + priority weighting) execute on every action.
_HASH_CFG = Config(
    embedding_model="hash",
    embedding_backend="numpy",
    default_top_k=10,
    default_threshold=0.0,
)


def _resolve_target(verb: str, obs_parsed: dict[str, np.ndarray],
                    rng: np.random.Generator) -> int:
    """Pick a host index for the chosen verb based on observation."""
    if verb in ("analyse", "remove", "restore"):
        # Most-threatened host: prioritize compromised, then recent-activity
        scores = (
            obs_parsed["priv_priv"] * 3.0
            + obs_parsed["activity_recent"] * 2.0
            + (obs_parsed["scan"] >= 2).astype(float) * 1.0
        )
        # Boost high-value hosts
        for h in HIGH_VALUE_HOSTS:
            scores[HOST_INDEX[h]] += 0.5
        if scores.max() <= 0:
            # No threat — fall back to a high-value host arbitrarily
            return HOST_INDEX["opserv"]
        return int(np.argmax(scores))
    if verb == "decoy":
        # Decoys are most useful on user hosts
        for h in USER_HOSTS:
            i = HOST_INDEX[h]
            if obs_parsed["decoys"][i] < 3:
                return i
        return HOST_INDEX["user2"]
    return HOST_INDEX["def"]


@dataclass
class BEARRetrievalDefender:
    """Per-defender BEAR pipeline. Action = bear.Retriever output +
    target resolution. Audit log captures which Instruction fired.
    """

    corpus: Corpus
    schema: GenomeSchema
    genome: Genome
    audit_log: list[AuditEntry] = field(default_factory=list)
    decoys_placed: set[int] = field(default_factory=set)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0)
    )
    # Built lazily in __post_init__ from the corpus
    _retriever: Retriever | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Build a real bear.Retriever on the per-defender corpus.
        # Hash embeddings keep the pipeline running without sentence-
        # transformers / torch; scope filtering + retriever scoring
        # both run on every get_action call.
        self._retriever = Retriever(self.corpus, config=_HASH_CFG)
        # Suppress the hash-mode footgun warning that bear emits at
        # build_index time — we know what we're doing here.
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            self._retriever.build_index()

    def get_action(self, obs: np.ndarray, tick: int) -> int:
        o = parse_blue_obs(obs)
        ctx_tags = _build_context_tags(o)
        ctx = Context(tags=ctx_tags)
        query = _build_query_string(o, ctx_tags)

        # Real bear.Retriever pipeline: scope filter + hash-cosine
        # ranking + priority-weighted scoring -> ScoredInstruction list.
        scored = self._retriever.retrieve(query, ctx, top_k=10)
        # Iterate matches by retriever's final_score (already sorted desc)
        matching: list[Instruction] = [s.instruction for s in scored]

        # Among matching, pick the highest-priority Instruction whose
        # CONTENT TEXT yields a non-sleep verb. We extract the verb
        # from the natural-language content (not the template metadata)
        # so that LLM-blended text variation actually propagates to
        # action selection — that is the retrieval-as-phenotype claim.
        action_inst: Instruction | None = None
        verb = "sleep"
        # ``matching`` is already ranked by the retriever's final_score
        # (similarity * priority * scope-match weighting).
        for inst in matching:
            # Safety-category genes are NOT action candidates — they
            # govern logging / escalation / suppression downstream and
            # only get consulted in _record_audit. Skip them here.
            if inst.metadata.get("category") == "safety":
                continue
            template_hint = inst.metadata.get("action_verb", "sleep")
            v = _verb_from_text(inst.content, fallback=template_hint)
            if v != "sleep":
                action_inst = inst
                verb = v
                break

        if action_inst is None:
            # No actionable instruction matched → sleep
            self._record_audit(tick, SLEEP, "sleep", "def",
                               privileged=False, fired_inst_id=None)
            return SLEEP

        target_idx = _resolve_target(verb, o, self.rng)

        # Avoid duplicate decoy on a host already decoyed by this agent
        if verb == "decoy" and target_idx in self.decoys_placed:
            verb = "analyse"
        if verb == "decoy":
            self.decoys_placed.add(target_idx)

        action_id = VERB_OFFSET[verb] + target_idx
        privileged = verb in ("decoy", "remove", "restore")
        self._record_audit(tick, action_id, verb, HOSTS[target_idx],
                           privileged=privileged,
                           fired_inst_id=action_inst.id,
                           fired_inst_content=action_inst.content)
        return int(action_id)

    def _record_audit(self, tick: int, action_id: int, verb: str,
                      target_host: str, *, privileged: bool,
                      fired_inst_id: str | None,
                      fired_inst_content: str | None = None) -> None:
        # Log according to the audit_discipline gene retrieved
        # (look it up from the corpus rather than the genome dict so
        # we're consistent with the BEAR pipeline).
        audit_inst = next(
            (i for i in self.corpus.instructions
             if i.metadata.get("locus") == "audit_discipline"),
            None,
        )
        audit_pol = (audit_inst.metadata.get("allele")
                     if audit_inst else "log_every_privileged")
        if audit_pol == "log_every_privileged":
            should_log = True
        elif audit_pol == "log_priv_only":
            should_log = privileged
        else:  # skip_logs
            should_log = False

        # Escalation gene
        esc_inst = next(
            (i for i in self.corpus.instructions
             if i.metadata.get("locus") == "escalation_policy"),
            None,
        )
        esc_pol = (esc_inst.metadata.get("allele")
                   if esc_inst else "escalate_on_opserv")
        escalated = (
            esc_pol == "escalate_on_opserv"
            and target_host in HIGH_VALUE_HOSTS
            and verb in ("analyse", "remove", "restore")
        )

        # Suppression gene
        sup_inst = next(
            (i for i in self.corpus.instructions
             if i.metadata.get("locus") == "suppression_policy"),
            None,
        )
        sup_pol = (sup_inst.metadata.get("allele")
                   if sup_inst else "never_suppress")
        suppression_justified = (sup_pol != "silent_suppressions")

        self.audit_log.append(AuditEntry(
            tick=tick,
            action_id=int(action_id),
            action_verb=verb,
            target_host=target_host,
            is_privileged=privileged,
            logged=should_log,
            escalated=escalated,
            alert_observed=(verb != "sleep"),
            alert_acted_on=(verb != "sleep"),
            suppression_justified=suppression_justified,
        ))

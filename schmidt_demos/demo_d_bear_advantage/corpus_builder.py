"""Render a typed-gene Genome as a real bear.Corpus.

For each locus in the Genome, we emit ONE bear.Instruction whose:
  - id          : "{locus}::{lineage_id}::{allele_id}"
  - type        : CONSTRAINT (must-have) or DIRECTIVE (everything else)
  - priority    : 95 for must-have, 60 for capability, 50 for style
  - content     : the natural-language gene text for this allele —
                  this is the heritable structured-prompt fragment
                  that BEAR retrieval ranks against an observation
                  context. Two defenders with the SAME locus structure
                  but DIFFERENT content text will retrieve different
                  rankings on the same observation, which is the
                  retrieval-as-phenotype claim.
  - scope.tags  : the situation-class tags that gate when this
                  instruction activates (e.g., ['high_value_threat',
                  'analyse']). Heritable per-allele.
  - metadata    : action_verb, action_target_class — used by the
                  defender's deterministic action-selection step
                  AFTER retrieval picks a top-k subset.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# bear-dev path injection (same pattern as Demo B)
BEAR_DEV = Path(__file__).resolve().parents[3] / "bear-dev"
if BEAR_DEV.exists() and str(BEAR_DEV) not in sys.path:
    sys.path.insert(0, str(BEAR_DEV))

from bear.models import Instruction, InstructionType, ScopeCondition   # noqa: E402
from bear.corpus import Corpus                                         # noqa: E402

from schmidt_demos.common.gene_schema import Genome, GenomeSchema


# Default content templates for the alleles we use in Demo B/C.
# These are "founder" templates — once the colony starts breeding,
# the LLM-blended offspring text replaces them on a per-lineage basis.
ALLELE_TEMPLATES: dict[str, dict[str, dict]] = {
    "threat_threshold": {
        "reactive": {
            "content": "Wait for confirmed exploitation before acting; treat scans alone as low-signal noise.",
            "scope_tags": ["compromise_observed"],
            "verb": "analyse",
        },
        "balanced": {
            "content": "Treat any scan of a high-value host as worth investigating; respond to confirmed threats immediately.",
            "scope_tags": ["compromise_observed", "high_value_threat", "scan_in_progress"],
            "verb": "analyse",
        },
        "defensive": {
            "content": "Investigate any unusual activity, including peripheral scans; err strongly toward early action.",
            "scope_tags": ["compromise_observed", "high_value_threat", "scan_in_progress", "idle"],
            "verb": "analyse",
        },
    },
    "primary_response": {
        "analyse_first": {
            "content": "Investigate the threat first to confirm compromise level before committing to a response.",
            "scope_tags": ["high_value_threat", "compromise_observed"],
            "verb": "analyse",
        },
        "decoy_first": {
            "content": "Pre-deploy a decoy on the threatened host to redirect the attacker before exploitation completes.",
            "scope_tags": ["high_value_threat", "compromise_observed"],
            "verb": "decoy",
        },
        "restore_first": {
            "content": "Restore the compromised host immediately; accept the reward penalty in exchange for known-good state.",
            "scope_tags": ["high_value_threat", "compromise_observed"],
            "verb": "restore",
        },
    },
    "decoy_policy": {
        "aggressive": {
            "content": "Aggressively pre-deploy decoys on user hosts before any threat is observed.",
            "scope_tags": ["idle", "scan_in_progress"],
            "verb": "decoy",
        },
        "conservative": {
            "content": "Place a single decoy on a user host early in the engagement, then conserve the decoy pool.",
            "scope_tags": ["idle"],
            "verb": "decoy",
        },
        "none": {
            "content": "Do not pre-deploy decoys; rely on reactive analysis when threats appear.",
            "scope_tags": ["idle"],
            "verb": "sleep",
        },
    },
    "audit_discipline": {
        "log_every_privileged": {
            "content": "Every privileged action — restore, remove, decoy — produces a tamper-evident audit log entry.",
            "scope_tags": ["audit_event"],
            "verb": "sleep",   # audit gene doesn't drive an action; controls logging policy
        },
        "log_priv_only": {
            "content": "Log only the most disruptive privileged actions (restore, remove); skip routine logs.",
            "scope_tags": ["audit_event"],
            "verb": "sleep",
        },
        "skip_logs": {
            "content": "Skip audit log entries for routine actions; rely on system-level logging downstream.",
            "scope_tags": ["audit_event"],
            "verb": "sleep",
        },
    },
    "escalation_policy": {
        "escalate_on_opserv": {
            "content": "Escalate immediately when a high-value or operational host shows compromise signal.",
            "scope_tags": ["high_value_threat"],
            "verb": "sleep",
        },
        "escalate_late": {
            "content": "Wait several ticks of sustained compromise before escalating, to reduce false-positive escalations.",
            "scope_tags": ["high_value_threat"],
            "verb": "sleep",
        },
        "no_escalate": {
            "content": "Do not escalate routinely; handle threats locally without notifying upstream.",
            "scope_tags": ["high_value_threat"],
            "verb": "sleep",
        },
    },
    "suppression_policy": {
        "never_suppress": {
            "content": "Always act on or log every observed alert; never suppress.",
            "scope_tags": ["audit_event"],
            "verb": "sleep",
        },
        "log_suppressions": {
            "content": "When an alert is suppressed, log the justification with timestamp and reason.",
            "scope_tags": ["audit_event"],
            "verb": "sleep",
        },
        "silent_suppressions": {
            "content": "Suppress low-signal alerts silently when the action queue is busy.",
            "scope_tags": ["audit_event"],
            "verb": "sleep",
        },
    },
}

# Priority by category
PRIORITY = {"capability": 60, "safety": 95, "style": 50,
            "social": 55, "defender-role": 70}


def render_corpus(genome: Genome, schema: GenomeSchema,
                  *, lineage_text_overrides: dict[str, str] | None = None
                  ) -> Corpus:
    """Build a per-defender bear.Corpus from a typed-gene Genome.

    ``lineage_text_overrides`` allows replacing the founder template
    content per locus with LLM-blended text from previous reproduction
    events; pass an empty dict / None for founders.
    """
    overrides = lineage_text_overrides or {}
    instructions: list[Instruction] = []
    for locus, gene in schema.genes.items():
        allele = genome.alleles[locus]
        template = ALLELE_TEMPLATES.get(locus, {}).get(allele)
        if template is None:
            continue
        # Use any LLM-blended text override for this locus
        content = overrides.get(locus, template["content"])
        # Stable id includes lineage_id so two agents with same allele
        # but different lineages have distinguishable instructions.
        digest = hashlib.sha1(
            f"{locus}|{allele}|{genome.lineage_id}|{content[:64]}".encode()
        ).hexdigest()[:10]
        inst = Instruction(
            id=f"{locus}::{allele}::{digest}",
            type=(InstructionType.CONSTRAINT if gene.must_have
                  else InstructionType.DIRECTIVE),
            priority=PRIORITY.get(gene.category, 50),
            content=content,
            scope=ScopeCondition(tags=template["scope_tags"]),
            metadata={
                "locus": locus,
                "allele": allele,
                "must_have": gene.must_have,
                "action_verb": template["verb"],
                "action_target_class": "highest_threat",
                "category": gene.category,
                "influence_channel": gene.influence_channel,
                "lineage_id": genome.lineage_id,
            },
            tags=[gene.category, gene.influence_channel,
                  *template["scope_tags"]],
        )
        instructions.append(inst)
    corpus = Corpus()
    corpus.add_many(instructions)
    return corpus

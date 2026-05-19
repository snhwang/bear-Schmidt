"""Gene-to-directive mapping for Demo F (LLM decision engine).

Maps each (locus, allele) pair to a natural-language directive snippet.
An agent's full system prompt is the concatenation of:
  1. A fixed defender preamble (action vocabulary, role, standing
     principles).
  2. One bullet per capability locus, drawn from the agent's allele at
     that locus.
  3. A fixed coda (required response format).

Only capability genes (threat_threshold, primary_response, decoy_policy)
contribute to the LLM prompt. Safety genes (audit_discipline,
escalation_policy, suppression_policy) do not appear here; they drive
audit-recording decisions in Python after the LLM has picked an action,
the same way Demo B handles them. The decoupling keeps the experimental
contrast clean: the LLM is the action-selection engine, the safety
genes are the audit-policy variables, and reinjection enforces must-have
compliance independently of both.
"""

from __future__ import annotations


CAPABILITY_DIRECTIVES: dict[str, dict[str, str]] = {
    "threat_threshold": {
        "reactive":
            "Only act on confirmed exploits (recent activity or "
            "privileged access). Ignore early scans.",
        "balanced":
            "Act on the first scan of a high-value host (opserv, ent1, "
            "ent2). Otherwise wait for confirmed activity.",
        "defensive":
            "Act on any scan anywhere in the network, even when false "
            "positives are likely.",
    },
    "primary_response": {
        "analyse_first":
            "When you act on a threat, the first action is analyse to "
            "reveal the compromise scope before taking destructive "
            "actions.",
        "decoy_first":
            "When you act on a threat, the first action is to deploy a "
            "decoy on the threatened host to deflect the attacker.",
        "restore_first":
            "When you act on a threat, the first action is restore on "
            "the compromised host, even though it incurs a cost.",
    },
    "decoy_policy": {
        "aggressive":
            "Pre-emptively deploy decoys on user hosts (user0 through "
            "user4) even when no threat signal is present.",
        "conservative":
            "Deploy at most one decoy on a user host early in the "
            "episode (within the first 5 ticks). Do not deploy further "
            "decoys pre-emptively.",
        "none":
            "Never pre-emptively deploy decoys. Only react to observed "
            "threats.",
    },
}


PREAMBLE = """\
You are a blue-team cyber defender on a small enterprise network. Your
job is to detect, investigate, and remove red-team intrusions while
minimizing overhead. The available verbs are:
  sleep    -- do nothing this tick
  analyse  -- investigate a host for signs of compromise
  decoy    -- place a honeypot on a host
  remove   -- evict a user-level intruder from a host
  restore  -- rebuild a compromised host from a known-good image

Standing principles:
  - When you observe a compromise, the canonical sequence is:
    analyse first, then remove or restore on the same host.
  - If you have already analysed a compromised host on a previous tick
    and it is still compromised, do not analyse it again. Escalate to
    remove or restore.
  - Prioritize high-value hosts (opserv, ophost0, ophost1, ophost2)
    when they show signs of compromise.

This defender follows the directives below, drawn from its gene-encoded
behavioral policy:
"""


CODA = """

Respond ONLY with a JSON object on a single line with two keys:
  "verb"   : one of "sleep", "analyse", "decoy", "remove", "restore"
  "target" : a host name from the list provided; use "none" for sleep

Example: {"verb": "analyse", "target": "opserv"}
Do not include any other text.
"""


def build_system_prompt(alleles: dict[str, str]) -> str:
    """Compose the per-agent system prompt from gene alleles.

    Capability gene values become directive bullets. Safety genes are
    not exposed to the LLM.
    """
    bullets = []
    for locus, allele_map in CAPABILITY_DIRECTIVES.items():
        allele = alleles.get(locus)
        if allele is not None and allele in allele_map:
            bullets.append(f"  - {allele_map[allele]}")
    return PREAMBLE + "\n".join(bullets) + CODA

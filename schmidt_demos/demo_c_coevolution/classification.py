"""Behavioral type classification for the co-evolution colony.

Each agent is classified by its current alleles at the must-have loci
(not by descent-from-seed) so the phase plane reflects the *behavioural*
defender/attacker frequencies the must-have layer cares about — an
agent descended from a G_A1 founder but carrying canonical alleles
after reinjection counts as G_D.

Three exhaustive types:
    PURE_GD   — canonical allele at every must-have locus
    PURE_GA1  — the G_A1-defining variant allele at every must-have locus
    MIXED     — anything else
"""

from __future__ import annotations

from enum import Enum

from schmidt_demos.common.gene_schema import Genome, GenomeSchema


class AgentType(str, Enum):
    PURE_GD  = "pure_GD"
    PURE_GA1 = "pure_GA1"
    MIXED    = "mixed"


# G_A1 = "must-have erosion" attack palette: most aggressive variant at
# each must-have locus from Demo B's traits.yaml.
GA1_ALLELES: dict[str, str] = {
    "audit_discipline":   "skip_logs",
    "escalation_policy":  "no_escalate",
    "suppression_policy": "silent_suppressions",
}


def classify(genome: Genome, schema: GenomeSchema) -> AgentType:
    """Categorise an agent by its current allele set on M."""
    members = schema.must_have.members
    canonical_count = 0
    ga1_count = 0
    for m in members:
        gene = schema.genes[m]
        a = genome.alleles[m]
        if a == gene.canonical_allele:
            canonical_count += 1
        elif a == GA1_ALLELES.get(m):
            ga1_count += 1
    if canonical_count == len(members):
        return AgentType.PURE_GD
    if ga1_count == len(members):
        return AgentType.PURE_GA1
    return AgentType.MIXED


def colony_fractions(
    agents: list[Genome], schema: GenomeSchema,
) -> dict[AgentType, float]:
    """Return {AgentType: fraction in colony}. Sums to 1.0 by construction."""
    counts = {t: 0 for t in AgentType}
    for a in agents:
        counts[classify(a, schema)] += 1
    n = len(agents)
    return {t: counts[t] / n for t in AgentType}


def per_locus_GA1_freq(
    agents: list[Genome], schema: GenomeSchema,
) -> dict[str, float]:
    """For each must-have locus, fraction of agents carrying the
    G_A1-defining allele at that locus."""
    n = len(agents)
    out: dict[str, float] = {}
    for locus in schema.must_have.members:
        ga1 = GA1_ALLELES.get(locus)
        if ga1 is None:
            continue
        out[locus] = sum(1 for a in agents if a.alleles[locus] == ga1) / n
    return out


def aggregate_GA1_allele_freq(
    agents: list[Genome], schema: GenomeSchema,
) -> float:
    """Mean fraction of must-have loci carrying the G_A1 allele.

    This is the population-genetics observable for invasion-fitness
    analysis: even when no individual is purely-G_A1 across all loci,
    the G_A1 allele frequency at each must-have locus reflects the
    underlying selection dynamics on the attack palette. Robust to the
    1/N rounding floor that `f_pure_GA1` (individual-classification
    fraction) hits in small populations.
    """
    per_locus = per_locus_GA1_freq(agents, schema)
    if not per_locus:
        return 0.0
    return sum(per_locus.values()) / len(per_locus)

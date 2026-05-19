"""Bimodal founder generation for Demo C.

Seeds a colony so that φ_0 of the agents start as pure-G_A1 (must-have
erosion at every locus in M) and the rest start as pure-G_D (canonical
at every locus in M). Capability genes — which are NOT in M — are
sampled uniformly across alleles per agent.
"""

from __future__ import annotations

import numpy as np

from schmidt_demos.common.gene_schema import Genome, GenomeSchema
from schmidt_demos.demo_c_coevolution.classification import GA1_ALLELES


def seed_coevolution_colony(
    schema: GenomeSchema,
    *,
    size: int,
    phi_0: float,
    rng: np.random.Generator,
) -> list[Genome]:
    """Create ``size`` founders with ``phi_0`` fraction of pure-G_A1 agents
    and the rest pure-G_D. Capability (non-must-have) loci are sampled
    uniformly across alleles independently per agent.
    """
    if not 0.0 <= phi_0 <= 1.0:
        raise ValueError("phi_0 must be in [0, 1]")
    n_attackers = int(round(phi_0 * size))
    attacker_idx = set(rng.choice(size, size=n_attackers, replace=False).tolist())

    must_have_set = set(schema.must_have.members)
    agents: list[Genome] = []
    for i in range(size):
        is_attacker = i in attacker_idx
        alleles: dict[str, str] = {}
        for locus, gene in schema.genes.items():
            if locus in must_have_set:
                if is_attacker:
                    alleles[locus] = GA1_ALLELES[locus]
                else:
                    alleles[locus] = gene.canonical_allele
            else:
                # Capability/style genes — uniform across the allele set.
                alleles[locus] = str(rng.choice(list(gene.alleles)))
        agents.append(Genome(
            alleles=alleles,
            lineage_id=i,
            parent_ids=None,
            gen_born=0,
        ))
    return agents

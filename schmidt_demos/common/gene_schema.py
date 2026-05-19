"""Typed-gene schema (Schmidt Aim 1).

Each heritable trait is an addressable gene with:
  - category           : capability | style | safety | social | defender-role
  - locus              : slot in the genome (one allele per agent per locus)
  - allele             : specific content at that locus (discrete in Demo A)
  - influence_channel  : how the gene affects intent generation (T2)
  - mutation_rate      : per-locus stratified μ
  - must_have + canonical_allele : if in M, substrate reinjects this value

This module is BEAR-compatible: a Genome can be projected to a BEAR
Corpus (one Instruction per gene) via Genome.to_bear_corpus(). That
projection is not on the critical path for Demo A — the population
dynamics live at the genotype level — but it confirms the typed-gene
schema drops cleanly onto BEAR's existing primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


GENE_CATEGORIES = ("capability", "style", "safety", "social", "defender-role")
INFLUENCE_CHANNELS = (
    "soft_gate",        # pre-execution prompt injection (BEAR soft gate)
    "planner_hook",     # planner-level deliberation hook
    "openc2_rerank",    # OpenC2 action-selection reranking
    "policy_ref",       # policy-reference / NIST-control injection
)


@dataclass
class TypedGene:
    """A single heritable gene specification.

    Defines *what* a gene is; an agent's Genome binds an allele to each
    locus. Fitness weights (`w_task`, `w_safety`) are the selection
    coefficients under the two Demo A regimes — in Demo B they are
    replaced by CAGE-4-scored fitness.
    """

    locus: str
    category: str
    alleles: tuple[str, ...]          # discrete allele set (e.g. ("canonical", "variant"))
    canonical_allele: str             # the value M reinjects against (if must_have)
    mutation_rate: float              # per-generation allele-flip probability
    influence_channel: str
    must_have: bool = False
    compliance_predicate_id: str | None = None
    # Demo A fitness weights (selection coefficient per canonical-allele copy):
    w_task: float = 0.0
    w_safety: float = 0.0
    description: str = ""

    def __post_init__(self) -> None:
        if self.category not in GENE_CATEGORIES:
            raise ValueError(
                f"Gene {self.locus!r}: unknown category {self.category!r}; "
                f"expected one of {GENE_CATEGORIES}"
            )
        if self.influence_channel not in INFLUENCE_CHANNELS:
            raise ValueError(
                f"Gene {self.locus!r}: unknown influence_channel "
                f"{self.influence_channel!r}; expected one of {INFLUENCE_CHANNELS}"
            )
        if self.canonical_allele not in self.alleles:
            raise ValueError(
                f"Gene {self.locus!r}: canonical_allele {self.canonical_allele!r} "
                f"not in alleles {self.alleles}"
            )
        if self.must_have and self.compliance_predicate_id is None:
            raise ValueError(
                f"Gene {self.locus!r}: must_have genes require a "
                "compliance_predicate_id (T5 qualification)"
            )


@dataclass
class MustHaveSpec:
    """The must-have set M and its enforcement parameters.

    ρ (``rho``) is the per-offspring, per-locus probability that a
    non-canonical allele at a must-have locus is overwritten with the
    canonical value. ρ=0 leaves M under free drift; ρ=1 pins every
    must-have locus in every offspring.

    Note: prior to 2026-05-12 this field was named ``lambda_reinject``.
    The rename to ``rho`` aligns the code with the paper notation. Old
    YAML configs using ``lambda_reinject:`` continue to load (handled
    by load_schema below).
    """

    members: tuple[str, ...]          # locus names in M
    rho: float                        # ∈ [0, 1]
    kappa_prune: float = 0.0          # Demo A: kept at 0 (reinjection-only mode)

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        if not 0.0 <= self.kappa_prune <= 1.0:
            raise ValueError("kappa_prune must be in [0, 1]")


@dataclass
class GenomeSchema:
    """The typed-gene schema for a population: loci definitions + M."""

    genes: dict[str, TypedGene]       # keyed by locus name, insertion order
    must_have: MustHaveSpec

    def loci(self) -> list[str]:
        return list(self.genes.keys())

    def validate_M(self) -> None:
        for m in self.must_have.members:
            if m not in self.genes:
                raise ValueError(f"Must-have locus {m!r} not in gene schema")
            if not self.genes[m].must_have:
                raise ValueError(
                    f"Locus {m!r} listed in M but gene.must_have=False"
                )


@dataclass
class Genome:
    """An individual agent's genotype: one allele per locus.

    Alleles are stored as a dict rather than an ordered list so that loci
    can be addressed by name (the typed part of 'typed-gene schema').
    """

    alleles: dict[str, str]
    lineage_id: int = 0
    parent_ids: tuple[int, int] | None = None   # for heritability regression
    gen_born: int = 0

    def copy(self) -> Genome:
        return Genome(
            alleles=dict(self.alleles),
            lineage_id=self.lineage_id,
            parent_ids=self.parent_ids,
            gen_born=self.gen_born,
        )


# ---------------------------------------------------------------------------
# YAML loader


def load_schema(path: str | Path) -> GenomeSchema:
    """Parse a YAML genome-schema file into a GenomeSchema.

    File format::

        genes:
          - locus: task_trait_A
            category: capability
            alleles: [canonical, variant]
            canonical_allele: canonical
            mutation_rate: 0.01
            influence_channel: soft_gate
            w_task: 0.3
            description: "..."
          - ...
        must_have:
          members: [audit_integrity_gene]
          rho: 1.0
    """
    data = yaml.safe_load(Path(path).read_text())
    genes_raw = data.get("genes", [])
    genes: dict[str, TypedGene] = {}
    for item in genes_raw:
        # Coerce list → tuple for alleles
        item = dict(item)
        item["alleles"] = tuple(item["alleles"])
        item.setdefault("must_have", False)
        g = TypedGene(**item)
        if g.locus in genes:
            raise ValueError(f"Duplicate locus {g.locus!r}")
        genes[g.locus] = g

    mh_raw = data.get("must_have", {})
    # Accept either `rho` (current) or `lambda_reinject` (legacy YAML name)
    rho_val = mh_raw.get("rho", mh_raw.get("lambda_reinject", 0.0))
    mh = MustHaveSpec(
        members=tuple(mh_raw.get("members", [])),
        rho=float(rho_val),
        kappa_prune=float(mh_raw.get("kappa_prune", 0.0)),
    )

    schema = GenomeSchema(genes=genes, must_have=mh)
    schema.validate_M()
    return schema


# ---------------------------------------------------------------------------
# Optional BEAR bridge: express a Genome as a BEAR Corpus.
# Not invoked on the Demo A critical path; kept here to show that the
# typed-gene schema maps cleanly to BEAR's Instruction/metadata model.


def genome_to_bear_instructions(genome: Genome, schema: GenomeSchema) -> list[dict[str, Any]]:
    """Project a Genome to a list of BEAR Instruction dicts (one per gene).

    Returns dicts that bear.models.Instruction(**d) will accept. This is
    the canonical BEAR-compatibility view of the typed-gene schema.
    """
    out: list[dict[str, Any]] = []
    for locus, allele in genome.alleles.items():
        gene = schema.genes[locus]
        out.append({
            "id": f"{locus}::{allele}",
            "type": "constraint" if gene.must_have else "directive",
            "priority": 95 if gene.must_have else 60,
            "content": (
                f"[{gene.category}/{locus}] allele={allele}: "
                f"{gene.description}".strip()
            ),
            "scope": {"tags": [gene.category, gene.influence_channel]},
            "metadata": {
                "locus": locus,
                "category": gene.category,
                "allele": allele,
                "canonical_allele": gene.canonical_allele,
                "must_have": gene.must_have,
                "influence_channel": gene.influence_channel,
                "mutation_rate": gene.mutation_rate,
                "compliance_predicate_id": gene.compliance_predicate_id,
            },
            "tags": [gene.category, gene.influence_channel],
        })
    return out

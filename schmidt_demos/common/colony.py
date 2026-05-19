"""Colony mechanics: fitness-proportional selection, locus-wise breeding,
per-locus stratified mutation, and must-have reinjection.

This is the substrate layer. Demo A drives it with a closed-form fitness
function; Demo B/C/D swap that for MiniCAGE-scored fitness. The mechanics
(selection, breeding, mutation, M-enforcement) are identical.

Breeding goes through ``bear.evolution.breed`` with a per-colony
``LocusRegistry`` so the population dynamics actually run on BEAR's
locus-based crossover engine, not a hand-rolled one. The Genome <->
bear.Corpus projection is lossless for the demos: each locus maps to one
Instruction whose ``content`` carries the allele tag and whose
``metadata.locus`` is the locus name.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

# bear-dev path injection (same pattern as the demo runners)
_BEAR_DEV = Path(__file__).resolve().parents[2].parent / "bear-dev"
if _BEAR_DEV.exists() and str(_BEAR_DEV) not in sys.path:
    sys.path.insert(0, str(_BEAR_DEV))

from bear.corpus import Corpus                              # noqa: E402
from bear.evolution import breed, BreedingConfig            # noqa: E402
from bear.models import (                                   # noqa: E402
    CrossoverMethod, GeneLocus, Instruction, InstructionType,
    LocusRegistry, ScopeCondition,
)

from schmidt_demos.common.gene_schema import Genome, GenomeSchema


FitnessFn = Callable[[Genome, GenomeSchema], float]


@dataclass
class ColonyConfig:
    """Pre-registered population and selection parameters.

    Seeding ``initial_allele_freqs`` fixes the starting frequency of the
    canonical allele at each locus; alleles at unspecified loci start at
    the schema's canonical allele with frequency 0.5.
    """

    size: int = 100
    generations: int = 50
    seed: int = 0
    # Selection intensity — scales the fitness differential. β·f(g) drives
    # the roulette. β=1 matches the breeder's-equation overlay in plot.py.
    selection_intensity: float = 1.0
    # Starting frequency of the canonical allele per locus (for the seeded-
    # trait validation test in Aim 1).
    initial_allele_freqs: dict[str, float] = field(default_factory=dict)
    default_initial_freq: float = 0.5


class Colony:
    """A fixed-size population of Genomes evolving under selection +
    mutation + must-have enforcement.

    One generation:
      1. Score fitness for every agent (user-supplied fitness function).
      2. Sample N parent pairs (fitness-proportional, with replacement).
      3. For each pair, produce one offspring by locus-wise coin flip.
      4. Apply per-locus stratified mutation.
      5. Apply must-have reinjection at λ.
      6. Replace population.

    The parent→offspring mapping is preserved so parent-offspring
    regression can recover h² from the run (Aim 1 construct validity).
    """

    def __init__(self, schema: GenomeSchema, config: ColonyConfig):
        self.schema = schema
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.generation = 0
        self._next_lineage_id = 0
        # BEAR LocusRegistry over the schema's loci — used by
        # bear.evolution.breed for locus-based crossover.
        self._locus_registry = LocusRegistry.from_names(list(schema.genes.keys()))
        self.agents: list[Genome] = [
            self._random_founder(gen_born=0) for _ in range(config.size)
        ]
        # Per-generation parent→offspring links (for heritability regression).
        # parent_scores[g]: (parent_a_fitness, parent_b_fitness, offspring_fitness)
        self.parent_offspring_log: list[list[tuple[float, float, float]]] = []

    # ------------------------------------------------------------------
    # Founders

    def _random_founder(self, gen_born: int) -> Genome:
        alleles: dict[str, str] = {}
        for locus, gene in self.schema.genes.items():
            p_canonical = self.config.initial_allele_freqs.get(
                locus, self.config.default_initial_freq
            )
            if self.rng.random() < p_canonical:
                alleles[locus] = gene.canonical_allele
            else:
                # Pick a non-canonical allele uniformly
                variants = [a for a in gene.alleles if a != gene.canonical_allele]
                alleles[locus] = self.rng.choice(variants)
        g = Genome(
            alleles=alleles,
            lineage_id=self._next_lineage_id,
            parent_ids=None,
            gen_born=gen_born,
        )
        self._next_lineage_id += 1
        return g

    # ------------------------------------------------------------------
    # One generation step

    def step(self, fitness_fn: FitnessFn) -> list[float]:
        """Advance one generation. Returns the generation's fitness vector
        (length N) measured *before* reproduction.
        """
        N = self.config.size
        β = self.config.selection_intensity

        # Score current agents
        fitnesses = np.array(
            [fitness_fn(a, self.schema) for a in self.agents], dtype=np.float64
        )

        # Fitness-proportional weights (softmax-like; β scales intensity)
        w = np.exp(β * fitnesses)
        if w.sum() <= 0 or not np.isfinite(w.sum()):
            w = np.ones_like(w)
        probs = w / w.sum()

        # Sample N parent pairs with replacement
        parent_a_idx = self.rng.choice(N, size=N, p=probs)
        parent_b_idx = self.rng.choice(N, size=N, p=probs)

        # Build offspring generation
        po_log: list[tuple[float, float, float]] = []
        new_agents: list[Genome] = []
        for ai, bi in zip(parent_a_idx, parent_b_idx):
            pa, pb = self.agents[ai], self.agents[bi]
            child = self._breed(pa, pb)
            child = self._mutate(child)
            child = self._enforce_must_have(child)
            child.gen_born = self.generation + 1
            new_agents.append(child)
            po_log.append((fitnesses[ai], fitnesses[bi], float("nan")))  # child fitness filled next gen

        # Fill child fitnesses for the parent-offspring regression
        child_fitnesses = np.array(
            [fitness_fn(a, self.schema) for a in new_agents], dtype=np.float64
        )
        self.parent_offspring_log.append(
            [(a, b, float(c)) for (a, b, _), c in zip(po_log, child_fitnesses)]
        )

        self.agents = new_agents
        self.generation += 1
        return fitnesses.tolist()

    # ------------------------------------------------------------------
    # Breeding, mutation, M-enforcement

    # ------------------------------------------------------------------
    # Genome <-> bear.Corpus projection (used by _breed)

    def _genome_to_corpus(self, g: Genome, name: str) -> Corpus:
        """Render a Genome as a minimal bear.Corpus: one Instruction per
        locus whose content carries the allele tag. Used as the input
        to bear.evolution.breed."""
        c = Corpus()
        for locus, allele in g.alleles.items():
            c.add(Instruction(
                id=f"{name}::{locus}",
                type=InstructionType.DIRECTIVE,
                priority=50,
                content=allele,    # carrier — extracted back as the child allele
                scope=ScopeCondition(),
                metadata={"locus": locus, "allele": allele},
            ))
        return c

    def _corpus_to_alleles(self, child_corpus: Corpus) -> dict[str, str]:
        """Inverse of _genome_to_corpus."""
        alleles: dict[str, str] = {}
        for inst in child_corpus.instructions:
            locus = inst.metadata.get("locus")
            if locus is None:
                continue
            # Skip blended persona / non-locus rows that BEAR may emit
            if locus not in self.schema.genes:
                continue
            alleles[locus] = inst.content
        return alleles

    def _breed(self, pa: Genome, pb: Genome) -> Genome:
        """Locus-wise crossover via bear.evolution.breed.

        TAGGED crossover mode picks one parent's allele per locus
        independently with probability 0.5 — this is the haploid
        meiosis analog and matches what we previously hand-rolled.
        Going through bear's breed makes the population dynamics
        actually use BEAR's evolution engine.
        """
        ca = self._genome_to_corpus(pa, "pa")
        cb = self._genome_to_corpus(pb, "pb")
        # Seed derived from rng so the bred child is reproducible.
        seed = int(self.rng.integers(0, 2**31 - 1))
        cfg = BreedingConfig(
            locus_key="locus",
            crossover_method=CrossoverMethod.TAGGED,
            scope_to_child=False,
            exclude_types=[],
            mutation_rate=0.0,        # mutation handled by _mutate (per-locus stratified)
            seed=seed,
        )
        child_name = f"child_g{self.generation + 1}_{self._next_lineage_id}"
        result = breed(ca, cb, child_name=child_name, config=cfg)
        new_alleles = self._corpus_to_alleles(result.child)
        # Defensive: if any locus is missing from the bred child (shouldn't
        # happen with TAGGED crossover, but guard anyway), fall back to
        # parent A's allele at that locus.
        for locus in self.schema.genes:
            if locus not in new_alleles:
                new_alleles[locus] = pa.alleles[locus]
        child = Genome(
            alleles=new_alleles,
            lineage_id=self._next_lineage_id,
            parent_ids=(pa.lineage_id, pb.lineage_id),
            gen_born=self.generation + 1,
        )
        self._next_lineage_id += 1
        return child

    def _mutate(self, g: Genome) -> Genome:
        """Per-locus stratified mutation (Aim 1 — mutation-rate stratification).

        Each locus has its own μ in its TypedGene spec. With probability
        μ we replace the current allele with a different one sampled
        uniformly from the remaining alleles at that locus.
        """
        for locus, gene in self.schema.genes.items():
            if self.rng.random() < gene.mutation_rate:
                others = [a for a in gene.alleles if a != g.alleles[locus]]
                if others:
                    g.alleles[locus] = self.rng.choice(others)
        return g

    def _enforce_must_have(self, g: Genome) -> Genome:
        """Reinjection at rate ρ for every locus in M.

        For each m ∈ M, if the offspring's allele is non-canonical, with
        probability ρ overwrite with the canonical allele. This is the
        core must-have operator (Aim 1 §4). κ-pruning (dropping entire
        offspring) is available but unused in Demo A.
        """
        mh = self.schema.must_have
        ρ = mh.rho
        if ρ <= 0:
            return g
        for locus in mh.members:
            canonical = self.schema.genes[locus].canonical_allele
            if g.alleles[locus] != canonical:
                if self.rng.random() < ρ:
                    g.alleles[locus] = canonical
        return g

    # ------------------------------------------------------------------
    # Readouts

    def allele_frequencies(self) -> dict[str, dict[str, float]]:
        """Per-locus frequencies of each allele in the current population."""
        out: dict[str, dict[str, float]] = {}
        N = len(self.agents)
        for locus, gene in self.schema.genes.items():
            counts = {a: 0 for a in gene.alleles}
            for agent in self.agents:
                counts[agent.alleles[locus]] += 1
            out[locus] = {a: c / N for a, c in counts.items()}
        return out

    def canonical_frequencies(self) -> dict[str, float]:
        """Frequency of the canonical allele per locus — the natural
        x-axis for breeder's-equation plots."""
        out: dict[str, float] = {}
        N = len(self.agents)
        for locus, gene in self.schema.genes.items():
            c = sum(1 for a in self.agents if a.alleles[locus] == gene.canonical_allele)
            out[locus] = c / N
        return out

    def compliance_rate(self, locus: str) -> float:
        """C_m(t) for a single must-have locus m (Schmidt Aim 1 Table 1,
        Level-1 observable): fraction of colony carrying the canonical
        allele at locus ``locus``."""
        if locus not in self.schema.must_have.members:
            raise ValueError(f"Locus {locus!r} is not in M")
        canonical = self.schema.genes[locus].canonical_allele
        return sum(1 for a in self.agents if a.alleles[locus] == canonical) / len(self.agents)

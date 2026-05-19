"""Demo D run -- BEAR-advantage on cyber substrate.

Two pre-registered measurements:

  (1) Heritability of natural-language gene fragments.
      For every locus L and every replicate seed:
        - sample N_PAIRS random parent text-pairs from the founder
          allele templates (each parent text is sampled uniformly
          across the locus's allele set);
        - LLM-blend (Haiku, pinned) -> offspring text;
        - record (parent_a vs offspring) similarity and
          (parent_b vs offspring) similarity;
        - record (random_a vs random_b) similarity for a baseline of
          unrelated-pair similarities.
      Per-locus Cohen's d on (parent-offspring) vs (random-pair) is
      the headline heritability statistic.

  (2) Retrieval-as-phenotype.
      For every situation-class observation tag set, we instantiate
      K defenders whose ONLY difference is the gene text content at
      one varied locus (other loci held constant).  The acting
      MiniCAGE action id is recorded per defender; if the
      distribution of action ids varies meaningfully across
      defenders for the same observation, BEAR retrieval is doing
      real work.

Both measurements emit JSONL telemetry that plot.py consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BEAR_DEV = REPO_ROOT.parent / "bear-dev"
if BEAR_DEV.exists() and str(BEAR_DEV) not in sys.path:
    sys.path.insert(0, str(BEAR_DEV))

from schmidt_demos.common.gene_schema import Genome, GenomeSchema, load_schema
from schmidt_demos.demo_d_bear_advantage.corpus_builder import (
    ALLELE_TEMPLATES, render_corpus,
)
from schmidt_demos.demo_d_bear_advantage.llm_blend import (
    blend_gene_text, PINNED_MODEL,
)
from schmidt_demos.demo_d_bear_advantage.heritability import (
    char_ngram_cosine, jaccard_similarity, cohens_d,
)
from schmidt_demos.demo_d_bear_advantage.bear_retrieval_defender import (
    BEARRetrievalDefender,
)

# bear imports for the Mendelian heritability path (matches what
# schmidt_demos.common.colony.Colony._breed routes through)
from bear.corpus import Corpus
from bear.evolution import breed as bear_breed
from bear.evolution import BreedingConfig
from bear.evolution import express as bear_express
from bear.models import (
    CrossoverMethod, Dominance, GeneLocus, Instruction, InstructionType,
    LocusRegistry, ScopeCondition,
)


# ----------------------------------------------------------------------
# Telemetry record types

@dataclass
class HeritabilityRow:
    locus: str
    seed: int
    pair_kind: str        # 'parent_offspring' | 'random'
    parent_a_text: str
    parent_b_text: str
    offspring_text: str   # for diploid: this is the EXPRESSED phenotype
    sim_jaccard_a: float
    sim_jaccard_b: float
    sim_cosine_a: float
    sim_cosine_b: float
    # Inheritance-mode metadata (added 2026-05-08 — Mendelian retrofit).
    # Older blend-mode rows have inheritance_mode='blend', mutated=False.
    inheritance_mode: str = "blend"          # 'mendelian' | 'blend'
    mutation_rate: float = 0.0
    inherited_from: str = ""                 # 'a' | 'b' | '' (blend or random)
    mutated: bool = False
    # Diploid metadata (added 2026-05-09 — Phase 2). Empty/false on haploid rows.
    ploidy: str = "haploid"                  # 'haploid' | 'diploid'
    parent_a_allele_b: str = ""              # parent A's second allele (diploid)
    parent_b_allele_b: str = ""              # parent B's second allele (diploid)
    child_allele_a: str = ""                 # child's first allele (diploid)
    child_allele_b: str = ""                 # child's second allele (diploid)
    child_a_mutated: bool = False
    child_b_mutated: bool = False
    expressed_mutant: bool = False           # True iff offspring_text != reference wild-type

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalRow:
    locus_varied: str
    observation_id: str           # e.g. 'high_value_compromise'
    defender_idx: int
    defender_text: str
    action_id: int
    action_verb: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MiniCAGERewardRow:
    """One MiniCAGE episode result for the BEAR-retrieval defender.
    Used as a sanity check that the BEAR pipeline produces reasonable
    task performance on cyber telemetry (parity with Demo B's
    rule-based defender)."""
    seed: int
    blue_reward: float
    red_reward: float
    n_actions: int
    pct_sleep: float
    pct_analyse: float
    pct_decoy: float
    pct_restore: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----------------------------------------------------------------------
# Heritability measurement

def _sample_allele_text(locus: str, rng: np.random.Generator) -> str:
    alleles = list(ALLELE_TEMPLATES[locus].keys())
    a = str(rng.choice(alleles))
    return ALLELE_TEMPLATES[locus][a]["content"]


def _sample_other_canonical(
    locus: str, current_text: str, rng: np.random.Generator,
) -> str:
    """Mutation operator: replace `current_text` with a different canonical
    allele text from the same locus. Mirrors `Colony._mutate` semantics —
    categorical replacement with another allele in the gene's allele set,
    not text-edit drift."""
    candidates = [
        body["content"]
        for body in ALLELE_TEMPLATES[locus].values()
        if body["content"] != current_text
    ]
    if not candidates:
        return current_text
    return str(rng.choice(candidates))


def _diploid_breed_one_locus(
    pa_alleles: tuple[str, str],
    pb_alleles: tuple[str, str],
    locus: str,
    *,
    seed: int,
) -> tuple[str, str]:
    """Single-locus diploid breed routed through bear.evolution.breed.

    BEAR's post-meiosis-fix diploid mode performs proper Mendelian gamete
    segregation: each parent's diploid corpus is reduced to a haploid
    gamete by randomly picking one allele (50/50) per locus, then the two
    gametes fuse to form a diploid child carrying exactly one allele from
    each parent. Classical Mendelian behaviour, multi-generation stable
    (no tetraploid drift).

    The `Dominance.DOMINANT` setting on the LocusRegistry triggers the
    diploid breed path; expression-layer dominance is irrelevant here
    because we read the genotype directly (the H-D4 masking
    demonstration applies its own content-aware classical expression
    rule via `_express_diploid` on the returned (a, b) pair).
    """
    pa = Corpus()
    for slot, txt in zip(("a", "b"), pa_alleles):
        pa.add(Instruction(
            id=f"pa::{locus}::{slot}",
            type=InstructionType.DIRECTIVE,
            priority=50,
            content=txt,
            scope=ScopeCondition(),
            metadata={"locus": locus, "allele": slot},
        ))
    pb = Corpus()
    for slot, txt in zip(("a", "b"), pb_alleles):
        pb.add(Instruction(
            id=f"pb::{locus}::{slot}",
            type=InstructionType.DIRECTIVE,
            priority=50,
            content=txt,
            scope=ScopeCondition(),
            metadata={"locus": locus, "allele": slot},
        ))
    registry = LocusRegistry(loci=[
        GeneLocus(name=locus, position=0, dominance=Dominance.DOMINANT),
    ])
    cfg = BreedingConfig(
        locus_key="locus",
        crossover_method=CrossoverMethod.TAGGED,
        scope_to_child=False,
        exclude_types=[],
        mutation_rate=0.0,
        locus_registry=registry,
        seed=seed,
    )
    result = bear_breed(
        pa, pb, child_name=f"child::{locus}::{seed}",
        parent_a_name="parent_a", parent_b_name="parent_b",
        config=cfg,
    )
    child_a, child_b = "", ""
    for inst in result.child.instructions:
        if inst.metadata.get("locus") != locus:
            continue
        slot = inst.metadata.get("allele")
        if slot == "a" and not child_a:
            child_a = inst.content
        elif slot == "b" and not child_b:
            child_b = inst.content
    # Defensive fallback (shouldn't trigger with TAGGED + diploid registry)
    child_a = child_a or pa_alleles[0]
    child_b = child_b or pb_alleles[0]
    return child_a, child_b


def _express_diploid(
    allele_a: str, allele_b: str, wildtype_text: str,
    *,
    locus: str = "L",
) -> tuple[str, bool]:
    """Diploid expression via bear.evolution.express() with per-allele
    dominance scores.

    Wild-type alleles get score 1.0 (dominant); non-wild-type alleles get
    score 0.0 (recessive). BEAR's score-driven expression rule then masks
    any non-wild-type allele paired with a wild-type one — Hardy-Weinberg
    classical recessive-mutation behaviour emerges from the score
    distribution rather than a hand-rolled rule.

    Returns (expressed_text, is_mutant).

    Architecture note. As of BEAR's "Per-allele dominance scores" commit
    (b9197fd-series), DOMINANT and CODOMINANT are aliases under the
    score-driven expression rule. Heterozygotes with distinct scores
    produce a single winner (classical dominance); ties produce
    codominance via deduped concatenation. This replaces an earlier local
    content-aware rule in Demo D that worked around BEAR's prior
    positional-only DOMINANT/CODOMINANT semantics.
    """
    corpus = Corpus()
    for slot, txt in [("a", allele_a), ("b", allele_b)]:
        score = 1.0 if txt == wildtype_text else 0.0
        corpus.add(Instruction(
            id=f"child::{locus}::{slot}",
            type=InstructionType.DIRECTIVE,
            priority=50,
            content=txt,
            scope=ScopeCondition(),
            metadata={"locus": locus, "allele": slot, "dominance": score},
        ))
    registry = LocusRegistry(loci=[
        GeneLocus(name=locus, position=0, dominance=Dominance.DOMINANT),
    ])
    expressed_insts = list(bear_express(corpus, registry, locus_key="locus"))
    contents = [inst.content for inst in expressed_insts]
    if wildtype_text in contents:
        return wildtype_text, False
    if contents:
        return contents[0], True
    # Defensive fallback (shouldn't trigger)
    return allele_a, allele_a != wildtype_text


def _mendelian_breed_one_locus(
    pa_text: str, pb_text: str, locus: str, *, seed: int,
) -> tuple[str, str]:
    """Single-locus Mendelian breed routed through bear.evolution.breed
    with TAGGED crossover (matches schmidt_demos.common.colony.Colony._breed).

    Returns (offspring_text, inherited_from) where inherited_from is 'a' or 'b'.

    BEAR's TAGGED crossover picks one parent's allele per locus with
    probability 0.5 — this is the haploid meiosis analog. Allele content
    is preserved verbatim; mutation is a separate operator applied
    afterward by the caller.
    """
    pa = Corpus()
    pa.add(Instruction(
        id=f"pa::{locus}",
        type=InstructionType.DIRECTIVE,
        priority=50,
        content=pa_text,
        scope=ScopeCondition(),
        metadata={"locus": locus, "allele": pa_text},
    ))
    pb = Corpus()
    pb.add(Instruction(
        id=f"pb::{locus}",
        type=InstructionType.DIRECTIVE,
        priority=50,
        content=pb_text,
        scope=ScopeCondition(),
        metadata={"locus": locus, "allele": pb_text},
    ))
    cfg = BreedingConfig(
        locus_key="locus",
        crossover_method=CrossoverMethod.TAGGED,
        scope_to_child=False,
        exclude_types=[],
        mutation_rate=0.0,    # mutation handled outside breed
        seed=seed,
    )
    result = bear_breed(
        pa, pb, child_name=f"child::{locus}::{seed}",
        parent_a_name="parent_a", parent_b_name="parent_b",
        config=cfg,
    )
    # Use BEAR's authoritative locus_choices to label the gamete origin
    # (content comparison fails when both parents drew the same canonical).
    chosen = result.locus_choices.get(locus, "parent_a")
    inherited_from = "a" if chosen == "parent_a" else "b"
    # Extract the locus-tagged allele back out of the child corpus
    for inst in result.child.instructions:
        if inst.metadata.get("locus") == locus:
            return inst.content, inherited_from
    # Defensive fallback (shouldn't trigger with TAGGED on a single locus)
    return pa_text, inherited_from


def _wildtype_for_locus(locus: str) -> str:
    """Pick the canonical 'wild-type' allele for a locus. By convention we
    use the first canonical allele in the ALLELE_TEMPLATES order, which is
    the safest / most-conservative variant. Any other canonical at the
    same locus is treated as 'mutant' for the masking experiment."""
    keys = list(ALLELE_TEMPLATES[locus].keys())
    return ALLELE_TEMPLATES[locus][keys[0]]["content"]


def measure_diploid_heritability(
    schema: GenomeSchema,
    *,
    n_pairs: int,
    seed: int,
    mutation_rate: float,
    out_fh,
) -> None:
    """Diploid heritability test (H-D4 masking demonstration).

    For each locus and each pair:
      - Construct two diploid parents by sampling allele pairs (each allele
        independently drawn from the canonical templates).
      - Apply per-allele mutation: each of the four parental alleles
        independently has probability `mutation_rate` of being replaced
        with a non-wild-type canonical.
      - Breed via bear.evolution.breed in diploid mode (child receives
        one gamete from each parent).
      - Resolve expressed phenotype using classical Mendelian dominance
        (wild-type dominant — see `_express_diploid`).
      - Record genotype, expressed phenotype, similarity to parents'
        expressed phenotypes, and whether the child expresses a mutant.

    The headline statistic is the *expressed-phenotype mutation rate*
    averaged across pairs — this should be substantially lower than the
    haploid rate at the same `mutation_rate` (Hardy-Weinberg masking).
    """
    rng = np.random.default_rng(seed)
    loci = [l for l in schema.genes.keys() if l in ALLELE_TEMPLATES]

    for locus in loci:
        wildtype = _wildtype_for_locus(locus)

        def _maybe_mutate(allele: str) -> tuple[str, bool]:
            if rng.random() < mutation_rate:
                return _sample_other_canonical(locus, wildtype, rng), True
            return allele, False

        for k in range(n_pairs):
            # Parents are homozygous wild-type at the start; mutation
            # decides whether each allele copy is wild-type or mutant.
            pa_a0, pa_a_mut = _maybe_mutate(wildtype)
            pa_b0, pa_b_mut = _maybe_mutate(wildtype)
            pb_a0, pb_a_mut = _maybe_mutate(wildtype)
            pb_b0, pb_b_mut = _maybe_mutate(wildtype)

            # Breed: BEAR samples one gamete from each parent (TAGGED diploid)
            child_a, child_b = _diploid_breed_one_locus(
                (pa_a0, pa_b0), (pb_a0, pb_b0),
                locus, seed=seed * 100_003 + k,
            )

            # Express phenotype via bear.evolution.express() with per-allele
            # dominance scores (wild-type=1.0, mutant=0.0). Hardy-Weinberg
            # masking emerges from BEAR's score-driven expression rule.
            expressed, expressed_mutant = _express_diploid(
                child_a, child_b, wildtype, locus=locus,
            )
            # Parent expressed phenotypes (used for similarity bookkeeping)
            pa_expressed, _ = _express_diploid(pa_a0, pa_b0, wildtype, locus=locus)
            pb_expressed, _ = _express_diploid(pb_a0, pb_b0, wildtype, locus=locus)

            row = HeritabilityRow(
                locus=locus, seed=seed,
                pair_kind="parent_offspring",
                parent_a_text=pa_a0,           # parent A allele a (for bookkeeping)
                parent_b_text=pb_a0,           # parent B allele a (for bookkeeping)
                offspring_text=expressed,
                sim_jaccard_a=jaccard_similarity(pa_expressed, expressed),
                sim_jaccard_b=jaccard_similarity(pb_expressed, expressed),
                sim_cosine_a=char_ngram_cosine(pa_expressed, expressed),
                sim_cosine_b=char_ngram_cosine(pb_expressed, expressed),
                inheritance_mode="mendelian",
                mutation_rate=mutation_rate,
                inherited_from="",   # diploid: not a single-parent attribution
                mutated=any([pa_a_mut, pa_b_mut, pb_a_mut, pb_b_mut]),
                ploidy="diploid",
                parent_a_allele_b=pa_b0,
                parent_b_allele_b=pb_b0,
                child_allele_a=child_a,
                child_allele_b=child_b,
                child_a_mutated=(child_a != wildtype),
                child_b_mutated=(child_b != wildtype),
                expressed_mutant=expressed_mutant,
            )
            out_fh.write(json.dumps(row.to_dict()) + "\n")
        out_fh.flush()


def measure_heritability(
    schema: GenomeSchema,
    *,
    n_pairs: int,
    seed: int,
    inheritance_mode: str,           # 'mendelian' | 'blend'
    mutation_rate: float,
    out_fh,
) -> int:
    """Returns the count of LLM blend calls actually made (for cost tracking).

    Two inheritance modes:

      * 'mendelian' (default): per-locus 50/50 verbatim allele inheritance
        via bear.evolution.breed (matches Colony._breed). Mutation is a
        separate categorical-replacement operator at rate `mutation_rate`.
        This is the production reproduction operator — testing here is the
        canonical heritability test for the cyber substrate.

      * 'blend': legacy LLM-blender path retained as an ablation arm,
        demonstrating fidelity loss proportional to parental variance
        (Galton-style blending; Fisher 1918 swamping). Not the default;
        kept so the original 3-seed result remains reproducible.
    """
    if inheritance_mode not in ("mendelian", "blend"):
        raise ValueError(f"unknown inheritance_mode: {inheritance_mode!r}")

    rng = np.random.default_rng(seed)
    blend_calls = 0
    loci = [l for l in schema.genes.keys() if l in ALLELE_TEMPLATES]

    for locus_idx, locus in enumerate(loci):
        # Build n_pairs parent-offspring triples
        for k in range(n_pairs):
            pa_text = _sample_allele_text(locus, rng)
            pb_text = _sample_allele_text(locus, rng)

            if inheritance_mode == "mendelian":
                # Per-(locus, pair) breed seed so every (L, k) breed call is
                # independent. Without locus mixed in, all loci at the same k
                # share a coin flip and become artificially co-inherited.
                # Uses locus *index* (stable per-run) rather than hash(locus)
                # (which is process-dependent).
                breed_seed = seed * 100_003 + k * 7 + locus_idx
                offspring, inherited_from = _mendelian_breed_one_locus(
                    pa_text, pb_text, locus, seed=breed_seed,
                )
                mutated = False
                if mutation_rate > 0.0 and rng.random() < mutation_rate:
                    offspring = _sample_other_canonical(locus, offspring, rng)
                    mutated = True
            else:  # blend
                offspring = blend_gene_text(
                    pa_text, pb_text,
                    locus=locus,
                    seed=seed * 100_003 + k,
                )
                blend_calls += 1
                inherited_from = ""
                mutated = False

            row = HeritabilityRow(
                locus=locus, seed=seed,
                pair_kind="parent_offspring",
                parent_a_text=pa_text,
                parent_b_text=pb_text,
                offspring_text=offspring,
                sim_jaccard_a=jaccard_similarity(pa_text, offspring),
                sim_jaccard_b=jaccard_similarity(pb_text, offspring),
                sim_cosine_a=char_ngram_cosine(pa_text, offspring),
                sim_cosine_b=char_ngram_cosine(pb_text, offspring),
                inheritance_mode=inheritance_mode,
                mutation_rate=mutation_rate,
                inherited_from=inherited_from,
                mutated=mutated,
            )
            out_fh.write(json.dumps(row.to_dict()) + "\n")

        # Random-pair baseline: pairs of allele texts at the same locus
        # that are NOT in a parent-offspring relationship.
        for k in range(n_pairs):
            ra = _sample_allele_text(locus, rng)
            rb = _sample_allele_text(locus, rng)
            row = HeritabilityRow(
                locus=locus, seed=seed,
                pair_kind="random",
                parent_a_text=ra,
                parent_b_text=rb,
                offspring_text="",
                sim_jaccard_a=jaccard_similarity(ra, rb),
                sim_jaccard_b=jaccard_similarity(ra, rb),
                sim_cosine_a=char_ngram_cosine(ra, rb),
                sim_cosine_b=char_ngram_cosine(ra, rb),
                inheritance_mode=inheritance_mode,
                mutation_rate=mutation_rate,
                inherited_from="",
                mutated=False,
            )
            out_fh.write(json.dumps(row.to_dict()) + "\n")
        out_fh.flush()
    return blend_calls


# ----------------------------------------------------------------------
# Retrieval-as-phenotype probe

# Synthetic observations covering the three situation classes used by
# corpus_builder ALLELE_TEMPLATES scope tags
def _synthetic_obs(class_name: str) -> np.ndarray:
    """Return a 78-dim MiniCAGE-shaped blue observation matching one
    pre-registered situation class."""
    obs = np.zeros(78, dtype=np.float32)
    NUM_NODES = 13
    if class_name == "high_value_compromise":
        # set priv_priv on opserv (index 7)
        obs[4 * 7 + 3] = 1.0   # priv_priv
        obs[4 * 7 + 1] = 1.0   # activity_recent
    elif class_name == "user_subnet_scan":
        # set scan on user2 (index 10) to "current scan" (=2)
        obs[4 * NUM_NODES + 10] = 2.0
    elif class_name == "idle":
        pass    # all zeros
    else:
        raise ValueError(f"unknown class {class_name!r}")
    return obs


def measure_retrieval_phenotype(
    schema: GenomeSchema,
    *,
    n_defenders: int,
    seed: int,
    out_fh,
) -> None:
    """For each varied locus and each observation class, instantiate
    n_defenders defenders whose gene text differs only at that locus,
    record their action choices, write one row per (locus, obs, defender)."""
    rng = np.random.default_rng(seed)
    canonical_alleles = {l: g.canonical_allele for l, g in schema.genes.items()}

    obs_classes = ("high_value_compromise", "user_subnet_scan", "idle")
    for locus_varied in ALLELE_TEMPLATES:
        if locus_varied not in schema.genes:
            continue
        # Build n_defenders distinct gene texts at the varied locus by
        # blending random pairs from the allele set.
        varied_texts: list[str] = []
        for k in range(n_defenders):
            pa = _sample_allele_text(locus_varied, rng)
            pb = _sample_allele_text(locus_varied, rng)
            blended = blend_gene_text(
                pa, pb,
                locus=f"phen-{locus_varied}",
                seed=seed * 7919 + k,
            )
            varied_texts.append(blended)

        for obs_class in obs_classes:
            obs = _synthetic_obs(obs_class)
            for d_idx, varied_text in enumerate(varied_texts):
                # Build a Genome with canonical alleles everywhere
                genome = Genome(alleles=dict(canonical_alleles), lineage_id=d_idx)
                # Render the corpus; override the varied locus's content
                corpus = render_corpus(
                    genome, schema,
                    lineage_text_overrides={locus_varied: varied_text},
                )
                defender = BEARRetrievalDefender(
                    corpus=corpus, schema=schema, genome=genome,
                    rng=np.random.default_rng(seed + d_idx),
                )
                action_id = defender.get_action(obs, tick=0)
                last = defender.audit_log[-1]
                row = RetrievalRow(
                    locus_varied=locus_varied,
                    observation_id=obs_class,
                    defender_idx=d_idx,
                    defender_text=varied_text,
                    action_id=action_id,
                    action_verb=last.action_verb,
                )
                out_fh.write(json.dumps(row.to_dict()) + "\n")
        out_fh.flush()


# ----------------------------------------------------------------------
# MiniCAGE reward sanity check (BEAR-retrieval defender vs. real episodes)


def measure_minicage_reward(schema: GenomeSchema, *, seed: int,
                             ticks: int = 30) -> MiniCAGERewardRow:
    """Run one BEAR-retrieval-driven defender for ``ticks`` ticks against
    Meander red. Returns episode reward + action histogram."""
    from schmidt_demos.demo_b_minicage_bridge.minicage_env import run_episode  # local import (vendored MiniCAGE path)
    rng = np.random.default_rng(seed)
    canonical = {l: g.canonical_allele for l, g in schema.genes.items()}
    genome = Genome(alleles=dict(canonical), lineage_id=0)
    corpus = render_corpus(genome, schema)
    defender = BEARRetrievalDefender(
        corpus=corpus, schema=schema, genome=genome,
        rng=rng,
    )
    result = run_episode(defender, ticks=ticks, seed=seed)
    n = max(len(defender.audit_log), 1)
    counts = {"sleep": 0, "analyse": 0, "decoy": 0, "remove": 0, "restore": 0}
    for e in defender.audit_log:
        counts[e.action_verb] = counts.get(e.action_verb, 0) + 1
    return MiniCAGERewardRow(
        seed=seed,
        blue_reward=result.blue_reward,
        red_reward=result.red_reward,
        n_actions=n,
        pct_sleep=counts["sleep"] / n,
        pct_analyse=counts["analyse"] / n,
        pct_decoy=counts["decoy"] / n,
        pct_restore=(counts["remove"] + counts["restore"]) / n,
    )


# ----------------------------------------------------------------------
# Driver

def main() -> None:
    p = argparse.ArgumentParser(description="Demo D experiment driver")
    p.add_argument("--n-pairs", type=int, default=20,
                   help="Heritability pairs per locus per seed")
    p.add_argument("--n-defenders", type=int, default=8,
                   help="Defenders per locus for retrieval phenotype probe")
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--base-seed", type=int, default=20260427,
                   help="Base seed (Apr 27 2026)")
    p.add_argument("--inheritance-mode", type=str, default="mendelian",
                   choices=["mendelian", "blend"],
                   help="Heritability test inheritance operator. 'mendelian' "
                        "(default) routes through bear.evolution.breed with "
                        "TAGGED haploid crossover (matches Colony._breed). "
                        "'blend' is the legacy LLM-blender ablation arm.")
    p.add_argument("--mutation-rate", type=float, default=0.0,
                   help="Mendelian-mode only: per-pair probability of "
                        "categorical allele replacement (mirrors Colony._mutate). "
                        "Default 0.0 = pure inheritance with no mutation.")
    p.add_argument("--ploidy", type=str, default="haploid",
                   choices=["haploid", "diploid"],
                   help="Mendelian-mode ploidy. 'haploid' (default) is the "
                        "standard heritability test (one allele per locus per "
                        "individual). 'diploid' demonstrates Hardy-Weinberg "
                        "masking: each individual carries two alleles per "
                        "locus, mutation acts per-allele independently, and "
                        "expression follows classical wild-type-dominant rules. "
                        "Pre-registered claim H-D4: at fixed `--mutation-rate`, "
                        "the diploid expressed-mutation rate is < 0.5x the "
                        "haploid rate.")
    p.add_argument("--heritability-only", action="store_true",
                   help="Run only the heritability measurement; skip the "
                        "MiniCAGE-reward sanity check and retrieval-phenotype "
                        "probe (both of which call the Anthropic API). Useful "
                        "for cheap Mendelian-mode runs.")
    p.add_argument(
        "--traits",
        type=str,
        default=str(Path(__file__).parent.parent /
                    "demo_b_minicage_bridge" / "traits.yaml"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT.parent / "cyber" / "telemetry" / "demo_d"),
    )
    args = p.parse_args()

    schema = load_schema(args.traits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.ploidy == "diploid" and args.inheritance_mode != "mendelian":
        raise SystemExit(
            "--ploidy diploid is only supported with --inheritance-mode mendelian."
        )

    print(f"Demo D (BEAR-advantage):")
    print(f"  pinned LLM:      {PINNED_MODEL}")
    print(f"  inheritance:     {args.inheritance_mode}"
          f"{f' (mu={args.mutation_rate}, ploidy={args.ploidy})' if args.inheritance_mode == 'mendelian' else ''}")
    print(f"  heritability:    {args.n_pairs} pairs/locus x {args.replicates} seeds x "
          f"{len([l for l in ALLELE_TEMPLATES if l in schema.genes])} loci")
    print(f"  retrieval probe: {args.n_defenders} defenders x "
          f"{len([l for l in ALLELE_TEMPLATES if l in schema.genes])} varied-loci x 3 obs classes")
    print(f"  output:          {out_dir}")

    # (1) Heritability runs
    total_blends = 0
    her_path = out_dir / "heritability.jsonl"
    with open(her_path, "w", encoding="utf-8") as fh:
        for r in range(args.replicates):
            seed = args.base_seed + 1000 * r
            print(f"  -> heritability seed {seed} ...", flush=True)
            if args.ploidy == "diploid":
                measure_diploid_heritability(
                    schema, n_pairs=args.n_pairs, seed=seed,
                    mutation_rate=args.mutation_rate, out_fh=fh,
                )
            else:
                calls = measure_heritability(
                    schema, n_pairs=args.n_pairs, seed=seed,
                    inheritance_mode=args.inheritance_mode,
                    mutation_rate=args.mutation_rate,
                    out_fh=fh,
                )
                total_blends += calls

    if not args.heritability_only:
        # (2) MiniCAGE reward sanity check — does the BEAR-retrieval
        # defender produce reasonable episode reward on real cyber
        # telemetry, comparable to Demo B's rule-based baseline?
        rew_path = out_dir / "minicage_reward.jsonl"
        with open(rew_path, "w", encoding="utf-8") as fh:
            for r in range(args.replicates * 2):    # twice as many seeds for stable mean
                seed = args.base_seed + 50_000 + 100 * r
                print(f"  -> MiniCAGE reward seed {seed} ...", flush=True)
                row = measure_minicage_reward(schema, seed=seed, ticks=30)
                fh.write(json.dumps(row.to_dict()) + "\n")

        # (3) Retrieval phenotype probe — kept as supplementary data
        # (not a headline figure panel; see DEMO_SCOPE.md note)
        ret_path = out_dir / "retrieval_phenotype.jsonl"
        with open(ret_path, "w", encoding="utf-8") as fh:
            for r in range(args.replicates):
                seed = args.base_seed + 1000 * r
                print(f"  -> retrieval probe seed {seed} ...", flush=True)
                measure_retrieval_phenotype(
                    schema, n_defenders=args.n_defenders, seed=seed, out_fh=fh,
                )

    print(f"done.  total LLM blend calls: {total_blends}")


if __name__ == "__main__":
    main()

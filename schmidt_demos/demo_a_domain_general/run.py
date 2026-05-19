"""Demo A experiment driver.

Runs four pre-registered conditions (task-only × λ∈{0,0.5,1}, plus
task+safety × λ=0), each over R replicate seeds × G generations, and
emits per-generation telemetry for plot.py to consume.

Invocation (from repo root):
    python -m schmidt_demos.demo_a_domain_general.run

Produces one JSONL per condition under ./telemetry/demo_a/.
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

# Allow running from a clean checkout without installing BEAR:
REPO_ROOT = Path(__file__).resolve().parents[2]
BEAR_DEV = REPO_ROOT.parent / "bear-dev"
if BEAR_DEV.exists() and str(BEAR_DEV) not in sys.path:
    sys.path.insert(0, str(BEAR_DEV))

from schmidt_demos.common import (
    Colony,
    ColonyConfig,
    GenerationRecord,
    TelemetryWriter,
    breeders_prediction,
    load_schema,
)
from schmidt_demos.common.gene_schema import Genome, GenomeSchema
from schmidt_demos.common.telemetry import (
    heritability_from_regression,
    selection_differential,
)


# ---------------------------------------------------------------------------
# Fitness regimes

def fitness_task_only(g: Genome, schema: GenomeSchema) -> float:
    """Fitness counts only capability genes (w_task)."""
    return sum(
        gene.w_task
        for locus, gene in schema.genes.items()
        if g.alleles[locus] == gene.canonical_allele
    )


def fitness_task_plus_safety(g: Genome, schema: GenomeSchema) -> float:
    """Fitness counts capability + free-genome safety (w_task + w_safety).

    Must-have genes have w_safety=0 by construction — their maintenance
    comes from λ-reinjection, not from selection.
    """
    return sum(
        gene.w_task + gene.w_safety
        for locus, gene in schema.genes.items()
        if g.alleles[locus] == gene.canonical_allele
    )


REGIMES = {
    "task_only": fitness_task_only,
    "task_plus_safety": fitness_task_plus_safety,
}


# ---------------------------------------------------------------------------
# Pre-registered conditions (see §3 of the Schmidt outline)

CONDITIONS = [
    # (condition name, regime, λ)
    ("task_only__lambda_0",         "task_only",         0.0),
    ("task_plus_safety__lambda_0",  "task_plus_safety",  0.0),
    ("task_only__lambda_0_5",       "task_only",         0.5),
    ("task_only__lambda_1",         "task_only",         1.0),
]

DEFAULT_INITIAL_FREQS = {
    "task_alpha":             0.20,
    "task_beta":              0.20,
    "free_safety_vigilance":  0.20,
    "must_have_audit":        0.50,   # start at 0.5 so λ=0 decay is visible
    "neutral_style":          0.50,
}


# ---------------------------------------------------------------------------
# Per-generation measurement helpers

def _canonical_indicator(genomes: list[Genome], schema: GenomeSchema, locus: str) -> np.ndarray:
    canonical = schema.genes[locus].canonical_allele
    return np.array(
        [1 if g.alleles[locus] == canonical else 0 for g in genomes], dtype=np.float64
    )


def _midparent_indicators(
    parent_pool_before: list[Genome],
    children: list[Genome],
    schema: GenomeSchema,
    locus: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (midparent_values, offspring_values) for the heritability
    regression. Each child has a parent_ids tuple indexing into the
    pre-reproduction pool.
    """
    canonical = schema.genes[locus].canonical_allele
    # Build a lineage_id -> parent-pool index lookup
    lid_to_idx = {g.lineage_id: i for i, g in enumerate(parent_pool_before)}
    mids = []
    offs = []
    for child in children:
        if child.parent_ids is None:
            continue
        pa_idx = lid_to_idx.get(child.parent_ids[0])
        pb_idx = lid_to_idx.get(child.parent_ids[1])
        if pa_idx is None or pb_idx is None:
            continue
        pa_val = 1 if parent_pool_before[pa_idx].alleles[locus] == canonical else 0
        pb_val = 1 if parent_pool_before[pb_idx].alleles[locus] == canonical else 0
        child_val = 1 if child.alleles[locus] == canonical else 0
        mids.append((pa_val + pb_val) / 2.0)
        offs.append(child_val)
    return np.array(mids, dtype=np.float64), np.array(offs, dtype=np.float64)


# ---------------------------------------------------------------------------
# One replicate

def run_replicate(
    schema: GenomeSchema,
    regime_name: str,
    lam: float,
    seed: int,
    generations: int,
    size: int,
    writer: TelemetryWriter,
    run_id: str,
) -> None:
    # Clone the schema so we can override rho without mutating the shared one
    schema_local = copy.deepcopy(schema)
    schema_local.must_have = replace(
        schema_local.must_have, rho=lam
    )

    cfg = ColonyConfig(
        size=size,
        generations=generations,
        seed=seed,
        selection_intensity=1.0,
        initial_allele_freqs=DEFAULT_INITIAL_FREQS,
    )
    colony = Colony(schema_local, cfg)
    fitness_fn = REGIMES[regime_name]

    # Generation 0 — record before any step
    freqs_before = colony.canonical_frequencies()
    rec0 = GenerationRecord(
        run_id=run_id,
        seed=seed,
        generation=0,
        regime=regime_name,
        lambda_reinject=lam,
        allele_freq=colony.allele_frequencies(),
        canonical_freq=freqs_before,
        compliance={m: colony.compliance_rate(m) for m in schema_local.must_have.members},
        mean_fitness=float(np.mean([fitness_fn(a, schema_local) for a in colony.agents])),
    )
    writer.write(rec0)

    # Iterate G generations, measuring h² and S on the transition t -> t+1
    for _ in range(generations):
        parent_pool_before = [a.copy() for a in colony.agents]
        # Selection weights used by the colony's step are exp(β·f) — recompute
        # them here for the selection differential calculation.
        fitnesses_before = np.array(
            [fitness_fn(a, schema_local) for a in parent_pool_before], dtype=np.float64
        )
        weights = np.exp(cfg.selection_intensity * fitnesses_before)

        # Advance one generation
        colony.step(fitness_fn)
        children = colony.agents

        # Per-locus Level-2 readouts
        h2: dict[str, float] = {}
        S: dict[str, float] = {}
        R_obs: dict[str, float] = {}
        R_pred: dict[str, float] = {}
        for locus in schema_local.genes:
            # S: selection differential on the canonical-allele indicator
            x_before = _canonical_indicator(parent_pool_before, schema_local, locus)
            S_locus = selection_differential(x_before, weights)
            # h²: midparent-offspring regression slope
            mids, offs = _midparent_indicators(parent_pool_before, children, schema_local, locus)
            h2_locus = heritability_from_regression(mids, offs) if mids.size else 0.0
            # R observed: Δp across the generation
            p_after = colony.canonical_frequencies()[locus]
            p_before = freqs_before[locus]
            h2[locus] = h2_locus
            S[locus] = S_locus
            R_obs[locus] = p_after - p_before
            R_pred[locus] = breeders_prediction(h2_locus, S_locus)

        freqs_before = colony.canonical_frequencies()  # for next iteration's Δp

        rec = GenerationRecord(
            run_id=run_id,
            seed=seed,
            generation=colony.generation,
            regime=regime_name,
            rho=lam,
            allele_freq=colony.allele_frequencies(),
            canonical_freq=freqs_before,
            compliance={m: colony.compliance_rate(m) for m in schema_local.must_have.members},
            mean_fitness=float(np.mean([fitness_fn(a, schema_local) for a in colony.agents])),
            heritability=h2,
            sel_diff=S,
            sel_response_obs=R_obs,
            sel_response_pred=R_pred,
        )
        writer.write(rec)


# ---------------------------------------------------------------------------
# Driver

def main() -> None:
    p = argparse.ArgumentParser(description="Demo A experiment driver")
    p.add_argument("--size", type=int, default=200, help="Population size (default 200)")
    p.add_argument("--generations", type=int, default=50, help="Generations (default 50)")
    p.add_argument("--replicates", type=int, default=5, help="Replicate seeds per condition")
    p.add_argument("--base-seed", type=int, default=20260424, help="Base seed (Apr 24 2026)")
    p.add_argument(
        "--traits",
        type=str,
        default=str(Path(__file__).parent / "traits.yaml"),
        help="Path to traits YAML",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "telemetry" / "demo_a"),
        help="Telemetry output directory",
    )
    args = p.parse_args()

    schema = load_schema(args.traits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Demo A: {args.replicates} replicates x {args.generations} generations x "
          f"N={args.size} -> {len(CONDITIONS)} conditions")
    print(f"  traits:  {args.traits}")
    print(f"  output:  {out_dir}")

    for cond_name, regime, lam in CONDITIONS:
        out_path = out_dir / f"{cond_name}.jsonl"
        print(f"  -> {cond_name}  (regime={regime}, lambda={lam})")
        with TelemetryWriter(out_path) as writer:
            for r in range(args.replicates):
                seed = args.base_seed + 1000 * r + int(lam * 10)
                run_id = f"{cond_name}__seed{seed}"
                run_replicate(
                    schema=schema,
                    regime_name=regime,
                    lam=lam,
                    seed=seed,
                    generations=args.generations,
                    size=args.size,
                    writer=writer,
                    run_id=run_id,
                )
    print("done.")


if __name__ == "__main__":
    main()

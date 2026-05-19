"""Demo A -- failure-mode (operating envelope) sweep.

Maps the (mu, rho) plane to compliance C_m for the must-have audit gene.
Reinjection happens BEFORE mutation in the breeding cycle (Colony._step:
breed -> mutate -> enforce_must_have actually runs enforce LAST, but
mutation operates on the canonical allele if it was just restored).
At high enough mu, the post-reinjection mutation step re-randomizes
some fraction of canonical alleles back to variants. This sweep maps
where that failure mode bites: which (mu, rho) cells produce
compliance < some threshold (e.g. 0.95).

For each (mu, rho) cell: N_SEEDS replicates of Demo A's standard
pipeline (N=200, GENERATIONS gens). Records final-generation
compliance per (mu, rho, seed) plus the full trajectory.

Output JSONL row schema:
    {mu, rho, seed, generation, c_m, canonical_freq}

Invocation:
    python -m schmidt_demos.demo_a_domain_general.run_failure_mode_sweep
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from schmidt_demos.common.colony import Colony, ColonyConfig
from schmidt_demos.common.gene_schema import load_schema

from schmidt_demos.demo_a_domain_general.run import (
    DEFAULT_INITIAL_FREQS, REGIMES,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# Grid of (mu, rho) cells. Chosen to span:
#   mu: 0.0 (no mutation) to 0.50 (extreme mutation - every other gen)
#   rho: 0.0 (no enforcement) to 1.0 (full enforcement)
DEFAULT_MU_GRID  = [0.00, 0.05, 0.10, 0.20, 0.35, 0.50]
DEFAULT_RHO_GRID = [0.00, 0.25, 0.50, 0.75, 1.00]


def run_cell(
    *,
    mu: float,
    rho: float,
    seed: int,
    n_pop: int,
    generations: int,
    schema_template,
    must_have_locus: str,
    regime: str,
) -> list[dict]:
    """Run one replicate at one (mu, rho) cell. Return per-generation
    canonical-frequency rows for the must-have locus."""
    schema_local = copy.deepcopy(schema_template)
    schema_local.must_have = replace(schema_local.must_have, rho=rho)
    # Override the must-have locus's mutation rate; leave the others
    # at their schema defaults (those genes aren't the test subject).
    schema_local.genes[must_have_locus] = replace(
        schema_local.genes[must_have_locus],
        mutation_rate=mu,
    )

    cfg = ColonyConfig(
        size=n_pop,
        generations=generations,
        seed=seed,
        selection_intensity=1.0,
        initial_allele_freqs=DEFAULT_INITIAL_FREQS,
    )
    colony = Colony(schema_local, cfg)
    fitness_fn = REGIMES[regime]

    rows: list[dict] = []
    # Generation 0
    canon = colony.canonical_frequencies()[must_have_locus]
    rows.append({
        "mu": mu, "rho": rho, "seed": seed, "generation": 0,
        "canonical_freq": float(canon),
    })

    for g in range(generations):
        colony.step(fitness_fn)
        canon = colony.canonical_frequencies()[must_have_locus]
        rows.append({
            "mu": mu, "rho": rho, "seed": seed,
            "generation": int(colony.generation),
            "canonical_freq": float(canon),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="Demo A failure-mode sweep driver")
    p.add_argument("--n-pop", type=int, default=200)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--seeds", type=int, default=5,
                   help="Replicate seeds per (mu, rho) cell")
    p.add_argument("--regime", type=str, default="task_only",
                   choices=list(REGIMES.keys()),
                   help="Fitness regime; task_only is the silent-erosion "
                        "condition under which the failure mode is most visible.")
    p.add_argument("--must-have-locus", type=str, default="must_have_audit")
    p.add_argument("--base-seed", type=int, default=20260512)
    p.add_argument("--mu-grid", type=float, nargs="+", default=DEFAULT_MU_GRID)
    p.add_argument("--rho-grid", type=float, nargs="+", default=DEFAULT_RHO_GRID)
    p.add_argument(
        "--traits", type=str,
        default=str(Path(__file__).parent / "traits.yaml"),
    )
    p.add_argument(
        "--out", type=str,
        default=str(REPO_ROOT / "telemetry" / "demo_a_failure_mode" / "sweep.jsonl"),
    )
    args = p.parse_args()

    schema = load_schema(args.traits)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_cells = len(args.mu_grid) * len(args.rho_grid)
    total_replicates = n_cells * args.seeds
    print(
        f"Demo A failure-mode sweep: "
        f"|mu|={len(args.mu_grid)} x |rho|={len(args.rho_grid)} = {n_cells} cells, "
        f"{args.seeds} seeds/cell = {total_replicates} replicates "
        f"(N={args.n_pop}, G={args.generations}, regime={args.regime})"
    )
    print(f"  must-have locus: {args.must_have_locus}")
    print(f"  output:          {out_path}")

    t_start = time.time()
    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for mu in args.mu_grid:
            for rho in args.rho_grid:
                for r in range(args.seeds):
                    seed = args.base_seed + 1000 * r + int(mu * 100) + int(rho * 10)
                    rows = run_cell(
                        mu=mu, rho=rho, seed=seed,
                        n_pop=args.n_pop, generations=args.generations,
                        schema_template=schema,
                        must_have_locus=args.must_have_locus,
                        regime=args.regime,
                    )
                    for row in rows:
                        fh.write(json.dumps(row) + "\n")
                        written += 1
                    fh.flush()
                print(
                    f"  mu={mu:.3f} rho={rho:.3f} done "
                    f"({args.seeds} seeds; cumulative {time.time() - t_start:.1f}s)",
                    flush=True,
                )

    print(f"done. {written} rows written in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()

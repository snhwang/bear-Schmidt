"""Demo C experiment driver — two-strategy co-evolution on MiniCAGE.

Reuses Demo B's defender (gene→OpenC2 action mapping), Demo B's traits
YAML (six-locus genome with three must-have loci), and Demo B's
MiniCAGE bridge. Adds: bimodal G_D / G_A1 seeding, behavioural type
classification per generation, invasion-fitness readout.

Three pre-registered conditions:
    main_lambda_0     : φ_0 = 0.5, λ = 0   (no enforcement; expect G_A1 to spread)
    main_lambda_1     : φ_0 = 0.5, λ = 1   (full enforcement; expect G_D to dominate)
    invasion_lambda_0 : φ_0 = 0.05, λ = 0  (rare-mutant invasion; V_inv readout)

Invocation:
    .venv/Scripts/python.exe -m schmidt_demos.demo_c_coevolution.run
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BEAR_DEV = REPO_ROOT.parent / "bear-dev"
if BEAR_DEV.exists() and str(BEAR_DEV) not in sys.path:
    sys.path.insert(0, str(BEAR_DEV))

from schmidt_demos.common.colony import Colony, ColonyConfig
from schmidt_demos.common.gene_schema import Genome, GenomeSchema, load_schema
from schmidt_demos.demo_b_minicage_bridge.minicage_env import run_episode
from schmidt_demos.demo_b_minicage_bridge.defender import BEARDefender
from schmidt_demos.demo_c_coevolution.seeding import seed_coevolution_colony
from schmidt_demos.demo_c_coevolution.classification import (
    AgentType, classify, colony_fractions, aggregate_GA1_allele_freq,
)


# ---------------------------------------------------------------------------
# Telemetry record


@dataclass
class DemoCRecord:
    run_id: str
    seed: int
    condition: str            # 'main_lambda_0' | 'main_lambda_1' | 'invasion_lambda_0'
    rho: float
    phi_0: float
    generation: int
    f_pure_GD: float
    f_pure_GA1: float
    f_mixed: float
    mean_blue_reward: float
    # G_A1 allele frequency averaged across must-have loci.
    # Population-genetics observable for invasion-fitness analysis;
    # robust to 1/N floor that f_pure_GA1 (individual-fraction) hits.
    f_allele_GA1: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pre-registered conditions

CONDITIONS = [
    # (name, phi_0, lambda)
    ("main_lambda_0",     0.50, 0.0),
    ("main_lambda_1",     0.50, 1.0),
    ("invasion_lambda_0", 0.05, 0.0),
]

# φ_0 sensitivity sweep — gated by --sensitivity-sweep so the default
# behaviour is unchanged for the other runs. All three are invasion
# conditions and use --invasion-size at runtime.
SENSITIVITY_CONDITIONS = [
    ("invasion_phi_002", 0.02, 0.0),
    ("invasion_phi_005", 0.05, 0.0),
    ("invasion_phi_010", 0.10, 0.0),
]


# ---------------------------------------------------------------------------
# Per-generation scoring (reuses Demo B's MiniCAGE bridge)


def score_episode(
    genome: Genome, schema: GenomeSchema,
    *, episode_ticks: int, episode_seed: int,
) -> float:
    """Return blue_reward only (Demo C's selection is task-only;
    co-evolution dynamics are driven by the must-have layer + drift,
    not by an explicit safety co-selection term)."""
    defender = BEARDefender(genome=genome, schema=schema)
    result = run_episode(defender, ticks=episode_ticks, seed=episode_seed)
    return result.blue_reward


# ---------------------------------------------------------------------------
# One replicate

def run_replicate(
    schema: GenomeSchema,
    *,
    condition: str,
    phi_0: float,
    lam: float,
    seed: int,
    size: int,
    generations: int,
    episode_ticks: int,
    reward_scale: float,
    writer_fh,
    run_id: str,
) -> None:
    schema_local = copy.deepcopy(schema)
    schema_local.must_have = replace(schema_local.must_have, rho=lam)

    cfg = ColonyConfig(
        size=size,
        generations=generations,
        seed=seed,
        selection_intensity=1.0,
        # initial_allele_freqs unused — we replace agents with bimodal founders below
    )
    colony = Colony(schema_local, cfg)

    # Replace the per-locus-uniform founders with bimodal G_D / G_A1 seeding
    rng_seed = np.random.default_rng(seed + 1)
    colony.agents = seed_coevolution_colony(
        schema_local, size=size, phi_0=phi_0, rng=rng_seed,
    )

    rng = np.random.default_rng(seed + 7919)

    for g in range(generations):
        # Score every agent (single deterministic episode seed per agent)
        ep_seeds = rng.integers(0, 2**31 - 1, size=len(colony.agents))
        rewards = np.array([
            score_episode(a, schema_local,
                          episode_ticks=episode_ticks, episode_seed=int(s))
            for a, s in zip(colony.agents, ep_seeds)
        ], dtype=np.float64)

        # Telemetry: behavioural type fractions + mean reward + allele freq
        fracs = colony_fractions(colony.agents, schema_local)
        f_allele_GA1 = aggregate_GA1_allele_freq(colony.agents, schema_local)
        rec = DemoCRecord(
            run_id=run_id,
            seed=seed,
            condition=condition,
            rho=lam,
            phi_0=phi_0,
            generation=g,
            f_pure_GD=fracs[AgentType.PURE_GD],
            f_pure_GA1=fracs[AgentType.PURE_GA1],
            f_mixed=fracs[AgentType.MIXED],
            mean_blue_reward=float(rewards.mean()),
            f_allele_GA1=f_allele_GA1,
        )
        writer_fh.write(json.dumps(rec.to_dict()) + "\n")
        writer_fh.flush()

        # Selection weights (task-only; dynamics on M come from drift + λ)
        fits = rewards / reward_scale
        weights = np.exp(cfg.selection_intensity * fits)
        if not np.isfinite(weights.sum()) or weights.sum() <= 0:
            weights = np.ones_like(weights)
        probs = weights / weights.sum()

        # Sample N parent pairs, breed, mutate, enforce must-haves
        N = len(colony.agents)
        idx_a = rng.choice(N, size=N, p=probs)
        idx_b = rng.choice(N, size=N, p=probs)
        new_agents = []
        for ai, bi in zip(idx_a, idx_b):
            child = colony._breed(colony.agents[ai], colony.agents[bi])
            child = colony._mutate(child)
            child = colony._enforce_must_have(child)
            child.gen_born = colony.generation + 1
            new_agents.append(child)
        colony.agents = new_agents
        colony.generation += 1


# ---------------------------------------------------------------------------
# Driver

def main() -> None:
    p = argparse.ArgumentParser(description="Demo C experiment driver")
    p.add_argument("--size", type=int, default=30)
    p.add_argument("--invasion-size", type=int, default=None,
                   help="Population size for invasion_* conditions only. "
                        "Defaults to --size if unset; recommended N=200 for "
                        "the v0 paper to make φ_0·N exact (see DEMO_C_VARIANCE_NOTE.md).")
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--episode-ticks", type=int, default=30)
    p.add_argument("--reward-scale", type=float, default=20.0)
    p.add_argument("--base-seed", type=int, default=20260510,
                   help="Base seed (May 10 2026 — Demo C window)")
    p.add_argument("--sensitivity-sweep", action="store_true",
                   help="Run the φ_0 sensitivity sweep (φ_0 ∈ {0.02, 0.05, 0.10}, "
                        "all at λ=0) instead of the default 3 conditions.")
    p.add_argument(
        "--traits",
        type=str,
        # Demo C reuses Demo B's gene palette
        default=str(Path(__file__).parent.parent / "demo_b_minicage_bridge" / "traits.yaml"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT.parent / "cyber" / "telemetry" / "demo_c"),
    )
    args = p.parse_args()

    schema = load_schema(args.traits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions = SENSITIVITY_CONDITIONS if args.sensitivity_sweep else CONDITIONS
    invasion_size = args.invasion_size if args.invasion_size is not None else args.size

    print(
        f"Demo C{' (phi_0 sweep)' if args.sensitivity_sweep else ''}: "
        f"{args.replicates} replicates x {args.generations} gens x "
        f"N={args.size} (invasion N={invasion_size}, episodes={args.episode_ticks} ticks) -> "
        f"{len(conditions)} conditions"
    )
    print(f"  traits:  {args.traits}")
    print(f"  output:  {out_dir}")

    for cond_name, phi_0, lam in conditions:
        out_path = out_dir / f"{cond_name}.jsonl"
        cond_size = invasion_size if cond_name.startswith("invasion_") else args.size
        print(f"  -> {cond_name}  (phi_0={phi_0}, lambda={lam}, N={cond_size})")
        with open(out_path, "w", encoding="utf-8") as writer_fh:
            for r in range(args.replicates):
                seed = args.base_seed + 1000 * r + int(lam * 10) + int(phi_0 * 100)
                run_id = f"{cond_name}__seed{seed}"
                print(f"     seed {seed} ...", flush=True)
                run_replicate(
                    schema=schema,
                    condition=cond_name, phi_0=phi_0, lam=lam,
                    seed=seed, size=cond_size,
                    generations=args.generations,
                    episode_ticks=args.episode_ticks,
                    reward_scale=args.reward_scale,
                    writer_fh=writer_fh, run_id=run_id,
                )
    print("done.")


if __name__ == "__main__":
    main()

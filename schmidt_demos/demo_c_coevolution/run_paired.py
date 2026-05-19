"""Demo C — paired-defender variant.

Same colony evolution as run.py, but each generation's fitness is
evaluated by random-pairing the colony into two-defender teams and
running each pair through a paired MiniCAGE episode (paired_episode.py).
The m_3 separation-of-duty predicate is tested per-pair on the joint
audit log and aggregated into a colony-level C_{m_3}(t).

This addresses the multi-agent gap in Demo C's solo-episode variant:
the within-episode coordination dynamics (defender-pair concurrence
on destructive actions) are now part of the simulation rather than
being declared N/A.

Three pre-registered conditions, same as run.py.

Invocation:
    .venv/Scripts/python.exe -m schmidt_demos.demo_c_coevolution.run_paired
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
from schmidt_demos.demo_b_minicage_bridge.defender import BEARDefender
from schmidt_demos.demo_c_coevolution.paired_episode import run_paired_episode
from schmidt_demos.demo_c_coevolution.multi_defender_compliance import (
    m3_separation_of_duty_paired,
)
from schmidt_demos.demo_c_coevolution.seeding import seed_coevolution_colony
from schmidt_demos.demo_c_coevolution.classification import (
    AgentType, colony_fractions, aggregate_GA1_allele_freq,
)


@dataclass
class DemoCPairedRecord:
    run_id: str
    seed: int
    condition: str
    rho: float
    phi_0: float
    generation: int
    f_pure_GD: float
    f_pure_GA1: float
    f_mixed: float
    mean_blue_reward: float
    # New: colony-wide C_m3 from paired-episode evaluation
    compliance_m3: float
    # G_A1 allele frequency averaged across must-have loci.
    f_allele_GA1: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CONDITIONS = [
    ("main_lambda_0",     0.50, 0.0),
    ("main_lambda_1",     0.50, 1.0),
    ("invasion_lambda_0", 0.05, 0.0),
]


def score_pair(
    a: Genome, b: Genome, schema: GenomeSchema,
    *, episode_ticks: int, episode_seed: int,
) -> tuple[float, bool]:
    """Returns (joint_blue_reward, m3_compliant)."""
    da = BEARDefender(genome=a, schema=schema)
    db = BEARDefender(genome=b, schema=schema)
    res = run_paired_episode(da, db, ticks=episode_ticks, seed=episode_seed)
    m3_ok = m3_separation_of_duty_paired(res.log_a, res.log_b)
    return res.blue_reward, m3_ok


def run_replicate(
    schema: GenomeSchema,
    *,
    condition: str, phi_0: float, lam: float, seed: int,
    size: int, generations: int, episode_ticks: int, reward_scale: float,
    writer_fh, run_id: str,
) -> None:
    schema_local = copy.deepcopy(schema)
    schema_local.must_have = replace(schema_local.must_have, rho=lam)

    # size must be even for clean pairing — round up if needed
    if size % 2 != 0:
        size += 1

    cfg = ColonyConfig(size=size, generations=generations, seed=seed,
                       selection_intensity=1.0)
    colony = Colony(schema_local, cfg)

    rng_seed = np.random.default_rng(seed + 1)
    colony.agents = seed_coevolution_colony(
        schema_local, size=size, phi_0=phi_0, rng=rng_seed,
    )

    rng = np.random.default_rng(seed + 7919)

    for g in range(generations):
        # Random pairing of agents into N/2 pairs
        order = rng.permutation(len(colony.agents))
        pairs = [(int(order[2 * i]), int(order[2 * i + 1]))
                 for i in range(len(order) // 2)]
        ep_seeds = rng.integers(0, 2**31 - 1, size=len(pairs))

        # Per-agent reward (each member of a pair gets the joint reward)
        rewards = np.zeros(len(colony.agents), dtype=np.float64)
        m3_results: list[bool] = []
        for (ai, bi), eps in zip(pairs, ep_seeds):
            joint_R, m3_ok = score_pair(
                colony.agents[ai], colony.agents[bi], schema_local,
                episode_ticks=episode_ticks, episode_seed=int(eps),
            )
            rewards[ai] = joint_R
            rewards[bi] = joint_R
            m3_results.append(m3_ok)

        # Telemetry
        fracs = colony_fractions(colony.agents, schema_local)
        c_m3 = float(np.mean(m3_results)) if m3_results else float("nan")
        f_allele_GA1 = aggregate_GA1_allele_freq(colony.agents, schema_local)
        rec = DemoCPairedRecord(
            run_id=run_id, seed=seed, condition=condition,
            rho=lam, phi_0=phi_0, generation=g,
            f_pure_GD=fracs[AgentType.PURE_GD],
            f_pure_GA1=fracs[AgentType.PURE_GA1],
            f_mixed=fracs[AgentType.MIXED],
            mean_blue_reward=float(rewards.mean()),
            compliance_m3=c_m3,
            f_allele_GA1=f_allele_GA1,
        )
        writer_fh.write(json.dumps(rec.to_dict()) + "\n")
        writer_fh.flush()

        # Selection
        fits = rewards / reward_scale
        weights = np.exp(cfg.selection_intensity * fits)
        if not np.isfinite(weights.sum()) or weights.sum() <= 0:
            weights = np.ones_like(weights)
        probs = weights / weights.sum()

        # Breed
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


def main() -> None:
    p = argparse.ArgumentParser(description="Demo C paired-defender driver")
    p.add_argument("--size", type=int, default=30)
    p.add_argument("--invasion-size", type=int, default=None,
                   help="Population size for invasion_* conditions only. "
                        "Defaults to --size if unset; recommended N=200 for "
                        "the v0 paper (see DEMO_C_VARIANCE_NOTE.md).")
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--episode-ticks", type=int, default=30)
    p.add_argument("--reward-scale", type=float, default=20.0)
    p.add_argument("--base-seed", type=int, default=20260512,
                   help="Base seed (May 12 2026 — extension to Demo C)")
    p.add_argument(
        "--traits",
        type=str,
        default=str(Path(__file__).parent.parent / "demo_b_minicage_bridge" / "traits.yaml"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT.parent / "cyber" / "telemetry" / "demo_c_paired"),
    )
    args = p.parse_args()

    schema = load_schema(args.traits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    invasion_size = args.invasion_size if args.invasion_size is not None else args.size

    print(
        f"Demo C (paired): {args.replicates} replicates x {args.generations} gens x "
        f"N={args.size} (invasion N={invasion_size}, paired episodes={args.episode_ticks} ticks) -> "
        f"{len(CONDITIONS)} conditions"
    )
    print(f"  traits:  {args.traits}")
    print(f"  output:  {out_dir}")

    for cond_name, phi_0, lam in CONDITIONS:
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

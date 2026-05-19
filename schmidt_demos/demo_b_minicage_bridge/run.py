"""Demo B experiment driver.

N=30 defender colony, 15 generations, two regimes (task-only vs
task+safety co-selection), three replicate seeds per condition, λ=0.5
must-have reinjection in both regimes (the must-have pipeline is on in
Demo B; Demo A already swept λ).

Per agent per generation:
  1. Instantiate BEARDefender from the agent's Genome.
  2. Run a 30-tick MiniCAGE episode against Meander red.
  3. Collect (blue_reward, red_reward, audit_log).
  4. Evaluate the 5 compliance predicates on the audit log.

Fitness function:
  - task_only:         fitness = normalize(blue_reward)
  - task_plus_safety:  fitness = normalize(blue_reward) + γ · mean(C_m across m)

Telemetry emitted as JSONL to ./telemetry/demo_b/.

Invocation:
    .venv/Scripts/python.exe -m schmidt_demos.demo_b_minicage_bridge.run
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# BEAR (not used in Demo B's critical path, but available for future LLM blending)
BEAR_DEV = REPO_ROOT.parent / "bear-dev"
if BEAR_DEV.exists() and str(BEAR_DEV) not in sys.path:
    sys.path.insert(0, str(BEAR_DEV))

from schmidt_demos.common.colony import Colony, ColonyConfig
from schmidt_demos.common.gene_schema import Genome, GenomeSchema, load_schema
from schmidt_demos.common.telemetry import TelemetryWriter
from schmidt_demos.demo_b_minicage_bridge.defender import BEARDefender
from schmidt_demos.demo_b_minicage_bridge.minicage_env import run_episode
from schmidt_demos.demo_b_minicage_bridge.compliance import (
    PREDICATES, evaluate_all,
)


# ---------------------------------------------------------------------------
# Demo B record (richer than common.telemetry.GenerationRecord since we
# also capture episode reward mean/std and per-member C_m)


@dataclass
class DemoBRecord:
    run_id: str
    seed: int
    regime: str               # 'task_only' | 'task_plus_safety'
    rho: float
    generation: int
    # Level-1 observables
    canonical_freq: dict[str, float]
    compliance: dict[str, float | None]     # m_id -> C_m (or None for N/A)
    # CAGE-2 task signals
    mean_blue_reward: float
    std_blue_reward: float
    mean_red_reward: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pre-registered conditions

CONDITIONS = [
    # (condition name, regime, λ).
    # The two main regimes are run at λ=0.5 (substrate enforcement on);
    # task_only at λ=0 is included as a baseline to show the must-have
    # pipeline's contribution (curves diverge from the λ=0.5 traces).
    ("task_only__lambda_0",          "task_only",         0.0),
    ("task_only__lambda_0_5",        "task_only",         0.5),
    ("task_plus_safety__lambda_0_5", "task_plus_safety",  0.5),
]

# Initial allele frequencies: start each safety gene at canonical=0.5 so
# C_m(t) trajectories are visible. Capability genes start uniform across
# alleles so task-reward shaping is honest.
DEFAULT_INITIAL_FREQS: dict[str, float] = {
    "threat_threshold":    0.333,    # 1/3 uniform across 3 alleles
    "primary_response":    0.333,
    "decoy_policy":        0.333,
    "audit_discipline":    0.50,
    "escalation_policy":   0.50,
    "suppression_policy":  0.50,
}


# ---------------------------------------------------------------------------
# Per-generation scoring

def score_agent(
    genome: Genome,
    schema: GenomeSchema,
    *,
    episode_ticks: int,
    episode_seed: int,
) -> dict[str, Any]:
    """Run one MiniCAGE episode and evaluate all compliance predicates.

    Returns a dict with keys:
        blue_reward, red_reward, compliance (id -> bool | None),
        applicable_mean_compliance (float in [0,1]).
    """
    defender = BEARDefender(genome=genome, schema=schema)
    result = run_episode(defender, ticks=episode_ticks, seed=episode_seed)
    comp = evaluate_all(result.audit_log)
    applicable = [v for v in comp.values() if isinstance(v, bool)]
    mean_c = float(np.mean([1.0 if v else 0.0 for v in applicable])) if applicable else 1.0
    return {
        "blue_reward": result.blue_reward,
        "red_reward":  result.red_reward,
        "compliance":  comp,
        "applicable_mean_compliance": mean_c,
    }


def _fitness_from_scores(
    score: dict[str, Any], *, regime: str, gamma: float,
    reward_scale: float,
) -> float:
    """Compose the scalar fitness for selection.

    For stability we rescale rewards into a ~0-mean unit-scale range;
    γ is the co-selection weight on the mean of applicable C_m.
    """
    r = score["blue_reward"] / reward_scale
    if regime == "task_only":
        return r
    elif regime == "task_plus_safety":
        c = score["applicable_mean_compliance"]
        return r + gamma * c
    raise ValueError(f"unknown regime {regime!r}")


# ---------------------------------------------------------------------------
# One replicate (custom generation loop)


def run_replicate(
    schema: GenomeSchema,
    *,
    regime: str,
    lam: float,
    seed: int,
    size: int,
    generations: int,
    episode_ticks: int,
    gamma: float,
    reward_scale: float,
    writer_fh,
    run_id: str,
) -> None:
    # Clone the schema so we can override λ per condition
    schema_local = copy.deepcopy(schema)
    from dataclasses import replace
    schema_local.must_have = replace(schema_local.must_have, rho=lam)

    cfg = ColonyConfig(
        size=size,
        generations=generations,
        seed=seed,
        selection_intensity=1.0,
        initial_allele_freqs=DEFAULT_INITIAL_FREQS,
    )
    colony = Colony(schema_local, cfg)

    rng = np.random.default_rng(seed + 7919)

    for g in range(generations):
        # Score every agent (single deterministic episode seed per agent for replicability)
        ep_seeds = rng.integers(0, 2**31 - 1, size=len(colony.agents))
        scores = []
        for agent, ep_seed in zip(colony.agents, ep_seeds):
            s = score_agent(
                agent, schema_local,
                episode_ticks=episode_ticks, episode_seed=int(ep_seed),
            )
            scores.append(s)

        # Aggregates for telemetry
        blue_rewards = np.array([s["blue_reward"] for s in scores])
        red_rewards = np.array([s["red_reward"]  for s in scores])

        # Per-predicate C_m for this generation
        compliance_agg: dict[str, float | None] = {}
        for p in PREDICATES:
            vals = [s["compliance"][p.id] for s in scores]
            if not p.applicable:
                compliance_agg[p.id] = None
            else:
                bool_vals = [v for v in vals if isinstance(v, bool)]
                compliance_agg[p.id] = (
                    float(sum(1 for v in bool_vals if v)) / len(bool_vals)
                    if bool_vals else 0.0
                )

        rec = DemoBRecord(
            run_id=run_id,
            seed=seed,
            regime=regime,
            rho=lam,
            generation=g,
            canonical_freq=colony.canonical_frequencies(),
            compliance=compliance_agg,
            mean_blue_reward=float(blue_rewards.mean()),
            std_blue_reward=float(blue_rewards.std()),
            mean_red_reward=float(red_rewards.mean()),
        )
        writer_fh.write(json.dumps(rec.to_dict()) + "\n")
        writer_fh.flush()

        # Compute selection weights (fitness-proportional)
        fits = np.array([
            _fitness_from_scores(s, regime=regime, gamma=gamma,
                                 reward_scale=reward_scale)
            for s in scores
        ])
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
    p = argparse.ArgumentParser(description="Demo B experiment driver")
    p.add_argument("--size", type=int, default=30)
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--replicates", type=int, default=3)
    p.add_argument("--episode-ticks", type=int, default=30)
    p.add_argument("--gamma", type=float, default=0.5,
                   help="Co-selection weight on mean applicable compliance")
    p.add_argument("--reward-scale", type=float, default=20.0,
                   help="Normalization for blue_reward magnitude")
    p.add_argument("--base-seed", type=int, default=20260501,
                   help="Base seed (May 1 2026 — Demo B window)")
    p.add_argument(
        "--traits",
        type=str,
        default=str(Path(__file__).parent / "traits.yaml"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT.parent / "cyber" / "telemetry" / "demo_b"),
    )
    args = p.parse_args()

    schema = load_schema(args.traits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Demo B: {args.replicates} replicates x {args.generations} gens x "
        f"N={args.size} (episodes={args.episode_ticks} ticks) -> "
        f"{len(CONDITIONS)} conditions"
    )
    print(f"  traits:  {args.traits}")
    print(f"  output:  {out_dir}")

    for cond_name, regime, lam in CONDITIONS:
        out_path = out_dir / f"{cond_name}.jsonl"
        print(f"  -> {cond_name}  (regime={regime}, lambda={lam})")
        with open(out_path, "w", encoding="utf-8") as writer_fh:
            for r in range(args.replicates):
                seed = args.base_seed + 1000 * r + int(lam * 10)
                run_id = f"{cond_name}__seed{seed}"
                print(f"     seed {seed} ...", flush=True)
                run_replicate(
                    schema=schema,
                    regime=regime, lam=lam,
                    seed=seed, size=args.size,
                    generations=args.generations,
                    episode_ticks=args.episode_ticks,
                    gamma=args.gamma,
                    reward_scale=args.reward_scale,
                    writer_fh=writer_fh, run_id=run_id,
                )
    print("done.")


if __name__ == "__main__":
    main()

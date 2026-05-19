"""Demo E experiment driver -- BEAR on CAGE Challenge 4 (production).

5 within-zone populations (one per CC4 blue agent: blue_agent_0..4),
N_pop candidates each. Each generation, every candidate is scored on
one CC4 episode where it plays its zone and the other 4 zones are
filled with random picks from their respective populations. Per-zone
selection is fitness-proportional with the audit-log-derived fitness
function described in DEMO_E_PLAN.md section 6:

    task_fitness   = -mean_over_ticks(zone_compromise_count)
    safety_fitness =  fraction of compliance predicates passed
    fitness        =  task_fitness + gamma * safety_fitness

Four pre-registered conditions:

    task_only__rho_0           -- no reinjection; expect silent erosion
    task_only__rho_0_5         -- partial reinjection
    task_plus_safety__rho_0_5  -- partial reinjection + gamma co-selection
    task_only__rho_1           -- full reinjection; canonical alleles pinned

Telemetry emitted as JSONL to ./telemetry/demo_e_cage4_prod/. Existing
seeds are detected and skipped on restart, so the script is safe to
ctrl-C and re-launch.

Invocation:
    cd /mnt/c/Users/Scott/Documents/Work/cyber
    source .venv-cc4/bin/activate
    python -m schmidt_demos.demo_e_cage4.run
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from schmidt_demos.common.colony import Colony, ColonyConfig
from schmidt_demos.common.gene_schema import Genome, GenomeSchema, load_schema
from schmidt_demos.demo_e_cage4.cc4_env import (
    BLUE_AGENT_NAMES, get_action_labels, run_episode,
)
from schmidt_demos.demo_e_cage4.compliance import (
    PREDICATES, evaluate_all, safety_fitness,
)
from schmidt_demos.demo_e_cage4.defender import AuditEntry, BEARDefender


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Pre-registered conditions

CONDITIONS = [
    # (name, rho, gamma) -- four pre-registered cells (full design)
    ("task_only__rho_0",          0.0, 0.0),
    ("task_only__rho_0_5",        0.5, 0.0),
    ("task_plus_safety__rho_0_5", 0.5, 0.5),
    ("task_only__rho_1",          1.0, 0.0),
]


# ---------------------------------------------------------------------------
# Telemetry record


@dataclass
class DemoERecord:
    run_id: str
    seed: int
    condition: str
    rho: float
    generation: int
    zone: str                             # blue_agent_0..4
    # Per-zone aggregates over the N_pop candidates this generation
    mean_task_fitness: float
    mean_safety_fitness: float
    mean_zone_compromise_count: float     # average across ticks across candidates
    compliance: dict[str, float]          # m_id -> fraction of zone agents compliant
    canonical_freq: dict[str, float]      # locus -> fraction of zone agents canonical
    episode_ticks_completed: float        # mean ticks (early termination matters here)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Audit-log derived fitness


def task_fitness_from_log(log: list[AuditEntry]) -> float:
    """-mean_over_ticks(zone_compromise_count). Higher (less negative) = better."""
    if not log:
        return 0.0
    return -sum(e.zone_compromise_count for e in log) / len(log)


def fitness_for_selection(
    log: list[AuditEntry], *, gamma: float,
) -> float:
    """Combine task and safety. Under task_only conditions gamma=0."""
    t = task_fitness_from_log(log)
    if gamma <= 0.0:
        return t
    s = safety_fitness(log)
    return t + gamma * s


# ---------------------------------------------------------------------------
# Scoring one candidate by running one CC4 episode


def score_one_episode(
    *,
    zone: str,
    candidate_genome: Genome,
    other_genomes: dict[str, Genome],
    schema: GenomeSchema,
    action_labels: dict[str, list[str]],
    episode_seed: int,
    steps: int,
) -> dict[str, AuditEntry] | None:
    """Run one CC4 episode with `candidate_genome` slotted into `zone`
    and the other 4 zones filled from `other_genomes`. Return the
    full per-zone audit logs (so the caller can also see what the
    opponents did, which is useful for debugging though we only score
    the candidate's own zone).

    Returns None on episode error (CC4 raises occasionally on
    pathological action choices in pilot configs).
    """
    defenders: dict[str, BEARDefender] = {}
    for name in BLUE_AGENT_NAMES:
        if name == zone:
            g = candidate_genome
        else:
            g = other_genomes[name]
        defenders[name] = BEARDefender(
            genome=g,
            schema=schema,
            action_labels=action_labels[name],
            agent_name=name,
        )

    try:
        result = run_episode(defenders, steps=steps, seed=episode_seed)
    except Exception as e:
        print(f"  WARN: episode error at zone={zone}: {type(e).__name__}: {e}", flush=True)
        return None
    return {
        "audit_logs": {name: d.audit_log for name, d in defenders.items()},
        "ticks": result.episode_ticks,
    }


# ---------------------------------------------------------------------------
# One generation: score all candidates, then within-zone selection + breed


def run_generation(
    *,
    generation: int,
    zone_colonies: dict[str, Colony],
    schema_local: GenomeSchema,
    action_labels: dict[str, list[str]],
    condition_name: str,
    rho: float,
    gamma: float,
    steps: int,
    rng: np.random.Generator,
    writer_fh,
    seed: int,
    run_id: str,
) -> None:
    """Run one generation of the 5-zone co-evolution.

    Sequence:
      1. For each zone, for each candidate in that zone's colony, run
         one CC4 episode with the candidate paired against random
         picks from the other 4 zones' colonies. Record the candidate's
         audit log.
      2. Compute per-candidate task_fitness and safety_fitness from
         the audit log.
      3. Emit per-zone telemetry for generation `generation`.
      4. Within each zone, do fitness-proportional parent sampling +
         locus-tagged breeding + mutation + must-have reinjection.
    """
    # ---- Step 1: score every candidate in every zone ----
    per_zone_scores: dict[str, list[float]] = {z: [] for z in BLUE_AGENT_NAMES}
    per_zone_task: dict[str, list[float]] = {z: [] for z in BLUE_AGENT_NAMES}
    per_zone_safety: dict[str, list[float]] = {z: [] for z in BLUE_AGENT_NAMES}
    per_zone_compromise: dict[str, list[float]] = {z: [] for z in BLUE_AGENT_NAMES}
    per_zone_compliance_bits: dict[str, dict[str, list[bool]]] = {
        z: {p.id: [] for p in PREDICATES} for z in BLUE_AGENT_NAMES
    }
    per_zone_ticks: dict[str, list[int]] = {z: [] for z in BLUE_AGENT_NAMES}

    for zone in BLUE_AGENT_NAMES:
        candidates = zone_colonies[zone].agents
        for ci, cand in enumerate(candidates):
            # Pair the candidate against one random pick from each
            # other zone's current colony.
            other_genomes: dict[str, Genome] = {}
            for z in BLUE_AGENT_NAMES:
                if z == zone:
                    continue
                idx = int(rng.integers(0, len(zone_colonies[z].agents)))
                other_genomes[z] = zone_colonies[z].agents[idx]

            ep_seed = int(rng.integers(0, 2**31 - 1))
            outcome = score_one_episode(
                zone=zone, candidate_genome=cand,
                other_genomes=other_genomes,
                schema=schema_local,
                action_labels=action_labels,
                episode_seed=ep_seed, steps=steps,
            )
            if outcome is None:
                # Episode error -- treat as neutral fitness
                per_zone_scores[zone].append(0.0)
                per_zone_task[zone].append(0.0)
                per_zone_safety[zone].append(0.0)
                per_zone_compromise[zone].append(0.0)
                for p in PREDICATES:
                    per_zone_compliance_bits[zone][p.id].append(False)
                per_zone_ticks[zone].append(0)
                continue

            log = outcome["audit_logs"][zone]
            t_fit = task_fitness_from_log(log)
            s_fit = safety_fitness(log)
            total = t_fit + gamma * s_fit if gamma > 0 else t_fit
            comp = evaluate_all(log)

            per_zone_scores[zone].append(total)
            per_zone_task[zone].append(t_fit)
            per_zone_safety[zone].append(s_fit)
            per_zone_compromise[zone].append(-t_fit)   # = mean compromise
            for pid, v in comp.items():
                per_zone_compliance_bits[zone][pid].append(v)
            per_zone_ticks[zone].append(outcome["ticks"])

    # ---- Step 2: emit per-zone telemetry ----
    for zone in BLUE_AGENT_NAMES:
        n = len(per_zone_scores[zone])
        canon_freq = zone_colonies[zone].canonical_frequencies()
        compliance_agg = {
            pid: float(sum(bits) / len(bits)) if bits else 0.0
            for pid, bits in per_zone_compliance_bits[zone].items()
        }
        rec = DemoERecord(
            run_id=run_id,
            seed=seed,
            condition=condition_name,
            rho=rho,
            generation=generation,
            zone=zone,
            mean_task_fitness=float(np.mean(per_zone_task[zone])) if n else 0.0,
            mean_safety_fitness=float(np.mean(per_zone_safety[zone])) if n else 0.0,
            mean_zone_compromise_count=float(np.mean(per_zone_compromise[zone])) if n else 0.0,
            compliance=compliance_agg,
            canonical_freq=canon_freq,
            episode_ticks_completed=float(np.mean(per_zone_ticks[zone])) if n else 0.0,
        )
        writer_fh.write(json.dumps(rec.to_dict()) + "\n")
    writer_fh.flush()

    # ---- Step 3: within-zone selection + breed ----
    for zone in BLUE_AGENT_NAMES:
        colony = zone_colonies[zone]
        fits = np.array(per_zone_scores[zone], dtype=np.float64)
        weights = np.exp(colony.config.selection_intensity * fits)
        if not np.isfinite(weights.sum()) or weights.sum() <= 0:
            weights = np.ones_like(weights)
        probs = weights / weights.sum()
        N = len(colony.agents)
        idx_a = rng.choice(N, size=N, p=probs)
        idx_b = rng.choice(N, size=N, p=probs)
        new_agents: list[Genome] = []
        for ai, bi in zip(idx_a, idx_b):
            child = colony._breed(colony.agents[int(ai)], colony.agents[int(bi)])
            child = colony._mutate(child)
            child = colony._enforce_must_have(child)
            child.gen_born = colony.generation + 1
            new_agents.append(child)
        colony.agents = new_agents
        colony.generation += 1


# ---------------------------------------------------------------------------
# One replicate seed: build colonies + run all generations


def run_replicate(
    *,
    schema: GenomeSchema,
    condition_name: str,
    rho: float,
    gamma: float,
    seed: int,
    n_pop: int,
    generations: int,
    steps: int,
    action_labels: dict[str, list[str]],
    writer_fh,
    run_id: str,
) -> None:
    schema_local = copy.deepcopy(schema)
    schema_local.must_have = replace(schema_local.must_have, rho=rho)

    # 5 separate per-zone colonies, all using the same schema/rho but
    # independent random seeds so the within-zone allele frequencies
    # diverge naturally.
    zone_colonies: dict[str, Colony] = {}
    for i, zone in enumerate(BLUE_AGENT_NAMES):
        cfg = ColonyConfig(
            size=n_pop,
            generations=generations,
            seed=seed * 1000 + i,
            selection_intensity=1.0,
            # initial_allele_freqs: balanced (1/n per allele) per locus
            # gives the substrate the most room to differentiate zones.
        )
        zone_colonies[zone] = Colony(schema_local, cfg)

    rng = np.random.default_rng(seed + 7919)

    t_start = time.time()
    for g in range(generations):
        gen_t0 = time.time()
        run_generation(
            generation=g,
            zone_colonies=zone_colonies,
            schema_local=schema_local,
            action_labels=action_labels,
            condition_name=condition_name,
            rho=rho, gamma=gamma,
            steps=steps,
            rng=rng,
            writer_fh=writer_fh,
            seed=seed,
            run_id=run_id,
        )
        elapsed = time.time() - gen_t0
        print(
            f"     gen {g+1}/{generations} ({elapsed:.1f}s, "
            f"cumulative {time.time() - t_start:.1f}s)",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Driver


def _load_complete_seeds(
    out_path: Path, generations: int, n_zones: int = 5,
) -> tuple[set[int], list[str]]:
    """Return (set of fully-complete seeds, lines belonging to them).

    A seed is complete when it has exactly generations*n_zones rows
    (one record per generation per zone). Partial seeds get dropped
    so re-running the seed produces a clean record set.
    """
    if not out_path.exists():
        return set(), []
    by_seed: dict[int, list[str]] = defaultdict(list)
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_seed[int(r["seed"])].append(line)
    complete: set[int] = set()
    keep: list[str] = []
    expected = generations * n_zones
    for seed, lines in by_seed.items():
        if len(lines) >= expected:
            complete.add(seed)
            keep.extend(lines)
    return complete, keep


def main() -> None:
    p = argparse.ArgumentParser(description="Demo E production driver")
    p.add_argument("--n-pop", type=int, default=20,
                   help="Per-zone population size (default 20)")
    p.add_argument("--generations", type=int, default=12,
                   help="Generations per replicate (default 12)")
    p.add_argument("--replicates", type=int, default=20,
                   help="Replicate seeds per condition (default 20)")
    p.add_argument("--steps", type=int, default=50,
                   help="CC4 episode ticks")
    p.add_argument("--gamma", type=float, default=0.0,
                   help="Co-selection weight on safety (task_only uses 0)")
    p.add_argument("--base-seed", type=int, default=20260512,
                   help="Base seed (May 12, 2026 -- Demo E window)")
    p.add_argument(
        "--traits",
        type=str,
        default=str(Path(__file__).parent / "traits.yaml"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(REPO_ROOT / "telemetry" / "demo_e_cage4_prod"),
    )
    args = p.parse_args()

    schema = load_schema(args.traits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching CC4 action labels ...", flush=True)
    action_labels = get_action_labels(seed=0)
    print(f"  per-agent sizes: {[(n, len(action_labels[n])) for n in BLUE_AGENT_NAMES]}")

    print(
        f"Demo E: {args.replicates} replicates x {args.generations} gens x "
        f"N_pop={args.n_pop} x 5 zones x {args.steps}-tick episodes "
        f"-> {len(CONDITIONS)} conditions"
    )
    print(f"  traits:  {args.traits}")
    print(f"  output:  {out_dir}")

    for cond_name, rho, gamma in CONDITIONS:
        out_path = out_dir / f"{cond_name}.jsonl"
        print(f"  -> {cond_name}  (rho={rho}, gamma={gamma})", flush=True)

        complete_seeds, keep_lines = _load_complete_seeds(
            out_path, args.generations, n_zones=len(BLUE_AGENT_NAMES),
        )
        if complete_seeds:
            print(f"     resuming: {len(complete_seeds)} seed(s) already "
                  f"complete; skipping them")
            with open(out_path, "w", encoding="utf-8") as fh:
                for line in keep_lines:
                    fh.write(line + "\n")
        elif out_path.exists():
            out_path.write_text("", encoding="utf-8")

        with open(out_path, "a", encoding="utf-8") as writer_fh:
            for r in range(args.replicates):
                seed = args.base_seed + 1000 * r + int(rho * 10)
                if seed in complete_seeds:
                    continue
                run_id = f"{cond_name}__seed{seed}"
                print(f"     seed {seed} ...", flush=True)
                run_replicate(
                    schema=schema,
                    condition_name=cond_name,
                    rho=rho, gamma=gamma,
                    seed=seed,
                    n_pop=args.n_pop,
                    generations=args.generations,
                    steps=args.steps,
                    action_labels=action_labels,
                    writer_fh=writer_fh,
                    run_id=run_id,
                )
    print("done.")


if __name__ == "__main__":
    main()

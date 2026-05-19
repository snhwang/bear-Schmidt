"""Demo F experiment driver: LLM decision engine on MiniCAGE.

Two pre-registered conditions, 50 replicate seeds each, 15 generations,
N=20 defenders per colony, 30-tick episodes against Meander red. LLM
calls dispatched concurrently per tick via a thread pool so the
per-tick latency is roughly one LLM round-trip rather than N.

Conditions:
  task_only__rho_0   -- task-only selection, no reinjection. Expected
                        silent erosion of canonical audit_discipline.
  task_only__rho_1   -- task-only selection, full reinjection. Expected
                        canonical allele pinned at 1.0.

Output: JSONL telemetry under ./telemetry/demo_f_llm_decision/.

Smoke run for sanity check (N=5, 1 seed, 3 generations):
    python -m schmidt_demos.demo_f_llm_decision.run \\
        --size 5 --generations 3 --replicates 1

Production:
    python -m schmidt_demos.demo_f_llm_decision.run

Server prerequisite: vLLM serving "gemma-4-e2b" at localhost:8355.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BEAR_DEV = REPO_ROOT.parent / "bear-dev"
if BEAR_DEV.exists() and str(BEAR_DEV) not in sys.path:
    sys.path.insert(0, str(BEAR_DEV))

# Vendored MiniCAGE
CYBORG_PP = REPO_ROOT / "third_party" / "CybORG_plus_plus"
if str(CYBORG_PP) not in sys.path:
    sys.path.insert(0, str(CYBORG_PP))

from mini_CAGE import SimplifiedCAGE, Meander_minimal  # noqa: E402

from schmidt_demos.common.colony import Colony, ColonyConfig  # noqa: E402
from schmidt_demos.common.gene_schema import (  # noqa: E402
    Genome, GenomeSchema, load_schema,
)
from schmidt_demos.demo_b_minicage_bridge.compliance import (  # noqa: E402
    PREDICATES, evaluate_all,
)
from schmidt_demos.demo_f_llm_decision.defender import (  # noqa: E402
    LLMDefender, get_client, DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS, USE_GUIDED_JSON,
)


CONDITIONS = [
    ("task_only__rho_0", "task_only", 0.0),
    ("task_only__rho_1", "task_only", 1.0),
]


DEFAULT_INITIAL_FREQS: dict[str, float] = {
    "threat_threshold":    0.333,
    "primary_response":    0.333,
    "decoy_policy":        0.333,
    "audit_discipline":    0.50,
    "escalation_policy":   0.50,
    "suppression_policy":  0.50,
}


@dataclass
class DemoFRecord:
    run_id: str
    seed: int
    regime: str
    rho: float
    generation: int
    canonical_freq: dict[str, float]
    compliance: dict[str, float | None]
    mean_blue_reward: float
    std_blue_reward: float
    mean_red_reward: float
    parse_failure_rate: float
    mean_tick_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _llm_call(
    messages: list[dict], model: str, max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """One synchronous LLM call. Returns "" on failure (defender falls
    back to SLEEP on empty response)."""
    client = get_client()
    kwargs: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if USE_GUIDED_JSON:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


def run_generation_episodes(
    defenders: list[LLMDefender],
    *,
    episode_ticks: int,
    episode_seeds: np.ndarray,
    model: str,
    max_workers: int,
) -> tuple[list[float], list[float], list[float]]:
    """Run one MiniCAGE episode per defender, in lockstep across ticks
    so LLM calls can be dispatched concurrently.

    Returns (blue_rewards, red_rewards, tick_latencies_ms).
    """
    n = len(defenders)
    envs = [SimplifiedCAGE(num_envs=1) for _ in range(n)]
    states = []
    for env, s in zip(envs, episode_seeds):
        np.random.seed(int(s))
        state, _info = env.reset()
        states.append(state)
    reds = [Meander_minimal() for _ in range(n)]

    blue_sums = [0.0] * n
    red_sums = [0.0] * n
    tick_latencies_ms: list[float] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for t in range(episode_ticks):
            # Prepare obs and messages for each defender
            blue_obs = [
                np.asarray(states[i]["Blue"]).reshape(-1)[:78]
                for i in range(n)
            ]
            messages_list = [
                defenders[i].prepare_messages(blue_obs[i], t) for i in range(n)
            ]

            # Concurrent LLM calls
            t0 = time.time()
            response_texts = list(pool.map(
                lambda msgs: _llm_call(msgs, model), messages_list
            ))
            tick_latencies_ms.append((time.time() - t0) * 1000.0)

            # Consume responses, step envs
            for i in range(n):
                action_id = defenders[i].consume_response(
                    response_texts[i], blue_obs[i], t,
                )
                blue_a = np.array([action_id])
                red_a = np.asarray(
                    reds[i].get_action(observation=states[i]["Red"])
                ).reshape(1,)
                state, reward, _done, _info = envs[i].step(
                    blue_action=blue_a, red_action=red_a,
                )
                states[i] = state
                blue_sums[i] += float(np.asarray(reward["Blue"]).flatten()[0])
                red_sums[i] += float(np.asarray(reward["Red"]).flatten()[0])

    return blue_sums, red_sums, tick_latencies_ms


def score_population(
    agents: list[Genome],
    schema: GenomeSchema,
    *,
    episode_ticks: int,
    episode_seeds: np.ndarray,
    model: str,
    max_workers: int,
) -> tuple[list[dict[str, Any]], float, float, list[dict]]:
    """Score every agent in one generation. Returns (scores,
    parse_failure_rate, mean_tick_latency_ms, parse_failure_records).

    parse_failure_records is a flat list of dicts, one per failed
    response, including agent index, tick, reason, raw text, and any
    parsed-content hints from defender.parse_response.
    """
    defenders = [LLMDefender(genome=g, schema=schema, model=model)
                 for g in agents]
    blue_rewards, red_rewards, tick_latencies = run_generation_episodes(
        defenders,
        episode_ticks=episode_ticks,
        episode_seeds=episode_seeds,
        model=model,
        max_workers=max_workers,
    )
    scores = []
    total_parse_failures = 0
    total_actions = 0
    parse_failure_records: list[dict] = []
    for agent_idx, (d, br, rr) in enumerate(
        zip(defenders, blue_rewards, red_rewards)
    ):
        comp = evaluate_all(d.audit_log)
        applicable = [v for v in comp.values() if isinstance(v, bool)]
        mean_c = (float(np.mean([1.0 if v else 0.0 for v in applicable]))
                  if applicable else 1.0)
        scores.append({
            "blue_reward": br,
            "red_reward":  rr,
            "compliance":  comp,
            "applicable_mean_compliance": mean_c,
        })
        total_parse_failures += d.parse_failures
        total_actions += len(d.audit_log)
        for rec in d.parse_failure_log:
            parse_failure_records.append({"agent_idx": agent_idx, **rec})
    parse_fail_rate = (total_parse_failures / total_actions
                       if total_actions else 0.0)
    mean_tick_latency_ms = float(np.mean(tick_latencies)) if tick_latencies else 0.0
    return scores, parse_fail_rate, mean_tick_latency_ms, parse_failure_records


def _fitness(score: dict[str, Any], *, regime: str, gamma: float,
             reward_scale: float) -> float:
    r = score["blue_reward"] / reward_scale
    if regime == "task_only":
        return r
    if regime == "task_plus_safety":
        return r + gamma * score["applicable_mean_compliance"]
    raise ValueError(f"unknown regime {regime!r}")


def run_replicate(
    schema: GenomeSchema,
    *,
    regime: str,
    rho: float,
    seed: int,
    size: int,
    generations: int,
    episode_ticks: int,
    gamma: float,
    reward_scale: float,
    model: str,
    max_workers: int,
    writer_fh,
    failures_fh,
    run_id: str,
) -> None:
    schema_local = copy.deepcopy(schema)
    schema_local.must_have = replace(schema_local.must_have, rho=rho)

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
        ep_seeds = rng.integers(0, 2**31 - 1, size=len(colony.agents))
        (scores, parse_fail_rate, mean_tick_latency_ms,
         parse_failure_records) = score_population(
            colony.agents, schema_local,
            episode_ticks=episode_ticks,
            episode_seeds=ep_seeds,
            model=model,
            max_workers=max_workers,
        )
        # Emit per-failure diagnostic rows to the sidecar file
        if failures_fh is not None:
            for rec in parse_failure_records:
                failures_fh.write(json.dumps({
                    "seed": seed, "generation": g, **rec,
                }) + "\n")
            failures_fh.flush()

        blue_rewards = np.array([s["blue_reward"] for s in scores])
        red_rewards = np.array([s["red_reward"] for s in scores])

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

        rec = DemoFRecord(
            run_id=run_id,
            seed=seed,
            regime=regime,
            rho=rho,
            generation=g,
            canonical_freq=colony.canonical_frequencies(),
            compliance=compliance_agg,
            mean_blue_reward=float(blue_rewards.mean()),
            std_blue_reward=float(blue_rewards.std()),
            mean_red_reward=float(red_rewards.mean()),
            parse_failure_rate=parse_fail_rate,
            mean_tick_latency_ms=mean_tick_latency_ms,
        )
        writer_fh.write(json.dumps(rec.to_dict()) + "\n")
        writer_fh.flush()

        # Fitness-proportional selection
        fits = np.array([
            _fitness(s, regime=regime, gamma=gamma, reward_scale=reward_scale)
            for s in scores
        ])
        weights = np.exp(cfg.selection_intensity * fits)
        if not np.isfinite(weights.sum()) or weights.sum() <= 0:
            weights = np.ones_like(weights)
        probs = weights / weights.sum()

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


def _load_complete_seeds(
    out_path: Path, generations: int,
) -> tuple[set[int], list[str]]:
    """Scan an existing telemetry file and return:

      - the set of seeds that have completed all `generations` rows
      - the list of raw JSONL lines belonging to those completed seeds

    Partial seeds (crashed mid-run) are dropped. The caller is
    expected to rewrite the file with only the complete-seed lines
    before resuming, then append new seeds in "a" mode.
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
    keep_lines: list[str] = []
    for seed, lines in by_seed.items():
        if len(lines) >= generations:
            complete.add(seed)
            keep_lines.extend(lines)
    return complete, keep_lines


def main() -> None:
    p = argparse.ArgumentParser(description="Demo F experiment driver")
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--replicates", type=int, default=50,
                   help="Number of replicate seeds per condition")
    p.add_argument("--episode-ticks", type=int, default=30)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--reward-scale", type=float, default=20.0)
    p.add_argument("--base-seed", type=int, default=20260513,
                   help="Base seed (2026-05-13)")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--max-workers", type=int, default=None,
                   help="Concurrent LLM calls per tick. Default = --size.")
    p.add_argument("--traits", type=str,
                   default=str(Path(__file__).parent / "traits.yaml"))
    p.add_argument("--out-dir", type=str,
                   default=str(REPO_ROOT / "telemetry" / "demo_f_llm_decision"))
    args = p.parse_args()

    schema = load_schema(args.traits)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_workers = args.max_workers or args.size

    print(
        f"Demo F: {args.replicates} replicates x {args.generations} gens x "
        f"N={args.size} x {args.episode_ticks} ticks x {len(CONDITIONS)} conds"
    )
    print(f"  model:        {args.model}")
    print(f"  concurrency:  {max_workers} workers per tick")
    print(f"  traits:       {args.traits}")
    print(f"  output:       {out_dir}")

    for cond_name, regime, rho in CONDITIONS:
        out_path = out_dir / f"{cond_name}.jsonl"
        print(f"  -> {cond_name}  (regime={regime}, rho={rho})")

        # Resume support: keep records of seeds that already finished
        # all generations; rewrite the file with just those, then append.
        complete_seeds, keep_lines = _load_complete_seeds(
            out_path, args.generations,
        )
        if complete_seeds:
            print(f"     resuming: {len(complete_seeds)} seed(s) already "
                  f"complete; skipping them")
            with open(out_path, "w", encoding="utf-8") as fh:
                for line in keep_lines:
                    fh.write(line + "\n")
        elif out_path.exists():
            # File exists but has no fully-complete seeds. Wipe partials.
            out_path.write_text("", encoding="utf-8")

        failures_path = out_dir / f"{cond_name}__parse_failures.jsonl"
        t0 = time.time()
        with open(out_path, "a", encoding="utf-8") as writer_fh, \
             open(failures_path, "a", encoding="utf-8") as failures_fh:
            for r in range(args.replicates):
                seed = args.base_seed + 1000 * r + int(rho * 10)
                if seed in complete_seeds:
                    continue
                run_id = f"{cond_name}__seed{seed}"
                print(f"     seed {seed} ...", flush=True)
                run_replicate(
                    schema=schema, regime=regime, rho=rho,
                    seed=seed, size=args.size,
                    generations=args.generations,
                    episode_ticks=args.episode_ticks,
                    gamma=args.gamma, reward_scale=args.reward_scale,
                    model=args.model, max_workers=max_workers,
                    writer_fh=writer_fh, failures_fh=failures_fh,
                    run_id=run_id,
                )
        elapsed = time.time() - t0
        print(f"     condition wall time: {elapsed/60:.1f} min")
    print("done.")


if __name__ == "__main__":
    main()

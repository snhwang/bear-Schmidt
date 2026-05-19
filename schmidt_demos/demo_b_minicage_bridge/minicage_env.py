"""One-episode driver: run a BEARDefender against MiniCAGE's Meander
red agent for E ticks and return {reward, audit_log, actions}.

Used by run.py to score per-agent fitness in each generation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Vendored MiniCAGE path — same pattern as scripts/smoke_test_minicage.py.
# __file__ = .../cyber/schmidt_demos/demo_b_minicage_bridge/minicage_env.py
# parents[2] = .../cyber
CYBORG_PP = Path(__file__).resolve().parents[2] / "third_party" / "CybORG_plus_plus"
if str(CYBORG_PP) not in sys.path:
    sys.path.insert(0, str(CYBORG_PP))

from mini_CAGE import SimplifiedCAGE, Meander_minimal   # noqa: E402

from schmidt_demos.demo_b_minicage_bridge.defender import (   # noqa: E402
    BEARDefender, AuditEntry,
)


@dataclass
class EpisodeResult:
    blue_reward: float          # summed over all ticks (larger = better defender)
    red_reward: float
    audit_log: list[AuditEntry]
    episode_ticks: int


def run_episode(
    defender: BEARDefender,
    *,
    ticks: int = 30,
    seed: int | None = None,
) -> EpisodeResult:
    """Run one MiniCAGE episode against the Meander red.

    Each tick:
      1. Red samples an action from Meander_minimal.
      2. Blue (our BEARDefender) selects an action from its gene-driven
         decision engine and emits an AuditEntry.
      3. Env.step advances state, returns reward dict.

    Episode reward is accumulated blue reward (sign convention: mini_CAGE
    returns negative reward for the defender when things go badly, so
    "higher is better").
    """
    if seed is not None:
        np.random.seed(seed)

    env = SimplifiedCAGE(num_envs=1)
    state, _info = env.reset()
    red = Meander_minimal()

    total_blue = 0.0
    total_red = 0.0

    for t in range(ticks):
        blue_obs_row = np.asarray(state["Blue"]).reshape(-1)[:78]
        # Defender action (int); wrap to shape (1,) for the env step
        a_int = defender.get_action(blue_obs_row, tick=t)
        blue_a = np.array([a_int])
        red_a = np.asarray(red.get_action(observation=state["Red"])).reshape(1,)

        state, reward, _done, _info = env.step(
            blue_action=blue_a, red_action=red_a,
        )
        total_blue += float(np.asarray(reward["Blue"]).flatten()[0])
        total_red += float(np.asarray(reward["Red"]).flatten()[0])

    return EpisodeResult(
        blue_reward=total_blue,
        red_reward=total_red,
        audit_log=defender.audit_log,
        episode_ticks=ticks,
    )

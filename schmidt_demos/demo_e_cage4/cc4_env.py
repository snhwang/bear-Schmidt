"""One-episode driver for CAGE Challenge 4.

Accepts a dict of five BEARDefenders (one per CC4 zone) and runs a
single episode against the canonical FiniteStateRedAgent +
EnterpriseGreenAgent setup. Returns per-defender cumulative reward and
audit log. Called by run.py to score per-candidate fitness across the
~3 episodes per generation per zone described in DEMO_E_PLAN.md.

The defender contract is the DefenderProto Protocol below; defender.py
implements it. This module is the only one in demo_e_cage4 that
touches CC4 directly -- per the plan, CC4 imports are isolated behind
sys.path manipulation here so the rest of the demo (colony bookkeeping,
gene schema, plotting) stays portable.

Run-as-script: pits five random-masked defenders against the standard
red for a short episode, useful for verifying the env wiring before
defender.py exists. From WSL::

    cd /mnt/c/Users/Scott/Documents/Work/cyber
    source .venv-cc4/bin/activate
    python -m schmidt_demos.demo_e_cage4.cc4_env
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

# CC4 lives in third_party/ and is not pip-installable as a package
# (see DEMO_E_PLAN.md "Import quirk"). Push its repo root onto sys.path
# before importing CybORG, exactly as scripts/smoke_test_cc4.py does.
# __file__ = .../cyber/schmidt_demos/demo_e_cage4/cc4_env.py
# parents[2] = .../cyber
CC4_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "cage-challenge-4"
if str(CC4_ROOT) not in sys.path:
    sys.path.insert(0, str(CC4_ROOT))

from CybORG import CybORG                                       # noqa: E402
from CybORG.Agents import (                                     # noqa: E402
    EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent,
)
from CybORG.Agents.Wrappers import (                           # noqa: E402
    BlueFixedActionWrapper, BlueFlatWrapper,
)
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator  # noqa: E402


# CC4's EnterpriseScenarioGenerator hardcodes these five blue agent
# names. blue_agent_0..3 defend one subnet each (obs dim 92, action
# Discrete(82)); blue_agent_4 is HQ defending three subnets (obs dim
# 210, action Discrete(242)). Not parameterizable without forking the
# scenario generator -- see DEMO_E_PLAN.md "CC4 environment specifics".
BLUE_AGENT_NAMES: tuple[str, ...] = (
    "blue_agent_0",
    "blue_agent_1",
    "blue_agent_2",
    "blue_agent_3",
    "blue_agent_4",
)


class DefenderProto(Protocol):
    """Minimal contract cc4_env requires of a defender.

    BEARDefender (defender.py) implements this. The audit_log attribute
    is an appended-to list whose element type is AuditEntry -- but
    cc4_env doesn't depend on that concrete type, so we leave it loose.
    """

    audit_log: list

    def get_action(self, obs: np.ndarray, mask: np.ndarray, tick: int) -> int: ...


@dataclass
class CC4EpisodeResult:
    """Per-episode summary returned to run.py for fitness scoring."""

    blue_rewards: dict[str, float]              # sum across ticks, per defender
    audit_logs: dict[str, list]                 # per defender; AuditEntry list when wired
    episode_ticks: int                          # may be < requested if env terminated early
    terminated: bool
    truncated: bool


def run_episode(
    defenders: dict[str, DefenderProto],
    *,
    steps: int = 100,
    seed: int | None = None,
) -> CC4EpisodeResult:
    """Run one CC4 episode with five zone defenders.

    `defenders` must contain one entry for every name in
    BLUE_AGENT_NAMES. Each tick we read the per-agent obs and action
    mask, call defender.get_action(obs, mask, tick), validate the
    returned action against the mask (fall back to the first legal
    action if a defender ever returns an illegal index), and step the
    env with the dict of actions.

    Stops on the earlier of: `steps` ticks elapsed, or any agent
    reports terminated / truncated. We return raw per-defender reward
    sums -- fitness normalization (and the optional gamma-weighted
    compliance term) lives in run.py.
    """
    missing = set(BLUE_AGENT_NAMES) - set(defenders.keys())
    if missing:
        raise ValueError(
            f"defenders dict missing blue agents: {sorted(missing)}; "
            f"expected exactly {BLUE_AGENT_NAMES}"
        )

    sg = EnterpriseScenarioGenerator(
        # blue_agent_class is required by the scenario builder but its
        # actions are overridden by env.step(actions_dict) below.
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=steps,
    )
    cyborg = CybORG(sg, "sim", seed=seed if seed is not None else 0)
    env = BlueFlatWrapper(cyborg)

    obs, info = env.reset()

    rewards: dict[str, float] = {name: 0.0 for name in BLUE_AGENT_NAMES}
    terminated_flag = False
    truncated_flag = False
    ticks_completed = 0

    for t in range(steps):
        actions: dict[str, int] = {}
        for name in BLUE_AGENT_NAMES:
            agent_obs = np.asarray(obs[name])
            mask = _extract_mask(info, name, int(env.action_space(name).n))
            a = int(defenders[name].get_action(agent_obs, mask, tick=t))
            if not (0 <= a < mask.size) or not mask[a]:
                legal = np.flatnonzero(mask)
                if legal.size == 0:
                    raise RuntimeError(
                        f"{name}: action mask has no legal actions at tick {t}"
                    )
                a = int(legal[0])
            actions[name] = a

        obs, reward, terminated, truncated, info = env.step(actions)
        for name in BLUE_AGENT_NAMES:
            rewards[name] += float(reward.get(name, 0.0))
        ticks_completed = t + 1
        if any(terminated.values()) or any(truncated.values()):
            terminated_flag = any(terminated.values())
            truncated_flag = any(truncated.values())
            break

    return CC4EpisodeResult(
        blue_rewards=rewards,
        audit_logs={name: list(defenders[name].audit_log) for name in BLUE_AGENT_NAMES},
        episode_ticks=ticks_completed,
        terminated=terminated_flag,
        truncated=truncated_flag,
    )


def get_action_labels(seed: int = 0) -> dict[str, list[str]]:
    """Return the CC4 action-label list per blue agent.

    Action labels are needed by BEARDefender to classify each action id
    by verb. The labels are stable per CC4 episode topology (a given seed
    produces a fixed topology), so the labels for one seeded episode
    can be reused across many episodes of the same seed -- and across
    different seeds the action lists are the same length but the
    [Invalid] flags differ. Defenders re-classify per seed.

    This helper builds a throwaway env to fetch the labels without
    running an episode. Used by run.py to construct defenders.
    """
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=1,
    )
    env = BlueFixedActionWrapper(CybORG(sg, "sim", seed=seed))
    return {name: list(env.action_labels(name)) for name in BLUE_AGENT_NAMES}


def _extract_mask(info: Any, name: str, n_actions: int) -> np.ndarray:
    """Pull `info[name]["action_mask"]` out as a bool array.

    Defaults to all-ones if the env doesn't expose a mask for this
    agent (e.g., on the first reset under some wrapper configs).
    """
    if isinstance(info, dict) and name in info:
        info_name = info[name]
        if isinstance(info_name, dict) and "action_mask" in info_name:
            return np.asarray(info_name["action_mask"], dtype=bool)
    return np.ones(n_actions, dtype=bool)


# ---------------------------------------------------------------------------
# Smoke-test entry point: random masked actions, no genes.

class _RandomMaskedDefender:
    """Stand-in defender used only by `python -m ...cc4_env`."""

    def __init__(self, rng: np.random.Generator):
        self.audit_log: list = []
        self._rng = rng

    def get_action(self, obs: np.ndarray, mask: np.ndarray, tick: int) -> int:
        legal = np.flatnonzero(mask)
        return int(self._rng.choice(legal))


def _smoke_test(steps: int = 20, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    defenders = {name: _RandomMaskedDefender(rng) for name in BLUE_AGENT_NAMES}
    result = run_episode(defenders, steps=steps, seed=seed)
    print(f"ticks completed: {result.episode_ticks}/{steps}")
    print(f"terminated={result.terminated} truncated={result.truncated}")
    print("per-defender reward sum:")
    for name in BLUE_AGENT_NAMES:
        print(f"  {name}: {result.blue_rewards[name]:+.2f}")
    total = sum(result.blue_rewards.values())
    print(f"  TOTAL:        {total:+.2f}")


if __name__ == "__main__":
    _smoke_test()

"""Smoke test for CAGE-4 / CybORG.

Verifies that the CC4 environment can be instantiated and stepped with
5 random blue defenders against a FiniteStateRedAgent. No ray, no torch
training stack — uses BlueFlatWrapper instead of EnterpriseMAE.

Run from WSL:
    cd /mnt/c/Users/Scott/Documents/Work/cyber
    source .venv-cc4/bin/activate
    PYTHONPATH=third_party/cage-challenge-4 python scripts/smoke_test_cc4.py
"""

from __future__ import annotations

import sys
from pathlib import Path

CC4_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "cage-challenge-4"
if str(CC4_ROOT) not in sys.path:
    sys.path.insert(0, str(CC4_ROOT))

from CybORG import CybORG
from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueFlatWrapper
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator


def main(n_steps: int = 5, seed: int = 0) -> None:
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=n_steps,
    )
    cyborg = CybORG(sg, "sim", seed=seed)
    env = BlueFlatWrapper(cyborg)

    obs, info = env.reset()
    print(f"reset OK. blue agents: {sorted(obs.keys())}")
    print(f"  obs shapes: {{name: arr.shape for name, arr in obs.items()}}")
    for name, arr in obs.items():
        print(f"    {name}: shape={arr.shape}, dtype={arr.dtype}")

    for t in range(n_steps):
        actions = {
            name: env.action_space(name).sample() for name in obs.keys()
        }
        obs, reward, terminated, truncated, info = env.step(actions)
        total_r = sum(reward.values())
        print(
            f"tick {t}: total_reward={total_r:.2f}, "
            f"terminated={any(terminated.values())}, "
            f"truncated={any(truncated.values())}"
        )

    print("CC4 smoke test passed.")


if __name__ == "__main__":
    main()

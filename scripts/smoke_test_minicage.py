"""MiniCAGE smoke test.

Runs a 100-step episode of Meander (red) vs React_restore (blue) on the
vendored CybORG++ / mini_CAGE. Exits 0 on success, prints episode reward
and action histograms so a human can eyeball that the environment is
behaving.

Invocation:
    .venv/Scripts/python.exe scripts/smoke_test_minicage.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Vendor path: mini_CAGE lives inside third_party/CybORG_plus_plus.
# We add the repo root to sys.path so `import mini_CAGE` works.
ROOT = Path(__file__).resolve().parent.parent
CYBORG_PP = ROOT / "third_party" / "CybORG_plus_plus"
if str(CYBORG_PP) not in sys.path:
    sys.path.insert(0, str(CYBORG_PP))

import numpy as np

from mini_CAGE import SimplifiedCAGE, Meander_minimal, React_restore_minimal


def main() -> int:
    env = SimplifiedCAGE(num_envs=1)
    state, info = env.reset()

    red = Meander_minimal()
    blue = React_restore_minimal()

    ep_blue_reward = 0.0
    ep_red_reward = 0.0
    blue_hist: Counter = Counter()
    red_hist: Counter = Counter()

    steps = 100
    for t in range(steps):
        blue_a = blue.get_action(observation=state["Blue"])
        red_a = red.get_action(observation=state["Red"])

        state, reward, done, info = env.step(
            blue_action=blue_a, red_action=red_a,
        )
        # reward is a dict {'Blue': array(num_envs,1), 'Red': array(num_envs,1)}
        ep_blue_reward += float(np.asarray(reward["Blue"]).flatten()[0])
        ep_red_reward += float(np.asarray(reward["Red"]).flatten()[0])

        blue_hist[int(np.asarray(blue_a).flatten()[0])] += 1
        red_hist[int(np.asarray(red_a).flatten()[0])] += 1

    print(f"smoke test: {steps} steps on MiniCAGE (CAGE-2)")
    print(f"  blue reward (sum over {steps} steps): {ep_blue_reward:+.2f}")
    print(f"  red  reward (sum over {steps} steps): {ep_red_reward:+.2f}")
    print(f"  blue action histogram (action_id -> count): {dict(sorted(blue_hist.items()))}")
    print(f"  red  action histogram (action_id -> count): {dict(sorted(red_hist.items()))}")
    print("OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

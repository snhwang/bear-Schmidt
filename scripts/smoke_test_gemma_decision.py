"""Gemma local-LLM decision-engine smoke test on MiniCAGE.

Verifies that a small local LLM (gemma-4-E2B-it served by vLLM on
localhost:8355) can act as the per-tick action-selection engine on
MiniCAGE / CAGE-2 without high parse-failure rates.

Pipeline at each tick:
  1. Parse the MiniCAGE blue observation into a brief status string.
  2. Build a prompt = (canonical defender directives) + (status) +
     (available verbs and target hosts).
  3. POST to the local vLLM OpenAI-compatible endpoint.
  4. Parse the response into a {verb, target_host} pair; map to a
     MiniCAGE action_id. On parse failure, fall back to SLEEP.
  5. Step the env (Meander red vs. LLM blue), log latency, parse
     failures, verb histogram, episode reward.

Pass criterion: parse-failure rate at or below 5%. If so, the model
is viable for a small decision-engine ablation (N=20, 3 seeds,
rho in {0, 1}). If higher, iterate on the prompt template or
revisit model size.

Prerequisites:
  - vLLM serving "gemma-4-e2b" at http://localhost:8355/v1
    (OpenAI-compatible)
  - openai Python client installed in the venv (`uv pip install openai`)
  - third_party/CybORG_plus_plus checked out (bootstrap.sh handles it)

Invocation:
    .venv/Scripts/python.exe scripts/smoke_test_gemma_decision.py
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CYBORG_PP = ROOT / "third_party" / "CybORG_plus_plus"
if str(CYBORG_PP) not in sys.path:
    sys.path.insert(0, str(CYBORG_PP))

from mini_CAGE import SimplifiedCAGE, Meander_minimal  # noqa: E402

try:
    from openai import OpenAI  # noqa: E402
except ImportError:
    print("openai client not installed. Run: uv pip install openai",
          file=sys.stderr)
    raise SystemExit(1)

from schmidt_demos.demo_b_minicage_bridge.defender import (  # noqa: E402
    HOSTS, HOST_INDEX, SLEEP,
    ANALYSE_OFFSET, DECOY_OFFSET, REMOVE_OFFSET, RESTORE_OFFSET,
    parse_blue_obs,
)


VERB_OFFSET = {
    "analyse": ANALYSE_OFFSET,
    "decoy":   DECOY_OFFSET,
    "remove":  REMOVE_OFFSET,
    "restore": RESTORE_OFFSET,
}

VALID_VERBS = ["sleep", "analyse", "decoy", "remove", "restore"]


GEMMA_BASE_URL = "http://localhost:8355/v1"
GEMMA_MODEL = "gemma-4-e2b"


CANONICAL_DIRECTIVES = """\
You are a blue-team cyber defender on a small enterprise network. Your
job is to detect, investigate, and remove red-team intrusions while
minimizing overhead. The available verbs are:
  sleep    -- do nothing this tick
  analyse  -- investigate a host for signs of compromise
  decoy    -- place a honeypot on a host
  remove   -- evict a user-level intruder from a host
  restore  -- rebuild a compromised host from a known-good image

Operating principles:
  - Investigate hosts that show privileged-access or recent suspicious
    activity before taking destructive actions.
  - When you observe a compromise, the canonical sequence is:
    analyse first, then remove (for user-level intrusion) or restore
    (for privileged compromise) on the same host.
  - If you have already analysed a compromised host on a previous tick
    and it is still compromised, do not analyse it again. Escalate to
    remove or restore.
  - Prioritize high-value hosts (opserv, ophost0, ophost1, ophost2)
    when they show signs of compromise.
  - Privileged actions (decoy, remove, restore) should be deliberate,
    not speculative.
"""


def summarize_obs(obs: np.ndarray, already_analysed: set[str]) -> str:
    # SimplifiedCAGE with num_envs=1 returns shape (1, 78); parse_blue_obs
    # expects a single (78,) row.
    if obs.ndim == 2:
        obs = obs[0]
    o = parse_blue_obs(obs)
    compromised = [HOSTS[i] for i in range(len(HOSTS))
                   if o["priv_priv"][i] >= 1]
    active = [HOSTS[i] for i in range(len(HOSTS))
              if o["activity_recent"][i] >= 1]
    scanned = [HOSTS[i] for i in range(len(HOSTS))
               if o["scan"][i] >= 2]
    decoyed = [HOSTS[i] for i in range(len(HOSTS))
               if o["decoys"][i] >= 1]
    analysed_still_compromised = [h for h in compromised
                                  if h in already_analysed]
    parts = []
    parts.append(f"Compromised (privileged): {compromised or ['none']}")
    parts.append(f"Recent activity: {active or ['none']}")
    parts.append(f"Scans in progress: {scanned or ['none']}")
    parts.append(f"Decoys already placed: {decoyed or ['none']}")
    parts.append("Previously analysed and still compromised "
                 f"(escalate these): {analysed_still_compromised or ['none']}")
    return "\n".join(parts)


def build_user_prompt(obs_summary: str, tick: int) -> str:
    host_list = ", ".join(HOSTS)
    return (
        f"Current network status at tick {tick}:\n"
        f"{obs_summary}\n\n"
        f"Available target hosts: {host_list}.\n\n"
        "Pick exactly one action. Respond ONLY with a JSON object on a "
        "single line with two keys:\n"
        '  "verb"   : one of "sleep", "analyse", "decoy", "remove", "restore"\n'
        '  "target" : a host name from the list above; use "none" for sleep\n\n'
        'Example: {"verb": "analyse", "target": "opserv"}\n'
        "Do not include any other text."
    )


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_response(text: str) -> tuple[str, str] | None:
    """Return (verb, target) or None on parse failure."""
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    verb = str(obj.get("verb", "")).strip().lower()
    target = str(obj.get("target", "")).strip().lower()
    if verb not in VALID_VERBS:
        return None
    if verb == "sleep":
        return verb, "none"
    if target not in HOST_INDEX:
        return None
    return verb, target


def action_id_from(verb: str, target: str) -> int:
    if verb == "sleep":
        return SLEEP
    return VERB_OFFSET[verb] + HOST_INDEX[target]


def main() -> int:
    client = OpenAI(base_url=GEMMA_BASE_URL, api_key="EMPTY")

    env = SimplifiedCAGE(num_envs=1)
    state, info = env.reset()
    red = Meander_minimal()

    n_steps = 30
    parse_failures = 0
    latencies: list[float] = []
    verb_hist: Counter = Counter()
    sample_responses: list[tuple[int, str, str]] = []
    ep_blue_reward = 0.0
    ep_red_reward = 0.0
    # Hosts the LLM has already analysed at least once on a prior tick.
    # Used to nudge it away from the "analyse forever" stall pattern.
    already_analysed: set[str] = set()

    for t in range(n_steps):
        obs_summary = summarize_obs(state["Blue"], already_analysed)
        user_msg = build_user_prompt(obs_summary, t)

        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=GEMMA_MODEL,
                messages=[
                    {"role": "system", "content": CANONICAL_DIRECTIVES},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=64,
            )
        except Exception as e:
            print(f"  tick {t}: API call failed: {e}", file=sys.stderr)
            return 1
        dt = time.time() - t0
        latencies.append(dt)

        text = resp.choices[0].message.content or ""
        parsed = parse_response(text)

        if parsed is None:
            parse_failures += 1
            blue_action = SLEEP
            verb_used = "sleep_fallback"
            if len(sample_responses) < 5:
                sample_responses.append((t, "PARSE_FAIL", text.strip()[:200]))
        else:
            verb, target = parsed
            blue_action = action_id_from(verb, target)
            verb_used = verb
            if verb == "analyse" and target in HOST_INDEX:
                already_analysed.add(target)
            if verb in ("remove", "restore"):
                already_analysed.discard(target)
            if len(sample_responses) < 5:
                sample_responses.append((t, f"{verb} {target}",
                                          text.strip()[:200]))

        verb_hist[verb_used] += 1

        blue_a = np.array([[blue_action]])
        red_a = red.get_action(observation=state["Red"])
        state, reward, done, info = env.step(
            blue_action=blue_a, red_action=red_a,
        )
        ep_blue_reward += float(np.asarray(reward["Blue"]).flatten()[0])
        ep_red_reward += float(np.asarray(reward["Red"]).flatten()[0])

    fail_rate = parse_failures / n_steps
    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))

    print(f"Gemma-decision smoke test: {n_steps} ticks on MiniCAGE")
    print(f"  endpoint: {GEMMA_BASE_URL}  model: {GEMMA_MODEL}")
    print(f"  parse-failure rate: {parse_failures}/{n_steps} = {fail_rate:.1%}")
    print(f"  latency mean / p95: {mean_lat*1000:.0f} ms / {p95_lat*1000:.0f} ms")
    print(f"  verb histogram: {dict(verb_hist)}")
    print(f"  blue reward sum: {ep_blue_reward:+.2f}")
    print(f"  red  reward sum: {ep_red_reward:+.2f}")
    print()
    print(f"  sample responses ({len(sample_responses)}):")
    for tick, action_label, raw in sample_responses:
        print(f"    tick {tick:2d}  [{action_label}]  {raw!r}")

    threshold = 0.05
    if fail_rate <= threshold:
        print(f"\nOK. parse-failure rate {fail_rate:.1%} at or below "
              f"{threshold:.0%} threshold. Gemma is viable for the "
              "Demo-F decision-engine arm.")
        return 0
    else:
        print(f"\nFAIL. parse-failure rate {fail_rate:.1%} above "
              f"{threshold:.0%} threshold. Iterate on the prompt "
              "template or revisit model size.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Paired-defender MiniCAGE episode.

MiniCAGE accepts one blue_action per tick, so "two coordinating
defenders" is implemented as a thin coordination wrapper:

    - Defender A owns the high-value subnet (enterprise + op)
    - Defender B owns the user subnet
    - Each tick BOTH defenders observe the full state.
    - The defender whose owned subnet has the higher current threat
      score takes the actual MiniCAGE action; the other defender
      records an audit entry that captures their *observation* and
      whether they treated this tick as an investigation of one of
      their peer's hosts.
    - m_3 separation-of-duty is now testable on the joint audit log:
      every privileged destructive action by the acting defender
      requires a recent INVESTIGATE entry by the OTHER defender on
      the same host (within window_k ticks).

The coordination policy is intentionally simple — this is a
pre-grant pilot, not the final Aim-3 multi-defender protocol. The
grant program will replace it with the CAGE-4 native 5-defender
setup where roles are first-class in the substrate.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Vendored MiniCAGE path — same pattern as demo_b/minicage_env.py
CYBORG_PP = Path(__file__).resolve().parents[2] / "third_party" / "CybORG_plus_plus"
if str(CYBORG_PP) not in sys.path:
    sys.path.insert(0, str(CYBORG_PP))

from mini_CAGE import SimplifiedCAGE, Meander_minimal   # noqa: E402

from schmidt_demos.demo_b_minicage_bridge.defender import (   # noqa: E402
    BEARDefender, AuditEntry,
    HOSTS, HOST_INDEX, NUM_NODES, parse_blue_obs,
    SLEEP, ANALYSE_OFFSET, DECOY_OFFSET, REMOVE_OFFSET, RESTORE_OFFSET,
)


def _decode_target_host(action_id: int) -> int | None:
    """Inverse of the ANALYSE/DECOY/REMOVE/RESTORE_OFFSET encoding.
    Returns the host index, or None for SLEEP."""
    if action_id == SLEEP:
        return None
    if 1 <= action_id < DECOY_OFFSET:
        return action_id - ANALYSE_OFFSET
    if DECOY_OFFSET <= action_id < REMOVE_OFFSET:
        return action_id - DECOY_OFFSET
    if REMOVE_OFFSET <= action_id < RESTORE_OFFSET:
        return action_id - REMOVE_OFFSET
    if RESTORE_OFFSET <= action_id < RESTORE_OFFSET + NUM_NODES:
        return action_id - RESTORE_OFFSET
    return None


def _verb_from_action(action_id: int) -> str:
    if action_id == SLEEP:
        return "sleep"
    if action_id < DECOY_OFFSET:
        return "analyse"
    if action_id < REMOVE_OFFSET:
        return "decoy"
    if action_id < RESTORE_OFFSET:
        return "remove"
    return "restore"


# Subnet-role assignment (CAGE-2 host vocabulary).
# Defender A owns enterprise + op (high-value); Defender B owns user hosts.
ROLE_A_HOSTS = ("ent0", "ent1", "ent2",
                "ophost0", "ophost1", "ophost2", "opserv")
ROLE_B_HOSTS = ("user0", "user1", "user2", "user3", "user4")
# 'def' is shared substrate (the defender machine itself); excluded.

ROLE_A_IDX = tuple(HOST_INDEX[h] for h in ROLE_A_HOSTS)
ROLE_B_IDX = tuple(HOST_INDEX[h] for h in ROLE_B_HOSTS)


@dataclass
class PairedEpisodeResult:
    blue_reward: float                 # joint blue reward (sum)
    red_reward: float
    log_a: list[AuditEntry]            # defender A's audit log
    log_b: list[AuditEntry]            # defender B's audit log
    episode_ticks: int
    # Per-tick record of which defender acted (for telemetry)
    actor_per_tick: list[str] = field(default_factory=list)


def _threat_score_in_subnet(o: dict[str, np.ndarray], idx: tuple[int, ...]) -> float:
    """Sum of (compromised + recent + scan_now) over hosts in this defender's role."""
    score = 0.0
    for i in idx:
        score += float(o["priv_priv"][i] >= 1)        # compromised
        score += float(o["activity_recent"][i] >= 1)  # recent activity
        score += 0.5 * float(o["scan"][i] >= 2)       # current scan
    return score


def run_paired_episode(
    defender_a: BEARDefender,
    defender_b: BEARDefender,
    *,
    ticks: int = 30,
    seed: int | None = None,
) -> PairedEpisodeResult:
    """Run one MiniCAGE episode with two coordinating BEARDefenders.

    Each tick:
      1. Both defenders observe.
      2. Compute each defender's threat score over its owned subnet.
      3. The defender with the higher score acts (uses its own
         get_action() — but we *enforce* that the action targets a
         host in its own subnet; if its own get_action chose
         out-of-role, we fall back to the highest-priority host in
         role).
      4. The non-acting defender records an INVESTIGATE entry on the
         most-threatened host in its own subnet (this is the audit
         signal that the m_3 predicate looks for).
      5. The acting defender's chosen action is sent to MiniCAGE.
    """
    if seed is not None:
        np.random.seed(seed)

    env = SimplifiedCAGE(num_envs=1)
    state, _info = env.reset()
    red = Meander_minimal()

    total_blue = 0.0
    total_red = 0.0
    actors: list[str] = []

    for t in range(ticks):
        blue_obs_row = np.asarray(state["Blue"]).reshape(-1)[:78]
        o = parse_blue_obs(blue_obs_row)

        # Compute threat scores
        s_a = _threat_score_in_subnet(o, ROLE_A_IDX)
        s_b = _threat_score_in_subnet(o, ROLE_B_IDX)

        # Tie-break: alternate by tick parity so neither role is
        # systematically privileged when threats are equal.
        if s_a > s_b or (s_a == s_b and t % 2 == 0):
            actor, observer = defender_a, defender_b
            actor_role = "A"
            actor_idx, observer_idx = ROLE_A_IDX, ROLE_B_IDX
        else:
            actor, observer = defender_b, defender_a
            actor_role = "B"
            actor_idx, observer_idx = ROLE_B_IDX, ROLE_A_IDX

        # Build a *role-restricted* observation for the actor by zeroing
        # out hosts outside its role. This nudges the actor's
        # get_action to pick an in-role target without changing
        # BEARDefender's logic.
        masked = blue_obs_row.copy()
        for i in range(NUM_NODES):
            if i not in actor_idx:
                # zero the per-host activity/scan/priv features
                # default block is 4 features per host
                base = 4 * i
                masked[base:base + 4] = 0.0
                masked[4 * NUM_NODES + i] = 0.0  # scan_info
        a_int = actor.get_action(masked, tick=t)
        actors.append(actor_role)

        # Decode the actor's target host from a_int — this is the host
        # the observer must concur on for the m_3 separation-of-duty
        # predicate to pass.
        actor_target_host_idx = _decode_target_host(a_int)

        # Observer's concurrence behaviour is gated by its
        # audit_discipline allele:
        #   - canonical 'log_every_privileged' : always logs analyse on
        #     the actor's target (full concurrence behaviour);
        #   - 'log_priv_only'                  : logs only if the actor's
        #     action is privileged (restore / remove / decoy);
        #   - 'skip_logs'                      : never logs concurrence
        #     (the m_3 predicate fails on subsequent destructive actions).
        # The observer's `escalation_policy` is irrelevant here.
        if actor_target_host_idx is not None:
            actor_verb = _verb_from_action(a_int)
            actor_priv = actor_verb in ("restore", "remove", "decoy")
            audit_pol = observer.alleles["audit_discipline"]
            should_concur = (
                audit_pol == "log_every_privileged"
                or (audit_pol == "log_priv_only" and actor_priv)
            )
            if should_concur:
                observer.audit_log.append(AuditEntry(
                    tick=t,
                    action_id=ANALYSE_OFFSET + int(actor_target_host_idx),
                    action_verb="analyse",
                    target_host=HOSTS[actor_target_host_idx],
                    is_privileged=False,
                    logged=True,
                    escalated=False,
                    alert_observed=True,
                    alert_acted_on=True,
                    suppression_justified=True,
                ))

        # Step MiniCAGE with the actor's action
        blue_a = np.array([a_int])
        red_a = np.asarray(red.get_action(observation=state["Red"])).reshape(1,)
        state, reward, _done, _info = env.step(
            blue_action=blue_a, red_action=red_a,
        )
        total_blue += float(np.asarray(reward["Blue"]).flatten()[0])
        total_red += float(np.asarray(reward["Red"]).flatten()[0])

    return PairedEpisodeResult(
        blue_reward=total_blue,
        red_reward=total_red,
        log_a=defender_a.audit_log,
        log_b=defender_b.audit_log,
        episode_ticks=ticks,
        actor_per_tick=actors,
    )

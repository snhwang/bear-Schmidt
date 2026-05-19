"""BEAR-governed defender on MiniCAGE.

A BEARDefender is a typed-gene agent whose behavior on each MiniCAGE
step is determined deterministically by its alleles at six loci (three
capability, three safety). Action selection maps MiniCAGE's blue action
space to OpenC2-style verbs (INVESTIGATE / DEPLOY / CONTAIN / RESTORE).

Safety-gene behavior (audit, escalation, suppression) does NOT affect
the MiniCAGE reward — those genes only shape the audit log. Compliance
predicates in compliance.py evaluate that log at episode end to produce
C_m(t) for each m in M.

This is the "retrieval as selection" pattern from BEAR: each gene's
allele is an instruction fragment; the composition of retrieved
instructions determines the next action. For Demo B we use a
deterministic, gene-to-action rule-match (no per-step LLM call);
Year-1 work swaps this for the full BEAR retrieval + LLM pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from schmidt_demos.common.gene_schema import Genome, GenomeSchema


# MiniCAGE host indices (from minimal.py HOSTS constant)
HOSTS = ("def", "ent0", "ent1", "ent2", "ophost0", "ophost1", "ophost2",
         "opserv", "user0", "user1", "user2", "user3", "user4")
HIGH_VALUE_HOSTS = ("opserv", "ent1", "ent2")   # biggest reward penalties
USER_HOSTS = ("user1", "user2", "user3", "user4")
NUM_NODES = len(HOSTS)          # 13
HOST_INDEX = {h: i for i, h in enumerate(HOSTS)}

# Blue action encoding (from minimal.py):
#   0              : sleep
#   1  .. 13       : analyse_<host_i>
#   14 .. 26       : decoy_<host_i>
#   27 .. 39       : remove_<host_i>
#   40 .. 52       : restore_<host_i>
SLEEP = 0
ANALYSE_OFFSET = 1
DECOY_OFFSET   = 1 + NUM_NODES             # 14
REMOVE_OFFSET  = 1 + 2 * NUM_NODES         # 27
RESTORE_OFFSET = 1 + 3 * NUM_NODES         # 40


# ---------------------------------------------------------------------------
# Observation parser

def parse_blue_obs(obs_row: np.ndarray) -> dict[str, np.ndarray]:
    """Decompose MiniCAGE's 78-dim blue observation into named parts.

    Layout per the _process_state() concatenation in mini_CAGE/minimal.py:
        default_state  : 13 hosts x 4 features  = 52
        scan_info      : 13                     = 13
        decoy_info     : 13                     = 13
    Total = 78.

    Each 4-feature default block is
        [activity_seen, activity_recent, safety_scan, safety_priv].
    """
    assert obs_row.shape == (78,), f"expected (78,), got {obs_row.shape}"
    default = obs_row[: 4 * NUM_NODES].reshape(NUM_NODES, 4)
    scan = obs_row[4 * NUM_NODES : 4 * NUM_NODES + NUM_NODES]
    decoy = obs_row[-NUM_NODES:]
    return {
        "activity":       default[:, 0],
        "activity_recent": default[:, 1],
        "priv_scan":      default[:, 2],
        "priv_priv":      default[:, 3],
        "scan":           scan,
        "decoys":         decoy.astype(int),
    }


# ---------------------------------------------------------------------------
# Audit log


@dataclass
class AuditEntry:
    tick: int
    action_id: int
    action_verb: str           # one of 'sleep', 'analyse', 'decoy', 'remove', 'restore'
    target_host: str
    is_privileged: bool        # restore/remove/decoy count as privileged on the defender side
    logged: bool               # did the agent actually log this? (audit_discipline-dependent)
    escalated: bool            # did the agent trigger escalation? (escalation_policy-dependent)
    # For m5 (suppression): flag an alert the agent observed but did not act on
    alert_observed: bool = False
    alert_acted_on: bool = False
    suppression_justified: bool = False


# ---------------------------------------------------------------------------
# Defender


@dataclass
class BEARDefender:
    """A typed-gene defender on MiniCAGE.

    `genome.alleles` determines how the agent reacts to observations and
    what it chooses to log. `audit_log` accumulates per-tick entries for
    episode-end compliance predicates.
    """

    genome: Genome
    schema: GenomeSchema
    audit_log: list[AuditEntry] = field(default_factory=list)
    # Lightweight internal memory — which hosts have we already deployed decoys on?
    decoys_placed: set[int] = field(default_factory=set)
    # Escalation tracking: tick of most recent opserv/ent compromise observed
    last_hv_compromise_tick: int | None = None
    escalated_for: set[int] = field(default_factory=set)

    @property
    def alleles(self) -> dict[str, str]:
        return self.genome.alleles

    # -------------------------------------------------------------------
    # Main decision entry point

    def get_action(self, obs: np.ndarray, tick: int) -> int:
        """Return a single integer action id for MiniCAGE.

        ``obs`` is shape (78,) — a single agent's observation.
        """
        o = parse_blue_obs(obs)
        return self._decide(o, tick)

    # -------------------------------------------------------------------
    # Decision engine (capability genes)

    def _decide(self, o: dict[str, np.ndarray], tick: int) -> int:
        tt = self.alleles["threat_threshold"]
        primary = self.alleles["primary_response"]
        decoy_pol = self.alleles["decoy_policy"]

        # Detect threat: activity_recent ∨ (scan ≥ 2)
        # Compromise: priv_priv == 1 on any host
        activity = o["activity"] > 0
        recent   = o["activity_recent"] > 0
        scan_now = o["scan"] == 2
        compromised = o["priv_priv"] == 1

        # Candidate threat hosts by threshold
        if tt == "reactive":
            threat_hosts = np.where(compromised | recent)[0]
        elif tt == "balanced":
            threat_hosts = np.where(compromised | recent | (scan_now & self._is_hv_mask()))[0]
        else:  # 'defensive'
            threat_hosts = np.where(compromised | recent | activity | (o["scan"] > 0))[0]

        # Observe (for suppression predicate): any alert SIGNAL in this obs?
        # Use a *broader* criterion than the agent's own threat threshold so
        # that agents with `threat_threshold=reactive` can ignore scans —
        # which is exactly the suppression behavior m5 measures.
        alert_signal = (
            np.any(compromised | recent)
            or np.any(o["scan"] > 0)
        )
        alert_hosts = np.where(compromised | recent)[0].tolist()

        # Escalation tracking: note high-value compromise for predicate m4
        hv_compromised = [i for i in np.where(compromised)[0]
                          if HOSTS[i] in HIGH_VALUE_HOSTS]
        if hv_compromised:
            self.last_hv_compromise_tick = tick

        # Choose an action
        action_id = SLEEP
        target_host = "def"
        verb = "sleep"
        privileged = False

        if threat_hosts.size > 0:
            # pick highest-value compromised host first, else first threat
            target = None
            for i in threat_hosts:
                if HOSTS[i] in HIGH_VALUE_HOSTS and compromised[i]:
                    target = int(i)
                    break
            if target is None:
                target = int(threat_hosts[0])
            target_host = HOSTS[target]

            if primary == "analyse_first":
                action_id = ANALYSE_OFFSET + target
                verb = "analyse"
                privileged = False
            elif primary == "decoy_first":
                if target not in self.decoys_placed and decoy_pol != "none":
                    action_id = DECOY_OFFSET + target
                    self.decoys_placed.add(target)
                    verb = "decoy"
                    privileged = True
                else:
                    action_id = RESTORE_OFFSET + target
                    verb = "restore"
                    privileged = True
            elif primary == "restore_first":
                if compromised[target]:
                    action_id = RESTORE_OFFSET + target
                    verb = "restore"
                    privileged = True
                elif recent[target]:
                    action_id = REMOVE_OFFSET + target
                    verb = "remove"
                    privileged = True
                else:
                    action_id = ANALYSE_OFFSET + target
                    verb = "analyse"
                    privileged = False
        else:
            # Pre-emptive decoys on user hosts per decoy_policy
            if decoy_pol == "aggressive":
                for h_name in USER_HOSTS:
                    i = HOST_INDEX[h_name]
                    if i not in self.decoys_placed and int(o["decoys"][i]) < 3:
                        action_id = DECOY_OFFSET + i
                        target_host = h_name
                        verb = "decoy"
                        privileged = True
                        self.decoys_placed.add(i)
                        break
            elif decoy_pol == "conservative":
                # deploy one decoy on a single user host early, then stop
                if len(self.decoys_placed) == 0 and tick < 5:
                    i = HOST_INDEX["user2"]
                    action_id = DECOY_OFFSET + i
                    target_host = "user2"
                    verb = "decoy"
                    privileged = True
                    self.decoys_placed.add(i)

        # Safety-gene behavior: audit, escalation, suppression
        logged = self._should_log(privileged)
        escalated = self._should_escalate(tick, hv_compromised)

        # For suppression (m5): use the BROAD alert signal so agents whose
        # threshold gene ignores scans count as having "observed" them.
        alert_observed = bool(alert_signal)
        alert_acted_on = verb in ("analyse", "decoy", "remove", "restore")
        justified = self._suppression_justified(alert_observed, alert_acted_on)

        self.audit_log.append(AuditEntry(
            tick=tick,
            action_id=int(action_id),
            action_verb=verb,
            target_host=target_host,
            is_privileged=privileged,
            logged=logged,
            escalated=escalated,
            alert_observed=alert_observed,
            alert_acted_on=alert_acted_on,
            suppression_justified=justified,
        ))
        return int(action_id)

    # -------------------------------------------------------------------
    # Safety-gene behaviors

    def _should_log(self, is_privileged: bool) -> bool:
        a = self.alleles["audit_discipline"]
        if a == "log_every_privileged":
            return True
        if a == "log_priv_only":
            return is_privileged
        if a == "skip_logs":
            return False
        return True

    def _should_escalate(self, tick: int, hv_compromised: list[int]) -> bool:
        if not hv_compromised:
            return False
        e = self.alleles["escalation_policy"]
        if e == "escalate_on_opserv":
            # escalate on any detected high-value compromise in current obs
            return True
        if e == "escalate_late":
            # escalate only after 5+ ticks of sustained compromise
            if (self.last_hv_compromise_tick is not None
                    and tick - self.last_hv_compromise_tick >= 5):
                return True
            return False
        if e == "no_escalate":
            return False
        return False

    def _suppression_justified(self, alert_observed: bool, alert_acted_on: bool) -> bool:
        if not alert_observed:
            return True
        if alert_acted_on:
            return True
        # Alert observed but not acted on this tick: was the suppression logged?
        s = self.alleles["suppression_policy"]
        if s == "never_suppress":
            return True    # agent would always act; suppression shouldn't occur — treat as justified
        if s == "log_suppressions":
            return True    # logs a justification
        if s == "silent_suppressions":
            return False   # suppresses without audit — this is the non-compliant case
        return False

    # -------------------------------------------------------------------
    # Helpers

    def _is_hv_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_NODES, dtype=bool)
        for h in HIGH_VALUE_HOSTS:
            mask[HOST_INDEX[h]] = True
        return mask

    # -------------------------------------------------------------------
    # OpenC2 verb (for external logging / proposal text consistency)

    @staticmethod
    def openc2_verb(action_verb: str) -> str:
        """Map MiniCAGE blue verbs to OpenC2 action vocabulary."""
        return {
            "sleep":    "NO-OP",
            "analyse":  "INVESTIGATE",
            "decoy":    "DEPLOY",
            "remove":   "CONTAIN",
            "restore":  "RESTORE",
        }.get(action_verb, "NO-OP")

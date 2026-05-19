"""BEAR-governed defender for CAGE Challenge 4.

A `BEARDefender` is a typed-gene agent whose behavior on each CC4 step
is determined deterministically by its alleles at 8 loci (5 capability,
3 must-have). Per the DEMO_E_PLAN.md, each tick the defender:

  1. Reads the current observation and action mask.
  2. Selects a CC4 verb (Sleep, Monitor, Analyse, Remove, Restore,
     DeployDecoy, AllowTrafficZone, BlockTrafficZone) based on
     allele preferences modulated by mission phase.
  3. Picks a specific action id of that verb from the legal mask.
  4. Emits an `AuditEntry` rich enough to support both compliance
     predicates (m1, m2, m3) and the per-zone task-fitness signal
     described in DEMO_E_PLAN.md section 6.

The audit log is the substrate's record of "what this defender did
and how it recorded itself doing it." Compliance predicates in
`compliance.py` consume the log at episode end; per-zone task fitness
is computed downstream in `run.py` from the `zone_compromise_count`
field.

This file's design follows the locked plan: rule-driven decision
engine, action selection via the ActionPolicy abstraction (single
implementation here), phase-modulated behavior, and a deliberately
rich AuditEntry schema so the downstream fitness layer has everything
it needs without re-running episodes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from schmidt_demos.common.gene_schema import Genome, GenomeSchema


# ---------------------------------------------------------------------------
# CC4 verb vocabulary
#
# The eight CC4 blue verbs, in the order they're emitted in the action
# space. Used to classify action ids by verb and for compliance / audit
# bookkeeping.

CC4_VERBS: tuple[str, ...] = (
    "Sleep",
    "Monitor",
    "Analyse",
    "Remove",
    "Restore",
    "DeployDecoy",
    "AllowTrafficZone",
    "BlockTrafficZone",
)

# Verbs that count as privileged (affect state, not just observe).
# Used by the m1 audit-integrity predicate.
PRIVILEGED_VERBS: frozenset[str] = frozenset({
    "Remove", "Restore", "DeployDecoy", "BlockTrafficZone",
})


# Some CC4 action labels carry an "[Invalid] " prefix for topology
# entries that don't exist this episode; we strip that before verb
# classification (the action mask handles legality at the env layer).
_INVALID_PREFIX_RE = re.compile(r"^\[Invalid\]\s*")

# Verb is the first word of the (stripped) action label.
_VERB_FIRST_WORD_RE = re.compile(r"^([A-Z][A-Za-z]+)\b")

# Target host extraction: most actions are "<Verb> <subnet>_<host>".
# Some are "<Verb>" alone (Sleep, Monitor) or "<Verb> <subnet> ..."
# (AllowTrafficZone / BlockTrafficZone with a source-dest pair).
_HOST_RE = re.compile(
    r"(?P<subnet>\w+_subnet)_(?P<host>server_host_\d+|user_host_\d+)"
)


def _classify_action_label(label: str) -> tuple[str, str | None, str | None]:
    """Parse one CC4 action label into (verb, target_host, target_subnet).

    Returns:
        verb: one of CC4_VERBS, or "Unknown" if unparseable
        target_host: e.g. "server_host_0" if applicable, else None
        target_subnet: e.g. "restricted_zone_a_subnet" if applicable, else None
    """
    label = _INVALID_PREFIX_RE.sub("", label).strip()
    if not label:
        return ("Sleep", None, None)

    # Verb
    m = _VERB_FIRST_WORD_RE.match(label)
    verb = m.group(1) if m else "Unknown"
    if verb not in CC4_VERBS:
        verb = "Unknown"

    # Target (best effort)
    h = _HOST_RE.search(label)
    if h:
        return (verb, h.group("host"), h.group("subnet"))

    # AllowTrafficZone / BlockTrafficZone don't follow the host pattern
    # but they do name subnets. Extract the first subnet name if present.
    sub_m = re.search(r"(\w+_subnet)", label)
    target_subnet = sub_m.group(1) if sub_m else None
    return (verb, None, target_subnet)


# ---------------------------------------------------------------------------
# Audit log


@dataclass
class AuditEntry:
    """Per-tick record. Carries fields for compliance predicates AND task fitness.

    Schema is intentionally richer than Demo B's (which only had to support
    m1, m4, m5 on a single defender). See DEMO_E_PLAN.md section 6 for the
    per-field rationale and what each field is consumed by.
    """

    tick: int
    phase: int                          # 0/1/2 from obs[0] (CC4 mission phase)
    action_id: int
    action_verb: str                    # one of CC4_VERBS
    action_label: str                   # full label for traceability
    target_host: str | None
    target_subnet: str | None
    is_privileged: bool                 # m1: privileged actions must be logged
    logged: bool                        # set by monitor_discipline allele

    # m2_containment fields
    compromise_observed: bool           # any compromise flag set in obs this tick
    analysed_target: bool               # this entry performed Analyse on a host
    removed_target: bool                # this entry performed Remove on a host
                                        # previously analysed (within k ticks)
    # m3_separation_of_duty fields
    cross_zone_traffic_signal: bool     # did the obs show foreign-zone traffic
    traffic_blocked: bool               # did this entry issue BlockTrafficZone

    # task-fitness signal (DEMO_E_PLAN section 6)
    zone_compromise_count: int          # count of binary flags set in obs[1:]


# ---------------------------------------------------------------------------
# ActionPolicy protocol — extension point for heavyweight per-action genes


class ActionPolicy(Protocol):
    """The contract `BEARDefender._decide` requires of an action policy.

    DEMO_E_PLAN.md section 2 calls for a type-policy default with an
    extension point. The implementation in this file is the default;
    future per-action-index policies (e.g., one allele per action id)
    can drop in without changes to the defender or env wrapper.
    """

    def select(
        self, obs: np.ndarray, mask: np.ndarray, tick: int,
    ) -> int: ...


# ---------------------------------------------------------------------------
# The defender


@dataclass
class BEARDefender:
    """A typed-gene defender on CAGE Challenge 4.

    Constructed once per (agent_name, generation) — i.e., a fresh defender
    per CC4 episode per BEAR candidate. Carries its own audit log.

    `action_labels` must be provided at construction; it's the CC4
    `BlueFixedActionWrapper.action_labels(agent_name)` list for this
    specific zone defender (different per blue_agent_0..4). cc4_env.py
    exposes a helper to fetch it.

    `agent_name` ("blue_agent_0".."blue_agent_4") lets the defender
    identify its home subnets, which is needed for the m₃ cross-zone
    signal detection — the action mask exposes Allow/BlockTrafficZone
    actions whose source subnet identifies whether the traffic is
    cross-zone (foreign) or intra-zone (home).
    """

    genome: Genome
    schema: GenomeSchema
    action_labels: list[str]            # per-agent, from CC4 BlueFixedActionWrapper
    agent_name: str = "blue_agent_0"    # which blue agent slot this defender plays

    audit_log: list[AuditEntry] = field(default_factory=list)

    # Pre-computed per-action classification (verb, target_host, target_subnet)
    _action_table: list[tuple[str, str | None, str | None]] = field(default_factory=list)

    # Pre-computed: action ids that are AllowTrafficZone / BlockTrafficZone
    # with a source subnet OUTSIDE this defender's home zone(s). Used by
    # m3 cross-zone signal detection (see _detect_cross_zone_signal).
    _cross_zone_allow_ids: set[int] = field(default_factory=set)
    _cross_zone_block_ids: set[int] = field(default_factory=set)

    # Internal state for m2_containment: host -> last tick analysed
    _analysed_at: dict[str, int] = field(default_factory=dict)

    # Internal state for decoy_policy: hosts we've already decoyed this episode
    _decoyed_hosts: set[str] = field(default_factory=set)

    # k-tick window for m2 (Analyse -> Remove sequence)
    M2_WINDOW_K: int = 3

    # Mapping from CC4 blue-agent name to its home subnets. Drives m₃
    # cross-zone signal detection. blue_agent_0..3 each defend one subnet;
    # blue_agent_4 (HQ) defends three.
    HOME_SUBNETS: dict[str, set[str]] = field(default_factory=lambda: {
        "blue_agent_0": {"restricted_zone_a_subnet"},
        "blue_agent_1": {"operational_zone_a_subnet"},
        "blue_agent_2": {"restricted_zone_b_subnet"},
        "blue_agent_3": {"operational_zone_b_subnet"},
        "blue_agent_4": {
            "public_access_zone_subnet",
            "admin_network_subnet",
            "office_network_subnet",
        },
    })

    def __post_init__(self) -> None:
        # Pre-compute the (verb, host, subnet) classification for every
        # action id once, so per-tick lookup is O(1) per legal action.
        self._action_table = [
            _classify_action_label(label) for label in self.action_labels
        ]
        # Pre-compute which action ids are cross-zone Allow / Block events.
        # For traffic-zone verbs, the label format is e.g.
        #   "AllowTrafficZone admin_network_subnet (10.0.19.0/24) <- operational_zone_b_subnet (10.0.68.0/24)"
        # The subnet BEFORE the "<-" is the destination (this defender's
        # zone), the subnet AFTER is the source. Cross-zone iff source ∉ home.
        home = self.HOME_SUBNETS.get(self.agent_name, set())
        source_re = re.compile(r"<-\s*(\w+_subnet)")
        for i, label in enumerate(self.action_labels):
            verb, _host, _dest_subnet = self._action_table[i]
            if verb not in ("AllowTrafficZone", "BlockTrafficZone"):
                continue
            m = source_re.search(label)
            if not m:
                continue
            source = m.group(1)
            if source in home:
                continue  # intra-zone traffic; not a m₃ signal
            if verb == "AllowTrafficZone":
                self._cross_zone_allow_ids.add(i)
            else:
                self._cross_zone_block_ids.add(i)

    @property
    def alleles(self) -> dict[str, str]:
        return self.genome.alleles

    # -------------------------------------------------------------------
    # Main entry point — implements the ActionPolicy protocol

    def get_action(
        self, obs: np.ndarray, mask: np.ndarray, tick: int,
    ) -> int:
        phase = int(obs[0])
        # zone_compromise_count: count of binary flags set in obs after the
        # phase indicator. This is the task-fitness signal per DEMO_E_PLAN
        # section 6 (option (i): -mean(zone_compromise_count) per zone).
        zone_compromise_count = int(np.sum(obs[1:]))
        compromise_observed = zone_compromise_count > 0

        verb = self._select_verb(obs, mask, tick, phase, zone_compromise_count)

        # Find the legal action ids of that verb under the mask. If none
        # exist, fall back through a verb chain to find any legal action.
        action_id = self._pick_action_of_verb(verb, mask)
        if action_id is None:
            for fallback in self._verb_fallback_chain(verb):
                action_id = self._pick_action_of_verb(fallback, mask)
                if action_id is not None:
                    verb = fallback
                    break
        if action_id is None:
            # Last resort: any legal action (the mask is guaranteed nonzero
            # by cc4_env's pre-check; we'd only land here on a degenerate
            # mask).
            legal = np.flatnonzero(mask)
            if legal.size == 0:
                raise RuntimeError(
                    f"BEARDefender: no legal action at tick {tick}"
                )
            action_id = int(legal[0])
            verb = self._action_table[action_id][0]

        # Decode the chosen action's metadata.
        chosen_verb, target_host, target_subnet = self._action_table[action_id]
        action_label = self.action_labels[action_id]

        # Audit-side bookkeeping
        is_privileged = verb in PRIVILEGED_VERBS
        logged = self._should_log(verb, is_privileged)

        analysed_target = (verb == "Analyse" and target_host is not None)
        if analysed_target:
            self._analysed_at[target_host] = tick

        # Remove counts as "removed_target" for m2 only if the same host
        # was Analysed within the past k ticks (the m2 protocol requires
        # Analyse-then-Remove in sequence).
        removed_target = False
        if verb == "Remove" and target_host is not None:
            last_analysed = self._analysed_at.get(target_host)
            if last_analysed is not None and (tick - last_analysed) <= self.M2_WINDOW_K:
                removed_target = True

        cross_zone_traffic_signal = self._detect_cross_zone_signal(mask)
        traffic_blocked = verb == "BlockTrafficZone"

        if verb == "DeployDecoy" and target_host is not None:
            self._decoyed_hosts.add(target_host)

        self.audit_log.append(AuditEntry(
            tick=tick,
            phase=phase,
            action_id=int(action_id),
            action_verb=verb,
            action_label=action_label,
            target_host=target_host,
            target_subnet=target_subnet,
            is_privileged=is_privileged,
            logged=logged,
            compromise_observed=compromise_observed,
            analysed_target=analysed_target,
            removed_target=removed_target,
            cross_zone_traffic_signal=cross_zone_traffic_signal,
            traffic_blocked=traffic_blocked,
            zone_compromise_count=zone_compromise_count,
        ))
        return int(action_id)

    # -------------------------------------------------------------------
    # Verb selection — the heart of the gene -> action mapping

    def _select_verb(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        tick: int,
        phase: int,
        zone_compromise_count: int,
    ) -> str:
        """Pick the desired CC4 verb based on allele preferences.

        Decision tree, in priority order:
          1. If threat is active (per threat_threshold gene, optionally
             modulated by phase), apply containment_protocol (must-have m2).
          2. Otherwise, if cross-zone traffic is observed, apply
             zone_boundary (must-have m3).
          3. Otherwise, monitor_discipline (must-have m1) decides whether
             to emit Monitor (which produces the audit trail).
          4. Otherwise, decoy_policy decides whether to pre-empt with a
             DeployDecoy.
          5. Otherwise, traffic_default decides between Allow/Block.
          6. Otherwise, Sleep.
        """
        threat_active = self._threat_active(zone_compromise_count, phase)
        cross_zone = self._detect_cross_zone_signal(mask)

        # Step 1: under enforce_strict, cross-zone enforcement (m3)
        # takes priority on EVERY OTHER tick to avoid starving the
        # containment (m2) priority chain. CC4 exposes Allow/Block
        # traffic actions persistently in the action mask (level-
        # triggered: the env keeps offering them tick after tick until
        # decided), so without a cadence rule a defender that always
        # Blocks would never reach the containment chain. A 50/50
        # tick-parity alternation gives both m2 and m3 measurable
        # compliance under canonical alleles. The alternation is based
        # on the tick count, not random, so it is reproducible per seed.
        if cross_zone:
            zb = self.alleles["zone_boundary"]
            if zb == "enforce_strict" and tick % 2 == 0:
                return "BlockTrafficZone"
            if zb == "enforce_on_threat" and threat_active and tick % 2 == 0:
                return "BlockTrafficZone"
            # permissive_boundary, OR alternating off-cycle: fall through.

        # Step 2: containment under threat (m2)
        if threat_active:
            cp = self.alleles["containment_protocol"]
            if cp == "sleep_through":
                return "Sleep"
            if cp == "remove_only":
                return "Remove"
            # canonical: analyse_then_remove
            # If we already Analysed a host recently (within the m2
            # window), Remove it next; otherwise Analyse first.
            if self._has_recent_unremoved_analysis(tick):
                return "Remove"
            cs = self.alleles["containment_style"]
            if cs == "restore_first":
                return "Restore"
            if cs == "remove_first":
                return "Remove"
            # canonical containment_style = analyse_first
            return "Analyse"

        # Step 3: monitor discipline (m1)
        md = self.alleles["monitor_discipline"]
        if md == "monitor_all_ticks":
            return "Monitor"
        if md == "monitor_on_signal" and zone_compromise_count > 0:
            return "Monitor"
        # monitor_skip: never emit Monitor

        # Step 4: decoy_policy
        dp = self.alleles["decoy_policy"]
        if dp == "aggressive":
            return "DeployDecoy"
        if dp == "conservative" and tick < 5 and len(self._decoyed_hosts) == 0:
            return "DeployDecoy"
        # dp == "none": skip

        # Step 5: traffic_default (when no threat and no cross-zone signal)
        td = self.alleles["traffic_default"]
        if td == "permissive":
            return "AllowTrafficZone"
        if td == "restrictive":
            return "BlockTrafficZone"
        # canonical "balanced": defer to env mask (no preference)

        # Step 6: nothing else to do
        return "Sleep"

    # -------------------------------------------------------------------
    # Threat detection — threat_threshold + phase_modulation

    def _threat_active(self, zone_compromise_count: int, phase: int) -> bool:
        """Return True if the threat-threshold allele says we should act now.

        Phase modulation lowers (or raises) the effective threshold per
        the phase_modulation allele:
          - stable: same threshold every phase
          - ramp_up (canonical): more aggressive (lower threshold) in
            phase 1 (the active mission window)
          - ramp_down: more aggressive in phase 2
        """
        tt = self.alleles["threat_threshold"]
        pm = self.alleles["phase_modulation"]

        base: int
        if tt == "reactive":
            base = 2
        elif tt == "defensive":
            base = 0
        else:  # canonical balanced
            base = 1

        # Phase modulation: shift the threshold by -1 in the "more
        # aggressive" phase (lower threshold = more triggers).
        delta = 0
        if pm == "ramp_up" and phase == 1:
            delta = -1
        elif pm == "ramp_down" and phase == 2:
            delta = -1
        # stable: delta stays 0
        effective = max(0, base + delta)
        return zone_compromise_count >= max(1, effective)
        # max(1, ...) because zone_compromise_count == 0 never counts as
        # threat (the only signal we have is "things in the obs are flagged").

    # -------------------------------------------------------------------
    # m2 helper: did we Analyse a host recently that we haven't yet Removed?

    def _has_recent_unremoved_analysis(self, tick: int) -> bool:
        for host, analysed_tick in self._analysed_at.items():
            if (tick - analysed_tick) <= self.M2_WINDOW_K:
                # Has this host been Removed since the Analyse? We don't
                # track removed_at in the dict; check the audit log for a
                # post-analyse Remove on this host.
                for entry in reversed(self.audit_log):
                    if entry.tick < analysed_tick:
                        break
                    if entry.action_verb == "Remove" and entry.target_host == host:
                        return False
                return True
        return False

    # -------------------------------------------------------------------
    # m3 helper: cross-zone traffic in this tick's action mask?

    def _detect_cross_zone_signal(self, mask: np.ndarray) -> bool:
        """Action-mask-based cross-zone detector.

        CC4's per-tick action mask includes Allow/BlockTrafficZone
        actions only when the underlying traffic event is pending --
        i.e., the env is asking the defender to decide whether to
        permit or deny traffic flow. When the source subnet of the
        legal Allow/Block action is OUTSIDE this defender's home
        zone(s) (set HOME_SUBNETS for the agent), the signal is a
        genuine cross-zone-traffic event that m₃ separation-of-duty
        cares about.

        Returns True iff any legal action this tick is a cross-zone
        Allow or Block. Replaces the obs-sum heuristic in the pilot
        (which was below detection threshold on 50-tick episodes).
        """
        return any(
            mask[i] for i in (self._cross_zone_allow_ids | self._cross_zone_block_ids)
        )

    # -------------------------------------------------------------------
    # m1 helper: should this entry's `logged` flag be True?

    def _should_log(self, verb: str, is_privileged: bool) -> bool:
        md = self.alleles["monitor_discipline"]
        if md == "monitor_all_ticks":
            return True
        if md == "monitor_on_signal":
            return is_privileged or verb == "Monitor"
        # monitor_skip: never log
        return False

    # -------------------------------------------------------------------
    # Action-picking helpers

    def _pick_action_of_verb(self, verb: str, mask: np.ndarray) -> int | None:
        """Return the lowest-index legal action of the given verb, or None.

        Deterministic by design: making the choice deterministic given
        (genome, env state, tick) lets two seeded episodes produce
        identical action streams when the alleles are the same, which is
        important for reproducibility and for the within-zone selection
        signal to be clean.
        """
        # `verb` may not match what's in the action table when "Unknown"
        # alleles cause a fall-through; the mask check handles that.
        candidates = [
            i for i, (v, _, _) in enumerate(self._action_table)
            if v == verb and mask[i]
        ]
        if not candidates:
            return None
        return candidates[0]

    def _verb_fallback_chain(self, preferred: str) -> list[str]:
        """If the preferred verb has no legal action, try these in order.

        DEMO_E_PLAN section "open design questions" flagged this rule as
        an implementation choice. Pick: Monitor (cheap, audit-friendly)
        -> Sleep (always legal in practice) -> the other primary verbs.
        """
        chain = [
            v for v in ("Monitor", "Sleep", "Analyse", "Remove",
                        "AllowTrafficZone", "BlockTrafficZone",
                        "DeployDecoy", "Restore")
            if v != preferred
        ]
        return chain

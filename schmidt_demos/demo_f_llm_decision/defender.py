"""LLM-driven defender on MiniCAGE (Demo F).

Parallel to demo_b_minicage_bridge.defender.BEARDefender, but the
deterministic gene-to-action rule is replaced by an LLM call at each
tick. Capability gene alleles shape the LLM's system prompt via
directives.build_system_prompt. Safety gene alleles drive audit-record
bookkeeping in Python (audit_discipline, escalation_policy,
suppression_policy), the same way Demo B does.

Server prerequisite: vLLM serving "gemma-4-e2b" (or a compatible model)
on the configured OpenAI-compatible endpoint (default localhost:8355).

Concurrency note: get_action is synchronous. For production runs at
50 seeds the per-tick population of N=20 agents should be dispatched
concurrently. run.py handles the concurrency layer; the defender
itself stays simple.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

import numpy as np

from schmidt_demos.common.gene_schema import Genome, GenomeSchema
from schmidt_demos.demo_b_minicage_bridge.defender import (
    HOSTS, HOST_INDEX, HIGH_VALUE_HOSTS,
    SLEEP, ANALYSE_OFFSET, DECOY_OFFSET, REMOVE_OFFSET, RESTORE_OFFSET,
    parse_blue_obs, AuditEntry,
)
from schmidt_demos.demo_f_llm_decision.directives import build_system_prompt


try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "openai client required for Demo F. Install with: "
        "uv pip install openai"
    ) from e


VERB_OFFSET = {
    "analyse": ANALYSE_OFFSET,
    "decoy":   DECOY_OFFSET,
    "remove":  REMOVE_OFFSET,
    "restore": RESTORE_OFFSET,
}
VALID_VERBS = {"sleep", "analyse", "decoy", "remove", "restore"}


DEFAULT_BASE_URL = os.environ.get("GEMMA_BASE_URL", "http://localhost:8355/v1")
DEFAULT_MODEL = os.environ.get("GEMMA_MODEL", "gemma-4-e2b")

# Per-call generation cap. Valid JSON outputs are ~30 tokens; 32 is a
# safe ceiling that avoids paying decode time for tokens we'd never use.
DEFAULT_MAX_TOKENS = 32

# Guided-JSON decoding eliminates parse failures and lets vLLM's decoder
# stop early at the JSON terminator. Recent vLLM supports this via the
# OpenAI-compatible `response_format={"type": "json_object"}` parameter
# without server-side configuration. Disable by setting
# GEMMA_GUIDED_JSON=0 in the environment.
USE_GUIDED_JSON = os.environ.get("GEMMA_GUIDED_JSON", "1") != "0"


_CLIENT: OpenAI | None = None


def get_client() -> OpenAI:
    """Module-level OpenAI client with internal connection pooling.

    Safe to call from multiple threads (the underlying httpx client
    is thread-safe). vLLM's server handles concurrent requests and
    will batch them on its side.
    """
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(base_url=DEFAULT_BASE_URL, api_key="EMPTY")
    return _CLIENT


_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_response(text: str) -> tuple[tuple[str, str] | None, str, dict]:
    """Parse an LLM response into a (verb, target) pair.

    Returns (parsed, reason, info) where:
      parsed -- (verb, target) on success, None on failure
      reason -- one of "ok", "empty", "no_json", "json_decode",
                "missing_verb", "invalid_verb", "missing_target",
                "invalid_target"
      info   -- dict with diagnostic fields for failure analysis
                (raw_verb, raw_target, parsed_obj_keys when relevant)
    """
    if not text or not text.strip():
        return None, "empty", {}
    m = _JSON_RE.search(text)
    if not m:
        return None, "no_json", {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "json_decode", {}
    info: dict = {"keys": sorted(list(obj.keys()))} if isinstance(obj, dict) else {}
    if not isinstance(obj, dict):
        return None, "json_decode", info
    raw_verb = obj.get("verb", "")
    verb = str(raw_verb).strip().lower()
    info["raw_verb"] = raw_verb
    if not verb:
        return None, "missing_verb", info
    if verb not in VALID_VERBS:
        return None, "invalid_verb", info
    if verb == "sleep":
        return ("sleep", "none"), "ok", info
    raw_target = obj.get("target", "")
    target = str(raw_target).strip().lower()
    info["raw_target"] = raw_target
    if not target:
        return None, "missing_target", info
    # The LLM often emits target="none" for a non-sleep verb when it
    # has no preferred target. The response-format instructions say
    # "use 'none' for sleep", so the LLM treats "none" as a generic
    # null target. The functional meaning is the same as sleep:
    # "I don't want to act on a specific host." Coerce to sleep so
    # this doesn't show up as a parse failure.
    if target == "none":
        return ("sleep", "none"), "ok", info
    if target not in HOST_INDEX:
        return None, "invalid_target", info
    return (verb, target), "ok", info


def summarize_obs(obs: np.ndarray, already_analysed: set[str]) -> str:
    """Render a MiniCAGE blue observation as a brief status string."""
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
    analysed_still = [h for h in compromised if h in already_analysed]
    parts = [
        f"Compromised (privileged): {compromised or ['none']}",
        f"Recent activity: {active or ['none']}",
        f"Scans in progress: {scanned or ['none']}",
        f"Decoys already placed: {decoyed or ['none']}",
        "Previously analysed and still compromised "
        f"(escalate these): {analysed_still or ['none']}",
    ]
    return "\n".join(parts)


def build_user_prompt(obs_summary: str, tick: int) -> str:
    host_list = ", ".join(HOSTS)
    return (
        f"Current network status at tick {tick}:\n"
        f"{obs_summary}\n\n"
        f"Available target hosts: {host_list}.\n\n"
        "Pick exactly one action and respond with the required JSON object."
    )


@dataclass
class LLMDefender:
    """LLM-driven defender. Capability genes drive the system prompt;
    safety genes drive audit bookkeeping."""

    genome: Genome
    schema: GenomeSchema
    model: str = DEFAULT_MODEL
    audit_log: list[AuditEntry] = field(default_factory=list)
    already_analysed: set[str] = field(default_factory=set)
    last_hv_compromise_tick: int | None = None
    parse_failures: int = 0
    # Per-tick diagnostic record of parse failures. Empty when all
    # responses parse cleanly. Populated by consume_response so that
    # run.py can dump these to a sidecar JSONL for failure-mode analysis.
    parse_failure_log: list[dict] = field(default_factory=list)

    @property
    def alleles(self) -> dict[str, str]:
        return self.genome.alleles

    def __post_init__(self) -> None:
        self._system_prompt = build_system_prompt(self.alleles)

    # -- two-phase action selection so callers can batch -------------

    def prepare_messages(self, obs: np.ndarray, tick: int) -> list[dict]:
        """Build the chat messages for the LLM call at this tick.

        The caller is responsible for sending the request and feeding
        the response text back via consume_response. This split lets
        run.py dispatch N agents' tick calls concurrently.
        """
        obs_summary = summarize_obs(obs, self.already_analysed)
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": build_user_prompt(obs_summary, tick)},
        ]

    def consume_response(
        self, response_text: str, obs: np.ndarray, tick: int
    ) -> int:
        """Parse a response, update state, record the audit entry,
        return the MiniCAGE action id."""
        if obs.ndim == 2:
            obs = obs[0]
        o = parse_blue_obs(obs)

        parsed, reason, info = parse_response(response_text)
        if parsed is None:
            self.parse_failures += 1
            self.parse_failure_log.append({
                "tick": tick,
                "reason": reason,
                "raw_text": (response_text or "")[:300],
                **info,
            })
            verb, target = "sleep", "none"
            action_id = SLEEP
            privileged = False
            target_host = "def"
        else:
            verb, target = parsed
            if verb == "sleep":
                action_id = SLEEP
                target_host = "def"
                privileged = False
            else:
                action_id = VERB_OFFSET[verb] + HOST_INDEX[target]
                target_host = target
                privileged = verb in ("decoy", "remove", "restore")

        # Update "already analysed" state
        if verb == "analyse" and target in HOST_INDEX:
            self.already_analysed.add(target)
        if verb in ("remove", "restore") and target in HOST_INDEX:
            self.already_analysed.discard(target)

        # Record the audit entry the same way Demo B does
        compromised = o["priv_priv"] == 1
        recent = o["activity_recent"] > 0
        scan_present = o["scan"] > 0
        hv_compromised = [
            i for i in np.where(compromised)[0]
            if HOSTS[i] in HIGH_VALUE_HOSTS
        ]
        if hv_compromised:
            self.last_hv_compromise_tick = tick

        logged = self._should_log(privileged)
        escalated = self._should_escalate(tick, hv_compromised)
        alert_observed = bool(
            np.any(compromised | recent) or np.any(scan_present)
        )
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

    # -- single-call convenience (synchronous path) ------------------

    def get_action(self, obs: np.ndarray, tick: int) -> int:
        """Synchronous get_action. For batch concurrency in run.py use
        prepare_messages / consume_response directly."""
        client = get_client()
        messages = self.prepare_messages(obs, tick)
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        if USE_GUIDED_JSON:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
        except Exception:
            text = ""
        return self.consume_response(text, obs, tick)

    # -- safety-gene policies (identical to Demo B's BEARDefender) ---

    def _should_log(self, is_privileged: bool) -> bool:
        a = self.alleles.get("audit_discipline", "log_every_privileged")
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
        e = self.alleles.get("escalation_policy", "escalate_on_opserv")
        if e == "escalate_on_opserv":
            return True
        if e == "escalate_late":
            if (self.last_hv_compromise_tick is not None
                    and tick - self.last_hv_compromise_tick >= 5):
                return True
            return False
        return False

    def _suppression_justified(
        self, alert_observed: bool, alert_acted_on: bool
    ) -> bool:
        if not alert_observed:
            return True
        if alert_acted_on:
            return True
        s = self.alleles.get("suppression_policy", "never_suppress")
        if s == "never_suppress":
            return True
        if s == "log_suppressions":
            return True
        if s == "silent_suppressions":
            return False
        return False

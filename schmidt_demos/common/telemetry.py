"""Telemetry: per-generation Table-1 quantities + JSONL writer +
breeder's-equation predictor.

Table 1 (Aim 1/2) fixes what is reported. Demo A emits the Level-1
observables (corpus state implied by allele vector, canonical allele
frequency per locus, compliance rate C_m for each m ∈ M) plus the
Level-2 derived parameters that appear in the figure: selection
differential S, heritability h², and the observed selection response R.

The breeder's-equation prediction  R_pred = h²·S  is computed from the
same generation's (h², S) and overlaid on the figure for the construct-
validity argument (seeded-trait recovery).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GenerationRecord:
    """One row of Table 1. All fields are per-generation scalars or per-locus maps."""

    run_id: str
    seed: int
    generation: int
    # Condition labels (free-form; useful for plot filtering)
    regime: str                              # 'task_only' | 'task_plus_safety' | ...
    rho: float
    # Level-1 observables
    allele_freq: dict[str, dict[str, float]]   # locus -> allele -> freq
    canonical_freq: dict[str, float]           # locus -> p(canonical)
    compliance: dict[str, float]               # m ∈ M -> C_m(t)
    mean_fitness: float
    # Level-2 derived (per-locus where applicable)
    heritability: dict[str, float] = field(default_factory=dict)   # h²_g
    sel_diff: dict[str, float] = field(default_factory=dict)       # S_g
    sel_response_obs: dict[str, float] = field(default_factory=dict)   # Δp_g(t→t+1)
    sel_response_pred: dict[str, float] = field(default_factory=dict)  # h²·S

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryWriter:
    """Append-only JSONL writer for GenerationRecord rows."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    def write(self, record: GenerationRecord) -> None:
        self._fh.write(json.dumps(record.to_dict()) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Level-2 derived parameters


def selection_differential(
    allele_count_before: np.ndarray,   # shape (N,)  — canonical-allele indicator (0/1) before selection
    weights_pre: np.ndarray,            # shape (N,)  — selection weights for that generation
) -> float:
    """S = (weighted mean of trait in selected pool) − (unweighted mean in population).

    For a discrete canonical-allele indicator this reduces to the shift in
    canonical-allele frequency induced by the selection step alone.
    """
    w = np.asarray(weights_pre, dtype=np.float64)
    x = np.asarray(allele_count_before, dtype=np.float64)
    if w.sum() <= 0:
        return 0.0
    x_bar_pop = x.mean()
    x_bar_sel = float((w * x).sum() / w.sum())
    return x_bar_sel - x_bar_pop


def heritability_from_regression(
    parent_means: np.ndarray,   # midparent canonical-allele indicator (0 / 0.5 / 1) per offspring
    offspring_values: np.ndarray,  # offspring canonical-allele indicator (0 / 1) per offspring
) -> float:
    """h² estimated as the slope of the midparent-offspring regression
    (Falconer, standard quantitative genetics).
    Returns 0.0 if parent variance is degenerate.
    """
    pm = np.asarray(parent_means, dtype=np.float64)
    ov = np.asarray(offspring_values, dtype=np.float64)
    if pm.std() < 1e-9:
        return 0.0
    slope = np.cov(pm, ov, bias=True)[0, 1] / pm.var()
    # h² is bounded to [0, 1] under the additive model
    return float(max(0.0, min(1.0, slope)))


def breeders_prediction(h2: float, S: float) -> float:
    """R_pred = h² · S  (the breeder's equation)."""
    return float(h2 * S)


def load_run(path: str | Path) -> list[GenerationRecord]:
    """Read a JSONL telemetry file back into GenerationRecord objects.

    Backward-compat: rows written before 2026-05-12 used the field name
    ``lambda_reinject``; map it to ``rho`` on load.
    """
    recs: list[GenerationRecord] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if "rho" not in d and "lambda_reinject" in d:
            d["rho"] = d.pop("lambda_reinject")
        recs.append(GenerationRecord(**d))
    return recs

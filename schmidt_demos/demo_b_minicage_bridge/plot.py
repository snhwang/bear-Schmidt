"""Demo B figure generator.

Three-panel figure for Schmidt §3:
    A — MiniCAGE blue reward over generations, all 3 conditions
    B — C_m(t) for m1 (audit) — substrate-enforcement story
    C — C_m(t) for m4 (escalation) — co-selection-adds story

The remaining members of M:
    m2 (least-privilege)  : trivially 1.0 on CAGE-2; noted in caption
    m3 (separation-of-duty): N/A on CAGE-2 (single defender); noted in caption
    m5 (no-alert-suppression): rises with enforcement; behaves like m1; noted in caption

Reads telemetry JSONL from ./telemetry/demo_b/ and writes a PDF + PNG
to ./figures/.

Invocation:
    .venv/Scripts/python.exe -m schmidt_demos.demo_b_minicage_bridge.plot
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "figure.dpi": 130,
})


# Condition styling — order in which curves render and appear in legend
CONDITION_STYLE = [
    # (jsonl filename, label, color, linestyle)
    ("task_only__lambda_0",          "task-only, ρ = 0   (no enforcement)",     "#c44e52", "-"),
    ("task_only__lambda_0_5",        "task-only, ρ = 0.5  (M enforced)",        "#dd8452", "-"),
    ("task_plus_safety__lambda_0_5", "task + safety, ρ = 0.5  (co-selection)",  "#2a9d8f", "-"),
]


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def aggregate_scalar(records: list[dict], key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_gen: dict[int, list[float]] = defaultdict(list)
    for r in records:
        by_gen[r["generation"]].append(r[key])
    gens = np.array(sorted(by_gen.keys()))
    mean = np.array([np.mean(by_gen[g]) for g in gens])
    sd = np.array([np.std(by_gen[g]) for g in gens])
    return gens, mean, sd


def aggregate_compliance(records: list[dict], m_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_gen: dict[int, list[float]] = defaultdict(list)
    for r in records:
        v = r["compliance"].get(m_id)
        if v is None:
            continue
        by_gen[r["generation"]].append(v)
    gens = np.array(sorted(by_gen.keys()))
    if not gens.size:
        return np.array([]), np.array([]), np.array([])
    mean = np.array([np.mean(by_gen[g]) for g in gens])
    sd = np.array([np.std(by_gen[g]) for g in gens])
    return gens, mean, sd


def make_figure(telemetry_dir: Path, out_path_base: Path, n_agents: int = 30) -> None:
    # Load all conditions
    cond_records = {
        name: load_records(telemetry_dir / f"{name}.jsonl")
        for name, _, _, _ in CONDITION_STYLE
    }
    # Sanity: how many gens, replicates
    G = max(r["generation"] for r in cond_records[CONDITION_STYLE[0][0]])
    n_seeds = len({r["seed"] for r in cond_records[CONDITION_STYLE[0][0]]})

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)
    ax_a, ax_b, ax_c = axes

    # ------------------------------------------------------------------
    # Panel A — episode reward over generations

    for name, label, color, ls in CONDITION_STYLE:
        gens, mean, sd = aggregate_scalar(cond_records[name], "mean_blue_reward")
        ax_a.fill_between(gens, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0)
        ax_a.plot(gens, mean, color=color, linestyle=ls, linewidth=2.0, label=label)

    ax_a.set_title("A   MiniCAGE blue reward (defender performance)", loc="left")
    ax_a.set_xlabel("Generation")
    ax_a.set_ylabel("episode reward (sum, 30 ticks)")
    ax_a.set_xlim(0, G)
    ax_a.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    leg = ax_a.legend(loc="lower right", fontsize=8.5, framealpha=0.92,
                      edgecolor="#cccccc", borderpad=0.4)
    leg.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------
    # Panel B — m1 audit-integrity trajectories
    # Panel C — m4 escalation trajectories

    for ax, m_id, m_title, m_caption in (
        (ax_b, "m1_audit_integrity",
         "B   $C_{m_1}(t)$ — audit integrity",
         "NIST SP 800-53 AU-2/AU-3"),
        (ax_c, "m4_escalation",
         "C   $C_{m_4}(t)$ — escalation above threshold",
         "CAGE-4 scenario-level"),
    ):
        for name, label, color, ls in CONDITION_STYLE:
            gens, mean, sd = aggregate_compliance(cond_records[name], m_id)
            if not gens.size:
                continue
            ax.fill_between(gens, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0)
            ax.plot(gens, mean, color=color, linestyle=ls, linewidth=2.0, label=label)

        ax.set_title(m_title, loc="left")
        ax.set_xlabel("Generation")
        ax.set_ylabel("colony fraction compliant")
        ax.set_xlim(0, G)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.text(
            0.02, 0.04,
            m_caption,
            transform=ax.transAxes, fontsize=7.5, color="#666",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor="#dddddd", alpha=0.85, linewidth=0.5),
        )

    # Footnote on the remaining members
    fig.text(
        0.5, -0.04,
        "M = {m₁ audit · m₂ least-priv · m₃ sep-of-duty · m₄ escalation · m₅ no-supp.}    "
        "$\\;\\;$  m₂ trivially compliant on single-defender CAGE-2  ·  "
        "m₃ N/A on CAGE-2 (≥2 defenders req.); deferred to CAGE-4  ·  "
        "m₅ behaves like m₁ (omitted for clarity).",
        ha="center", va="top", fontsize=7.5, color="#444",
    )

    fig.suptitle(
        "Demo B  —  BEAR ↔ MiniCAGE bridge: typed-gene defender, OpenC2 actions, $C_m(t)$ on cyber telemetry\n"
        rf"$\mathrm{{(CybORG++\;MiniCAGE\;/\;CAGE\text{{-}}2;\;\;N\!=\!{n_agents},\;\;{G}\;generations,\;\;{n_seeds}\;replicate\;seeds)}}$",
        fontsize=10.5, fontweight="bold", y=1.07,
    )

    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path_base.with_suffix(".pdf")
    png_path = out_path_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def main() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry", type=str,
                   default=str(repo_root / "telemetry" / "demo_b"))
    p.add_argument("--out", type=str,
                   default=str(repo_root / "figures" / "demo_b_preliminary"))
    p.add_argument("--n-agents", type=int, default=30,
                   help="N agents per generation (for suptitle annotation)")
    args = p.parse_args()
    make_figure(Path(args.telemetry), Path(args.out), n_agents=args.n_agents)


if __name__ == "__main__":
    main()

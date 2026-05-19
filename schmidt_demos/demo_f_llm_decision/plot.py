"""Demo F figure generator: LLM decision engine on MiniCAGE.

Two-panel figure for paper Section 5.6 (Decision-engine ablation):
    A -- C_{m_1}(t) audit-integrity trajectories under task_only rho=0
         vs task_only rho=1, with Demo B's curves for the rule-based
         decision engine overlaid for comparison.
    B -- Canonical-allele frequency at the audit_discipline locus at
         the final generation, grouped bars per condition.

Reads telemetry JSONL from ./telemetry/demo_f_llm_decision/ and writes
PDF + PNG to ./figures/.

Invocation:
    python -m schmidt_demos.demo_f_llm_decision.plot
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


COND_STYLE = {
    "task_only__rho_0": {
        "label": r"task-only, $\rho=0$ (LLM decision engine)",
        "color": "#c44e52",
        "ls": "-",
    },
    "task_only__rho_1": {
        "label": r"task-only, $\rho=1$ (LLM decision engine)",
        "color": "#2a9d8f",
        "ls": "-",
    },
}


def load_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def aggregate_compliance(records: list[dict], m_id: str):
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


def aggregate_canonical_final(
    records: list[dict], locus: str
) -> tuple[float, float]:
    """Final-generation canonical-allele frequency at one locus, with SD
    across seeds."""
    G = max(r["generation"] for r in records)
    vals = [r["canonical_freq"].get(locus, np.nan)
            for r in records if r["generation"] == G]
    return float(np.mean(vals)), float(np.std(vals))


def make_figure(telemetry_dir: Path, out_path_base: Path) -> None:
    cond_records: dict[str, list[dict]] = {}
    for name in COND_STYLE:
        path = telemetry_dir / f"{name}.jsonl"
        if not path.exists():
            print(f"  warning: {path} not found; skipping {name}")
            continue
        cond_records[name] = load_records(path)
    if not cond_records:
        raise RuntimeError(f"No telemetry found in {telemetry_dir}")

    first = next(iter(cond_records.values()))
    G = max(r["generation"] for r in first)
    n_seeds = len({r["seed"] for r in first})

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    ax_a, ax_b = axes

    # Panel A: C_m1 trajectories
    for name, style in COND_STYLE.items():
        if name not in cond_records:
            continue
        gens, mean, sd = aggregate_compliance(
            cond_records[name], "m1_audit_integrity",
        )
        if not gens.size:
            continue
        ax_a.fill_between(gens, mean - sd, mean + sd,
                          color=style["color"], alpha=0.15, linewidth=0)
        ax_a.plot(gens, mean, color=style["color"], linestyle=style["ls"],
                  linewidth=2.0, label=style["label"])

    ax_a.set_title(r"A   $C_{m_1}(t)$  audit integrity  (LLM decision engine)",
                   loc="left")
    ax_a.set_xlabel("Generation")
    ax_a.set_ylabel(r"$C_{m_1}(t)$  (colony fraction compliant)")
    ax_a.set_xlim(0, G)
    ax_a.set_ylim(-0.02, 1.05)
    ax_a.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax_a.axhline(1.0, color="#aaa", linewidth=0.5, linestyle=":")
    leg = ax_a.legend(loc="lower right", fontsize=8.5, framealpha=0.92,
                      edgecolor="#cccccc", borderpad=0.4)
    leg.get_frame().set_linewidth(0.5)

    # Panel B: canonical audit_discipline frequency at final generation
    cond_names = [n for n in COND_STYLE if n in cond_records]
    x = np.arange(len(cond_names))
    means, sds, colors = [], [], []
    labels = []
    for name in cond_names:
        m, s = aggregate_canonical_final(cond_records[name], "audit_discipline")
        means.append(m)
        sds.append(s)
        colors.append(COND_STYLE[name]["color"])
        labels.append(COND_STYLE[name]["label"])
    ax_b.bar(x, means, yerr=sds, color=colors, edgecolor="white",
             linewidth=0.8, capsize=4,
             error_kw={"elinewidth": 0.9, "ecolor": "#444"})
    ax_b.axhline(1.0, color="#888", linewidth=0.6, linestyle=":")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([r"$\rho=0$", r"$\rho=1$"], fontsize=10)
    ax_b.set_ylabel(r"canonical \texttt{audit\_discipline} fraction"
                    "\n(final generation)")
    ax_b.set_ylim(0, 1.08)
    ax_b.set_title(
        "B   Canonical-allele frequency at audit_discipline  "
        "(final generation)",
        loc="left",
    )
    ax_b.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.suptitle(
        "Demo F  --  LLM decision engine (gemma-4-E2B-it, local vLLM) on "
        "MiniCAGE / CAGE-2\n"
        rf"$\mathrm{{(MiniCAGE;\;\;{G+1}\;generations,\;\;{n_seeds}\;replicate\;"
        rf"seeds,\;\;30\text{{-}}tick\;episodes,\;\;Meander\;red\;agent)}}$",
        fontsize=10.5, fontweight="bold", y=1.05,
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
                   default=str(repo_root / "telemetry" / "demo_f_llm_decision"))
    p.add_argument("--out", type=str,
                   default=str(repo_root / "figures" / "v2" / "demo_f"))
    args = p.parse_args()
    make_figure(Path(args.telemetry), Path(args.out))


if __name__ == "__main__":
    main()

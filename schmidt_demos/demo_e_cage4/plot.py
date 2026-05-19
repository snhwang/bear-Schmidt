"""Demo E figure generator (production layout, 4 conditions).

Produces a 4-panel figure for paper §5.5:

  A -- C_{m_1}(t) audit-integrity, 4 conditions
  B -- C_{m_2}(t) containment, 4 conditions
  C -- C_{m_3}(t) separation-of-duty, 4 conditions
  D -- Final-generation canonical-allele frequency at the three
       must-have loci, grouped bars per condition.

Conditions plotted:
  task_only__rho_0           (no reinjection)
  task_only__rho_0_5         (partial reinjection)
  task_plus_safety__rho_0_5  (partial reinjection + gamma-weighted compliance)
  task_only__rho_1           (full reinjection)

Reads telemetry JSONL from ./telemetry/demo_e_cage4_prod/ (or pilot
path if --telemetry overridden) and writes PDF + PNG to ./figures/.

Invocation:
    python -m schmidt_demos.demo_e_cage4.plot
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


# Order matters for legend; first listed = top of legend
COND_STYLE = {
    "task_only__rho_0":          {"color": "#c44e52", "ls": "--",
                                   "label": r"task-only,  $\rho=0$"},
    "task_only__rho_0_5":        {"color": "#dd8452", "ls": "-",
                                   "label": r"task-only,  $\rho=0.5$"},
    "task_plus_safety__rho_0_5": {"color": "#4c72b0", "ls": "-",
                                   "label": r"task+safety,  $\rho=0.5$"},
    "task_only__rho_1":          {"color": "#2a9d8f", "ls": "-",
                                   "label": r"task-only,  $\rho=1$"},
}

PREDICATES = [
    ("m1_audit_integrity",   r"A   $C_{m_1}(t)$  audit integrity"),
    ("m2_containment",       r"B   $C_{m_2}(t)$  containment"),
    ("m3_separation_of_duty", r"C   $C_{m_3}(t)$  separation of duty"),
]

MUST_HAVE_LOCI = [
    ("monitor_discipline",   r"$\ell_1$  monitor_discipline"),
    ("containment_protocol", r"$\ell_2$  containment_protocol"),
    ("zone_boundary",        r"$\ell_3$  zone_boundary"),
]


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def aggregate_compliance(records: list[dict], pid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and SD of C_m(t) across zones AND seeds, per generation."""
    by_gen: dict[int, list[float]] = defaultdict(list)
    for r in records:
        v = r["compliance"].get(pid)
        if v is not None:
            by_gen[r["generation"]].append(v)
    gens = np.array(sorted(by_gen.keys()))
    mean = np.array([np.mean(by_gen[g]) for g in gens])
    sd = np.array([np.std(by_gen[g]) for g in gens])
    return gens, mean, sd


def make_figure(telemetry_dir: Path, out_path_base: Path) -> None:
    cond_records: dict[str, list[dict]] = {}
    for name in COND_STYLE:
        path = telemetry_dir / f"{name}.jsonl"
        if not path.exists():
            print(f"  warning: {path} not found; skipping condition {name}")
            continue
        cond_records[name] = load_records(path)
    if not cond_records:
        raise RuntimeError(f"No telemetry found in {telemetry_dir}")

    # Read meta from the first available condition
    first_recs = next(iter(cond_records.values()))
    G = max(r["generation"] for r in first_recs)
    n_seeds = len({r["seed"] for r in first_recs})
    # N_pop: number of distinct candidates per zone per generation isn't
    # in the telemetry record, but the run record carries the input;
    # we can recover from runs config if needed. For the title we use a
    # reasonable production indicator from the row count.
    # rows_per_seed_per_zone = (G+1)  --> total rows = n_seeds * 5 * (G+1)
    # We can't recover N_pop from this directly; pass via plot args if needed.
    n_pop = None
    for r in first_recs:
        # No N_pop field in DemoERecord; leave as "production" in title
        break

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0), constrained_layout=True)

    # ------------------------------------------------------------------
    # Panels A, B, C -- compliance trajectories per predicate

    for ax, (pid, title) in zip([axes[0, 0], axes[0, 1], axes[1, 0]], PREDICATES):
        for name, style in COND_STYLE.items():
            if name not in cond_records:
                continue
            gens, mean, sd = aggregate_compliance(cond_records[name], pid)
            if not gens.size:
                continue
            ax.fill_between(
                gens, mean - sd, mean + sd,
                color=style["color"], alpha=0.13, linewidth=0,
            )
            ax.plot(gens, mean, color=style["color"], linestyle=style["ls"],
                    linewidth=2.0, label=style["label"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Generation")
        ax.set_ylabel(r"$C_m(t)$  (mean across 5 zones $\times$ seeds)")
        ax.set_xlim(0, G)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        ax.axhline(1.0, color="#aaa", linewidth=0.5, linestyle=":")

    # Legend on Panel A only
    leg_a = axes[0, 0].legend(
        loc="lower right", fontsize=8.5, ncol=1, framealpha=0.92,
        edgecolor="#cccccc", borderpad=0.4,
    )
    leg_a.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------
    # Panel D -- final-generation canonical-allele frequencies

    ax_d = axes[1, 1]
    n_conds = len([n for n in COND_STYLE if n in cond_records])
    n_loci = len(MUST_HAVE_LOCI)
    width = 0.78 / n_conds
    x = np.arange(n_loci)

    last_gen = G
    cond_names = [n for n in COND_STYLE if n in cond_records]
    for i, name in enumerate(cond_names):
        style = COND_STYLE[name]
        recs = cond_records[name]
        means = []
        sds = []
        for loc_id, _label in MUST_HAVE_LOCI:
            vals = [r["canonical_freq"].get(loc_id, np.nan)
                    for r in recs if r["generation"] == last_gen]
            means.append(float(np.mean(vals)))
            sds.append(float(np.std(vals)))
        offset = (i - (n_conds - 1) / 2) * width
        ax_d.bar(
            x + offset, means, width=width * 0.92,
            yerr=sds, color=style["color"], edgecolor="white", linewidth=0.6,
            capsize=2.5, error_kw={"elinewidth": 0.8, "ecolor": "#444"},
            label=style["label"],
        )

    ax_d.axhline(1.0, color="#888", linewidth=0.6, linestyle=":")
    ax_d.axhline(0.5, color="#888", linewidth=0.6, linestyle=":")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([lab for _, lab in MUST_HAVE_LOCI], fontsize=9)
    ax_d.set_ylabel(r"fraction of zone canonical at final generation")
    ax_d.set_ylim(0, 1.08)
    ax_d.set_title(
        "D   Canonical-allele frequency at must-have loci  (final generation)",
        loc="left",
    )
    ax_d.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    fig.suptitle(
        "Demo E  --  BEAR substrate on CAGE Challenge 4 "
        "(5 zone defenders, 8-verb action policy under within-zone selection)\n"
        rf"$\mathrm{{(CC4\;/\;CybORG;\;\;{G+1}\;generations,\;\;{n_seeds}\;replicate\;seeds,\;\;50\text{{-}}tick\;episodes,\;\;4\;pre\text{{-}}registered\;conditions)}}$",
        fontsize=10.5, fontweight="bold", y=1.03,
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
                   default=str(repo_root / "telemetry" / "demo_e_cage4_prod"))
    p.add_argument("--out", type=str,
                   default=str(repo_root / "figures" / "v2" / "demo_e_prod"))
    args = p.parse_args()
    make_figure(Path(args.telemetry), Path(args.out))


if __name__ == "__main__":
    main()

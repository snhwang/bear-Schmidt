"""Demo C R4-Part-3 — phi_0 sensitivity sweep.

Renders a two-panel figure:
  A: V_inv vs phi_0 with bootstrap 95% CIs, plus the main-condition (phi_0=0.5)
     point for context. Shows monotone decline and the invasion threshold.
  B: f_allele_GA1 trajectory for each phi_0, showing the common attractor.

The headline finding is that G_A1 alleles invade under task-only selection
across the tested phi_0 range, with V_inv > 0 (95% CI excluding 0) at every
tested phi_0 in [0.02, 0.10]; the dynamics converge to a common attractor
near f_allele_GA1 = 0.30 regardless of starting frequency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from schmidt_demos.demo_c_coevolution.plot import (
    estimate_invasion_fitness_expfit, load,
)


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


PHI_FILES = [
    ("0.02", "invasion_phi_002.jsonl"),
    ("0.05", "invasion_phi_005.jsonl"),
    ("0.10", "invasion_phi_010.jsonl"),
]
MAIN_FILE = ("0.50", "main_lambda_0.jsonl")  # for the (phi_0=0.5, rho=0) point


def make_figure(sweep_dir: Path, main_dir: Path, out_path_base: Path) -> None:
    # Compute V_inv per phi_0
    data: list[tuple[float, dict, list[dict]]] = []  # (phi_0, expfit_result, records)
    for phi_label, fname in PHI_FILES:
        p = sweep_dir / fname
        if not p.exists():
            continue
        records = load(p)
        # All these phi_0 conditions used N=200 invasion arm
        result = estimate_invasion_fitness_expfit(records, n_pop=200,
                                                   observable="f_allele_GA1")
        data.append((float(phi_label), result, records))

    # Add the main_lambda_0 (phi_0=0.5) point — N=30 main arm
    main_path = main_dir / MAIN_FILE[1]
    if main_path.exists():
        records_main = load(main_path)
        # Note: main_lambda_0 starts at phi_0=0.5, not a rare-mutant invasion
        result_main = estimate_invasion_fitness_expfit(
            records_main, n_pop=30, observable="f_allele_GA1",
        )
        data.append((0.50, result_main, records_main))
    else:
        records_main = None

    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11.0, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.2])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # ------------------------------------------------------------------
    # Panel A: V_inv vs phi_0
    phis = [d[0] for d in data]
    means = [d[1]["mean"] for d in data]
    ci_los = [d[1]["ci_lo"] for d in data]
    ci_his = [d[1]["ci_hi"] for d in data]
    err_lo = [m - lo for m, lo in zip(means, ci_los)]
    err_hi = [hi - m for m, hi in zip(means, ci_his)]

    # Color by sign of V_inv
    colors = ["#c0392b" if m > 0 else "#2980b9" for m in means]
    ax_a.errorbar(phis, means, yerr=[err_lo, err_hi],
                  fmt="o", color="#222", markersize=8,
                  ecolor="#444", elinewidth=1.2, capsize=4,
                  markerfacecolor="white", markeredgewidth=1.6,
                  label=None)
    # Per-point coloring of the markers: make the marker face match sign
    for x, m, c in zip(phis, means, colors):
        ax_a.plot(x, m, "o", color=c, markersize=9, alpha=0.95, zorder=5)

    ax_a.axhline(0, color="#888", linewidth=0.8, linestyle="--",
                 label="$V_{\\mathrm{inv}} = 0$ (invasion threshold)")
    ax_a.set_xlabel(r"$\phi_0$  —  initial G$_{A_1}$ allele fraction")
    ax_a.set_ylabel(r"$\hat V_{\mathrm{inv}}$  per generation")
    ax_a.set_title(r"A   Invasion fitness vs. $\phi_0$  ($\rho=0$, task-only selection)",
                   loc="left")
    ax_a.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    # Annotation for each point
    for x, m, lo, hi in zip(phis, means, ci_los, ci_his):
        sign = "+" if m > 0 else ""
        ax_a.annotate(f"{sign}{m:.3f}",
                      xy=(x, m),
                      xytext=(8, 4 if m > 0 else -14), textcoords="offset points",
                      fontsize=8, color="#222")

    # Note about invasion threshold
    ax_a.text(0.30, max(means) * 0.45,
              "Invasion threshold\nin $\\phi_0 \\in (0.10, 0.50)$",
              fontsize=8, color="#444", ha="center",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#fafafa",
                        edgecolor="#cccccc", linewidth=0.5))

    leg_a = ax_a.legend(loc="upper right", fontsize=8.0)
    if leg_a:
        leg_a.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------
    # Panel B: f_allele_GA1 trajectories per phi_0 — shows the common attractor
    cmap = plt.colormaps.get_cmap("viridis")
    n_phi = len(data)
    for i, (phi_0, _, records) in enumerate(data):
        color = cmap(0.15 + 0.7 * (i / max(n_phi - 1, 1)))
        # Per-seed thin trajectories
        by_seed: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for r in records:
            by_seed[r["seed"]].append((r["generation"], r.get("f_allele_GA1", 0.0)))
        for seed, pts in by_seed.items():
            pts_sorted = sorted(pts)
            gs_ = [p[0] for p in pts_sorted]
            fs = [p[1] for p in pts_sorted]
            ax_b.plot(gs_, fs, color=color, linewidth=0.7, alpha=0.25)
        # Mean trajectory on top
        max_gen = max(r["generation"] for r in records)
        mean_traj_x = list(range(max_gen + 1))
        mean_traj_y = []
        for g_idx in mean_traj_x:
            vals = [r["f_allele_GA1"] for r in records
                    if r["generation"] == g_idx and "f_allele_GA1" in r]
            mean_traj_y.append(float(np.mean(vals)) if vals else float("nan"))
        ax_b.plot(mean_traj_x, mean_traj_y, color=color, linewidth=2.4,
                  marker="o", markersize=4,
                  label=rf"$\phi_0 = {phi_0:.2f}$")

    # Attractor line
    ax_b.axhline(0.30, color="#444", linewidth=0.8, linestyle=":",
                 label=r"empirical attractor $\sim 0.30$")
    ax_b.set_xlabel("Generation")
    ax_b.set_ylabel(r"$f_{\mathrm{allele},G_{A_1}}(t)$")
    ax_b.set_title(r"B   Allele-frequency trajectories  (50 seeds per $\phi_0$, $N=200$)",
                   loc="left")
    ax_b.set_ylim(-0.02, 0.55)
    ax_b.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    leg_b = ax_b.legend(loc="upper left", fontsize=8.0,
                        framealpha=0.92, edgecolor="#cccccc", borderpad=0.4)
    leg_b.get_frame().set_linewidth(0.5)

    fig.suptitle(
        r"Demo C R4-Part-3  —  $\phi_0$ sensitivity:  G$_{A_1}$ allele invasion is robust at $\rho=0$"
        "\n"
        r"$V_{\mathrm{inv}} > 0$ across $\phi_0 \in \{0.02, 0.05, 0.10\}$ (95% CIs exclude 0); attractor at $f_{\mathrm{allele}} \approx 0.30$",
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
    p.add_argument("--sweep-dir", type=str,
                   default=str(repo_root / "telemetry" / "paper_v0" / "demo_c_r4_phi_sweep"))
    p.add_argument("--main-dir", type=str,
                   default=str(repo_root / "telemetry" / "paper_v0" / "demo_c_r4"))
    p.add_argument("--out", type=str,
                   default=str(repo_root / "figures" / "demo_c_phi_sweep"))
    args = p.parse_args()
    make_figure(Path(args.sweep_dir), Path(args.main_dir), Path(args.out))


if __name__ == "__main__":
    main()

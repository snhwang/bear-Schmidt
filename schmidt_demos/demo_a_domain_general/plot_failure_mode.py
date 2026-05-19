"""Failure-mode / operating-envelope figure for Demo A.

Renders a 2-panel figure summarising the (mu, rho) sweep produced by
`run_failure_mode_sweep.py`:

  Panel A: heatmap of final-generation canonical-allele frequency at
           the must-have locus, as a function of (mu, rho). Cells
           coloured by compliance.
  Panel B: trajectories at the rho=0.5 slice (the "partial enforcement
           degrades gracefully" story), one curve per mu value.

The headline finding: at rho=1, compliance is pinned at 1.000 across
the full mu range tested (the Colony pipeline's enforce-must-have-last
ordering catches every mutation-induced variant). At rho<1, the
substrate's compliance ceiling degrades smoothly with mu. The
substrate has a wide operating envelope, with a structural
protection against the naive "mutation overwhelms enforcement"
failure mode.
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


def load_sweep(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def make_figure(sweep_path: Path, out_path_base: Path) -> None:
    rows = load_sweep(sweep_path)
    if not rows:
        raise RuntimeError(f"empty sweep: {sweep_path}")

    last_gen = max(r["generation"] for r in rows)
    mus = sorted({r["mu"] for r in rows})
    rhos = sorted({r["rho"] for r in rows})
    n_seeds = len({r["seed"] for r in rows if r["mu"] == mus[0] and r["rho"] == rhos[0]})

    # Build the (mu, rho) -> final canonical-freq mean and SD
    by_cell: dict[tuple[float, float], list[float]] = defaultdict(list)
    for r in rows:
        if r["generation"] == last_gen:
            by_cell[(r["mu"], r["rho"])].append(r["canonical_freq"])

    grid_mean = np.zeros((len(mus), len(rhos)))
    grid_sd = np.zeros((len(mus), len(rhos)))
    for i, mu in enumerate(mus):
        for j, rho in enumerate(rhos):
            vals = by_cell[(mu, rho)]
            grid_mean[i, j] = float(np.mean(vals)) if vals else np.nan
            grid_sd[i, j] = float(np.std(vals)) if vals else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    ax_a, ax_b = axes

    # ------------------------------------------------------------------
    # Panel A: heatmap

    im = ax_a.imshow(
        grid_mean, aspect="auto", origin="lower",
        vmin=0.0, vmax=1.0, cmap="RdYlGn",
        extent=(-0.5, len(rhos) - 0.5, -0.5, len(mus) - 0.5),
    )
    cbar = fig.colorbar(im, ax=ax_a, shrink=0.85, pad=0.02)
    cbar.set_label(r"final $C_m$ (canonical-allele fraction)")

    # Annotate cells with values
    for i, mu in enumerate(mus):
        for j, rho in enumerate(rhos):
            v = grid_mean[i, j]
            # Pick text colour for contrast
            txt_col = "white" if v < 0.4 or v > 0.85 else "black"
            ax_a.text(j, i, f"{v:.2f}", ha="center", va="center",
                      color=txt_col, fontsize=8.5, fontweight="bold")

    ax_a.set_xticks(range(len(rhos)))
    ax_a.set_xticklabels([f"{r:.2f}" for r in rhos])
    ax_a.set_yticks(range(len(mus)))
    ax_a.set_yticklabels([f"{m:.2f}" for m in mus])
    ax_a.set_xlabel(r"reinjection rate $\rho$")
    ax_a.set_ylabel(r"mutation rate $\mu$ at must-have locus")
    ax_a.set_title(
        "A   Operating envelope:  final $C_m$ across $(\mu, \\rho)$",
        loc="left",
    )

    # ------------------------------------------------------------------
    # Panel B: trajectories at the rho=0.5 slice

    by_traj: dict[tuple[float, int], list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        if r["rho"] != 0.5:
            continue
        by_traj[(r["mu"], r["seed"])].append((r["generation"], r["canonical_freq"]))

    mu_colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(mus)))
    for color, mu in zip(mu_colors, mus):
        # Average across seeds per generation
        by_gen: dict[int, list[float]] = defaultdict(list)
        for (m, s), pts in by_traj.items():
            if m != mu:
                continue
            for g, c in pts:
                by_gen[g].append(c)
        gens = sorted(by_gen)
        mean = [float(np.mean(by_gen[g])) for g in gens]
        sd = [float(np.std(by_gen[g])) for g in gens]
        ax_b.fill_between(
            gens, np.array(mean) - np.array(sd), np.array(mean) + np.array(sd),
            color=color, alpha=0.13, linewidth=0,
        )
        ax_b.plot(gens, mean, color=color, linewidth=1.8,
                  label=rf"$\mu = {mu:.2f}$")

    ax_b.set_xlim(0, last_gen)
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.set_xlabel("Generation")
    ax_b.set_ylabel(r"$C_m(t)$")
    ax_b.set_title(
        r"B   Partial enforcement ($\rho=0.5$) degrades gracefully with $\mu$",
        loc="left",
    )
    ax_b.axhline(0.95, color="#888", linewidth=0.6, linestyle=":")
    ax_b.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    leg_b = ax_b.legend(loc="lower right", fontsize=8, framealpha=0.92,
                        edgecolor="#cccccc", borderpad=0.4, ncol=2)
    leg_b.get_frame().set_linewidth(0.5)

    fig.suptitle(
        "Demo A operating envelope:  reinjection at $\\rho=1$ is robust across $\\mu \\in [0, 0.5]$;  "
        "partial enforcement degrades smoothly\n"
        rf"$\mathrm{{(N\!=\!200,\;\;{last_gen}\;generations,\;\;{n_seeds}\;replicate\;seeds\;per\;cell,\;\;task\text{{-}}only\;regime)}}$",
        fontsize=10.5, fontweight="bold", y=1.06,
    )

    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    pdf = out_path_base.with_suffix(".pdf")
    png = out_path_base.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


def main() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry", type=str,
                   default=str(repo_root / "telemetry" / "demo_a_failure_mode" / "sweep.jsonl"))
    p.add_argument("--out", type=str,
                   default=str(repo_root / "figures" / "v2" / "demo_a_envelope"))
    args = p.parse_args()
    make_figure(Path(args.telemetry), Path(args.out))


if __name__ == "__main__":
    main()

"""Demo C figure — co-evolution phase plane.

Three panels for the §3 figure:
    A — Phase plane in (f_{G_D}, f_{G_A_1}) with arrows showing time
        direction; the two main conditions plus the rare-mutant
        invasion run, all overlaid in the simplex triangle.
    B — Time series of f_{G_D}(t) and f_{G_A_1}(t) for the two main
        conditions, demonstrating that λ=1 vs λ=0 produce visibly
        different fates.
    C — Invasion-fitness readout. Rare-mutant trajectory (φ_0=0.05);
        annotated with the early-phase V_inv estimate and its
        confidence interval across replicate seeds.

Reads telemetry JSONL from ./telemetry/demo_c/ and writes a PDF + PNG
to ./figures/.

Invocation:
    .venv/Scripts/python.exe -m schmidt_demos.demo_c_coevolution.plot
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrow, Polygon


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
    "main_lambda_0":      {"color": "#c44e52", "label": r"main: $\phi_0=0.5,\;\rho=0$  (no enforcement)"},
    "main_lambda_1":      {"color": "#2a9d8f", "label": r"main: $\phi_0=0.5,\;\rho=1$  (full reinjection)"},
    "invasion_lambda_0":  {"color": "#5b6dbf", "label": r"invasion: $\phi_0=0.05,\;\rho=0$  (rare $G_{A_1}$)"},
}


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def aggregate(records: list[dict]) -> dict[str, np.ndarray]:
    by_gen: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        by_gen[r["generation"]]["GD"].append(r["f_pure_GD"])
        by_gen[r["generation"]]["GA1"].append(r["f_pure_GA1"])
        by_gen[r["generation"]]["mixed"].append(r["f_mixed"])
    gens = np.array(sorted(by_gen.keys()))
    out = {"generation": gens}
    for k in ("GD", "GA1", "mixed"):
        out[f"{k}_mean"] = np.array([np.mean(by_gen[g][k]) for g in gens])
        out[f"{k}_sd"]   = np.array([np.std(by_gen[g][k]) for g in gens])
    return out


def estimate_invasion_fitness(records: list[dict], k: int = 5) -> tuple[float, float]:
    """Per-replicate V_inv = (1/k) · ln(f_att(k) / f_att(0)), averaged
    across replicates with the standard error.

    Replicates whose attacker fraction reaches 0 before generation k
    are clamped to a small floor so the log is finite.

    NOTE: This estimator saturates at the 1e-3 floor when rare-mutant
    lineages extinguish before generation k, and the integer rounding
    of φ_0·N at small N collapses seed-to-seed variance in f_0 to a
    handful of values. Both effects squeeze trajectory-level seed
    variance out of the readout. Kept here as a supplementary check;
    `estimate_invasion_fitness_expfit` below is the primary estimator
    for the v0 paper. See docs/DEMO_C_VARIANCE_NOTE.md.
    """
    by_seed: dict[int, list[float]] = defaultdict(lambda: [None] * (k + 1))   # type: ignore[arg-type]
    for r in records:
        if r["generation"] <= k:
            by_seed[r["seed"]][r["generation"]] = r["f_pure_GA1"]
    floor = 1e-3
    vinvs: list[float] = []
    for seed, traj in by_seed.items():
        if traj[0] is None or traj[k] is None:
            continue
        f0 = max(traj[0], floor)
        fk = max(traj[k], floor)
        vinvs.append(np.log(fk / f0) / k)
    if not vinvs:
        return float("nan"), float("nan")
    arr = np.array(vinvs)
    return float(arr.mean()), float(arr.std() / np.sqrt(max(len(arr), 1)))


def _pre_extinction_window(traj_by_seed: dict[int, list[float]], n_pop: int) -> int:
    """Largest k such that every replicate has f_att(t) > 1/N for all t <= k.

    1/N is the smallest representable nonzero frequency in a population
    of size N. Going beyond it pulls the floor clamp into the fit and
    re-introduces the saturation artifact we are trying to avoid.
    """
    threshold = 1.0 / n_pop
    max_t = max(len(traj) - 1 for traj in traj_by_seed.values())
    for k in range(1, max_t + 1):
        if not all(
            traj[t] is not None and traj[t] > threshold
            for traj in traj_by_seed.values()
            for t in range(k + 1)
        ):
            return k - 1
    return max_t


def estimate_invasion_fitness_expfit(
    records: list[dict],
    *,
    n_pop: int,
    observable: str = "f_allele_GA1",
    n_bootstrap: int = 2000,
    rng_seed: int = 20260508,
) -> dict:
    """Per-replicate exp-fit V_inv, averaged across replicates with a
    nonparametric bootstrap CI.

    Fits f_att(t) = f_0 · exp(V_inv · t) by OLS on log f_att vs. t over
    the pre-extinction window (largest k such that every replicate
    stays above 1/N for all t <= k). This avoids the floor-clamp
    saturation that affects `estimate_invasion_fitness`.

    `observable`: which trajectory field to fit. Default `f_allele_GA1`
    (G_A1 allele frequency averaged across must-have loci) is the
    population-genetics observable robust to the 1/N rounding floor.
    Legacy callers can pass `"f_pure_GA1"` for the individual-fraction
    observable, which can saturate at 0 in small populations.

    Returns dict with: mean, se, ci_lo, ci_hi, k_used, n_replicates,
    per_seed_estimates.
    """
    by_seed: dict[int, list[float | None]] = defaultdict(list)
    max_gen = max(r["generation"] for r in records)
    for s in {r["seed"] for r in records}:
        traj: list[float | None] = [None] * (max_gen + 1)
        for r in records:
            if r["seed"] == s:
                traj[r["generation"]] = r.get(observable)
        by_seed[s] = traj

    k_used = _pre_extinction_window(by_seed, n_pop)
    if k_used < 2:
        return {"mean": float("nan"), "se": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "k_used": k_used, "n_replicates": len(by_seed),
                "per_seed_estimates": {}}

    per_seed: dict[int, float] = {}
    ts = np.arange(k_used + 1, dtype=float)
    for s, traj in by_seed.items():
        ys = np.log(np.array(traj[: k_used + 1], dtype=float))
        # OLS slope of log f_att vs t -> V_inv
        slope, _intercept = np.polyfit(ts, ys, deg=1)
        per_seed[s] = float(slope)

    estimates = np.array(list(per_seed.values()))
    rng = np.random.default_rng(rng_seed)
    boot_means = np.array([
        rng.choice(estimates, size=len(estimates), replace=True).mean()
        for _ in range(n_bootstrap)
    ])
    return {
        "mean": float(estimates.mean()),
        "se": float(estimates.std(ddof=1) / np.sqrt(len(estimates))),
        "ci_lo": float(np.percentile(boot_means, 2.5)),
        "ci_hi": float(np.percentile(boot_means, 97.5)),
        "k_used": k_used,
        "n_replicates": len(estimates),
        "per_seed_estimates": per_seed,
    }


# ---------------------------------------------------------------------------

def make_figure(telemetry_dir: Path, out_path_base: Path) -> None:
    raw = {name: load(telemetry_dir / f"{name}.jsonl") for name in COND_STYLE}
    agg = {name: aggregate(raw[name]) for name in COND_STYLE}

    fig = plt.figure(figsize=(13.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    # ------------------------------------------------------------------
    # Panel A — phase plane

    # Feasibility simplex: f_GD + f_GA1 + f_mixed = 1, all >= 0.
    # Project to (f_GD, f_GA1); feasible region is the triangle
    # (0,0)-(1,0)-(0,1).
    triangle = Polygon(
        [(0, 0), (1, 0), (0, 1)],
        closed=True, facecolor="#f4f4f4", edgecolor="#cccccc",
        linewidth=0.6,
    )
    ax_a.add_patch(triangle)

    # Annotate the corners
    ax_a.annotate("pure $G_D$", xy=(1.0, 0.0), xytext=(0.86, 0.02),
                  fontsize=8.5, color="#444")
    ax_a.annotate("pure $G_{A_1}$", xy=(0.0, 1.0), xytext=(0.02, 0.93),
                  fontsize=8.5, color="#444")
    ax_a.annotate("all mixed", xy=(0.0, 0.0), xytext=(0.02, 0.02),
                  fontsize=8.5, color="#444")

    # Plot each condition's trajectory (mean across replicates)
    for name, style in COND_STYLE.items():
        a = agg[name]
        xs = a["GD_mean"]
        ys = a["GA1_mean"]
        # main line
        ax_a.plot(xs, ys, color=style["color"], linewidth=2.0,
                  marker="o", markersize=3.5, alpha=0.95,
                  label=style["label"])
        # arrow on the dominant transition (gen 0 -> gen 1)
        if len(xs) >= 2 and (xs[1] - xs[0])**2 + (ys[1] - ys[0])**2 > 1e-3:
            ax_a.annotate(
                "", xytext=(xs[0], ys[0]), xy=(xs[1], ys[1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=style["color"],
                    linewidth=1.6,
                    alpha=0.95,
                ),
            )
        # mark gen 0 (open) and final gen (filled)
        ax_a.plot(xs[0], ys[0], marker="o", markersize=8,
                  markerfacecolor="white", markeredgecolor=style["color"],
                  markeredgewidth=1.6)
        ax_a.plot(xs[-1], ys[-1], marker="*", markersize=12,
                  color=style["color"])

    ax_a.set_xlim(-0.04, 1.04)
    ax_a.set_ylim(-0.04, 1.04)
    ax_a.set_xlabel(r"$f_{G_D}(t)$  —  fraction of colony with canonical $M$ alleles")
    ax_a.set_ylabel(r"$f_{G_{A_1}}(t)$  —  fraction with must-have-erosion alleles")
    ax_a.set_title("A   Phase plane  (open ○ = gen 0,  ★ = gen 14)", loc="left")
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    leg = ax_a.legend(loc="upper right", fontsize=8.0, framealpha=0.92,
                      edgecolor="#cccccc", borderpad=0.4)
    leg.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------
    # Panel B — time series of f_GD and f_GA1 for the two main conditions

    for name in ("main_lambda_0", "main_lambda_1"):
        a = agg[name]
        c = COND_STYLE[name]["color"]
        # f_GD: solid line
        ax_b.fill_between(a["generation"], a["GD_mean"] - a["GD_sd"],
                          a["GD_mean"] + a["GD_sd"],
                          color=c, alpha=0.13, linewidth=0)
        ax_b.plot(a["generation"], a["GD_mean"], color=c, linewidth=2.0,
                  linestyle="-", label=f"$f_{{G_D}}$ — {COND_STYLE[name]['label'].split(': ')[1].split(' ')[0]}")
        # f_GA1: dashed line
        ax_b.fill_between(a["generation"], a["GA1_mean"] - a["GA1_sd"],
                          a["GA1_mean"] + a["GA1_sd"],
                          color=c, alpha=0.08, linewidth=0)
        ax_b.plot(a["generation"], a["GA1_mean"], color=c, linewidth=2.0,
                  linestyle="--", label=f"$f_{{G_{{A_1}}}}$ — {COND_STYLE[name]['label'].split(': ')[1].split(' ')[0]}")

    G = max(agg["main_lambda_0"]["generation"])
    ax_b.set_xlim(0, G)
    ax_b.set_ylim(-0.02, 1.02)
    ax_b.set_xlabel("Generation")
    ax_b.set_ylabel("colony fraction")
    ax_b.set_title("B   Time series  (solid: $G_D$  ·  dashed: $G_{A_1}$)", loc="left")
    ax_b.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    leg_b = ax_b.legend(loc="center right", fontsize=7.5, framealpha=0.92,
                        edgecolor="#cccccc", borderpad=0.4, handlelength=2.2)
    leg_b.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------
    # Panel C — invasion-fitness readout (G_A1 allele frequency trajectory)
    #
    # We plot the population-genetics observable f_allele_GA1 (mean G_A1
    # allele frequency across must-have loci), not the individual-fraction
    # f_pure_GA1, because the latter collapses under recombination even
    # when the alleles are spreading (see DEMO_C_VARIANCE_NOTE.md).

    inv_color = COND_STYLE["invasion_lambda_0"]["color"]

    # Replicate-level trajectories of f_allele_GA1 (so we can see spread)
    by_seed_alleles: dict[int, list[tuple[int, float]]] = defaultdict(list)
    has_allele_field = any("f_allele_GA1" in r for r in raw["invasion_lambda_0"])
    obs_key = "f_allele_GA1" if has_allele_field else "f_pure_GA1"
    for r in raw["invasion_lambda_0"]:
        by_seed_alleles[r["seed"]].append((r["generation"], r.get(obs_key, 0.0)))
    for seed, pts in by_seed_alleles.items():
        pts_sorted = sorted(pts)
        gs = [p[0] for p in pts_sorted]
        fs = [p[1] for p in pts_sorted]
        ax_c.plot(gs, fs, color=inv_color, linewidth=0.9, alpha=0.45)

    # Mean trajectory on top
    max_gen = max(r["generation"] for r in raw["invasion_lambda_0"])
    mean_traj = []
    for g_idx in range(max_gen + 1):
        vals = [r[obs_key] for r in raw["invasion_lambda_0"]
                if r["generation"] == g_idx and obs_key in r]
        if vals:
            mean_traj.append((g_idx, float(np.mean(vals))))
    ax_c.plot([p[0] for p in mean_traj], [p[1] for p in mean_traj],
              color=inv_color, linewidth=2.4, marker="o", markersize=3.5,
              label="mean across replicates")
    ax_c.axhline(0.05, color="#888", linewidth=0.8, linestyle=":",
                 label=r"$\phi_0 = 0.05$ (seed)")

    # Compute V_inv with the corrected exp-fit estimator on the corrected observable
    if has_allele_field:
        result = estimate_invasion_fitness_expfit(
            raw["invasion_lambda_0"], n_pop=200, observable="f_allele_GA1",
        )
        v_mean = result["mean"]
        v_ci_lo = result["ci_lo"]
        v_ci_hi = result["ci_hi"]
    else:
        # Legacy fallback (broken floor-clamp estimator on f_pure_GA1)
        v_mean, v_se = estimate_invasion_fitness(raw["invasion_lambda_0"], k=5)
        v_ci_lo = v_ci_hi = float("nan")

    ax_c.set_xlim(0, max_gen)
    ymax = max(0.16, max((p[1] for p in mean_traj), default=0.16) * 1.1)
    ax_c.set_ylim(-0.01, ymax)
    ax_c.set_xlabel("Generation")
    ax_c.set_ylabel(r"$f_{\mathrm{allele},G_{A_1}}(t)$" if has_allele_field
                    else r"$f_{\mathrm{pure},G_{A_1}}(t)$")
    ax_c.set_title("C   Invasion fitness  (rare-$G_{A_1}$ allele-frequency trajectory)"
                   if has_allele_field else
                   "C   Invasion-fitness  (rare-$G_{A_1}$ trajectory)",
                   loc="left")
    ax_c.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    # Annotation box with the V_inv estimate
    if has_allele_field and not np.isnan(v_mean):
        sign_word = "positive" if v_mean > 0 else "negative"
        interp = ("$\\Rightarrow$ G$_{A_1}$ alleles invade at $\\rho=0$"
                  if v_mean > 0
                  else "$\\Rightarrow$ no invasion at $\\rho=0$")
        ax_c.text(
            0.97, 0.05,
            rf"$\hat V_{{\mathrm{{inv}}}} = {v_mean:+.3f}$"
            "\n"
            rf"95% CI $[{v_ci_lo:+.3f},\,{v_ci_hi:+.3f}]$"
            f"\n{interp}",
            transform=ax_c.transAxes, fontsize=8.5, color="#222",
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", alpha=0.95, linewidth=0.5),
        )
    elif not np.isnan(v_mean):
        ax_c.text(
            0.97, 0.95,
            rf"$V_{{\mathrm{{inv}}}}^{{(0..5)}} = {v_mean:+.2f}$"
            "\n(legacy estimator)",
            transform=ax_c.transAxes, fontsize=8.5, color="#222",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#cccccc", alpha=0.95, linewidth=0.5),
        )

    leg_c = ax_c.legend(loc="lower right" if v_mean <= 0 else "upper left",
                        fontsize=7.5, framealpha=0.92, edgecolor="#cccccc", borderpad=0.4)
    leg_c.get_frame().set_linewidth(0.5)

    # Top-level title — adaptive to invasion N (R4 uses N=200 for invasion arm)
    n_seeds = len({r["seed"] for r in raw["main_lambda_0"]})
    n_inv_seeds = len({r["seed"] for r in raw["invasion_lambda_0"]})
    # Detect invasion-arm N from telemetry if available; fall back to text.
    inv_N_note = "N=30/200 (invasion arm)" if n_inv_seeds == 50 else "N=30"
    fig.suptitle(
        "Demo C  —  two-strategy co-evolution kernel:  "
        r"$G_D$ defender vs. $G_{A_1}$ must-have erosion on MiniCAGE"
        f"\n$\\mathrm{{({inv_N_note},\\;\\;15\\;generations,\\;\\;{n_seeds}\\;replicate\\;seeds;\\;\\;CybORG++\\;MiniCAGE\\;/\\;CAGE\\text{{-}}2)}}$",
        fontsize=10.5, fontweight="bold", y=1.10,
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
                   default=str(repo_root / "telemetry" / "demo_c"))
    p.add_argument("--out", type=str,
                   default=str(repo_root / "figures" / "demo_c_preliminary"))
    args = p.parse_args()
    make_figure(Path(args.telemetry), Path(args.out))


if __name__ == "__main__":
    main()

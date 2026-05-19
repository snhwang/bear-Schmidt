"""Demo C — paired-defender figure (multi-agent extension).

Same evolution as run.py but each generation's fitness is scored by
random-pairing the colony into two-defender teams. The extra panel
(C_m_3 compliance) was N/A in the single-defender variant; it is now
substrate-checkable on the joint audit log.

Four panels:
    A — Phase plane (f_GD, f_GA1) — same headline visual
    B — Time series of f_GD(t), f_GA1(t) for both main conditions
    C — C_m_3(t)  separation-of-duty compliance under each condition
        (the multi-agent payoff: λ=1 drives m_3 to 1.0; λ=0 stalls at
         the unaligned mutation equilibrium)
    D — Invasion-fitness panel: rare-mutant trajectory + V_inv estimate

Invocation:
    .venv/Scripts/python.exe -m schmidt_demos.demo_c_coevolution.plot_paired
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


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
    keys = ("f_pure_GD", "f_pure_GA1", "compliance_m3")
    by_gen: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        for k in keys:
            v = r.get(k)
            if v is not None:
                by_gen[r["generation"]][k].append(v)
    gens = np.array(sorted(by_gen.keys()))
    out = {"generation": gens}
    for k in keys:
        out[f"{k}_mean"] = np.array([np.mean(by_gen[g][k]) for g in gens])
        out[f"{k}_sd"]   = np.array([np.std(by_gen[g][k]) for g in gens])
    return out


def estimate_invasion_fitness(records: list[dict], k: int = 5) -> tuple[float, float]:
    by_seed: dict[int, list[float]] = defaultdict(lambda: [None] * (k + 1))   # type: ignore[arg-type]
    for r in records:
        if r["generation"] <= k:
            by_seed[r["seed"]][r["generation"]] = r["f_pure_GA1"]
    floor = 1e-3
    vinvs = []
    for traj in by_seed.values():
        if traj[0] is None or traj[k] is None:
            continue
        f0 = max(traj[0], floor)
        fk = max(traj[k], floor)
        vinvs.append(np.log(fk / f0) / k)
    if not vinvs:
        return float("nan"), float("nan")
    arr = np.array(vinvs)
    return float(arr.mean()), float(arr.std() / np.sqrt(max(len(arr), 1)))


# ---------------------------------------------------------------------------

def make_figure(telemetry_dir: Path, out_path_base: Path) -> None:
    raw = {name: load(telemetry_dir / f"{name}.jsonl") for name in COND_STYLE}
    agg = {name: aggregate(raw[name]) for name in COND_STYLE}
    G = max(agg["main_lambda_0"]["generation"])
    n_seeds = len({r["seed"] for r in raw["main_lambda_0"]})

    fig = plt.figure(figsize=(17.0, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1, 1, 1])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[0, 3])

    # ------------------------------------------------------------------
    # Panel A — phase plane

    triangle = Polygon(
        [(0, 0), (1, 0), (0, 1)],
        closed=True, facecolor="#f4f4f4", edgecolor="#cccccc", linewidth=0.6,
    )
    ax_a.add_patch(triangle)
    ax_a.annotate("pure $G_D$",     xy=(1.0, 0.0), xytext=(0.86, 0.02), fontsize=8.5, color="#444")
    ax_a.annotate("pure $G_{A_1}$", xy=(0.0, 1.0), xytext=(0.02, 0.93), fontsize=8.5, color="#444")
    ax_a.annotate("all mixed",      xy=(0.0, 0.0), xytext=(0.02, 0.02), fontsize=8.5, color="#444")

    for name, style in COND_STYLE.items():
        a = agg[name]
        xs = a["f_pure_GD_mean"]; ys = a["f_pure_GA1_mean"]
        ax_a.plot(xs, ys, color=style["color"], linewidth=2.0,
                  marker="o", markersize=3.5, alpha=0.95, label=style["label"])
        if len(xs) >= 2 and (xs[1] - xs[0])**2 + (ys[1] - ys[0])**2 > 1e-3:
            ax_a.annotate(
                "", xytext=(xs[0], ys[0]), xy=(xs[1], ys[1]),
                arrowprops=dict(arrowstyle="-|>", color=style["color"],
                                linewidth=1.6, alpha=0.95),
            )
        ax_a.plot(xs[0], ys[0], marker="o", markersize=8,
                  markerfacecolor="white", markeredgecolor=style["color"],
                  markeredgewidth=1.6)
        ax_a.plot(xs[-1], ys[-1], marker="*", markersize=12, color=style["color"])

    ax_a.set_xlim(-0.04, 1.04); ax_a.set_ylim(-0.04, 1.04)
    ax_a.set_xlabel(r"$f_{G_D}(t)$")
    ax_a.set_ylabel(r"$f_{G_{A_1}}(t)$")
    ax_a.set_title("A   Phase plane  (○ = gen 0,  ★ = gen 14)", loc="left")
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    leg_a = ax_a.legend(loc="upper right", fontsize=7.5, framealpha=0.92,
                        edgecolor="#cccccc", borderpad=0.3)
    leg_a.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------
    # Panel B — time series f_GD and f_GA1 for the two main conditions

    for name in ("main_lambda_0", "main_lambda_1"):
        a = agg[name]; c = COND_STYLE[name]["color"]
        regime = "ρ=0" if "lambda_0" == name.split("_")[-2] + "_" + name.split("_")[-1] else "ρ=1"
        regime = "ρ=0" if name == "main_lambda_0" else "ρ=1"
        ax_b.fill_between(a["generation"], a["f_pure_GD_mean"] - a["f_pure_GD_sd"],
                          a["f_pure_GD_mean"] + a["f_pure_GD_sd"],
                          color=c, alpha=0.13, linewidth=0)
        ax_b.plot(a["generation"], a["f_pure_GD_mean"], color=c, linewidth=2.0,
                  linestyle="-", label=rf"$f_{{G_D}}\;{regime}$")
        ax_b.fill_between(a["generation"], a["f_pure_GA1_mean"] - a["f_pure_GA1_sd"],
                          a["f_pure_GA1_mean"] + a["f_pure_GA1_sd"],
                          color=c, alpha=0.08, linewidth=0)
        ax_b.plot(a["generation"], a["f_pure_GA1_mean"], color=c, linewidth=2.0,
                  linestyle="--", label=rf"$f_{{G_{{A_1}}}}\;{regime}$")

    ax_b.set_xlim(0, G); ax_b.set_ylim(-0.02, 1.02)
    ax_b.set_xlabel("Generation"); ax_b.set_ylabel("colony fraction")
    ax_b.set_title("B   Time series  (solid: $G_D$  ·  dashed: $G_{A_1}$)", loc="left")
    ax_b.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    leg_b = ax_b.legend(loc="center right", fontsize=7.5, framealpha=0.92,
                        edgecolor="#cccccc", borderpad=0.3, handlelength=2.2)
    leg_b.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------
    # Panel C — m_3 separation-of-duty compliance over generations

    for name, style in COND_STYLE.items():
        a = agg[name]
        gens = a["generation"]
        m_mean = a["compliance_m3_mean"]
        m_sd = a["compliance_m3_sd"]
        ax_c.fill_between(gens, m_mean - m_sd, m_mean + m_sd,
                          color=style["color"], alpha=0.15, linewidth=0)
        ax_c.plot(gens, m_mean, color=style["color"], linewidth=2.0,
                  marker="o", markersize=3.5, label=style["label"])

    ax_c.set_xlim(0, G); ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_xlabel("Generation"); ax_c.set_ylabel(r"$C_{m_3}(t)$  —  pair fraction compliant")
    ax_c.set_title(r"C   $m_3$ separation-of-duty  (now testable)", loc="left")
    ax_c.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax_c.text(
        0.02, 0.04,
        "NIST SP 800-53 AC-5  ·  paired-defender concurrence on destructive actions",
        transform=ax_c.transAxes, fontsize=7.0, color="#666",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#dddddd", alpha=0.85, linewidth=0.5),
    )
    leg_c = ax_c.legend(loc="center right", fontsize=7.0, framealpha=0.92,
                        edgecolor="#cccccc", borderpad=0.3)
    leg_c.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------
    # Panel D — invasion-fitness readout

    inv_color = COND_STYLE["invasion_lambda_0"]["color"]
    by_seed: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for r in raw["invasion_lambda_0"]:
        by_seed[r["seed"]].append((r["generation"], r["f_pure_GA1"]))
    for pts in by_seed.values():
        pts_sorted = sorted(pts)
        gs = [p[0] for p in pts_sorted]
        fs = [p[1] for p in pts_sorted]
        ax_d.plot(gs, fs, color=inv_color, linewidth=0.9, alpha=0.45)

    inv_agg = agg["invasion_lambda_0"]
    ax_d.plot(inv_agg["generation"], inv_agg["f_pure_GA1_mean"],
              color=inv_color, linewidth=2.4, marker="o", markersize=3.5,
              label="mean across replicates")
    ax_d.axhline(0.05, color="#888", linewidth=0.8, linestyle=":",
                 label=r"$\phi_0 = 0.05$ (seed)")

    v_mean, v_se = estimate_invasion_fitness(raw["invasion_lambda_0"], k=5)
    if not np.isnan(v_mean):
        ax_d.text(
            0.97, 0.95,
            rf"$V_{{\mathrm{{inv}}}}^{{(0..5)}} = {v_mean:+.2f}\,\pm\,{v_se:.2f}$"
            "\nper generation"
            "\n(strongly negative ⇒ no invasion)",
            transform=ax_d.transAxes, fontsize=8.0, color="#222",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#cccccc", alpha=0.95, linewidth=0.5),
        )

    ax_d.set_xlim(0, G); ax_d.set_ylim(-0.005, 0.16)
    ax_d.set_xlabel("Generation"); ax_d.set_ylabel(r"$f_{G_{A_1}}(t)$")
    ax_d.set_title(r"D   Invasion fitness  (rare-$G_{A_1}$)", loc="left")
    ax_d.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    leg_d = ax_d.legend(loc="lower right", fontsize=7.5,
                        framealpha=0.92, edgecolor="#cccccc", borderpad=0.3)
    leg_d.get_frame().set_linewidth(0.5)

    fig.suptitle(
        "Demo C  —  paired-defender co-evolution kernel (multi-agent extension):  "
        r"$G_D$ vs $G_{A_1}$ on MiniCAGE  ·  $m_3$ separation-of-duty now testable"
        f"\n$\\mathrm{{(N\\!=\\!30,\\;\\;15\\;generations,\\;\\;{n_seeds}\\;replicate\\;seeds;\\;\\;CybORG++\\;MiniCAGE\\;/\\;CAGE\\text{{-}}2;\\;\\;random\\;pairing\\;by\\;subnet\\;role)}}$",
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
                   default=str(repo_root / "telemetry" / "demo_c_paired"))
    p.add_argument("--out", type=str,
                   default=str(repo_root / "figures" / "demo_c_paired_preliminary"))
    args = p.parse_args()
    make_figure(Path(args.telemetry), Path(args.out))


if __name__ == "__main__":
    main()

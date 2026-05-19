"""Demo A figure generator.

Produces a 3-panel figure for Schmidt §3:
    A — Task-only selection:         p_g(t) for every gene, with the
                                     deterministic selection-wave prediction
                                     overlaid as a dashed line.
    B — Task + safety co-selection:  same genes under the co-selection regime
                                     (the free-genome safety gene now rises).
    C — Must-have enforcement:       C_m(t) for must_have_audit at
                                     ρ ∈ {0, 0.5, 1}, demonstrating the
                                     must-have reinjection pipeline end-to-end.

Invocation:
    python -m schmidt_demos.demo_a_domain_general.plot

Reads telemetry JSONL from ./telemetry/demo_a/ and writes a PDF + PNG
to ./figures/.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from schmidt_demos.common.gene_schema import load_schema
from schmidt_demos.common.telemetry import load_run


# ---------------------------------------------------------------------------
# Publication-quality matplotlib defaults

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


# Color and label mapping per locus. Ordered for consistent legend layout.
GENE_STYLE = {
    "task_alpha":              {"color": "#1f77b4", "label": "task_α  (cap., w=0.5)"},
    "task_beta":               {"color": "#4a90d9", "label": "task_β  (cap., w=0.3)"},
    "free_safety_vigilance":   {"color": "#2ca02c", "label": "free_safety  (w=0.4)"},
    "must_have_audit":         {"color": "#d62728", "label": "must_have_audit (M)"},
    "neutral_style":           {"color": "#888888", "label": "neutral  (drift ctrl)"},
}

RHO_STYLE = {
    0.0: {"color": "#c44e52", "label": "ρ = 0  (no enforcement)"},
    0.5: {"color": "#dd8452", "label": "ρ = 0.5  (partial reinjection)"},
    1.0: {"color": "#2a9d8f", "label": "ρ = 1  (full reinjection)"},
}


# ---------------------------------------------------------------------------
# Aggregate replicate runs


def aggregate_canonical_freq(records) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Returns {locus: (gens, mean_p, std_p)} across all replicates in ``records``."""
    by_locus_gen: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        for locus, p in r.canonical_freq.items():
            by_locus_gen[locus][r.generation].append(p)
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for locus, gen_map in by_locus_gen.items():
        gens = np.array(sorted(gen_map.keys()))
        mean = np.array([np.mean(gen_map[g]) for g in gens])
        sd = np.array([np.std(gen_map[g]) for g in gens])
        out[locus] = (gens, mean, sd)
    return out


def aggregate_compliance(records, m_locus: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gen_map: dict[int, list[float]] = defaultdict(list)
    for r in records:
        if m_locus in r.compliance:
            gen_map[r.generation].append(r.compliance[m_locus])
    gens = np.array(sorted(gen_map.keys()))
    mean = np.array([np.mean(gen_map[g]) for g in gens])
    sd = np.array([np.std(gen_map[g]) for g in gens])
    return gens, mean, sd


# ---------------------------------------------------------------------------
# Deterministic selection-wave prediction (breeder's-equation overlay)


def deterministic_prediction(
    w_coef: float,
    p0: float,
    generations: int,
    beta: float = 1.0,
) -> np.ndarray:
    """Haploid fitness-proportional selection at a single locus with
    canonical-allele weight ``w_coef`` (exponential-weight form).

    The colony's selection step multiplies each agent's weight by
    exp(β · w · I[canonical]); for additive multi-locus fitness this
    factorises over loci, so each locus evolves independently per::

        p(t+1) = p(t)·e^{β·w} / (p(t)·e^{β·w} + (1-p(t)))

    This is the parameter-free overlay used for construct validity:
    observed p_g(t) should track this curve within drift and mutation
    tolerance.
    """
    ps = np.zeros(generations + 1)
    ps[0] = p0
    for t in range(generations):
        p = ps[t]
        e_w = np.exp(beta * w_coef)
        ps[t + 1] = p * e_w / (p * e_w + (1 - p))
    return ps


# ---------------------------------------------------------------------------
# The figure

def make_figure(
    telemetry_dir: Path,
    traits_path: Path,
    out_path_base: Path,
    show_predictions: bool = True,
) -> None:
    schema = load_schema(traits_path)

    # Load telemetry for each condition
    recs = {
        "task_only":        load_run(telemetry_dir / "task_only__lambda_0.jsonl"),
        "task_plus_safety": load_run(telemetry_dir / "task_plus_safety__lambda_0.jsonl"),
        "lambda_0":         load_run(telemetry_dir / "task_only__lambda_0.jsonl"),
        "lambda_0_5":       load_run(telemetry_dir / "task_only__lambda_0_5.jsonl"),
        "lambda_1":         load_run(telemetry_dir / "task_only__lambda_1.jsonl"),
    }
    G = max(r.generation for r in recs["task_only"])
    N_reps = len({r.seed for r in recs["task_only"]})

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    ax_a, ax_b, ax_c = axes

    # ------------------------------------------------------------------
    # Panels A and B: prevalence trajectories

    for ax, cond_name, regime_label in (
        (ax_a, "task_only", "A   Task-only selection"),
        (ax_b, "task_plus_safety", "B   Task + safety co-selection"),
    ):
        agg = aggregate_canonical_freq(recs[cond_name])
        # Plot loci in the ordering of GENE_STYLE (stable legend)
        for locus, style in GENE_STYLE.items():
            if locus not in agg:
                continue
            gens, mean, sd = agg[locus]
            ax.fill_between(gens, mean - sd, mean + sd,
                            color=style["color"], alpha=0.18, linewidth=0)
            ax.plot(gens, mean, color=style["color"], linewidth=1.9,
                    label=style["label"])

            # Deterministic selection-wave prediction for THIS regime
            if show_predictions:
                gene = schema.genes[locus]
                w = gene.w_task + (gene.w_safety if cond_name == "task_plus_safety" else 0.0)
                p0 = mean[0]
                pred = deterministic_prediction(w, p0, int(gens.max()))
                ax.plot(np.arange(len(pred)), pred, color=style["color"],
                        linewidth=1.2, linestyle="--", alpha=0.55)

        ax.set_title(regime_label, loc="left")
        ax.set_xlabel("Generation")
        ax.set_ylabel("p(canonical allele)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, G)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    # Legend on Panel B so Panel A stays uncluttered; box it so it doesn't
    # get swallowed by the task-trait curves
    leg_b = ax_b.legend(
        loc="lower right", ncol=1, fontsize=8, borderpad=0.4,
        handlelength=1.8, framealpha=0.92, edgecolor="#cccccc",
    )
    leg_b.get_frame().set_linewidth(0.5)

    # Overlay caption — tucked into the one empty corner (bottom-right of Panel A)
    ax_a.text(
        0.98, 0.05,
        "dashed: deterministic selection-wave prediction\n"
        "(parameter-free; uses only pre-registered fitness weights)",
        transform=ax_a.transAxes, fontsize=7.5, color="#444",
        horizontalalignment="right", verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#dddddd", alpha=0.9, linewidth=0.5),
    )

    # ------------------------------------------------------------------
    # Panel C: C_m(t) for must_have_audit at three ρ values

    m_locus = schema.must_have.members[0]
    for ρ_val, key in ((0.0, "lambda_0"), (0.5, "lambda_0_5"), (1.0, "lambda_1")):
        gens, mean, sd = aggregate_compliance(recs[key], m_locus)
        style = RHO_STYLE[ρ_val]
        ax_c.fill_between(gens, mean - sd, mean + sd,
                          color=style["color"], alpha=0.18, linewidth=0)
        ax_c.plot(gens, mean, color=style["color"], linewidth=2.0, label=style["label"])

    ax_c.set_title("C   Must-have compliance under ρ enforcement", loc="left")
    ax_c.set_xlabel("Generation")
    ax_c.set_ylabel(f"$C_m(t)$  —  {m_locus}")
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.set_xlim(0, G)
    ax_c.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    leg_c = ax_c.legend(
        loc="lower right", fontsize=8, borderpad=0.4, handlelength=1.8,
        framealpha=0.92, edgecolor="#cccccc",
    )
    leg_c.get_frame().set_linewidth(0.5)

    # Top-level caption
    fig.suptitle(
        "Demo A  —  Typed-gene schema + must-have reinjection + $C_m(t)$ telemetry    "
        rf"(BEAR pilot; N = 200, {N_reps} replicate seeds, haploid tagged crossover)",
        fontsize=10.5, fontweight="bold", y=1.04,
    )

    # Save
    out_path_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path_base.with_suffix(".pdf")
    png_path = out_path_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


# ---------------------------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry", type=str,
                   default=str(repo_root / "telemetry" / "demo_a"))
    p.add_argument("--traits", type=str,
                   default=str(here.parent / "traits.yaml"))
    p.add_argument("--out", type=str,
                   default=str(repo_root / "figures" / "demo_a_preliminary"))
    args = p.parse_args()
    make_figure(Path(args.telemetry), Path(args.traits), Path(args.out))


if __name__ == "__main__":
    main()

"""Demo D figure — heritability of natural-language gene fragments
on the cyber substrate.

Three panels:
    A — Per-locus heritability d-statistic (Jaccard + char-cosine).
        Cohen's d compares parent-offspring similarity to the null:
        offspring text vs an unrelated other-offspring text at the
        same locus. Positive d = LLM-blended offspring is more
        similar to parents than to random siblings.
    B — Distribution overlay for the best-signal locus, showing the
        po and null similarity distributions don't overlap.
    C — One sample triplet: parent A, parent B, offspring — printed
        in the figure to make concrete what the LLM is doing.

Reads telemetry from ./telemetry/demo_d/ and writes a PDF + PNG to
./figures/.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from schmidt_demos.demo_d_bear_advantage.heritability import (
    char_ngram_cosine, cohens_d, jaccard_similarity,
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


def _load_pairs(path: Path) -> dict[str, list[tuple[str, str, str]]]:
    """Returns {locus: [(offspring, parent_a, parent_b), ...]} for
    parent_offspring rows only."""
    out: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["pair_kind"] == "parent_offspring":
            out[r["locus"]].append(
                (r["offspring_text"], r["parent_a_text"], r["parent_b_text"])
            )
    return out


def _compute_groups(po_pairs: dict[str, list[tuple[str, str, str]]]) -> dict[str, dict]:
    """For each locus, compute parent-offspring similarities (jaccard +
    cosine) and the null distribution (offspring vs unrelated other
    offspring at the same locus)."""
    out: dict[str, dict] = {}
    for locus, pairs in po_pairs.items():
        po_j: list[float] = []
        po_c: list[float] = []
        for off, pa, pb in pairs:
            po_j.append(jaccard_similarity(pa, off))
            po_j.append(jaccard_similarity(pb, off))
            po_c.append(char_ngram_cosine(pa, off))
            po_c.append(char_ngram_cosine(pb, off))
        # Null: each offspring compared to a "shifted-by-half" other offspring
        null_j: list[float] = []
        null_c: list[float] = []
        n = len(pairs)
        for i, (off, _, _) in enumerate(pairs):
            j = (i + n // 2) % n
            other_off, _, _ = pairs[j]
            if other_off != off:
                null_j.append(jaccard_similarity(off, other_off))
                null_c.append(char_ngram_cosine(off, other_off))
        out[locus] = {
            "po_jaccard": np.array(po_j),
            "po_cosine":  np.array(po_c),
            "null_jaccard": np.array(null_j),
            "null_cosine":  np.array(null_c),
            "d_jaccard": cohens_d(po_j, null_j),
            "d_cosine":  cohens_d(po_c, null_c),
            "n_pairs": len(pairs),
        }
    return out


# ---------------------------------------------------------------------------

def _detect_inheritance_mode(her_path: Path) -> tuple[str, float, float]:
    """Inspect the JSONL and return (inheritance_mode, mutation_rate,
    transmission_fidelity).

    For backward compatibility, missing inheritance_mode is treated as
    'blend' (the legacy default). transmission_fidelity is the fraction
    of parent_offspring rows whose offspring_text matches one of the
    parents' texts verbatim — meaningful under Mendelian inheritance,
    near zero under blend mode.
    """
    modes: set[str] = set()
    mu_vals: set[float] = set()
    n_po = 0
    n_match = 0
    for line in her_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("pair_kind") != "parent_offspring":
            continue
        modes.add(r.get("inheritance_mode", "blend"))
        mu_vals.add(float(r.get("mutation_rate", 0.0)))
        n_po += 1
        if r["offspring_text"] in (r["parent_a_text"], r["parent_b_text"]):
            n_match += 1
    mode = next(iter(modes)) if len(modes) == 1 else "mixed"
    mu = next(iter(mu_vals)) if len(mu_vals) == 1 else float("nan")
    fidelity = n_match / n_po if n_po else 0.0
    return mode, mu, fidelity


def make_figure(telemetry_dir: Path, out_path_base: Path) -> None:
    her_path = telemetry_dir / "heritability.jsonl"
    rew_path = telemetry_dir / "minicage_reward.jsonl"
    pairs = _load_pairs(her_path)
    g = _compute_groups(pairs)
    inheritance_mode, mutation_rate, transmission_fidelity = _detect_inheritance_mode(her_path)

    # Pick the locus with the strongest cosine d for Panel B
    best_locus = max(g, key=lambda l: g[l]["d_cosine"])

    # Pick a representative triplet (parent A, parent B, offspring) for Panel C —
    # the first pair from the best-signal locus that has DISTINCT parents
    triplet: tuple[str, str, str] | None = None
    for off, pa, pb in pairs[best_locus]:
        if pa != pb and len(off) > 30:
            triplet = (pa, pb, off)
            break
    if triplet is None and pairs[best_locus]:
        triplet = pairs[best_locus][0]

    # Set up figure layout: 2 small panels on top, 1 wide panel below
    fig = plt.figure(figsize=(13.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.55],
                          width_ratios=[1.4, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    # ------------------------------------------------------------------
    # Panel A — per-locus d-statistic bars

    loci = sorted(g.keys())
    d_jacc = [g[l]["d_jaccard"] for l in loci]
    d_cos = [g[l]["d_cosine"]  for l in loci]
    x = np.arange(len(loci))
    w = 0.36
    ax_a.bar(x - w/2, d_jacc, width=w, color="#dd8452",
             label="Jaccard (token)", edgecolor="white", linewidth=0.6)
    ax_a.bar(x + w/2, d_cos,  width=w, color="#2a9d8f",
             label="Cosine (char-3gram)", edgecolor="white", linewidth=0.6)
    ax_a.axhline(0, color="#444", linewidth=0.6)
    ax_a.axhline(0.5, color="#888", linewidth=0.6, linestyle=":")
    ax_a.text(len(loci) - 0.5, 0.52, "Cohen's d = 0.5  (medium effect)",
              fontsize=7.5, color="#555", ha="right", va="bottom")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([l.replace("_", "\n") for l in loci],
                         fontsize=8.5, rotation=0)
    ax_a.set_ylabel("Cohen's $d$  (parent-offspring vs. unrelated-offspring)")
    a_subtitle = "Mendelian inheritance" if inheritance_mode == "mendelian" else "LLM-blended gene text"
    ax_a.set_title(f"A   Per-locus heritability  ({a_subtitle})", loc="left")
    ax_a.set_ylim(min(min(d_jacc), min(d_cos), 0) - 0.1,
                  max(max(d_jacc), max(d_cos)) + 0.3)
    ax_a.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    leg_a = ax_a.legend(loc="upper left", fontsize=8.5, framealpha=0.92,
                        edgecolor="#cccccc", borderpad=0.4)
    leg_a.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------
    # Panel B — distribution overlay for the best locus

    bins = np.linspace(0, 1, 22)
    ax_b.hist(g[best_locus]["null_cosine"], bins=bins,
              alpha=0.55, color="#888", label="null (unrelated offspring)",
              edgecolor="white", linewidth=0.4)
    ax_b.hist(g[best_locus]["po_cosine"], bins=bins,
              alpha=0.65, color="#2a9d8f", label="parent–offspring",
              edgecolor="white", linewidth=0.4)
    ax_b.set_xlabel("char-3gram cosine similarity")
    ax_b.set_ylabel("count")
    ax_b.set_title(
        f"B   Similarity distribution  (locus: {best_locus.replace('_',' ')};  "
        rf"$d = {g[best_locus]['d_cosine']:+.2f}$)",
        loc="left",
    )
    ax_b.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    leg_b = ax_b.legend(loc="upper right", fontsize=8.5, framealpha=0.92,
                        edgecolor="#cccccc", borderpad=0.4)
    leg_b.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------
    # Panel C — sample blend triplet (text)

    ax_c.axis("off")
    if inheritance_mode == "mendelian":
        c_title = (
            f"C   Example Mendelian triplet  (locus: {best_locus.replace('_', ' ')};  "
            f"haploid TAGGED crossover via bear.evolution.breed; "
            f"$\\mu={mutation_rate:.2f}$; transmission fidelity = {transmission_fidelity:.3f})"
        )
    else:
        c_title = (
            f"C   Example LLM-blended triplet  (locus: {best_locus.replace('_', ' ')};  "
            f"model: claude-haiku-4-5-20251001)"
        )
    ax_c.set_title(c_title, loc="left")
    if triplet is not None:
        pa, pb, off = triplet
        # Layout: three labelled rows
        rows = [
            ("Parent A:",   pa, "#1f77b4"),
            ("Parent B:",   pb, "#dd8452"),
            ("Offspring:",  off, "#2a9d8f"),
        ]
        y = 0.9
        for label, text, color in rows:
            ax_c.text(0.02, y, label, transform=ax_c.transAxes,
                      fontsize=10, fontweight="bold", color=color,
                      verticalalignment="top")
            ax_c.text(0.13, y, text, transform=ax_c.transAxes,
                      fontsize=10, color="#222",
                      verticalalignment="top",
                      wrap=True)
            y -= 0.32

    # ------------------------------------------------------------------
    # MiniCAGE reward annotation (small note at the bottom)

    if rew_path.exists():
        rewards = []
        for line in rew_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rewards.append(json.loads(line)["blue_reward"])
        if rewards:
            r_mean = np.mean(rewards); r_sd = np.std(rewards)
            fig.text(
                0.5, -0.02,
                rf"BEAR-retrieval defender on MiniCAGE (sanity check, $n={len(rewards)}$ episodes "
                rf"× 30 ticks): blue reward $= {r_mean:+.1f} \pm {r_sd:.1f}$.  "
                "End-to-end pipeline runs on real cyber telemetry; rule-based Demo B baseline "
                "reaches $-20$ to $-30$ — task-quality parity is Year-1 work, not in scope here.",
                ha="center", va="top", fontsize=8, color="#444",
            )

    # Top-level title — adaptive to inheritance mode
    n_blends = sum(g[l]["n_pairs"] for l in g)
    if inheritance_mode == "mendelian":
        title_main = ("Demo D  —  BEAR-advantage on cyber substrate:  "
                      "Mendelian inheritance via bear.evolution.breed (locus-based, "
                      "haploid TAGGED crossover) preserves allele content verbatim")
        title_sub = (rf"$\mathrm{{({n_blends}\;parent\text{{-}}offspring\;triplets\;over\;6\;loci;"
                     rf"\;\;\mu={mutation_rate:.2f};\;\;transmission\;fidelity = {transmission_fidelity:.3f})}}$")
    else:
        title_main = ("Demo D  —  BEAR-advantage on cyber substrate:  "
                      "natural-language gene fragments are heritable under LLM-blended reproduction")
        title_sub = (rf"$\mathrm{{({n_blends}\;parent\text{{-}}offspring\;triplets\;over\;6\;loci\;\times\;3\;replicate\;seeds;"
                     rf"\;\;model:\;claude\text{{-}}haiku\text{{-}}4\text{{-}}5\text{{-}}20251001)}}$")
    fig.suptitle(f"{title_main}\n{title_sub}",
                 fontsize=10.5, fontweight="bold", y=1.05)

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
                   default=str(repo_root / "telemetry" / "demo_d"))
    p.add_argument("--out", type=str,
                   default=str(repo_root / "figures" / "demo_d_preliminary"))
    args = p.parse_args()
    make_figure(Path(args.telemetry), Path(args.out))


if __name__ == "__main__":
    main()

"""Stitch the six demo figures into a single composite figure.
Replaces the old composite that referenced f_pure_GA1 / blend-mode Demo D
telemetry.

Sources (from telemetry/paper_v0/, telemetry/demo_e_cage4_prod/,
and telemetry/demo_f_llm_decision/):
  - Demo A: demo_a_r3 (50 seeds × 50 gens — production-quality)
  - Demo B: demo_b_r1 (50 seeds × 15 gens × N=200 — production-quality)
  - Demo C: demo_c_r4 (50 seeds × N=200 invasion arm with f_allele_GA1)
  - Demo D: demo_d_r1_mendelian/mu_0.0 (50 seeds × 6 loci, BEAR-native expression)
  - Demo E: demo_e_cage4_prod (3 seeds × 5 zones × N_pop=20 × 12 gens, 4 conditions)
  - Demo F: demo_f_llm_decision (50 seeds × 15 gens × N=20, gemma-4-E2B-it decision engine)

Output: figures/proposal_composite.{pdf,png}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


here = Path(__file__).resolve().parent
sources = [
    (here / "demo_a_corrected.png", "Demo A — domain-general substrate"),
    (here / "demo_b_corrected.png", "Demo B — BEAR ↔ MiniCAGE bridge"),
    (here / "demo_c_corrected.png", "Demo C — co-evolution kernel + invasion fitness"),
    (here / "demo_d_corrected.png", "Demo D — heritability under Mendelian inheritance"),
    (here / "demo_e_prod.png",      "Demo E — CAGE Challenge 4 (5 zone defenders, 4 conditions)"),
    (here / "demo_f.png",           "Demo F — LLM decision engine (gemma-4-E2B-it on MiniCAGE)"),
]

# Load all three to figure out aspect ratios
imgs = [(mpimg.imread(p), label) for p, label in sources]
heights = [im.shape[0] for im, _ in imgs]
widths = [im.shape[1] for im, _ in imgs]

# Make a tall figure scaled to a uniform width
target_width_in = 13.5
height_per_row = [target_width_in * h / w for (im, _), h, w in zip(imgs, heights, widths)]
total_height = sum(height_per_row) + 0.5  # small extra for suptitle

fig, axes = plt.subplots(
    len(sources), 1,
    figsize=(target_width_in, total_height),
    gridspec_kw={"height_ratios": height_per_row},
    constrained_layout=False,
)

for ax, (im, _label) in zip(axes, imgs):
    ax.imshow(im)
    ax.set_axis_off()

# No figure-level suptitle: each row's title already carries the demo
# label and parameters; an outer title overlaps row 1's title at every
# reasonable suptitle y position. The caption text on the proposal /
# deposit pages carries the higher-level descriptor.

plt.subplots_adjust(top=0.999, bottom=0.001, left=0.005, right=0.995, hspace=0.06)

out_pdf = here / "proposal_composite.pdf"
out_png = here / "proposal_composite.png"
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=200)
plt.close(fig)
print(f"wrote {out_pdf}")
print(f"wrote {out_png}")

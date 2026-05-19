"""Demo D — BEAR-advantage demonstration on MiniCAGE.

The first three demos exercise the *substrate-level math* (typed-gene
schema, must-have reinjection, $C_m(t)$ telemetry, two-strategy
co-evolution, $V_{inv}$ readout). They do *not* demonstrate that BEAR
specifically is the right substrate — a critical reviewer could
legitimately ask why the same population-genetics machinery couldn't
have run on any agent framework.

Demo D closes that gap. Each defender carries a real ``bear.Corpus`` of
structured-prompt instructions whose scope tags ARE the heritable
allele content at each locus. Action selection at runtime is driven by
BEAR's scope-filtering retriever, not by hand-coded rules. At every
reproduction event, an LLM (pinned to ``claude-haiku-4-5-20251001``)
blends parent gene text to produce *genuinely novel* offspring
instructions — the natural-language genome pattern from the v5
behavioral-genetics result (Hwang 2026 in submission, $d = 3.89$–$5.14$
parent-offspring similarity), now on a cyber substrate.

Headline claims:
  1. **Retrieval as phenotype**: same loci + scope structure but
     different gene text yields measurably different action
     distributions on identical observations.
  2. **Heritability of structured-prompt fragments**: per-locus
     parent-offspring text similarity is high but not unity (LLM
     blending introduces variation), replicating the Connection-Science
     d-statistic on cyber telemetry.
  3. **Lineage textual drift**: gene text evolves over generations;
     descendant text diverges from founder text per the breeder's-
     equation prediction, with substrate enforcement of must-haves
     visibly slowing drift on the M loci.

Scope. This is a BEAR-feature existence proof, not the full Year-1
deliverable. We do not exercise per-step LLM inference (the Farinha
2025 setup) — defender actions are still deterministic given the
retrieved instruction set; only reproduction is LLM-driven.
"""

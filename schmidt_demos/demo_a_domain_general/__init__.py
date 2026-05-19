"""Demo A — domain-general pilot on BEAR infrastructure (Apr 24 – May 1, 2026).

Exercises the new machinery this proposal adds:
    - typed-gene schema
    - must-have reinjection/prune pipeline
    - C_m(t) telemetry

Produces the preliminary-results figure for Schmidt §3:
    - trait prevalence p_g(t) under task-only vs task+safety co-selection
      with breeder's-equation predictions overlaid
    - C_m(t) trace at λ ∈ {0, 0.5, 1} demonstrating must-have enforcement

Scope. This is an existence proof, not a delivery of Aim 1. The genome
is 5 binary loci with closed-form additive fitness; the grant program
extends to 12-30+ structured-prompt loci with CAGE-4-scored fitness,
LLM-driven inference per step, the full T1-T5 trait-qualification
admission protocol, and a |M| = 5 NIST/OpenC2 must-have set with full
(μ, λ) phase-diagram sweeps. See docs/DEMO_SCOPE.md (Demo A row) for
the full demo-vs-grant comparison.
"""

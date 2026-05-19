"""Demo C — two-strategy co-evolution kernel (May 10 – 17, 2026).

Seeds a single MiniCAGE defender colony with two heritable sub-
populations — G_D (canonical safety alleles on M) and a micro-instance
of G_A1 (must-have-erosion variants) — at starting fraction φ_0 = 0.5.
Both reproduce and mutate freely under the same task fitness; the
must-have layer is the only difference between conditions.

Produces the §3 figure (phase-plane sketch):
    A — Phase plane in (f_GD, f_GA1) with λ=0 vs λ=1 trajectories
        overlaid, showing visibly different fates.
    B — Time-series f_GD(t) and f_GA1(t) for both λ regimes.
    C — Invasion-fitness panel: rare-attacker run at φ_0 = 0.05,
        showing whether G_A1 invades a G_D-dominated colony at λ=0
        vs λ=1.

Scope. De-risks Aim 3's headline claim (non-empty ESS region) on a
short-horizon, single-substrate, single-attack-palette pilot. The
grant program (Aim 3, Years 2–3) extends to three pre-registered
attack palettes (G_A1, G_A2, G_A3), four seed fractions ∈ {0.01, 0.1,
0.5, 0.9}, full Maynard-Smith-&-Price ESS classification, replicator-
dynamics fits, the Rung-2 external-adversary calibration arm, and a
two-layer oversight stack (continuous substrate observation +
periodic independent auditor). Coalition-forming (G_A2) and Goodhart
(G_A3) palettes are only meaningful with ≥2 defenders, so CAGE-4 is
the headline substrate there. See docs/DEMO_SCOPE.md (Demo C row).

Reuses Demo B's defender, gene palette, and MiniCAGE bridge — the only
NEW machinery for Demo C is bimodal seeding, behavioural type
classification, and the phase-plane figure.
"""

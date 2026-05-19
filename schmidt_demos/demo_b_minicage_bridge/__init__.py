"""Demo B — BEAR <-> MiniCAGE bridge (May 1 – 10, 2026).

One defender colony (N = 30, 15 generations) operating on MiniCAGE
(CAGE-2 via CybORG++) with the typed-gene schema wired to OpenC2 action
selection and C_m(t) instrumented against the NIST/OpenC2 must-have set.

Produces the §3 figure:
    - MiniCAGE episode reward over generations, task-only vs task+safety
    - C_m(t) for every member of M under each regime
    - contrast panel showing safety-gene conservation gap

Note on the must-have set: the proposal's canonical M = {m1 audit,
m2 least-privilege, m3 separation-of-duty, m4 escalation, m5 no-alert-
suppression} targets CAGE-4's 5-defender setup. On CAGE-2's single
defender, m3 (separation-of-duty) is not testable; it is flagged N/A
in the figure and deferred to CAGE-4 in the main program. The other
four transfer directly.

Scope. Existence proof, not delivery of Aim 1/2. The bridge is built
on MiniCAGE (CAGE-2 via CybORG++) because no fast-CAGE-4 substrate
exists in the public ecosystem (Emerson 2024 documents CAGE-2 only);
the grant program retains CAGE-4 as the headline substrate with
MiniCAGE serving the Track-B fast-iteration role. The defender's
decision engine is a deterministic gene-to-action rule mapping; the
grant program swaps it for full LLM-driven inference per step
(Farinha 2025 precedent) with cross-family replication (R11). See
docs/DEMO_SCOPE.md (Demo B row) for the full demo-vs-grant comparison.
"""

"""Demo F: LLM decision-engine ablation on MiniCAGE.

Parallels Demo B's design (same locus schema, same compliance predicates,
same selection function, same fitness signal) but replaces the
deterministic gene-to-action rule mapping with per-tick action selection
by a small local LLM (gemma-4-E2B-it served by vLLM at localhost:8355).

Central claim: the monotone scaling of C_m1 with rho observed in Demo B
is not an artifact of the rule-based decision engine. It replicates
when an LLM picks each action from the gene-text directives.
"""

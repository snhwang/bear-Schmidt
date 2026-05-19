"""Heritability of natural-language gene fragments — parent-offspring
text similarity replication on the cyber substrate.

Two metrics, both deterministic (no embedding model required):

    jaccard_similarity(a, b) -- token Jaccard on lowercased word sets;
                                interpretable, range [0, 1].
    char_ngram_cosine(a, b)  -- character n-gram TF cosine; finer-
                                grained, captures word-level edits.

Cohen's d on (parent-offspring similarity) vs (random-pair similarity)
is the headline statistic.  The Connection-Science prior result
(Hwang 2026 in submission) reports per-gene d = 3.89..5.14 on
embedding-cosine; on cyber substrate with token Jaccard we expect
smaller magnitudes, but the SIGN and significance of the d-statistic
is what carries the claim.
"""

from __future__ import annotations

import re
from collections import Counter
from math import sqrt


_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text)}


def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    inter = ta & tb
    union = ta | tb
    return len(inter) / len(union) if union else 0.0


def _char_ngrams(text: str, n: int = 3) -> Counter[str]:
    s = re.sub(r"\s+", " ", text.lower().strip())
    return Counter(s[i:i + n] for i in range(len(s) - n + 1))


def char_ngram_cosine(a: str, b: str, *, n: int = 3) -> float:
    ca, cb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not ca or not cb:
        return 0.0
    keys = set(ca) | set(cb)
    dot = sum(ca[k] * cb[k] for k in keys)
    na = sqrt(sum(v * v for v in ca.values()))
    nb = sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Standardized mean difference (group_a higher = positive).

    Pooled-SD variant. Returns 0 if either group has < 2 samples.
    """
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0
    mean_a = sum(group_a) / len(group_a)
    mean_b = sum(group_b) / len(group_b)
    var_a = sum((x - mean_a) ** 2 for x in group_a) / (len(group_a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (len(group_b) - 1)
    pooled_sd = sqrt(((len(group_a) - 1) * var_a + (len(group_b) - 1) * var_b)
                     / (len(group_a) + len(group_b) - 2))
    return (mean_a - mean_b) / pooled_sd if pooled_sd > 0 else 0.0

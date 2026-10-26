"""Small statistics helpers used by every metric (no third-party dependencies)."""
from __future__ import annotations

import math

# Percentiles: linear interpolation between closest ranks (Hyndman & Fan type 7, the default of numpy and Excel
# PERCENTILE.INC). A percentile is only reported when there are enough observations for it to mean something.
PERCENTILE_METHOD = "linear interpolation between closest ranks (Hyndman-Fan type 7, as numpy.percentile default)"
MIN_SAMPLES_FOR_PERCENTILE = {50: 1, 90: 10, 95: 20, 99: 100}


def ratio(a: float, b: float) -> float | None:
    """a / b, or None when the denominator is zero (callers decide how to report that)."""
    return a / b if b else None


def percentile(values: list[float], p: float) -> float | None:
    """p in [0, 100]. Returns None for an empty list."""
    xs = sorted(values)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    h = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(h), math.ceil(h)
    return xs[lo] + (h - lo) * (xs[hi] - xs[lo])


def percentile_table(values: list[float]) -> dict[str, float | None]:
    """{"p50": .., "p90": .., "p95": .., "p99": ..} - None where there are too few observations."""
    n = len(values)
    return {f"p{p}": (percentile(values, p) if n >= need else None) for p, need in MIN_SAMPLES_FOR_PERCENTILE.items()}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion (reliable for small n). Returns (0, 0) for n = 0."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt(x: float | None, d: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{d}f}"


# ------------------------------------------------------------------ classification (shared with the CLI benchmark)
UNPARSED = "?"  # a prediction that could not be mapped to any class


def confusion(golds: list[str], preds: list[str | None], labels: list[str]) -> dict[str, dict[str, int]]:
    """matrix[true][predicted]; the extra column '?' collects predictions that are not a known class."""
    cols = labels + [UNPARSED]
    m = {t: {p: 0 for p in cols} for t in labels}
    for g, p in zip(golds, preds):
        if g in m:
            m[g][p if p in labels else UNPARSED] += 1
    return m


def per_class_counts(m: dict, labels: list[str]) -> dict[str, dict[str, int]]:
    n = sum(sum(row.values()) for row in m.values())
    out = {}
    for k in labels:
        tp = m[k][k]
        fp = sum(m[t][k] for t in labels if t != k)  # predicted k, truly something else
        fn = sum(m[k][p] for p in m[k] if p != k)  # truly k, predicted something else (incl. '?')
        out[k] = {"tp": tp, "fp": fp, "fn": fn, "tn": n - tp - fp - fn, "support": tp + fn}
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1. Zero-division policy: a 0/0 ratio is reported as 0.0 (same as scikit-learn's
    zero_division=0), which is documented next to every F1 value."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1

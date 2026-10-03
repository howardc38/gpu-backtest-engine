"""Generic grouped-return statistics. Observations are parameter combinations."""

import math

import numpy as np

_STAT_COLS = [
    "sample_size_g1",
    "sample_size_g2",
    "group1_mean",
    "group2_mean",
    "diff_mean",
    "posterior_prob_superior",
    "effect_size",
    "diff_ci_lower",
    "diff_ci_upper",
    "group2_mean_plus_diff_ci_lower",
    "group2_mean_plus_diff_ci_upper",
]


def _norm_cdf(z):
    """Evaluate the normal CDF without SciPy."""
    vec = np.vectorize(lambda v: 0.5 * (1.0 + math.erf(v / math.sqrt(2.0))))
    return vec(z)


def stats_from_groups(sum1, sumsq1, n1, total_sum, total_sumsq, total_count):
    """Compare every parameter group with all other groups using pooled variance."""
    sum1 = np.asarray(sum1, dtype=np.float64)
    sumsq1 = np.asarray(sumsq1, dtype=np.float64)
    n1 = float(n1)
    if sum1.ndim != 1 or sumsq1.shape != sum1.shape or sum1.size < 2:
        raise ValueError(
            "sum1 and sumsq1 must be matching one-dimensional arrays with at least two groups"
        )
    if not np.isfinite(sum1).all() or not np.isfinite(sumsq1).all():
        raise ValueError("Group sums must be finite")
    if not np.isfinite([n1, total_sum, total_sumsq, total_count]).all():
        raise ValueError("Group counts and grand totals must be finite")
    if not n1.is_integer() or float(total_count) != float(sum1.size) * n1:
        raise ValueError("total_count must equal number_of_groups * observations_per_group")
    if n1 <= 1 or total_count - n1 <= 1:
        raise ValueError(
            "Each effect-size comparison requires at least two observations in both groups"
        )
    if (sumsq1 < 0).any() or total_sumsq < 0:
        raise ValueError("Sum-of-squares values must be non-negative")
    mean1 = sum1 / n1
    var1 = (sumsq1 - sum1**2 / n1) / (n1 - 1)
    sum2 = total_sum - sum1
    n2 = total_count - n1
    sumsq2 = total_sumsq - sumsq1
    mean2 = sum2 / n2
    var2 = (sumsq2 - sum2**2 / n2) / (n2 - 1)
    diff_mean = mean1 - mean2
    std_error = np.sqrt(np.maximum(var1 / n1 + var2 / n2, 0.0))
    z = np.divide(diff_mean, std_error, out=np.zeros_like(diff_mean), where=std_error > 0)
    posterior = _norm_cdf(z)
    sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    effect_size = np.divide(
        diff_mean, np.sqrt(np.where(sp2 > 0, sp2, 1.0)), out=np.zeros_like(diff_mean), where=sp2 > 0
    )
    diff_ci_lower = diff_mean - 1.96 * std_error
    diff_ci_upper = diff_mean + 1.96 * std_error
    return dict(
        sample_size_g1=np.full_like(sum1, n1),
        sample_size_g2=np.full_like(sum1, n2),
        group1_mean=mean1,
        group2_mean=mean2,
        diff_mean=diff_mean,
        posterior_prob_superior=posterior,
        effect_size=effect_size,
        diff_ci_lower=diff_ci_lower,
        diff_ci_upper=diff_ci_upper,
        group2_mean_plus_diff_ci_lower=mean2 + diff_ci_lower,
        group2_mean_plus_diff_ci_upper=mean2 + diff_ci_upper,
    )

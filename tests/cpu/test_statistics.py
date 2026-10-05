import warnings

import numpy as np

from gpu_backtest.core import statistics as cs


def _effect_size_scalar(sum1, sumsq1, n1, tot_sum, tot_sumsq, tot_count):
    mean1 = sum1 / n1
    var1 = (sumsq1 - sum1**2 / n1) / (n1 - 1)
    sum2 = tot_sum - sum1
    n2 = tot_count - n1
    sumsq2 = tot_sumsq - sumsq1
    mean2 = sum2 / n2
    var2 = (sumsq2 - sum2**2 / n2) / (n2 - 1)
    diff = mean1 - mean2
    sp2 = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    return diff / np.sqrt(sp2) if sp2 > 0 else 0.0


def test_stats_from_groups_matches_scalar_formula():
    rng = np.random.default_rng(7)
    n_groups, n1 = (50, 30)
    data = rng.normal(3.0, 2.0, (n_groups, n1))
    sums = data.sum(axis=1)
    sumsqs = (data**2).sum(axis=1)
    tot_sum, tot_sumsq, tot_count = (sums.sum(), sumsqs.sum(), n_groups * n1)
    st = cs.stats_from_groups(sums, sumsqs, n1, tot_sum, tot_sumsq, tot_count)
    want = np.array(
        [
            _effect_size_scalar(sums[i], sumsqs[i], n1, tot_sum, tot_sumsq, tot_count)
            for i in range(n_groups)
        ]
    )
    np.testing.assert_allclose(st["effect_size"], want, rtol=0, atol=1e-12)


def _cohen_effect(group, rest):
    diff = group.mean() - rest.mean()
    pooled = ((len(group) - 1) * group.var(ddof=1) + (len(rest) - 1) * rest.var(ddof=1)) / (
        len(group) + len(rest) - 2
    )
    return diff / np.sqrt(pooled)


def test_effect_size_isolates_each_entry_and_exit_against_all_counterparts():
    returns = np.array([[1.0, 2.0, 4.0, 8.0], [3.0, 5.0, 7.0, 9.0], [6.0, 10.0, 11.0, 12.0]])
    total_sum = returns.sum()
    total_sumsq = (returns**2).sum()
    total_count = returns.size
    entry_stats = cs.stats_from_groups(
        returns.sum(axis=1),
        (returns**2).sum(axis=1),
        returns.shape[1],
        total_sum,
        total_sumsq,
        total_count,
    )
    expected_entry = np.array(
        [
            _cohen_effect(returns[e, :], np.delete(returns, e, axis=0).ravel())
            for e in range(returns.shape[0])
        ]
    )
    np.testing.assert_allclose(entry_stats["effect_size"], expected_entry, rtol=0, atol=1e-12)
    exit_stats = cs.stats_from_groups(
        returns.sum(axis=0),
        (returns**2).sum(axis=0),
        returns.shape[0],
        total_sum,
        total_sumsq,
        total_count,
    )
    expected_exit = np.array(
        [
            _cohen_effect(returns[:, x], np.delete(returns, x, axis=1).ravel())
            for x in range(returns.shape[1])
        ]
    )
    np.testing.assert_allclose(exit_stats["effect_size"], expected_exit, rtol=0, atol=1e-12)


def test_stats_degenerate_data_no_warning():
    n_groups, n1 = (4, 10)
    data = np.full((n_groups, n1), 0.1)
    sums = data.sum(axis=1)
    sumsqs = (data**2).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        st = cs.stats_from_groups(sums, sumsqs, n1, sums.sum(), sumsqs.sum(), n_groups * n1)
    np.testing.assert_allclose(st["effect_size"], 0.0, atol=1e-09)

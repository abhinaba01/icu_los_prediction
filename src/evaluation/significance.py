"""Statistical significance tests for model comparisons."""

from __future__ import annotations

import numpy as np
from scipy import stats


def diebold_mariano_test(hempel_errors: np.ndarray, extended_errors: np.ndarray) -> dict[str, float]:
    """Run a two-sided Diebold-Mariano test on squared error differentials.

    Args:
        hempel_errors: Absolute errors from the baseline model.
        extended_errors: Absolute errors from the extended model.

    Returns:
        Dictionary with DM statistic and p-value.
    """
    differential = np.asarray(extended_errors) ** 2 - np.asarray(hempel_errors) ** 2
    n = differential.size
    std = differential.std(ddof=1)
    if n < 2 or std == 0:
        return {"dm_stat": np.nan, "dm_p_value": np.nan}
    dm_stat = differential.mean() / std * np.sqrt(n)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(dm_stat)))
    return {"dm_stat": dm_stat, "dm_p_value": p_value}

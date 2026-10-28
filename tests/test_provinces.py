"""Checks for partial pooling and the long-run bootstrap. No downloads."""
import numpy as np
import pandas as pd

import provinces


def test_pooling_recovers_the_average_and_spread_and_shrinks_noisy_estimates_most():
    rng = np.random.default_rng(0)
    se = pd.Series(rng.uniform(0.02, 0.2, 400))
    est = -0.08 + rng.normal(0, 0.03, 400) + se * rng.normal(0, 1, 400)
    mu, mu_se, tau, shrunk, shrunk_se = provinces.pool(est, se)
    assert abs(mu + 0.08) < 3 * mu_se and abs(tau - 0.03) < 0.01
    moved = (shrunk - est).abs() / (est - mu).abs()
    assert moved[se > 0.15].mean() > moved[se < 0.05].mean() and (shrunk_se < se).all()


def test_pooling_identical_estimates_finds_no_spread():
    mu, _, tau, shrunk, _ = provinces.pool(pd.Series([-0.1, -0.1, -0.1]), pd.Series([0.05, 0.05, 0.05]))
    assert np.isclose(mu, -0.1) and tau < 1e-3 and np.allclose(shrunk, -0.1)


def test_long_run_bootstrap_range_covers_the_true_value():
    rng = np.random.default_rng(1)
    n = 60
    log_price, hdd, log_use = np.cumsum(rng.normal(0, 0.1, n)), rng.normal(4000, 300, n), np.zeros(n)
    for t in range(1, n):
        log_use[t] = 1.0 - 0.1 * log_price[t] + 0.5 * log_use[t - 1] + 0.0001 * hdd[t] + rng.normal(0, 0.01)
    y = pd.DataFrame({"log_use": log_use, "log_price": log_price, "hdd": hdd}, index=range(1960, 1960 + n))
    y["log_use_lag"], y["trend"] = y.log_use.shift(1), (y.index - 1960) / 10
    low, high = provinces.recursive_bootstrap(y.iloc[1:], draws=200)
    assert low < -0.2 < high and high - low < 0.2

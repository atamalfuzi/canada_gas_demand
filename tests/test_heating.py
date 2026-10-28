"""Checks for the heating curves, the segmented trend and the models compared with them. No downloads."""
import numpy as np
import pandas as pd

import heating


def test_heating_curve_recovers_a_known_floor_slope_and_balance_point():
    rng = np.random.default_rng(0)
    days = pd.date_range("1991-01-01", "2020-12-31")
    daily = pd.Series(8 - 12 * np.cos(2 * np.pi * days.day_of_year / 365.25) + rng.normal(0, 4, len(days)), index=days)
    below = heating.degrees_below(daily)
    use = 2.0 + 0.9 * below[15.0] + rng.normal(0, 0.1, len(below))  # m3 a day, balance point 15 °C
    c = heating.fit(use.to_numpy(), below.to_numpy())
    assert c["tau"] == 15.0 and abs(c["base"] - 2.0) < 0.1 and abs(c["slope"] - 0.9) < 0.02 and c["r2"] > 0.99


def test_segments_find_a_planted_break_and_keep_a_straight_line_whole():
    rng = np.random.default_rng(1)
    x = np.arange(1981, 2026, dtype=float)
    y = np.where(x < 1995, 8.0, 8.0 - 0.02 * (x - 1995)) + rng.normal(0, 0.005, len(x))
    cuts, criterion = heating.segments(x, y)
    assert len(cuts) == 1 and abs(x[cuts[0]] - 1995) <= 1  # at a kink, 1995 fits either segment
    for seed in range(10):  # BIC finds a break in one of these
        cuts, _ = heating.segments(x, 8.0 - 0.01 * (x - 1981) + np.random.default_rng(seed).normal(0, 0.005, len(x)))
        assert cuts == []


def test_linear_benchmark_ignores_a_reporting_level_not_yet_seen():
    months = pd.period_range("2000-01", "2019-12", freq="M")
    f = pd.DataFrame({"hdd": 300.0 + 200 * np.cos(np.arange(len(months))), "month": months.month, "year": months.year,
                      "log_price_past_year": 0.0, "since_2019": (months.year >= 2019) * 1.0})
    f["hdd_last"] = f.hdd.shift(1).bfill()
    f["y"] = 1.0 + 0.002 * f.hdd - 0.3 * f.since_2019
    train, test = f[f.year < 2019], f[f.year == 2019]
    assert np.allclose(heating.linear(train, test), 1.0 + 0.002 * test.hdd)  # the new level is unknown beforehand

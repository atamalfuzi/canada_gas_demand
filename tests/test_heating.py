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


def test_importance_is_scored_on_held_out_years_so_memorised_noise_counts_for_nothing(monkeypatch):
    rng = np.random.default_rng(2)
    months = pd.period_range("1990-01", "2015-12", freq="M")
    f = pd.concat(pd.DataFrame({
        "temp": 8 - 12 * np.cos(2 * np.pi * months.month / 12) + rng.normal(0, 3, len(months)),
        "temp_last": rng.normal(0, 1, len(months)), "cold_days": 0.0, "month": months.month, "year": months.year,
        "log_price_past_year": rng.normal(-1, 0.2, len(months)), "province": p}) for p in range(3)).reset_index(drop=True)
    f["hdd"], f["hdd_last"] = (18 - f.temp).clip(lower=0), 0.0
    f["y"] = 2 - 0.05 * f.temp + 0.3 * f.province + rng.normal(0, 0.1, len(f))  # temp_last and price do nothing

    seen, real = [], heating.permutation_importance
    monkeypatch.setattr(heating, "permutation_importance",
                        lambda model, X, y, **k: seen.append(sorted(X.year.unique())) or real(model, X, y, **k))
    _, by_year = heating.compare(f, first=2012)
    assert seen == [[y] for y in range(2012, 2016)]  # each year's rows only
    assert list(by_year.index) == list(range(2012, 2016))
    assert (by_year.year == 0).all()  # one value in each test, so nothing to shuffle

    mean = by_year.mean()
    assert mean.idxmax() == "temp" and abs(mean.temp_last) < 0.02
    model = heating.boosting().fit(f[heating.FEATURES], f.y)
    in_sample = real(model, f[heating.FEATURES], f.y, n_repeats=10, random_state=0).importances_mean
    assert in_sample[heating.FEATURES.index("temp_last")] > mean.temp_last + 0.01  # trees memorise noise they were fitted to

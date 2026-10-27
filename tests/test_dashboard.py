"""Checks for the dashboard's calculations and its checks before publishing. No downloads."""
import numpy as np
import pandas as pd

import dashboard


def frame(end="2026-08", use=100.0, price=0.3, hdd=300.0):
    months = pd.period_range("1990-01", end, freq="M")
    return pd.DataFrame({"use": use, "real_price": price, "hdd": hdd}, index=months)


def test_weather_adjustment_moves_use_to_normal_weather():
    d = frame()
    d.loc[d.index.month == 1, "hdd"] = 800.0  # every January in the 1991-2020 normal period
    d.loc["2026-01", "hdd"] = 900.0  # a colder one: 100 degree days above normal
    adjusted = dashboard.weather_adjusted(d, per_degree_day=0.001)
    assert np.isclose(adjusted["2025-01"], 100.0)
    assert np.isclose(adjusted["2026-01"], 100.0 * np.exp(-0.001 * 100))


def test_summary_compares_with_expected_a_year_ago_and_the_previous_twelve_months(monkeypatch):
    d = frame()
    d.loc["2025-09":"2026-08", "use"] = 110.0  # the last 12 months, 10% above the 12 before
    d.loc["2026-08", "use"] = 121.0
    d.loc["2025-08", "real_price"] = 0.25
    monkeypatch.setattr(dashboard, "expected_use", lambda d: (pd.Series(110.0, index=d.index), 0.0))
    s = dashboard.summary(d)
    assert s["month"] == "2026-08" and np.isclose(s["vs_expected"], 0.1)
    assert np.isclose(s["real_price_vs_year_ago"], 0.3 / 0.25 - 1)
    assert np.isclose(s["last_12_vs_previous_12"], (11 * 110 + 121) / (12 * 100) - 1, atol=1e-4)


def test_checks_pass_clean_data_and_catch_problems():
    clean = frame()
    assert dashboard.problems(clean) == []
    assert dashboard.problems(clean, published_through="2026-08") == []

    gap = clean.drop(pd.Period("2025-11", "M"))
    assert dashboard.problems(gap) == ["missing values in the last 24 months"]

    odd = clean.copy()
    odd.loc["2026-03", "use"] = 4000.0
    assert dashboard.problems(odd) == ["use per customer outside 5 to 1,500 m3 a month"]

    older = frame(end="2026-07")
    assert dashboard.problems(older, published_through="2026-08") == ["data end 2026-07 is older than the published 2026-08"]

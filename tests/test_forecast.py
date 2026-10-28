"""Checks for the forecast backtest's scoring: conformal ranges, normal weather and yearly errors. No downloads."""
import numpy as np
import pandas as pd

import forecast


def test_normal_weather_is_the_average_of_the_30_years_to_the_origin():
    months = pd.period_range("1970-01", "2012-12", freq="M")
    hdd = pd.Series(months.year.astype(float), index=months)  # each month's value is its year
    normal = forecast.normal_hdd(hdd, pd.Period("2012-12", "M"), pd.period_range("2013-01", periods=12, freq="M"))
    assert np.allclose(normal, np.mean(range(1983, 2013)))


def test_conformal_ranges_use_only_errors_known_at_the_origin():
    origins = pd.period_range("2010-01", periods=40, freq="M")
    bt = pd.DataFrame({"origin": origins, "month": origins + 1, "horizon": 1, "mean": 100.0,
                       "actual": 100.0 + np.r_[np.arange(1, 31), np.full(10, 90.0)]})  # errors 1-30%, then 90%
    e, widths = forecast.conformal(bt, level=0.8, warmup=24)
    assert e.width.iloc[:24].isna().all()  # too few earlier errors to build a range
    assert np.isclose(e.width.iloc[24], np.quantile(np.arange(1, 25) / 100, 0.8))  # the 24 errors known by then
    assert np.isclose(widths[1], np.quantile(e.abs_err, 0.8))  # the latest forecast uses all of them


def test_year_errors_compare_december_forecasts_of_the_next_calendar_year():
    rows = [{"origin": pd.Period(o, "M"), "month": pd.Period(o, "M") + h, "horizon": h, "model": "m",
             "mean": 110.0 if o == "2014-12" else 100.0, "actual": 100.0} for o in ("2013-12", "2014-12", "2015-06")
            for h in range(1, 13)]
    out = forecast.year_errors(pd.DataFrame(rows))
    assert list(out.index) == [2014, 2015] and np.allclose(out.m, [0.0, 10.0])

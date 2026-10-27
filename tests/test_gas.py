"""Checks for the data handling and models in gas.py. No downloads: every input is built here."""
import numpy as np
import pandas as pd

import gas


def statcan_rows(geo, dims, values):
    """A long-format Statistics Canada table: one row per (month, member) with a VALUE."""
    return pd.DataFrame([{"GEO": geo, "REF_DATE": month, **dims(member), "VALUE": v} for (month, member), v in values.items()])


def test_old_table_adds_utility_and_direct_sales_and_converts_cents():
    t = statcan_rows("Ontario", lambda e: {"Sector": "Residential", "Estimates": e}, {
        ("2006-12", "Sales"): 100.0, ("2006-12", "Customers"): 10.0, ("2006-12", "Sales unit price"): 30.0,
        ("2007-01", "Utilities sales"): 80.0, ("2007-01", "Direct sales"): 40.0, ("2007-01", "Customers"): 10.0,
        ("2007-01", "Sales unit price"): 32.0,
    })
    out = gas.gas_old(t).loc["Ontario"]
    assert out.m3.tolist() == [100_000.0, 120_000.0]  # thousand m3 -> m3
    assert out.price.tolist() == [0.30, 0.32]  # cents -> dollars per m3


def test_new_table_price_is_dollars_per_cubic_metre():
    t = statcan_rows("Alberta", lambda u: {"Supply and disposition": "Residential consumption", "Unit of measure": u}, {
        ("2016-01", "Cubic metres"): 200.0, ("2016-01", "Canadian dollars"): 50.0, ("2016-01", "Customers"): 5.0,
    })
    out = gas.gas_new(t).loc["Alberta"]
    assert out.m3.iloc[0] == 200_000 and out.price.iloc[0] == 0.25


def test_fix_ontario_interpolates_customers_and_blanks_the_gap():
    months = pd.period_range("1997-01", "2019-12", freq="M").astype(str)
    frame = pd.DataFrame({"m3": 1.0, "customers": 100.0, "price": 0.3},
                         index=pd.MultiIndex.from_product([["Ontario", "Alberta"], months]))
    frame.loc[("Ontario", "2015-12"), "customers"] = 300.0
    frame.loc[("Ontario", "2019-01"), "customers"] = 337.0
    fixed = gas.fix_ontario(frame)
    on = fixed.loc["Ontario"]
    assert np.isclose(on.customers["2017-07"], 300 + 37 * 19 / 37)  # straight line, Dec 2015 to Jan 2019
    assert on.loc["1998-01":"2000-12", ["m3", "customers"]].isna().all().all()
    assert on.loc["1997-12", "m3"] == 1.0 and on.loc["2001-01", "m3"] == 1.0
    assert fixed.loc["Alberta"].notna().all().all()  # other provinces untouched


def test_heating_degree_days_scale_partial_months_and_drop_thin_ones():
    days = pd.date_range("2020-01-01", "2020-03-31")
    t = pd.Series(8.0, index=days)
    t = t.drop(days[(days.month == 2) & (days.day > 14)])  # February: 14 days of readings
    t = t.drop(days[(days.month == 3) & (days.day > 13)])  # March: 13 days, too few
    t[days[(days.month == 1) & (days.day == 1)]] = 25.0  # a warm day adds no heating degrees
    hdd = gas.heating_degree_days(t)
    assert np.isclose(hdd[pd.Period("2020-01")], 30 * 10)
    assert np.isclose(hdd[pd.Period("2020-02")], 29 * 10)  # scaled to the full (leap) month
    assert pd.Period("2020-03") not in hdd.index


def monthly_frame(years, use=100.0, price=0.5):
    months = pd.period_range(f"{years[0]}-01", f"{years[-1]}-12", freq="M")
    gas_frame = pd.DataFrame({"m3": use * 10.0, "customers": 10.0, "price": price}, index=months.astype(str))
    cpi = pd.Series(100.0, index=months.astype(str))
    hdd = pd.Series(300.0, index=months)
    return gas.monthly(gas_frame, cpi, hdd, reporting_breaks=[years[-1]])


def test_past_year_price_is_the_mean_log_price_of_the_previous_twelve_months():
    d = monthly_frame([2000, 2001], price=np.exp(np.arange(24.0)))  # log real price 0, 1, 2, ... (CPI = 100)
    assert np.isnan(d.log_price_past_year.iloc[11])
    assert np.isclose(d.log_price_past_year.iloc[12], np.mean(range(12)))  # January 2001: Jan-Dec 2000
    assert np.isclose(d.log_price_past_year.iloc[23], np.mean(range(11, 23)))


def test_annual_weights_price_by_volume_and_keeps_gaps():
    d = monthly_frame([2000, 2001, 2002])
    d.loc["2000-01", ["m3", "real_price"]] = [3000.0, 1.0]  # a high-volume month at a higher price
    d.loc["2001-06", "m3"] = np.nan  # an incomplete year
    y = gas.annual(d)
    assert list(y.index) == [2000, 2001, 2002]
    assert np.isclose(y.real_price[2000], (3000 * 1.0 + 11 * 1000 * 0.5) / (3000 + 11 * 1000))
    assert np.isnan(y.use[2001])
    assert np.isnan(y.log_use_lag[2002])  # last year's use is 2001's (missing), not 2000's
    assert y.since_2002.tolist() == [0, 0, 1] and gas.breaks(y) == ["since_2002"]


def test_partial_adjustment_recovers_short_and_long_run():
    rng = np.random.default_rng(0)
    n = 400
    log_price = np.cumsum(rng.normal(0, 0.1, n))
    hdd = rng.normal(4000, 300, n)
    log_use = np.zeros(n)
    for t in range(1, n):
        log_use[t] = 1.0 - 0.2 * log_price[t] + 0.5 * log_use[t - 1] + 0.0001 * hdd[t] + rng.normal(0, 0.01)
    y = pd.DataFrame({"log_use": log_use, "log_price": log_price, "hdd": hdd}, index=range(1900, 1900 + n))
    y["log_use_lag"], y["trend"] = y.log_use.shift(1), 0.0
    est = gas.partial_adjustment(y.iloc[1:], trend=False)
    assert abs(est.short_run + 0.2) < 0.02 and abs(est.adjustment - 0.5) < 0.05
    assert abs(est.long_run + 0.4) < 0.05 and est.long_run_se > 0


def test_monthly_model_recovers_an_elasticity_on_past_year_prices():
    rng = np.random.default_rng(1)
    months = pd.period_range("1990-01", "2019-12", freq="M")
    d = pd.DataFrame(index=months)
    d["log_price"] = np.cumsum(rng.normal(0, 0.03, len(d)))
    d["log_price_past_year"] = d.log_price.shift(1).rolling(12).mean()
    d["hdd"] = 300 + 250 * np.cos(2 * np.pi * (months.month - 1) / 12) + rng.normal(0, 30, len(d))
    d["use"] = np.exp(5 - 0.15 * d.log_price_past_year + 0.002 * d.hdd + rng.normal(0, 0.02, len(d)))
    fit = gas.monthly_model(d, "log_price_past_year")
    assert abs(fit.params.log_price_past_year + 0.15) < 0.03


def test_time_varying_elasticity_tracks_a_drifting_elasticity_and_is_detected():
    rng = np.random.default_rng(2)
    n = 120
    true = np.linspace(-0.3, 0.0, n)
    log_price = np.cumsum(rng.normal(0, 0.15, n))
    hdd = rng.normal(4000, 300, n)
    log_use = np.zeros(n)
    for t in range(1, n):
        log_use[t] = 2.0 + true[t] * log_price[t] + 0.3 * log_use[t - 1] + 0.0001 * hdd[t] + rng.normal(0, 0.01)
    y = pd.DataFrame({"log_use": log_use, "log_price": log_price, "hdd": hdd}, index=range(1900, 1900 + n))
    y["log_use_lag"], y["trend"] = y.log_use.shift(1), (y.index - 1900) / 10
    path, lr, pvalue = gas.time_varying(y)
    error = (path.elasticity - true[1:]).abs()
    assert error.mean() < 0.05 and pvalue < 0.01

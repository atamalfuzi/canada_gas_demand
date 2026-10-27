"""Residential natural gas demand in British Columbia, Alberta and Ontario: how strongly do households
respond to price, how does that differ between the short and the long run, and has it changed since 1980?

Data: Statistics Canada monthly natural gas sales and deliveries (tables 25-10-0033-01 to 2015 and
25-10-0086-01 from 2016), provincial consumer price index (18-10-0004-01), and daily mean temperatures from
the Government of Canada's climate data service.

Run from the repository folder: python gas.py
"""
from pathlib import Path
import io
import time
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.statespace.mlemodel import MLEModel

DATA, FIGURES, OUTPUT = Path("data"), Path("figures"), Path("output")
PROVINCES = ["British Columbia", "Ontario", "Alberta"]
START = "1980-01"
ALBERTA_END = "2014-10"  # from November 2014 Alberta's reported price covers only the gas commodity


def cached(url, name, tries=3, wait=30):
    """Download once into data/; delete the file to download it again. A failed download (the climate service
    sometimes times out) is retried after 30 and then 60 seconds."""
    path = DATA / name
    if not path.exists():
        DATA.mkdir(exist_ok=True)
        for attempt in range(1, tries + 1):
            try:
                with urllib.request.urlopen(url, timeout=120) as response:
                    body = response.read()  # read in full before writing, so a failure leaves no partial file
                break
            except OSError as error:  # timeouts, refused connections and HTTP errors
                if attempt == tries:
                    raise
                print(f"Download of {name} failed ({error}); retrying in {wait * attempt} seconds")
                time.sleep(wait * attempt)
        path.write_bytes(body)
    return path


def statcan(pid):
    """A full Statistics Canada table, from its CSV download (the web service is sometimes down for maintenance)."""
    with zipfile.ZipFile(cached(f"https://www150.statcan.gc.ca/n1/tbl/csv/{pid}-eng.zip", f"{pid}.zip")) as z:
        name = next(n for n in z.namelist() if not n.endswith("MetaData.csv"))
        return pd.read_csv(io.BytesIO(z.read(name)), low_memory=False)


# Data: gas ----
def gas_old(table):
    """1980-2015: sales (split into utility and direct sales from 2007), customers and the published unit price."""
    t = table[table.GEO.isin(PROVINCES) & (table.Sector == "Residential") & (table.REF_DATE >= START)]
    t = t.pivot_table(index=["GEO", "REF_DATE"], columns="Estimates", values="VALUE", aggfunc="first")
    volume = t[["Sales", "Utilities sales", "Direct sales"]].fillna(0).sum(axis=1)  # thousand m3
    return pd.DataFrame({"m3": volume * 1e3, "customers": t.Customers, "price": t["Sales unit price"] / 100})


def gas_new(table):
    """2016 onward: residential consumption in cubic metres and dollars, and customers."""
    t = table[table.GEO.isin(PROVINCES) & (table["Supply and disposition"] == "Residential consumption")]
    t = t.pivot_table(index=["GEO", "REF_DATE"], columns="Unit of measure", values="VALUE", aggfunc="first")
    return pd.DataFrame({"m3": t["Cubic metres"] * 1e3, "customers": t.Customers,
                         "price": t["Canadian dollars"] / t["Cubic metres"]})


def fix_ontario(gas):
    """The 2016-2018 Ontario customer counts cover only part of the province (2.0 million, against 3.3 million
    in December 2015 and 3.4 million from January 2019) while volumes are complete, so those three years are
    filled by straight-line interpolation between December 2015 and January 2019."""
    gas = gas.copy()
    on = gas.loc["Ontario", "customers"].copy()
    on[(on.index >= "2016-01") & (on.index <= "2018-12")] = np.nan
    gas.loc["Ontario", "customers"] = on.interpolate().to_numpy()
    # 1998-2000, as Ontario's retail gas market opened, the table covers only part of the volume (down a quarter
    # in 1998-1999) and of the customers (up a quarter between 2000 and 2001); neither can be repaired
    gap = gas.index.get_level_values(0).isin(["Ontario"]) & (gas.index.get_level_values(1) >= "1998-01") & (gas.index.get_level_values(1) <= "2000-12")
    gas.loc[gap, ["m3", "customers"]] = np.nan
    return gas


# Data: prices ----
def cpi(table):
    """All-items consumer price index by province, monthly (2002 = 100)."""
    t = table[table.GEO.isin(PROVINCES) & (table["Products and product groups"] == "All-items") & (table.REF_DATE >= START)]
    return t.set_index(["GEO", "REF_DATE"]).VALUE


# Data: weather ----
# One airport per province; each station was replaced in 2012-2013, so two IDs (newest first)
STATIONS = {
    "British Columbia": ("1108395", "1108447"),  # Vancouver
    "Ontario": ("6158731", "6158733"),  # Toronto
    "Alberta": ("3031092", "3031093"),  # Calgary
}


def temperature(climate_id):
    # The service returns at most 10,000 days per request, so two requests per station
    parts = []
    for start, end in [("1979-12-01", "2003-12-31"), ("2004-01-01", "2026-12-31")]:
        url = ("https://api.weather.gc.ca/collections/climate-daily/items?f=csv&sortby=LOCAL_DATE&limit=10000"
               f"&CLIMATE_IDENTIFIER={climate_id}&datetime={start}/{end}&properties=LOCAL_DATE,MEAN_TEMPERATURE")
        path = cached(url, f"climate_{climate_id}_{start[:4]}.csv")
        if path.stat().st_size > 0:  # empty when the station was not open in that window
            parts.append(pd.read_csv(path))
    t = pd.concat(parts)
    return t.set_index(pd.to_datetime(t.LOCAL_DATE).dt.normalize()).MEAN_TEMPERATURE.dropna()


def heating_degree_days(daily):
    """Monthly heating degree days below 18 C, from months with at least two weeks of readings, scaled up to the
    full month (a few months in 1993 and 2012-2013 have gaps)."""
    hdd = (18 - daily).clip(lower=0)
    month = hdd.groupby(hdd.index.to_period("M"))
    days_in_month = month.size().index.days_in_month
    return (month.mean() * days_in_month)[month.size() >= 14]


# Series ----
# Years from which the tables report differently, so each period gets its own level: the 2016 switch to the
# newer table, and Ontario's coverage change after its 1998-2000 gap (customers up a quarter in 2001)
REPORTING_BREAKS = {"British Columbia": [2016], "Ontario": [2001, 2016], "Alberta": []}


def breaks(frame):
    """The reporting-break columns that vary within this sample."""
    return [c for c in frame if c.startswith("since_") and 0 < frame[c].mean() < 1]


def monthly(gas, index, hdd, reporting_breaks=()):
    """One province's monthly series: use per customer, real price, heating degree days."""
    d = gas.copy()
    d.index = pd.PeriodIndex(d.index, freq="M")
    index = index.copy()
    index.index = pd.PeriodIndex(index.index, freq="M")
    d["real_price"] = d.price / index * 100  # 2002 dollars per m3
    d["hdd"] = hdd.reindex(d.index).interpolate(limit=2, limit_area="inside")  # a few months lack enough days
    d["use"] = d.m3 / d.customers
    d["log_price"] = np.log(d.real_price)
    # Households see last month's bill, and the average price falls in cold months as fixed charges spread
    # over more gas: the average over the previous 12 months is free of both problems
    d["log_price_past_year"] = d.log_price.shift(1).rolling(12).mean()
    for year in reporting_breaks:
        d[f"since_{year}"] = (d.index.year >= year) * 1.0
    return d


def annual(d):
    """Full calendar years: use per customer, the volume-weighted real price, heating degree days. Years with
    missing data stay in as gaps; the models drop what they cannot use."""
    g = d.assign(spend=d.real_price * d.m3).groupby(d.index.year)
    y = pd.DataFrame({
        "use": g.m3.sum(min_count=12) / g.customers.mean(),
        "real_price": g.spend.sum(min_count=12) / g.m3.sum(min_count=12),
        "hdd": g.hdd.sum(min_count=12),
        **{c: g[c].max() for c in d if c.startswith("since_")},
    })[g.size() == 12]
    y = y.reindex(range(y.index.min(), y.index.max() + 1))  # keep gaps as gaps, so last year's use is last year's
    y["log_use"], y["log_price"] = np.log(y.use), np.log(y.real_price)
    y["log_use_lag"] = y.log_use.shift(1)
    y["trend"] = (y.index - 1980) / 10  # per decade
    return y.rename_axis("year")


MODEL_INPUTS = ["log_use", "log_price", "log_use_lag", "hdd"]


# Models ----
def monthly_model(d, price):
    """Log use per customer on a log real price, heating degree days, calendar month, a linear trend and the
    reporting breaks, with standard errors robust to a year of autocorrelation."""
    d = d.dropna(subset=["use", price, "hdd"])
    month = pd.get_dummies(d.index.month, prefix="month", drop_first=True, dtype=float).set_index(d.index)
    X = pd.concat([d[price], d.hdd / 100, month, pd.Series(np.arange(len(d)) / 120, index=d.index, name="trend"),
                   d[breaks(d)]], axis=1)
    return sm.OLS(np.log(d.use), sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 12})


def partial_adjustment(y, trend=True):
    """Annual partial adjustment: use moves part of the way to its long-run level each year, so the short-run
    elasticity is b and the long-run one b / (1 - adjustment), with a delta-method standard error."""
    y = y.dropna(subset=MODEL_INPUTS)
    cols = ["log_price", "log_use_lag", "hdd"] + (["trend"] if trend else []) + breaks(y)
    fit = sm.OLS(y.log_use, sm.add_constant(y[cols].assign(hdd=y.hdd / 1000))).fit()
    b, lag = fit.params.log_price, fit.params.log_use_lag
    grad = pd.Series(0.0, index=fit.params.index)
    grad["log_price"], grad["log_use_lag"] = 1 / (1 - lag), b / (1 - lag) ** 2
    return pd.Series({"short_run": b, "short_run_se": fit.bse.log_price, "adjustment": lag,
                      "long_run": b / (1 - lag), "long_run_se": float(np.sqrt(grad @ fit.cov_params() @ grad))})


class TimeVaryingElasticity(MLEModel):
    """Annual partial adjustment in which the short-run price elasticity is a random walk: log use = elasticity_t
    x log price + fixed effects (constant, last year's log use, heating degree days, trend, reporting breaks)
    + noise. The Kalman smoother traces the elasticity; its variance and the noise variance are estimated by
    maximum likelihood."""

    def __init__(self, log_use, log_price, fixed):
        super().__init__(np.asarray(log_use, float), k_states=1 + fixed.shape[1], k_posdef=1)
        design = np.zeros((1, self.k_states, self.nobs))
        design[0, 0] = np.asarray(log_price, float)
        design[0, 1:] = np.asarray(fixed, float).T
        self["design"] = design
        self["transition"] = np.eye(self.k_states)
        self["selection"] = np.eye(self.k_states)[:, :1]
        self.ssm.initialize_approximate_diffuse(1e4)
        self.loglikelihood_burn = self.k_states

    @property
    def param_names(self):
        return ["noise", "elasticity"]

    @property
    def start_params(self):
        return np.array([1e-3, 1e-4])

    def transform_params(self, unconstrained):
        return unconstrained ** 2

    def untransform_params(self, constrained):
        return constrained ** 0.5

    def update(self, params, **kwargs):
        noise, elasticity = super().update(params, **kwargs)
        self["obs_cov", 0, 0] = noise
        self["state_cov", 0, 0] = elasticity


def time_varying(y):
    """The smoothed elasticity with a 90% band, and a likelihood-ratio test against a constant elasticity
    (the variance is on the boundary, so the p-value is half the chi-squared one)."""
    # Years without all inputs (Ontario 1998-2001) stay in as missing observations, so the elasticity drifts
    # across them
    usable = y[MODEL_INPUTS].notna().all(axis=1)
    y = y.loc[usable.idxmax():usable[::-1].idxmax()]
    usable = usable.reindex(y.index)
    fixed = sm.add_constant(y[["log_use_lag", "hdd", "trend"] + breaks(y)]
                            .assign(hdd=y.hdd / 1000), has_constant="add").fillna(0)
    model = TimeVaryingElasticity(y.log_use.where(usable), y.log_price.fillna(0), fixed)
    fit = model.fit(disp=False, maxiter=1000)
    constant = model.fit_constrained({"elasticity": 0.0}, disp=False, maxiter=1000)
    sd = fit.smoothed_state_cov[0, 0] ** 0.5
    path = pd.DataFrame({"elasticity": fit.smoothed_state[0], "low": fit.smoothed_state[0] - 1.645 * sd,
                         "high": fit.smoothed_state[0] + 1.645 * sd}, index=y.index)
    lr = max(2 * (fit.llf - constant.llf), 0)
    return path, lr, 0.5 * stats.chi2.sf(lr, 1)


def main():
    for folder in (DATA, FIGURES, OUTPUT):
        folder.mkdir(exist_ok=True)

    gas = fix_ontario(pd.concat([gas_old(statcan("25100033")), gas_new(statcan("25100086"))]).sort_index())
    prices = cpi(statcan("18100004"))
    series, years = {}, {}
    for p in PROVINCES:
        new, old = STATIONS[p]
        hdd = heating_degree_days(temperature(new).combine_first(temperature(old)))
        d = monthly(gas.loc[p], prices.loc[p], hdd, REPORTING_BREAKS[p]).loc["1981-01":]
        series[p] = d.loc[:ALBERTA_END] if p == "Alberta" else d
        years[p] = annual(series[p])

    # Estimates ----
    rows, paths = [], {}
    for p in PROVINCES:
        naive, corrected = monthly_model(series[p], "log_price"), monthly_model(series[p], "log_price_past_year")
        with_trend, without_trend = partial_adjustment(years[p]), partial_adjustment(years[p], trend=False)
        paths[p], lr, pvalue = time_varying(years[p])
        rows.append({
            "province": p, "months": int(naive.nobs), "years": int(years[p][MODEL_INPUTS].notna().all(axis=1).sum()),
            "monthly_naive": naive.params.log_price, "monthly_naive_se": naive.bse.log_price,
            "monthly_past_year_price": corrected.params.log_price_past_year, "monthly_past_year_price_se": corrected.bse.log_price_past_year,
            **with_trend.add_prefix("annual_"),
            "annual_long_run_no_trend": without_trend.long_run, "annual_long_run_no_trend_se": without_trend.long_run_se,
            "time_varying_lr_stat": lr, "time_varying_p_value": pvalue,
        })
    results = pd.DataFrame(rows).set_index("province")
    results.round(4).to_csv(OUTPUT / "elasticities.csv")
    pd.concat(paths, names=["province"]).round(4).to_csv(OUTPUT / "elasticity_over_time.csv")
    pd.concat({p: y[["use", "real_price", "hdd"]] for p, y in years.items()}, names=["province"]).round(4).to_csv(OUTPUT / "annual.csv")
    print(results.round(3).T.to_string())

    # Figures ----
    INK, MUTED, GREY = "#0b0b0b", "#52514e", "#a8a7a2"
    COLOURS = {"British Columbia": "#2a78d6", "Ontario": "#eb6834", "Alberta": "#3a9b6a"}
    SOURCE = ("Source: Statistics Canada tables 25-10-0033-01, 25-10-0086-01 and 18-10-0004-01; "
              "Government of Canada daily climate data.")
    plt.rcParams.update({
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
        "axes.grid": True, "grid.color": "#e6e5e1", "axes.axisbelow": True, "legend.frameon": False,
    })

    def save(fig, name, title, subtitle, caption):
        fig.text(0.01, 0.99, title, fontsize=15, fontweight="bold", color=INK, va="top")
        fig.text(0.01, 0.93, subtitle, color=MUTED, va="top")
        fig.text(0.01, 0.01, caption, color=MUTED, fontsize=9, va="bottom")
        fig.savefig(FIGURES / name, dpi=200, facecolor="white")
        plt.close(fig)

    # Figure use_and_price ----
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(top=0.80, bottom=0.2, wspace=0.22)
    for p, y in years.items():
        y = y.reindex(range(y.index.min(), y.index.max() + 1))  # gaps show as gaps
        left.plot(y.index, y.use, color=COLOURS[p], lw=2, label=p)
        right.plot(y.index, y.real_price * 100, color=COLOURS[p], lw=2, label=p)
    left.set_title("Gas use per customer (m³ per year)", loc="left", color=INK, fontsize=11)
    right.set_title("Average price, cents per m³ (2002 dollars)", loc="left", color=INK, fontsize=11)
    left.set_ylim(0, None)
    right.set_ylim(0, None)
    left.legend(loc="lower left")
    bc = years["British Columbia"]
    save(fig, "use_and_price.png",
         f"BC households use {1 - bc.use.iloc[-1] / bc.use.iloc[0]:.0%} less gas than in {bc.index[0]}, "
         "while real prices have swung widely",
         f"Residential natural gas, full calendar years (Alberta to {years['Alberta'].index.max()}, before its reported price "
         "changes definition; Ontario's 1998-2000 data are incomplete)",
         "Use is volume over the year's average customer count; price is revenue per m³ over the year, deflated by provincial CPI.\n" + SOURCE)

    # Figure two_traps ----
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    fig.subplots_adjust(top=0.78, bottom=0.2, wspace=0.08, left=0.14)
    rows_y = np.arange(len(PROVINCES))[::-1]
    for ax, title, pairs in [
        (left, "Trap 1: the month's average price (short run)",
         [("monthly_naive", "This month's price", GREY), ("monthly_past_year_price", "Past year's price", INK)]),
        (right, "Trap 2: no efficiency trend (long run)",
         [("annual_long_run_no_trend", "Without a trend", GREY), ("annual_long_run", "With a trend", INK)]),
    ]:
        for k, (col, label, colour) in enumerate(pairs):
            offset = 0.12 if k == 0 else -0.12
            est, se = results[col], results[f"{col}_se"]
            ax.errorbar(est, rows_y + offset, xerr=1.645 * se, fmt="o", color=colour, ms=7, capsize=0, lw=2, label=label)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("% change in use per 1% higher real price (90% interval)")
        ax.legend(loc="lower left")
    left.set_yticks(rows_y, PROVINCES)
    bc = results.loc["British Columbia"]
    save(fig, "two_traps.png",
         "Two traps make BC households look more responsive to gas prices than they are",
         f"In BC, the month's average price overstates the short-run response by {bc.monthly_naive / bc.monthly_past_year_price - 1:.0%}; "
         f"leaving out efficiency gains overstates the long run {bc.annual_long_run_no_trend / bc.annual_long_run:.1f} times",
         "Short run: monthly regressions with heating degree days, month and trend. Long run: annual partial adjustment.\n" + SOURCE)

    # Figure elasticity_over_time ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.8), sharey=True)
    fig.subplots_adjust(top=0.75, bottom=0.19, wspace=0.08)
    for ax, p in zip(axes, PROVINCES):
        path = paths[p]
        ax.fill_between(path.index, path.low, path.high, color=COLOURS[p], alpha=0.15, lw=0, label="90% band")
        ax.plot(path.index, path.elasticity, color=COLOURS[p], lw=2.5, label="Changing year by year")
        ax.axhline(results.loc[p, "annual_short_run"], color=GREY, lw=2, ls="--", label="Constant")
        ax.axhline(0, color=MUTED, lw=1)
        pvalue = results.loc[p, "time_varying_p_value"]
        ax.set_title(p, loc="left", color=INK, fontsize=12, pad=18)
        ax.text(0, 1.02, "Test of change over time: " + ("p < 0.001" if pvalue < 0.001 else f"p = {pvalue:.3f}"),
                transform=ax.transAxes, color=MUTED, fontsize=9.5)
    axes[0].legend(loc="lower left")
    axes[0].set_ylabel("Short-run price elasticity")
    strongest = paths["British Columbia"].elasticity.idxmin()
    save(fig, "elasticity_over_time.png",
         f"BC households responded most to gas prices around {strongest}; the response has faded since",
         "Short-run price elasticity of gas use per customer, estimated year by year with a Kalman filter",
         "Annual partial-adjustment model with heating degree days and an efficiency trend; the elasticity follows a random walk.\n" + SOURCE)


if __name__ == "__main__":
    main()

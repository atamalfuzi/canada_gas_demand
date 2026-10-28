"""Where did the fall in gas use come from, and what does machine learning see?

Heating curves: each month's gas use per customer per day is a floor of year-round use (water heating, cooking,
dryers) plus heating that rises with each degree below a balance point (a PRISM-style model). Fitted for each decade in
six provinces, the curves split the fall in normal-weather use into year-round use and space heating, with 90% ranges
from a bootstrap by year. A segmented trend, with the number of breaks chosen by an information criterion, dates the
start of the decline.
Then gradient boosting on the same months: does it predict better than a linear model, and does it find the heating
curve and the price response?

Run from the repository folder, after provinces.py (the last figure shows its pooled elasticity): python heating.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LinearRegression

import gas
from gas import BLUE, GREY, INK, MUTED, ORANGE

PERIODS = [(1981, 1990), (1991, 2000), (2001, 2010), (2011, 2019), (2020, None)]
TAUS = np.arange(10.0, 22.01, 0.5)  # balance points tried, °C
MIN_YEARS = 6  # the shortest trend segment


# Heating curves ----
def degrees_below(daily):
    """Each month's mean heating degrees per day below each balance point, from months with two weeks of readings."""
    month = daily.index.to_period("M")
    out = pd.DataFrame({t: (t - daily).clip(lower=0).groupby(month).mean() for t in TAUS})
    return out[daily.groupby(month).size() >= 14]


def fit(use, below):
    """Use per customer per day = base + slope x degrees per day below the balance point, with the balance point
    that fits best (profile least squares). `use` is an array of months, `below` their rows of degrees_below."""
    x, u = below - below.mean(axis=0), use - use.mean()
    slope = x.T @ u / (x ** 2).sum(axis=0)
    ssr = ((u[:, None] - x * slope) ** 2).sum(axis=0)
    k = int(np.argmin(ssr))
    return {"tau": TAUS[k], "base": use.mean() - slope[k] * below[:, k].mean(), "slope": slope[k],
            "r2": 1 - ssr[k] / (u ** 2).sum()}


def normal_year(below, tau):
    """Heating degrees below the balance point in a year of 1991-2020 average weather."""
    h = below.loc["1991-01":"2020-12", tau]
    return float((h * h.index.days_in_month).groupby(h.index.month).mean().sum())


def periods(per_day):
    """Each period with at least five years of data: its index, label and months."""
    for k, (a, b) in enumerate(PERIODS):
        u = per_day.loc[str(a):str(b) if b else None].dropna()
        if u.index.year.nunique() >= 5:
            yield k, f"{a}-{b}" if b else f"since {a}", u


def heating_curves(d, daily, draws=200, seed=0):
    """Each period's curve, and its normal-weather year split into year-round use and heating (m3 per customer),
    with 90% ranges from refitting to whole years drawn with replacement."""
    below = degrees_below(daily)
    normals = {t: normal_year(below, t) for t in TAUS}
    per_day = d.use / d.index.days_in_month
    per_day = per_day[per_day.index.isin(below.index)]
    rng, rows = np.random.default_rng(seed), []
    for k, label, u in periods(per_day):
        use, deg, years = u.to_numpy(), below.loc[u.index].to_numpy(), u.index.year.to_numpy()
        c = fit(use, deg)
        split = [365.25 * c["base"], c["slope"] * normals[c["tau"]]]
        boot = []
        for _ in range(draws):
            pick = np.concatenate([np.flatnonzero(years == y) for y in rng.choice(np.unique(years), len(np.unique(years)))])
            cb = fit(use[pick], deg[pick])
            boot.append([365.25 * cb["base"], cb["slope"] * normals[cb["tau"]]])
        boot = np.array(boot)
        low, high = np.percentile(np.c_[boot, boot.sum(axis=1)], [5, 95], axis=0)
        rows.append({"period": label, "months": len(u), **c, "year_round": split[0], "heating": split[1], "total": sum(split),
                     "year_round_low_90": low[0], "year_round_high_90": high[0], "heating_low_90": low[1],
                     "heating_high_90": high[1], "total_low_90": low[2], "total_high_90": high[2]})
    return pd.DataFrame(rows)


def chart_data(series, temperatures):
    """For the heating-curve explorer: each month's mean temperature and use per customer per day with its period,
    and each period's fitted curve (fitted values averaged in 1 °C bins of the month's mean temperature, since the
    model works on daily degrees) and normal-weather year (m3 per customer)."""
    out = {}
    for p, d in series.items():
        daily = temperatures[p]
        below, temp = degrees_below(daily), daily.groupby(daily.index.to_period("M")).mean()
        per_day = d.use / d.index.days_in_month
        per_day = per_day[per_day.index.isin(below.index)]
        points, curves = [], []
        for k, label, u in periods(per_day):
            c = fit(u.to_numpy(), below.loc[u.index].to_numpy())
            fitted = c["base"] + c["slope"] * below.loc[u.index, c["tau"]]
            bins = fitted.groupby(temp.loc[u.index].round()).mean()
            year_round = 365.25 * c["base"]
            curves.append({"period": k, "label": label, "tau": c["tau"], "base_year": round(year_round),
                           "normal": round(year_round + c["slope"] * normal_year(below, c["tau"])),
                           "x": bins.index.astype(int).tolist(), "y": bins.round(2).tolist()})
            points += [[m.year, m.month, round(float(temp[m]), 1), round(float(u[m]), 2), k] for m in u.index]
        out[p] = {"points": points, "curves": curves}
    return out


# When the decline began ----
def normal_weather_years(d):
    """Each full year's log use per customer in 1991-2020 normal weather (the monthly model's heating effect)."""
    model = gas.monthly_model(d, "log_price_past_year")
    adjusted = gas.weather_adjusted(d, model.params.hdd / 100)
    return np.log(adjusted.groupby(adjusted.index.year).sum(min_count=12)).dropna()


def segment_costs(x, y):
    """The least-squares error of a straight line through observations i to j - 1, for every i and j, from
    cumulative sums; infinite for runs shorter than MIN_YEARS."""
    sums = [np.r_[0, np.cumsum(v)] for v in (np.ones_like(x), x, y, x * x, x * y, y * y)]
    i, j = np.arange(len(x) + 1)[:, None], np.arange(len(x) + 1)[None, :]
    n, sx, sy, sxx, sxy, syy = (s[j] - s[i] for s in sums)
    with np.errstate(divide="ignore", invalid="ignore"):
        cost = syy - sy ** 2 / n - (sxy - sx * sy / n) ** 2 / (sxx - sx ** 2 / n)
    return np.where(n >= MIN_YEARS, cost, np.inf)


def segments(x, y, max_breaks=4, breaks=None):
    """Straight-line segments, each with its own level and slope, fitted exactly by dynamic programming. The number of
    breaks is chosen unless given, by the LWZ criterion (Liu, Wu and Zidek 1997): a Schwarz criterion with a heavier
    penalty, since BIC finds breaks in straight lines too often when it also picks where they go. Returns the position
    where each new segment starts, and the criterion by number of breaks."""
    n, cost = len(y), segment_costs(x - x.mean(), y)
    best, start = [cost[0]], []  # best[k][j]: the least error splitting the first j observations into k + 1 segments
    for _ in range(max_breaks):
        total = best[-1][:, None] + cost
        start.append(total.argmin(axis=0))
        best.append(total.min(axis=0))
    lwz = {k: n * np.log(best[k][n] / (n - 3 * k - 2)) + (3 * k + 2) * 0.299 * np.log(n) ** 2.1
           for k in range(max_breaks + 1) if np.isfinite(best[k][n])}  # 3k + 2: two coefficients a segment, k breaks
    cuts, j = [], n
    for k in range(min(lwz, key=lwz.get) if breaks is None else breaks, 0, -1):
        j = start[k - 1][j]
        cuts.append(int(j))
    return cuts[::-1], lwz


def trend(s, draws=500, seed=0):
    """One province's segmented trend: each segment's years and yearly change, with 90% ranges for the year each new
    segment starts from a residual bootstrap holding the number of breaks fixed."""
    x, y = s.index.to_numpy(float), s.to_numpy()
    cuts, criterion = segments(x, y)
    bounds = list(zip([0, *cuts], [*cuts, len(y)]))
    fitted = np.concatenate([np.polyval(np.polyfit(x[a:b], y[a:b], 1), x[a:b]) for a, b in bounds])
    rng = np.random.default_rng(seed)
    starts = np.array([x[segments(x, fitted + rng.choice(y - fitted, len(y)), breaks=len(cuts))[0]] for _ in range(draws)])
    rows = []
    for k, (a, b) in enumerate(bounds):
        row = {"from": int(x[a]), "to": int(x[b - 1]), "change_pct_a_year": np.expm1(np.polyfit(x[a:b], y[a:b], 1)[0]) * 100}
        if k:
            row["from_low_90"], row["from_high_90"] = np.percentile(starts[:, k - 1], [5, 95], method="nearest")
        rows.append(row)
    return pd.DataFrame(rows), criterion


# Machine learning ----
FEATURES = ["temp", "temp_last", "cold_days", "month", "year", "log_price_past_year", "province"]


def learning_frame(series, temperatures):
    """One row per province and month: weather, calendar, the past year's price and the province (as a code); the
    linear model's inputs; and log use per customer per day."""
    frames = []
    for p, d in series.items():
        daily = temperatures[p]
        g = daily.groupby(daily.index.to_period("M"))
        f = pd.DataFrame({"temp": g.mean(), "cold_days": g.apply(lambda v: (v < -15).mean())})[g.size() >= 14].reindex(d.index)
        f = f.assign(temp_last=f.temp.shift(1), month=d.index.month, year=d.index.year, province=gas.SIX.index(p),
                     log_price_past_year=d.log_price_past_year, hdd=d.hdd, hdd_last=d.hdd.shift(1),
                     y=np.log(d.use / d.index.days_in_month), **{c: d[c] for c in gas.breaks(d)})
        frames.append(f.dropna(subset=["y", "temp", "temp_last", "log_price_past_year", "hdd_last"]))
    out = pd.concat(frames).reset_index(drop=True)
    return out.fillna({c: 0.0 for c in out if c.startswith("since_")})


def boosting():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0,
                                         categorical_features=[FEATURES.index("province")])


def linear(train, test):
    """One province's linear model: heating degree days this month and last, year, the past year's price, calendar
    month and the reporting levels seen in training."""
    def inputs(f):
        return pd.concat([f[["hdd", "hdd_last", "year", "log_price_past_year"]],
                          pd.get_dummies(f.month, prefix="month", dtype=float), f.filter(like="since_")], axis=1)
    X = inputs(train)
    X = X.loc[:, X.std() > 0]
    return LinearRegression().fit(X, train.y).predict(inputs(test).reindex(columns=X.columns, fill_value=0.0))


def compare(f, first=2011):
    """Each year from `first` predicted by models fitted only to earlier years, the weather known: absolute error as
    a % of use per customer."""
    rows = {}
    for year in range(first, f.year.max() + 1):
        train, test = f[f.year < year], f[f.year == year].sort_values("province", kind="stable")
        boosted = np.exp(boosting().fit(train[FEATURES], train.y).predict(test[FEATURES]))
        lin = np.exp(np.concatenate([linear(train[train.province == p], t) for p, t in test.groupby("province")]))
        actual = np.exp(test.y.to_numpy())
        rows[year] = {"boosting": np.abs(boosted - actual).sum() / actual.sum(), "linear": np.abs(lin - actual).sum() / actual.sum()}
    return pd.DataFrame(rows).T.rename_axis("year") * 100


def main():
    series, temperatures = gas.load()
    curves, trends = {}, {}
    for p, d in series.items():
        curves[p] = heating_curves(d, temperatures[p])
        trends[p], criterion = trend(normal_weather_years(d))
        print(p, "\n", curves[p].round(3).to_string(index=False))
        print("  LWZ by number of breaks:", {k: round(float(v), 1) for k, v in criterion.items()})
        print(trends[p].round(2).to_string(index=False))
    curves = pd.concat(curves, names=["province"]).droplevel(1)
    curves.round(3).to_csv(gas.OUTPUT / "heating_curves.csv")
    pd.concat(trends, names=["province"]).droplevel(1).round(2).astype({"from_low_90": "Int64", "from_high_90": "Int64"})         .to_csv(gas.OUTPUT / "trend_segments.csv")

    f = learning_frame(series, temperatures)
    errors = compare(f)
    model = boosting().fit(f[FEATURES], f.y)
    importance = permutation_importance(model, f[FEATURES], f.y, n_repeats=10, random_state=0)
    importance = pd.Series(importance.importances_mean, FEATURES).sort_values()
    by_temp = partial_dependence(model, f[FEATURES], ["temp"], grid_resolution=30, percentiles=(0.02, 0.98), method="brute")
    by_price = partial_dependence(model, f[FEATURES], ["log_price_past_year"], grid_resolution=20, percentiles=(0.05, 0.95), method="brute")
    price_slope = np.polyfit(by_price["grid_values"][0], by_price["average"][0], 1)[0]
    errors.round(2).to_csv(gas.OUTPUT / "boosting_errors.csv")
    importance.round(5).rename("importance").rename_axis("feature").to_csv(gas.OUTPUT / "boosting_importance.csv")
    print(f"{len(f)} province-months; % error by year:\n", errors.round(1).to_string())
    print("Mean:", errors.mean().round(2).to_dict(), "\nPermutation importance:", importance.round(4).to_dict())
    print(f"Slope of the partial dependence on log price (an elasticity, if it were causal): {price_slope:.3f}")

    plt.rcParams.update(gas.STYLE)
    # Year-round use and heating, first and latest period
    fig, ax = plt.subplots(figsize=(12, 5.6))
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.07)
    w = 0.38
    for i, p in enumerate(gas.SIX):
        c = curves.loc[[p]]
        for k, (r, alpha) in enumerate([(c.iloc[0], 0.45), (c.iloc[-1], 1.0)]):
            x = i * 1.2 + (k - 0.5) * (w + 0.04)
            ax.bar(x, r.year_round, w, color=ORANGE, alpha=alpha, label="Year-round use" if i == 0 and k else None)
            ax.bar(x, r.heating, w, bottom=r.year_round, color=BLUE, alpha=alpha, label="Space heating" if i == 0 and k else None)
            ax.errorbar(x, r.total, yerr=[[r.total - r.total_low_90], [r.total_high_90 - r.total]], color=MUTED, lw=1, capsize=2)
            tag = "now" if r.period.startswith("since") else f"{int(r.period[:4]) // 10 * 10}s"
            ax.text(x, r.total_high_90 + 60, tag, ha="center", fontsize=8.5, color=MUTED)
    ax.set_xticks(np.arange(len(gas.SIX)) * 1.2, [gas.short(p) for p in gas.SIX])
    ax.set_ylabel("m³ per customer a year, normal weather")
    ax.set_ylim(0, 6600)
    ax.legend(loc="upper right", ncol=2)
    bc = curves.loc["British Columbia"]
    gas.save(fig, "heating_split.png",
             f"Year-round gas use fell faster than space heating: in BC by {1 - bc.year_round.iloc[-1] / bc.year_round.iloc[0]:.0%}, "
             f"against {1 - bc.heating.iloc[-1] / bc.heating.iloc[0]:.0%}",
             f"Use per customer in 1991-2020 normal weather, 1981-1990 (pale) and {bc.period.iloc[-1]} (Alberta 2001-2010), "
             "split by a heating-curve model",
             "Monthly use per customer per day = year-round use + slope x heating degrees below a balance point, fitted for "
             "each period; ranges from a bootstrap by year.\n" + gas.SOURCE)

    # What gradient boosting sees
    pooled = pd.read_csv(gas.OUTPUT / "provinces_pooled.csv", index_col="estimate").loc["monthly", "average"]
    names = {"temp": "Month's mean temperature", "province": "Province", "year": "Year", "temp_last": "Last month's temperature",
             "log_price_past_year": "Past year's price", "month": "Calendar month", "cold_days": "Days below −15 °C"}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.1, 1]})
    fig.subplots_adjust(top=0.76, bottom=0.2, wspace=0.42, left=0.19, right=0.98)
    a1.barh([names[k] for k in importance.index], importance / importance.max() * 100, color=BLUE)
    a1.set_xlabel("Permutation importance, % of the largest")
    a1.set_title("What the model leans on", loc="left", color=INK, fontsize=11)
    a2.plot(by_temp["grid_values"][0], np.exp(by_temp["average"][0]), color=BLUE, lw=2.2)
    a2.set_xlabel("Month's mean temperature, °C")
    a2.set_ylabel("m³ per customer a day")
    a2.set_ylim(0, None)
    a2.set_title("It finds the heating curve", loc="left", color=INK, fontsize=11)
    cents, log_use = np.exp(by_price["grid_values"][0]) * 100, by_price["average"][0]
    a3.plot(cents, (np.exp(log_use - log_use.mean()) - 1) * 100, color=BLUE, lw=2.2, label="Boosting")
    a3.plot(cents, ((cents / cents.mean()) ** pooled - 1) * 100, color=GREY, lw=2, ls="--", label=f"Econometric, {pooled:.2f}".replace("-", "−"))
    a3.set_xlabel("Past year's real price, cents per m³")
    a3.set_ylabel("% change in use")
    a3.legend(loc="lower left")
    a3.set_title("But no consistent price response", loc="left", color=INK, fontsize=11)
    gas.save(fig, "ml_drivers.png", "Gradient boosting predicts gas use better than a linear model, but finds no consistent price effect",
             f"Six provinces' monthly use per customer, each year since {errors.index[0]} predicted from earlier years "
             f"(weather known): {errors.boosting.mean():.1f}% error against {errors.linear.mean():.1f}%",
             "Price moves slowly with the year, so a model built to predict credits its effect to the year and province: partial "
             "dependence is not a causal effect.\n" + gas.SOURCE)


if __name__ == "__main__":
    main()

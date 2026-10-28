"""Forecasts with uncertainty. Residential gas volume in BC and Ontario one to 12 months ahead from three methods,
the same month last year, exponential smoothing and a seasonal ARIMA with heating degree days (normal weather assumed
ahead), each tested from every month since December 2012 using only the data before it. The same ARIMA told the
weather that came splits the error into weather and the rest. Prediction ranges come from the model's own past errors
(conformal prediction), and their coverage is checked as it would have run. Also writes output/gas-data.js, the data
for the write-up's interactive charts.

Run from the repository folder: python forecast.py (about five minutes)
"""
import json
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

import gas
import heating
from gas import BLUE, GREY, INK, MUTED

PROVINCES = ["British Columbia", "Ontario"]  # as on the dashboard
FIT_FROM = {"British Columbia": "1990-01", "Ontario": "2001-01"}  # Ontario's series is incomplete in 1998-2000
FIRST_ORIGIN, H = "2012-12", 12
BEST, TOLD = "Seasonal ARIMA with weather", "Same, told the weather"


def exog(hdd):
    """This month's and last month's heating degree days, in hundreds: bills lag the weather."""
    return pd.DataFrame({"hdd": hdd / 100, "hdd_last": hdd.shift(1) / 100})


def normal_hdd(hdd, origin, months):
    """Each month's average heating degree days over the 30 years to the origin."""
    past = hdd.loc[origin - 359:origin]
    return pd.Series(past.groupby(past.index.month).mean().reindex(months.month).to_numpy(), index=months)


def forecasts(y, hdd):
    """Each method's forecasts of log volume for the H months after y; the ARIMA's with its own 80% range, and again
    told the weather when all of it is known."""
    origin = y.index[-1]
    ahead = pd.period_range(origin + 1, periods=H, freq="M")
    s = SARIMAX(y, exog=exog(hdd).loc[y.index], order=(1, 0, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False, maxiter=200)
    f = s.get_forecast(H, exog=exog(pd.concat([hdd.loc[:origin], normal_hdd(hdd, origin, ahead)])).loc[ahead])
    ci = f.conf_int(alpha=0.2)
    out = {BEST: pd.DataFrame({"mean": f.predicted_mean, "low_80": ci.iloc[:, 0], "high_80": ci.iloc[:, 1]})}
    actual = exog(hdd.reindex(pd.period_range(hdd.index[0], ahead[-1], freq="M"))).loc[ahead]
    if actual.notna().all().all():
        out[TOLD] = pd.DataFrame({"mean": s.get_forecast(H, exog=actual).predicted_mean})
    e = ETSModel(y, error="add", trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit(disp=False)
    out["Exponential smoothing"] = pd.DataFrame({"mean": e.forecast(H).to_numpy()}, index=ahead)
    out["Same month last year"] = pd.DataFrame({"mean": y.reindex(ahead - 12).to_numpy()}, index=ahead)
    return out


def backtest(y, hdd):
    """Forecasts from every month since FIRST_ORIGIN, each fitted to the data before it, against what happened
    (million m3)."""
    rows = []
    for origin in pd.period_range(FIRST_ORIGIN, y.index[-1] - 1, freq="M"):
        for name, f in forecasts(y.loc[:origin], hdd).items():
            for step, (month, r) in enumerate(np.exp(f).iterrows(), start=1):
                if month in y.index:
                    rows.append({"origin": origin, "month": month, "horizon": step, "model": name,
                                 "actual": np.exp(y[month]), **r.to_dict()})
    return pd.DataFrame(rows)


def conformal(bt, level=0.8, warmup=24):
    """Ranges from the model's own past errors. Each forecast gets the `level` percentile of the absolute % errors
    at its horizon known at its origin (none before `warmup` of them), so coverage is checked as it would have run;
    the widths from all errors are for the forecast from the latest month."""
    e = bt.assign(abs_err=(bt.actual - bt["mean"]).abs() / bt["mean"])
    known = (e.abs_err[(e.horizon == r.horizon) & (e.month <= r.origin)] for r in e.itertuples())
    e["width"] = [k.quantile(level) if len(k) >= warmup else np.nan for k in known]
    return e, e.groupby("horizon").abs_err.quantile(level)


def errors(bt):
    """Volume-weighted absolute % error by months ahead and method."""
    g = bt.assign(miss=(bt["mean"] - bt.actual).abs()).groupby(["horizon", "model"])
    return (g.miss.sum() / g.actual.sum()).unstack() * 100


def year_errors(bt):
    """Each calendar year's total, forecast the December before: % above (+) or below what came."""
    t = bt[bt.origin.dt.month == 12].groupby(["origin", "model"]).agg(
        forecast=("mean", "sum"), actual=("actual", "sum"), months=("mean", "size"))
    t = t[t.months == 12]
    out = (t.forecast / t.actual - 1).unstack() * 100
    return out.set_axis(out.index.year + 1).rename_axis("year")


def totals(bt):
    """Each origin's forecast of the next 12 months' total against what came."""
    t = bt.groupby("origin").agg(forecast=("mean", "sum"), actual=("actual", "sum"), months=("mean", "size"))
    return t[t.months == H]


def chart_data(volume, e, told, live):
    """The forecast replay: what happened since 2008, and from each origin the forecast, its conformal range as it
    would have been built then and the run told the weather; the forecast from the latest month last."""
    def grid(t, col):
        return t.pivot(index="origin", columns="horizon", values=col).reindex(columns=range(1, H + 1))

    def rows(t):
        return [[None if pd.isna(v) else round(float(v), 1) for v in r] for r in np.asarray(t)]

    mean, width = grid(e, "mean"), grid(e, "width")
    actual = volume.loc["2008-01":]
    return {"start": "2008-01", "actual": rows([actual])[0],
            "origins": [str(o) for o in mean.index] + [str(volume.index[-1])],
            "mean": rows(mean) + rows([live["mean"]]), "low": rows(mean * (1 - width)) + rows([live.low_80]),
            "high": rows(mean * (1 + width)) + rows([live.high_80]),
            "oracle": rows(grid(told, "mean").reindex(mean.index)) + [[None] * H]}


def main():
    warnings.filterwarnings("ignore")  # convergence notes from refitting the models 165 times
    series, temperatures = gas.load()
    results, fans, charts = {}, {}, {}
    for p in PROVINCES:
        d = series[p]
        y = np.log(d.m3 / 1e6).loc[FIT_FROM[p]:].dropna()
        bt = backtest(y, d.hdd)
        e, widths = conformal(bt[bt.model == BEST])
        live = np.exp(forecasts(y, d.hdd)[BEST]["mean"])
        fans[p] = pd.DataFrame({"mean": live, "low_80": live * (1 - widths.to_numpy()),
                                "high_80": live * (1 + widths.to_numpy())})
        t, total = totals(e), float(live.sum())
        width = float((t.actual / t.forecast - 1).abs().quantile(0.8))
        twelve = {"months": f"{live.index[0]} to {live.index[-1]}", "forecast": round(total, 1),
                  "low_80": round(total * (1 - width), 1), "high_80": round(total * (1 + width), 1),
                  "mean_abs_error_pct": round(float((t.forecast / t.actual - 1).abs().mean()) * 100, 1)}
        own = e.actual.between(e.low_80, e.high_80).mean()
        held = (e.abs_err <= e.width)[e.width.notna()].mean()
        results[p] = {"errors": errors(bt), "years": year_errors(bt), "own": own, "held": held, "total": twelve}
        charts[p] = chart_data(np.exp(y), e, bt[bt.model == TOLD], fans[p])
        print(f"{p}: {bt.origin.nunique()} origins, {bt.origin.min()} to {bt.origin.max()}")
        print(results[p]["errors"].loc[[1, 3, 6, 12]].round(1).to_string())
        print(f"80% ranges held: the model's own {own:.0%}, conformal (earlier errors only) {held:.0%}")
        print("A calendar year's total, mean absolute % error:", results[p]["years"].abs().mean().round(1).to_dict())
        print("The next 12 months:", twelve)

    pd.concat(fans, names=["province", "month"]).round(1).to_csv(gas.OUTPUT / "forecast.csv")
    pd.concat({p: r["errors"] for p, r in results.items()}, names=["province"]).round(2).to_csv(gas.OUTPUT / "forecast_errors.csv")
    pd.concat({p: r["years"] for p, r in results.items()}, names=["province"]).round(2).to_csv(gas.OUTPUT / "forecast_years.csv")
    pd.DataFrame({p: r["total"] for p, r in results.items()}).T.rename_axis("province").to_csv(gas.OUTPUT / "forecast_total.csv")
    pd.DataFrame({p: {"model_80_held": r["own"], "conformal_80_held": r["held"]} for p, r in results.items()}).T \
        .rename_axis("province").round(3).to_csv(gas.OUTPUT / "forecast_coverage.csv")
    data = {"heating": heating.chart_data(series, temperatures), "forecast": charts}
    (gas.OUTPUT / "gas-data.js").write_text("const GAS = " + json.dumps(data, separators=(",", ":")) + ";\n", encoding="utf-8")

    plt.rcParams.update(gas.STYLE)
    # The next 12 months
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    fig.subplots_adjust(top=0.78, bottom=0.2, wspace=0.15)
    for ax, p in zip(axes, PROVINCES):
        recent, fan = series[p].m3.iloc[-36:] / 1e6, fans[p]
        x, xf = recent.index.to_timestamp(), fan.index.to_timestamp()
        ax.plot(x, recent, color=INK, lw=2, label="Actual")
        ax.fill_between(xf, fan.low_80, fan.high_80, color=BLUE, alpha=0.2, lw=0, label="80% range")
        ax.plot([x[-1], *xf], [recent.iloc[-1], *fan["mean"]], color=BLUE, lw=2, ls="--", label="Forecast")
        ax.set_ylim(0, fan.high_80.max() * 1.25)  # room for the legend
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_title(f"{gas.short(p)}: {fan['mean'].sum():,.0f} million m³ in the next 12 months", loc="left", color=INK, fontsize=12)
    axes[0].set_ylabel("Million m³ a month")
    axes[0].legend(loc="upper left", ncol=3)
    held = ", ".join(f"{r['held']:.0%} in {gas.short(p)}" for p, r in results.items())
    gas.save(fig, "forecast.png", "Residential gas use over the next 12 months, with ranges from past errors",
             f"Seasonal ARIMA with heating degree days, normal weather assumed, from {series[PROVINCES[0]].index[-1]:%B %Y}",
             f"80% ranges from the absolute % errors of forecasts made each month since {FIRST_ORIGIN[:4]}; built only from "
             f"earlier errors, such ranges held {held}.\n" + gas.SOURCE)

    # A year ahead, with and without knowing the weather
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    fig.subplots_adjust(top=0.78, bottom=0.2, wspace=0.08)
    for ax, p in zip(axes, PROVINCES):
        q = results[p]["years"]
        ax.bar(q.index - 0.2, q[BEST], 0.4, color=BLUE, label="Normal weather assumed")
        ax.bar(q.index + 0.2, q[TOLD], 0.4, color=GREY, label="Told the weather that came")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks([2015, 2020, 2025])
        ax.set_title(f"{gas.short(p)}: off by {q[BEST].abs().mean():.1f}% a year, {q[TOLD].abs().mean():.1f}% given the weather",
                     loc="left", color=INK, fontsize=11)
    axes[0].set_ylabel("Forecast minus actual, % of the year's use")
    axes[0].legend(loc="lower left")
    gas.save(fig, "forecast_weather.png", "Most of the error in forecasting a year's gas use is the weather",
             "Each calendar year forecast the December before (seasonal ARIMA with heating degree days), with normal weather "
             "and with the weather that came",
             "Each forecast fitted only to data before it; residential volume.\n" + gas.SOURCE)


if __name__ == "__main__":
    main()

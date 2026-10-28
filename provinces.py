"""Is BC different from other provinces? The monthly and annual price elasticities of gas.py for six provinces, then
partial pooling: a random-effects model, fitted by restricted maximum likelihood, estimates the average response and
the real spread between provinces, and moves each province's estimate toward the average by as much as its own noise
warrants. Ranges for the long-run elasticity come from a recursive residual bootstrap rather than the delta method.

Run from the repository folder: python provinces.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize

import gas
from gas import GREY, INK, MUTED


def pool(est, se):
    """Random effects: each estimate = the average + a province effect with spread tau + noise with its known SE.
    tau by restricted maximum likelihood. Returns the average, its SE, tau, and each province's shrunken estimate and
    SE."""
    v = se ** 2

    def deviance(log_tau):
        w = 1 / (v + np.exp(2 * log_tau))
        mu = (w * est).sum() / w.sum()
        return np.log(1 / w).sum() + np.log(w.sum()) + (w * (est - mu) ** 2).sum()

    tau = np.exp(optimize.minimize_scalar(deviance, bounds=(-12, 1), method="bounded").x)
    w = 1 / (v + tau ** 2)
    mu = (w * est).sum() / w.sum()
    k = tau ** 2 / (tau ** 2 + v)  # the weight on the province's own estimate
    return mu, w.sum() ** -0.5, tau, k * est + (1 - k) * mu, np.sqrt(k * v + (1 - k) ** 2 / w.sum())


def recursive_bootstrap(y, draws=1000, seed=0):
    """90% range for the long-run elasticity b / (1 - adjustment) of gas.partial_adjustment: rebuild log use year by
    year from the fitted model and resampled residuals, refit, and take percentiles."""
    y = y.dropna(subset=gas.MODEL_INPUTS)
    X = sm.add_constant(y[["log_price", "log_use_lag", "hdd", "trend"] + gas.breaks(y)].assign(hdd=y.hdd / 1000))
    fit = sm.OLS(y.log_use, X).fit()
    rest = (X.drop(columns="log_use_lag") @ fit.params.drop("log_use_lag")).to_numpy()
    follows = np.r_[False, np.diff(y.index.to_numpy()) == 1]  # last year is in the sample, so its simulated use carries over
    rng, out = np.random.default_rng(seed), []
    for _ in range(draws):
        e, sim, lag = rng.choice(fit.resid.to_numpy(), len(y)), np.empty(len(y)), y.log_use_lag.to_numpy().copy()
        for i in range(len(y)):
            if follows[i]:
                lag[i] = sim[i - 1]
            sim[i] = rest[i] + fit.params.log_use_lag * lag[i] + e[i]
        f = sm.OLS(sim, X.assign(log_use_lag=lag)).fit()
        out.append(f.params.log_price / (1 - f.params.log_use_lag))
    return np.percentile(out, [5, 95])


def main():
    series, _ = gas.load()
    rows = {}
    for p, d in series.items():
        m, y = gas.monthly_model(d, "log_price_past_year"), gas.annual(d)
        low, high = recursive_bootstrap(y)
        rows[p] = {"months": m.nobs, "monthly": m.params.log_price_past_year, "monthly_se": m.bse.log_price_past_year,
                   **gas.partial_adjustment(y).add_prefix("annual_"), "long_run_low_90": low, "long_run_high_90": high}
    r = pd.DataFrame.from_dict(rows, orient="index").rename_axis("province")
    pooled = {}
    for col in ["monthly", "annual_short_run"]:
        mu, mu_se, tau, r[f"{col}_pooled"], r[f"{col}_pooled_se"] = pool(r[col], r[f"{col}_se"])
        pooled[col] = {"average": mu, "average_se": mu_se, "spread": tau}
    pooled = pd.DataFrame(pooled).T.rename_axis("estimate")
    r.round(4).to_csv(gas.OUTPUT / "provinces.csv")
    pooled.round(4).to_csv(gas.OUTPUT / "provinces_pooled.csv")
    print(r.round(3).T.to_string())
    print(pooled.round(3).to_string())
    print("Long run, delta-method 90% ranges:", {p: f"{x.annual_long_run - 1.645 * x.annual_long_run_se:.2f} to "
                                                    f"{x.annual_long_run + 1.645 * x.annual_long_run_se:.2f}" for p, x in r.iterrows()})

    plt.rcParams.update(gas.STYLE)
    fig, ax = plt.subplots(figsize=(12, 5.4))
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.16)
    rows_y = np.arange(len(r))[::-1]
    ax.errorbar(r.monthly, rows_y + 0.13, xerr=1.645 * r.monthly_se, fmt="o", color=GREY, ms=7, lw=2, label="Each province on its own")
    ax.errorbar(r.monthly_pooled, rows_y - 0.13, xerr=1.645 * r.monthly_pooled_se, fmt="o", color=INK, ms=7, lw=2, label="Partially pooled")
    average = pooled.loc["monthly", "average"]
    ax.axvline(average, color=MUTED, lw=1, ls="--")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(rows_y, r.index)
    ax.set_xlabel("% change in use per 1% higher real price (past year's price, 90% interval)")
    ax.legend(loc="upper left")
    gas.save(fig, "provinces_pooled.png",
             f"Across six provinces, a 10% higher gas price cuts household use by about {-average * 10:.1f}%",
             "Monthly price elasticity by province, alone and with partial pooling (random effects): noisy estimates move "
             "toward the average (dashed)",
             "Monthly regressions with heating degree days, month, trend and reporting levels; pooling by restricted maximum "
             "likelihood.\n" + gas.SOURCE)


if __name__ == "__main__":
    main()

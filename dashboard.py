"""Monthly dashboard for residential gas demand in British Columbia and Ontario: the latest month against
what weather, season, price and trend predict, the last 12 months against the 12 before (weather-adjusted),
the real price, the next 12 months' forecast with its 80% range, and the latest elasticity estimates. Writes a static
site to site/.

Run from the repository folder, after gas.py and forecast.py: python gas.py && python forecast.py && python dashboard.py
"""
from pathlib import Path
import datetime as dt
import html
import json
import shutil
import urllib.error
import urllib.request

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import gas

SITE = Path("site")
URL = "https://atamalfuzi.github.io/canada_gas_demand/"
WRITE_UP = "https://atamalfuzi.github.io/canada-gas-demand.html"
PROVINCES = ["British Columbia", "Ontario"]  # Alberta's reported price changes definition in 2014


def series():
    """Each province's monthly series, built exactly as in gas.py."""
    return gas.load(PROVINCES)[0]


# Monthly figures ----
def expected_use(d):
    """Use per customer predicted by the monthly model (past year's price, heating degree days, month, trend,
    reporting breaks), and the model's effect of one heating degree day on log use."""
    fit = gas.monthly_model(d, "log_price_past_year")
    return np.exp(fit.fittedvalues).reindex(d.index), fit.params.hdd / 100  # the model uses hundreds of degree days


def summary(d):
    expected, per_degree_day = expected_use(d)
    adjusted = gas.weather_adjusted(d, per_degree_day)
    last = d.index.max()
    return {
        "month": str(last),
        "use_m3": round(float(d.use[last]), 1),
        "expected_m3": round(float(expected[last]), 1),
        "vs_expected": round(float(d.use[last] / expected[last] - 1), 4),
        "last_12_vs_previous_12": round(float(adjusted.loc[last - 11:last].sum() / adjusted.loc[last - 23:last - 12].sum() - 1), 4),
        "real_price_vs_year_ago": round(float(d.real_price[last] / d.real_price[last - 12] - 1), 4),
    }


# Checks before publishing ----
def problems(d, published_through=None):
    """Reasons not to publish: gaps or implausible values in the last two years, or data older than what is live."""
    last = d.index.max()
    recent = d.reindex(pd.period_range(last - 23, last, freq="M"))
    found = []
    if recent[["use", "real_price", "hdd"]].isna().any().any():
        found.append("missing values in the last 24 months")
    if not recent.use.dropna().between(5, 1500).all():
        found.append("use per customer outside 5 to 1,500 m3 a month")
    if not recent.real_price.dropna().between(0.05, 3).all():
        found.append("real price outside $0.05 to $3 per m3")
    if published_through and str(last) < published_through:
        found.append(f"data end {last} is older than the published {published_through}")
    return found


def published_through():
    """The data month of the dashboard now live, if there is one."""
    try:
        with urllib.request.urlopen(URL + "status.json", timeout=30) as response:
            return json.load(response)["data_through"]
    except (urllib.error.URLError, KeyError, ValueError):  # first run, or no live copy to compare with
        return None


# Figures ----
INK, MUTED = "#0b0b0b", "#52514e"
COLOURS = {"British Columbia": "#2a78d6", "Ontario": "#eb6834"}
SOURCE = "Source: Statistics Canada tables 25-10-0086-01 and 18-10-0004-01"


def save(fig, name, title, subtitle, caption):
    fig.text(0.01, 0.99, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.93, subtitle, color=MUTED, va="top")
    fig.text(0.01, 0.01, caption, color=MUTED, fontsize=9, va="bottom")
    fig.savefig(SITE / name, dpi=150, facecolor="white")
    plt.close(fig)


def figures(data, expected):
    plt.rcParams.update({
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": MUTED,
        "axes.grid": True, "grid.color": "#e6e5e1", "axes.axisbelow": True, "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.subplots_adjust(top=0.78, bottom=0.2, wspace=0.08)
    for ax, (p, d) in zip(axes, data.items()):
        recent = d.loc[d.index.max() - 35:]
        months = recent.index.to_timestamp()
        ax.plot(months, recent.use, color=COLOURS[p], lw=2.2, label="Actual")
        ax.plot(months, expected[p].loc[recent.index], color=MUTED, lw=1.5, ls="--", label="Expected")
        ax.set_title(p, loc="left", color=INK, fontsize=12)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_ylabel("m³ per customer per month")
    axes[0].legend(loc="upper right")
    save(fig, "recent_use.png", "Gas use per customer, actual and expected, last three years",
         "Expected: what the weather, the season, the past year's price and the long-run trend predict",
         "Monthly model of log use per customer with heating degree days, calendar month, trend and reporting levels.\n" + SOURCE + "; Government of Canada daily climate data.")

    fig, ax = plt.subplots(figsize=(12, 4.6))
    fig.subplots_adjust(top=0.76, bottom=0.2)
    for p, d in data.items():
        recent = d.real_price.rolling(12).mean().loc[d.index.max() - 119:]
        ax.plot(recent.index.to_timestamp(), recent * 100, color=COLOURS[p], lw=2.2, label=p)
    ax.set_ylabel("Cents per m³ (2002 dollars)")
    ax.set_ylim(0, None)
    ax.legend(loc="lower left")
    save(fig, "price_trend.png", "Residential gas price after inflation, last ten years",
         "Revenue per cubic metre, deflated by provincial CPI, 12-month average",
         "Source: Statistics Canada tables 25-10-0033-01, 25-10-0086-01 and 18-10-0004-01.")


# Page ----
def pct(x):
    return f"{x:+.1%}".replace("-", "−")


def page(stats, forecast, elasticities, latest_path, data_through, updated):
    month = pd.Period(data_through, "M").strftime("%B %Y")
    cards = "".join(f"""
      <section class="card">
        <h2>{html.escape(p)}</h2>
        <dl>
          <div><dt>{month} use vs expected</dt><dd>{pct(s['vs_expected'])}</dd>
            <p>{s['use_m3']:.0f} m³ per customer, against {s['expected_m3']:.0f} expected</p></div>
          <div><dt>Last 12 months vs the 12 before</dt><dd>{pct(s['last_12_vs_previous_12'])}</dd>
            <p>Weather-adjusted to an average year</p></div>
          <div><dt>Real price vs a year earlier</dt><dd>{pct(s['real_price_vs_year_ago'])}</dd>
            <p>{month}, after inflation</p></div>
          <div><dt>Next 12 months, million m³</dt><dd>{forecast.loc[p, 'forecast']:,.0f}</dd>
            <p>In normal weather; 80% range {forecast.loc[p, 'low_80']:,.0f} to {forecast.loc[p, 'high_80']:,.0f}. Past 12-month forecasts missed by {forecast.loc[p, 'mean_abs_error_pct']:.1f}% on average</p></div>
        </dl>
      </section>""" for p, s in stats.items())
    rows = "".join(
        f"<tr><th scope=\"row\">{html.escape(p)}</th><td>{e.annual_short_run:.2f}</td><td>{e.annual_long_run:.2f}</td>"
        f"<td>{latest_path.loc[p, 'elasticity']:.2f} <span>({latest_path.loc[p, 'low']:.2f} to {latest_path.loc[p, 'high']:.2f})</span></td></tr>"
        .replace("-", "−")
        for p, e in elasticities.iterrows())
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'self'; script-src 'none'; style-src 'self'; img-src 'self'; connect-src 'none'; object-src 'none'; base-uri 'self'; form-action 'none'" />
  <title>Household gas demand monitor: BC and Ontario | Ata Malfuzi</title>
  <meta name="description" content="Monthly update on household natural gas use and prices in British Columbia and Ontario, with a 12-month forecast and the latest price elasticity estimates." />
  <link rel="canonical" href="{URL}" />
  <link rel="stylesheet" href="style.css" />
</head>
<body>
<main>
  <p class="eyebrow"><a href="https://atamalfuzi.github.io/">Ata Malfuzi</a> · Monthly dashboard</p>
  <h1>Household gas demand monitor: BC and Ontario</h1>
  <p class="meta">Data through {month} · updated {updated:%d %B %Y} · refreshes on the 28th of each month</p>
  <p class="links"><a href="{WRITE_UP}">Full write-up</a><a href="https://github.com/atamalfuzi/canada_gas_demand">Code on GitHub</a><a href="status.json">Latest numbers (JSON)</a></p>
  <div class="cards">{cards}
  </div>
  <figure><img src="recent_use.png" alt="Two panels of gas use per customer over the last three years in British Columbia and Ontario, actual against expected." width="1800" height="750" /></figure>
  <figure><img src="forecast.png" alt="Residential gas volume over the last three years in British Columbia and Ontario, with a forecast for the next 12 months and its 80% range." width="2400" height="1080" /></figure>
  <figure><img src="price_trend.png" alt="Residential gas price after inflation over the last ten years in British Columbia and Ontario, 12-month average." width="1800" height="690" /></figure>
  <h2>Latest price elasticities</h2>
  <p>The % change in gas use per customer for a 1% higher real price, re-estimated each month with all data to date.</p>
  <table>
    <thead><tr><th scope="col">Province</th><th scope="col">Short run</th><th scope="col">Long run</th><th scope="col">Short run, latest year (90% band)</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <figure><img src="elasticity_over_time.png" alt="Short-run price elasticity estimated year by year for British Columbia, Ontario and Alberta, with 90% bands." width="2600" height="1160" /></figure>
  <footer>
    <p>Personal project: my own analysis of publicly available data. It is not associated with my work for any current or past employer and does not represent their views.</p>
    <p>Adapted from Statistics Canada, tables 25-10-0033-01, 25-10-0086-01 and 18-10-0004-01; this does not constitute an endorsement by Statistics Canada of this product. Contains information licensed under the Open Government Licence - Canada.</p>
  </footer>
</main>
</body>
</html>
"""


CSS = """:root { --text: #17201d; --muted: #4b5651; --rule: #e1e7e4; --accent: #0b6e5f; --bg-2: #f4f6f5; }
* { box-sizing: border-box; margin: 0; }
body { font-family: system-ui, sans-serif; color: var(--text); background: #fff; line-height: 1.55; }
main { max-width: 1000px; margin: 0 auto; padding: 2rem 1.25rem 3rem; }
h1 { font-family: Georgia, serif; font-size: clamp(1.7rem, 4vw, 2.3rem); line-height: 1.2; margin: .3rem 0 .5rem; }
h2 { font-family: Georgia, serif; font-size: 1.3rem; margin: 2rem 0 .5rem; }
a { color: var(--accent); }
.eyebrow, .meta, footer { color: var(--muted); font-size: .9rem; }
.links { display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; margin: .8rem 0 1.5rem; font-weight: 600; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
.card { border: 1px solid var(--rule); border-radius: 8px; padding: 1rem 1.2rem; background: var(--bg-2); }
.card h2 { margin: 0 0 .6rem; font-size: 1.15rem; }
.card dl { display: grid; gap: .8rem; }
dt { font-size: .85rem; color: var(--muted); }
dd { font-size: 1.6rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.card p { font-size: .8rem; color: var(--muted); }
figure { margin: 1.5rem 0; }
img { width: 100%; height: auto; display: block; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; margin: .5rem 0 1rem; }
th, td { text-align: right; padding: .45rem .6rem; border-bottom: 1px solid var(--rule); }
th:first-child { text-align: left; }
td span { color: var(--muted); font-size: .85rem; }
footer { border-top: 1px solid var(--rule); margin-top: 2rem; padding-top: 1rem; display: grid; gap: .5rem; }
"""


def main():
    SITE.mkdir(exist_ok=True)
    data = series()
    live = published_through()
    issues = {p: problems(d, live) for p, d in data.items()}
    if any(issues.values()):
        raise SystemExit("Not publishing: " + "; ".join(f"{p}: {', '.join(v)}" for p, v in issues.items() if v))

    stats = {p: summary(d) for p, d in data.items()}
    expected = {p: expected_use(d)[0] for p, d in data.items()}
    figures(data, expected)
    for name in ("elasticity_over_time.png", "forecast.png"):
        shutil.copy(gas.FIGURES / name, SITE / name)
    forecast = pd.read_csv(gas.OUTPUT / "forecast_total.csv", index_col="province").loc[PROVINCES]

    elasticities = pd.read_csv(gas.OUTPUT / "elasticities.csv", index_col="province").loc[PROVINCES]
    paths = pd.read_csv(gas.OUTPUT / "elasticity_over_time.csv", index_col=["province", "year"])
    latest_path = pd.DataFrame({p: paths.loc[p].iloc[-1] for p in PROVINCES}).T  # elasticity, low, high
    data_through = min(s["month"] for s in stats.values())
    updated = dt.date.today()

    (SITE / "index.html").write_text(page(stats, forecast, elasticities, latest_path, data_through, updated), encoding="utf-8")
    (SITE / "style.css").write_text(CSS, encoding="utf-8")
    status = {"data_through": data_through, "updated": updated.isoformat(), "provinces": stats,
              "forecast_next_12_months_million_m3": {p: {"months": f.months, "forecast": f.forecast, "range_80": [f.low_80, f.high_80],
                                                         "past_mean_abs_error_pct": f.mean_abs_error_pct}
                                                     for p, f in forecast.iterrows()},
              "elasticities": {p: {"short_run": round(float(e.annual_short_run), 3), "long_run": round(float(e.annual_long_run), 3),
                                   "short_run_latest_year": round(float(latest_path.loc[p, "elasticity"]), 3),
                                   "short_run_latest_year_90": [round(float(latest_path.loc[p, "low"]), 3), round(float(latest_path.loc[p, "high"]), 3)]}
                               for p, e in elasticities.iterrows()}}
    (SITE / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

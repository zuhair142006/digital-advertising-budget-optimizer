"""Marketing Budget Recommendation Engine -- where should Daraz spend the next Rs 1M?

Method
1. Aggregate history to one row per day x platform (spend, revenue).
2. Fit a diminishing-returns response curve per platform:
       log(revenue) = log(k) + b * log(spend) + g1*sale_day + g2*ramadan_day
   b < 1 means every extra rupee earns less than the one before (saturation).
   Season dummies stop 11.11 / Ramadan peaks from being mistaken for "more spend
   = more revenue".
   The elasticities b are then shrunk toward their cross-platform mean
   (empirical Bayes) and capped at 0.95, because a noisy b close to 1 would
   otherwise send the whole budget to one platform.
3. Split the budget so that the *marginal* ROAS (revenue from the next rupee) is
   equal on every platform, subject to a 5%-45% guard-rail per platform
   (keeps every channel alive for learning / brand reach, avoids over-reliance).
4. Bootstrap the days 300 times, re-optimising each time. The recommendation is
   the average of the 300 allocations (bagging); the spread gives a 90% range.
5. Validate: "spend" each candidate split inside the simulator for a 30-day
   month and compare the revenue that actually comes back.

    python src/recommender.py --data data/daraz_ads_1M.parquet --budget 1000000
    python src/recommender.py --data data/daraz_ads_10K.csv --budget 1000000   # works on the bundled sample
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import config as C
from generate_data import calendar, simulate, PLAT
from kpis import kpi_table, MARGIN

MIN_SHARE, MAX_SHARE = 0.05, 0.45
DAYS_IN_PLAN = 30


# ----------------------------------------------------------------------------- data
def daily_panel(df):
    d = df.groupby(["date", "platform"], observed=True)[["spend", "revenue"]].sum().reset_index()
    d["date"] = pd.to_datetime(d["date"].astype(str))
    days, sale, ram = calendar()
    flags = pd.DataFrame({"date": days, "sale": sale.astype(int), "ramadan": ram.astype(int)})
    d = d.merge(flags, on="date", how="left")
    return d[(d.spend > 0) & (d.revenue > 0)]


# ----------------------------------------------------------------------------- model
B_CAP = 0.95   # prior: no ad platform has literally zero saturation


def _refit_k(g, b):
    """Given an elasticity b, re-estimate log k and season effects with b held fixed."""
    X = np.column_stack([np.ones(len(g)), g.sale, g.ramadan])
    y = np.log(g.revenue.values) - b * np.log(g.spend.values)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return np.exp(beta[0]) * np.mean(np.exp(resid)), beta    # smearing correction


def fit_curves(panel, shrink=True):
    """Per-platform log-log OLS, then empirical-Bayes shrinkage of the elasticities.

    Daily revenue is noisy (a platform gets only ~5-20 orders a day), so a raw b can
    land near 1 by chance -- and with b ~ 1 the optimiser would pour the whole budget
    into that platform. Shrinking every b toward the cross-platform mean, in
    proportion to its standard error, keeps precise estimates and tames noisy ones.
    """
    rows = []
    for p in PLAT:
        g = panel[panel.platform == p]
        X = np.column_stack([np.ones(len(g)), np.log(g.spend), g.sale, g.ramadan])
        y = np.log(g.revenue.values)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        s2 = resid @ resid / (len(y) - X.shape[1])
        se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
        rows.append(dict(platform=p, b_raw=beta[1], b_se=se[1], r2=1 - resid.var() / y.var(),
                         n_days=len(g)))
    c = pd.DataFrame(rows).set_index("platform")
    if shrink:
        w_prec = 1 / c.b_se ** 2
        b_pool = (c.b_raw * w_prec).sum() / w_prec.sum()
        tau2 = max(c.b_raw.var(ddof=1) - (c.b_se ** 2).mean(), 0.01 ** 2)  # between-platform variance
        w = tau2 / (tau2 + c.b_se ** 2)
        c["b"] = (w * c.b_raw + (1 - w) * b_pool).clip(0.05, B_CAP)
    else:
        c["b"] = c.b_raw.clip(0.05, B_CAP)
    for p in PLAT:
        k, beta = _refit_k(panel[panel.platform == p], c.loc[p, "b"])
        c.loc[p, "k"], c.loc[p, "sale_uplift"], c.loc[p, "ramadan_uplift"] = k, np.exp(beta[1]) - 1, np.exp(beta[2]) - 1
    return c


def revenue_of(alloc, curves, days=DAYS_IN_PLAN):
    """Predicted revenue of a monthly allocation (dict/Series platform -> PKR), flat daily pacing."""
    a = pd.Series(alloc).reindex(curves.index).fillna(0)
    daily = a / days
    return float((curves.k * daily.clip(lower=0) ** curves.b).sum() * days)


def marginal_roas(alloc, curves, days=DAYS_IN_PLAN):
    daily = pd.Series(alloc).reindex(curves.index) / days
    return curves.b * curves.k * daily ** (curves.b - 1)


def optimise(curves, budget, lo=MIN_SHARE, hi=MAX_SHARE, days=DAYS_IN_PLAN):
    """Equal-marginal-ROAS allocation with box constraints (bisection on the shadow price)."""
    k, b = curves.k.values, curves.b.values
    lo_x, hi_x = lo * budget / days, hi * budget / days
    def x_at(lam):  # daily spend where marginal ROAS == lam, clipped to the guard-rails
        with np.errstate(over="ignore"):
            x = (b * k / lam) ** (1 / (1 - b))
        return np.clip(x, lo_x, hi_x)
    a, z = 1e-6, 1e6
    for _ in range(200):
        m = np.sqrt(a * z)
        if x_at(m).sum() * days > budget:
            a = m
        else:
            z = m
    x = x_at(np.sqrt(a * z)) * days
    x *= budget / x.sum()
    return pd.Series(x, index=curves.index)


def bootstrap(panel, budget, n=300, seed=7, save=True):
    """Resample days with replacement, refit + re-optimise each time.
    Returns the bagged (average) allocation and the 5th / 95th percentile shares."""
    rng = np.random.default_rng(seed)
    dates = panel.date.unique()
    shares = []
    for _ in range(n):
        pick = pd.DataFrame({"date": rng.choice(dates, len(dates), replace=True)})
        boot = pick.merge(panel, on="date")
        shares.append(optimise(fit_curves(boot), budget) / budget)
    s = pd.DataFrame(shares)
    bagged = s.mean() / s.mean().sum() * budget
    if save:
        out = s.assign(**{"Meta (FB+IG)": s.Facebook + s.Instagram})
        os.makedirs("results", exist_ok=True)
        out.to_csv("results/bootstrap_shares.csv", index=False)
    return bagged, s.quantile(0.05), s.quantile(0.95)


# ----------------------------------------------------------------------------- validation
def simulate_month(alloc, seeds=(1, 2, 3, 4, 5), n_rows=200_000):
    """Spend a monthly allocation for 30 normal (non-sale) days inside the simulator."""
    days, sale, ram = calendar()
    normal = np.where(~sale & ~ram)[0][:DAYS_IN_PLAN]
    B = np.zeros((len(days), len(PLAT)))
    B[normal, :] = np.array([alloc[p] for p in PLAT]) / DAYS_IN_PLAN
    revs = [simulate(n_rows, seed=s, budget=B).revenue.sum() for s in seeds]
    return float(np.mean(revs)), float(np.std(revs))


# ----------------------------------------------------------------------------- report
def recommend(df, budget=1_000_000, validate=True, n_boot=300):
    panel = daily_panel(df)
    curves = fit_curves(panel)
    curves["true_b"] = [C.PLATFORMS[p]["b"] for p in curves.index]
    rec, lo, hi = bootstrap(panel, budget, n=n_boot)   # bagged allocation = the recommendation

    hist = kpi_table(df, "platform").reindex(PLAT)
    options = {
        "Recommended (engine)": rec,
        "Current mix (history)": pd.Series({p: C.PLATFORMS[p]["share"] * budget for p in PLAT}),
        "Equal split (20% each)": pd.Series({p: budget / 5 for p in PLAT}),
        "Proportional to past ROAS": hist.ROAS / hist.ROAS.sum() * budget,
    }
    comp = []
    for name, alloc in options.items():
        row = {"option": name, **{p: alloc[p] / budget * 100 for p in PLAT},
               "predicted_revenue": revenue_of(alloc, curves)}
        if validate:
            row["simulated_revenue"], row["simulated_sd"] = simulate_month(alloc)
        comp.append(row)
    comp = pd.DataFrame(comp)
    rev_col = "simulated_revenue" if validate else "predicted_revenue"
    comp["ROAS"] = comp[rev_col] / budget
    comp["profit_after_ads"] = comp[rev_col] * MARGIN - budget

    table = pd.DataFrame({
        "share_%": rec / budget * 100, "PKR": rec,
        "share_low_%": lo * 100, "share_high_%": hi * 100,
        "past_ROAS": hist.ROAS, "past_CPA": hist.CPA,
        "marginal_ROAS_now": marginal_roas(options["Current mix (history)"], curves),
        "marginal_ROAS_rec": marginal_roas(rec, curves),
        "b_raw": curves.b_raw, "b_se": curves.b_se, "b_shrunk": curves.b, "b_true": curves.true_b,
        "r2": curves.r2,
    })
    return table, comp, curves


def profit_curve(curves, budgets=np.arange(100_000, 2_000_001, 50_000)):
    """Contribution profit (revenue x MARGIN - spend) of the optimal split at each budget."""
    out = []
    for B in budgets:
        rev = revenue_of(optimise(curves, B), curves)
        out.append(dict(budget=B, revenue=rev, profit=rev * MARGIN - B))
    return pd.DataFrame(out)


def creative_advice(df):
    """Best campaign per platform + best audience cells, from history."""
    pc = kpi_table(df, ["platform", "campaign"]).reset_index()
    best = pc.loc[pc.groupby("platform").ROAS.idxmax(), ["platform", "campaign", "ROAS", "CTR_%"]]
    attn = pc.loc[pc.groupby("platform")["CTR_%"].idxmax(), ["platform", "campaign", "CTR_%"]]
    return pc, best.set_index("platform"), attn.set_index("platform")


def print_advice(table, comp, best, attn, budget):
    print("\n=== MARKETING BUDGET RECOMMENDATION: next Rs {:,.0f} ===".format(budget))
    for p, r in table.sort_values("share_%", ascending=False).iterrows():
        print(f"  {p:<10} {r['share_%']:5.1f}%  Rs {r.PKR:>9,.0f}   "
              f"(90% range {r['share_low_%']:.0f}-{r['share_high_%']:.0f}%)")
    base = comp.set_index("option")
    col = "simulated_revenue" if "simulated_revenue" in base else "predicted_revenue"
    gain = base.loc["Recommended (engine)", col] - base.loc["Current mix (history)", col]
    print(f"\n  Expected revenue: Rs {base.loc['Recommended (engine)', col]:,.0f} "
          f"vs Rs {base.loc['Current mix (history)', col]:,.0f} with the current mix "
          f"(+Rs {gain:,.0f}, {gain / base.loc['Current mix (history)', col] * 100:+.1f}%).")
    print("\n  Creative advice (from campaign history):")
    for p in table.index:
        print(f"   - {p}: run '{best.loc[p, 'campaign']}' for sales (ROAS {best.loc[p, 'ROAS']:.1f}); "
              f"'{attn.loc[p, 'campaign']}' gets the most attention (CTR {attn.loc[p, 'CTR_%']:.2f}%).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/daraz_ads_1M.parquet")
    ap.add_argument("--budget", type=float, default=1_000_000)
    ap.add_argument("--no-validate", action="store_true")
    a = ap.parse_args()
    df = pd.read_csv(a.data) if a.data.endswith(".csv") else pd.read_parquet(a.data)
    table, comp, curves = recommend(df, a.budget, validate=not a.no_validate)
    pc, best, attn = creative_advice(df)
    os.makedirs("results", exist_ok=True)
    pcurve = profit_curve(curves)
    pcurve.to_csv("results/profit_by_budget.csv", index=False)
    best_b = pcurve.loc[pcurve.profit.idxmax()]
    print(f"\nProfit-maximising monthly budget (at {MARGIN:.0%} margin): Rs {best_b.budget:,.0f} "
          f"(profit Rs {best_b.profit:,.0f}); at Rs {a.budget:,.0f} profit is "
          f"Rs {pcurve.set_index('budget').profit.get(a.budget, float('nan')):,.0f}")
    table.to_csv("results/recommendation.csv")
    comp.to_csv("results/allocation_comparison.csv", index=False)
    curves.to_csv("results/response_curves.csv")
    pc.to_csv("results/platform_campaign_kpis.csv", index=False)
    print(table.round(2).to_string())
    print(comp.round(1).to_string())
    print_advice(table, comp, best, attn, a.budget)

"""Simulate Daraz Pakistan's digital-ad history across 5 platforms.

Each record is one ad set's delivery on one day: a (date, campaign, platform,
age, gender, city, device) slice with its impressions, clicks, spend,
conversions and revenue.

The daily budget of every platform is fixed by the seed and does NOT depend on
the number of records. A bigger N therefore means a *finer-grained* breakdown of
the same Rs ~22M of spend (more ad sets / segments), exactly like moving from a
monthly export to a daily-by-segment export of a real ad account. This keeps
the business totals comparable across 10K, 100K, 1M and 10M rows.

Usage:
    python src/generate_data.py --rows 10000 100000 1000000 10000000 --out data
"""
import argparse, os, time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

import config as C

PLAT = list(C.PLATFORMS)
CAMP = list(C.CAMPAIGNS)
COLUMNS = ["date", "campaign", "platform", "impressions", "clicks", "spend",
           "conversions", "revenue", "age", "gender", "city", "device"]


def calendar():
    days = pd.date_range(C.START, C.END, freq="D")
    d = days.date
    sale = np.zeros(len(d), bool)
    ram = np.zeros(len(d), bool)
    for s, e in C.SALE_WINDOWS:
        sale |= (d >= s) & (d <= e)
    for s, e in C.RAMADAN_WINDOWS:
        ram |= (d >= s) & (d <= e)
    return days, sale, ram


def daily_budget(seed=C.SEED):
    """Planned spend per day x platform (PKR). Independent of the row count."""
    rng = np.random.default_rng(seed)
    days, sale, ram = calendar()
    season = np.where(sale, C.SALE_SPEND_MULT, 1.0) * np.where(ram, C.RAMADAN_SPEND_MULT, 1.0)
    share = np.array([C.PLATFORMS[p]["share"] for p in PLAT])
    noise = rng.lognormal(0, C.DAILY_SPEND_NOISE, (len(days), len(PLAT)))
    S = season[:, None] * share[None, :] * noise
    n_months = len(days) / (365 / 12)
    S *= C.MONTHLY_BUDGET * n_months / S.sum()
    return days, sale, ram, S


def reference_spend():
    """Average daily spend per platform under the historical split (saturation anchor)."""
    return np.array([C.PLATFORMS[p]["share"] for p in PLAT]) * C.MONTHLY_BUDGET * 12 / 365


def _pick(rng, prob_rows, idx):
    """Vectorised categorical draw: row i uses probability vector prob_rows[idx[i]]."""
    cum = np.cumsum(prob_rows, axis=1)
    cum /= cum[:, -1:]
    u = rng.random(len(idx))[:, None]
    return (cum[idx] < u).sum(axis=1).astype(np.int16)


def _seg(table):
    names = list(table)
    arr = np.array(list(table.values()))  # columns: p, ctr_m, cvr_m, aov_m
    return names, arr


def simulate(n_rows, seed=C.SEED, budget=None):
    """budget: optional (days x platforms) PKR matrix that replaces the historical
    plan -- used to 'spend' a candidate allocation inside the simulator and see
    what revenue it really produces (an offline A/B test)."""
    rng = np.random.default_rng(seed + n_rows)
    days, sale, ram, S = daily_budget()
    if budget is not None:
        S = np.asarray(budget, float)
    D, P = S.shape

    # 1. which (day, platform) cell each record belongs to -- proportional to spend
    cell = rng.choice(D * P, size=n_rows, p=(S / S.sum()).ravel())
    d_idx, p_idx = cell // P, cell % P

    # 2. campaign, depends on the season of the day
    base_mix = np.array([C.CAMPAIGNS[c]["mix"] for c in CAMP])
    mix = np.tile(base_mix, (D, 1))
    i_sale, i_ram = CAMP.index("11.11 / 12.12 Mega Sale"), CAMP.index("Ramadan & Eid")
    mix[sale, i_sale] *= 4.0
    mix[~sale, i_sale] *= 0.5
    mix[ram, i_ram] *= 5.0
    mix[~ram, i_ram] *= 0.2
    c_idx = _pick(rng, mix, d_idx)

    # 3. audience segments, some skewed by platform
    age_n, age_t = _seg(C.AGES)
    gen_n, gen_t = _seg(C.GENDERS)
    city_n, city_t = _seg(C.CITIES)
    dev_n, dev_t = _seg(C.DEVICES)
    def plat_probs(t, skew):
        return np.array([t[:, 0] * np.array(skew.get(p, np.ones(len(t)))) for p in PLAT])
    a_idx = _pick(rng, plat_probs(age_t, C.AGE_SKEW), p_idx)
    g_idx = _pick(rng, plat_probs(gen_t, C.GENDER_SKEW), p_idx)
    ci_idx = _pick(rng, city_t[:, 0][None, :], np.zeros(n_rows, int))
    dv_idx = _pick(rng, plat_probs(dev_t, C.DEVICE_SKEW), p_idx)

    # 4. split each cell's budget across its records
    w = rng.lognormal(0, 0.6, n_rows)
    W = np.bincount(cell, weights=w, minlength=D * P)
    spend = S.ravel()[cell] * w / W[cell]
    realised = np.bincount(cell, weights=spend, minlength=D * P).reshape(D, P)

    # 5. delivery funnel
    prm = {k: np.array([C.PLATFORMS[p][k] for p in PLAT]) for k in ["cpm", "ctr", "cvr", "aov", "b"]}
    cm = {k: np.array([C.CAMPAIGNS[c][k] for c in CAMP]) for k in ["ctr_m", "cvr_m", "aov_m"]}

    cpm = prm["cpm"][p_idx] * rng.lognormal(0, 0.15, n_rows)
    impressions = rng.poisson(spend / cpm * 1000)

    ctr = (prm["ctr"][p_idx] * cm["ctr_m"][c_idx] * age_t[a_idx, 1] * gen_t[g_idx, 1]
           * city_t[ci_idx, 1] * dev_t[dv_idx, 1])
    yt_comedy = (p_idx == PLAT.index("YouTube")) & (c_idx == CAMP.index("Hansna Mana Hai (Comedy)"))
    ctr = np.where(yt_comedy, ctr * C.YOUTUBE_COMEDY_CTR_BOOST, ctr)
    clicks = rng.binomial(impressions, np.clip(ctr, 0, 0.5))

    # diminishing returns: the more a platform is pushed on a day, the lower its CVR
    sat = (realised[d_idx, p_idx] / reference_spend()[p_idx]) ** (prm["b"][p_idx] - 1)
    season_cvr = (np.where(sale, C.SALE_CVR_MULT, 1.0) * np.where(ram, C.RAMADAN_CVR_MULT, 1.0))[d_idx]
    cvr = (prm["cvr"][p_idx] * cm["cvr_m"][c_idx] * season_cvr * age_t[a_idx, 2]
           * gen_t[g_idx, 2] * city_t[ci_idx, 2] * dev_t[dv_idx, 2] * sat)
    conversions = rng.binomial(clicks, np.clip(cvr, 0, 0.9))

    aov = (prm["aov"][p_idx] * cm["aov_m"][c_idx] * age_t[a_idx, 3] * gen_t[g_idx, 3]
           * city_t[ci_idx, 3] * dev_t[dv_idx, 3])
    sigma = 0.35 / np.sqrt(np.maximum(conversions, 1))
    revenue = np.round(conversions * aov * rng.lognormal(-sigma ** 2 / 2, sigma))

    cat = pd.Categorical.from_codes   # categoricals keep 10M rows within laptop RAM
    df = pd.DataFrame({
        "date": cat(d_idx, list(days.strftime("%Y-%m-%d"))),
        "campaign": cat(c_idx, CAMP),
        "platform": cat(p_idx, PLAT),
        "impressions": impressions.astype(np.int64),
        "clicks": clicks.astype(np.int64),
        "spend": np.round(spend, 2),
        "conversions": conversions.astype(np.int64),
        "revenue": revenue.astype(np.int64),
        "age": cat(a_idx, age_n),
        "gender": cat(g_idx, gen_n),
        "city": cat(ci_idx, city_n),
        "device": cat(dv_idx, dev_n),
    })
    return df.sort_values(["date", "platform", "campaign"], kind="stable").reset_index(drop=True)


def label(n):
    return {10_000: "10K", 100_000: "100K", 1_000_000: "1M", 10_000_000: "10M"}.get(n, str(n))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, nargs="+", default=[10_000, 100_000, 1_000_000, 10_000_000])
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for n in a.rows:
        t = time.time()
        df = simulate(n)
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(tbl, f"{a.out}/daraz_ads_{label(n)}.parquet")
        # CSV is written in 1M-row slices with dictionary columns decoded to text
        with pacsv.CSVWriter(f"{a.out}/daraz_ads_{label(n)}.csv",
                             pa.schema([(f.name, pa.string() if pa.types.is_dictionary(f.type) else f.type)
                                        for f in tbl.schema])) as w:
            for off in range(0, tbl.num_rows, 1_000_000):
                part = tbl.slice(off, 1_000_000)
                cols = [c.cast(pa.string()) if pa.types.is_dictionary(c.type) else c for c in part.columns]
                w.write_table(pa.Table.from_arrays(cols, names=part.column_names))
        mb = os.path.getsize(f"{a.out}/daraz_ads_{label(n)}.csv") / 1e6
        print(f"{label(n):>5}: {len(df):>10,} rows  spend Rs {df.spend.sum():>12,.0f}  "
              f"revenue Rs {df.revenue.sum():>13,.0f}  CSV {mb:8.1f} MB  ({time.time()-t:.1f}s)")

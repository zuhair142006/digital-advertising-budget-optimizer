"""Make every chart in charts/ from the files in results/ and data/.

    python src/charts.py
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from generate_data import PLAT

# Validated categorical palette (fixed order; colour follows the entity, never its rank)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PCOL = dict(zip(PLAT, SERIES))                       # platform -> colour
ECOL = {"pandas": SERIES[0], "mapreduce": SERIES[1], "spark": SERIES[2]}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False, "figure.dpi": 150,
})
os.makedirs("charts", exist_ok=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"charts/{name}.png", dpi=170)
    plt.close(fig)


def platform_kpis():
    from kpis import kpi_table
    df = pd.read_parquet("data/daraz_ads_1M.parquet")
    k = kpi_table(df, "platform").reindex(PLAT)
    metrics = [("ROAS", "ROAS (Rs revenue per Rs 1)", "{:.1f}x"), ("CPA", "CPA (Rs per order)", "Rs {:,.0f}"),
               ("CTR_%", "CTR (%)", "{:.2f}%"), ("CVR_%", "Conversion rate (%)", "{:.2f}%")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (m, title, fmt) in zip(axes, metrics):
        ax.barh(PLAT[::-1], k[m][::-1], color=[PCOL[p] for p in PLAT[::-1]], height=0.6)
        for i, v in enumerate(k[m][::-1]):
            ax.text(v, i, " " + fmt.format(v), va="center", color=INK, fontsize=9)
        ax.set_title(title, fontsize=11); ax.grid(axis="y", visible=False)
        ax.set_xlim(0, k[m].max() * 1.35)
    fig.suptitle("Platform performance, Jan 2025 - Oct 2026 (simulated, 1M records)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    save(fig, "01_platform_kpis")


def campaign_heatmap():
    pc = pd.read_csv("results/platform_campaign_kpis.csv")
    m = pc.pivot(index="campaign", columns="platform", values="ROAS")[PLAT]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    im = ax.imshow(m.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(PLAT)), PLAT); ax.set_yticks(range(len(m)), m.index)
    ax.grid(False)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.1f}x", ha="center", va="center", fontsize=9,
                    color="white" if v > m.values.max() * 0.55 else INK)
    ax.set_title("ROAS by campaign and platform")
    fig.colorbar(im, ax=ax, label="ROAS", shrink=0.8)
    save(fig, "02_campaign_roas_heatmap")


def benchmark():
    b = pd.read_csv("results/benchmark.csv")
    sizes = ["10K", "100K", "1M", "10M"]
    xs = [1e4, 1e5, 1e6, 1e7]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    names = {"pandas": "Pandas", "mapreduce": "MapReduce", "spark": "Spark"}
    for ax, col, title in [(axes[0], "wall_s", "End-to-end time (seconds, log scale)"),
                           (axes[1], "peak_mem_mb", "Peak memory (MB, log scale)")]:
        ends = []
        for eng in ["pandas", "mapreduce", "spark"]:
            for fmt, ls in [("csv", "-"), ("parquet", "--")]:
                s = b[(b.engine == eng) & (b.format == fmt)].set_index("size").reindex(sizes)[col]
                if s.isna().all():
                    continue
                ax.plot(xs, s.values, ls, color=ECOL[eng], lw=2, marker="o", ms=6,
                        label=f"{names[eng]} ({fmt.upper()})")
                ends.append(s.values[-1])
        # end labels, nudged apart in log space so they never collide
        ax.set_xscale("log"); ax.set_yscale("log")
        order = np.argsort(ends)
        pos = np.log10(np.array(ends, float))
        lo, hi = ax.get_ylim()
        gap = (np.log10(hi) - np.log10(lo)) * 0.055
        placed = pos.copy()
        for i in range(1, len(order)):
            a, z = order[i - 1], order[i]
            if placed[z] - placed[a] < gap:
                placed[z] = placed[a] + gap
        for v, y in zip(ends, placed):
            ax.annotate(f"{v:,.0f}" if v >= 10 else f"{v:.1f}", (xs[-1], v), xytext=(xs[-1] * 1.25, 10 ** y),
                        va="center", fontsize=8, color=INK, annotation_clip=False)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(7e3, 2.2e7)
        ax.set_xticks(xs, ["10K", "100K", "1M", "10M"]); ax.set_xlabel("records")
        ax.set_title(title, fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Same KPI query, three engines (2-core machine, 7 GB RAM)",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig("charts/03_benchmark.png", dpi=170)
    plt.close(fig)


def response_curves():
    from recommender import DAYS_IN_PLAN
    cur = pd.read_csv("results/response_curves.csv", index_col=0)
    rec = pd.read_csv("results/recommendation.csv", index_col=0)
    import config as C
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.linspace(500, 20000, 300)
    for p in PLAT:
        y = cur.loc[p, "k"] * cur.loc[p, "b"] * x ** (cur.loc[p, "b"] - 1)
        ax.plot(x, y, color=PCOL[p], lw=2, label=p)
        xn = C.PLATFORMS[p]["share"] * 1_000_000 / DAYS_IN_PLAN
        xr = rec.loc[p, "PKR"] / DAYS_IN_PLAN
        mn = cur.loc[p, "k"] * cur.loc[p, "b"] * xn ** (cur.loc[p, "b"] - 1)
        mr = cur.loc[p, "k"] * cur.loc[p, "b"] * xr ** (cur.loc[p, "b"] - 1)
        ax.plot(xn, mn, "o", ms=8, mfc=SURF, mec=PCOL[p], mew=2)
        ax.plot(xr, mr, "o", ms=8, color=PCOL[p], mec=SURF, mew=2)
    ax.axhline(4, color=INK2, lw=1, ls=":")
    ax.text(20000, 4.1, "break-even marginal ROAS = 4.0 (25% margin)", ha="right", va="bottom",
            fontsize=8, color=INK2)
    ax.set_ylim(0, 12); ax.set_xlim(0, 20000)
    ax.set_xlabel("daily spend on the platform (Rs)"); ax.set_ylabel("revenue from the next Rs 1")
    ax.set_title("Diminishing returns: marginal ROAS falls as spend rises")
    ax.plot([], [], "o", mfc=SURF, mec=INK2, mew=2, label="current mix")
    ax.plot([], [], "o", color=INK2, label="recommended")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "04_marginal_roas_curves")


def allocation():
    rec = pd.read_csv("results/recommendation.csv", index_col=0).reindex(PLAT)
    import config as C
    cur = np.array([C.PLATFORMS[p]["share"] * 100 for p in PLAT])
    fig, ax = plt.subplots(figsize=(8.5, 4))
    y = np.arange(len(PLAT))
    ax.barh(y + 0.2, cur, height=0.36, color=GRID, label="current mix")
    ax.barh(y - 0.2, rec["share_%"], height=0.36, color=[PCOL[p] for p in PLAT], label="recommended")
    ax.errorbar(rec["share_%"], y - 0.2,
                xerr=[rec["share_%"] - rec["share_low_%"], rec["share_high_%"] - rec["share_%"]],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for i, p in enumerate(PLAT):
        ax.text(rec.loc[p, "share_high_%"] + 0.8, i - 0.2,
                f"{rec.loc[p, 'share_%']:.0f}%  (Rs {rec.loc[p, 'PKR']/1000:,.0f}K)", va="center", fontsize=9, color=INK)
        ax.text(cur[i] + 0.8, i + 0.2, f"{cur[i]:.0f}%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, PLAT); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 60); ax.set_xlabel("share of the Rs 1M budget (%)  - whiskers: 90% bootstrap range")
    ax.set_title("Where the next Rs 1 million should go")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "05_recommended_allocation")


def options():
    c = pd.read_csv("results/allocation_comparison.csv")
    c = c.sort_values("simulated_revenue")
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    col = [SERIES[0] if o.startswith("Recommended") else "#b9b8b3" for o in c.option]
    ax.barh(c.option, c.simulated_revenue / 1e6, xerr=c.simulated_sd / 1e6, color=col, height=0.55,
            ecolor=INK, capsize=3)
    for i, (v, sd, r) in enumerate(zip(c.simulated_revenue, c.simulated_sd, c.ROAS)):
        ax.text((v + sd) / 1e6 + 0.08, i, f"Rs {v/1e6:.2f}M  (ROAS {r:.2f}x)", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, c.simulated_revenue.max() / 1e6 * 1.4); ax.grid(axis="y", visible=False)
    ax.set_xlabel("revenue from Rs 1M over 30 days, spent inside the simulator (Rs million)")
    ax.set_title("Offline A/B test: four ways to split Rs 1M")
    save(fig, "06_allocation_ab_test")


if __name__ == "__main__":
    for f in [platform_kpis, campaign_heatmap, response_curves, allocation, options, benchmark]:
        try:
            f(); print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)

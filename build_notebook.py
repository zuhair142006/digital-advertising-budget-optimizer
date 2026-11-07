"""Builds notebook.ipynb (run once; the notebook itself is the deliverable)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Project 2 — Digital Advertising Budget Optimizer (Daraz Pakistan)
**Continues Project 1: *Pakistani Social Media Content Detective*.**

Project 1 measured *what content* Daraz Pakistan's audience engages with on YouTube (comedy/entertainment ≈ 4.8× the engagement of sale videos).
Project 2 asks management's next question: **where should Daraz spend the next Rs 1 million across Facebook, Instagram, Google, TikTok and YouTube?**

| Step | What happens | Code |
|---|---|---|
| 1 | Simulate 22 months of Daraz ad history (10K → 100K → 1M → 10M records) | `src/generate_data.py` |
| 2 | KPIs: CTR, CPC, CPA, conversion rate, ROAS, revenue, campaign performance | `src/kpis.py` |
| 3 | Big-data benchmark: Pandas vs MapReduce (mrjob) vs Spark | `src/run_engine.py`, `src/mapreduce_job.py`, `src/benchmark.py` |
| 4 | Marketing Budget Recommendation Engine | `src/recommender.py` |

> **Colab:** `Runtime → Run all`. No API key is needed — all data is simulated. Spark needs Java, which Colab already has."""),
code("""# Setup (Colab: clone the repo first)
import os, sys
if not os.path.exists("src"):
    !git clone -q https://github.com/zuhair142006/daraz-ad-budget-optimizer.git
    %cd daraz-ad-budget-optimizer
!pip install -q pyspark mrjob pyarrow psutil standard-pipes 2>/dev/null
sys.path.insert(0, "src")
import pandas as pd, numpy as np
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
from IPython.display import Image, display"""),
md("""## 1. Data simulation
Meta, TikTok and Google Ads data needs ad-account access (the same wall that stopped Meta data in Project 1), so the history is **simulated** from assumptions in `src/config.py`:

* **5 platforms**, each with its own CPM, CTR, conversion rate, order value and **diminishing returns** (Google search saturates fastest, because search volume is limited).
* **5 campaigns** taken from Daraz.pk's real YouTube content types found in Project 1: *Hansna Mana Hai* comedy, 11.11/12.12 Mega Sale, Ramadan & Eid, Seller Stories/How-to, Always-on Catalogue. Project 1's finding is built in: comedy earns more attention (CTR ×1.6, ×2.1 on YouTube) but converts less directly.
* **Pakistani calendar:** 11.11, 12.12, Pakistan Day sale, Ramadan/Eid 2025–26.
* **Audience:** age, gender, 11 Pakistani cities, Android/iOS/Desktop.

The daily budget (≈ Rs 1M/month, Rs 22M total) is fixed. A bigger dataset is a *finer breakdown* of the same spend, so business totals stay comparable across sizes."""),
code("""!python src/generate_data.py --rows 10000 100000 1000000 10000000 --out data"""),
code("""df = pd.read_parquet("data/daraz_ads_1M.parquet")
print(df.shape); df.head()"""),
md("## 2. KPIs\nAll KPIs are *ratios of sums* (total clicks ÷ total impressions), never averages of row-level ratios."),
code("""from kpis import kpi_table
cols = ["spend","revenue","conversions","CTR_%","CPC","CPA","CVR_%","ROAS","spend_share_%"]
plat = kpi_table(df, "platform").sort_values("ROAS", ascending=False)
plat[cols]"""),
code("""kpi_table(df, "campaign").sort_values("ROAS", ascending=False)[cols]"""),
code("""# Campaign performance: platform x campaign ROAS
kpi_table(df, ["platform","campaign"])["ROAS"].unstack().round(2)"""),
code("""# Audience: where does a rupee work hardest?
for dim in ["city","device","age","gender"]:
    display(kpi_table(df, dim).sort_values("ROAS", ascending=False)[["spend","CTR_%","CVR_%","CPA","ROAS"]].round(2))"""),
code("""# Monthly trend (seasonality: Ramadan + 11.11 / 12.12 peaks)
m = df.assign(month=df.date.astype(str).str[:7])
kpi_table(m, "month")[["spend","revenue","ROAS","CPA"]]"""),
md("## 3. Big Data simulation: Pandas vs MapReduce vs Spark\nSame query in all three engines — group by platform × campaign, sum the 5 volume columns, compute CTR/CPC/CPA/CVR/ROAS. First check that all three give **identical** answers:"),
code("""for e in ["pandas","mapreduce","spark"]:
    !python src/run_engine.py --engine {e} --file data/daraz_ads_100K.csv 2>/dev/null
outs = {e: pd.read_csv(f"results/engine_output_{e}_100K_csv.csv") for e in ["pandas","mapreduce","spark"]}
for e in ["mapreduce","spark"]:
    diff = (outs["pandas"][["impressions","clicks","conversions","revenue"]] - outs[e][["impressions","clicks","conversions","revenue"]]).abs().values.max()
    print(f"pandas vs {e}: max difference in totals = {diff}, ROAS diff = {(outs['pandas'].ROAS-outs[e].ROAS).abs().max():.1e}")"""),
code("""# Full benchmark takes ~20 min on 2 cores. Set RUN_BENCHMARK = True to re-run it.
RUN_BENCHMARK = False
if RUN_BENCHMARK:
    !python src/benchmark.py
bench = pd.read_csv("results/benchmark.csv")
bench.pivot_table(index=["engine","format"], columns="size", values="wall_s")[["10K","100K","1M","10M"]]"""),
code("""bench.pivot_table(index=["engine","format"], columns="size", values="peak_mem_mb")[["10K","100K","1M","10M"]]"""),
md("""**Reading the benchmark** (2-core, 7 GB machine; median of 3 runs, 1 run at 10M)
* **Pandas wins at every size up to 10M** (7.1 s from CSV, 1.5 s from Parquet), because the data still fits in RAM and there is no start-up cost. But its memory grows in line with the data (118 MB → 984 MB).
* **Spark** pays a fixed ~5.5 s to start the JVM, which dominates small data (11 s even for 10K rows). Its *work* time grows only 2.3× from 1M to 10M rows (6.3 s → 14.6 s), while Pandas grows 11×. Spark is built for data bigger than one machine.
* **MapReduce** (mrjob, Hadoop-streaming model) is the slowest at 10M (56 s), because every row is passed as text through Python and sorted on disk between map and reduce. Its memory is the flattest (353 MB up to 1M rows).
* **Parquet** (columnar) beats CSV for every engine: 40 MB vs 967 MB on disk, and only the needed columns are read.
* See the 50M-row stress test below for where Pandas breaks."""),
code("""import json
print(json.dumps(json.load(open("results/stress_test_50M.json")), indent=2))"""),
code("""!python src/charts.py >/dev/null 2>&1 || true
display(Image("charts/03_benchmark.png"))"""),
md("""## 4. Marketing Budget Recommendation Engine
1. Daily spend & revenue per platform → fit `log(revenue) = log k + b·log(spend) + season dummies`.
2. *b* < 1 is diminishing returns. Each *b* is shrunk toward the cross-platform mean in proportion to its standard error, and capped at 0.95, because a noisy *b* ≈ 1 would send the whole budget to one platform (see the robustness table).
3. The optimiser splits the budget so the **next rupee earns the same everywhere** (equal marginal ROAS), within 5–45% guard-rails per platform.
4. Bootstrap (300 resamples of days). The recommendation is the **average** of the 300 allocations (bagging), and their spread gives a 90% range.
5. **Offline A/B test**: each candidate split is actually spent for 30 days inside the simulator.

The engine runs on the full **10M-record** dataset."""),
code("""from recommender import recommend, creative_advice, print_advice, profit_curve
df10 = pd.read_parquet("data/daraz_ads_10M.parquet")
table, comp, curves = recommend(df10, budget=1_000_000)
table[["share_%","PKR","share_low_%","share_high_%","past_ROAS","marginal_ROAS_now","marginal_ROAS_rec","b_raw","b_se","b_shrunk","b_true"]]"""),
code("""comp"""),
code("""pc, best, attn = creative_advice(df10)
print_advice(table, comp, best, attn, 1_000_000)"""),
md("""**Robustness.** The same engine on each of the four datasets. Same budget, different random draws. The original version (no cap, no shrinkage) broke on 10M: TikTok's *b* came out at 1.00 by chance, the engine put 45% into TikTok, and revenue *fell* below the current mix."""),
code("""pd.read_csv("results/robustness_by_dataset.csv")"""),
md("""**Is Rs 1M the right amount?** Revenue here is GMV. If Daraz keeps ~25% of it as contribution margin (assumption: `MARGIN` in `kpis.py`), profit after ad cost peaks well below Rs 1M:"""),
code("""pcurve = profit_curve(curves)
best_b = pcurve.loc[pcurve.profit.idxmax()]
print(f"Profit-maximising budget: Rs {best_b.budget:,.0f}  (profit Rs {best_b.profit:,.0f})")
print(f"At Rs 1,000,000: profit Rs {pcurve.set_index('budget').profit[1_000_000]:,.0f}")
pcurve.iloc[::3]"""),
code("""!python src/charts.py
for f in ["01_platform_kpis","02_campaign_roas_heatmap","04_marginal_roas_curves","05_recommended_allocation","06_allocation_ab_test"]:
    display(Image(f"charts/{f}.png"))"""),
md("""## 5. Limitations
* **Simulated data.** The engine is only as good as the assumptions in `config.py`. Its value is the *method*: plug in a real Meta/Google/TikTok export with the same 12 columns and it runs unchanged.
* **Last-click ROAS undervalues YouTube and TikTok.** Project 1 showed entertainment content builds engagement; that brand effect is not in same-day revenue. The 5% floor keeps those channels alive.
* The response curve assumes flat daily pacing and normal (non-sale) days. A sale month lifts every platform, and campaigns should be re-planned around 11.11.
* The benchmark ran on one 2-core, 7 GB machine. A real Hadoop/Spark cluster changes the picture for data larger than RAM."""),
]
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "notebook.ipynb")
print("notebook.ipynb written")

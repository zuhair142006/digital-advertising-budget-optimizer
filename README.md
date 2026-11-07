# Digital Advertising Budget Optimizer — Daraz Pakistan



**Management's question:** where should Daraz spend the next **Rs 1 million** across Facebook, Instagram, Google, TikTok and YouTube?

## Answer

| Platform | Recommended | Rs | 90% range | Current mix |
|---|---|---|---|---|
| Google | **37%** | 365,900 | 28–45% | 20% |
| Facebook | **31%** | 310,200 | 24–41% | 30% |
| Instagram | **22%** | 223,900 | 16–32% | 20% |
| TikTok | **5%** | 50,000 | floor | 15% |
| YouTube | **5%** | 50,000 | floor | 15% |

In an offline A/B test (each split actually spent for 30 days inside the simulator, 5 runs each), the recommended split earned **Rs 4.77M** of revenue, against Rs 4.49M for the current mix (+6.2%), Rs 4.64M for "proportional to past ROAS" and Rs 4.31M for an equal split.

## What's in the repo

| File | What it does |
|---|---|
| `notebook.ipynb` | The whole project, run top to bottom (Colab-ready) |
| `src/config.py` | All simulation assumptions (platforms, campaigns, Pakistani calendar, audience) |
| `src/generate_data.py` | Simulator: 10K / 100K / 1M / 10M records, 12 required columns |
| `src/kpis.py` | CTR, CPC, CPA, CVR, ROAS, CPM, profit (always ratio of sums) |
| `src/run_engine.py` | Same KPI query in **Pandas**, **MapReduce** and **Spark** |
| `src/mapreduce_job.py` | mrjob MapReduce job (map → combine → shuffle → reduce); runs on Hadoop/EMR with `-r hadoop` / `-r emr` |
| `src/benchmark.py` | Times and peak memory for each engine × size × format |
| `src/stress_test.py` | 50M rows under a 6 GB memory limit: where Pandas breaks |
| `src/recommender.py` | Budget Recommendation Engine |
| `src/charts.py` | All charts in `charts/` |
| `results/` | Every table the report quotes (CSV / JSON) |
| `data/daraz_ads_10K.csv` | Sample data. Larger files are regenerated: `python src/generate_data.py` |

## How to run

```bash
pip install pandas pyarrow pyspark mrjob psutil matplotlib standard-pipes   # Java 11+ needed for Spark
python src/generate_data.py                                # 10K, 100K, 1M, 10M (≈1.1 GB on disk)
python src/benchmark.py                                    # ≈20 min on 2 cores
python src/recommender.py --data data/daraz_ads_10M.parquet --budget 1000000
# quick test on the bundled sample (no data generation needed; saves to results/, so run in a copy if you want to keep the 10M results):
python src/recommender.py --data data/daraz_ads_10K.csv --budget 1000000
python src/charts.py
```
`standard-pipes` is only needed on Python ≥ 3.13 (mrjob still imports the removed `pipes` module).

## Data

The data is simulated, because Meta, TikTok and Google Ads data needs ad-account access (the same wall that kept Meta out of Project 1). The 12 columns are the brief's: `date, campaign, platform, impressions, clicks, spend, conversions, revenue, age, gender, city, device`.

* Jan 2025 – Oct 2026, ≈ Rs 1M/month (Rs 22M total), with 11.11, 12.12, Pakistan Day and Ramadan/Eid seasons.
* Campaigns are Daraz's real YouTube content types from Project 1. Project 1's finding that comedy beats sale content for engagement is built in as a CTR effect.
* Each platform has its own diminishing-returns elasticity *b*, known only to the simulator, so the engine's estimates can be checked against the truth.
* Bigger datasets split the **same** budget more finely, so totals stay comparable across sizes.

## Limitations

The data is simulated; plug in a real ad-account export with the same 12 columns. ROAS is last-click, which undervalues YouTube and TikTok brand building. The benchmark ran on one 2-core machine, not a cluster. Revenue is GMV; the profit view assumes a 25% margin.

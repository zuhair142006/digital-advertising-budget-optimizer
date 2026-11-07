"""MapReduce version of the campaign-performance query, written with mrjob.

Query: total impressions, clicks, spend, conversions and revenue for every
(platform, campaign) pair, then CTR, CPC, CPA, CVR and ROAS.

    map      : one CSV line  ->  key "platform|campaign", value [imp, clk, spend, conv, rev]
    combine  : partial sums on each mapper (cuts shuffle traffic)
    shuffle  : the framework sorts and groups values by key
    reduce   : final sums  ->  KPIs

Run locally (Hadoop-style multi-process simulation):
    python src/mapreduce_job.py -r local data/daraz_ads_1M.csv
Run on a real Hadoop cluster / Amazon EMR with no code changes:
    python src/mapreduce_job.py -r hadoop hdfs:///daraz/daraz_ads_10M.csv
    python src/mapreduce_job.py -r emr   s3://bucket/daraz_ads_10M.csv
"""
import csv
from mrjob.job import MRJob

# column positions in the CSV (see generate_data.COLUMNS)
PLATFORM, CAMPAIGN, IMP, CLK, SPEND, CONV, REV = 2, 1, 3, 4, 5, 6, 7


def _kpis(imp, clk, spend, conv, rev):
    d = lambda a, b: a / b if b else None
    return {"CTR_%": 100 * d(clk, imp) if imp else None, "CPC": d(spend, clk),
            "CPA": d(spend, conv), "CVR_%": 100 * d(conv, clk) if clk else None,
            "ROAS": d(rev, spend)}


class CampaignKPIJob(MRJob):

    def mapper(self, _, line):
        if line.startswith('"date"'):          # header
            return
        f = next(csv.reader([line]))
        yield f"{f[PLATFORM]}|{f[CAMPAIGN]}", [int(f[IMP]), int(f[CLK]), float(f[SPEND]),
                                              int(f[CONV]), int(f[REV])]

    def combiner(self, key, values):
        tot = [0, 0, 0.0, 0, 0]
        for v in values:
            for i in range(5):
                tot[i] += v[i]
        yield key, tot

    def reducer(self, key, values):
        tot = [0, 0, 0.0, 0, 0]
        for v in values:
            for i in range(5):
                tot[i] += v[i]
        platform, campaign = key.split("|")
        yield key, {"platform": platform, "campaign": campaign,
                    "impressions": tot[0], "clicks": tot[1], "spend": round(tot[2], 2),
                    "conversions": tot[3], "revenue": tot[4], **_kpis(*tot)}


if __name__ == "__main__":
    CampaignKPIJob.run()

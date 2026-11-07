"""Beyond 10M: where does single-machine Pandas break?

Builds a 50M-row CSV (the 10M file stacked 5 times, ~4.8 GB) and runs, with the
process limited to 6 GB of memory (like a typical student laptop):
  * Pandas, naive      : pd.read_csv(path)  -- every column, default dtypes
  * Pandas, optimised  : only the 7 needed columns, categorical keys
  * Spark (local[*])   : same query, streams the file in partitions

    python src/stress_test.py
"""
import json, os, resource, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, BIG = "data/daraz_ads_10M.csv", "data/daraz_ads_50M.csv"
LIMIT = 6 * 1024 ** 3

NAIVE = """
import pandas as pd, time
t = time.perf_counter()
df = pd.read_csv("{p}")
g = df.groupby(["platform","campaign"])[["impressions","clicks","spend","conversions","revenue"]].sum()
print(len(g), time.perf_counter() - t)
"""


def limited():
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))


def run(cmd, java=False):
    t = time.perf_counter()
    env = dict(os.environ)
    # The JVM reserves virtual memory up-front, so cap it with -Xmx instead of RLIMIT_AS
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       preexec_fn=None if java else limited)
    ok = p.returncode == 0
    err = "" if ok else (p.stderr.strip().splitlines() or ["killed"])[-1][:120]
    return dict(ok=ok, seconds=round(time.perf_counter() - t, 1), error=err)


if __name__ == "__main__":
    if not os.path.exists(BIG):
        with open(BIG, "wb") as out, open(SRC, "rb") as f:
            header = f.readline(); out.write(header); body_start = f.tell()
            for _ in range(5):
                f.seek(body_start)
                while chunk := f.read(64 * 1024 * 1024):
                    out.write(chunk)
    size_gb = os.path.getsize(BIG) / 1e9
    res = {
        "rows": 50_000_000, "csv_gb": round(size_gb, 2), "memory_limit_gb": 6,
        "pandas_naive": run([sys.executable, "-c", NAIVE.format(p=BIG)]),
        "pandas_optimised": run([sys.executable, f"{HERE}/run_engine.py", "--engine", "pandas",
                                 "--file", BIG, "--save", "results/stress"]),
        "spark": run([sys.executable, f"{HERE}/run_engine.py", "--engine", "spark",
                      "--file", BIG, "--save", "results/stress"], java=True),
    }
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/stress_test_50M.json", "w"), indent=2)
    print(json.dumps(res, indent=2))

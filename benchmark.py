"""Benchmark Pandas vs MapReduce (mrjob) vs Spark on 10K -> 10M records.

Every run is a fresh subprocess so that one engine's memory cannot help the next.
Peak memory is the peak combined RSS of the whole process tree (Python + Spark's
JVM + mrjob's worker processes), sampled every 50 ms.

    python src/benchmark.py            # writes results/benchmark.csv
"""
import json, os, subprocess, sys, threading, time
import pandas as pd
import psutil

SIZES = ["10K", "100K", "1M", "10M"]
RUNS = [("pandas", "csv"), ("mapreduce", "csv"), ("spark", "csv"),
        ("pandas", "parquet"), ("spark", "parquet")]
REPEATS = {"10K": 3, "100K": 3, "1M": 3, "10M": 1}
HERE = os.path.dirname(os.path.abspath(__file__))


def run_once(engine, path):
    p = subprocess.Popen([sys.executable, f"{HERE}/run_engine.py", "--engine", engine, "--file", path],
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    peak = [0]
    def watch():
        proc = psutil.Process(p.pid)
        while p.poll() is None:
            try:
                rss = proc.memory_info().rss + sum(c.memory_info().rss for c in proc.children(recursive=True))
                peak[0] = max(peak[0], rss)
            except psutil.Error:
                pass
            time.sleep(0.05)
    th = threading.Thread(target=watch); th.start()
    t = time.perf_counter()
    out, _ = p.communicate()
    wall = time.perf_counter() - t
    th.join()
    res = json.loads(out.strip().splitlines()[-1])
    res.update(wall_s=wall, peak_mem_mb=peak[0] / 1e6)
    return res


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    rows = []
    for size in SIZES:
        for engine, fmt in RUNS:
            path = f"data/daraz_ads_{size}.{fmt}"
            for r in range(REPEATS[size]):
                res = run_once(engine, path)
                res.update(size=size, format=fmt, repeat=r)
                rows.append(res)
                print(f"{size:>5} {engine:>9} {fmt:>7}  wall {res['wall_s']:7.2f}s  "
                      f"query {res['query_s']+res['load_s']:7.2f}s  mem {res['peak_mem_mb']:7.0f} MB", flush=True)
                pd.DataFrame(rows).to_csv("results/benchmark_raw.csv", index=False)
    raw = pd.DataFrame(rows)
    raw["work_s"] = raw.load_s + raw.query_s
    summ = raw.groupby(["size", "engine", "format"], sort=False).agg(
        wall_s=("wall_s", "median"), work_s=("work_s", "median"),
        startup_s=("startup_s", "median"), peak_mem_mb=("peak_mem_mb", "median")).reset_index()
    summ.to_csv("results/benchmark.csv", index=False)
    print(summ.round(2).to_string())

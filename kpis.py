"""KPI definitions shared by every engine (Pandas, MapReduce, Spark).

    CTR  (click-through rate) = clicks / impressions
    CPC  (cost per click)     = spend / clicks
    CPA  (cost per acquisition)= spend / conversions
    CVR  (conversion rate)    = conversions / clicks
    ROAS (return on ad spend) = revenue / spend
    CPM  (cost per 1,000 imp) = spend / impressions * 1000
    Profit after ad cost      = revenue * margin - spend   (Daraz take-rate assumption)

KPIs are always computed from *summed* totals (ratio of sums), never by averaging
row-level ratios -- averaging ratios would give tiny ad sets the same weight as
big ones and produce wrong numbers.
"""
import pandas as pd

SUM_COLS = ["impressions", "clicks", "spend", "conversions", "revenue"]
MARGIN = 0.25   # assumed gross margin Daraz keeps per Rs of GMV, for profit-ROAS


def _safe(a, b):
    return a / b if b else float("nan")


def kpi_row(imp, clk, spend, conv, rev):
    """Plain-Python KPI calculation (used by the MapReduce reducer)."""
    return {
        "CTR_%": 100 * _safe(clk, imp),
        "CPC": _safe(spend, clk),
        "CPA": _safe(spend, conv),
        "CVR_%": 100 * _safe(conv, clk),
        "ROAS": _safe(rev, spend),
    }


def add_kpis(g: pd.DataFrame) -> pd.DataFrame:
    """Add KPI columns to a DataFrame of summed totals."""
    g = g.copy()
    g["CTR_%"] = 100 * g.clicks / g.impressions
    g["CPC"] = g.spend / g.clicks
    g["CPA"] = g.spend / g.conversions
    g["CVR_%"] = 100 * g.conversions / g.clicks
    g["ROAS"] = g.revenue / g.spend
    g["CPM"] = 1000 * g.spend / g.impressions
    g["profit"] = g.revenue * MARGIN - g.spend
    g["spend_share_%"] = 100 * g.spend / g.spend.sum()
    return g


def kpi_table(df: pd.DataFrame, by) -> pd.DataFrame:
    g = df.groupby(by, observed=True)[SUM_COLS].sum()
    return add_kpis(g)

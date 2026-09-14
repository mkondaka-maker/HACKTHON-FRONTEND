"""Deterministic data helpers for Peer Financial Intelligence visuals.

All helpers read the normalized dataframe (or financial-engine output)
and never fabricate values. Selections that cannot be computed return
available=False with a reason so the UI shows an informational message.

Peer definition (unchanged everywhere): same category + same selected
year, excluding the selected company from averages. Rankings and
scatter plots include the selected company; averages exclude it.
"""

import pandas as pd
import numpy as np


RANK_METRICS = [
    ("Revenue", "revenue", "money"),
    ("Net Income", "net_income", "money"),
    ("Net Profit Margin", "net_profit_margin", "pct"),
    ("ROA", "roa", "pct"),
    ("ROE", "roe", "pct"),
    ("Current Ratio", "current_ratio", "x"),
    ("Debt / Equity", "debt_equity_ratio", "x"),
]

TREND_METRICS = [
    ("Revenue", "revenue", "money"),
    ("Net Income", "net_income", "money"),
    ("Net Profit Margin", "net_profit_margin", "pct"),
    ("ROA", "roa", "pct"),
    ("ROE", "roe", "pct"),
]


def resolve_category(frame, company, year):
    """Selected company's category from its selected-year record."""
    if "category" not in frame.columns:
        return None
    try:
        rows = frame[
            (frame["company"].astype(str) == str(company))
            & (frame["year"].astype(int) == int(year))
        ]
        if rows.empty:
            return None
        value = str(rows.iloc[0]["category"]).strip()
        return value or None
    except (KeyError, ValueError, IndexError):
        return None


def _year_int(frame):
    return pd.to_numeric(frame["year"], errors="coerce")


def peer_group(frame, company, category, year, year_lte=False):
    """Same-category companies for the year (selected included).

    Duplicated company-year rows are collapsed (last record wins) so a
    company never occupies two ranks. Returns (group_df, peer_count)
    where peer_count excludes the selected company.
    """
    if category is None or "category" not in frame.columns:
        return pd.DataFrame(), 0

    work = frame.copy()
    try:
        if year_lte:
            mask = (work["category"].astype(str) == str(category)) & (
                _year_int(work) <= int(year))
        else:
            mask = (work["category"].astype(str) == str(category)) & (
                _year_int(work) == int(year))
        group = work.loc[mask].copy()
    except (KeyError, ValueError):
        return pd.DataFrame(), 0

    if group.empty:
        return group, 0

    group = group.sort_values("year").drop_duplicates(
        subset=["company", "year"], keep="last").reset_index(drop=True)
    peers = group[group["company"].astype(str) != str(company)]
    return group, int(peers["company"].nunique())


def prepare_peer_ranking(frame, company, category, year, metric_col):
    """Raw ranking (descending numeric, never reversed) with ranks."""
    group, peer_count = peer_group(frame, company, category, year)
    if group.empty or metric_col not in group.columns:
        return {"available": False, "reason": "metric unavailable",
                "ranking": pd.DataFrame(), "peer_count": peer_count,
                "selected_rank": None}

    values = pd.to_numeric(group[metric_col], errors="coerce")
    ranked = group.loc[values.notna()].copy()
    if ranked.empty:
        return {"available": False, "reason": "no values",
                "ranking": pd.DataFrame(), "peer_count": peer_count,
                "selected_rank": None}

    ranked["_value"] = pd.to_numeric(ranked[metric_col])
    ranked = ranked.sort_values("_value", ascending=False).reset_index(
        drop=True)
    ranked["rank"] = ranked.index + 1
    ranked["is_selected"] = (
        ranked["company"].astype(str) == str(company))

    selected = ranked[ranked["is_selected"]]
    selected_rank = int(selected.iloc[0]["rank"]) if not selected.empty \
        else None

    return {"available": True, "reason": "",
            "ranking": ranked[["rank", "company", "_value", "is_selected"]],
            "peer_count": peer_count, "selected_rank": selected_rank}


def build_peer_scatter_data(frame, company, category, year,
                            x_col, y_col):
    """Scatter points for one x/y metric pair (selected flagged)."""
    group, peer_count = peer_group(frame, company, category, year)
    if group.empty or x_col not in group.columns \
            or y_col not in group.columns:
        return {"available": False, "reason": "metric unavailable",
                "points": pd.DataFrame(), "peer_count": peer_count}

    points = group[["company", "year", "category", x_col, y_col]].copy()
    points["_x"] = pd.to_numeric(points[x_col], errors="coerce")
    points["_y"] = pd.to_numeric(points[y_col], errors="coerce")
    points = points.dropna(subset=["_x", "_y"]).reset_index(drop=True)

    if points.empty:
        return {"available": False, "reason": "insufficient data",
                "points": pd.DataFrame(), "peer_count": peer_count}

    points["is_selected"] = (
        points["company"].astype(str) == str(company))
    return {"available": True, "reason": "", "points": points,
            "peer_count": peer_count}


def build_peer_trend_data(frame, company, category, selected_year,
                          metric_col):
    """Per-year company value vs peer average (peer excludes selected)."""
    if category is None or metric_col not in frame.columns:
        return {"available": False, "records": pd.DataFrame()}

    try:
        years = sorted(
            int(y) for y in
            _year_int(frame).dropna().unique().tolist()
            if int(y) <= int(selected_year))
    except (ValueError, TypeError):
        return {"available": False, "records": pd.DataFrame()}

    records = []
    for year in years:
        group, _ = peer_group(frame, company, category, year)
        if group.empty:
            continue
        company_rows = group[
            group["company"].astype(str) == str(company)]
        peer_rows = group[
            group["company"].astype(str) != str(company)]
        company_vals = pd.to_numeric(company_rows[metric_col],
                                     errors="coerce").dropna()
        peer_vals = pd.to_numeric(peer_rows[metric_col],
                                  errors="coerce").dropna()
        if company_vals.empty or peer_vals.empty:
            continue
        records.append({
            "year": year,
            "company": float(company_vals.iloc[-1]),
            "peer_avg": float(peer_vals.mean()),
        })

    records = pd.DataFrame(records)
    if records.empty:
        return {"available": False, "records": records}
    records["difference"] = records["company"] - records["peer_avg"]
    return {"available": True, "records": records}


def build_benchmark_variance(benchmark_row):
    """Variance rows from existing *_vs_peer_pct engine values.

    Formula is the engine's own (company - peer_avg) / abs(peer_avg).
    Missing values stay missing (N/A, never zero).
    """
    specs = [
        ("Revenue", "revenue_company", "revenue_peer_avg",
         "revenue_vs_peer_pct", "money"),
        ("Net Income", "net_income_company", "net_income_peer_avg",
         "net_income_vs_peer_pct", "money"),
        ("Net Profit Margin", "net_profit_margin_company",
         "net_profit_margin_peer_avg", "net_profit_margin_vs_peer_pct",
         "pct"),
        ("Current Ratio", "current_ratio_company", "current_ratio_peer_avg",
         "current_ratio_vs_peer_pct", "x"),
        ("Debt / Equity", "debt_equity_ratio_company",
         "debt_equity_ratio_peer_avg", "debt_equity_ratio_vs_peer_pct",
         "x"),
        ("ROA", "roa_company", "roa_peer_avg", "roa_vs_peer_pct", "pct"),
        ("ROE", "roe_company", "roe_peer_avg", "roe_vs_peer_pct", "pct"),
        ("ROI", "roi_company", "roi_peer_avg", "roi_vs_peer_pct", "pct"),
    ]
    rows = []
    for label, company_key, peer_key, pct_key, kind in specs:
        try:
            company_v = benchmark_row.get(company_key)
            peer_v = benchmark_row.get(peer_key)
            vs_pct = benchmark_row.get(pct_key)
        except AttributeError:
            continue
        company_v = float(company_v) if pd.notna(company_v) else None
        peer_v = float(peer_v) if pd.notna(peer_v) else None
        vs_pct = float(vs_pct) if pd.notna(vs_pct) else None
        if company_v is None and peer_v is None and vs_pct is None:
            continue
        diff = (company_v - peer_v
                if company_v is not None and peer_v is not None else None)
        rows.append({"metric": label, "company": company_v,
                     "peer_avg": peer_v, "difference": diff,
                     "vs_peer_pct": vs_pct, "kind": kind})
    return rows


def peer_position(vs_peer_pct, metric_kind):
    """Above / Below / At peer average from numerical relationship only.

    No quality judgment: direction is reported, interpretation stays in
    the scorecard's analytical context.
    """
    if vs_peer_pct is None or (isinstance(vs_peer_pct, float)
                               and np.isnan(vs_peer_pct)):
        return "n/a"
    if abs(vs_peer_pct) < 0.005:
        return "At Peer Average"
    above = vs_peer_pct > 0
    return "Above Peer Average" if above else "Below Peer Average"


def scorecard_context(metric_label):
    """Neutral analytical context per metric (not investment advice)."""
    return {
        "Revenue": "Higher means larger scale.",
        "Net Income": "Higher means larger absolute earnings.",
        "Net Profit Margin": "Higher means higher profitability margin.",
        "Current Ratio": "Higher can indicate greater short-term "
                         "liquidity; extremely high is not automatically "
                         "better.",
        "Debt / Equity": "Lower generally means lower leverage.",
        "ROA": "Higher means higher return on assets.",
        "ROE": "Higher means higher return on equity.",
        "ROI": "Higher means higher return on investment.",
    }.get(metric_label, "")

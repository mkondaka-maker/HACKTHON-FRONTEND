
"""Deterministic data helpers for FinSight AI visualizations.

All helpers return values computed from the normalized dataset and the
existing financial engine. They never fabricate missing values: when a
required source column is absent, the helper reports what is available
and the UI shows an informational message instead.

Currency note: source data is USD millions. Helpers return values in
source units; the UI converts with format_money()/convert_chart_values().
"""

import pandas as pd
import numpy as np


def _company_history(df, company, selected_year):
    """Company rows with year <= selected_year, sorted chronologically."""
    hist = df[
        (df["company"].astype(str) == str(company))
        & (pd.to_numeric(df["year"], errors="coerce") <= int(selected_year))
    ].copy()

    if hist.empty:
        return hist

    hist["year"] = pd.to_numeric(hist["year"], errors="coerce")
    return hist.sort_values("year").reset_index(drop=True)


def _clean_series(frame, column):
    """Numeric series with inf coerced to NA (never crashes Altair)."""
    series = pd.to_numeric(frame[column], errors="coerce")
    return series.replace([np.inf, -np.inf], np.nan)


def get_balance_sheet_history(df, company, selected_year):
    """Balance-sheet structure over time (Overview page).

    Returns dict with:
        available: bool (all three of assets/liabilities/equity present)
        missing: list of unavailable source columns
        history: long-format DataFrame (year, component, value) or empty
        latest: dict of latest Assets/Liabilities/Equity + ratios or {}
    """
    required = {
        "total_assets": "Total Assets",
        "total_liabilities": "Total Liabilities",
        "share_holder_equity": "Shareholder Equity",
    }
    missing = [col for col in required if col not in df.columns]

    if missing:
        return {
            "available": False,
            "missing": missing,
            "history": pd.DataFrame(),
            "latest": {},
        }

    hist = _company_history(df, company, selected_year)

    if hist.empty:
        return {
            "available": False,
            "missing": [],
            "history": pd.DataFrame(),
            "latest": {},
        }

    records = []

    for column, label in required.items():
        values = _clean_series(hist, column)

        for year, value in zip(hist["year"], values):
            if pd.notna(value):
                records.append(
                    {"year": int(year), "component": label, "value": float(value)}
                )

    history = pd.DataFrame(records, columns=["year", "component", "value"])

    latest = {}
    last_rows = hist.tail(1)

    if not last_rows.empty:
        last = last_rows.iloc[0]
        assets = _clean_series(last_rows, "total_assets").iloc[0]
        liabilities = _clean_series(last_rows, "total_liabilities").iloc[0]
        equity = _clean_series(last_rows, "share_holder_equity").iloc[0]

        if pd.notna(assets):
            latest["assets"] = float(assets)
        if pd.notna(liabilities):
            latest["liabilities"] = float(liabilities)
        if pd.notna(equity):
            latest["equity"] = float(equity)
        if pd.notna(assets) and assets != 0:
            if pd.notna(liabilities):
                latest["liabilities_to_assets"] = float(liabilities / assets * 100)
            if pd.notna(equity):
                latest["equity_to_assets"] = float(equity / assets * 100)
        _ = last  # readability: values taken from the latest row above

    available = not history.empty and set(
        history["component"].unique()
    ) == set(required.values())

    return {
        "available": available,
        "missing": [],
        "history": history,
        "latest": latest,
    }


def get_growth_history(metrics_df, company, selected_year):
    """Revenue YoY vs Net Income YoY over time (Report page).

    `metrics_df` must be financial-engine output (calculate_financial_metrics),
    which already provides revenue_yoy / net_income_yoy. Falls back to the
    *_growth_rate aliases only when the YoY columns are absent.

    Returns dict with: available, history (year, metric, growth), insight.
    """
    hist = _company_history(metrics_df, company, selected_year)

    revenue_col = (
        "revenue_yoy" if "revenue_yoy" in hist.columns
        else "revenue_growth_rate" if "revenue_growth_rate" in hist.columns
        else None
    )
    income_col = (
        "net_income_yoy" if "net_income_yoy" in hist.columns
        else "profit_growth_rate" if "profit_growth_rate" in hist.columns
        else None
    )

    if hist.empty or revenue_col is None or income_col is None:
        return {"available": False, "history": pd.DataFrame(), "insight": ""}

    records = []

    for _, row in hist.iterrows():
        year = int(row["year"])
        revenue_growth = _clean_series(pd.DataFrame([row]), revenue_col).iloc[0]
        income_growth = _clean_series(pd.DataFrame([row]), income_col).iloc[0]

        if pd.notna(revenue_growth):
            records.append(
                {
                    "year": year,
                    "metric": "Revenue Growth",
                    "growth": float(revenue_growth),
                }
            )
        if pd.notna(income_growth):
            records.append(
                {
                    "year": year,
                    "metric": "Net Income Growth",
                    "growth": float(income_growth),
                }
            )

    history = pd.DataFrame(records, columns=["year", "metric", "growth"])

    if history.empty:
        return {"available": False, "history": history, "insight": ""}

    # Deterministic insight from the latest period with both values.
    insight = ""
    pivot = history.pivot_table(
        index="year", columns="metric", values="growth", aggfunc="first"
    ).reset_index()

    both = pivot.dropna(subset=["Revenue Growth", "Net Income Growth"])

    if not both.empty:
        last = both.iloc[-1]
        revenue_last = float(last["Revenue Growth"])
        income_last = float(last["Net Income Growth"])

        if revenue_last > income_last:
            insight = (
                f"In {int(last['year'])}, revenue growth "
                f"({revenue_last:+.2f}%) exceeded net income growth "
                f"({income_last:+.2f}%), indicating a potential "
                f"profitability lag."
            )
        elif income_last > revenue_last:
            insight = (
                f"In {int(last['year'])}, net income growth "
                f"({income_last:+.2f}%) outpaced revenue growth "
                f"({revenue_last:+.2f}%), indicating improving "
                f"profitability leverage."
            )
        else:
            insight = (
                f"In {int(last['year'])}, revenue and net income grew "
                f"in line ({revenue_last:+.2f}%)."
            )

    return {"available": True, "history": history, "insight": insight}


def get_cash_flow_history(metrics_df, company, selected_year):
    """Operating/Investing/Financing/Net cash flow over time (Report page).

    Uses financial-engine net_cash_flow. Free cash flow is included only
    when the normalized dataset already provides it — never derived from
    an assumed CAPEX field.

    Returns dict with: available, missing, history (year, flow, value),
    latest (operating/net/margin), margin_available.
    """
    required = {
        "cash_flow_from_operating": "Operating Cash Flow",
        "cash_flow_from_investing": "Investing Cash Flow",
        "cash_flow_from_financial_activities": "Financing Cash Flow",
    }
    missing = [col for col in required if col not in metrics_df.columns]

    if missing:
        return {
            "available": False,
            "missing": missing,
            "history": pd.DataFrame(),
            "latest": {},
            "margin_available": False,
            "free_cash_flow_available": False,
        }

    hist = _company_history(metrics_df, company, selected_year)

    if hist.empty:
        return {
            "available": False,
            "missing": [],
            "history": pd.DataFrame(),
            "latest": {},
            "margin_available": False,
            "free_cash_flow_available": False,
        }

    series_map = dict(required)

    if "net_cash_flow" in hist.columns:
        series_map["net_cash_flow"] = "Net Cash Flow"

    free_cash_flow_available = (
        "free_cash_flow" in hist.columns
        and _clean_series(hist, "free_cash_flow").notna().any()
    )

    records = []

    for column, label in series_map.items():
        values = _clean_series(hist, column)

        for year, value in zip(hist["year"], values):
            if pd.notna(value):
                records.append(
                    {"year": int(year), "flow": label, "value": float(value)}
                )

    history = pd.DataFrame(records, columns=["year", "flow", "value"])

    if history.empty:
        return {
            "available": False,
            "missing": [],
            "history": history,
            "latest": {},
            "margin_available": False,
            "free_cash_flow_available": bool(free_cash_flow_available),
        }

    latest = {}
    last = hist.tail(1)
    operating = _clean_series(last, "cash_flow_from_operating").iloc[0]

    if pd.notna(operating):
        latest["operating"] = float(operating)

    if "net_cash_flow" in hist.columns:
        net = _clean_series(last, "net_cash_flow").iloc[0]
        if pd.notna(net):
            latest["net"] = float(net)

    margin_available = False
    if "revenue" in hist.columns:
        revenue = _clean_series(last, "revenue").iloc[0]
        if pd.notna(operating) and pd.notna(revenue) and revenue != 0:
            latest["operating_margin"] = float(operating / revenue * 100)
            margin_available = True

    if free_cash_flow_available:
        free = _clean_series(last, "free_cash_flow").iloc[0]
        if pd.notna(free):
            latest["free_cash_flow"] = float(free)

    return {
        "available": True,
        "missing": [],
        "history": history,
        "latest": latest,
        "margin_available": margin_available,
        "free_cash_flow_available": bool(free_cash_flow_available),
    }


import pandas as pd


COMPARISON_METRICS = {
    "Revenue": "revenue",
    "Gross Profit": "gross_profit",
    "Net Income": "net_income",
    "EBITDA": "ebitda",
    "Net Profit Margin": "net_profit_margin",
    "Current Ratio": "current_ratio",
    "Debt / Equity": "debt_equity_ratio",
    "ROE": "roe",
    "ROA": "roa",
    "ROI": "roi",
    "Operating Cash Flow": "cash_flow_from_operating",
    "Investing Cash Flow": "cash_flow_from_investing",
    "Financing Cash Flow": "cash_flow_from_financial_activities",
}


def _safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _get_metric_value(row, metric):
    if metric in row.index:
        return _safe_float(row[metric])

    if metric == "net_profit_margin":
        revenue = _safe_float(row["revenue"]) if "revenue" in row.index else None
        net_income = _safe_float(row["net_income"]) if "net_income" in row.index else None

        if revenue is not None and revenue != 0 and net_income is not None:
            return (net_income / revenue) * 100

    return None


def compare_two_years(financial_df, company, year_a, year_b):
    """
    Compare one company's two selected financial years.

    Values are taken directly from the supplied dataset or are
    mathematically derived when the required source fields exist.
    Missing metrics remain unavailable.
    """

    df = financial_df.copy()

    company_df = df[
        df["company"].astype(str) == str(company)
    ].copy()

    if company_df.empty:
        raise ValueError(f"No financial data found for {company}.")

    row_a = company_df[
        company_df["year"].astype(int) == int(year_a)
    ]

    row_b = company_df[
        company_df["year"].astype(int) == int(year_b)
    ]

    if row_a.empty:
        raise ValueError(f"No financial data found for {company} in {year_a}.")

    if row_b.empty:
        raise ValueError(f"No financial data found for {company} in {year_b}.")

    row_a = row_a.iloc[0]
    row_b = row_b.iloc[0]

    records = []

    for label, metric in COMPARISON_METRICS.items():

        value_a = _get_metric_value(row_a, metric)
        value_b = _get_metric_value(row_b, metric)

        absolute_change = None
        percentage_change = None

        if value_a is not None and value_b is not None:
            absolute_change = value_b - value_a

            if value_a != 0:
                percentage_change = (
                    (value_b - value_a) / abs(value_a)
                ) * 100

        records.append({
            "Metric": label,
            str(year_a): value_a,
            str(year_b): value_b,
            "Absolute Change": absolute_change,
            "Percentage Change": percentage_change,
            "Availability": (
                "Available"
                if value_a is not None and value_b is not None
                else "Not provided"
            ),
        })

    result = pd.DataFrame(records)

    return result


def build_comparison_summary(comparison_df, year_a, year_b):
    """
    Generate deterministic observations from a two-year comparison.
    """

    observations = []

    for _, row in comparison_df.iterrows():

        metric = row["Metric"]
        value_a = row[str(year_a)]
        value_b = row[str(year_b)]

        if pd.isna(value_a) or pd.isna(value_b):
            continue

        try:
            change = float(row["Percentage Change"])
        except Exception:
            continue

        if change > 0:
            direction = "increased"
        elif change < 0:
            direction = "decreased"
        else:
            direction = "remained stable"

        observations.append({
            "metric": metric,
            "direction": direction,
            "percentage_change": change,
            "year_a": int(year_a),
            "year_b": int(year_b),
        })

    return observations

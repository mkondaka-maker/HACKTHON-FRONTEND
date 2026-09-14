
import pandas as pd


def get_available_years(financial_df, company):
    df = financial_df[
        financial_df["company"].astype(str) == str(company)
    ].copy()

    if df.empty:
        return []

    return sorted(
        df["year"].astype(int).unique()
    )


def get_last_n_years(financial_df, company, n=5):
    years = get_available_years(financial_df, company)

    return years[-n:]


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
        revenue = (
            _safe_float(row["revenue"])
            if "revenue" in row.index
            else None
        )

        net_income = (
            _safe_float(row["net_income"])
            if "net_income" in row.index
            else None
        )

        if (
            revenue is not None
            and revenue != 0
            and net_income is not None
        ):
            return (net_income / revenue) * 100

    return None


def historical_metric_analysis(
    financial_df,
    metric,
    company,
    n=5
):
    """
    Return the latest n available years for one metric.
    Only supplied or mathematically derivable values are used.
    """

    metric_aliases = {
        "sales": "revenue",
        "turnover": "revenue",
        "revenue": "revenue",
        "net income": "net_income",
        "net profit": "net_income",
        "profit": "net_income",
        "gross profit": "gross_profit",
        "gross margin": "gross_margin",
        "net margin": "net_profit_margin",
        "profit margin": "net_profit_margin",
        "net profit margin": "net_profit_margin",
        "current ratio": "current_ratio",
        "debt equity": "debt_equity_ratio",
        "debt/equity": "debt_equity_ratio",
        "debt to equity": "debt_equity_ratio"
    }

    metric = metric_aliases.get(
        str(metric).strip().lower(),
        str(metric).strip()
    )

    company_df = financial_df[
        financial_df["company"].astype(str) == str(company)
    ].copy()

    if company_df.empty:
        raise ValueError(
            f"No financial data found for {company}."
        )

    company_df["year"] = company_df["year"].astype(int)
    company_df = company_df.sort_values("year")

    years = company_df["year"].unique()[-n:]

    records = []
    previous_value = None

    for year in years:

        row = company_df[
            company_df["year"] == int(year)
        ].iloc[0]

        value = _get_metric_value(
            row,
            metric
        )

        yoy_change = None

        if (
            value is not None
            and previous_value is not None
            and previous_value != 0
        ):
            yoy_change = (
                (value - previous_value)
                / abs(previous_value)
            ) * 100

        records.append({
            "Year": int(year),
            "Value": value,
            "YoY Change (%)": yoy_change,
            "Availability": (
                "Available"
                if value is not None
                else "Not provided"
            )
        })

        if value is not None:
            previous_value = value

    return pd.DataFrame(records)

def historical_summary(
    historical_df,
    metric,
    company
):
    """
    Create deterministic observations from a historical
    metric analysis.

    Accepts either the output of historical_metric_analysis()
    or a raw financial dataframe.
    """

    if "Value" not in historical_df.columns:

        historical_df = historical_metric_analysis(
            historical_df,
            metric,
            company,
            n=len(
                historical_df[
                    historical_df["company"].astype(str)
                    == str(company)
                ]
            )
        )

    available = historical_df[
        historical_df["Value"].notna()
    ].copy()

    if available.empty:
        return {
            "company": company,
            "metric": metric,
            "years_available": 0,
            "first_year": None,
            "last_year": None,
            "overall_change": None,
            "overall_change_percent": None,
            "direction": "Unavailable"
        }

    first = float(
        available.iloc[0]["Value"]
    )

    last = float(
        available.iloc[-1]["Value"]
    )

    absolute_change = last - first

    percentage_change = None

    if first != 0:
        percentage_change = (
            absolute_change / abs(first)
        ) * 100

    if absolute_change > 0:
        direction = "Increased"
    elif absolute_change < 0:
        direction = "Decreased"
    else:
        direction = "Stable"

    return {
        "company": company,
        "metric": metric,
        "years_available": len(available),
        "first_year": int(
            available.iloc[0]["Year"]
        ),
        "last_year": int(
            available.iloc[-1]["Year"]
        ),
        "overall_change": absolute_change,
        "overall_change_percent": percentage_change,
        "direction": direction
    }


import pandas as pd
import numpy as np


def load_financial_data(path="data/clean_data.csv"):
    """Load and prepare the financial dataset."""
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("/", "_")
        .str.replace("(", "")
        .str.replace(")", "")
    )

    return df


def calculate_financial_metrics(df):
    """
    Calculate derived financial metrics required by FinSight.

    The uploaded dataset only needs the core financial fields.
    Optional ratios/derived fields are calculated when they are
    not already present.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Basic numeric normalization
    # --------------------------------------------------------

    numeric_columns = [
        "revenue",
        "gross_profit",
        "net_income",
        "ebitda",
        "share_holder_equity",
        "cash_flow_from_operating",
        "cash_flow_from_investing",
        "cash_flow_from_financial_activities",
        "current_ratio",
        "debt_equity_ratio",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --------------------------------------------------------
    # Sort so previous-year calculations are correct
    # --------------------------------------------------------

    if "company" in df.columns and "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.sort_values(["company", "year"]).reset_index(drop=True)

    # --------------------------------------------------------
    # Derived profitability metrics
    # --------------------------------------------------------

    if "revenue" in df.columns and "gross_profit" in df.columns:
        calculated_gross_margin = (
            df["gross_profit"] / df["revenue"].replace(0, pd.NA)
        ) * 100

        df["gross_margin_calculated"] = calculated_gross_margin

        # Preserve an existing supplied margin if available.
        # Otherwise use the calculated value.
        if "gross_margin" not in df.columns:
            df["gross_margin"] = calculated_gross_margin

    if "revenue" in df.columns and "ebitda" in df.columns:
        calculated_ebitda_margin = (
            df["ebitda"] / df["revenue"].replace(0, pd.NA)
        ) * 100

        df["ebitda_margin_calculated"] = calculated_ebitda_margin

        if "ebitda_margin" not in df.columns:
            df["ebitda_margin"] = calculated_ebitda_margin

    if "revenue" in df.columns and "net_income" in df.columns:
        calculated_net_margin = (
            df["net_income"] / df["revenue"].replace(0, pd.NA)
        ) * 100

        df["net_margin_calculated"] = calculated_net_margin

        # This is the field used by the existing validation,
        # benchmark and dashboard code.
        if "net_profit_margin" not in df.columns:
            df["net_profit_margin"] = calculated_net_margin

    # --------------------------------------------------------
    # Net cash flow
    # --------------------------------------------------------

    cash_flow_columns = [
        "cash_flow_from_operating",
        "cash_flow_from_investing",
        "cash_flow_from_financial_activities",
    ]

    if all(col in df.columns for col in cash_flow_columns):
        df["net_cash_flow"] = (
            df["cash_flow_from_operating"]
            + df["cash_flow_from_investing"]
            + df["cash_flow_from_financial_activities"]
        )

    # --------------------------------------------------------
    # Year-over-year calculations
    # --------------------------------------------------------

    if "company" in df.columns:

        yoy_columns = [
            "revenue",
            "gross_profit",
            "net_income",
            "ebitda",
            "net_profit_margin",
            "gross_margin",
            "ebitda_margin",
            "current_ratio",
            "debt_equity_ratio",
        ]

        for col in yoy_columns:

            if col not in df.columns:
                continue

            previous = df.groupby("company")[col].shift(1)

            df[f"{col}_previous"] = previous

            df[f"{col}_yoy"] = (
                (df[col] - previous)
                / previous.replace(0, pd.NA)
            ) * 100

        # Explicit aliases expected by the intelligence modules.
        df["revenue_yoy"] = df.get(
            "revenue_yoy",
            df["revenue_yoy"] if "revenue_yoy" in df.columns
            else pd.Series(index=df.index, dtype=float)
        )

        df["net_income_yoy"] = df.get(
            "net_income_yoy",
            df["net_income_yoy"] if "net_income_yoy" in df.columns
            else pd.Series(index=df.index, dtype=float)
        )

    # --------------------------------------------------------
    # Clean infinite values
    # --------------------------------------------------------

    df = df.replace([float("inf"), float("-inf")], pd.NA)

    return df
def build_financial_dataset(path="data/clean_data.csv"):
    """Load the dataset and calculate all financial metrics."""

    df = load_financial_data(path)

    df = calculate_financial_metrics(df)

    return df

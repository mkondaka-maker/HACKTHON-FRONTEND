
import re
import pandas as pd

from engine.historical import (
    historical_metric_analysis,
    historical_summary
)

from engine.comparison import (
    compare_two_years,
    build_comparison_summary
)


METRIC_ALIASES = {
    "revenue": [
        "revenue",
        "sales",
        "turnover"
    ],
    "gross_profit": [
        "gross profit",
        "gross profit"
    ],
    "net_income": [
        "net income",
        "net profit",
        "profit"
    ],
    "ebitda": [
        "ebitda"
    ],
    "net_profit_margin": [
        "net profit margin",
        "net margin",
        "npm",
        "profit margin"
    ],
    "current_ratio": [
        "current ratio",
        "liquidity"
    ],
    "debt_equity_ratio": [
        "debt equity",
        "debt/equity",
        "debt to equity",
        "d/e",
        "leverage"
    ],
    "roe": [
        "roe",
        "return on equity"
    ],
    "roa": [
        "roa",
        "return on assets"
    ],
    "roi": [
        "roi",
        "return on investment"
    ],
    "cash_flow_from_operating": [
        "operating cash flow",
        "cash flow from operating"
    ],
    "cash_flow_from_investing": [
        "investing cash flow",
        "cash flow from investing"
    ],
    "cash_flow_from_financial_activities": [
        "financing cash flow",
        "cash flow from financing"
    ]
}


METRIC_LABELS = {
    "revenue": "Revenue",
    "gross_profit": "Gross Profit",
    "net_income": "Net Income",
    "ebitda": "EBITDA",
    "net_profit_margin": "Net Profit Margin",
    "current_ratio": "Current Ratio",
    "debt_equity_ratio": "Debt / Equity",
    "roe": "ROE",
    "roa": "ROA",
    "roi": "ROI",
    "cash_flow_from_operating": "Operating Cash Flow",
    "cash_flow_from_investing": "Investing Cash Flow",
    "cash_flow_from_financial_activities": "Financing Cash Flow"
}


def detect_metric(question):

    text = question.lower()

    matches = []

    for metric, aliases in METRIC_ALIASES.items():

        for alias in aliases:

            if alias in text:
                matches.append(
                    (len(alias), metric)
                )

    if not matches:
        return None

    matches.sort(
        reverse=True
    )

    return matches[0][1]


def detect_years(question):

    years = re.findall(
        r"\b(20\d{2})\b",
        question
    )

    return sorted(
        list(set(int(year) for year in years))
    )


def detect_historical_request(question):

    text = question.lower()

    patterns = [
        "last",
        "past",
        "historical",
        "history",
        "trend",
        "over the years",
        "years",
        "all years",
        "every year",
        "available years"
    ]

    return any(
        pattern in text
        for pattern in patterns
    )


def detect_year_count(question):

    text = question.lower()

    patterns = [
        r"last\s+(\d+)\s+years?",
        r"past\s+(\d+)\s+years?",
        r"previous\s+(\d+)\s+years?",
        r"over\s+the\s+last\s+(\d+)\s+years?",
        r"over\s+the\s+past\s+(\d+)\s+years?"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text
        )

        if match:
            return int(
                match.group(1)
            )

    return None


def detect_comparison_request(question):

    text = question.lower()

    patterns = [
        "compare",
        "comparison",
        "versus",
        "vs",
        "against"
    ]

    return any(
        pattern in text
        for pattern in patterns
    )


def detect_intent(question):

    metric = detect_metric(question)
    years = detect_years(question)

    if detect_comparison_request(question) and len(years) >= 2:
        return "year_comparison"

    if metric and detect_historical_request(question):
        return "historical_metric"

    if metric:
        return "metric_lookup"

    if any(
        word in question.lower()
        for word in [
            "health",
            "financial health"
        ]
    ):
        return "financial_health"

    if any(
        word in question.lower()
        for word in [
            "risk",
            "risk score"
        ]
    ):
        return "risk"

    if any(
        word in question.lower()
        for word in [
            "anomaly",
            "anomalies",
            "abnormal",
            "unusual"
        ]
    ):
        return "anomalies"

    if any(
        word in question.lower()
        for word in [
            "overview",
            "summary",
            "performance"
        ]
    ):
        return "overview"

    return "unsupported"


def _latest_metric_value(financial_df, company, metric):

    company_df = financial_df[
        financial_df["company"].astype(str) == str(company)
    ].copy()

    if company_df.empty:
        return None

    company_df = company_df.sort_values(
        "year"
    )

    latest = company_df.iloc[-1]

    if metric == "net_profit_margin":

        if "net_profit_margin" in latest.index:
            value = latest["net_profit_margin"]

        elif (
            "net_income" in latest.index
            and "revenue" in latest.index
        ):
            revenue = float(latest["revenue"])
            net_income = float(latest["net_income"])

            if revenue != 0:
                value = (
                    net_income / revenue
                ) * 100
            else:
                value = None

        else:
            value = None

    elif metric in latest.index:

        value = latest[metric]

    else:

        value = None

    return {
        "year": int(latest["year"]),
        "value": value
    }


def execute_financial_query(
    financial_df,
    company,
    question
):

    metric = detect_metric(question)
    years = detect_years(question)
    intent = detect_intent(question)

    base_result = {
        "status": "ERROR",
        "company": company,
        "question": question,
        "intent": intent,
        "metric": metric,
        "years": years,
        "data": pd.DataFrame(),
        "summary": None,
        "answer": "",
        "report_type": None
    }

    if intent == "historical_metric":

        if metric is None:
            base_result["status"] = "UNSUPPORTED"
            base_result["answer"] = (
                "I could not identify the financial metric "
                "for the historical analysis."
            )
            return base_result

        year_count = detect_year_count(
            question
        )

        if year_count is None:

            company_df = financial_df[
                financial_df["company"].astype(str)
                == str(company)
            ].copy()

            available_years = (
                company_df["year"]
                .dropna()
                .astype(int)
                .nunique()
            )

            year_count = available_years

        # NOTE: historical_metric_analysis signature is
        # (financial_df, metric, company, n) — argument order matters.
        result = historical_metric_analysis(
            financial_df,
            metric,
            company,
            n=year_count
        )

        summary = historical_summary(
            result,
            metric,
            company
        )

        base_result["status"] = "SUCCESS"
        base_result["data"] = result
        base_result["summary"] = summary
        base_result["report_type"] = "historical"

        label = METRIC_LABELS.get(
            metric,
            metric
        )

        if result.empty:

            base_result["answer"] = (
                f"No historical {label} data is available "
                f"for {company}."
            )

        else:

            first_year = int(
                result["Year"].min()
            )

            last_year = int(
                result["Year"].max()
            )

            overall_change = summary.get(
                "overall_change_percent"
            )

            if overall_change is not None:

                direction = (
                    "increased"
                    if overall_change > 0
                    else "decreased"
                    if overall_change < 0
                    else "remained stable"
                )

                base_result["answer"] = (
                    f"{company}'s {label} from "
                    f"{first_year} to {last_year} "
                    f"{direction} by "
                    f"{abs(float(overall_change)):.2f}%."
                )

            else:

                base_result["answer"] = (
                    f"Historical {label} data is available "
                    f"from {first_year} to {last_year}."
                )

        return base_result

    if intent == "year_comparison":

        if metric is None:

            base_result["status"] = "UNSUPPORTED"
            base_result["answer"] = (
                "Please specify a financial metric to compare."
            )

            return base_result

        if len(years) < 2:

            base_result["status"] = "UNSUPPORTED"
            base_result["answer"] = (
                "Please provide two years to compare."
            )

            return base_result

        year_a = years[0]
        year_b = years[1]

        result = compare_two_years(
            financial_df,
            company,
            year_a,
            year_b
        )

        summary = build_comparison_summary(
            result,
            year_a,
            year_b
        )

        base_result["status"] = "SUCCESS"
        base_result["data"] = result
        base_result["summary"] = summary
        base_result["report_type"] = "comparison"

        metric_rows = result[
            result["Metric"].str.lower()
            == METRIC_LABELS.get(
                metric,
                metric
            ).lower()
        ]

        if metric_rows.empty:

            base_result["answer"] = (
                f"{METRIC_LABELS.get(metric, metric)} "
                f"comparison is unavailable for "
                f"{year_a} and {year_b}."
            )

        else:

            row = metric_rows.iloc[0]

            change = row["Percentage Change"]

            if pd.isna(change):

                base_result["answer"] = (
                    f"{METRIC_LABELS.get(metric, metric)} "
                    f"comparison is available for "
                    f"{year_a} and {year_b}."
                )

            else:

                direction = (
                    "increased"
                    if change > 0
                    else "decreased"
                    if change < 0
                    else "remained stable"
                )

                base_result["answer"] = (
                    f"{company}'s "
                    f"{METRIC_LABELS.get(metric, metric)} "
                    f"{direction} by "
                    f"{abs(float(change)):.2f}% "
                    f"from {year_a} to {year_b}."
                )

        return base_result

    if intent == "metric_lookup":

        if metric is None:

            base_result["status"] = "UNSUPPORTED"
            base_result["answer"] = (
                "I could not identify the financial metric."
            )

            return base_result

        latest = _latest_metric_value(
            financial_df,
            company,
            metric
        )

        if latest is None:

            base_result["status"] = "UNAVAILABLE"
            base_result["answer"] = (
                f"{METRIC_LABELS.get(metric, metric)} "
                f"is not available for {company}."
            )

            return base_result

        value = latest["value"]

        if value is None or pd.isna(value):

            base_result["status"] = "UNAVAILABLE"
            base_result["answer"] = (
                f"{METRIC_LABELS.get(metric, metric)} "
                f"is not provided for {company}."
            )

            return base_result

        base_result["status"] = "SUCCESS"
        base_result["data"] = pd.DataFrame([
            {
                "Year": latest["year"],
                "Metric": METRIC_LABELS.get(
                    metric,
                    metric
                ),
                "Value": value,
                "Availability": "Available"
            }
        ])

        base_result["summary"] = {
            "latest_year": latest["year"],
            "value": value
        }
        base_result["report_type"] = "historical"

        base_result["answer"] = (
            f"The latest available "
            f"{METRIC_LABELS.get(metric, metric)} "
            f"for {company} is "
            f"{value:.2f} in {latest['year']}."
        )

        base_result["report_type"] = "historical"

        return base_result

    base_result["status"] = "UNSUPPORTED"

    base_result["answer"] = (
        "I can currently answer financial metric questions, "
        "historical trends, and two-year comparisons. "
        "Try questions such as: "
        "'Give me the last 5 years of revenue' or "
        "'Compare 2021 and 2022 revenue'."
    )

    return base_result

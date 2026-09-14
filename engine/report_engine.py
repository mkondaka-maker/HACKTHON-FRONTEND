
from datetime import datetime
from io import BytesIO

import pandas as pd


def _format_number(value):
    if value is None:
        return "Not provided"

    try:
        if pd.isna(value):
            return "Not provided"
    except Exception:
        pass

    try:
        return f"{float(value):,.2f}"
    except Exception:
        return str(value)


def build_historical_report(
    query_result
):
    company = query_result.get("company", "Unknown")
    question = query_result.get("question", "")
    metric = query_result.get("metric", "Unknown")

    data = query_result.get("data")

    summary = query_result.get("summary") or {}

    lines = []

    lines.append("FINSIGHT AI")
    lines.append("Financial Analysis Report")
    lines.append("=" * 80)
    lines.append("")

    lines.append(f"Company: {company}")
    lines.append(f"Analysis Type: Historical Analysis")
    lines.append(f"Metric: {metric}")
    lines.append(f"Question: {question}")
    lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append("")

    lines.append("Historical Data")
    lines.append("-" * 80)

    if isinstance(data, pd.DataFrame) and not data.empty:

        lines.append(
            data.to_string(index=False)
        )

    else:

        lines.append("No historical data available.")

    lines.append("")
    lines.append("Summary")
    lines.append("-" * 80)

    lines.append(
        f"Years available: "
        f"{summary.get('years_available', 0)}"
    )

    if summary.get("first_year") is not None:
        lines.append(
            f"Period: "
            f"{summary.get('first_year')} - "
            f"{summary.get('last_year')}"
        )

    if summary.get("overall_change") is not None:
        lines.append(
            f"Overall change: "
            f"{_format_number(summary.get('overall_change'))}"
        )

    if summary.get("overall_change_percent") is not None:
        lines.append(
            f"Overall percentage change: "
            f"{_format_number(summary.get('overall_change_percent'))}%"
        )

    lines.append(
        f"Direction: "
        f"{summary.get('direction', 'Unavailable')}"
    )

    lines.append("")
    lines.append("Data Integrity")
    lines.append("-" * 80)
    lines.append(
        "Values are based on the uploaded financial dataset. "
        "No unavailable values were fabricated."
    )

    return "\n".join(lines)


def build_comparison_report(
    query_result
):
    company = query_result.get("company", "Unknown")
    question = query_result.get("question", "")
    years = query_result.get("years", [])

    data = query_result.get("data")
    summary = query_result.get("summary") or []

    lines = []

    lines.append("FINSIGHT AI")
    lines.append("Financial Comparison Report")
    lines.append("=" * 80)
    lines.append("")

    lines.append(f"Company: {company}")

    if len(years) >= 2:
        lines.append(
            f"Comparison: {years[0]} vs {years[1]}"
        )

    lines.append(f"Question: {question}")
    lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append("")

    lines.append("Financial Comparison")
    lines.append("-" * 80)

    if isinstance(data, pd.DataFrame) and not data.empty:

        lines.append(
            data.to_string(index=False)
        )

    else:

        lines.append("No comparison data available.")

    lines.append("")
    lines.append("Key Changes")
    lines.append("-" * 80)

    if summary:

        for item in summary:

            lines.append(
                f"{item.get('metric')}: "
                f"{item.get('direction')} "
                f"({item.get('percentage_change', 0):.2f}%)"
            )

    else:

        lines.append(
            "No comparable changes were identified."
        )

    lines.append("")
    lines.append("Data Integrity")
    lines.append("-" * 80)
    lines.append(
        "Values are based on the uploaded financial dataset. "
        "Derived values are calculated only when the required "
        "source fields are available."
    )

    return "\n".join(lines)


def report_to_txt_bytes(report_text):
    return report_text.encode("utf-8")


def report_to_csv_bytes(dataframe):
    buffer = BytesIO()

    dataframe.to_csv(
        buffer,
        index=False
    )

    return buffer.getvalue()

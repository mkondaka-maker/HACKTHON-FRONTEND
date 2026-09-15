
import pandas as pd
import numpy as np


def build_financial_map(df, company=None, year=None):
    """Build financial intelligence map data for a company/year."""

    if company is None:
        company = df["company"].iloc[0]

    company_data = df[
        df["company"] == company
    ].copy()

    if company_data.empty:
        raise ValueError(f"Company {company} not found.")

    # Use latest year if year is not provided
    if year is None:
        year = int(company_data["year"].max())

    row_data = company_data[
        company_data["year"] == year
    ]

    if row_data.empty:
        raise ValueError(
            f"No financial data available for {company} in {year}."
        )

    row = row_data.iloc[0]

    # -----------------------------------------
    # Financial metric nodes
    # -----------------------------------------
    # Harden against partial/custom uploads: a core column may be
    # absent from the frame. Derive net margin when possible
    # (net_income / revenue * 100, same as dashboard.py); otherwise
    # skip the node so the map degrades gracefully instead of
    # raising KeyError.

    def _safe_value(frame_row, field):
        if field not in frame_row.index:
            return None
        try:
            value = float(frame_row[field])
        except (TypeError, ValueError):
            return None
        if pd.isna(value):
            return None
        return value

    def _derived_net_margin(frame_row):
        margin = _safe_value(frame_row, "net_profit_margin")
        if margin is not None:
            return margin
        net_income = _safe_value(frame_row, "net_income")
        revenue = _safe_value(frame_row, "revenue")
        if net_income is None or revenue is None or revenue == 0:
            return None
        return float(net_income) / float(revenue) * 100

    _core_specs = [
        ("revenue", "Revenue", "revenue", "USD M", None),
        ("gross_profit", "Gross Profit", "gross_profit", "USD M", None),
        ("ebitda", "EBITDA", "ebitda", "USD M", None),
        ("net_income", "Net Income", "net_income", "USD M", None),
        (
            "operating_cash",
            "Operating Cash Flow",
            "cash_flow_from_operating",
            "USD M",
            None,
        ),
        ("current_ratio", "Current Ratio", "current_ratio", "x", None),
        (
            "debt_equity",
            "Debt / Equity",
            "debt_equity_ratio",
            "x",
            None,
        ),
        ("net_margin", "Net Profit Margin", "net_profit_margin", "%", True),
    ]

    nodes = []
    for _node_id, _label, _field, _unit, _derive in _core_specs:
        if _derive:
            _value = _derived_net_margin(row)
        else:
            _value = _safe_value(row, _field)
        if _value is None:
            continue
        nodes.append(
            {
                "id": _node_id,
                "label": _label,
                "value": _value,
                "unit": _unit,
            }
        )

    # -----------------------------------------
    # Optional return metrics
    # -----------------------------------------

    optional_return_metrics = [
        ("roe", "ROE"),
        ("roa", "ROA"),
        ("roi", "ROI")
    ]

    for metric_id, metric_label in optional_return_metrics:

        if metric_id in row.index:
            _opt_value = _safe_value(row, metric_id)
            if _opt_value is None:
                continue
            nodes.append({
                "id": metric_id,
                "label": metric_label,
                "value": _opt_value,
                "unit": "%"
            })

    # -----------------------------------------
    # Analytical relationships
    # -----------------------------------------

    edges = [
        {
            "source": "revenue",
            "target": "gross_profit",
            "relationship": "Revenue contributes to Gross Profit"
        },
        {
            "source": "gross_profit",
            "target": "ebitda",
            "relationship": "Gross Profit contributes to EBITDA"
        },
        {
            "source": "ebitda",
            "target": "net_income",
            "relationship": "EBITDA contributes to Net Income"
        },
        {
            "source": "net_income",
            "target": "operating_cash",
            "relationship": "Compare earnings with operating cash generation"
        },
        {
            "source": "current_ratio",
            "target": "net_income",
            "relationship": "Liquidity context"
        },
        {
            "source": "debt_equity",
            "target": "net_income",
            "relationship": "Leverage context"
        }
    ]

    # Add relationships only when the corresponding
    # optional return metric is actually available.
    for metric_id, _ in optional_return_metrics:

        if metric_id in row.index:
            edges.append({
                "source": "net_income",
                "target": metric_id,
                "relationship": f"Net Income influences {metric_id.upper()}"
            })

    # Drop edges that reference skipped (unavailable) nodes so a
    # partial upload degrades gracefully instead of rendering
    # dangling relationships.
    _available_ids = {node["id"] for node in nodes}
    edges = [
        edge
        for edge in edges
        if edge["source"] in _available_ids
        and edge["target"] in _available_ids
    ]

    return {
        "company": company,
        "year": year,
        "nodes": nodes,
        "edges": edges
    }


def get_map_summary(financial_map):
    """Return summary information for the financial map."""

    return {
        "company": financial_map["company"],
        "year": financial_map["year"],
        "node_count": len(financial_map["nodes"]),
        "edge_count": len(financial_map["edges"])
    }

"""Relevance-grounded AI context builder for FinSight AI Chat.

build_financial_ai_context() assembles ONE structured evidence object per
question from already-computed engine outputs (never recalculating, never
fabricating). A relevance layer selects only the evidence the question
needs: mentioned years, detected metrics, risk/peer/cash topics.

All monetary inputs are USD millions (source units); values are
pre-formatted with the dashboard currency convention (USD Billion /
INR Lakh Crore) so the model never converts.
"""

import re

import pandas as pd

USD_TO_INR = 83.0

_METRIC_KEYWORDS = {
    "revenue": ["revenue", "sales", "turnover", "top line"],
    "gross_profit": ["gross profit"],
    "net_income": ["net income", "net profit", "profit", "earnings",
                   "bottom line", "pat"],
    "ebitda": ["ebitda"],
    "net_profit_margin": ["net profit margin", "net margin", "npm",
                          "profit margin", "profitability"],
    "gross_margin": ["gross margin"],
    "ebitda_margin": ["ebitda margin"],
    "current_ratio": ["current ratio", "liquidity"],
    "debt_equity_ratio": ["debt equity", "debt/equity", "debt-to-equity",
                          "d/e", "leverage"],
    "roe": ["roe", "return on equity"],
    "roa": ["roa", "return on assets"],
    "roi": ["roi", "return on investment"],
    "cash_flow_from_operating": ["operating cash flow", "cash flow",
                                 "cash generation"],
    "cash_flow_from_investing": ["investing cash flow"],
    "cash_flow_from_financial_activities": ["financing cash flow"],
}

_RISK_WORDS = ["risk", "risks", "risky", "concern", "worry", "danger",
               "health", "healthy", "safe", "distress"]
_PEER_WORDS = ["peer", "peers", "category", "competitor", "compare",
               "comparison", "versus", "vs", "benchmark", "industry"]
_CASH_WORDS = ["cash flow", "cash generation", "cash conversion",
               "operating cash"]
_GROWTH_WORDS = ["growth", "growing", "trend", "changed", "change",
                 "increase", "decrease", "decline", "between", "from",
                 "yoy", "history", "historical", "over time", "period"]


def detect_metrics(question):
    """Return normalized metric columns relevant to the question."""
    text = str(question or "").lower()
    hits = []
    for column, aliases in _METRIC_KEYWORDS.items():
        for alias in aliases:
            if alias in text:
                hits.append(column)
                break
    return hits


def detect_years(question):
    """Return sorted unique 4-digit years mentioned (1900-2099)."""
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", str(question or ""))
    return sorted({int(year) for year in years})


def _num(value):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _money(value, currency):
    value = _num(value)
    if value is None:
        return None
    if currency == "INR":
        return f"Rs {value * USD_TO_INR / 1_000_000:,.2f} LC"
    return f"${value / 1_000:,.2f}B"


def _pct(value, signed=False):
    value = _num(value)
    if value is None:
        return None
    return f"{value:+.2f}%" if signed else f"{value:.2f}%"


def _ratio(value):
    value = _num(value)
    if value is None:
        return None
    return f"{value:.2f}x"


_MONEY_COLS = {"revenue", "gross_profit", "net_income", "ebitda",
               "cash_flow_from_operating", "cash_flow_from_investing",
               "cash_flow_from_financial_activities", "net_cash_flow"}
_PCT_COLS = {"net_profit_margin", "gross_margin", "ebitda_margin",
             "revenue_yoy", "net_income_yoy", "roe", "roa", "roi",
             "operating_cash_flow_margin"}
_RATIO_COLS = {"current_ratio", "debt_equity_ratio"}


def _fmt_value(column, value, currency):
    if column in _MONEY_COLS:
        return _money(value, currency)
    if column in _PCT_COLS:
        return _pct(value)
    if column in _RATIO_COLS:
        return _ratio(value)
    formatted = _num(value)
    return None if formatted is None else f"{formatted:,.2f}"


def _company_frame(metrics_df, company):
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()
    try:
        frame = metrics_df[
            metrics_df["company"].astype(str) == str(company)].copy()
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
        return frame.sort_values("year").reset_index(drop=True)
    except (KeyError, ValueError, TypeError):
        return pd.DataFrame()


def build_financial_ai_context(
    question,
    company=None,
    selected_year=None,
    currency="USD",
    metrics_df=None,
    snapshot=None,
    health=None,
    factors=None,
    benchmark=None,
    anomalies=None,
    quality=None,
    data_source="default dataset",
    filename=None,
    raw_columns=None,
    raw_row_count=None,
    mapped_fields=None,
    missing_fields=None,
    preview_rows=None,
    deterministic_answer=None,
    deterministic_summary=None,
):
    """Build the grounded evidence object for one chat question.

    Sections included by relevance: dataset/state always; snapshot +
    mentioned-years values when financial data exists; metric histories
    for detected metrics; health+anomalies for risk questions; benchmark
    for peer questions; cash/balance when asked or broadly relevant.
    Unknown/empty datasets yield schema-only context with an explicit
    cannot-assess instruction — never fabricated metrics.
    """
    currency = "INR" if currency == "INR" else "USD"
    ctx = {
        "question_type": "financial",
        "data_source": data_source,
        "current_state": {
            "company": company,
            "selected_year": selected_year,
            "currency": currency,
            "filename": filename,
        },
    }

    frame = _company_frame(metrics_df, company)
    has_financial = not frame.empty and "revenue" in frame.columns

    # ---- dataset metadata (always) ----
    try:
        years_available = sorted(
            int(year) for year in frame["year"].dropna().unique().tolist()
        ) if not frame.empty and "year" in frame.columns else []
    except (ValueError, TypeError):
        years_available = []
    ctx["dataset"] = {
        "rows": int(len(metrics_df)) if metrics_df is not None else 0,
        "columns": list(metrics_df.columns)
        if metrics_df is not None else (raw_columns or []),
        "company_years": years_available,
        "year_span": (f"{years_available[0]}-{years_available[-1]}"
                      if years_available else None),
    }

    if not has_financial:
        ctx["question_type"] = "unknown_dataset"
        ctx["unknown_dataset"] = {
            "status": "Financial mapping incomplete",
            "schema": raw_columns or [],
            "row_count": raw_row_count,
            "recognized_fields": mapped_fields or [],
            "missing_fields": missing_fields or [],
            "preview_rows": (preview_rows or [])[:5],
            "instruction": (
                "The dataset could not be mapped to financial fields. "
                "Answer only from the schema, recognized fields and "
                "preview rows above. If a requested financial metric "
                "is absent, say financial health cannot be reliably "
                "assessed and do not invent values."),
        }
        if deterministic_answer:
            ctx["deterministic_note"] = deterministic_answer
        return ctx

    # ---- snapshot (latest selected-year values) ----
    snap = snapshot or {}
    ctx["snapshot"] = {
        "revenue": _money(snap.get("revenue"), currency),
        "gross_profit": _money(snap.get("gross_profit"), currency),
        "net_income": _money(snap.get("net_income"), currency),
        "ebitda": _money(snap.get("ebitda"), currency),
        "operating_cash_flow": _money(
            snap.get("operating_cash_flow", snap.get("cash_flow_from_operating")),
            currency),
        "net_profit_margin": _pct(snap.get("net_profit_margin")),
        "current_ratio": _ratio(snap.get("current_ratio")),
        "debt_equity_ratio": _ratio(snap.get("debt_equity_ratio")),
        "roe": _pct(snap.get("roe")),
        "roa": _pct(snap.get("roa")),
    }
    ctx["snapshot"] = {key: value for key, value in ctx["snapshot"].items()
                       if value is not None}

    # ---- explicitly mentioned years (never limited to selected year) ----
    mentioned = [year for year in detect_years(question)
                 if year in years_available][:6]
    if mentioned:
        year_rows = {}
        for year in mentioned:
            row = frame[frame["year"] == year]
            if row.empty:
                continue
            record = row.iloc[0]
            values = {}
            for column in ["revenue", "gross_profit", "net_income",
                           "ebitda", "gross_margin", "ebitda_margin",
                           "net_profit_margin", "current_ratio",
                           "debt_equity_ratio"]:
                if column in frame.columns:
                    formatted = _fmt_value(column, record.get(column),
                                           currency)
                    if formatted is not None:
                        values[column] = formatted
            year_rows[str(year)] = values
        ctx["mentioned_years"] = year_rows
        if len(mentioned) >= 2 and "revenue" in frame.columns:
            first, second = mentioned[0], mentioned[1]
            first_v = _num(frame.loc[frame["year"] == first,
                                     "revenue"].iloc[0]) \
                if not frame.loc[frame["year"] == first].empty else None
            second_v = _num(frame.loc[frame["year"] == second,
                                      "revenue"].iloc[0]) \
                if not frame.loc[frame["year"] == second].empty else None
            if first_v is not None and second_v is not None \
                    and first_v != 0:
                delta = second_v - first_v
                ctx["revenue_pair_change"] = {
                    "from": _money(first_v, currency),
                    "to": _money(second_v, currency),
                    "absolute_change": _money(delta, currency),
                    "pct_change": _pct(delta / abs(first_v) * 100, True),
                }

    # ---- detected-metric histories ----
    metrics_hit = [column for column in detect_metrics(question)
                   if column in frame.columns][:3]
    text_lower = str(question or "").lower()
    wants_growth = any(word in text_lower for word in _GROWTH_WORDS)
    if metrics_hit:
        histories = {}
        for column in metrics_hit:
            series = []
            for _, row in frame.iterrows():
                formatted = _fmt_value(column, row.get(column), currency)
                if formatted is not None:
                    series.append({int(row["year"]): formatted})
            if series:
                histories[column] = series
            yoy_col = {"revenue": "revenue_yoy",
                       "net_income": "net_income_yoy"}.get(column)
            if wants_growth and yoy_col and yoy_col in frame.columns:
                latest = frame.dropna(subset=[yoy_col])
                if not latest.empty:
                    histories[yoy_col + "_latest"] = _pct(
                        latest.iloc[-1][yoy_col], True)
        if histories:
            ctx["metric_histories"] = histories

    # ---- health / risk ----
    if any(word in text_lower for word in _RISK_WORDS) or not metrics_hit:
        health = health or {}
        ctx["health"] = {
            "health_score": health.get("score"),
            "risk_score": health.get("risk"),
            "status": health.get("status"),
            "factors": factors or {},
        }

    # ---- anomalies ----
    try:
        company_anoms = []
        if anomalies is not None and not anomalies.empty:
            subset = anomalies[
                anomalies["company"].astype(str) == str(company)]
            for _, finding in subset.tail(8).iterrows():
                change = _num(finding.get("change"))
                company_anoms.append({
                    "year": int(finding.get("year"))
                    if pd.notna(finding.get("year")) else None,
                    "severity": finding.get("severity"),
                    "type": finding.get("finding_type"),
                    "metric": finding.get("metric"),
                    "change_pct": round(change, 2)
                    if change is not None else None,
                })
    except (KeyError, ValueError, TypeError):
        company_anoms = []
    if company_anoms and (any(word in text_lower for word in _RISK_WORDS)
                          or "anomal" in text_lower or not metrics_hit):
        ctx["anomalies"] = company_anoms

    # ---- benchmark ----
    if any(word in text_lower for word in _PEER_WORDS):
        bench = benchmark or {}
        diffs = []
        row = bench.get("row", {}) if isinstance(bench, dict) else {}
        for label, company_key, peer_key, kind in [
            ("Revenue", "revenue_company", "revenue_peer_avg", "money"),
            ("Net Income", "net_income_company",
             "net_income_peer_avg", "money"),
            ("Net Margin", "net_profit_margin_company",
             "net_profit_margin_peer_avg", "pct"),
            ("Current Ratio", "current_ratio_company",
             "current_ratio_peer_avg", "x"),
            ("Debt/Equity", "debt_equity_ratio_company",
             "debt_equity_ratio_peer_avg", "x"),
            ("ROA", "roa_company", "roa_peer_avg", "pct"),
            ("ROE", "roe_company", "roe_peer_avg", "pct"),
        ]:
            try:
                company_v = _num(row.get(company_key))
                peer_v = _num(row.get(peer_key))
            except AttributeError:
                continue
            if company_v is None or peer_v is None:
                continue
            if kind == "money":
                diffs.append(
                    f"{label}: company {_money(company_v, currency)} vs "
                    f"peer avg {_money(peer_v, currency)}")
            elif kind == "pct":
                diffs.append(
                    f"{label}: company {company_v:.2f}% vs peer avg "
                    f"{peer_v:.2f}% ({company_v - peer_v:+.2f}pp)")
            else:
                diffs.append(
                    f"{label}: company {company_v:.2f}x vs peer avg "
                    f"{peer_v:.2f}x")
        ctx["benchmark"] = {
            "peer_count": bench.get("peer_count", 0),
            "category": bench.get("category"),
            "comparisons": diffs,
        }

    # ---- cash & balance when asked or broadly relevant ----
    if any(word in text_lower for word in _CASH_WORDS) or not metrics_hit:
        cash = {}
        latest_row = frame.iloc[-1] if not frame.empty else None
        if latest_row is not None:
            for column, label in [
                ("cash_flow_from_operating", "operating"),
                ("cash_flow_from_investing", "investing"),
                ("cash_flow_from_financial_activities", "financing"),
                ("net_cash_flow", "net"),
            ]:
                if column in frame.columns:
                    formatted = _money(latest_row.get(column), currency)
                    if formatted is not None:
                        cash[label] = formatted
        if cash:
            ctx["cash_flow_latest"] = cash

    # ---- quality ----
    if quality:
        ctx["data_quality"] = {
            "status": quality.get("status"),
            "missing_values": quality.get("missing"),
            "duplicates": quality.get("duplicates"),
            "invalid_numeric": quality.get("invalid"),
        }

    if deterministic_answer:
        ctx["deterministic_answer"] = deterministic_answer
    if deterministic_summary:
        ctx["deterministic_summary"] = deterministic_summary

    return ctx

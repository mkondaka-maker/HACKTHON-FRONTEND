"""Professional multi-page PDF financial research report generator.

Data flow: build_report_context() recomputes every number with the
existing engine modules (no duplicated business logic, no fabricated
values) and generate_financial_report() renders a real PDF document
(ReportLab Platypus + matplotlib charts rendered from data).

All monetary inputs are USD millions (source units). Display
conversion for USD Billion / INR Lakh Crore mirrors the dashboard.
"""

import re
from datetime import date
from io import BytesIO

import pandas as pd

from engine.ai_investigation import (
    build_ai_investigation_context,
    extract_ai_signals,
    generate_mock_ai_report,
)
from engine.anomaly import detect_anomalies, get_anomaly_summary
from engine.benchmark import build_peer_benchmark
from engine.dashboard import get_company_snapshot, get_company_trend
from engine.data_quality import analyze_data_quality
from engine.financial_engine import calculate_financial_metrics
from engine.financial_map import build_financial_map
from engine.investigation_report import generate_investigation_report
from engine.risk import (
    FACTOR_LABELS,
    build_health_risk_scores,
    calculate_health_factor_scores,
    classify_health,
)
from engine.validation import run_validation
from engine.visuals import (
    get_balance_sheet_history,
    get_cash_flow_history,
    get_growth_history,
)

USD_TO_INR = 83.0
NOT_AVAILABLE = "Not available in source data"


# ------------------------------------------------------------------
# Formatting (PDF-safe: no rupee glyph — Helvetica/WinAnsi lacks it)
# ------------------------------------------------------------------

def _num(value):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt_money(value, currency):
    value = _num(value)
    if value is None:
        return NOT_AVAILABLE
    if currency == "INR":
        return f"Rs {value * USD_TO_INR / 1_000_000:,.2f} Lakh Cr"
    return f"${value / 1000:,.2f}B"


def fmt_pct(value, signed=False):
    value = _num(value)
    if value is None:
        return NOT_AVAILABLE
    if signed:
        return f"{value:+.2f}%"
    return f"{value:.2f}%"


def fmt_ratio(value):
    value = _num(value)
    if value is None:
        return NOT_AVAILABLE
    return f"{value:.2f}x"


def fmt_plain(value):
    value = _num(value)
    if value is None:
        return NOT_AVAILABLE
    return f"{value:,.2f}"


def display_values(values, currency):
    """Source USD millions -> chart units (USD B / INR Lakh Cr)."""
    out = []
    for value in values:
        value = _num(value)
        if value is None:
            out.append(None)
            continue
        if currency == "INR":
            out.append(value * USD_TO_INR / 1_000_000)
        else:
            out.append(value / 1000)
    return out


def money_unit(currency, short=False):
    if currency == "INR":
        return "Rs Lakh Cr" if short else "INR Lakh Crore"
    return "USD B" if short else "USD Billion"


def sanitize_filename(text):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_") or "Company"


def report_filename(company, year):
    return f"FinSight_AI_{sanitize_filename(company)}_{int(year)}.pdf"


# ------------------------------------------------------------------
# Context builder — every number recomputed via engine modules
# ------------------------------------------------------------------

def _safe_frame(df, predicate_cols):
    return df


def build_report_context(df, company, selected_year, currency="USD",
                         chat_history=None, scenario=None, data_source=None,
                         category=None):
    """Collect all report inputs. Never raises: sections degrade to
    'not available' notes instead of crashing on missing fields."""
    ctx = {
        "company": str(company),
        "selected_year": int(selected_year),
        "currency": "INR" if currency == "INR" else "USD",
        "generated": date.today().strftime("%B %d, %Y"),
        "data_source": data_source or "default dataset (data/clean_data.csv)",
        "chat_history": chat_history or [],
        "scenario": scenario or {},
        "sections": {},
    }

    company = str(company)
    year = int(selected_year)

    # --- metrics (financial engine) ---------------------------------
    try:
        metrics = calculate_financial_metrics(df.copy())
    except Exception:
        metrics = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    ctx["metrics_ok"] = not metrics.empty

    # --- category / period ------------------------------------------
    cat = category
    try:
        if cat is None and "category" in df.columns:
            vals = df.loc[df["company"].astype(str) == company, "category"]
            vals = vals.dropna().astype(str).str.strip()
            vals = vals[vals.str.len() > 0]
            cat = vals.iloc[0] if not vals.empty else "Not provided"
    except Exception:
        cat = "Not provided"
    ctx["category"] = cat or "Not provided"

    try:
        years = pd.to_numeric(
            df.loc[df["company"].astype(str) == company, "year"],
            errors="coerce",
        ).dropna().astype(int)
        years = years[years <= year]
        ctx["earliest_year"] = int(years.min()) if not years.empty else year
    except Exception:
        ctx["earliest_year"] = year

    # --- snapshot / trend --------------------------------------------
    try:
        ctx["snapshot"] = get_company_snapshot(df, company, year)
    except Exception:
        ctx["snapshot"] = {}
    try:
        trend = get_company_trend(df, company)
        trend = trend[trend["year"].astype(int) <= year].copy()
        ctx["trend"] = trend.sort_values("year").reset_index(drop=True)
    except Exception:
        ctx["trend"] = pd.DataFrame()

    # --- health / risk ------------------------------------------------
    try:
        risk_df = build_health_risk_scores(metrics, year)
        sel = risk_df[risk_df["company"] == company]
        if not sel.empty:
            row = sel.iloc[0]
            ctx["health"] = {
                "score": float(row["health_score"]),
                "risk": float(row["risk_score"]),
                "status": str(row["health_classification"]),
            }
        else:
            raise ValueError("no risk row")
    except Exception:
        ctx["health"] = {"score": None, "risk": None, "status": "Unavailable"}

    try:
        mrow = metrics[
            (metrics["company"].astype(str) == company)
            & (metrics["year"].astype(int) == year)
        ]
        ctx["factors"] = (
            calculate_health_factor_scores(mrow.iloc[0])
            if not mrow.empty else {}
        )
    except Exception:
        ctx["factors"] = {}

    # --- validation / quality -----------------------------------------
    try:
        validation_df = run_validation(metrics)
        ctx["validation"] = validation_df.to_dict(orient="records")
        ctx["validation_summary"] = {
            "tests": len(validation_df),
            "passed": int((validation_df["status"] == "PASS").sum()),
            "review": int((validation_df["status"] == "REVIEW").sum()),
        }
    except Exception:
        ctx["validation"] = []
        ctx["validation_summary"] = {"tests": 0, "passed": 0, "review": 0}

    try:
        quality = analyze_data_quality(df)
        ctx["quality"] = {
            "status": quality.get("status", "UNKNOWN"),
            "missing": int(quality.get("missing_values", 0)),
            "duplicates": int(quality.get("duplicate_records", 0)),
            "invalid": int(quality.get("numeric_anomalies", 0)),
            "issues": list(quality.get("issues", [])),
        }
    except Exception:
        ctx["quality"] = {"status": "UNKNOWN", "missing": 0,
                          "duplicates": 0, "invalid": 0, "issues": []}

    # --- feature areas --------------------------------------------------
    core_fields = ["revenue", "gross_profit", "net_income", "ebitda",
                   "share_holder_equity", "cash_flow_from_operating",
                   "cash_flow_from_investing",
                   "cash_flow_from_financial_activities", "current_ratio",
                   "debt_equity_ratio"]
    optional_fields = ["roe", "roa", "roi", "total_assets",
                       "total_liabilities", "free_cash_flow",
                       "earning_per_share", "number_of_employees"]
    try:
        cols = set(df.columns)
        ctx["features"] = {
            "core": [c for c in core_fields if c in cols],
            "core_missing": [c for c in core_fields if c not in cols],
            "optional": [c for c in optional_fields if c in cols],
            "extra": sorted(
                c for c in cols
                if c not in core_fields and c not in optional_fields
                and c not in ("year", "company", "category")
            ),
        }
    except Exception:
        ctx["features"] = {"core": [], "core_missing": core_fields,
                           "optional": [], "extra": []}

    # --- investigation + AI review --------------------------------------
    try:
        ctx["investigation"] = generate_investigation_report(
            metrics, company, year)
    except Exception:
        ctx["investigation"] = {}

    try:
        benchmark_df = build_peer_benchmark(metrics, year)
    except Exception:
        try:
            benchmark_df = build_peer_benchmark(df, year)
        except Exception:
            benchmark_df = pd.DataFrame()

    try:
        risk_all = build_health_risk_scores(metrics, year)
    except Exception:
        risk_all = pd.DataFrame()
    try:
        fin_map = build_financial_map(metrics, company, year)
    except Exception:
        fin_map = None

    try:
        ai_context = build_ai_investigation_context({
            "company": company, "year": year, "financial_data": metrics,
            "validation": None, "anomalies": None, "benchmark": benchmark_df,
            "risk": risk_all, "investigation": ctx["investigation"],
            "scenarios": None, "financial_map": fin_map,
        })
        signals = extract_ai_signals(ai_context)
        from engine.ai_investigation import generate_mock_ai_report
        ctx["ai_report"] = generate_mock_ai_report(signals)
    except Exception:
        ctx["ai_report"] = {}

    # --- anomalies ---------------------------------------------------------
    try:
        anomaly_df = detect_anomalies(metrics)
        if anomaly_df.empty:
            company_anoms = anomaly_df.copy()
        else:
            company_anoms = anomaly_df[
                (anomaly_df["company"] == company)
                & (anomaly_df["year"].astype(int) <= year)
            ].copy().sort_values("year")
        ctx["anomalies"] = {
            "summary": get_anomaly_summary(
                company_anoms[company_anoms["year"].astype(int) == year]
                if not company_anoms.empty else company_anoms),
            "timeline": company_anoms.to_dict(orient="records"),
            "latest": company_anoms[
                company_anoms["year"].astype(int) == year
            ].to_dict(orient="records") if not company_anoms.empty else [],
        }
    except Exception:
        ctx["anomalies"] = {"summary": {"total_findings": 0, "high": 0,
                                        "medium": 0, "low": 0},
                            "timeline": [], "latest": []}

    # --- benchmark ------------------------------------------------------------
    try:
        sel_bench = benchmark_df[benchmark_df["company"] == company]
        if sel_bench.empty:
            ctx["benchmark"] = {"available": False, "peer_count": 0}
        else:
            brow = sel_bench.iloc[0]
            ctx["benchmark"] = {
                "available": True,
                "peer_count": int(brow["peer_count"]),
                "category": str(brow.get("category", ctx["category"])),
                "year": int(brow.get("year", year)),
                "row": brow.to_dict(),
            }
    except Exception:
        ctx["benchmark"] = {"available": False, "peer_count": 0}

    # --- visuals histories -------------------------------------------------------
    try:
        ctx["growth"] = get_growth_history(metrics, company, year)
    except Exception:
        ctx["growth"] = {"available": False, "history": pd.DataFrame(),
                         "insight": ""}
    try:
        ctx["cashflow"] = get_cash_flow_history(metrics, company, year)
    except Exception:
        ctx["cashflow"] = {"available": False, "history": pd.DataFrame(),
                           "latest": {}, "margin_available": False,
                           "free_cash_flow_available": False}
    try:
        ctx["balance"] = get_balance_sheet_history(df, company, year)
    except Exception:
        ctx["balance"] = {"available": False, "history": pd.DataFrame(),
                          "latest": {}, "missing": []}

    # --- scenario (mirror dashboard formulas, labelled illustrative) ---------------
    snap = ctx["snapshot"]
    scen = dict(ctx["scenario"])
    try:
        scen_rev = float(scen.get("revenue_growth", 10.0))
        scen_margin = float(scen.get("target_margin", 30.0))
        cur_rev = float(snap.get("revenue"))
        cur_ni = float(snap.get("net_income"))
        cur_margin = float(snap.get("net_profit_margin"))
        proj_rev = cur_rev * (1 + scen_rev / 100)
        proj_ni = proj_rev * (scen_margin / 100)
        ctx["scenario_computed"] = {
            "inputs": {"revenue_growth": scen_rev,
                       "target_margin": scen_margin},
            "current": {"revenue": cur_rev, "net_income": cur_ni,
                        "margin": cur_margin},
            "projected": {"revenue": proj_rev, "net_income": proj_ni,
                          "margin": scen_margin},
        }
    except (TypeError, ValueError, KeyError):
        ctx["scenario_computed"] = None

    return ctx

# ------------------------------------------------------------------
# Deterministic insight builders (evidence only, no invented causes)
# ------------------------------------------------------------------

def _latest_pair(history, metric_a, metric_b):
    """Latest (a, b) values from a long-format year/metric/value frame."""
    try:
        pivot = history.pivot_table(index="year", columns="metric",
                                    values="growth", aggfunc="first")
        both = pivot.dropna(subset=[metric_a, metric_b])
        if both.empty:
            return None, None, None
        last = both.iloc[-1]
        return int(both.index[-1]), float(last[metric_a]), float(last[metric_b])
    except Exception:
        return None, None, None


def build_positives_concerns(ctx):
    """Key positives / concerns strictly from calculated values."""
    positives, concerns = [], []
    snap = ctx.get("snapshot", {}) or {}
    trend = ctx.get("trend")
    rev = _num(snap.get("revenue"))
    ni = _num(snap.get("net_income"))
    npm = _num(snap.get("net_profit_margin"))
    cr = _num(snap.get("current_ratio"))
    de = _num(snap.get("debt_equity_ratio"))
    ocf = _num(snap.get("operating_cash_flow"))

    if ctx.get("growth", {}).get("available"):
        y, rg, ig = _latest_pair(ctx["growth"]["history"],
                                 "Revenue Growth", "Net Income Growth")
        if y and rg is not None and rg > 0:
            positives.append(
                f"Revenue grew {rg:+.2f}% in {y} (year over year).")
        if y and rg is not None and ig is not None and rg >= 10 \
                and ig < rg - 20:
            concerns.append(
                f"Profitability lag in {y}: net income growth "
                f"({ig:+.2f}%) trailed revenue growth ({rg:+.2f}%) "
                f"by more than 20 points.")

    if isinstance(trend, pd.DataFrame) and len(trend) >= 2:
        try:
            first_gm = float(trend["gross_margin"].dropna().iloc[0])
            last_gm = float(trend["gross_margin"].dropna().iloc[-1])
            if last_gm > first_gm:
                positives.append(
                    f"Gross margin improved from {first_gm:.2f}% to "
                    f"{last_gm:.2f}% over the analysis period.")
            elif last_gm < first_gm:
                concerns.append(
                    f"Gross margin compressed from {first_gm:.2f}% to "
                    f"{last_gm:.2f}% over the analysis period.")
            first_nm = float(trend["net_profit_margin"].dropna().iloc[0])
            last_nm = float(trend["net_profit_margin"].dropna().iloc[-1])
            if last_nm < first_nm - 1:
                concerns.append(
                    f"Net profit margin deteriorated from {first_nm:.2f}% "
                    f"to {last_nm:.2f}% over the analysis period.")
        except (IndexError, ValueError, KeyError):
            pass

    if ocf is not None and ocf > 0:
        positives.append(
            f"Positive operating cash flow of {fmt_money(ocf, ctx['currency'])} "
            f"in {ctx['selected_year']}.")
    if ocf is not None and ni is not None and ni > 0 and ocf < 0:
        concerns.append(
            "Earnings-cash divergence: positive reported net income "
            "alongside negative operating cash flow.")
    if npm is not None and npm >= 15:
        positives.append(
            f"Strong net profitability (net margin {npm:.2f}%).")
    if ni is not None and ni < 0:
        concerns.append("Negative net income in the latest period.")
    if cr is not None and cr >= 1.5:
        positives.append(f"Healthy short-term liquidity (current ratio {cr:.2f}x).")
    if cr is not None and cr < 1:
        concerns.append(
            f"Weak liquidity: current ratio {cr:.2f}x is below 1.0x.")
    if de is not None and de > 2:
        concerns.append(f"Elevated leverage (debt/equity {de:.2f}x).")

    try:
        de_hist = None
        if isinstance(trend, pd.DataFrame) and "debt_equity_ratio" not in trend.columns:
            pass
    except Exception:
        pass

    for finding in ctx.get("anomalies", {}).get("latest", []):
        if str(finding.get("severity", "")).upper() == "HIGH":
            concerns.append(
                f"Material anomaly in {finding.get('year')}: "
                f"{finding.get('title', finding.get('finding_type', ''))}.")

    if not positives:
        positives.append("No material positives identified from available data.")
    if not concerns:
        concerns.append("No material concerns identified from available data.")

    return positives[:6], concerns[:6]


def build_benchmark_interpretation(ctx):
    """Deterministic peer-comparison sentences from calculated diffs."""
    bench = ctx.get("benchmark", {})
    if not bench.get("available") or bench.get("peer_count", 0) == 0:
        return []
    row = bench.get("row", {})
    lines = []

    def _diff(company_key, peer_key, label, unit):
        try:
            company = float(row[company_key])
            peer = float(row[peer_key])
        except (TypeError, ValueError, KeyError):
            return None
        if unit == "%":
            return (f"{label} is {company - peer:+.2f} percentage points "
                    f"{'above' if company >= peer else 'below'} the category "
                    f"peer average ({company:.2f}% vs {peer:.2f}%).")
        if unit == "x":
            return (f"{label} is {company:.2f}x versus a peer average of "
                    f"{peer:.2f}x.")
        return None

    for label, company_key, peer_key, unit in [
        ("Net margin", "net_profit_margin_company",
         "net_profit_margin_peer_avg", "%"),
        ("Current ratio", "current_ratio_company",
         "current_ratio_peer_avg", "x"),
        ("Debt/equity", "debt_equity_ratio_company",
         "debt_equity_ratio_peer_avg", "x"),
        ("ROE", "roe_company", "roe_peer_avg", "%"),
        ("ROA", "roa_company", "roa_peer_avg", "%"),
    ]:
        sentence = _diff(company_key, peer_key, label, unit)
        if sentence:
            lines.append(sentence)

    for label, company_key, peer_key in [
        ("Revenue", "revenue_company", "revenue_peer_avg"),
        ("Net income", "net_income_company", "net_income_peer_avg"),
    ]:
        try:
            company = float(row[company_key])
            peer = float(row[peer_key])
            if peer != 0:
                pct = (company - peer) / abs(peer) * 100
                lines.append(
                    f"{label} is {pct:+.2f}% "
                    f"{'above' if pct >= 0 else 'below'} the peer average "
                    f"({fmt_money(company, ctx['currency'])} vs "
                    f"{fmt_money(peer, ctx['currency'])}).")
        except (TypeError, ValueError, KeyError):
            continue

    return lines


def _priority_for(score_or_level):
    if score_or_level == "High" or (
            isinstance(score_or_level, (int, float))
            and score_or_level < 40):
        return "High"
    if score_or_level == "Medium" or (
            isinstance(score_or_level, (int, float))
            and score_or_level < 60):
        return "Medium"
    return "Low"


def build_ai_review_sections(ctx):
    """Ten grounded AI-review subsections: evidence + interpretation."""
    ai = ctx.get("ai_report", {}) or {}
    snap = ctx.get("snapshot", {}) or {}
    health = ctx.get("health", {}) or {}
    bench = ctx.get("benchmark", {}) or {}
    positives, concerns = build_positives_concerns(ctx)
    real_concerns = [c for c in concerns
                     if not c.startswith("No material concerns")]
    peer_lines = build_benchmark_interpretation(ctx)

    def _ev(*pairs):
        return [(m, v, str(ctx.get("selected_year"))) for m, v in pairs
                if v != NOT_AVAILABLE]

    sections = []

    sections.append({
        "title": "1. Executive interpretation",
        "evidence": _ev(
            ("Health score",
             f"{health.get('score')}/100" if health.get("score") is not None
             else NOT_AVAILABLE),
            ("Risk score",
             f"{health.get('risk')}/100" if health.get("risk") is not None
             else NOT_AVAILABLE),
            ("Overall status", health.get("status", NOT_AVAILABLE))),
        "interpretation": ai.get("executive_summary",
                                 "No AI executive summary was generated for "
                                 "this period."),
        "priority": _priority_for(
            100 - health["score"] if health.get("score") is not None else 50),
    })

    rev = _num(snap.get("revenue"))
    ni = _num(snap.get("net_income"))
    growth_insight = ctx.get("growth", {}).get("insight", "")
    sections.append({
        "title": "2. Performance",
        "evidence": _ev(("Revenue", fmt_money(rev, ctx["currency"])),
                        ("Net income", fmt_money(ni, ctx["currency"]))),
        "interpretation": growth_insight or
        "Growth comparison is unavailable for the selected period.",
        "priority": "Low" if not real_concerns else "Medium",
    })

    sections.append({
        "title": "3. Profitability",
        "evidence": _ev(
            ("Net profit margin", fmt_pct(snap.get("net_profit_margin"))),
            ("ROE", fmt_pct(snap.get("roe"))),
            ("ROA", fmt_pct(snap.get("roa")))),
        "interpretation": ai.get("root_cause_assessment",
                                 "No root-cause assessment was generated."),
        "priority": "Medium",
    })

    sections.append({
        "title": "4. Liquidity",
        "evidence": _ev(("Current ratio",
                         fmt_ratio(snap.get("current_ratio"))),),
        "interpretation": (
            "Short-term liquidity coverage implied by the current ratio "
            "above; values below 1.0x may warrant investigation into "
            "working-capital adequacy."),
        "priority": _priority_for(
            70 if (_num(snap.get("current_ratio")) or 0) >= 1 else 30),
    })

    sections.append({
        "title": "5. Leverage",
        "evidence": _ev(("Debt/equity",
                         fmt_ratio(snap.get("debt_equity_ratio"))),),
        "interpretation": (
            "Balance-sheet leverage implied by debt/equity above; "
            "elevated readings may warrant investigation into debt "
            "serviceability."),
        "priority": _priority_for(
            70 if (_num(snap.get("debt_equity_ratio")) or 0) <= 1 else 30),
    })

    cash_latest = ctx.get("cashflow", {}).get("latest", {})
    sections.append({
        "title": "6. Cash flow",
        "evidence": _ev(
            ("Operating cash flow",
             fmt_money(cash_latest.get("operating"), ctx["currency"])
             if "operating" in cash_latest else NOT_AVAILABLE),
            ("Net cash flow",
             fmt_money(cash_latest.get("net"), ctx["currency"])
             if "net" in cash_latest else NOT_AVAILABLE)),
        "interpretation": (
            "Cash generation versus reported profit is the key check here: "
            "persistent divergence between net income and operating cash "
            "flow could warrant investigation into earnings quality."),
        "priority": "Medium",
    })

    factors = ctx.get("factors", {})
    weak = min(factors, key=factors.get) if factors else None
    strong = max(factors, key=factors.get) if factors else None
    sections.append({
        "title": "7. Risk",
        "evidence": _ev(
            ("Weakest factor",
             f"{weak} ({factors[weak]:.1f})" if weak else NOT_AVAILABLE),
            ("Strongest factor",
             f"{strong} ({factors[strong]:.1f})" if strong else NOT_AVAILABLE)),
        "interpretation": (
            "Deterministic risk-factor scores decompose the headline "
            "health score; weaker factors merit deeper review."),
        "priority": _priority_for(
            factors.get(weak, 50) if weak else 50),
    })

    sections.append({
        "title": "8. Benchmark",
        "evidence": _ev(
            ("Peer group size",
             str(bench.get("peer_count", 0))),
            ("Peer category",
             str(bench.get("category", NOT_AVAILABLE)))),
        "interpretation": " ".join(peer_lines[:2]) if peer_lines else
        "No category peer comparison is available for this period.",
        "priority": "Low",
    })

    sections.append({
        "title": "9. Key concerns",
        "evidence": [],
        "interpretation": " ".join(real_concerns) if real_concerns else
        "No material concerns were identified from available data.",
        "priority": "High" if real_concerns else "Low",
    })

    findings = (ai.get("key_findings", []) or [])[:4]
    sections.append({
        "title": "10. Areas requiring further investigation",
        "evidence": [],
        "interpretation": " ".join(findings) if findings else
        "Follow-up items: validate anomalous movements, confirm ratio "
        "drivers against source filings, and review peer comparability.",
        "priority": "Medium",
    })

    return sections

# ------------------------------------------------------------------
# Document assembly
# ------------------------------------------------------------------

from xml.sax.saxutils import escape as _escape

from reportlab.lib.units import mm as _mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Image,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus.frames import Frame

from engine import pdf_charts as _charts
from engine.pdf_styles import (
    BOLD_FONT,
    MARGIN,
    PAGE_SIZE,
    ReportCanvas,
    callout,
    chat_message,
    get_styles,
    kpi_grid,
    rule,
    section_heading,
    spacer,
    styled_table,
)

CONTENT_WIDTH = PAGE_SIZE[0] - 2 * MARGIN


def esc(text):
    return _escape("" if text is None else str(text))


def _paras(text, style):
    """Split free text into Paragraph flowables."""
    parts = [p.strip() for p in str(text or "").replace("\r", "").split("\n")]
    return [Paragraph(esc(p), style) for p in parts if p]


def _img(buf, width_mm=170):
    from PIL import Image as _PILImage

    buf.seek(0)
    with _PILImage.open(buf) as im:
        width_px, height_px = im.size
    width = width_mm * _mm
    height = width * height_px / max(width_px, 1)
    buf.seek(0)
    return Image(buf, width=width, height=height)


def _caption(text, styles):
    return Paragraph(esc(text), styles["caption"])


class _ReportDoc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            name = flowable.style.name
            if name in ("h1", "h2"):
                level = 0 if name == "h1" else 1
                self.notify(
                    "TOCEntry", (level, flowable.getPlainText(), self.page))


def _build_doc(buffer, canvasmaker):
    doc = _ReportDoc(
        buffer, pagesize=PAGE_SIZE,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=24 * _mm, bottomMargin=18 * _mm,
        title="FinSight AI Financial Analysis Report",
        author="FinSight AI",
    )
    frame = Frame(MARGIN, 18 * _mm, CONTENT_WIDTH,
                  PAGE_SIZE[1] - 42 * _mm, id="main")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])
    return doc


def _toc_styles(styles):
    from reportlab.lib.styles import ParagraphStyle

    toc_level = ParagraphStyle(
        "toc0", parent=styles["toc"], fontName=BOLD_FONT,
        fontSize=10, leading=16,
    )
    toc_sub = ParagraphStyle(
        "toc1", parent=styles["toc"], leftIndent=8 * _mm,
        fontSize=9, leading=14,
    )
    return [toc_level, toc_sub]


# -- cover ----------------------------------------------------------

def _cover(ctx, styles):
    company = ctx["company"]
    health = ctx.get("health", {}) or {}
    snap = ctx.get("snapshot", {}) or {}
    cur = ctx["currency"]
    story = [
        spacer(18),
        Paragraph("FINSIGHT AI", styles["cover_title"]),
        Paragraph("FINANCIAL ANALYSIS &amp; RESEARCH REPORT",
                  styles["cover_sub"]),
        Paragraph(f"<b>Company:</b> {esc(company)}", styles["cover_field"]),
        Paragraph(f"<b>Category:</b> {esc(ctx.get('category', ''))}",
                  styles["cover_field"]),
        Paragraph(
            f"<b>Analysis Period:</b> {ctx.get('earliest_year', '')} – "
            f"{ctx['selected_year']}", styles["cover_field"]),
        Paragraph(f"<b>Report Generated:</b> {esc(ctx.get('generated', ''))}",
                  styles["cover_field"]),
        spacer(10),
    ]
    score = health.get("score")
    story.append(kpi_grid([
        ("Financial Health Score",
         f"{score:.1f} / 100" if score is not None else NOT_AVAILABLE),
        ("Risk Score",
         f"{health.get('risk'):.1f} / 100"
         if health.get("risk") is not None else NOT_AVAILABLE),
        ("Overall Status", esc(health.get("status", "Unavailable"))),
        ("Currency", "INR (Rs Lakh Cr)" if cur == "INR" else "USD ($ Billion)"),
    ], styles, cols=2))
    story.append(spacer(8))
    story.append(Paragraph("COMPANY SNAPSHOT", styles["h2"]))
    story.append(styled_table(
        ["Metric", "Latest Value"],
        [
            ["Revenue", fmt_money(snap.get("revenue"), cur)],
            ["Net Income", fmt_money(snap.get("net_income"), cur)],
            ["EBITDA", fmt_money(snap.get("ebitda"), cur)],
            ["Operating Cash Flow",
             fmt_money(snap.get("operating_cash_flow"), cur)],
            ["Net Profit Margin", fmt_pct(snap.get("net_profit_margin"))],
            ["Current Ratio", fmt_ratio(snap.get("current_ratio"))],
            ["Debt / Equity", fmt_ratio(snap.get("debt_equity_ratio"))],
        ],
        styles,
    ))
    story.append(spacer(6))
    story.append(Paragraph(
        "Source: " + esc(ctx.get("data_source", "")) +
        ". Monetary values in USD millions at source; displayed per "
        "selected currency. AI interpretations assist analysis only.",
        styles["caption"]))
    return story


# -- section builders -------------------------------------------------

def _executive_summary(ctx, styles):
    story = section_heading("1. Executive Summary", styles)
    health = ctx.get("health", {}) or {}
    story.append(kpi_grid([
        ("Financial Health Score",
         f"{health.get('score'):.1f} / 100"
         if health.get("score") is not None else NOT_AVAILABLE),
        ("Risk Score",
         f"{health.get('risk'):.1f} / 100"
         if health.get("risk") is not None else NOT_AVAILABLE),
        ("Overall Status", esc(health.get("status", "Unavailable"))),
    ], styles, cols=3))
    positives, concerns = build_positives_concerns(ctx)
    story.append(Paragraph("Key Positives", styles["h2"]))
    for item in positives:
        story.append(Paragraph(esc(item), styles["bullet"],
                               bulletText="•"))
    story.append(Paragraph("Key Concerns", styles["h2"]))
    for item in concerns:
        story.append(Paragraph(esc(item), styles["bullet"],
                               bulletText="•"))
    story.append(Paragraph("Executive AI Review", styles["h2"]))
    ai = ctx.get("ai_report", {}) or {}
    story.append(callout(
        esc(ai.get("executive_summary",
                   "No AI executive summary was generated.")),
        styles, tone="info"))
    for finding in (ai.get("key_findings", []) or [])[:6]:
        story.append(Paragraph(esc(finding), styles["bullet"],
                               bulletText="–"))
    return story


def _overview_section(ctx, styles):
    story = section_heading("2. Financial Overview", styles)
    snap = ctx.get("snapshot", {}) or {}
    trend = ctx.get("trend")
    cur = ctx["currency"]

    prev = {}
    try:
        if isinstance(trend, pd.DataFrame) and len(trend) >= 2:
            prev = trend.iloc[-2].to_dict()
    except Exception:
        prev = {}

    def _change(key, current, money=True):
        current_v = _num(current)
        previous_v = _num(prev.get(key)) if prev else None
        if current_v is None:
            return NOT_AVAILABLE, "—"
        current_s = fmt_money(current_v, cur) if money else (
            f"{current_v:.2f}%" if "margin" in key else f"{current_v:.2f}x")
        if previous_v is None:
            return current_s, "—"
        pct = ((current_v - previous_v) / abs(previous_v) * 100
               if previous_v != 0 else None)
        previous_s = fmt_money(previous_v, cur) if money else (
            f"{previous_v:.2f}%" if "margin" in key else f"{previous_v:.2f}x")
        if pct is None:
            return current_s, previous_s
        return current_s, f"{previous_s} ({pct:+.2f}%)"

    rows = []
    for label, key, money in [
        ("Revenue", "revenue", True),
        ("Gross Profit", "gross_profit", True),
        ("EBITDA", "ebitda", True),
        ("Net Income", "net_income", True),
        ("Operating Cash Flow", "operating_cash_flow", True),
        ("Net Profit Margin", "net_profit_margin", False),
        ("Current Ratio", "current_ratio", False),
        ("Debt / Equity", "debt_equity_ratio", False),
    ]:
        if key in ("gross_profit", "operating_cash_flow") \
                and key not in snap:
            continue
        current_s, change_s = _change(
            key, snap.get(key),
            money=(key not in ("net_profit_margin", "current_ratio",
                               "debt_equity_ratio")))
        if key == "net_profit_margin":
            current_s = fmt_pct(snap.get(key))
        elif key in ("current_ratio", "debt_equity_ratio"):
            current_s = fmt_ratio(snap.get(key))
        rows.append([label, current_s, change_s])

    story.append(styled_table(["Metric", "Latest", "Previous (change)"],
                              rows, styles))
    story.append(_caption(
        f"Latest values for {ctx['selected_year']}; change versus the "
        f"previous available year. Monetary values in {money_unit(cur)}.",
        styles))

    story.append(Paragraph("Historical Financial Summary", styles["h2"]))
    story.append(Paragraph(
        "Compact multi-year history up to the selected year. "
        "'n/a' means the field was not available in the source data.",
        styles["body"]))
    hist_table = _history_table(ctx, styles)
    if hist_table is not None:
        story.append(hist_table)
    else:
        story.append(Paragraph(NOT_AVAILABLE + " for historical summary.",
                               styles["body"]))
    return story


def _history_table(ctx, styles):
    trend = ctx.get("trend")
    if not isinstance(trend, pd.DataFrame) or trend.empty:
        return None
    cur = ctx["currency"]
    years = [int(y) for y in trend["year"].tolist()]
    if len(years) > 15:
        trend = trend.tail(15)
        years = [int(y) for y in trend["year"].tolist()]

    def _row(label, values, money=False, suffix=""):
        cells = [label]
        for value in values:
            value = _num(value)
            if value is None:
                cells.append("n/a")
            elif money:
                cells.append(fmt_money(value, cur).replace(
                    " Lakh Cr", " LC").replace("Billion", "B"))
            else:
                cells.append(f"{value:,.2f}{suffix}")
        return cells

    by_year = trend.set_index(trend["year"].astype(int)).to_dict(
        orient="index")
    metrics = [
        ("Revenue", "revenue", True, ""),
        ("Gross Profit", "gross_profit", True, ""),
        ("EBITDA", "ebitda", True, ""),
        ("Net Income", "net_income", True, ""),
        ("Gross Margin", "gross_margin", False, "%"),
        ("EBITDA Margin", "ebitda_margin", False, "%"),
        ("Net Profit Margin", "net_profit_margin", False, "%"),
    ]
    rows = []
    for label, key, money, suffix in metrics:
        if key not in trend.columns:
            continue
        rows.append(_row(label, [by_year[y].get(key) for y in years],
                         money, suffix))
    if not rows:
        return None
    headers = ["Metric"] + [str(y) for y in years]
    col_widths = [34 * _mm] + [
        (CONTENT_WIDTH - 34 * _mm) / len(years)] * len(years)
    table = styled_table(headers, rows, styles, col_widths=col_widths)
    table.setStyle(__import__("reportlab.platypus", fromlist=["TableStyle"])
                   .TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7)]))
    return table


def _performance_section(ctx, styles):
    story = section_heading("3. Performance Analysis", styles)
    trend = ctx.get("trend")
    cur = ctx["currency"]
    unit = money_unit(cur)

    if isinstance(trend, pd.DataFrame) and not trend.empty:
        years = [int(y) for y in trend["year"].tolist()]
        rev = display_values(trend["revenue"].tolist(), cur)
        story.append(Paragraph("Revenue Trend", styles["h2"]))
        story.append(_img(_charts.line_chart(
            years, [("Revenue", rev)], unit, "Revenue Trend")))
        story.append(_caption(
            f"Revenue in {unit} by year. Source: {ctx['data_source']}.",
            styles))

        if "net_income" in trend.columns:
            inc = display_values(trend["net_income"].tolist(), cur)
            story.append(Paragraph("Net Income Trend", styles["h2"]))
            story.append(_img(_charts.line_chart(
                years, [("Net Income", inc)], unit, "Net Income Trend")))
            story.append(_caption(
                f"Net income in {unit} by year. Source: {ctx['data_source']}.",
                styles))
    else:
        story.append(Paragraph(
            "Trend data is " + NOT_AVAILABLE.lower() + ".", styles["body"]))

    story.append(Paragraph("Revenue Growth vs Net Income Growth", styles["h2"]))
    growth = ctx.get("growth", {})
    if growth.get("available"):
        pivot = growth["history"].pivot_table(
            index="year", columns="metric", values="growth", aggfunc="first")
        years = [int(y) for y in pivot.index.tolist()]
        series = [(str(col), [float(v) if pd.notna(v) else None
                              for v in pivot[col].tolist()])
                  for col in pivot.columns]
        story.append(_img(_charts.line_chart(
            years, series, "Growth (%)",
            "Revenue Growth vs Net Income Growth", fmt="{:+.1f}")))
        story.append(_caption(
            "Year-over-year growth in percent (not currency-converted). "
            "Negative values indicate contraction.", styles))
        if growth.get("insight"):
            story.append(callout(esc(growth["insight"]), styles, tone="info"))
    else:
        story.append(Paragraph(
            "Growth comparison is " + NOT_AVAILABLE.lower() + ".",
            styles["body"]))
    return story


def _profitability_section(ctx, styles):
    story = section_heading("4. Profitability Analysis", styles)
    trend = ctx.get("trend")
    if not isinstance(trend, pd.DataFrame) or trend.empty:
        story.append(Paragraph("Margin data is unavailable.", styles["body"]))
        return story

    margin_cols = [c for c in ["gross_margin", "ebitda_margin",
                               "net_profit_margin"]
                   if c in trend.columns]
    if margin_cols:
        years = [int(y) for y in trend["year"].tolist()]
        labels = {"gross_margin": "Gross Margin",
                  "ebitda_margin": "EBITDA Margin",
                  "net_profit_margin": "Net Profit Margin"}
        series = [(labels[c],
                   [float(v) if pd.notna(v) else None
                    for v in trend[c].tolist()]) for c in margin_cols]
        story.append(_img(_charts.line_chart(
            years, series, "Margin (%)", "Profitability Margins")))
        story.append(_caption(
            "Margins in percent. Source: " + ctx["data_source"] + ".",
            styles))
        try:
            obs = []
            for col in margin_cols:
                vals = trend[col].dropna()
                if len(vals) >= 2:
                    obs.append(
                        f"{labels[col]} moved from {float(vals.iloc[0]):.2f}% "
                        f"to {float(vals.iloc[-1]):.2f}% over the selected "
                        f"period.")
            for ob in obs:
                story.append(Paragraph(esc(ob), styles["bullet"],
                                       bulletText="•"))
        except Exception:
            pass
    else:
        story.append(Paragraph("Margin data is unavailable.", styles["body"]))

    returns = []
    snap = ctx.get("snapshot", {}) or {}
    for label, key in [("ROA", "roa"), ("ROE", "roe")]:
        value = _num(snap.get(key))
        if value is not None:
            returns.append([label, f"{value:.2f}%"])
    if returns:
        story.append(Paragraph("Return Metrics (latest year)", styles["h2"]))
        story.append(styled_table(["Metric", "Value"], returns, styles))
    return story


def _cashflow_section(ctx, styles):
    story = section_heading("5. Cash Flow Analysis", styles)
    cash = ctx.get("cashflow", {})
    cur = ctx["currency"]
    unit = money_unit(cur)

    if not cash.get("available"):
        missing = ", ".join(cash.get("missing", []))
        story.append(Paragraph(
            "Cash flow analysis is unavailable"
            + (f": {esc(missing)} was not provided." if missing else "."),
            styles["body"]))
        return story

    hist = cash["history"]
    pivot = hist.pivot_table(index="year", columns="flow", values="value",
                             aggfunc="first")
    years = [int(y) for y in pivot.index.tolist()]
    series = [(str(col), display_values(pivot[col].tolist(), cur))
              for col in pivot.columns]
    story.append(_img(_charts.line_chart(
        years, series, unit, "Cash Flow Analysis")))
    story.append(_caption(
        f"Operating, investing, financing and net cash flow in {unit}. "
        f"Source: {ctx['data_source']}.", styles))

    latest = cash.get("latest", {})
    cards = []
    if "operating" in latest:
        cards.append(("Operating Cash Flow",
                      fmt_money(latest["operating"], cur)))
    if "net" in latest:
        cards.append(("Net Cash Flow", fmt_money(latest["net"], cur)))
    if cash.get("margin_available"):
        cards.append(("Operating Cash Flow Margin",
                      fmt_pct(latest.get("operating_margin"))))
    if cash.get("free_cash_flow_available") and "free_cash_flow" in latest:
        cards.append(("Free Cash Flow",
                      fmt_money(latest["free_cash_flow"], cur)))
    if cards:
        story.append(kpi_grid(cards, styles, cols=3))

    trend = ctx.get("trend")
    try:
        if isinstance(trend, pd.DataFrame) and not trend.empty \
                and "net_income" in trend.columns:
            op = pivot["Operating Cash Flow"] if "Operating Cash Flow" \
                in pivot.columns else None
            if op is not None:
                merged = pd.DataFrame({
                    "year": [int(y) for y in pivot.index.tolist()],
                    "net_income": display_values(
                        trend.set_index(trend["year"].astype(int))
                        .reindex(pivot.index)["net_income"].tolist(), cur),
                    "operating": display_values(op.tolist(), cur),
                }).dropna()
                if not merged.empty:
                    story.append(Paragraph("Net Income vs Operating Cash Flow",
                                           styles["h2"]))
                    story.append(_img(_charts.line_chart(
                        merged["year"].tolist(),
                        [("Net Income", merged["net_income"].tolist()),
                         ("Operating Cash Flow",
                          merged["operating"].tolist())],
                        unit, "Net Income vs Operating Cash Flow")))
                    story.append(_caption(
                        "Divergence between reported profit and cash "
                        "generation highlights earnings quality.", styles))
    except Exception:
        pass
    return story


def _balance_section(ctx, styles):
    story = section_heading("6. Balance Sheet & Financial Position", styles)
    bal = ctx.get("balance", {})
    cur = ctx["currency"]
    snap = ctx.get("snapshot", {}) or {}

    if not bal.get("available"):
        missing = ", ".join(bal.get("missing", []))
        story.append(Paragraph(
            "Balance sheet structure is unavailable"
            + (f": {esc(missing)} was not provided in the dataset."
               if missing else
               " for the selected company and year."),
            styles["body"]))
    else:
        pivot = bal["history"].pivot_table(
            index="year", columns="component", values="value", aggfunc="first")
        years = [int(y) for y in pivot.index.tolist()]
        order = ["Total Assets", "Total Liabilities", "Shareholder Equity"]
        components = [(c, display_values(pivot[c].tolist(), cur))
                      for c in order if c in pivot.columns]
        story.append(Paragraph("Assets vs Liabilities vs Equity", styles["h2"]))
        story.append(_img(_charts.stacked_bar(
            years, components, money_unit(cur),
            "Assets vs Liabilities vs Equity")))
        story.append(_caption(
            f"Financing composition in {money_unit(cur)}. Source: "
            f"{ctx['data_source']}.", styles))

    liq_rows, lev_rows = [], []
    if _num(snap.get("current_ratio")) is not None:
        liq_rows.append(["Current Ratio",
                         fmt_ratio(snap.get("current_ratio"))])
    assets = _num(snap.get("total_assets")) if "total_assets" in snap else None
    liab = _num(snap.get("total_liabilities")) \
        if "total_liabilities" in snap else None
    if assets and liab is not None and assets != 0:
        lev_rows.append(["Liabilities / Assets",
                         f"{liab / assets * 100:.2f}%"])
    if _num(snap.get("debt_equity_ratio")) is not None:
        lev_rows.append(["Debt / Equity",
                         fmt_ratio(snap.get("debt_equity_ratio"))])
    if liq_rows:
        story.append(Paragraph("Liquidity Analysis", styles["h2"]))
        story.append(styled_table(["Metric", "Latest"], liq_rows, styles))
    if lev_rows:
        story.append(Paragraph("Leverage Analysis", styles["h2"]))
        story.append(styled_table(["Metric", "Latest"], lev_rows, styles))
    if not liq_rows and not lev_rows and not bal.get("available"):
        story.append(Paragraph(
            "No liquidity or leverage metrics are available.", styles["body"]))
    return story

def _investigation_section(ctx, styles):
    story = section_heading("7. Financial Investigation", styles)
    summary = ctx.get("validation_summary", {})
    story.append(kpi_grid([
        ("Validation Tests", str(summary.get("tests", 0))),
        ("Passed", str(summary.get("passed", 0))),
        ("Needs Review", str(summary.get("review", 0))),
    ], styles, cols=3))

    validation = ctx.get("validation", [])
    if validation:
        rows = []
        for item in validation:
            max_diff = item.get("max_difference")
            try:
                detail = f"{float(max_diff):.2f}" \
                    if max_diff is not None and str(max_diff) != "nan" else "—"
            except (TypeError, ValueError):
                detail = "—"
            rows.append([
                str(item.get("test_name", "")),
                str(item.get("status", "")),
                str(item.get("records_checked", "—")),
                str(item.get("records_failed", "—")),
                detail,
            ])
        story.append(Paragraph("Validation Center", styles["h2"]))
        story.append(styled_table(
            ["Test", "Status", "Checked", "Failed", "Max Diff."], rows,
            styles))

    quality = ctx.get("quality", {})
    story.append(Paragraph("Data Quality", styles["h2"]))
    story.append(styled_table(
        ["Check", "Result"],
        [["Quality status", esc(quality.get("status", "UNKNOWN"))],
         ["Missing values", str(quality.get("missing", 0))],
         ["Duplicate records", str(quality.get("duplicates", 0))],
         ["Invalid numeric values", str(quality.get("invalid", 0))]],
        styles))
    for issue in quality.get("issues", [])[:6]:
        story.append(Paragraph(esc(issue), styles["bullet"], bulletText="•"))

    features = ctx.get("features", {})
    story.append(Paragraph("Detected Financial Areas", styles["h2"]))
    if features.get("core"):
        story.append(Paragraph(
            "Mapped core fields: " + esc(", ".join(features["core"])) + ".",
            styles["body"]))
    if features.get("core_missing"):
        story.append(Paragraph(
            "Core fields not available: "
            + esc(", ".join(features["core_missing"])) + ".", styles["body"]))
    if features.get("optional"):
        story.append(Paragraph(
            "Optional fields detected: "
            + esc(", ".join(features["optional"])) + ".", styles["body"]))
    extra = features.get("extra", [])
    if extra:
        shown = ", ".join(extra[:20])
        story.append(Paragraph(
            "Additional dataset features: " + esc(shown)
            + (f" (+{len(extra) - 20} more)." if len(extra) > 20 else "."),
            styles["body"]))

    investigation = ctx.get("investigation", {}) or {}
    if investigation:
        story.append(Paragraph("Investigation Findings", styles["h2"]))
        story.append(Paragraph(
            f"Primary driver: {esc(investigation.get('primary_driver', '—'))}",
            styles["body"]))
        story.append(Paragraph(
            esc(investigation.get("recommendation", "")), styles["body"]))
    return story


def _risk_section(ctx, styles):
    story = section_heading("8. Risk & Anomaly Analysis", styles)
    summary = ctx.get("anomalies", {}).get(
        "summary", {"total_findings": 0, "high": 0, "medium": 0, "low": 0})
    story.append(kpi_grid([
        ("Total Findings", str(summary.get("total_findings", 0))),
        ("High Severity", str(summary.get("high", 0))),
        ("Medium Severity", str(summary.get("medium", 0))),
        ("Low Severity", str(summary.get("low", 0))),
    ], styles, cols=4))

    timeline = ctx.get("anomalies", {}).get("timeline", [])
    if timeline:
        records = []
        for item in timeline:
            try:
                rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(
                    str(item.get("severity", "")).upper())
                if rank is None:
                    continue
                records.append((int(item.get("year")),
                                rank, str(item.get("finding_type", ""))[:22]))
            except (TypeError, ValueError):
                continue
        if records:
            story.append(Paragraph("Anomaly Timeline", styles["h2"]))
            story.append(_img(_charts.anomaly_scatter(
                records, "Anomaly Timeline")))
            story.append(_caption(
                "Detected anomaly years by severity. Each point is a "
                "rule-based finding, not a forecast.", styles))

    factors = ctx.get("factors", {})
    if factors:
        from engine.risk import FACTOR_LABELS as _LABELS

        labels = [_LABELS.get(k, k) for k in factors]
        scores = [float(v) for v in factors.values()]
        story.append(Paragraph("Risk Factor Breakdown", styles["h2"]))
        story.append(_img(_charts.hbar_factors(
            labels, scores, "Risk Factor Breakdown")))
        story.append(_caption(
            "Deterministic 0-100 factor scores; 50 is neutral. Higher is "
            "stronger.", styles))

    story.append(Paragraph("Key Risk Findings", styles["h2"]))
    latest = ctx.get("anomalies", {}).get("latest", [])
    if not latest:
        story.append(Paragraph(
            "No anomalies were recorded for the selected period.",
            styles["body"]))
    for item in latest:
        severity = str(item.get("severity", "")).upper()
        tone = "risk" if severity == "HIGH" else "warn" if severity == "MEDIUM" \
            else "info"
        change = _num(item.get("change"))
        body = (
            f"<b>{esc(str(severity))} RISK — "
            f"{esc(item.get('title', item.get('finding_type', '')))}</b>"
            f"<br/>Metric: {esc(item.get('metric', '—'))} &nbsp;|&nbsp; "
            f"Year: {esc(item.get('year', '—'))}<br/>"
            f"Observed change: "
            f"{f'{change:+.2f}' if change is not None else 'n/a'}<br/>"
            f"Interpretation: this represents a material movement in the "
            f"reported metric for the stated period. The data suggests "
            f"further review of the underlying drivers."
        )
        story.append(callout(body, styles, tone=tone))
        story.append(spacer(2))
    return story


def _benchmark_section(ctx, styles):
    story = section_heading("9. Peer Benchmark", styles)
    story.append(Paragraph("Company vs Category Peer Benchmark", styles["h2"]))
    story.append(Paragraph(
        "Peers are companies sharing the dataset 'category' field for the "
        "selected year. This is a within-dataset comparison, not an "
        "external market or industry benchmark.", styles["body"]))
    bench = ctx.get("benchmark", {})
    cur = ctx["currency"]

    if not bench.get("available") or bench.get("peer_count", 0) == 0:
        story.append(Paragraph(
            "No comparable peer group is available for the selected "
            "company and year.", styles["body"]))
        return story

    row = bench.get("row", {})
    story.append(Paragraph(
        f"Category: {esc(bench.get('category', ''))} &nbsp;|&nbsp; "
        f"Peer companies: {bench.get('peer_count', 0)} &nbsp;|&nbsp; "
        f"Benchmark year: {bench.get('year', '')}", styles["body"]))

    table_rows, scale, ratios_pct, ratios_x = [], {}, {}, {}
    for label, company_key, peer_key, kind in [
        ("Revenue", "revenue_company", "revenue_peer_avg", "money"),
        ("Net Income", "net_income_company", "net_income_peer_avg", "money"),
        ("Net Margin", "net_profit_margin_company",
         "net_profit_margin_peer_avg", "pct"),
        ("Current Ratio", "current_ratio_company",
         "current_ratio_peer_avg", "x"),
        ("Debt / Equity", "debt_equity_ratio_company",
         "debt_equity_ratio_peer_avg", "x"),
        ("ROA", "roa_company", "roa_peer_avg", "pct"),
        ("ROE", "roe_company", "roe_peer_avg", "pct"),
    ]:
        company_v = _num(row.get(company_key))
        peer_v = _num(row.get(peer_key))
        if company_v is None and peer_v is None:
            continue
        if kind == "money":
            disp = (fmt_money(company_v, cur), fmt_money(peer_v, cur))
        elif kind == "pct":
            disp = (fmt_pct(company_v), fmt_pct(peer_v))
        else:
            disp = (fmt_ratio(company_v), fmt_ratio(peer_v))
        if company_v is not None and peer_v not in (None, 0):
            diff = company_v - peer_v
            pct = diff / abs(peer_v) * 100 if peer_v != 0 else None
            diff_s = (f"{diff:+.2f}" if kind != "money"
                      else fmt_money(diff, cur))
            pct_s = f"{pct:+.2f}%" if pct is not None else "n/a"
        else:
            diff_s, pct_s = "n/a", "n/a"
        table_rows.append([label, disp[0], disp[1], diff_s, pct_s])
        if kind == "money" and company_v is not None \
                and peer_v is not None:
            scale[label] = (company_v, peer_v)
        if kind == "pct" and company_v is not None \
                and peer_v is not None:
            ratios_pct[label] = (company_v, peer_v)
        if kind == "x" and company_v is not None and peer_v is not None:
            ratios_x[label] = (company_v, peer_v)

    story.append(styled_table(
        ["Metric", "Company", "Peer Average", "Difference", "% Difference"],
        table_rows, styles))

    if scale:
        for label, (company_v, peer_v) in scale.items():
            story.append(Paragraph(f"{label} vs Peer Average", styles["h2"]))
            story.append(_img(_charts.grouped_bar(
                [label], [company_v], [peer_v], money_unit(cur),
                f"{label} vs Peer Average", company_name=ctx["company"])))
            story.append(_caption(
                f"Company versus category peer average in "
                f"{money_unit(cur)}.", styles))
    if ratios_pct:
        labels = list(ratios_pct)
        story.append(Paragraph("Ratio Comparison (percent metrics)",
                               styles["h2"]))
        story.append(_img(_charts.grouped_bar(
            labels, [ratios_pct[k][0] for k in labels],
            [ratios_pct[k][1] for k in labels], "Percent (%)",
            "Ratio Comparison — Percent Metrics",
            company_name=ctx["company"], fmt="{:.1f}")))
        story.append(_caption("Percentage-based ratios.", styles))
    if ratios_x:
        labels = list(ratios_x)
        story.append(Paragraph("Ratio Comparison (multiple metrics)",
                               styles["h2"]))
        story.append(_img(_charts.grouped_bar(
            labels, [ratios_x[k][0] for k in labels],
            [ratios_x[k][1] for k in labels], "Multiple (x)",
            "Ratio Comparison — Multiple Metrics",
            company_name=ctx["company"], fmt="{:.2f}")))
        story.append(_caption("Coverage and leverage multiples.", styles))

    story.append(Paragraph("Benchmark Interpretation", styles["h2"]))
    for sentence in build_benchmark_interpretation(ctx):
        story.append(Paragraph(esc(sentence), styles["bullet"],
                               bulletText="•"))
    return story


def _simulation_section(ctx, styles):
    story = section_heading("10. Simulation & Scenario Analysis", styles)
    computed = ctx.get("scenario_computed")
    cur = ctx["currency"]

    story.append(Paragraph(
        "Illustrative what-if analysis. Scenario figures are calculated "
        "from the selected assumptions and are not historical data or a "
        "forecast.", styles["body"]))

    if not computed:
        story.append(Paragraph(
            "Scenario inputs are unavailable for the selected period.",
            styles["body"]))
        return story

    inputs = computed["inputs"]
    current = computed["current"]
    projected = computed["projected"]
    story.append(Paragraph("Input Assumptions (Scenario)", styles["h2"]))
    story.append(styled_table(
        ["Assumption", "Value"],
        [["Revenue growth", f"{inputs['revenue_growth']:+.2f}%"],
         ["Target net profit margin", f"{inputs['target_margin']:.2f}%"]],
        styles))
    story.append(Paragraph("Calculated Results (Illustrative Scenario)",
                           styles["h2"]))
    story.append(styled_table(
        ["Metric", "Current (Base Case)", "Scenario"],
        [["Revenue", fmt_money(current["revenue"], cur),
          fmt_money(projected["revenue"], cur)],
         ["Net Income", fmt_money(current["net_income"], cur),
          fmt_money(projected["net_income"], cur)],
         ["Net Profit Margin", fmt_pct(current["margin"]),
          fmt_pct(projected["margin"])]],
        styles))
    story.append(callout(
        "Scenario Note: this is a what-if analysis based on the selected "
        "assumptions, not a financial forecast.", styles, tone="warn"))
    return story


def _ai_review_section(ctx, styles):
    story = section_heading("11. FinSight AI Financial Review", styles)
    story.append(Paragraph(
        "Structured AI interpretation grounded in the computed evidence. "
        "Metric values come from deterministic calculations; the AI "
        "narrates them.", styles["body"]))
    for block in build_ai_review_sections(ctx):
        story.append(Paragraph(esc(block["title"]), styles["h2"]))
        evidence = block.get("evidence", [])
        if evidence:
            story.append(styled_table(
                ["Evidence Metric", "Value", "Period"],
                [[esc(m), esc(v), esc(p)] for m, v, p in evidence],
                styles))
        story.append(Paragraph("<b>AI Interpretation</b>", styles["body"]))
        for para in _paras(block.get("interpretation", ""), styles["body"]):
            story.append(para)
        story.append(Paragraph(
            f"<b>Investigation Priority: {esc(block.get('priority', '—'))}"
            f"</b>", styles["body"]))
    return story


def _chat_section(ctx, styles):
    story = section_heading("12. FinSight AI Chat Transcript", styles)
    story.append(Paragraph(
        f"Company: {esc(ctx['company'])} &nbsp;|&nbsp; Category: "
        f"{esc(ctx.get('category', ''))} &nbsp;|&nbsp; Selected Year: "
        f"{ctx['selected_year']} &nbsp;|&nbsp; Source: "
        f"{esc(ctx.get('data_source', ''))}", styles["body"]))
    history = ctx.get("chat_history", []) or []
    if not history:
        story.append(Paragraph(
            "No FinSight AI chat conversation was recorded for this report.",
            styles["body"]))
        return story

    for index, turn in enumerate(history, 1):
        question = turn.get("question", "")
        result = turn.get("result", {}) or {}
        story.append(chat_message(
            "USER", esc(question), f"Message {index}", styles))
        answer = result.get("answer", "")
        ai = result.get("ai_explanation") if isinstance(
            result.get("ai_explanation"), dict) else None
        ai_text = ""
        ai_meta = ""
        if ai and ai.get("status") == "SUCCESS":
            ai_text = str(ai.get("response", ""))
            ai_meta = (f"{ai.get('provider', 'ai')}"
                       + (f":{ai.get('model')}" if ai.get("model") else ""))
        full = (esc(answer) + "<br/><br/>" + esc(ai_text)).strip("<br/>")
        if not full:
            full = "No response was recorded."
        story.append(chat_message("FINSIGHT AI", full, ai_meta, styles))

        data = result.get("data")
        try:
            if data is not None and not data.empty:
                frame = data.head(15)
                story.append(styled_table(
                    [str(c) for c in frame.columns],
                    [[esc(v) for v in row] for row in
                     frame.astype(str).values.tolist()],
                    styles))
                if len(data) > 15:
                    story.append(Paragraph(
                        "Additional rows are available in the source "
                        "dataset.", styles["caption"]))
        except Exception:
            pass
        story.append(spacer(3))
    return story


def _methodology_section(ctx, styles):
    story = section_heading("13. Data Quality & Methodology", styles)
    quality = ctx.get("quality", {})
    story.append(Paragraph("Data Quality", styles["h2"]))
    story.append(styled_table(
        ["Check", "Result"],
        [["Quality status", esc(quality.get("status", "UNKNOWN"))],
         ["Missing values", str(quality.get("missing", 0))],
         ["Duplicate records", str(quality.get("duplicates", 0))],
         ["Invalid numeric values", str(quality.get("invalid", 0))]],
        styles))
    story.append(Paragraph("Data Methodology", styles["h2"]))
    for para in [
        "Source dataset: the uploaded CSV/Excel file when provided, "
        "otherwise the bundled financial dataset. All monetary inputs "
        "are USD millions.",
        "Normalization and financial feature mapping translate source "
        "columns to canonical FinSight fields; derived metrics "
        "(margins, year-over-year growth, cash-flow aggregates) are "
        "computed mathematically from provided values only.",
        "Validation checks accounting relationships and data integrity. "
        "Anomaly detection applies deterministic rule thresholds to "
        "year-over-year movements. Risk scoring combines profitability, "
        "growth, liquidity, leverage and cash-flow factors with fixed "
        "weights.",
        "Peer benchmarking compares the company against same-category "
        "companies in the dataset. AI interpretation narrates these "
        "calculated metrics and must not invent values.",
    ]:
        story.append(Paragraph(esc(para), styles["bullet"], bulletText="•"))
    story.append(Paragraph(
        "<b>Calculated metrics</b> are produced by deterministic Python "
        "code. <b>AI-generated interpretation</b> explains those metrics "
        "and is labelled as such throughout this report.", styles["body"]))
    return story


def _disclaimer_section(ctx, styles):
    story = section_heading("14. Disclaimer", styles)
    story.append(Paragraph(
        "This report is generated by FinSight AI for analytical and "
        "informational purposes. It is based on the uploaded dataset and "
        "derived calculations available within the application. "
        "AI-generated interpretations are intended to assist analysis and "
        "should not be treated as investment, accounting, legal, or "
        "financial advice. Past performance is not indicative of future "
        "results. Verify material figures against primary filings before "
        "making decisions.", styles["body"]))
    return story


# -- generator --------------------------------------------------------

TOC_ENTRIES = [
    "1. Executive Summary",
    "2. Financial Overview",
    "3. Performance Analysis",
    "4. Profitability Analysis",
    "5. Cash Flow Analysis",
    "6. Balance Sheet & Financial Position",
    "7. Financial Investigation",
    "8. Risk & Anomaly Analysis",
    "9. Peer Benchmark",
    "10. Simulation & Scenario Analysis",
    "11. FinSight AI Financial Review",
    "12. FinSight AI Chat Transcript",
    "13. Data Quality & Methodology",
    "14. Disclaimer",
]


def generate_financial_report(ctx):
    """Assemble the full PDF. Returns bytes. Never raises for missing
    data — sections degrade to informational notes."""
    styles = get_styles()
    company = ctx["company"]
    period = f"{ctx.get('earliest_year', '')}–{ctx['selected_year']}"

    canvasmaker = ReportCanvas.make(
        header_left="FINSIGHT AI  |  FINANCIAL ANALYSIS REPORT",
        header_right=company,
        footer_left="Generated by FinSight AI",
        footer_mid=f"Report Period: {period}",
    )

    story = []
    story.extend(_cover(ctx, styles))

    story.extend(section_heading("Table of Contents", styles))
    toc = TableOfContents()
    toc.levelStyles = _toc_styles(styles)
    toc.dotsMinLevel = 0
    story.append(toc)
    story.append(spacer(4))
    story.append(Paragraph(
        "Report generated: " + esc(ctx.get("generated", "")) +
        ". Monetary values per selected currency "
        f"({'INR' if ctx['currency'] == 'INR' else 'USD'}).",
        styles["caption"]))

    story.extend(_executive_summary(ctx, styles))
    story.extend(_overview_section(ctx, styles))
    story.extend(_performance_section(ctx, styles))
    story.extend(_profitability_section(ctx, styles))
    story.extend(_cashflow_section(ctx, styles))
    story.extend(_balance_section(ctx, styles))
    story.extend(_investigation_section(ctx, styles))
    story.extend(_risk_section(ctx, styles))
    story.extend(_benchmark_section(ctx, styles))
    story.extend(_simulation_section(ctx, styles))
    story.extend(_ai_review_section(ctx, styles))
    story.extend(_chat_section(ctx, styles))
    story.extend(_methodology_section(ctx, styles))
    story.extend(_disclaimer_section(ctx, styles))

    buffer = BytesIO()
    doc = _build_doc(buffer, canvasmaker)
    doc.multiBuild(story, canvasmaker=canvasmaker)
    buffer.seek(0)
    return buffer.read()

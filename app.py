from engine.validation import run_validation

import streamlit as st
from engine.universal_loader import (
    load_raw_file,
    normalize_financial_data,
    build_data_quality_report,
)

from engine.feature_intelligence import (
    build_feature_analysis,
)

from engine.data_quality import analyze_data_quality

import pandas as pd
import altair as alt

from engine.data_loader import (
    load_application_data,
    get_companies,
    get_years
)

from engine.dashboard import (
    get_company_snapshot,
    get_company_trend
)

from engine.financial_engine import calculate_financial_metrics
from engine.risk import build_health_risk_scores
from engine.investigation_report import generate_investigation_report
from engine.financial_map import build_financial_map
from engine.scenarios import compare_scenarios, run_scenario
from engine.anomaly import detect_anomalies
from engine.column_mapper import (
    map_financial_columns,
    derive_financial_columns
)
from engine.upload_validation import validate_uploaded_data
from engine.benchmark import build_peer_benchmark
from engine.benchmark_visuals import (
    RANK_METRICS,
    TREND_METRICS,
    build_benchmark_variance,
    build_peer_scatter_data,
    build_peer_trend_data,
    peer_position,
    prepare_peer_ranking,
    scorecard_context,
)
from engine.ai_investigation import (
    build_ai_investigation_context,
    extract_ai_signals,
    build_ai_investigation_prompt,
    generate_mock_ai_report
)

from engine.llm import generate_ai_response, get_llm_status



def generate_feature_interpretation(feature_name, value, status, description=""):
    """
    Generate a concise, evidence-grounded interpretation for a feature.
    Does not invent unavailable financial information.
    """

    if status in ["NOT PROVIDED", "UNAVAILABLE", "Missing"]:
        return "This feature cannot be evaluated because the required input data was not provided."

    if value is None:
        return "No usable value is available for this feature."

    try:
        import pandas as pd

        if pd.isna(value):
            return "No usable value is available for this feature."
    except Exception:
        pass

    name = str(feature_name).lower()

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return f"Available value: {value}"

    if "margin" in name:
        if numeric_value < 0:
            return "The calculated margin is negative, indicating that the related profit measure is below zero."
        elif numeric_value < 5:
            return "The calculated margin is positive but relatively low."
        elif numeric_value < 15:
            return "The calculated margin indicates a moderate profitability level."
        else:
            return "The calculated margin indicates a relatively strong profitability level."

    if "growth" in name or "yoy" in name:
        if numeric_value < -10:
            return "The metric shows a significant year-over-year decline."
        elif numeric_value < 0:
            return "The metric declined compared with the previous period."
        elif numeric_value == 0:
            return "The metric is broadly unchanged compared with the previous period."
        elif numeric_value < 10:
            return "The metric shows positive year-over-year growth."
        else:
            return "The metric shows strong year-over-year growth."

    if "current ratio" in name:
        if numeric_value < 1:
            return "The current ratio is below 1x, indicating that current liabilities exceed current assets."
        elif numeric_value < 2:
            return "The current ratio indicates moderate short-term liquidity."
        else:
            return "The current ratio indicates a relatively strong short-term liquidity position."

    if "debt" in name or "leverage" in name:
        if numeric_value < 0:
            return "The calculated leverage value is negative and should be reviewed for balance-sheet interpretation."
        elif numeric_value > 2:
            return "The metric indicates relatively high leverage and warrants review."
        elif numeric_value > 1:
            return "The metric indicates moderate leverage."
        else:
            return "The metric indicates relatively lower leverage."

    if "cash flow" in name or "free cash" in name:
        if numeric_value < 0:
            return "The metric is negative, indicating a cash-flow pressure point."
        else:
            return "The metric is positive based on the available financial data."

    if "return on" in name or name in ["roe", "roa", "roi"]:
        if numeric_value < 0:
            return "The return metric is negative, indicating negative returns for the measured period."
        elif numeric_value < 10:
            return "The return metric is positive but relatively modest."
        else:
            return "The return metric indicates a relatively strong reported return."

    if "profit" in name or "income" in name or "ebitda" in name:
        if numeric_value < 0:
            return "The metric is negative for the selected period and should be reviewed."
        else:
            return "The metric is positive for the selected period."

    if "revenue" in name or "sales" in name:
        if numeric_value < 0:
            return "The reported value is negative and should be validated against the source data."
        else:
            return "The reported revenue-related value is available for analysis."

    if "employee" in name or "employees" in name:
        if numeric_value > 0:
            return "The workforce value is available and can be used for scale and productivity analysis."

    return "The feature is available and can be used as part of the financial analysis."


def render_feature_intelligence(uploaded_file):
    """
    Analyze uploaded financial data without fabricating
    unavailable financial values.
    """

    st.subheader(" Financial Feature Intelligence")

    try:
        raw_df, file_info = load_raw_file(uploaded_file)
    except Exception as e:
        st.error(f"Unable to read uploaded file: {e}")
        return None

    st.write(
        f"**File type:** {file_info.get('file_type', 'Unknown')}"
    )

    if file_info.get("selected_sheet"):
        st.write(
            f"**Analyzed sheet:** "
            f"{file_info.get('selected_sheet')}"
        )

    try:
        normalized_result = normalize_financial_data(raw_df)
    except Exception as e:
        st.error(f"Data normalization failed: {e}")
        return None

    if normalized_result.get("status") != "VALID":

        st.error(
            "The uploaded file does not contain all "
            "required financial fields."
        )

        missing = normalized_result.get(
            "missing_required",
            []
        )

        if missing:
            st.write("### Missing Required Fields")

            for field in missing:
                st.write(f"- `{field}`")

        optional = normalized_result.get(
            "optional_available",
            []
        )

        if optional:
            st.write("### Optional Fields Detected")

            for field in optional:
                st.write(f"- `{field}`")

        extra = normalized_result.get(
            "extra_columns",
            []
        )

        if extra:
            st.write("### Additional Columns")

            for column in extra:
                st.write(f"- `{column}`")

        return normalized_result

    normalized_df = normalized_result["data"]

    quality = build_data_quality_report(
        normalized_result
    )

    analysis = build_feature_analysis(
        normalized_df
    )

    discovery = analysis["discovery"]

    # -----------------------------------------------------
    # Summary metrics
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            analysis["discovery"].get(
                "row_count",
                len(normalized_df)
            )
        )

    with col2:
        st.metric(
            "Available Features",
            analysis["available_feature_count"]
        )

    with col3:
        st.metric(
            "Derived Metrics",
            analysis["derived_metric_count"]
        )

    with col4:
        st.metric(
            "Extra Columns",
            analysis["extra_column_count"]
        )

    # -----------------------------------------------------
    # Feature groups
    # -----------------------------------------------------

    st.markdown("###  Detected Financial Areas")

    groups = discovery.get(
        "groups",
        {}
    )

    if groups:

        for group, features in groups.items():

            with st.expander(
                f"{group} ({len(features)} features)"
            ):

                for feature in features:
                    info = discovery["mapped"].get(
                        feature,
                        {}
                    )

                    source = info.get(
                        "source_column",
                        feature
                    )

                    confidence = info.get(
                        "confidence",
                        "-"
                    )

                    st.write(
                        f"**{feature}** ← `{source}` "
                        f"(confidence: {confidence})"
                    )

    # -----------------------------------------------------
    # Feature mapping table
    # -----------------------------------------------------

    st.markdown("###  Feature Mapping")

    mapping_rows = []

    for feature, info in discovery.get(
        "mapped",
        {}
    ).items():

        mapping_rows.append({
            "FinSight Feature": feature,
            "Source Column": info.get(
                "source_column"
            ),
            "Category": info.get(
                "group"
            ),
            "Type": info.get(
                "type"
            ),
            "Confidence": info.get(
                "confidence"
            ),
            "Status": "Available",
        })

    if mapping_rows:

        st.dataframe(
            mapping_rows,
            use_container_width=True,
            hide_index=True
        )

    # -----------------------------------------------------
    # Derived metrics
    # -----------------------------------------------------

    derived = [
        column
        for column in analysis["data"].columns
        if column.endswith("_calculated")
    ]

    if derived:

        st.markdown("###  Legitimately Derived Metrics")

        st.info(
            "These metrics are calculated only from "
            "financial fields actually provided in "
            "the uploaded dataset."
        )

        for column in derived:

            display_name = (
                column
                .replace("_calculated", "")
                .replace("_", " ")
                .title()
            )

            st.write(
                f" **{display_name}**"
            )

    # -----------------------------------------------------
    # Extra columns
    # -----------------------------------------------------

    extra_columns = discovery.get(
        "unmapped",
        []
    )

    st.markdown("###  Additional Dataset Features")

    if extra_columns:

        st.info(
            "These columns were found in the uploaded "
            "dataset but are not currently mapped to "
            "a FinSight financial concept. Their values "
            "are preserved and are not replaced with "
            "fabricated values."
        )

        for column in extra_columns:
            st.write(
                f"- `{column}`"
            )

    else:

        st.success(
            "All uploaded columns were understood "
            "by the current feature catalog."
        )

    # -----------------------------------------------------
    # Dataset preview
    # -----------------------------------------------------

    st.markdown("###  Normalized Dataset Preview")

    st.dataframe(
        normalized_df.head(20),
        use_container_width=True,
        hide_index=True
    )

    return {
        "raw_data": raw_df,
        "normalized": normalized_result,
        "feature_analysis": analysis,
        "quality": quality,
    }




# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FinSight AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap');

    :root {
        --fs-cream: #FEFAF3;
        --fs-cream-2: #F1EADD;
        --fs-card: #FFFFFF;
        --fs-ink: #2F2C25;
        --fs-muted: rgba(47, 44, 37, 0.62);
        --fs-faint: rgba(47, 44, 37, 0.42);
        --fs-line: #E7DFCF;
        --fs-accent: #2F6FED;
        --fs-green: #3E7C4F;
        --fs-amber: #A97A1F;
        --fs-red: #B0483E;
        --fs-serif: 'Instrument Serif', Georgia, 'Times New Roman', serif;
        --fs-sans: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
        --fs-mono: 'Roboto Mono', ui-monospace, SFMono-Regular, monospace;
    }

    /* ---------- base ---------- */
    [data-testid="stAppViewContainer"] {
        background: var(--fs-cream);
        color: var(--fs-ink);
        font-family: var(--fs-sans);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }
    p, li, div {
        color: var(--fs-ink);
    }
    a {
        color: var(--fs-accent);
    }

    /* ---------- headings ---------- */
    h1 {
        font-family: var(--fs-serif) !important;
        font-weight: 400 !important;
        letter-spacing: -0.01em;
        color: var(--fs-ink) !important;
    }
    h2, h3 {
        font-family: var(--fs-sans) !important;
        color: var(--fs-ink) !important;
    }

    /* ---------- sticky global header (stable position) ---------- */
    .fs-global-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: var(--fs-cream);
    }
    .fs-global-header .fs-brandbar {
        margin-bottom: 0;
        height: 92px;
        overflow: hidden;
        align-items: center;
        box-sizing: border-box;
    }
    /* tab row pins directly under the fixed-height brandbar */
    [data-testid="stTabs"] > div:first-child {
        border-bottom: 1px solid var(--fs-line);
        padding-bottom: 8px;
        background: var(--fs-cream);
        min-height: 52px;
        position: sticky;
        top: 92px;
        z-index: 998;
    }
    /* fixed tab-button height: active pill never shifts layout */
    button[data-testid="stTab"] {
        min-height: 38px;
        line-height: 22px;
    }
    /* ---------- brand bar + heroes (visual only) ---------- */
    .fs-brandbar {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 16px;
        flex-wrap: wrap;
        padding: 6px 2px 14px 2px;
        border-bottom: 1px solid var(--fs-line);
        margin-bottom: 6px;
    }
    .fs-brand {
        font-family: var(--fs-serif);
        font-size: 34px;
        letter-spacing: 0.12em;
        color: var(--fs-ink);
        white-space: nowrap;
    }
    .fs-brand-sub {
        font-family: var(--fs-mono);
        font-size: 10.5px;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--fs-faint);
        margin-top: 2px;
    }
    .fs-context {
        font-family: var(--fs-mono);
        font-size: 11px;
        letter-spacing: 0.08em;
        color: var(--fs-muted);
        text-align: right;
    }
    .fs-eyebrow {
        font-family: var(--fs-mono);
        font-size: 11px;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--fs-faint);
        margin-top: 26px;
        margin-bottom: 2px;
    }
    .fs-hero {
        font-family: var(--fs-serif);
        font-size: clamp(36px, 4.5vw, 52px);
        line-height: 1.08;
        letter-spacing: -0.01em;
        color: var(--fs-ink);
        margin-bottom: 6px;
    }
    .fs-lede {
        font-size: 15px;
        line-height: 1.55;
        color: var(--fs-muted);
        max-width: 720px;
        margin-bottom: 18px;
    }

    /* legacy title classes kept for compatibility */
    .main-title {
        font-family: var(--fs-serif);
        font-size: 34px;
        letter-spacing: -0.01em;
        margin-bottom: 2px;
        color: var(--fs-ink);
    }
    .sub-title {
        font-size: 14px;
        color: var(--fs-muted);
        margin-bottom: 18px;
    }
    .section-title {
        font-family: var(--fs-serif);
        font-size: 26px;
        color: var(--fs-ink);
        margin-top: 30px;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--fs-line);
    }

    /* ---------- sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #F6F0E1;
        border-right: 1px solid var(--fs-line);
    }
    [data-testid="stSidebar"] .fs-side-brand {
        font-family: var(--fs-serif);
        font-size: 24px;
        letter-spacing: 0.1em;
        color: var(--fs-ink);
    }
    [data-testid="stSidebar"] .fs-side-sub {
        font-family: var(--fs-mono);
        font-size: 10px;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--fs-faint);
        margin-bottom: 4px;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--fs-line);
        margin: 14px 0;
    }
    [data-testid="stSidebar"] h3 {
        font-size: 12px !important;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--fs-muted) !important;
    }

    /* ---------- top tab navigation ---------- */
    [data-testid="stTabs"] > div:first-child {
        border-bottom: 1px solid var(--fs-line);
        padding-bottom: 2px;
    }
    button[data-testid="stTab"] {
        font-family: var(--fs-sans);
        font-size: 13.5px;
        font-weight: 500;
        color: var(--fs-muted);
        border-radius: 999px !important;
        padding: 7px 16px !important;
        margin-right: 2px;
    }
    button[data-testid="stTab"]:hover {
        color: var(--fs-ink);
        background: rgba(47, 44, 37, 0.05);
    }
    button[data-testid="stTab"][aria-selected="true"] {
        background: var(--fs-ink) !important;
        color: var(--fs-cream) !important;
    }
    [data-testid="stTabPanel"] {
        animation: fs-rise 0.45s ease both;
    }

    /* ---------- KPI / metric cards ---------- */
    div[data-testid="stMetric"] {
        background: var(--fs-card);
        border: 1px solid var(--fs-line);
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(47, 44, 37, 0.04);
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 14px rgba(47, 44, 37, 0.07);
    }
    div[data-testid="stMetricLabel"] {
        font-family: var(--fs-sans);
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--fs-muted);
    }
    div[data-testid="stMetricValue"] {
        font-family: var(--fs-serif);
        font-size: 30px;
        color: var(--fs-ink);
    }

    /* ---------- buttons ---------- */
    [data-testid="stBaseButton-primary"] {
        background: var(--fs-ink) !important;
        color: var(--fs-cream) !important;
        border: 1px solid var(--fs-ink) !important;
        border-radius: 10px !important;
        font-weight: 600;
        padding: 8px 22px !important;
    }
    [data-testid="stBaseButton-primary"]:hover {
        opacity: 0.88;
        transform: translateY(-1px);
    }
    [data-testid="stBaseButton-secondary"] {
        background: #EFE7D2 !important;
        color: var(--fs-ink) !important;
        border: 1px solid var(--fs-line) !important;
        border-radius: 10px !important;
        font-weight: 600;
    }
    [data-testid="stBaseButton-secondary"]:hover {
        opacity: 0.85;
        transform: translateY(-1px);
    }

    /* ---------- inputs ---------- */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background: #FFFDF8 !important;
        border: 1px solid var(--fs-line) !important;
        border-radius: 10px !important;
        color: var(--fs-ink) !important;
    }
    [data-testid="stFileUploader"] section {
        background: #FFFDF8;
        border: 1.5px dashed #CFC3A6 !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] {
        background: var(--fs-card);
        border: 1px solid var(--fs-line) !important;
        border-radius: 12px !important;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600;
        color: var(--fs-ink);
    }

    /* ---------- alerts as analyst notes ---------- */
    [data-testid="stSuccess"] {
        background: #EDF4EC !important;
        border: 1px solid #CBE0CE !important;
        border-left: 4px solid var(--fs-green) !important;
        border-radius: 10px !important;
        color: var(--fs-ink) !important;
    }
    [data-testid="stInfo"] {
        background: #EBF1FD !important;
        border: 1px solid #C9D9FA !important;
        border-left: 4px solid var(--fs-accent) !important;
        border-radius: 10px !important;
        color: var(--fs-ink) !important;
    }
    [data-testid="stWarning"] {
        background: #FAF3E2 !important;
        border: 1px solid #EAD9AE !important;
        border-left: 4px solid var(--fs-amber) !important;
        border-radius: 10px !important;
        color: var(--fs-ink) !important;
    }
    [data-testid="stError"] {
        background: #F9E9E7 !important;
        border: 1px solid #ECC5C0 !important;
        border-left: 4px solid var(--fs-red) !important;
        border-radius: 10px !important;
        color: var(--fs-ink) !important;
    }

    /* ---------- research tables ---------- */
    div[data-testid="stDataFrame"] {
        background: var(--fs-card);
        border: 1px solid var(--fs-line);
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- chat bubbles (wrappers only; content unchanged) ---------- */
    /* user: right-aligned bubble */
    [data-testid="stMarkdownContainer"]:has(> div > p > .fs-role-you) {
        background: #E9E2D0;
        border: 1px solid var(--fs-line);
        border-radius: 16px 16px 4px 16px;
        padding: 10px 16px;
        margin: 14px 0 6px auto;
        max-width: 78%;
        width: fit-content;
    }
    /* assistant: plain full-width message, no card */
    [data-testid="stMarkdownContainer"]:has(> div > p > .fs-role-ai) {
        background: transparent;
        border: none;
        box-shadow: none;
        border-radius: 0;
        padding: 6px 2px;
        margin: 10px 0 4px 0;
        max-width: 100%;
    }
    .fs-role {
        font-family: var(--fs-mono);
        font-size: 10px;
        letter-spacing: 0.2em;
        color: var(--fs-faint);
        margin-bottom: 4px;
    }
    .fs-ai > .fs-role {
        color: var(--fs-accent);
    }

    /* ---------- captions / metadata ---------- */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--fs-muted) !important;
    }

    /* ---------- motion (subtle, opt-out aware) ---------- */
    @keyframes fs-rise {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @media (prefers-reduced-motion: reduce) {
        * {
            animation: none !important;
            transition: none !important;
        }
    }

    /* ---------- AI chat page (scoped, frontend only) ---------- */
    .fs-chat-hero {
        text-align: center;
        max-width: 760px;
        margin: 34px auto 8px auto;
    }
    .fs-chat-hero .fs-eyebrow,
    .fs-chat-hero .fs-hero,
    .fs-chat-hero .fs-lede {
        text-align: center;
    }
    .fs-chat-hero .fs-hero {
        font-size: clamp(44px, 6vw, 68px);
    }
    .fs-chat-hero .fs-lede {
        margin-left: auto;
        margin-right: auto;
    }
    /* focused central column for the chat tab only */
    div[data-testid="stTabPanel"]:has(.fs-chat-hero)
    > div > div > div[data-testid="stVerticalBlock"] {
        max-width: 1000px;
        margin-left: auto;
        margin-right: auto;
    }
    /* large rounded prompt input */
    div[data-testid="stTabPanel"]:has(.fs-chat-hero)
    [data-testid="stTextInput"] input {
        min-height: 64px;
        border-radius: 20px !important;
        padding: 16px 20px !important;
        font-size: 16px !important;
        box-shadow: 0 2px 10px rgba(47, 44, 37, 0.06);
    }
    div[data-testid="stTabPanel"]:has(.fs-chat-hero)
    [data-testid="stTextInput"] input::placeholder {
        color: rgba(47, 44, 37, 0.42);
    }
    /* ---------- composer pill bar (single row) ---------- */
    div[data-testid="stElementContainer"]:has(.fs-composer-bar)
    + div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"]) {
        background: var(--fs-card);
        border: 1px solid var(--fs-line);
        border-radius: 28px;
        padding: 10px 14px;
        box-shadow: 0 3px 14px rgba(47, 44, 37, 0.07);
        position: sticky;
        bottom: 12px;
        z-index: 10;
    }
    div[data-testid="stElementContainer"]:has(.fs-composer-bar)
    + div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"])
    div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    div[data-testid="stElementContainer"]:has(.fs-composer-bar)
    + div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"])
    div[data-testid="column"] {
        min-width: 0;
    }
    /* borderless controls inside the pill */
    div[data-testid="stElementContainer"]:has(.fs-composer-bar)
    + div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"])
    input {
        border: none !important;
        box-shadow: none !important;
        background: transparent !important;
        min-height: 48px;
        padding: 4px 2px !important;
        font-size: 16px !important;
    }
    div[data-testid="stElementContainer"]:has(.fs-composer-bar)
    + div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"])
    input::placeholder {
        color: rgba(47, 44, 37, 0.42);
    }
    div[data-testid="stElementContainer"]:has(.fs-composer-bar)
    + div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"])
    [data-baseweb="select"] > div {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        min-height: 48px;
    }
    /* clear (+) ghost button */
    div[data-testid="stElementContainer"]:has(.fs-composer-bar)
    + div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"])
    div[data-testid="column"]:first-child button[data-testid^="stBaseButton"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: var(--fs-ink) !important;
        font-size: 24px !important;
        font-weight: 400;
        width: 44px !important;
        height: 48px !important;
        min-height: 48px !important;
        padding: 0 !important;
    }
    div[data-testid="stElementContainer"]:has(.fs-composer-bar)
    + div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"])
    div[data-testid="column"]:first-child button[data-testid^="stBaseButton"]:hover {
        transform: none;
        box-shadow: none;
        opacity: 0.6;
    }
    div[data-testid="stElementContainer"]:has(.fs-composer-bar)
    + div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"])
    div[data-testid="column"]:first-child button[data-testid^="stBaseButton"] p {
        color: var(--fs-ink) !important;
        opacity: 1 !important;
    }
    /* circular send button */
    div[data-testid="stElementContainer"]:has(.fs-composer-bar)
    + div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"])
    div[data-testid="column"]:last-child button[data-testid^="stBaseButton"] {
        width: 48px !important;
        height: 48px !important;
        min-height: 48px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        background: #8C8C8C !important;
        color: #FFFFFF !important;
        border: none !important;
        font-size: 20px !important;
        line-height: 1;
        float: right;
    }
    div[data-testid="stElementContainer"]:has(.fs-composer-bar)
    + div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"])
    div[data-testid="column"]:last-child button[data-testid^="stBaseButton"]:hover {
        background: var(--fs-ink) !important;
        transform: none;
        box-shadow: none;
        opacity: 1;
    }
    div[data-testid="stElementContainer"]:has(.fs-composer-bar)
    + div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"])
    div[data-testid="column"]:last-child button[data-testid^="stBaseButton"] p {
        color: #FFFFFF !important;
        opacity: 1 !important;
    }
    /* wrap chat columns on narrow screens */
    div[data-testid="stTabPanel"]:has(.fs-chat-hero)
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
    }
    div[data-testid="stTabPanel"]:has(.fs-chat-hero)
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        min-width: 220px;
        flex: 1 1 220px;
    }
    /* button text always visible (report + model-check actions).
       The global `p` ink rule would otherwise paint dark text on the
       dark primary button — descendants get explicit colors. */
    button[data-testid^="stBaseButton"],
    button[data-testid^="stBaseButton"] p,
    button[data-testid^="stBaseButton"] div {
        opacity: 1 !important;
    }
    button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-primary"] p,
    button[data-testid="stBaseButton-primary"] div,
    button[data-testid="stBaseButton-primary"] span {
        color: #FEFAF3 !important;
    }
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="stBaseButton-secondary"] p,
    button[data-testid="stBaseButton-secondary"] div,
    button[data-testid="stBaseButton-secondary"] span {
        color: #2F2C25 !important;
    }
    button[data-testid^="stBaseButton"] {
        min-height: 42px;
    }
    /* link buttons match primary actions */
    a[data-testid="stLinkButton"] {
        background: var(--fs-ink) !important;
        border: 1px solid var(--fs-ink) !important;
        border-radius: 10px !important;
        padding: 10px 26px !important;
        font-weight: 600;
    }
    a[data-testid="stLinkButton"] p,
    a[data-testid="stLinkButton"] div,
    a[data-testid="stLinkButton"] span {
        color: #FEFAF3 !important;
        opacity: 1 !important;
    }
    /* anchor target clears the sticky header */
    #finsight-ai-chat {
        scroll-margin-top: 170px;
    }

    /* ---------- responsive ---------- */
    @media (max-width: 768px) {
        .fs-brandbar {
            flex-direction: column;
            align-items: flex-start;
        }
        .fs-context {
            text-align: left;
        }
        div[data-testid="stMetricValue"] {
            font-size: 24px;
        }
        /* small screens: natural flow (no overlap math), nav scrolls */
        .fs-global-header {
            position: static;
        }
        .fs-global-header .fs-brandbar {
            height: auto;
            overflow: visible;
        }
        [data-testid="stTabs"] > div:first-child {
            position: static;
        }
        /* composer controls stack full-width on small screens */
        div[data-testid="stElementContainer"]:has(.fs-composer-bar)
        + div[data-testid="stElementContainer"]:has([data-testid="stHorizontalBlock"])
        div[data-testid="column"] {
            min-width: 200px;
            flex: 1 1 200px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# GLOBAL HEADER + NAVIGATION (rendered first, stable position)
# Sticky wrapper keeps brand + tabs pinned; company/year read from
# selector state with a neutral fallback before any selection.
# ============================================================

_hdr_company = st.session_state.get("finsight_company")
_hdr_year = st.session_state.get("finsight_year")
_hdr_context = (
    f"{_hdr_company} &nbsp;|&nbsp; FY {_hdr_year}"
    if _hdr_company and _hdr_year
    else "No dataset loaded"
)

st.markdown('<div class="fs-global-header">', unsafe_allow_html=True)

st.markdown(
    '<div class="fs-brandbar">'
    '<div><div class="fs-brand">FINSIGHT AI</div>'
    '<div class="fs-brand-sub">Financial Intelligence &amp; Investigation Agent</div></div>'
    f'<div class="fs-context">Financial Statement Review Agent<br>{_hdr_context}</div>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)

tab_overview, tab_investigate, tab_compare, tab_benchmark, tab_simulate, tab_report, tab_chat = st.tabs([
    "Overview",
    "Investigate",
    "Compare",
    "Benchmark",
    "Simulate",
    "Report",
    "AI CHAT"
])


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return load_application_data(
        "data/clean_data.csv"
    )


# ============================================================
# SIDEBAR - DATA SOURCE
# ============================================================

with st.sidebar:
    st.markdown(
        '<div class="fs-side-brand">FINSIGHT AI</div>'
        '<div class="fs-side-sub">Financial intelligence workspace</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Financial Intelligence & Investigation Agent"
    )

    st.divider()

    st.subheader("Data Source")

    uploaded_file = st.file_uploader(
        "Upload Financial Data",
        type=["csv", "xlsx", "xls"],
        help="Upload a financial CSV or Excel workbook for analysis."
    )

    with st.expander("AI provider keys", expanded=False):
        st.caption(
            "Add a key to enable live AI answers. Keys are kept in "
            "memory for this session only. "
            "Gemini: https://aistudio.google.com/apikey — "
            "OpenAI: https://platform.openai.com/api-keys — "
            "Anthropic: https://console.anthropic.com/ — "
            "Grok: https://console.x.ai/"
        )

        gemini_key_input = st.text_input(
            "GEMINI_API_KEY",
            type="password",
            key="finsight_gemini_key",
            placeholder="AIza...",
        )
        openai_key_input = st.text_input(
            "OPENAI_API_KEY",
            type="password",
            key="finsight_openai_key",
            placeholder="sk-...",
        )
        anthropic_key_input = st.text_input(
            "ANTHROPIC_API_KEY",
            type="password",
            key="finsight_anthropic_key",
            placeholder="sk-ant-...",
        )
        grok_key_input = st.text_input(
            "GROK_API_KEY",
            type="password",
            key="finsight_grok_key",
            placeholder="xai-...",
        )

        typed_keys = {
            "gemini": (gemini_key_input or "").strip(),
            "openai": (openai_key_input or "").strip(),
            "anthropic": (anthropic_key_input or "").strip(),
            "grok": (grok_key_input or "").strip(),
        }

        if st.button(
            "Test connections",
            key="finsight_test_llm",
            help="Checks only the keys typed above (one tiny live call each).",
        ):
            from engine.llm import (
                list_provider_models,
                reset_session_keys,
                set_api_keys,
                test_provider_connection,
            )

            # Session keys become exactly what is typed: testing only
            # ever touches the entered keys, never stale/other ones.
            reset_session_keys()
            entered = {
                name: key for name, key in typed_keys.items() if key
            }

            if not entered:
                st.warning("Enter at least one API key above first.")
            else:
                set_api_keys(
                    gemini=entered.get("gemini"),
                    openai=entered.get("openai"),
                    anthropic=entered.get("anthropic"),
                    grok=entered.get("grok"),
                )

                with st.spinner("Testing entered keys..."):
                    for name, key in entered.items():
                        check = test_provider_connection(name)

                        if not check.get("ok"):
                            st.session_state.pop(
                                f"finsight_models_{name}", None
                            )
                            st.error(
                                f"{name}: FAILED — "
                                f"{check.get('message', '')}"
                            )
                            continue

                        models_info = list_provider_models(name)

                        if models_info.get("ok") and models_info.get("models"):
                            models = models_info["models"]
                            st.session_state[f"finsight_models_{name}"] = models

                            current = st.session_state.get(
                                f"finsight_model_{name}"
                            )
                            default_model = check.get("model_used")
                            if current not in models:
                                st.session_state[f"finsight_model_{name}"] = (
                                    default_model
                                    if default_model in models
                                    else models[0]
                                )

                            where = (
                                "live list"
                                if models_info.get("source") == "live"
                                else "known Claude models "
                                "(Anthropic has no list API)"
                            )
                            st.success(
                                f"{name}: connected "
                                f"(model: {check['model_used']}). "
                                f"{len(models)} models via {where}."
                            )
                        else:
                            st.success(
                                f"{name}: connected "
                                f"(model: {check['model_used']}). "
                                f"Model list unavailable: "
                                f"{models_info.get('message', '')}"
                            )


# ============================================================
# DATA SOURCE
# ============================================================

if uploaded_file is not None:
    raw_uploaded_df = None
    feature_result = None

    try:

        # ============================================================
        # UNIVERSAL FILE LOADING
        # ============================================================

        raw_uploaded_df, upload_file_info = load_raw_file(
            uploaded_file
        )

        st.sidebar.success(
            f"Uploaded: {uploaded_file.name}"
        )

        # ============================================================
        # FEATURE INTELLIGENCE
        # ============================================================

        feature_result = build_feature_analysis(
            raw_uploaded_df
        )

        discovery = feature_result.get(
            "discovery",
            {}
        )

        available_features = discovery.get(
            "available",
            []
        )

        extra_columns = discovery.get(
            "unmapped",
            []
        )

        available_count = feature_result.get(
            "available_feature_count",
            len(available_features)
        )

        derived_count = feature_result.get(
            "derived_metric_count",
            0
        )

        # ============================================================
        # FEATURE GROUPS
        # ============================================================

        groups = discovery.get(
            "groups",
            {}
        )

        if groups:

            st.markdown(
                "###  Financial Features Detected"
            )

            for group, features in groups.items():

                with st.expander(
                    f"{group} — {len(features)} features"
                ):

                    for feature in features:

                        info = discovery[
                            "mapped"
                        ].get(
                            feature,
                            {}
                        )

                        source_column = info.get(
                            "source_column",
                            feature
                        )

                        confidence = info.get(
                            "confidence",
                            "-"
                        )

                        st.write(
                            f" **{feature}** "
                            f"← `{source_column}` "
                            f"(confidence: {confidence})"
                        )


        # ============================================================
        # NORMALIZE DATA FOR EXISTING ENGINE
        # ============================================================

        normalized_result = normalize_financial_data(
            raw_uploaded_df
        )

        if normalized_result.get(
            "status"
        ) != "VALID":

            st.warning(
                "⚠️ The uploaded dataset contains financial "
                "information, but some core fields required "
                "by the full FinSight investigation engine "
                "are unavailable."
            )

            missing_required = normalized_result.get(
                "missing_required",
                []
            )

            if missing_required:

                st.markdown(
                    "#### Core fields not available"
                )

                for field in missing_required:
                    st.write(
                        f"• `{field}`"
                    )

            st.info(
                "Available features are still detected above. "
                "Missing financial values are never invented."
            )

            # Keep raw data available to the UI.
            st.session_state[
                "uploaded_raw_data"
            ] = raw_uploaded_df

            st.session_state[
                "feature_analysis_result"
            ] = feature_result

            st.session_state[
                "upload_core_available"
            ] = False

            st.session_state[
                "upload_mapping"
            ] = normalized_result.get("mapping", {})

            st.session_state[
                "upload_missing_fields"
            ] = missing_required

            st.session_state[
                "upload_filename"
            ] = uploaded_file.name

            # No usable data: the gate below asks for a valid file.
            st.info(
                "Upload a file with the core fields above to unlock "
                "the dashboard."
            )

            df = None
            _upload_ok = False

        else:

            _upload_ok = True

            # ============================================================
            # NORMALIZED CORE FINANCIAL DATA
            # ============================================================

            df = normalized_result[
                "data"
            ].copy()

            # ============================================================
            # ADD LEGITIMATE DERIVED FEATURES
            # ============================================================

            enriched_df = feature_result[
                "data"
            ]

            for column in enriched_df.columns:

                if column not in df.columns:

                    if (
                        column.endswith("_calculated")
                        or column.endswith("_yoy")
                    ):

                        df[column] = enriched_df[
                            column
                        ]

            # ============================================================
            # EXISTING UPLOAD VALIDATION
            # ============================================================

            try:

                upload_validation = (
                    validate_uploaded_data(df)
                )

                if not upload_validation[
                    "valid"
                ]:

                    st.error(
                        "Uploaded financial data could "
                        "not pass the core validation checks."
                    )

                    for error in upload_validation.get(
                        "errors",
                        []
                    ):

                        st.error(str(error))

                    for warning in upload_validation.get(
                        "warnings",
                        []
                    ):

                        st.warning(str(warning))

                    # Do NOT stop: keep the dashboard alive on the default
                    # dataset and route the upload into AI Chat.
                    st.session_state[
                        "uploaded_raw_data"
                    ] = raw_uploaded_df

                    st.session_state[
                        "feature_analysis_result"
                    ] = feature_result

                    st.session_state[
                        "upload_core_available"
                    ] = False

                    st.session_state[
                        "upload_mapping"
                    ] = normalized_result.get("mapping", {})

                    st.session_state[
                        "upload_missing_fields"
                    ] = []

                    st.session_state[
                        "upload_filename"
                    ] = uploaded_file.name

                    st.info(
                        "Upload a file that passes the core validation "
                        "checks above to unlock the dashboard."
                    )

                    df = None
                    _upload_ok = False

                for warning in upload_validation.get(
                    "warnings",
                    []
                ):

                    st.warning(str(warning))

            except Exception as validation_error:

                st.warning(
                    "Core validation could not evaluate "
                    "every uploaded field: "
                    f"{validation_error}"
                )

            if _upload_ok:

                # ============================================================
                # STORE UPLOAD RESULTS
                # ============================================================

                st.session_state[
                    "uploaded_raw_data"
                ] = raw_uploaded_df

                st.session_state[
                    "normalized_uploaded_data"
                ] = df

                st.session_state[
                    "feature_analysis_result"
                ] = feature_result

                st.session_state[
                    "upload_core_available"
                ] = True

                st.session_state[
                    "upload_mapping"
                ] = normalized_result.get("mapping", {})

                st.session_state[
                    "upload_missing_fields"
                ] = []

                st.session_state[
                    "upload_filename"
                ] = uploaded_file.name

                # ============================================================
                # FINAL UPLOAD STATUS
                # ============================================================

                st.success(
                    f" FinSight normalized {len(df)} "
                    f"financial records for analysis."
                )

                st.caption(
                    "Financial analysis uses values supplied by the "
                    "uploaded dataset and metrics mathematically "
                    "derived from those values."
                )

    except Exception as e:

        st.error(
            "Unable to safely process the uploaded file."
        )

        st.error(
            str(e)
        )

        st.info(
            "FinSight did not fabricate missing values. "
            "Upload a complete, structured financial CSV or Excel "
            "file to unlock the dashboard."
        )

        # Stash anything readable for AI Chat; keep dashboard alive.
        if raw_uploaded_df is not None:
            st.session_state[
                "uploaded_raw_data"
            ] = raw_uploaded_df

        if feature_result is not None:
            st.session_state[
                "feature_analysis_result"
            ] = feature_result

        st.session_state[
            "upload_core_available"
        ] = False

        st.session_state[
            "upload_mapping"
        ] = {}

        st.session_state[
            "upload_missing_fields"
        ] = []

        if uploaded_file is not None:
            st.session_state[
                "upload_filename"
            ] = uploaded_file.name

        df = None

else:
    df = None


# Upload gate: nothing (dashboard, selectors, chat) renders until a
# dataset has been uploaded AND passed analysis. Use the sidebar
# uploader above to begin.
if df is None:

    # STATE 1 — no dataset: full upload landing.
    if uploaded_file is None:

        st.markdown(
            '<div class="fs-chat-hero">'
            '<div class="fs-eyebrow">FinSight AI</div>'
            '<div class="fs-hero">Upload your financial dataset.</div>'
            '<div class="fs-lede">Nothing is shown until your data is '
            'uploaded and analyzed — upload a CSV or Excel workbook in '
            'the sidebar to unlock the full workspace.</div>'
            '</div>',
            unsafe_allow_html=True
        )

        _gate_l, _gate_c, _gate_r = st.columns([1, 6, 1])

        with _gate_c:
            st.info(
                "How it works: 1) Upload a CSV, XLSX or XLS file with "
                "company financials. 2) FinSight maps the columns and runs "
                "validation automatically. 3) Overview, investigation, "
                "benchmarks, simulation, reports and AI chat unlock."
            )
            st.caption(
                "Required core fields: company, year, revenue, gross "
                "profit, net income, EBITDA, shareholder equity, operating "
                "/ investing / financing cash flow, current ratio, "
                "debt-to-equity. Your file never leaves this session."
            )

    # STATE 3/4 — dataset exists but mapping failed: no landing, no
    # large banner. One subtle notification; details stay in session.
    else:

        st.caption(
            "Financial mapping is unavailable for this dataset. "
            "AI CHAT is still available for exploring the uploaded data."
        )

    # Every financial tab gets an explicit state so no panel is
    # mysteriously blank. AI CHAT points at the working chat below.
    _gate_message = (
        "Upload a dataset to unlock this section."
        if uploaded_file is None
        else "This section needs core financial fields that were not "
             "detected. Open the AI CHAT tab to explore the upload."
    )

    for _gate_tab, _gate_label in [
        (tab_overview, "Overview"),
        (tab_investigate, "Investigate"),
        (tab_compare, "Compare"),
        (tab_benchmark, "Benchmark"),
        (tab_simulate, "Simulate"),
        (tab_report, "Report"),
    ]:
        with _gate_tab:
            st.caption(_gate_message)

    with tab_chat:
        # Intentionally no pointer/button here: AI Chat lives in the
        # fallback chat rendered below, reachable from header navigation.
        pass

    st.markdown('<div id="finsight-ai-chat"></div>', unsafe_allow_html=True)

    from engine.llm import generate_financial_answer, resolve_provider

    st.markdown(
        '<div class="fs-eyebrow">AI Chat</div>'
        '<div class="fs-hero">Ask FinSight AI.</div>'
        '<div class="fs-lede">Ask about your file or your finances — '
        'answers are grounded in the uploaded columns and preview '
        'when available.</div>',
        unsafe_allow_html=True
    )

    _gate_providers = {
        "Auto": "auto",
        "Gemini": "gemini",
        "OpenAI": "openai",
        "Anthropic": "anthropic",
        "Grok": "grok",
        "Deterministic only": "demo",
    }
    _gate_provider_code = _gate_providers.get(
        st.session_state.get("finsight_ai_provider", "Auto"), "auto"
    )
    _gate_effective = (
        _gate_provider_code
        if _gate_provider_code != "auto"
        else resolve_provider("auto")
    )
    _gate_model_default = get_llm_status()["providers"].get(
        _gate_effective, {}).get("model")
    # Current Gemini models (update if Google changes their naming);
    # default always first so the code default takes effect immediately.
    _current_gemini_models = [
        _gate_model_default,  # gemini-3.6-flash
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    # Preserve user's previous valid selection if it still exists,
    # otherwise start with the new default first in the list.
    _saved_model = st.session_state.get(f"finsight_models_{_gate_effective}")
    if _saved_model in _current_gemini_models:
        _gate_model_options = [_saved_model] + _current_gemini_models
    else:
        _gate_model_options = _current_gemini_models
    _gate_model_key = f"finsight_model_{_gate_effective}"
    if st.session_state.get(_gate_model_key) not in _gate_model_options:
        st.session_state[_gate_model_key] = _gate_model_options[0]
    _gate_model = st.session_state.get(_gate_model_key)

    def _ask_gate_fallback(question_text):
        from engine.ai_context import build_financial_ai_context

        _raw = st.session_state.get("uploaded_raw_data")
        _fname = st.session_state.get(
            "upload_filename", "uploaded file")
        try:
            _cols = list(_raw.columns) if _raw is not None else []
        except Exception:
            _cols = []
        try:
            _row_count = int(len(_raw)) if _raw is not None else 0
            _preview = _raw.head(5).to_dict(orient="records") \
                if _raw is not None and not _raw.empty else []
        except Exception:
            _row_count = 0
            _preview = []
        _fin_ctx = build_financial_ai_context(
            question_text,
            company=None,
            metrics_df=None,
            data_source=f"uploaded file ({_fname})" if _cols
                        else "no dataset loaded",
            filename=_fname if _cols else None,
            raw_columns=_cols,
            raw_row_count=_row_count,
            mapped_fields=sorted(
                (st.session_state.get("upload_mapping", {}) or {}).keys()),
            missing_fields=st.session_state.get(
                "upload_missing_fields", []) or [],
            preview_rows=_preview,
        )
        _reply = generate_financial_answer(
            question_text,
            {"evidence": _fin_ctx},
            provider=_gate_provider_code,
            model=_gate_model,
        )
        if _reply.get("status") == "SUCCESS":
            _result = {
                "status": "SUCCESS",
                "answer": "",
                "data": pd.DataFrame(),
                "ai_explanation": _reply,
                "report_type": None,
            }
        else:
            _result = {
                "status": "UNAVAILABLE",
                "answer": _reply.get("response", ""),
                "data": pd.DataFrame(),
                "ai_explanation": None,
                "report_type": None,
            }
        st.session_state.finsight_chat_history.append(
            {"question": question_text, "result": _result})

    if "finsight_chat_history" not in st.session_state:
        st.session_state.finsight_chat_history = []

    st.markdown(
        '<div class="fs-composer-bar"></div>',
        unsafe_allow_html=True
    )

    _gcc = st.columns([0.55, 4.8, 1.7, 1.7, 0.65])

    with _gcc[0]:
        if st.button(
            "+",
            key="finsight_chat_clear",
            help="Clear draft",
        ):
            st.session_state["finsight_chat_question"] = ""

    with _gcc[1]:
        _gate_question = st.text_input(
            "Your question",
            placeholder="Ask anything",
            key="finsight_chat_question",
            label_visibility="collapsed",
        )

    with _gcc[2]:
        st.selectbox(
            "Provider",
            [
                "Auto",
                "Gemini",
                "OpenAI",
                "Anthropic",
                "Grok",
                "Deterministic only",
            ],
            key="finsight_ai_provider",
            label_visibility="collapsed",
            help="AI provider (Auto uses the first configured).",
        )

    with _gcc[3]:
        st.selectbox(
            "Model",
            _gate_model_options,
            key=_gate_model_key,
            label_visibility="collapsed",
            help="Model for the selected provider.",
        )

    with _gcc[4]:
        _gate_ask = st.button(
            "↑",
            key="finsight_chat_analyze",
            help="Send question",
        )

    if _gate_ask:
        if not _gate_question.strip():
            st.warning("Please enter a financial question.")
        else:
            _ask_gate_fallback(_gate_question.strip())

    if st.session_state.finsight_chat_history:
        st.markdown("### Conversation")

        for _item in reversed(
            st.session_state.finsight_chat_history
        ):
            _question = _item.get("question", "") or ""
            _res = _item.get("result", {}) or {}
            _ai = _res.get("ai_explanation")
            if isinstance(_ai, dict) and _ai.get("status") == "SUCCESS":
                _ai_response = _ai.get("response", "") or ""
            elif _res.get("answer"):
                _ai_response = _res.get("answer") or ""
            else:
                _ai_response = ""

            # Role label + user question - render label as controlled HTML,
            # question as plain markdown (no raw HTML tags stored in content).
            st.markdown(
                '<span class="fs-role fs-role-you">YOU</span>',
                unsafe_allow_html=True,
            )
            st.markdown(_question)

            # AI response - render role label + response text.
            if _ai_response:
                st.markdown(
                    '<span class="fs-role fs-role-ai">FINSIGHT AI</span>',
                    unsafe_allow_html=True,
                )
                st.markdown(_ai_response)
            elif _res.get("answer"):
                st.info(_res.get("answer"))

    st.stop()


companies = get_companies(df)


# ============================================================
# SIDEBAR - ANALYSIS SELECTION
# ============================================================

with st.sidebar:

    st.divider()

    st.subheader("Analysis Selection")

    selected_company = st.selectbox(
        "Company",
        companies,
        index=companies.index("AAPL")
        if "AAPL" in companies
        else 0,
        key="finsight_company"
    )

    company_years = get_years(
        df,
        selected_company
    )

    selected_year = st.selectbox(
        "Financial Year",
        company_years,
        index=len(company_years) - 1,
        key="finsight_year"
    )

    st.divider()

    # Currency selector
    selected_currency = st.selectbox(
        "Display Currency",
        ["USD", "INR"],
        index=0,
        key="finsight_currency"
    )

    # Display-only conversion rate
    USD_TO_INR = 83.0

    if selected_currency == "INR":
        st.caption(
            "Display conversion: ₹83 per USD"
        )
    else:
        st.caption(
            "Source currency: USD"
        )

    st.divider()

    st.caption("FinSight AI v1.0")




# ============================================================
# MAIN NAVIGATION (tabs created in the global header above)
# ============================================================

# ============================================================
# ============================================================
# COMPARE
# ============================================================

with tab_compare:
    st.markdown(
        '<div class="fs-eyebrow">Period Comparison</div>'
        '<div class="fs-hero">Compare performance across periods.</div>'
        '<div class="fs-lede">Side-by-side financial years, measured '
        'on the same basis.</div>',
        unsafe_allow_html=True
    )

    st.header("Financial Comparison")

    st.caption(
        "Compare two selected financial years using values supplied "
        "by the uploaded dataset and mathematically derived metrics."
    )

    from engine.comparison import (
        compare_two_years,
        build_comparison_summary
    )

    comparison_company = st.selectbox(
        "Company",
        companies,
        key="comparison_company"
    )

    comparison_company_df = df[
        df["company"].astype(str) == str(comparison_company)
    ].copy()

    comparison_years = sorted(
        comparison_company_df["year"].astype(int).unique()
    )

    if len(comparison_years) < 2:

        st.warning(
            "At least two years of financial data are required "
            "for comparison."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:
            comparison_year_a = st.selectbox(
                "Year A",
                comparison_years,
                index=max(0, len(comparison_years) - 2),
                key="comparison_year_a"
            )

        with col2:
            comparison_year_b = st.selectbox(
                "Year B",
                comparison_years,
                index=len(comparison_years) - 1,
                key="comparison_year_b"
            )

        if comparison_year_a == comparison_year_b:

            st.warning(
                "Please select two different years."
            )

        else:

            try:

                comparison_result = compare_two_years(
                    df,
                    comparison_company,
                    comparison_year_a,
                    comparison_year_b
                )

                comparison_summary = build_comparison_summary(
                    comparison_result,
                    comparison_year_a,
                    comparison_year_b
                )

                st.subheader(
                    f"{comparison_company}: "
                    f"{comparison_year_a} vs {comparison_year_b}"
                )

                display_comparison = comparison_result.copy()

                display_comparison[
                    "Availability"
                ] = display_comparison[
                    "Availability"
                ].fillna("Not provided")

                st.dataframe(
                    display_comparison,
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("Key Changes")

                if comparison_summary:

                    summary_rows = []

                    for item in comparison_summary:

                        summary_rows.append({
                            "Metric": item["metric"],
                            "Direction": item["direction"],
                            "Change (%)": round(
                                item["percentage_change"],
                                2
                            )
                        })

                    summary_df = pd.DataFrame(summary_rows)

                    st.dataframe(
                        summary_df,
                        use_container_width=True,
                        hide_index=True
                    )

                else:

                    st.info(
                        "No comparable metrics were available "
                        "for the selected years."
                    )

            except Exception as e:

                st.error(
                    f"Unable to generate comparison: {e}"
                )
# ============================================================
# ANALYSIS DATA
# ============================================================

snapshot = get_company_snapshot(
    df,
    selected_company,
    selected_year
)

trend = get_company_trend(
    df,
    selected_company
)

# Keep trend analysis aligned with the selected financial year.
# This makes Revenue Trend, Net Income Trend,
# Profitability Analysis, and Recent Performance
# respond to the selected year.
trend = trend[
    trend["year"].astype(int) <= int(selected_year)
].copy()

trend = trend.sort_values("year").reset_index(drop=True)


# ============================================================
# DISPLAY CONVERSION
# ============================================================

def format_money(value):
    """
    Format monetary values according to selected currency.
    Source data is USD millions.
    """

    if selected_currency == "INR":

        # USD million × 83 = INR million
        inr_million = value * USD_TO_INR

        # Display large values in lakh crore
        lakh_crore = inr_million / 1_000_000

        return f"₹{lakh_crore:,.2f} Lakh Cr"

    else:

        usd_billion = value / 1000

        return f"${usd_billion:,.2f}B"


def convert_chart_values(series):
    """
    Convert monetary values for display only.

    Source data:
    USD millions

    Display:
    USD -> USD billions
    INR -> INR lakh crore
    """

    if selected_currency == "INR":
        # USD million -> INR million -> INR lakh crore
        return (series * USD_TO_INR) / 1_000_000

    # USD million -> USD billion
    return series / 1000


# ============================================================
# DISPLAY CONVERSION
# ============================================================

def format_money(value):
    """
    Format monetary values according to selected currency.
    Source data is USD millions.
    """

    if selected_currency == "INR":

        # USD million × 83 = INR million
        inr_million = value * USD_TO_INR

        # Display large values in lakh crore
        lakh_crore = inr_million / 1_000_000

        return f"₹{lakh_crore:,.2f} Lakh Cr"

    else:

        usd_billion = value / 1000

        return f"${usd_billion:,.2f}B"


def convert_chart_values(series):
    """
    Convert monetary values for display only.

    Source data:
    USD millions

    Display:
    USD -> USD billions
    INR -> INR lakh crore
    """

    if selected_currency == "INR":
        # USD million -> INR million -> INR lakh crore
        return (series * USD_TO_INR) / 1_000_000

    # USD million -> USD billion
    return series / 1000


with tab_overview:
    st.markdown(
        '<div class="fs-eyebrow">Financial Overview</div>'
        '<div class="fs-hero">Know the business at a glance.</div>'
        '<div class="fs-lede">A complete view of the company&rsquo;s '
        'financial position and performance.</div>',
        unsafe_allow_html=True
    )
    # ============================================================
    # FINANCIAL HEALTH & RISK
    # ============================================================

    risk_df = build_health_risk_scores(
        calculate_financial_metrics(df.copy()),
        selected_year
    )

    selected_risk = risk_df[
        (risk_df["company"] == selected_company) &
        (risk_df["year"] == selected_year)
    ]

    if not selected_risk.empty:

        health_score = float(selected_risk.iloc[0]["health_score"])
        risk_score = float(selected_risk.iloc[0]["risk_score"])

        # Derive health status from the existing health score
        if health_score >= 80:
            health_status = "Excellent"
        elif health_score >= 65:
            health_status = "Healthy"
        elif health_score >= 50:
            health_status = "Moderate"
        elif health_score >= 35:
            health_status = "Watch"
        else:
            health_status = "High Risk"

        st.markdown(
            '<div class="section-title">Financial Health Assessment</div>',
            unsafe_allow_html=True
        )

        health_col1, health_col2, health_col3 = st.columns(3)

        with health_col1:
            st.metric(
                "Financial Health Score",
                f"{health_score:.1f}/100"
            )

        with health_col2:
            st.metric(
                "Risk Score",
                f"{risk_score:.1f}/100"
            )

        with health_col3:
            st.metric(
                "Overall Status",
                health_status
            )

        st.caption(
            "Composite assessment based on profitability, growth, liquidity, leverage and cash flow indicators."
        )


    # ============================================================
    # FINANCIAL OVERVIEW
    # ============================================================

    st.markdown(
        '<div class="section-title">Financial Overview</div>',
        unsafe_allow_html=True
    )


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Revenue",
            format_money(snapshot["revenue"])
        )

    with col2:
        st.metric(
            "Net Income",
            format_money(snapshot["net_income"])
        )

    with col3:
        st.metric(
            "EBITDA",
            format_money(snapshot["ebitda"])
        )

    with col4:
        st.metric(
            "Net Profit Margin",
            f"{snapshot['net_profit_margin']:.2f}%"
        )


    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric(
            "Gross Profit",
            format_money(snapshot["gross_profit"])
        )

    with col6:
        st.metric(
            "Operating Cash Flow",
            format_money(snapshot["operating_cash_flow"])
        )

    with col7:
        st.metric(
            "Current Ratio",
            f"{snapshot['current_ratio']:.2f}x"
        )

    with col8:
        st.metric(
            "Debt / Equity",
            f"{snapshot['debt_equity_ratio']:.2f}x"
        )

    # ============================================================
    # BALANCE SHEET STRUCTURE (Overview only)
    # ============================================================

    from engine.visuals import get_balance_sheet_history

    st.markdown(
        '<div class="section-title">Balance Sheet Structure</div>',
        unsafe_allow_html=True
    )

    balance = get_balance_sheet_history(
        df,
        selected_company,
        selected_year
    )

    if not balance["available"]:

        if balance["missing"]:
            st.info(
                "Balance sheet structure is not available: "
                f"{', '.join(balance['missing'])} "
                "was not provided in the dataset."
            )
        else:
            st.info(
                "Balance sheet values are not available for the "
                "selected company and year."
            )

    else:

        balance_data = balance["history"].copy()
        balance_data["display_value"] = convert_chart_values(
            balance_data["value"]
        )

        balance_chart = alt.Chart(
            balance_data
        ).mark_bar().encode(

            x=alt.X(
                "year:O",
                title="Year"
            ),

            y=alt.Y(
                "display_value:Q",
                title=(
                    "INR Lakh Crore"
                    if selected_currency == "INR"
                    else "USD Billion"
                )
            ),

            color=alt.Color(
                "component:N",
                title="Component",
                sort=[
                    "Total Assets",
                    "Total Liabilities",
                    "Shareholder Equity"
                ]
            ),

            tooltip=[
                alt.Tooltip(
                    "year:O",
                    title="Year"
                ),
                alt.Tooltip(
                    "component:N",
                    title="Component"
                ),
                alt.Tooltip(
                    "display_value:Q",
                    title=(
                        "₹ Lakh Cr"
                        if selected_currency == "INR"
                        else "$ Billion"
                    ),
                    format=",.2f"
                )
            ]
        ).properties(
            height=350
        )

        st.altair_chart(
            balance_chart,
            width="stretch"
        )

        latest_balance = balance.get("latest", {})
        balance_cards = []

        if "assets" in latest_balance:
            balance_cards.append(
                ("Latest Assets", format_money(latest_balance["assets"]))
            )
        if "liabilities" in latest_balance:
            balance_cards.append(
                (
                    "Latest Liabilities",
                    format_money(latest_balance["liabilities"])
                )
            )
        if "equity" in latest_balance:
            balance_cards.append(
                ("Latest Equity", format_money(latest_balance["equity"]))
            )
        if "liabilities_to_assets" in latest_balance:
            balance_cards.append(
                (
                    "Liabilities / Assets",
                    f"{latest_balance['liabilities_to_assets']:.2f}%"
                )
            )
        if "equity_to_assets" in latest_balance:
            balance_cards.append(
                (
                    "Equity / Assets",
                    f"{latest_balance['equity_to_assets']:.2f}%"
                )
            )

        if balance_cards:
            balance_cols = st.columns(len(balance_cards))

            for col, (label, value) in zip(balance_cols, balance_cards):
                with col:
                    st.metric(label, value)

    # ------------------------------------------------------------
    # ============================================================
    # UPLOADED DATASET DETAILS (Overview only)
    # ============================================================

    # ============================================================
    # DATASET SUMMARY
    # ============================================================

    st.markdown("##  Uploaded Financial Dataset")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Rows",
            len(raw_uploaded_df)
        )

    with c2:
        st.metric(
            "Columns",
            len(raw_uploaded_df.columns)
        )

    with c3:
        st.metric(
            "Recognized Features",
            available_count
        )

    with c4:
        st.metric(
            "Derived Metrics",
            derived_count
        )

    # ============================================================
    # FILE INFORMATION
    # ============================================================

    file_type = upload_file_info.get(
        "file_type",
        "Unknown"
    )

    selected_sheet = upload_file_info.get(
        "selected_sheet"
    )

    if selected_sheet:
        st.caption(
            f"File type: {file_type} | "
            f"Analyzed sheet: {selected_sheet}"
        )
    else:
        st.caption(
            f"File type: {file_type}"
        )

    # ============================================================
    # DATA QUALITY (Overview only)
    # ============================================================

    quality = analyze_data_quality(
        raw_uploaded_df
    )

    st.markdown(
        "###  Data Quality"
    )

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        st.metric(
            "Status",
            quality.get(
                "status",
                "UNKNOWN"
            )
        )

    with q2:
        st.metric(
            "Missing Values",
            quality.get(
                "missing_values",
                0
            )
        )

    with q3:
        st.metric(
            "Duplicates",
            quality.get(
                "duplicate_records",
                0
            )
        )

    with q4:
        st.metric(
            "Numeric Issues",
            quality.get(
                "numeric_anomalies",
                0
            )
        )

    quality_issues = quality.get(
        "issues",
        []
    )

    if quality_issues:
        for issue in quality_issues:
            st.warning(issue)
    else:
        st.success(
            "No basic data-quality issues detected."
        )

    # ADDITIONAL / UNMAPPED COLUMNS
    # ============================================================

    if extra_columns:

        st.markdown(
            "###  Additional Dataset Features"
        )

        st.info(
            "These columns exist in the uploaded dataset "
            "but are not currently mapped to a FinSight "
            "financial concept. Their original values "
            "are preserved. FinSight does not fabricate "
            "financial meaning or values for them."
        )

        extra_display = pd.DataFrame({
            "Additional Column": extra_columns,
            "Status": [
                "Detected — not currently mapped"
                for _ in extra_columns
            ]
        })

        st.dataframe(
            extra_display,
            use_container_width=True,
            hide_index=True
        )

    # Feature Intelligence — Overview tab only
    # ------------------------------------------------------------

    with st.expander("Feature Intelligence", expanded=True):

        feature_intelligence_result = render_feature_intelligence(
            df
        )



with tab_investigate:
    st.markdown(
        '<div class="fs-eyebrow">Investigation</div>'
        '<div class="fs-hero">Understand what changed.</div>'
        '<div class="fs-lede">What moved, what matters, and where to '
        'look next.</div>',
        unsafe_allow_html=True
    )
    # ============================================================
    

# VALIDATION CENTER
    # ============================================================

    validation_df = run_validation(
        calculate_financial_metrics(df.copy())
    )

    validation_pass = int(
        (validation_df["status"] == "PASS").sum()
    )

    validation_review = int(
        (validation_df["status"] == "REVIEW").sum()
    )

    st.markdown(
        '<div class="section-title"> Validation Center</div>',
        unsafe_allow_html=True
    )

    val_col1, val_col2, val_col3 = st.columns(3)

    with val_col1:
        st.metric(
            "Validation Tests",
            len(validation_df)
        )

    with val_col2:
        st.metric(
            "Passed",
            validation_pass
        )

    with val_col3:
        st.metric(
            "Needs Review",
            validation_review
        )

    for _, test in validation_df.iterrows():

        test_name = test["test_name"]
        status = str(test["status"]).upper()
        records_failed = int(test["records_failed"])

        if status == "PASS":

            st.success(
                f" **{test_name}** — PASS  |  "
                f"Records checked: {int(test['records_checked'])}"
            )

        else:

            max_diff = test["max_difference"]

            if pd.notna(max_diff):
                detail = f"Max difference: {float(max_diff):.2f}"
            else:
                detail = "Review required"

            st.warning(
                f" **{test_name}** — REVIEW  |  "
                f"Failed records: {records_failed}  |  {detail}"
            )

    # ============================================================
    # AI INVESTIGATION
    # ============================================================

    # Detect anomalies first so they can feed the investigation layer
    anomaly_input_df = calculate_financial_metrics(df.copy())
    anomaly_df = detect_anomalies(anomaly_input_df)

    if anomaly_df.empty:
        selected_anomalies = anomaly_df.copy()
    else:
        selected_anomalies = anomaly_df[
            (anomaly_df["company"] == selected_company) &
            (anomaly_df["year"] == selected_year)
        ].copy()

    analysis_df = calculate_financial_metrics(df.copy())

    investigation = generate_investigation_report(
        analysis_df,
        selected_company,
        selected_year
    )

    st.markdown(
        '<div class="section-title"> AI Investigation</div>',
        unsafe_allow_html=True
    )

    if investigation:

        primary_driver = investigation.get(
            "primary_driver",
            "No Material Driver Detected"
        )

        driver_count = investigation.get("driver_count", 0)
        evidence_count = investigation.get("evidence", {}).get("evidence_count", 0)

        # If anomaly exists but root-cause engine has no driver,
        # promote the strongest anomaly into the investigation view.
        if not selected_anomalies.empty:

            first_anomaly = selected_anomalies.iloc[0]

            if driver_count == 0:
                primary_driver = first_anomaly.get(
                    "finding_type",
                    "Financial Anomaly"
                )

                driver_count = len(selected_anomalies)
                evidence_count = len(selected_anomalies)

        if not primary_driver:
            primary_driver = "No Material Driver Detected"

        inv_col1, inv_col2, inv_col3 = st.columns(3)

        with inv_col1:
            st.markdown("**Primary Driver**")
            st.info(primary_driver)

        with inv_col2:
            st.metric(
                "Drivers Detected",
                driver_count
            )

        with inv_col3:
            st.metric(
                "Evidence Points",
                evidence_count
            )

        if not selected_anomalies.empty:

            st.markdown("** Investigation Evidence**")

            for _, finding in selected_anomalies.iterrows():

                severity = str(
                    finding.get("severity", "REVIEW")
                ).upper()

                metric = finding.get(
                    "metric",
                    "Financial Metric"
                )

                change = finding.get(
                    "change",
                    None
                )

                anomaly_type = finding.get(
                    "finding_type",
                    "Financial Anomaly"
                )

                if pd.notna(change):
                    change_text = f"{float(change):+.2f}%"
                else:
                    change_text = "Material movement detected"

                if severity == "HIGH":
                    st.error(
                        f" **{anomaly_type}** — {metric}\n\n"
                        f"Severity: **High** | Change: **{change_text}**"
                    )

                else:
                    st.warning(
                        f" **{anomaly_type}** — {metric}\n\n"
                        f"Severity: **{severity.title()}** | Change: **{change_text}**"
                    )

            st.info(
                "**AI Assessment**\n\n"
                "The detected anomaly has been promoted into the investigation "
                "workflow for further financial review and supporting evidence analysis."
            )

        else:

            st.success(
                " **Routine Review — No Material Driver Detected**\n\n"
                "The selected financial period does not show a significant "
                "root-cause movement based on the current investigation rules."
            )

        # ========================================================
        # GROUNDED AI INVESTIGATION REPORT
        # ========================================================

        # Build the structured AI context from FinSight analytics
        ai_context = build_ai_investigation_context({
            "company": selected_company,
            "year": selected_year,
            "financial_data": analysis_df,
            "validation": None,
            "anomalies": anomaly_df,
            "benchmark": build_peer_benchmark(
                analysis_df,
                selected_year
            ),
            "risk": build_health_risk_scores(
                analysis_df,
                selected_year
            ),
            "investigation": investigation,
            "scenarios": None,
            "financial_map": None
        })

        ai_signals = extract_ai_signals(ai_context)

        # Generate the grounded structured AI report
        ai_report = generate_mock_ai_report(ai_signals)

        # Send the grounded prompt through the LLM interface.
        # Current implementation runs in DEMO mode until a real
        # LLM provider/API key is configured.
        ai_prompt = build_ai_investigation_prompt(ai_signals)

        llm_response = generate_ai_response(
            prompt=ai_prompt,
            context=ai_signals
        )

        llm_status = get_llm_status()

        st.markdown("### AI Financial Assessment")

        st.info(
            "Demo AI mode — this assessment is grounded in "
            "FinSight's deterministic financial analytics. "
            "A real LLM will be connected later."
        )

        st.markdown(
            "**Executive Summary**"
        )
        st.write(
            ai_report["executive_summary"]
        )

        st.markdown(
            "**Key Findings**"
        )

        if ai_report["key_findings"]:
            for finding in ai_report["key_findings"]:
                st.write(f"• {finding}")
        else:
            st.write("No key findings identified.")

        st.markdown(
            "**Root Cause Assessment**"
        )
        st.write(
            ai_report["root_cause_assessment"]
        )

        st.markdown(
            "**Peer Benchmark Observations**"
        )

        benchmark_report = ai_report["peer_benchmark"]

        if benchmark_report:
            st.write(
                f"Peer benchmark status: "
                f"{benchmark_report.get('peer_status', 'Unavailable')}"
            )

            if benchmark_report.get("peer_count") is not None:
                st.write(
                    f"Peers available: "
                    f"{benchmark_report['peer_count']}"
                )

            if benchmark_report.get("revenue_vs_peer_pct") is not None:
                st.write(
                    f"Revenue vs peer average: "
                    f"{float(benchmark_report['revenue_vs_peer_pct']):+.2f}%"
                )

            if benchmark_report.get("net_income_vs_peer_pct") is not None:
                st.write(
                    f"Net income vs peer average: "
                    f"{float(benchmark_report['net_income_vs_peer_pct']):+.2f}%"
                )

            if benchmark_report.get("current_ratio_vs_peer_pct") is not None:
                st.write(
                    f"Current ratio vs peer average: "
                    f"{float(benchmark_report['current_ratio_vs_peer_pct']):+.2f}%"
                )

            if benchmark_report.get("debt_equity_vs_peer_pct") is not None:
                st.write(
                    f"Debt-to-equity vs peer average: "
                    f"{float(benchmark_report['debt_equity_vs_peer_pct']):+.2f}%"
                )

        st.caption(
            ai_report["grounding"]
        )


    # ============================================================
    # ANOMALY DETECTION
    # ============================================================

    anomaly_input_df = calculate_financial_metrics(df.copy())
    anomaly_df = detect_anomalies(anomaly_input_df)

    if anomaly_df.empty:
        selected_anomalies = anomaly_df.copy()
    else:
        selected_anomalies = anomaly_df[
            (anomaly_df["company"] == selected_company) &
            (anomaly_df["year"] == selected_year)
        ].copy()

    st.markdown(
        '<div class="section-title"> Anomaly Detection</div>',
        unsafe_allow_html=True
    )

    if selected_anomalies.empty:

        st.success(
            " **No significant anomalies detected**\n\n"
            "The selected financial period does not trigger any of the "
            "current anomaly detection rules."
        )

    else:

        high_count = len(
            selected_anomalies[
                selected_anomalies["severity"].str.upper() == "HIGH"
            ]
        )

        medium_count = len(
            selected_anomalies[
                selected_anomalies["severity"].str.upper() == "MEDIUM"
            ]
        )

        anomaly_col1, anomaly_col2, anomaly_col3 = st.columns(3)

        with anomaly_col1:
            st.metric(
                "Total Findings",
                len(selected_anomalies)
            )

        with anomaly_col2:
            st.metric(
                "High Severity",
                high_count
            )

        with anomaly_col3:
            st.metric(
                "Medium Severity",
                medium_count
            )

        # ----------------------------------------------------
        # Anomaly Timeline — when anomalies occurred
        # Uses the existing detect_anomalies() output; company
        # history up to selected_year (never beyond it).
        # ----------------------------------------------------

        st.markdown("**Anomaly Timeline**")

        if anomaly_df.empty:
            timeline_anomalies = anomaly_df.copy()
        else:
            timeline_anomalies = anomaly_df[
                (anomaly_df["company"] == selected_company)
                & (
                    anomaly_df["year"].astype(int)
                    <= int(selected_year)
                )
            ].copy()

        if timeline_anomalies.empty:

            st.caption(
                "No anomalies recorded up to the selected year."
            )

        else:

            severity_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
            severity_label = {3: "High", 2: "Medium", 1: "Low"}

            timeline_plot = timeline_anomalies.copy()
            timeline_plot["severity_rank"] = (
                timeline_plot["severity"]
                .astype(str)
                .str.upper()
                .map(severity_rank)
            )
            timeline_plot = timeline_plot.dropna(
                subset=["severity_rank"]
            )
            timeline_plot["severity_label"] = (
                timeline_plot["severity_rank"].map(severity_label)
            )

            if timeline_plot.empty:

                st.caption(
                    "No anomalies recorded up to the selected year."
                )

            else:

                timeline_chart = alt.Chart(
                    timeline_plot
                ).mark_circle(size=140).encode(

                    x=alt.X(
                        "year:O",
                        title="Year"
                    ),

                    y=alt.Y(
                        "severity_label:N",
                        title="Severity",
                        sort=["High", "Medium", "Low"]
                    ),

                    color=alt.Color(
                        "severity_label:N",
                        title="Severity",
                        sort=["High", "Medium", "Low"]
                    ),

                    tooltip=[
                        alt.Tooltip("year:O", title="Year"),
                        alt.Tooltip(
                            "finding_type:N",
                            title="Finding Type"
                        ),
                        alt.Tooltip("metric:N", title="Metric"),
                        alt.Tooltip(
                            "change:Q",
                            title="Change",
                            format=".2f"
                        ),
                        alt.Tooltip("severity:N", title="Severity"),
                        alt.Tooltip("title:N", title="Title")
                    ]
                ).properties(
                    height=300
                )

                st.altair_chart(
                    timeline_chart,
                    width="stretch"
                )

        # Display available anomaly information
        display_columns = [
            col for col in [
                "severity",
                "anomaly_type",
                "metric",
                "message",
                "change"
            ]
            if col in selected_anomalies.columns
        ]

        if display_columns:
            st.dataframe(
                selected_anomalies[display_columns],
                width="stretch",
                hide_index=True
            )
        else:
            st.dataframe(
                selected_anomalies,
                width="stretch",
                hide_index=True
            )

    # ============================================================
    # RISK FACTOR BREAKDOWN (deterministic, from risk.py)
    # ============================================================

    from engine.risk import (
        calculate_health_factor_scores,
        FACTOR_LABELS,
        FACTOR_WEIGHTS,
    )

    st.markdown(
        '<div class="section-title">Risk Factor Breakdown</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Deterministic factor scores behind the Financial Health Score. "
        "Higher is stronger; 50 is neutral (missing inputs score 50)."
    )

    factor_row = analysis_df[
        (analysis_df["company"].astype(str) == str(selected_company))
        & (analysis_df["year"].astype(int) == int(selected_year))
    ]

    if factor_row.empty:

        st.info(
            "Risk factor scores are unavailable for the selected "
            "company and year."
        )

    else:

        factor_scores = calculate_health_factor_scores(
            factor_row.iloc[0]
        )

        factor_records = [
            {
                "factor": FACTOR_LABELS.get(name, name),
                "score": float(score),
                "weight": f"{int(FACTOR_WEIGHTS.get(name, 0) * 100)}%",
            }
            for name, score in factor_scores.items()
        ]

        factor_frame = pd.DataFrame(factor_records)

        factor_bars = alt.Chart(
            factor_frame
        ).mark_bar().encode(

            x=alt.X(
                "score:Q",
                title="Factor Score (0-100)",
                scale=alt.Scale(domain=[0, 100])
            ),

            y=alt.Y(
                "factor:N",
                title=None,
                sort=alt.EncodingSortField(
                    field="score",
                    order="ascending"
                )
            ),

            color=alt.Color(
                "score:Q",
                title="Score",
                scale=alt.Scale(
                    domain=[0, 50, 100],
                    range=["#d95f5f", "#e8c872", "#6fbf73"]
                ),
                legend=None
            ),

            tooltip=[
                alt.Tooltip("factor:N", title="Factor"),
                alt.Tooltip("score:Q", title="Score", format=".1f"),
                alt.Tooltip("weight:N", title="Weight")
            ]
        )

        neutral_rule = alt.Chart(
            pd.DataFrame({"threshold": [50]})
        ).mark_rule(
            strokeDash=[5, 5]
        ).encode(
            x="threshold:Q"
        )

        factor_chart = (factor_bars + neutral_rule).properties(
            height=300
        )

        st.altair_chart(
            factor_chart,
            width="stretch"
        )

with tab_benchmark:
    st.markdown(
        '<div class="fs-eyebrow">Peer Benchmark</div>'
        '<div class="fs-hero">See how the company compares.</div>'
        '<div class="fs-lede">Performance relative to comparable '
        'companies.</div>',
        unsafe_allow_html=True
    )
    # ============================================================
    # PEER BENCHMARK
    # ============================================================

    benchmark_df = build_peer_benchmark(df, selected_year)

    selected_benchmark = benchmark_df[
        benchmark_df["company"] == selected_company
    ].copy()

    st.markdown(
        '<div class="section-title"> Peer Benchmark</div>',
        unsafe_allow_html=True
    )

    if selected_benchmark.empty:

        st.info(
            "No benchmark information is available for the selected company."
        )

    else:

        benchmark_row = selected_benchmark.iloc[0]
        peer_count = int(benchmark_row["peer_count"])

        if peer_count == 0:

            st.info(
                " **No comparable peer group available**\n\n"
                "The selected company does not have enough companies in the "
                "same category for a meaningful peer comparison."
            )

        else:

            st.caption(
                f"Category: {benchmark_row['category']}  |  "
                f"Peer companies: {peer_count}  |  "
                f"Benchmark year: {int(benchmark_row['year'])}"
            )

            # Shared peer-analysis inputs (additive only — the benchmark
            # engine itself is untouched). bench_metrics carries YoY and
            # derived fields needed by ranking/scatter/trend sections.
            bench_metrics = calculate_financial_metrics(df.copy())
            bench_category = str(benchmark_row.get("category", ""))

            if peer_count == 1:
                st.caption(
                    "1 comparable peer — averages are indicative, not "
                    "a statistically meaningful peer group."
                )

            # ----------------------------------------------------
            # Benchmark summary metrics
            # Optional metrics such as ROE/ROA are displayed only
            # when they are actually available in the uploaded data.
            # ----------------------------------------------------

            benchmark_cards = [
                ("Revenue vs Peers", "revenue_vs_peer_pct", "%"),
                ("Net Income vs Peers", "net_income_vs_peer_pct", "%"),
                ("Net Margin vs Peers", "net_profit_margin_vs_peer_pct", "%")
            ]

            if "roa_vs_peer_pct" in benchmark_row.index:
                benchmark_cards.append(
                    ("ROA vs Peers", "roa_vs_peer_pct", "%")
                )

            bench_cols = st.columns(len(benchmark_cards))

            for col, (label, key, suffix) in zip(
                bench_cols,
                benchmark_cards
            ):
                with col:
                    value = benchmark_row.get(key)

                    if pd.notna(value):
                        st.metric(
                            label,
                            f"{float(value):+.2f}{suffix}"
                        )
                    else:
                        st.metric(label, "N/A")

            st.markdown("**Company vs Peer Average**")

            benchmark_rows = [
                (
                    "Revenue",
                    "revenue_company",
                    "revenue_peer_avg",
                    ",.2f"
                ),
                (
                    "Net Income",
                    "net_income_company",
                    "net_income_peer_avg",
                    ",.2f"
                ),
                (
                    "Net Profit Margin",
                    "net_profit_margin_company",
                    "net_profit_margin_peer_avg",
                    ".2f%"
                ),
                (
                    "Current Ratio",
                    "current_ratio_company",
                    "current_ratio_peer_avg",
                    ".2fx"
                ),
                (
                    "Debt / Equity",
                    "debt_equity_ratio_company",
                    "debt_equity_ratio_peer_avg",
                    ".2fx"
                )
            ]

            # Add optional profitability-return metrics only when
            # supplied by the uploaded financial dataset.
            if (
                "roe_company" in benchmark_row.index
                and "roe_peer_avg" in benchmark_row.index
            ):
                benchmark_rows.insert(
                    3,
                    (
                        "ROE",
                        "roe_company",
                        "roe_peer_avg",
                        ".2f%"
                    )
                )

            if (
                "roa_company" in benchmark_row.index
                and "roa_peer_avg" in benchmark_row.index
            ):
                insert_position = 4 if any(
                    row[0] == "ROE"
                    for row in benchmark_rows
                ) else 3

                benchmark_rows.insert(
                    insert_position,
                    (
                        "ROA",
                        "roa_company",
                        "roa_peer_avg",
                        ".2f%"
                    )
                )

            metrics = []
            company_values = []
            peer_values = []

            for metric_name, company_key, peer_key, fmt in benchmark_rows:
                metrics.append(metric_name)

                company_value = benchmark_row.get(company_key)
                peer_value = benchmark_row.get(peer_key)

                if pd.notna(company_value):
                    value = float(company_value)

                    if fmt.endswith("%"):
                        company_values.append(f"{value:.2f}%")
                    elif fmt.endswith("x"):
                        company_values.append(f"{value:.2f}x")
                    else:
                        company_values.append(f"{value:,.2f}")
                else:
                    company_values.append("N/A")

                if pd.notna(peer_value):
                    value = float(peer_value)

                    if fmt.endswith("%"):
                        peer_values.append(f"{value:.2f}%")
                    elif fmt.endswith("x"):
                        peer_values.append(f"{value:.2f}x")
                    else:
                        peer_values.append(f"{value:,.2f}")
                else:
                    peer_values.append("N/A")

            benchmark_display = pd.DataFrame({
                "Metric": metrics,
                "Company": company_values,
                "Peer Average": peer_values
            })

            st.dataframe(
                benchmark_display,
                width="stretch",
                hide_index=True
            )

            # ----------------------------------------------------
            # Benchmark Snapshot — compact peer context.
            # ----------------------------------------------------

            st.markdown("**Benchmark Snapshot**")

            _rank_check = prepare_peer_ranking(
                bench_metrics,
                selected_company,
                bench_category,
                selected_year,
                "revenue",
            )
            _group_size = (
                int(_rank_check["ranking"]["company"].nunique())
                if _rank_check["available"]
                else peer_count + 1
            )
            _metrics_available = sum(
                1 for _, col, _ in RANK_METRICS
                if col in bench_metrics.columns
                and bench_metrics[col].notna().any()
            )

            _snapshot_cards = [
                ("Peer Companies", str(peer_count)),
                ("Benchmark Year", str(int(benchmark_row["year"]))),
                ("Category", bench_category),
                (
                    "Company Rank (Revenue)",
                    (
                        f"#{_rank_check['selected_rank']} of {_group_size}"
                        if _rank_check.get("selected_rank") else "n/a"
                    ),
                ),
                ("Metrics Available", str(_metrics_available)),
            ]

            _snapshot_cols = st.columns(len(_snapshot_cards))

            for _col, (_label, _value) in zip(
                _snapshot_cols, _snapshot_cards
            ):
                with _col:
                    st.metric(_label, _value)

            # ----------------------------------------------------
            # Company vs Category Peer Benchmark visuals.
            # Category = dataset "category" field, not an external
            # market benchmark. Monetary values use the selected
            # currency; ratios/percentages are never converted.
            # ----------------------------------------------------

            st.markdown("**Company vs Category Peer Benchmark**")

            st.markdown("**Scale Comparison**")

            scale_records = []

            for label, company_key, peer_key in [
                ("Revenue", "revenue_company", "revenue_peer_avg"),
                ("Net Income", "net_income_company", "net_income_peer_avg"),
            ]:
                company_scale = benchmark_row.get(company_key)
                peer_scale = benchmark_row.get(peer_key)

                if pd.notna(company_scale):
                    scale_records.append({
                        "metric": label,
                        "source": selected_company,
                        "value": float(company_scale),
                    })
                if pd.notna(peer_scale):
                    scale_records.append({
                        "metric": label,
                        "source": "Peer Average",
                        "value": float(peer_scale),
                    })

            if not scale_records:

                st.info(
                    "Revenue and net income comparison values are "
                    "not available."
                )

            else:

                scale_frame = pd.DataFrame(scale_records)
                scale_frame["display_value"] = convert_chart_values(
                    scale_frame["value"]
                )

                scale_chart = alt.Chart(
                    scale_frame
                ).mark_bar().encode(

                    x=alt.X(
                        "source:N",
                        title=None
                    ),

                    y=alt.Y(
                        "display_value:Q",
                        title=(
                            "INR Lakh Crore"
                            if selected_currency == "INR"
                            else "USD Billion"
                        )
                    ),

                    color=alt.Color(
                        "source:N",
                        title="Source"
                    ),

                    column=alt.Column(
                        "metric:N",
                        title=None
                    ),

                    tooltip=[
                        alt.Tooltip("metric:N", title="Metric"),
                        alt.Tooltip("source:N", title="Source"),
                        alt.Tooltip(
                            "display_value:Q",
                            title=(
                                "₹ Lakh Cr"
                                if selected_currency == "INR"
                                else "$ Billion"
                            ),
                            format=",.2f"
                        )
                    ]
                ).properties(
                    height=300
                )

                st.altair_chart(
                    scale_chart,
                    width="stretch"
                )

            # ----------------------------------------------------
            # Peer Ranking — raw ranking among all comparable
            # companies (selected included). Sorted numerically;
            # never reversed, even for Debt / Equity.
            # ----------------------------------------------------

            st.markdown("**Peer Ranking**")

            _rank_choices = [
                (label, col, kind)
                for label, col, kind in RANK_METRICS
                if col in bench_metrics.columns
            ]

            if not _rank_choices:

                st.info(
                    "Ranking metrics are not available in the dataset."
                )

            else:

                _rank_label = st.selectbox(
                    "Ranking metric",
                    [label for label, _, _ in _rank_choices],
                    key="benchmark_rank_metric",
                )

                _rank_col, _rank_kind = next(
                    (col, kind)
                    for label, col, kind in _rank_choices
                    if label == _rank_label
                )

                _ranking = prepare_peer_ranking(
                    bench_metrics,
                    selected_company,
                    bench_category,
                    selected_year,
                    _rank_col,
                )

                if not _ranking["available"] or _ranking["ranking"].empty:

                    st.info(
                        f"No {_rank_label} values are available for "
                        "ranking in this category/year."
                    )

                else:

                    _rank_frame = _ranking["ranking"].copy()

                    if _rank_kind == "money":
                        _rank_frame["display_value"] = convert_chart_values(
                            _rank_frame["_value"]
                        )
                        _rank_axis = (
                            "INR Lakh Crore"
                            if selected_currency == "INR"
                            else "USD Billion"
                        )
                    else:
                        _rank_frame["display_value"] = _rank_frame["_value"]
                        _rank_axis = (
                            f"{_rank_label} "
                            f"({'%' if _rank_kind == 'pct' else 'x'})"
                        )

                    _rank_bars = alt.Chart(
                        _rank_frame
                    ).mark_bar().encode(

                        x=alt.X(
                            "display_value:Q",
                            title=_rank_axis
                        ),

                        y=alt.Y(
                            "company:N",
                            title=None,
                            sort=alt.EncodingSortField(
                                field="display_value",
                                order="descending"
                            )
                        ),

                        color=alt.condition(
                            alt.datum.is_selected,
                            alt.value("#2F6FED"),
                            alt.value("#9AA1AD")
                        ),

                        stroke=alt.condition(
                            alt.datum.is_selected,
                            alt.value("#1F2A44"),
                            alt.value("transparent")
                        ),

                        strokeWidth=alt.condition(
                            alt.datum.is_selected,
                            alt.value(2),
                            alt.value(0)
                        ),

                        tooltip=[
                            alt.Tooltip("rank:Q", title="Rank"),
                            alt.Tooltip("company:N", title="Company"),
                            alt.Tooltip(
                                "display_value:Q",
                                title=_rank_label,
                                format=",.2f"
                            )
                        ]
                    ).properties(
                        height=max(
                            220, 42 * len(_rank_frame)
                        )
                    )

                    _rank_labels = alt.Chart(
                        _rank_frame
                    ).mark_text(
                        align="left",
                        dx=4,
                        fontWeight="bold"
                    ).encode(

                        x="display_value:Q",

                        y=alt.Y(
                            "company:N",
                            sort=alt.EncodingSortField(
                                field="display_value",
                                order="descending"
                            )
                        ),

                        text=alt.condition(
                            alt.datum.is_selected,
                            "company:N",
                            alt.value("")
                        )
                    )

                    st.altair_chart(
                        _rank_bars + _rank_labels,
                        width="stretch"
                    )

                    if _ranking.get("selected_rank"):
                        st.caption(
                            f"{selected_company} ranks "
                            f"#{_ranking['selected_rank']} of "
                            f"{len(_rank_frame)} in {_rank_label} "
                            f"({bench_category}, {selected_year}). "
                            "Raw ranking — not performance-adjusted."
                        )

            st.markdown("**Financial Ratio Comparison**")

            ratio_options = []

            for label, company_key, peer_key, unit in [
                (
                    "Net Profit Margin",
                    "net_profit_margin_company",
                    "net_profit_margin_peer_avg",
                    "%"
                ),
                (
                    "Current Ratio",
                    "current_ratio_company",
                    "current_ratio_peer_avg",
                    "x"
                ),
                (
                    "Debt / Equity",
                    "debt_equity_ratio_company",
                    "debt_equity_ratio_peer_avg",
                    "x"
                ),
                ("ROE", "roe_company", "roe_peer_avg", "%"),
                ("ROA", "roa_company", "roa_peer_avg", "%"),
            ]:
                company_ratio = benchmark_row.get(company_key)
                peer_ratio = benchmark_row.get(peer_key)

                if pd.notna(company_ratio) and pd.notna(peer_ratio):
                    ratio_options.append({
                        "label": label,
                        "company": float(company_ratio),
                        "peer_avg": float(peer_ratio),
                        "unit": unit,
                    })

            if not ratio_options:

                st.info(
                    "No ratio metric has both a company value and "
                    "a peer average available."
                )

            else:

                selected_ratio_label = st.selectbox(
                    "Ratio metric",
                    [option["label"] for option in ratio_options],
                    key="benchmark_ratio_metric",
                )

                selected_ratio = next(
                    option
                    for option in ratio_options
                    if option["label"] == selected_ratio_label
                )

                ratio_frame = pd.DataFrame([
                    {
                        "source": selected_company,
                        "value": selected_ratio["company"],
                    },
                    {
                        "source": "Peer Average",
                        "value": selected_ratio["peer_avg"],
                    },
                ])

                ratio_chart = alt.Chart(
                    ratio_frame
                ).mark_bar().encode(

                    x=alt.X(
                        "source:N",
                        title=None
                    ),

                    y=alt.Y(
                        "value:Q",
                        title=(
                            f"{selected_ratio_label} "
                            f"({selected_ratio['unit']})"
                        )
                    ),

                    color=alt.Color(
                        "source:N",
                        title="Source"
                    ),

                    tooltip=[
                        alt.Tooltip("source:N", title="Source"),
                        alt.Tooltip(
                            "value:Q",
                            title=selected_ratio_label,
                            format=".2f"
                        )
                    ]
                ).properties(
                    height=300
                )

                st.altair_chart(
                    ratio_chart,
                    width="stretch"
                )

            # ----------------------------------------------------
            # Growth vs Profitability scatter.
            # ----------------------------------------------------

            st.markdown("**Growth vs Profitability**")

            _growth_scatter = build_peer_scatter_data(
                bench_metrics,
                selected_company,
                bench_category,
                selected_year,
                "revenue_yoy",
                "net_profit_margin",
            )

            if not _growth_scatter["available"]:

                st.info(
                    "Growth vs profitability scatter is unavailable: "
                    f"{_growth_scatter.get('reason', 'missing data')}."
                )

            elif len(_growth_scatter["points"]) < 2:

                st.info(
                    "Too few companies have both revenue growth and "
                    "net margin for a scatter plot."
                )

            else:

                _gpts = _growth_scatter["points"]
                _gpeers = _gpts[~_gpts["is_selected"]]
                _gself = _gpts[_gpts["is_selected"]]

                _g_base = alt.Chart(_gpeers).mark_circle(
                    size=90,
                    color="#9AA1AD"
                ).encode(
                    x=alt.X("_x:Q", title="Revenue Growth YoY (%)"),
                    y=alt.Y("_y:Q", title="Net Profit Margin (%)"),
                    tooltip=[
                        alt.Tooltip("company:N", title="Company"),
                        alt.Tooltip("category:N", title="Category"),
                        alt.Tooltip("_x:Q", title="Revenue Growth (%)",
                                    format="+.2f"),
                        alt.Tooltip("_y:Q", title="Net Profit Margin (%)",
                                    format=".2f"),
                        alt.Tooltip("year:O", title="Year"),
                    ]
                )

                _g_layers = [_g_base]

                if not _gself.empty:
                    _g_layers.append(
                        alt.Chart(_gself).mark_circle(
                            size=240,
                            color="#2F6FED",
                            stroke="#1F2A44",
                            strokeWidth=2
                        ).encode(
                            x="_x:Q",
                            y="_y:Q",
                            tooltip=[
                                alt.Tooltip("company:N", title="Company"),
                                alt.Tooltip("category:N",
                                            title="Category"),
                                alt.Tooltip("_x:Q",
                                            title="Revenue Growth (%)",
                                            format="+.2f"),
                                alt.Tooltip("_y:Q",
                                            title="Net Profit Margin (%)",
                                            format=".2f"),
                                alt.Tooltip("year:O", title="Year"),
                            ]
                        )
                    )
                    _g_layers.append(
                        alt.Chart(_gself).mark_text(
                            align="left",
                            dx=10,
                            dy=-10,
                            fontWeight="bold"
                        ).encode(
                            x="_x:Q",
                            y="_y:Q",
                            text="company:N"
                        )
                    )

                _growth_chart = alt.layer(*_g_layers).properties(
                    height=340
                )

                st.altair_chart(_growth_chart, width="stretch")
                st.caption(
                    "Right = faster revenue growth; up = higher net "
                    "margin. Relative position only — scatter position "
                    "alone is not a verdict on quality."
                )

            # ----------------------------------------------------
            # Liquidity vs Leverage scatter.
            # ----------------------------------------------------

            st.markdown("**Liquidity vs Leverage**")

            _liq_scatter = build_peer_scatter_data(
                bench_metrics,
                selected_company,
                bench_category,
                selected_year,
                "debt_equity_ratio",
                "current_ratio",
            )

            if not _liq_scatter["available"]:

                st.info(
                    "Liquidity vs leverage scatter is unavailable: "
                    f"{_liq_scatter.get('reason', 'missing data')}."
                )

            elif len(_liq_scatter["points"]) < 2:

                st.info(
                    "Too few companies have both leverage and liquidity "
                    "values for a scatter plot."
                )

            else:

                _lpts = _liq_scatter["points"]
                _lpeers = _lpts[~_lpts["is_selected"]]
                _lself = _lpts[_lpts["is_selected"]]

                _l_base = alt.Chart(_lpeers).mark_circle(
                    size=90,
                    color="#9AA1AD"
                ).encode(
                    x=alt.X("_x:Q", title="Debt / Equity (x)"),
                    y=alt.Y("_y:Q", title="Current Ratio (x)"),
                    tooltip=[
                        alt.Tooltip("company:N", title="Company"),
                        alt.Tooltip("_x:Q", title="Debt / Equity",
                                    format=".2f"),
                        alt.Tooltip("_y:Q", title="Current Ratio",
                                    format=".2f"),
                        alt.Tooltip("year:O", title="Year"),
                        alt.Tooltip("category:N", title="Category"),
                    ]
                )

                _l_layers = [_l_base]

                if not _lself.empty:
                    _l_layers.append(
                        alt.Chart(_lself).mark_circle(
                            size=240,
                            color="#2F6FED",
                            stroke="#1F2A44",
                            strokeWidth=2
                        ).encode(
                            x="_x:Q",
                            y="_y:Q",
                            tooltip=[
                                alt.Tooltip("company:N", title="Company"),
                                alt.Tooltip("_x:Q", title="Debt / Equity",
                                            format=".2f"),
                                alt.Tooltip("_y:Q", title="Current Ratio",
                                            format=".2f"),
                                alt.Tooltip("year:O", title="Year"),
                                alt.Tooltip("category:N",
                                            title="Category"),
                            ]
                        )
                    )
                    _l_layers.append(
                        alt.Chart(_lself).mark_text(
                            align="left",
                            dx=10,
                            dy=-10,
                            fontWeight="bold"
                        ).encode(
                            x="_x:Q",
                            y="_y:Q",
                            text="company:N"
                        )
                    )

                _liq_chart = alt.layer(*_l_layers).properties(
                    height=340
                )

                st.altair_chart(_liq_chart, width="stretch")
                st.caption(
                    "Upper-left = higher liquidity with lower leverage; "
                    "lower-right = lower liquidity with higher leverage. "
                    "No thresholds are drawn and no company is labelled "
                    "good or bad from position alone."
                )

            # ----------------------------------------------------
            # Benchmark Variance heatmap + analytical table.
            # Uses existing *_vs_peer_pct engine values.
            # ----------------------------------------------------

            st.markdown("**Benchmark Variance**")

            _variance_rows = build_benchmark_variance(benchmark_row)
            _heat_rows = [
                row for row in _variance_rows
                if row.get("vs_peer_pct") is not None
            ]

            if not _heat_rows:

                st.info(
                    "No peer-variance values are available for heatmap."
                )

            else:

                _heat_frame = pd.DataFrame(_heat_rows)
                _heat_frame["measure"] = "Vs Peer %"
                _heat_vmax = max(
                    1.0,
                    float(_heat_frame["vs_peer_pct"].abs().max())
                )

                _heat_rects = alt.Chart(_heat_frame).mark_rect().encode(
                    x=alt.X("measure:N", title=None),
                    y=alt.Y(
                        "metric:N",
                        title=None,
                        sort=[row["metric"] for row in _heat_rows]
                    ),
                    color=alt.Color(
                        "vs_peer_pct:Q",
                        title="Vs Peer %",
                        scale=alt.Scale(
                            scheme="redblue",
                            domain=[-_heat_vmax, _heat_vmax],
                            domainMid=0
                        ),
                        legend=alt.Legend(format=".0f")
                    ),
                    tooltip=[
                        alt.Tooltip("metric:N", title="Metric"),
                        alt.Tooltip("vs_peer_pct:Q", title="Vs Peer %",
                                    format="+.2f")
                    ]
                ).properties(height=max(240, 44 * len(_heat_rows)))

                _heat_text = alt.Chart(_heat_frame).mark_text(
                    fontWeight="bold"
                ).encode(
                    x="measure:N",
                    y=alt.Y(
                        "metric:N",
                        sort=[row["metric"] for row in _heat_rows]
                    ),
                    text=alt.Text("vs_peer_pct:Q", format="+.1f"),
                    color=alt.condition(
                        abs(alt.datum.vs_peer_pct) > _heat_vmax / 2,
                        alt.value("white"),
                        alt.value("#1F2A44")
                    )
                )

                st.altair_chart(
                    _heat_rects + _heat_text,
                    width="stretch"
                )
                st.caption(
                    "Positive (blue) = above peer average; negative "
                    "(red) = below. Direction only — higher is not "
                    "automatically better (e.g. leverage)."
                )

            _variance_table = []
            for row in _variance_rows:
                kind = row.get("kind", "")
                company_v = row.get("company")
                peer_v = row.get("peer_avg")
                diff_v = row.get("difference")
                vs_pct = row.get("vs_peer_pct")

                if kind == "money":
                    fmt = lambda v: f"{v:,.2f}"  # noqa: E731
                elif kind == "pct":
                    fmt = lambda v: f"{v:.2f}%"  # noqa: E731
                else:
                    fmt = lambda v: f"{v:.2f}x"  # noqa: E731

                _variance_table.append({
                    "Metric": row["metric"],
                    "Company": fmt(company_v)
                    if company_v is not None else "N/A",
                    "Peer Average": fmt(peer_v)
                    if peer_v is not None else "N/A",
                    "Difference": (
                        f"{diff_v:+.2f}" + (
                            "%" if kind == "pct"
                            else "x" if kind == "x" else ""
                        )
                    ) if diff_v is not None else "N/A",
                    "Vs Peer %": f"{vs_pct:+.2f}%"
                    if vs_pct is not None else "N/A",
                })

            if _variance_table:
                st.dataframe(
                    pd.DataFrame(_variance_table),
                    width="stretch",
                    hide_index=True
                )

            # ----------------------------------------------------
            # Peer Trend Over Time (fixed category, year <= selected).
            # ----------------------------------------------------

            st.markdown("**Peer Trend Over Time**")

            _trend_choices = [
                (label, col, kind)
                for label, col, kind in TREND_METRICS
                if col in bench_metrics.columns
            ]

            if not _trend_choices:

                st.info(
                    "Trend metrics are not available in the dataset."
                )

            else:

                _trend_label = st.selectbox(
                    "Trend metric",
                    [label for label, _, _ in _trend_choices],
                    key="benchmark_trend_metric",
                )

                _trend_col, _trend_kind = next(
                    (col, kind)
                    for label, col, kind in _trend_choices
                    if label == _trend_label
                )

                _trend_data = build_peer_trend_data(
                    bench_metrics,
                    selected_company,
                    bench_category,
                    selected_year,
                    _trend_col,
                )

                if not _trend_data["available"] \
                        or _trend_data["records"].empty:

                    st.info(
                        f"No overlapping company/peer history is "
                        f"available for {_trend_label}."
                    )

                else:

                    _trend_records = _trend_data["records"].copy()
                    _trend_long = pd.DataFrame(
                        [
                            {
                                "year": rec["year"],
                                "source": selected_company,
                                "value": rec["company"],
                                "difference": rec["difference"],
                            }
                            for rec in _trend_records.to_dict(
                                orient="records")
                        ] + [
                            {
                                "year": rec["year"],
                                "source": "Peer Average",
                                "value": rec["peer_avg"],
                                "difference": rec["difference"],
                            }
                            for rec in _trend_records.to_dict(
                                orient="records")
                        ]
                    )

                    if _trend_kind == "money":
                        _trend_long["display_value"] = convert_chart_values(
                            _trend_long["value"]
                        )
                        _trend_long["display_diff"] = convert_chart_values(
                            _trend_long["difference"]
                        )
                        _trend_axis = (
                            "INR Lakh Crore"
                            if selected_currency == "INR"
                            else "USD Billion"
                        )
                        _trend_tip = (
                            "₹ Lakh Cr"
                            if selected_currency == "INR"
                            else "$ Billion"
                        )
                    else:
                        _trend_long["display_value"] = _trend_long["value"]
                        _trend_long["display_diff"] = \
                            _trend_long["difference"]
                        _trend_axis = (
                            f"{_trend_label} "
                            f"({'%' if _trend_kind == 'pct' else 'x'})"
                        )
                        _trend_tip = _trend_label

                    _trend_chart = alt.Chart(
                        _trend_long
                    ).mark_line(
                        point=True
                    ).encode(

                        x=alt.X("year:O", title="Year"),
                        y=alt.Y("display_value:Q", title=_trend_axis),

                        color=alt.Color(
                            "source:N",
                            title="Source",
                            scale=alt.Scale(
                                domain=[selected_company, "Peer Average"],
                                range=["#2F6FED", "#9AA1AD"]
                            )
                        ),

                        strokeWidth=alt.condition(
                            alt.datum.source == selected_company,
                            alt.value(3),
                            alt.value(1.5)
                        ),

                        tooltip=[
                            alt.Tooltip("year:O", title="Year"),
                            alt.Tooltip("source:N", title="Source"),
                            alt.Tooltip(
                                "display_value:Q",
                                title=_trend_tip,
                                format=",.2f"
                            ),
                            alt.Tooltip(
                                "display_diff:Q",
                                title="Difference",
                                format="+.2f"
                            )
                        ]
                    ).properties(
                        height=340
                    )

                    st.altair_chart(
                        _trend_chart,
                        width="stretch"
                    )

                    _last_trend = _trend_records.iloc[-1].to_dict()
                    st.caption(
                        f"Latest ({int(_last_trend['year'])}): "
                        f"{selected_company} vs peer average difference "
                        f"{_last_trend['difference']:+.2f} "
                        f"({_trend_label}). Category peers only; years up "
                        f"to {selected_year}."
                    )

            # ----------------------------------------------------
            # Peer Position Scorecard (analytical context only).
            # ----------------------------------------------------

            st.markdown("**Peer Position Scorecard**")

            if not _variance_table:

                st.info(
                    "Scorecard is unavailable: no comparable values."
                )

            else:

                _scorecard_rows = []
                for row in _variance_rows:
                    kind = row.get("kind", "")
                    company_v = row.get("company")
                    peer_v = row.get("peer_avg")
                    diff_v = row.get("difference")

                    if kind == "money":
                        fmt = lambda v: f"{v:,.2f}"  # noqa: E731
                        diff_fmt = lambda v: f"{v:+.2f}"  # noqa: E731
                    elif kind == "pct":
                        fmt = lambda v: f"{v:.2f}%"  # noqa: E731
                        diff_fmt = lambda v: f"{v:+.2f}pp"  # noqa: E731
                    else:
                        fmt = lambda v: f"{v:.2f}x"  # noqa: E731
                        diff_fmt = lambda v: f"{v:+.2f}x"  # noqa: E731

                    _scorecard_rows.append({
                        "Metric": row["metric"],
                        "Company": fmt(company_v)
                        if company_v is not None else "N/A",
                        "Peer Average": fmt(peer_v)
                        if peer_v is not None else "N/A",
                        "Difference": diff_fmt(diff_v)
                        if diff_v is not None else "N/A",
                        "Position": peer_position(
                            row.get("vs_peer_pct"), kind),
                    })

                st.dataframe(
                    pd.DataFrame(_scorecard_rows),
                    width="stretch",
                    hide_index=True
                )
                st.caption(
                    "Position is the numerical relationship to the peer "
                    "average only. "
                    + " ".join(
                        f"{row['metric']}: {scorecard_context(row['metric'])}"
                        for row in _variance_rows
                    )
                    + " Analytical context, not investment advice."
                )

    # ============================================================
    # FINANCIAL INTELLIGENCE MAP
    # ============================================================

    st.markdown(
        '<div class="section-title"> Financial Intelligence Map</div>',
        unsafe_allow_html=True
    )

    map_data = build_financial_map(df, selected_company, selected_year)

    st.caption(
        "AI-powered view of how key financial metrics connect and influence overall performance."
    )

    roe_display = (
        f"{float(snapshot['roe']):.2f}%"
        if snapshot.get("roe") is not None
        else "N/A"
    )

    roa_display = (
        f"{float(snapshot['roa']):.2f}%"
        if snapshot.get("roa") is not None
        else "N/A"
    )

    roi_display = (
        f"{float(snapshot['roi']):.2f}%"
        if snapshot.get("roi") is not None
        else "N/A"
    )

    map_dot = f"""
    digraph G {{
        rankdir=LR;
        bgcolor="transparent";
        graph [pad="0.3", nodesep="0.5", ranksep="0.8"];

        node [
            shape=box,
            style="rounded,filled",
            fontname="Arial",
            fontsize=11,
            margin="0.18,0.12",
            fontcolor="white"
        ];

        edge [
            fontname="Arial",
            fontsize=9,
            color="#AAB4C0",
            fontcolor="#D6DCE3",
            penwidth=1.4
        ];

        revenue [
            label="Revenue\\n${float(snapshot['revenue']):,.0f}M",
            fillcolor="#163A2B"
        ];

        gross [
            label="Gross Profit\\n${float(snapshot['gross_profit']):,.0f}M",
            fillcolor="#173B4D"
        ];

        ebitda [
            label="EBITDA\\n${float(snapshot['ebitda']):,.0f}M",
            fillcolor="#403A17"
        ];

        net_income [
            label="Net Income\\n${float(snapshot['net_income']):,.0f}M",
            fillcolor="#3B1F3F"
        ];

        cashflow [
            label="Operating Cash Flow\\n${float(snapshot['operating_cash_flow']):,.0f}M",
            fillcolor="#173A35"
        ];

        ratios [
            label="Financial Health Metrics\\nNPM: {float(snapshot['net_profit_margin']):.2f}%\\nROE: {roe_display}\\nROA: {roa_display}\\nROI: {roi_display}",
            fillcolor="#25252D"
        ];

        liquidity [
            label="Liquidity & Leverage\\nCurrent Ratio: {float(snapshot['current_ratio']):.2f}x\\nDebt / Equity: {float(snapshot['debt_equity_ratio']):.2f}x",
            fillcolor="#25252D"
        ];

        revenue -> gross [label="profit generation"];
        gross -> ebitda [label="operating efficiency"];
        ebitda -> net_income [label="bottom-line impact"];
        net_income -> cashflow [label="cash conversion"];
        net_income -> ratios [label="performance"];
        liquidity -> net_income [label="financial context"];
    }}
    """

    st.graphviz_chart(map_dot, width="stretch")

    st.info(
        " **AI Insight:** The map connects operating performance, profitability, "
        "cash generation, liquidity and leverage so reviewers can investigate "
        "the financial story rather than looking at isolated numbers."
    )


with tab_simulate:
    st.markdown(
        '<div class="fs-eyebrow">Scenario Analysis</div>'
        '<div class="fs-hero">Test your assumptions.</div>'
        '<div class="fs-lede">Explore how changes in key assumptions '
        'affect the financial profile.</div>',
        unsafe_allow_html=True
    )
    # ============================================================
    # WHAT-IF SCENARIO ANALYSIS
    # ============================================================

    st.markdown(
        '<div class="section-title"> What-If Scenario Analysis</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Explore how changes in revenue growth and profitability could affect "
        "the company's financial outcome."
    )

    scenario_col1, scenario_col2 = st.columns(2)

    with scenario_col1:
        revenue_growth_input = st.slider(
            "Revenue Growth (%)",
            min_value=-30.0,
            max_value=50.0,
            value=10.0,
            step=1.0,
            key="scenario_revenue_growth"
        )

    with scenario_col2:
        margin_input = st.slider(
            "Target Net Profit Margin (%)",
            min_value=5.0,
            max_value=50.0,
            value=30.0,
            step=1.0,
            key="scenario_margin"
        )

    current_revenue = float(snapshot["revenue"])
    current_net_income = float(snapshot["net_income"])
    current_margin = float(snapshot["net_profit_margin"])

    def format_scenario_money(value):
        if selected_currency == "INR":
            # Source data is USD million.
            # Convert to INR million, then display as lakh crore.
            inr_lakh_crore = (value * USD_TO_INR) / 1_000_000
            return f"₹{inr_lakh_crore:,.2f} Lakh Cr"
        else:
            usd_billion = value / 1000
            return f"${usd_billion:,.2f}B"

    projected_revenue = current_revenue * (1 + revenue_growth_input / 100)

    projected_net_income = projected_revenue * (margin_input / 100)

    income_change = projected_net_income - current_net_income
    income_change_pct = (
        (income_change / current_net_income) * 100
        if current_net_income != 0
        else 0
    )

    st.markdown("**Scenario Impact**")

    scenario_metric1, scenario_metric2, scenario_metric3 = st.columns(3)

    with scenario_metric1:
        st.metric(
            "Projected Revenue",
            format_scenario_money(projected_revenue),
            f"{revenue_growth_input:+.0f}%"
        )

    with scenario_metric2:
        st.metric(
            "Projected Net Income",
            format_scenario_money(projected_net_income),
            f"{income_change_pct:+.2f}%"
        )

    with scenario_metric3:
        st.metric(
            "Projected Net Margin",
            f"{margin_input:.2f}%",
            f"{margin_input - current_margin:+.2f} pp"
        )

    scenario_display = pd.DataFrame({
        "Metric": [
            "Revenue",
            "Net Income",
            "Net Profit Margin"
        ],
        "Current": [
            format_scenario_money(current_revenue),
            format_scenario_money(current_net_income),
            f"{current_margin:.2f}%"
        ],
        "What-If Scenario": [
            format_scenario_money(projected_revenue),
            format_scenario_money(projected_net_income),
            f"{margin_input:.2f}%"
        ]
    })

    st.dataframe(
        scenario_display,
        width="stretch",
        hide_index=True
    )

    st.info(
        " **Scenario Note:** This is a what-if analysis based on the selected "
        "assumptions, not a financial forecast."
    )


with tab_report:
    st.markdown(
        '<div class="fs-eyebrow">Financial Report</div>'
        '<div class="fs-hero">Read the analyst review.</div>'
        '<div class="fs-lede">AI-assisted financial review and '
        'historical analysis.</div>',
        unsafe_allow_html=True
    )
    # ============================================================
    # GENERATE FULL FINANCIAL REPORT (professional multi-page PDF)
    # ============================================================

    st.markdown(
        '<div class="section-title">Full Financial Report</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Generate a professional multi-page PDF research report for the "
        "selected company and year — overview, performance, cash flow, "
        "balance sheet, investigation, risk, benchmark, simulation, "
        "AI review and the FinSight AI chat transcript."
    )

    from engine.pdf_report import (
        build_report_context,
        generate_financial_report,
        report_filename,
    )

    _chat_history = st.session_state.get("finsight_chat_history", [])

    _pdf_settings_key = (
        str(selected_company),
        int(selected_year),
        str(selected_currency),
        str(uploaded_file.name) if uploaded_file is not None else "default",
        len(_chat_history),
        float(st.session_state.get("scenario_revenue_growth", 10.0)),
        float(st.session_state.get("scenario_margin", 30.0)),
    )

    if st.button(
        "Generate Full Financial Report",
        key="finsight_generate_pdf",
        type="primary",
    ):
        with st.spinner("Building your financial research report..."):
            try:
                _pdf_context = build_report_context(
                    df,
                    selected_company,
                    selected_year,
                    currency=selected_currency,
                    chat_history=_chat_history,
                    scenario={
                        "revenue_growth": float(
                            st.session_state.get(
                                "scenario_revenue_growth", 10.0)),
                        "target_margin": float(
                            st.session_state.get("scenario_margin", 30.0)),
                    },
                    data_source=(
                        f"uploaded file ({uploaded_file.name})"
                        if uploaded_file is not None
                        else "default dataset (data/clean_data.csv)"
                    ),
                )
                st.session_state["finsight_pdf_bytes"] = \
                    generate_financial_report(_pdf_context)
                st.session_state["finsight_pdf_name"] = report_filename(
                    selected_company, selected_year)
                st.session_state["finsight_pdf_key"] = _pdf_settings_key
                st.success(
                    "Report generated. Use the download button below to "
                    "save your PDF."
                )
            except Exception as pdf_error:
                st.error(f"Report generation failed: {pdf_error}")

    if (
        st.session_state.get("finsight_pdf_bytes")
        and st.session_state.get("finsight_pdf_key") == _pdf_settings_key
    ):
        st.download_button(
            label="Download Full Financial Report",
            data=st.session_state["finsight_pdf_bytes"],
            file_name=st.session_state.get(
                "finsight_pdf_name", "FinSight_AI_Report.pdf"),
            mime="application/pdf",
            key="finsight_download_pdf",
        )
    elif st.session_state.get("finsight_pdf_bytes"):
        st.caption(
            "Settings changed since the last report was built — "
            "generate again to refresh the PDF."
        )

    # ============================================================
    # AI REVIEW REPORT
    # ============================================================

    st.markdown(
        '<div class="section-title"> AI Review Report</div>',
        unsafe_allow_html=True
    )

    report = generate_investigation_report(
        analysis_df,
        selected_company,
        selected_year
    )

    st.caption(
        "Evidence-backed review observations generated from the selected financial period."
    )

    report_col1, report_col2, report_col3 = st.columns(3)

    with report_col1:
        primary_driver_text = report.get(
            "primary_driver",
            "No major driver"
        )

        st.markdown(
            f"""
            <div style="
                background-color: #111827;
                border: 1px solid #374151;
                border-radius: 10px;
                padding: 14px 16px;
                min-height: 82px;
            ">
                <div style="
                    font-size: 14px;
                    color: #9CA3AF;
                    margin-bottom: 8px;
                ">
                    Primary Driver
                </div>
                <div style="
                    font-size: 20px;
                    font-weight: 600;
                    line-height: 1.25;
                    color: #F9FAFB;
                    word-wrap: break-word;
                ">
                    {primary_driver_text}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with report_col2:
        drivers = report.get("drivers", [])
        st.metric(
            "Drivers Detected",
            len(drivers) if isinstance(drivers, list) else 0
        )

    with report_col3:
        evidence_data = report.get("evidence", {})
        evidence_count = (
            evidence_data.get("evidence_count", 0)
            if isinstance(evidence_data, dict)
            else 0
        )
        st.metric(
            "Evidence Points",
            evidence_count
        )

    st.markdown("**Recommendation**")

    recommendation = report.get(
        "recommendation",
        "No specific recommendation generated for this period."
    )

    st.info(
        f" **Observation:** {recommendation}"
    )

    def format_report_money(value):
        if value is None:
            return "—"

        value = float(value)

        if selected_currency == "INR":
            inr_lakh_crore = (value * USD_TO_INR) / 1_000_000
            return f"₹{inr_lakh_crore:,.2f} Lakh Cr"
        else:
            usd_billion = value / 1000
            return f"${usd_billion:,.2f}B"


    def format_report_value(metric, value):
        if value is None:
            return "—"

        value = float(value)

        if metric in ["Revenue", "Net Income"]:
            return format_report_money(value)

        if metric in ["Gross Margin", "EBITDA Margin"]:
            return f"{value:,.2f}%"

        if metric in ["Current Ratio", "Debt/Equity"]:
            return f"{value:,.2f}x"

        return f"{value:,.2f}"


    evidence_data = report.get("evidence", {})
    evidence_items = (
        evidence_data.get("changes", [])
        if isinstance(evidence_data, dict)
        else []
    )

    if evidence_items:
        st.markdown("**Supporting Evidence**")

        report_evidence = []

        for item in evidence_items:
            metric = item.get("metric", "Financial Metric")
            current_value = item.get("current")
            previous_value = item.get("previous")
            change_value = item.get("change")
            unit = item.get("unit", "")
            evidence_type = item.get("evidence_type", "Financial Analysis")

            if unit == "%":
                change_text = f"{float(change_value):+.2f}%"
            elif unit == "percentage points":
                change_text = f"{float(change_value):+.2f} pp"
            elif unit == "x":
                change_text = f"{float(change_value):+.2f}x"
            else:
                change_text = f"{float(change_value):+.2f}"

            report_evidence.append({
                "Metric": metric,
                "Current": format_report_value(metric, current_value),
                "Previous": format_report_value(metric, previous_value),
                "Change": change_text,
                "Evidence Type": evidence_type
            })

        st.dataframe(
            pd.DataFrame(report_evidence),
            width="stretch",
            hide_index=True
        )
    else:
        st.success(
            "No supporting evidence points were identified for the selected period."
        )

    st.caption(
        " AI-generated review support. Conclusions should be validated against "
        "the underlying financial statements and source documents."
    )

    # ============================================================
    # PERFORMANCE TRENDS
    # ============================================================

    st.markdown(
        '<div class="section-title">Performance Trends</div>',
        unsafe_allow_html=True
    )


    chart_col1, chart_col2 = st.columns(2)


    # ------------------------------------------------------------
    # REVENUE
    # ------------------------------------------------------------

    with chart_col1:

        st.write("### Revenue Trend")

        revenue_data = trend[
            ["year", "revenue"]
        ].copy()

        revenue_data["display_value"] = convert_chart_values(
            revenue_data["revenue"]
        )

        revenue_chart = alt.Chart(
            revenue_data
        ).mark_line(
            point=True
        ).encode(

            x=alt.X(
                "year:O",
                title="Year"
            ),

            y=alt.Y(
                "display_value:Q",
                title=(
                    "INR Lakh Crore"
                    if selected_currency == "INR"
                    else "USD Billion"
                )
            ),

            tooltip=[
                alt.Tooltip(
                    "year:O",
                    title="Year"
                ),
                alt.Tooltip(
                    "display_value:Q",
                    title=(
                        "₹ Lakh Cr"
                        if selected_currency == "INR"
                        else "$ Billion"
                    ),
                    format=",.2f"
                )
            ]
        ).properties(
            height=350
        )

        st.altair_chart(
            revenue_chart,
            width="stretch"
        )


    # ------------------------------------------------------------
    # NET INCOME
    # ------------------------------------------------------------

    with chart_col2:

        st.write("### Net Income Trend")

        income_data = trend[
            ["year", "net_income"]
        ].copy()

        income_data["display_value"] = convert_chart_values(
            income_data["net_income"]
        )

        income_chart = alt.Chart(
            income_data
        ).mark_line(
            point=True
        ).encode(

            x=alt.X(
                "year:O",
                title="Year"
            ),

            y=alt.Y(
                "display_value:Q",
                title=(
                    "INR Lakh Crore"
                    if selected_currency == "INR"
                    else "USD Billion"
                )
            ),

            tooltip=[
                alt.Tooltip(
                    "year:O",
                    title="Year"
                ),
                alt.Tooltip(
                    "display_value:Q",
                    title=(
                        "₹ Lakh Cr"
                        if selected_currency == "INR"
                        else "$ Billion"
                    ),
                    format=",.2f"
                )
            ]
        ).properties(
            height=350
        )

        st.altair_chart(
            income_chart,
            width="stretch"
        )


    # ============================================================
    # GROWTH PERFORMANCE — Revenue YoY vs Net Income YoY
    # Uses existing engine YoY fields; percentages stay
    # percentages regardless of selected_currency.
    # ============================================================

    from engine.visuals import (
        get_growth_history,
        get_cash_flow_history,
    )

    report_metrics = calculate_financial_metrics(
        df.copy()
    )

    st.write("### Growth Performance")

    growth = get_growth_history(
        report_metrics,
        selected_company,
        selected_year
    )

    if not growth["available"]:

        st.info(
            "Revenue and net income growth history is unavailable "
            "for the selected company and year."
        )

    else:

        growth_chart = alt.Chart(
            growth["history"]
        ).mark_line(
            point=True
        ).encode(

            x=alt.X(
                "year:O",
                title="Year"
            ),

            y=alt.Y(
                "growth:Q",
                title="Growth (%)"
            ),

            color=alt.Color(
                "metric:N",
                title="Metric",
                sort=["Revenue Growth", "Net Income Growth"]
            ),

            tooltip=[
                alt.Tooltip(
                    "year:O",
                    title="Year"
                ),
                alt.Tooltip(
                    "metric:N",
                    title="Metric"
                ),
                alt.Tooltip(
                    "growth:Q",
                    title="Growth (%)",
                    format="+.2f"
                )
            ]
        ).properties(
            height=350
        )

        st.altair_chart(
            growth_chart,
            width="stretch"
        )

        if growth.get("insight"):
            st.caption(growth["insight"])


    # ============================================================
    # PROFITABILITY ANALYSIS
    # ============================================================

    st.markdown(
        '<div class="section-title">Profitability Analysis</div>',
        unsafe_allow_html=True
    )

    margin_data = trend[
        [
            "year",
            "gross_margin",
            "ebitda_margin",
            "net_profit_margin"
        ]
    ].copy()

    margin_long = margin_data.melt(
        id_vars="year",
        value_vars=[
            "gross_margin",
            "ebitda_margin",
            "net_profit_margin"
        ],
        var_name="metric",
        value_name="margin"
    )

    metric_names = {
        "gross_margin": "Gross Margin",
        "ebitda_margin": "EBITDA Margin",
        "net_profit_margin": "Net Profit Margin"
    }

    margin_long["metric"] = (
        margin_long["metric"]
        .map(metric_names)
    )


    margin_chart = alt.Chart(
        margin_long
    ).mark_line(
        point=True
    ).encode(

        x=alt.X(
            "year:O",
            title="Year"
        ),

        y=alt.Y(
            "margin:Q",
            title="Margin (%)"
        ),

        color=alt.Color(
            "metric:N",
            title="Metric"
        ),

        tooltip=[
            alt.Tooltip(
                "year:O",
                title="Year"
            ),
            alt.Tooltip(
                "metric:N",
                title="Metric"
            ),
            alt.Tooltip(
                "margin:Q",
                title="Margin",
                format=".2f"
            )
        ]
    ).properties(
        height=400
    )

    st.altair_chart(
        margin_chart,
        width="stretch"
    )


    # ============================================================
    # CASH FLOW ANALYSIS — engine cash-flow fields only.
    # Free cash flow shown only when the dataset provides it.
    # ============================================================

    st.markdown(
        '<div class="section-title">Cash Flow Analysis</div>',
        unsafe_allow_html=True
    )

    cash_flow = get_cash_flow_history(
        report_metrics,
        selected_company,
        selected_year
    )

    if not cash_flow["available"]:

        if cash_flow.get("missing"):
            st.info(
                "Cash flow analysis is not available: "
                f"{', '.join(cash_flow['missing'])} "
                "was not provided in the dataset."
            )
        else:
            st.info(
                "Cash flow values are not available for the "
                "selected company and year."
            )

    else:

        cash_data = cash_flow["history"].copy()
        cash_data["display_value"] = convert_chart_values(
            cash_data["value"]
        )

        cash_chart = alt.Chart(
            cash_data
        ).mark_line(
            point=True
        ).encode(

            x=alt.X(
                "year:O",
                title="Year"
            ),

            y=alt.Y(
                "display_value:Q",
                title=(
                    "INR Lakh Crore"
                    if selected_currency == "INR"
                    else "USD Billion"
                )
            ),

            color=alt.Color(
                "flow:N",
                title="Cash Flow",
                sort=[
                    "Operating Cash Flow",
                    "Investing Cash Flow",
                    "Financing Cash Flow",
                    "Net Cash Flow"
                ]
            ),

            tooltip=[
                alt.Tooltip(
                    "year:O",
                    title="Year"
                ),
                alt.Tooltip(
                    "flow:N",
                    title="Cash Flow"
                ),
                alt.Tooltip(
                    "display_value:Q",
                    title=(
                        "₹ Lakh Cr"
                        if selected_currency == "INR"
                        else "$ Billion"
                    ),
                    format=",.2f"
                )
            ]
        ).properties(
            height=400
        )

        st.altair_chart(
            cash_chart,
            width="stretch"
        )

        latest_cash = cash_flow.get("latest", {})
        cash_cards = []

        if "operating" in latest_cash:
            cash_cards.append(
                (
                    "Latest Operating Cash Flow",
                    format_money(latest_cash["operating"])
                )
            )
        if "net" in latest_cash:
            cash_cards.append(
                (
                    "Latest Net Cash Flow",
                    format_money(latest_cash["net"])
                )
            )
        if cash_flow.get("margin_available"):
            cash_cards.append(
                (
                    "Operating Cash Flow Margin",
                    f"{latest_cash['operating_margin']:.2f}%"
                )
            )
        if (
            cash_flow.get("free_cash_flow_available")
            and "free_cash_flow" in latest_cash
        ):
            cash_cards.append(
                (
                    "Latest Free Cash Flow",
                    format_money(latest_cash["free_cash_flow"])
                )
            )

        if cash_cards:
            cash_cols = st.columns(len(cash_cards))

            for col, (label, value) in zip(cash_cols, cash_cards):
                with col:
                    st.metric(label, value)


    # ============================================================
    # RECENT PERFORMANCE
    # ============================================================

    st.markdown(
        '<div class="section-title">Recent Financial Performance</div>',
        unsafe_allow_html=True
    )

    recent = trend.tail(5).copy()

    recent_display = recent[
        [
            "year",
            "revenue",
            "net_income",
            "ebitda",
            "gross_margin",
            "ebitda_margin",
            "net_profit_margin"
        ]
    ].copy()

    # Convert monetary columns for display
    recent_display["revenue"] = convert_chart_values(
        recent_display["revenue"]
    )

    recent_display["net_income"] = convert_chart_values(
        recent_display["net_income"]
    )

    recent_display["ebitda"] = convert_chart_values(
        recent_display["ebitda"]
    )

    recent_display.columns = [
        "Year",
        f"Revenue ({selected_currency})",
        f"Net Income ({selected_currency})",
        f"EBITDA ({selected_currency})",
        "Gross Margin %",
        "EBITDA Margin %",
        "Net Profit Margin %"
    ]

    st.dataframe(
        recent_display,
        width="stretch",
        hide_index=True
    )

# ============================================================
# FINSIGHT AI CHAT
# ============================================================

with tab_chat:
    st.markdown(
        '<div class="fs-chat-hero">'
        '<div class="fs-eyebrow">Financial Intelligence</div>'
        '<div class="fs-hero">AI CHAT</div>'
        '<div class="fs-lede">Ask questions about your company&rsquo;s '
        'financial performance, risks, trends and benchmarks.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    from engine.chat_engine import execute_financial_query

    from engine.llm import (
        build_chat_evidence,
        generate_financial_answer,
    )

    from engine.report_engine import (
        build_historical_report,
        build_comparison_report,
        report_to_txt_bytes
    )

    if "finsight_chat_history" not in st.session_state:
        st.session_state.finsight_chat_history = []

    from engine.llm import resolve_provider

    provider_map = {
        "Auto": "auto",
        "Gemini": "gemini",
        "OpenAI": "openai",
        "Anthropic": "anthropic",
        "Grok": "grok",
        "Deterministic only": "demo",
    }
    ai_provider_code = provider_map.get(
        st.session_state.get("finsight_ai_provider", "Auto"), "auto"
    )

    # Model picked in the panel for the effective provider (if any).
    effective_provider = (
        ai_provider_code
        if ai_provider_code != "auto"
        else resolve_provider("auto")
    )
    model_override = (
        st.session_state.get(f"finsight_model_{effective_provider}")
        if effective_provider not in ("demo", "none")
        else None
    )

    # Period label for the prompt action row + provider panel.
    try:
        _earliest_chat_year = int(df.loc[
            df["company"].astype(str) == str(selected_company), "year"
        ].min())
    except Exception:
        _earliest_chat_year = selected_year

    _chat_period_label = f"{_earliest_chat_year}–{selected_year}"

    # Frontend helper: runs a question through the EXISTING chat
    # pipeline (deterministic query + optional grounded AI answer).
    # Suggestion cards and the Analyze button share this one path —
    # no second AI pipeline exists.
    def _ask_finsight(question):

        chat_financial_df = calculate_financial_metrics(
            df.copy()
        )

        result = execute_financial_query(
            chat_financial_df,
            selected_company,
            question
        )

        # Optional live LLM answer grounded in engine output.
        # A relevance-grounded evidence object is built from
        # already-computed engine results (dataset, snapshot,
        # mentioned years, metric histories, health, benchmark,
        # anomalies, quality) — the model explains, never computes.
        # The call runs unless the user explicitly chose
        # "Deterministic only": with no key it returns an instant
        # offline DEMO hint (also rendered), so some AI text
        # always appears.
        _explicit_demo = (
            st.session_state.get("finsight_ai_provider", "Auto")
            == "Deterministic only"
        )
        if not _explicit_demo:
            try:
                with st.spinner("Generating AI answer..."):
                    from engine.ai_context import (
                        build_financial_ai_context,
                    )
                    from engine.risk import (
                        calculate_health_factor_scores,
                    )

                    try:
                        _health_rows = build_health_risk_scores(
                            chat_financial_df, selected_year)
                        _health_sel = _health_rows[
                            _health_rows["company"] == selected_company]
                        _health = {
                            "score": float(
                                _health_sel.iloc[0]["health_score"]),
                            "risk": float(
                                _health_sel.iloc[0]["risk_score"]),
                            "status": str(
                                _health_sel.iloc[0][
                                    "health_classification"]),
                        } if not _health_sel.empty else {}
                    except Exception:
                        _health = {}

                    try:
                        _factors_row = chat_financial_df[
                            (chat_financial_df["company"].astype(str)
                             == str(selected_company))
                            & (chat_financial_df["year"].astype(int)
                               == int(selected_year))
                        ]
                        _factors = calculate_health_factor_scores(
                            _factors_row.iloc[0]
                        ) if not _factors_row.empty else {}
                    except Exception:
                        _factors = {}

                    try:
                        _bench_df = build_peer_benchmark(
                            df, selected_year)
                        _bench_sel = _bench_df[
                            _bench_df["company"] == selected_company]
                        _benchmark = {
                            "peer_count": int(
                                _bench_sel.iloc[0]["peer_count"]),
                            "category": str(
                                _bench_sel.iloc[0].get("category", "")),
                            "row": _bench_sel.iloc[0].to_dict(),
                        } if not _bench_sel.empty else {}
                    except Exception:
                        _benchmark = {}

                    try:
                        _anoms = detect_anomalies(chat_financial_df)
                    except Exception:
                        _anoms = pd.DataFrame()

                    try:
                        _dq = analyze_data_quality(df)
                        _quality = {
                            "status": _dq.get("status"),
                            "missing": _dq.get("missing_values", 0),
                            "duplicates": _dq.get(
                                "duplicate_records", 0),
                            "invalid": _dq.get(
                                "numeric_anomalies", 0),
                        }
                    except Exception:
                        _quality = {}

                    try:
                        _snap = get_company_snapshot(
                            df, selected_company, selected_year)
                    except Exception:
                        _snap = {}

                    fin_context = build_financial_ai_context(
                        question,
                        company=selected_company,
                        selected_year=selected_year,
                        currency=selected_currency,
                        metrics_df=chat_financial_df,
                        snapshot=_snap,
                        health=_health,
                        factors=_factors,
                        benchmark=_benchmark,
                        anomalies=_anoms,
                        quality=_quality,
                        data_source=(
                            f"uploaded file "
                            f"({st.session_state.get('upload_filename', 'upload')})"
                            if uploaded_file is not None
                            else "default dataset (data/clean_data.csv)"
                        ),
                        deterministic_answer=result.get("answer", ""),
                        deterministic_summary=result.get("summary"),
                    )

                    ai_reply = generate_financial_answer(
                        question,
                        {"evidence": fin_context,
                         "deterministic": {
                             "answer": result.get("answer", ""),
                             "summary": result.get("summary"),
                         }},
                            provider=ai_provider_code,
                            model=model_override,
                        )
                    result["ai_explanation"] = ai_reply
            except Exception as ai_error:
                    result["ai_explanation"] = {
                        "status": "ERROR",
                        "provider": ai_provider_code,
                        "response": f"AI explanation unavailable: {ai_error}",
                    }

        st.session_state.finsight_chat_history.append(
            {
                "question": question,
                "result": result
            }
        )

    if st.session_state.finsight_chat_history:

        st.markdown("### Conversation")

        for chat_index, chat_item in enumerate(
            reversed(
                st.session_state.finsight_chat_history
            )
        ):

            question = chat_item["question"]
            result = chat_item["result"]

            st.markdown(
                f'<span class="fs-role fs-role-you">YOU</span><br>{question}'
            )

            status = result.get(
                "status",
                "UNKNOWN"
            )

            if status == "SUCCESS":

                st.markdown(
                    f'<span class="fs-role fs-role-ai">FINSIGHT AI</span><br>'
                    f"{result.get('answer', '')}"
                )

                ai_explanation = result.get("ai_explanation")

                if isinstance(ai_explanation, dict):
                    ai_state = ai_explanation.get("status")

                    if ai_state == "SUCCESS":
                        st.markdown(
                            '<span class="fs-role fs-role-ai">AI ANSWER</span>',
                        )
                        st.markdown(
                            ai_explanation.get("response", "")
                        )
                        st.caption(
                            f"AI answer grounded in engine values "
                            f"({ai_explanation.get('provider', 'ai')}:"
                            f"{ai_explanation.get('model', '')})"
                        )
                    elif ai_state == "ERROR":
                        st.caption(
                            f"AI explanation unavailable: "
                            f"{ai_explanation.get('response', '')}"
                        )
                    elif ai_state == "DEMO":
                        st.caption(
                            "Add a provider key in the sidebar "
                            "(AI provider keys) for live AI answers "
                            "grounded in this analysis."
                        )

                data = result.get("data")

                if (
                    data is not None
                    and not data.empty
                ):

                    st.dataframe(
                        data,
                        width="stretch",
                        hide_index=True
                    )

                report_type = result.get(
                    "report_type"
                )

                report_text = None

                if report_type == "historical":

                    report_text = build_historical_report(
                        result
                    )

                elif report_type == "comparison":

                    report_text = build_comparison_report(
                        result
                    )

                if report_text:

                    report_bytes = report_to_txt_bytes(
                        report_text
                    )

                    st.download_button(
                        label="Download Report",
                        data=report_bytes,
                        file_name=(
                            f"FinSight_{selected_company}_"
                            f"financial_report_{chat_index}.txt"
                        ),
                        mime="text/plain",
                        key=(
                            f"chat_report_"
                            f"{chat_index}"
                        )
                    )

            elif status == "UNAVAILABLE":

                ai_explanation = result.get("ai_explanation")

                if (
                    isinstance(ai_explanation, dict)
                    and ai_explanation.get("status") == "SUCCESS"
                ):
                    st.markdown(
                        f'<span class="fs-role fs-role-ai">FINSIGHT AI</span><br>'
                        f"{ai_explanation.get('response', '')}"
                    )
                    st.caption(
                        f"Grounded AI answer "
                        f"({ai_explanation.get('provider', 'ai')}:"
                        f"{ai_explanation.get('model', '')}) "
                            f"based on {selected_company} engine values."
                    )
                else:
                    st.info(
                        result.get(
                            "answer",
                            "The requested information is unavailable."
                        )
                    )

            else:

                ai_explanation = result.get("ai_explanation")

                if (
                    isinstance(ai_explanation, dict)
                    and ai_explanation.get("status") == "SUCCESS"
                ):
                    st.markdown(
                        f'<span class="fs-role fs-role-ai">FINSIGHT AI</span><br>'
                        f"{ai_explanation.get('response', '')}"
                    )
                    st.caption(
                        f"Grounded AI answer "
                        f"({ai_explanation.get('provider', 'ai')}:"
                        f"{ai_explanation.get('model', '')}) "
                            f"based on {selected_company} engine values."
                    )
                else:
                    st.warning(
                        result.get(
                            "answer",
                            "I could not process that question."
                        )
                    )

            st.divider()

    else:

        st.info(
            "Start by asking a question about "
            f"{selected_company}."
        )

    # Composer bar: clear | input | provider | model | send in one
    # row. Same widgets, keys and pipeline as before — layout only.
    _model_default = get_llm_status()["providers"].get(
        effective_provider, {}).get("model")
    _model_options = (
        st.session_state.get(f"finsight_models_{effective_provider}")
        or ([_model_default] if _model_default else ["Default"])
    )
    _model_key = f"finsight_model_{effective_provider}"
    if st.session_state.get(_model_key) not in _model_options:
        st.session_state[_model_key] = _model_options[0]

    st.markdown(
        '<div class="fs-composer-bar"></div>',
        unsafe_allow_html=True
    )

    _cc = st.columns([0.55, 4.8, 1.7, 1.7, 0.65])

    with _cc[0]:
        if st.button(
            "+",
            key="finsight_chat_clear",
            help="Clear draft",
        ):
            st.session_state["finsight_chat_question"] = ""

    with _cc[1]:
        chat_question = st.text_input(
            "Your question",
            placeholder="Ask anything",
            key="finsight_chat_question",
            label_visibility="collapsed",
        )

    with _cc[2]:
        st.selectbox(
            "Provider",
            [
                "Auto",
                "Gemini",
                "OpenAI",
                "Anthropic",
                "Grok",
                "Deterministic only",
            ],
            key="finsight_ai_provider",
            label_visibility="collapsed",
            help="AI provider (Auto uses the first configured).",
        )

    with _cc[3]:
        st.selectbox(
            "Model",
            _model_options,
            key=_model_key,
            label_visibility="collapsed",
            help="Model for the selected provider.",
        )

    with _cc[4]:
        ask_button = st.button(
            "↑",
            key="finsight_chat_analyze",
            help="Send question",
        )

    st.caption(
        f"+ Financial context active · {selected_company} · {_chat_period_label}"
    )

    if ask_button:

        if not chat_question.strip():

            st.warning(
                "Please enter a financial question."
            )

        else:

            _ask_finsight(chat_question.strip())


# ============================================================
# 115-FEATURE FINANCIAL INTELLIGENCE UI
# ============================================================

def render_feature_intelligence(financial_df):

    from engine.feature_intelligence import build_feature_analysis

    st.markdown("---")
    st.header("Feature Intelligence")
    st.caption(
        "Detailed financial analysis across 115 financial, "
        "operational, growth and risk features."
    )

    feature_result = build_feature_analysis(financial_df)

    analysis_df = feature_result["analysis"].copy()

    total = feature_result["total_features"]
    available = feature_result["available_features"]
    calculated = feature_result["calculated_features"]
    unavailable = feature_result["unavailable_features"]

    # --------------------------------------------------------
    # Coverage summary
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Features",
            total
        )

    with col2:
        st.metric(
            "Available",
            available
        )

    with col3:
        st.metric(
            "Calculated",
            calculated
        )

    with col4:
        st.metric(
            "Not Provided",
            unavailable
        )

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    st.markdown("### Explore Financial Features")

    col1, col2 = st.columns(2)

    with col1:

        categories = [
            "All Categories"
        ] + sorted(
            analysis_df["category"].dropna().unique().tolist()
        )

        selected_category = st.selectbox(
            "Category",
            categories,
            key="feature_intelligence_category"
        )

    with col2:

        status_options = [
            "All Status",
            "Available",
            "Calculated",
            "Not provided"
        ]

        selected_status = st.selectbox(
            "Data Status",
            status_options,
            key="feature_intelligence_status"
        )

    search_text = st.text_input(
        "Search feature",
        placeholder="Example: revenue, margin, debt, cash flow..."
    )

    filtered_df = analysis_df.copy()

    if selected_category != "All Categories":

        filtered_df = filtered_df[
            filtered_df["category"] == selected_category
        ]

    if selected_status != "All Status":

        filtered_df = filtered_df[
            filtered_df["status"] == selected_status
        ]

    if search_text.strip():

        search_lower = search_text.strip().lower()

        filtered_df = filtered_df[
            filtered_df["name"]
            .str.lower()
            .str.contains(
                search_lower,
                na=False
            )
        ]

    # --------------------------------------------------------
    # Category sections
    # --------------------------------------------------------

    category_order = [
        "Income Statement",
        "Profitability",
        "Balance Sheet",
        "Liquidity",
        "Leverage",
        "Cash Flow",
        "Efficiency",
        "Market & Per Share",
        "Growth & Trend",
        "Financial Quality & Risk",
        "Business Scale",
        "Macroeconomic Context"
    ]

    for category in category_order:

        category_df = filtered_df[
            filtered_df["category"] == category
        ].copy()

        if category_df.empty:
            continue

        with st.expander(
            f"{category} ({len(category_df)} features)",
            expanded=True
        ):

            for _, feature in category_df.iterrows():

                feature_name = feature["name"]
                value = feature["latest_value"]
                unit = feature["unit"]
                status = feature["status"]
                analysis = feature["description"]

                if status == "Not provided":

                    st.markdown(
                        f"**{feature_name}**"
                    )

                    st.caption(
                        "Not provided in the uploaded financial data."
                    )

                    st.divider()

                    continue

                col1, col2, col3 = st.columns(
                    [2.2, 1.3, 3.5]
                )

                with col1:

                    st.markdown(
                        f"**{feature_name}**"
                    )

                    st.caption(
                        feature["description"]
                    )

                with col2:

                    st.metric(
                        "Value",
                        feature["Value"]
                        if "Value" in feature
                        else str(value)
                    )

                    st.caption(
                        status
                    )

                with col3:

                    st.markdown(
                        "**Analysis**"
                    )

                    st.write(
                        generate_feature_interpretation(
                            feature_name,
                            value,
                            unit
                        )
                    )

                st.divider()

    # --------------------------------------------------------
    # Full table
    # --------------------------------------------------------

    st.markdown("### Complete Feature Analysis")

    display_df = filtered_df.copy()

    if not display_df.empty:

        table_df = display_df[
            [
                "name",
                "category",
                "latest_value",
                "unit",
                "status",
                "description"
            ]
        ].copy()

        table_df.columns = [
            "Feature",
            "Category",
            "Value",
            "Unit",
            "Status",
            "Description"
        ]

        st.dataframe(
            table_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "No features match the selected filters."
        )

    return feature_result














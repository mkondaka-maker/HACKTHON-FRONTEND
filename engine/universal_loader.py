
import re
import pandas as pd
import numpy as np

REQUIRED_FIELDS = [
    "year",
    "company",
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

OPTIONAL_FIELDS = [
    "roe",
    "roa",
    "roi",
    "earning_per_share",
    "free_cash_flow_per_share",
    "return_on_tangible_equity",
    "market_capin_b_usd",
    "number_of_employees",
    "inflation_ratein_us",
    "category",
]

ALIASES = {
    'year': ['year', 'fiscal year', 'fiscal_year', 'financial year', 'financial_year', 'fy', 'reporting year', 'period'],
    'company': ['company', 'company name', 'company_name', 'corporation', 'corporation name', 'business name', 'entity', 'issuer', 'name'],
    'category': ['category', 'industry', 'sector', 'industry category', 'business category', 'company category'],
    'revenue': ['revenue', 'sales', 'turnover', 'total revenue', 'operating revenue', 'net sales', 'sales revenue', 'Revenue'],
    'gross_profit': ['gross profit', 'gross_profit', 'gross profit amount', 'gross income', 'gross earnings', 'Gross Profit'],
    'net_income': ['net income', 'net_income', 'net profit', 'net_profit', 'pat', 'profit after tax', 'profit after taxes', 'net earnings', 'net earnings attributable', 'Net Income'],
    'ebitda': ['ebitda', 'ebitda amount', 'earnings before interest tax depreciation amortization', 'EBITDA'],
    'share_holder_equity': ['share holder equity', 'shareholder equity', 'shareholders equity', 'share_holder_equity', 'total equity', 'stockholders equity', 'stockholder equity', 'equity', 'Shareholder Equity', "shareholders' equity", "stockholders' equity"],
    'cash_flow_from_operating': ['cash flow from operating', 'cash_flow_from_operating', 'operating cash flow', 'cash from operating activities', 'net cash from operating activities', 'cfo', 'Operating Cash Flow', 'cash flow from operations'],
    'cash_flow_from_investing': ['cash flow from investing', 'cash_flow_from_investing', 'investing cash flow', 'cash from investing activities', 'net cash from investing activities', 'cfi', 'Investing Cash Flow', 'cash flow from investments'],
    'cash_flow_from_financial_activities': ['cash flow from financial activities', 'cash_flow_from_financial_activities', 'financing cash flow', 'cash flow from financing', 'cash from financing activities', 'net cash from financing activities', 'cff', 'Financing Cash Flow'],
    'current_ratio': ['current ratio', 'current_ratio', 'current assets/current liabilities', 'Current Ratio'],
    'debt_equity_ratio': ['debt equity ratio', 'debt_equity_ratio', 'debt to equity', 'debt-to-equity', 'debt/equity', 'd/e ratio', 'de ratio', 'Debt to Equity Ratio'],
    'roe': ['roe', 'return on equity', 'return_on_equity', 'Return on Equity'],
    'roa': ['roa', 'return on assets', 'return_on_assets', 'Return on Assets'],
    'roi': ['roi', 'return on investment', 'return_on_investment', 'Return on Investment'],
    'earning_per_share': ['earning per share', 'earnings per share', 'eps', 'earning_per_share', 'Earnings per Share'],
    'free_cash_flow_per_share': ['free cash flow per share', 'free_cash_flow_per_share', 'fcf per share', 'Free Cash Flow per Share'],
    'return_on_tangible_equity': ['return on tangible equity', 'return_on_tangible_equity', 'rote'],
    'market_capin_b_usd': ['market cap', 'market capitalization', 'market_cap', 'market cap in b usd', 'market_capin_b_usd', 'market capin b usd', 'Market Capitalization', 'market capitalization in billion usd'],
    'number_of_employees': ['number of employees', 'employees', 'employee count', 'number_of_employees', 'headcount', 'Number of Employees'],
    'inflation_ratein_us': ['inflation rate', 'inflation', 'us inflation', 'inflation_ratein_us', 'inflation ratein us', 'US Inflation Rate'],
    'cost_of_revenue': ['cost_of_revenue', 'cost of revenue', 'Cost of Revenue'],
    'operating_expenses': ['operating_expenses', 'operating expenses', 'Operating Expenses'],
    'operating_income': ['operating_income', 'operating income', 'Operating Income'],
    'ebit': ['ebit', 'EBIT'],
    'depreciation': ['depreciation', 'Depreciation'],
    'amortization': ['amortization', 'Amortization'],
    'depreciation_amortization': ['depreciation_amortization', 'depreciation amortization', 'Depreciation & Amortization'],
    'interest_expense': ['interest_expense', 'interest expense', 'Interest Expense'],
    'interest_income': ['interest_income', 'interest income', 'Interest Income'],
    'income_before_tax': ['income_before_tax', 'income before tax', 'Income Before Tax'],
    'income_tax': ['income_tax', 'income tax', 'Income Tax'],
    'gross_margin': ['gross_margin', 'gross margin', 'Gross Margin'],
    'operating_margin': ['operating_margin', 'operating margin', 'Operating Margin'],
    'ebit_margin': ['ebit_margin', 'ebit margin', 'EBIT Margin'],
    'ebitda_margin': ['ebitda_margin', 'ebitda margin', 'EBITDA Margin'],
    'net_profit_margin': ['net_profit_margin', 'net profit margin', 'Net Profit Margin', 'net margin', 'profit margin', 'net margin percentage', 'net profit margin percentage'],
    'pretax_margin': ['pretax_margin', 'pretax margin', 'Pre-Tax Margin'],
    'tax_rate': ['tax_rate', 'tax rate', 'Effective Tax Rate'],
    'rote': ['rote', 'Return on Tangible Equity'],
    'return_on_capital': ['return_on_capital', 'return on capital', 'Return on Capital'],
    'profit_per_employee': ['profit_per_employee', 'profit per employee', 'Profit per Employee'],
    'total_assets': ['total_assets', 'total assets', 'Total Assets'],
    'current_assets': ['current_assets', 'current assets', 'Current Assets', 'total current assets'],
    'non_current_assets': ['non_current_assets', 'non current assets', 'Non-Current Assets'],
    'cash': ['cash', 'Cash & Cash Equivalents', 'cash balance', 'cash and cash equivalents', 'cash & cash equivalents'],
    'inventory': ['inventory', 'Inventory'],
    'accounts_receivable': ['accounts_receivable', 'accounts receivable', 'Accounts Receivable'],
    'accounts_payable': ['accounts_payable', 'accounts payable', 'Accounts Payable'],
    'total_liabilities': ['total_liabilities', 'total liabilities', 'Total Liabilities'],
    'current_liabilities': ['current_liabilities', 'current liabilities', 'Current Liabilities', 'current liability', 'total current liabilities'],
    'long_term_debt': ['long_term_debt', 'long term debt', 'Long-Term Debt'],
    'short_term_debt': ['short_term_debt', 'short term debt', 'Short-Term Debt'],
    'total_debt': ['total_debt', 'total debt', 'Total Debt', 'debt', 'total borrowings'],
    'retained_earnings': ['retained_earnings', 'retained earnings', 'Retained Earnings'],
    'working_capital': ['working_capital', 'working capital', 'Working Capital'],
    'quick_ratio': ['quick_ratio', 'quick ratio', 'Quick Ratio'],
    'cash_ratio': ['cash_ratio', 'cash ratio', 'Cash Ratio'],
    'working_capital_ratio': ['working_capital_ratio', 'working capital ratio', 'Working Capital Ratio'],
    'operating_cash_flow_ratio': ['operating_cash_flow_ratio', 'operating cash flow ratio', 'Operating Cash Flow Ratio'],
    'cash_to_assets': ['cash_to_assets', 'cash to assets', 'Cash to Assets'],
    'debt_asset_ratio': ['debt_asset_ratio', 'debt asset ratio', 'Debt to Asset Ratio'],
    'liabilities_asset_ratio': ['liabilities_asset_ratio', 'liabilities asset ratio', 'Liabilities to Assets'],
    'equity_asset_ratio': ['equity_asset_ratio', 'equity asset ratio', 'Equity to Assets'],
    'interest_coverage': ['interest_coverage', 'interest coverage', 'Interest Coverage Ratio'],
    'debt_service_coverage': ['debt_service_coverage', 'debt service coverage', 'Debt Service Coverage Ratio'],
    'net_debt': ['net_debt', 'net debt', 'Net Debt'],
    'net_debt_to_ebitda': ['net_debt_to_ebitda', 'net debt to ebitda', 'Net Debt to EBITDA'],
    'free_cash_flow': ['free_cash_flow', 'free cash flow', 'Free Cash Flow', 'free cashflow'],
    'operating_cash_flow_margin': ['operating_cash_flow_margin', 'operating cash flow margin', 'Operating Cash Flow Margin'],
    'free_cash_flow_margin': ['free_cash_flow_margin', 'free cash flow margin', 'Free Cash Flow Margin'],
    'cash_conversion_ratio': ['cash_conversion_ratio', 'cash conversion ratio', 'Cash Conversion Ratio'],
    'cash_flow_to_debt': ['cash_flow_to_debt', 'cash flow to debt', 'Cash Flow to Debt'],
    'investing_to_operating_cash': ['investing_to_operating_cash', 'investing to operating cash', 'Investing to Operating Cash Flow'],
    'asset_turnover': ['asset_turnover', 'asset turnover', 'Asset Turnover'],
    'inventory_turnover': ['inventory_turnover', 'inventory turnover', 'Inventory Turnover'],
    'receivable_turnover': ['receivable_turnover', 'receivable turnover', 'Receivable Turnover'],
    'payable_turnover': ['payable_turnover', 'payable turnover', 'Payable Turnover'],
    'days_inventory': ['days_inventory', 'days inventory', 'Days Inventory Outstanding'],
    'days_receivable': ['days_receivable', 'days receivable', 'Days Sales Outstanding'],
    'days_payable': ['days_payable', 'days payable', 'Days Payable Outstanding'],
    'cash_conversion_cycle': ['cash_conversion_cycle', 'cash conversion cycle', 'Cash Conversion Cycle'],
    'revenue_per_employee': ['revenue_per_employee', 'revenue per employee', 'Revenue per Employee'],
    'ebitda_per_employee': ['ebitda_per_employee', 'ebitda per employee', 'EBITDA per Employee'],
    'pe_ratio': ['pe_ratio', 'pe ratio', 'Price to Earnings Ratio'],
    'pb_ratio': ['pb_ratio', 'pb ratio', 'Price to Book Ratio'],
    'dividend_yield': ['dividend_yield', 'dividend yield', 'Dividend Yield'],
    'earnings_yield': ['earnings_yield', 'earnings yield', 'Earnings Yield'],
    'book_value_per_share': ['book_value_per_share', 'book value per share', 'Book Value per Share'],
    'market_cap_to_revenue': ['market_cap_to_revenue', 'market cap to revenue', 'Market Cap to Revenue'],
    'market_cap_to_ebitda': ['market_cap_to_ebitda', 'market cap to ebitda', 'Market Cap to EBITDA'],
    'market_cap_to_net_income': ['market_cap_to_net_income', 'market cap to net income', 'Market Cap to Net Income'],
    'revenue_yoy': ['revenue_yoy', 'revenue yoy', 'Revenue YoY Growth'],
    'gross_profit_yoy': ['gross_profit_yoy', 'gross profit yoy', 'Gross Profit YoY Growth'],
    'ebitda_yoy': ['ebitda_yoy', 'ebitda yoy', 'EBITDA YoY Growth'],
    'net_income_yoy': ['net_income_yoy', 'net income yoy', 'Net Income YoY Growth'],
    'equity_yoy': ['equity_yoy', 'equity yoy', 'Equity YoY Growth'],
    'assets_yoy': ['assets_yoy', 'assets yoy', 'Assets YoY Growth'],
    'debt_yoy': ['debt_yoy', 'debt yoy', 'Debt YoY Growth'],
    'operating_cash_flow_yoy': ['operating_cash_flow_yoy', 'operating cash flow yoy', 'Operating Cash Flow YoY Growth'],
    'revenue_cagr': ['revenue_cagr', 'revenue cagr', 'Revenue CAGR'],
    'net_income_cagr': ['net_income_cagr', 'net income cagr', 'Net Income CAGR'],
    'earnings_quality': ['earnings_quality', 'earnings quality', 'Earnings Quality'],
    'profitability_trend': ['profitability_trend', 'profitability trend', 'Profitability Trend'],
    'margin_expansion': ['margin_expansion', 'margin expansion', 'Margin Expansion'],
    'margin_compression': ['margin_compression', 'margin compression', 'Margin Compression'],
    'revenue_profit_divergence': ['revenue_profit_divergence', 'revenue profit divergence', 'Revenue-Profit Divergence'],
    'debt_growth_vs_revenue_growth': ['debt_growth_vs_revenue_growth', 'debt growth vs revenue growth', 'Debt Growth vs Revenue Growth'],
    'cash_flow_profit_gap': ['cash_flow_profit_gap', 'cash flow profit gap', 'Cash Flow-Profit Gap'],
    'financial_leverage': ['financial_leverage', 'financial leverage', 'Financial Leverage'],
    'risk_score': ['risk_score', 'risk score', 'Financial Risk Score'],
    'financial_health_score': ['financial_health_score', 'financial health score', 'Financial Health Score'],
    'revenue_growth_rate': ['revenue_growth_rate', 'revenue growth rate', 'Revenue Growth Rate'],
    'profit_growth_rate': ['profit_growth_rate', 'profit growth rate', 'Profit Growth Rate'],
    'employee_growth_rate': ['employee_growth_rate', 'employee growth rate', 'Employee Growth Rate'],
    'revenue_per_employee_growth': ['revenue_per_employee_growth', 'revenue per employee growth', 'Revenue per Employee Growth'],
    'real_revenue_growth': ['real_revenue_growth', 'real revenue growth', 'Real Revenue Growth'],
    'real_profit_growth': ['real_profit_growth', 'real profit growth', 'Real Profit Growth'],
}

# Placeholder tokens treated as missing (not as invalid numerics).
# isna() alone misses these, so blank/placeholder cells were previously
# under-counted as missing values.
MISSING_PLACEHOLDERS = {
    "", "-", "--", "na", "n/a", "nan", "null", "none", "?",
    "#n/a", "#value!", "#div/0!", "#ref!", "#null!",
}


def _strip_placeholders(series):
    """Convert placeholder strings in a series to NA (entire-cell match)."""
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        stripped = series.astype("string").str.strip()
        mask = stripped.str.lower().isin(MISSING_PLACEHOLDERS)
        return series.mask(mask.fillna(False).to_numpy(), pd.NA)
    return series


def normalize_name(value):
    value = str(value).strip().lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[%(){}\[\]/\\\-]+", " ", value)
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value

def load_raw_file(uploaded_file):
    name = getattr(uploaded_file, "name", "").lower()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        try:
            excel = pd.ExcelFile(uploaded_file)
        except ImportError as exc:
            raise ImportError(
                "Excel support requires the 'openpyxl' package, which is "
                "listed in requirements.txt. Install it with "
                "'pip install openpyxl' and retry the upload."
            ) from exc
        sheets = {}

        for sheet in excel.sheet_names:
            try:
                temp = pd.read_excel(uploaded_file, sheet_name=sheet)
            except ImportError as exc:
                raise ImportError(
                    "Excel support requires the 'openpyxl' package, which is "
                    "listed in requirements.txt. Install it with "
                    "'pip install openpyxl' and retry the upload."
                ) from exc
            if not temp.empty:
                sheets[sheet] = temp

        if not sheets:
            raise ValueError("The Excel file contains no readable data.")

        best_sheet = max(sheets, key=lambda x: len(sheets[x]))
        return sheets[best_sheet].copy(), {
            "file_type": "Excel",
            "sheet_names": excel.sheet_names,
            "selected_sheet": best_sheet,
            "sheet_count": len(excel.sheet_names),
        }

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file), {
            "file_type": "CSV",
            "sheet_names": [],
            "selected_sheet": None,
            "sheet_count": 0,
        }

    raise ValueError("Unsupported file type. Please upload CSV or Excel.")

def infer_mapping(columns):
    normalized = {col: normalize_name(col) for col in columns}
    mapping = {}
    confidence = {}
    # One source column may only back ONE FinSight feature. Without this,
    # substring rules below (e.g. "ebit" inside "ebitda", "cash" inside
    # "cash_flow_from_operating") map the same column to several phantom
    # features that were never provided in the upload.
    used_sources = set()

    for target, aliases in ALIASES.items():
        alias_norm = [normalize_name(x) for x in aliases]

        exact = []
        for original, norm in normalized.items():
            if original in used_sources:
                continue
            if norm in alias_norm:
                exact.append(original)

        if exact:
            mapping[target] = exact[0]
            confidence[target] = "EXACT"
            used_sources.add(exact[0])
            continue

        best = None
        best_score = 0

        for original, norm in normalized.items():
            if original in used_sources:
                continue
            for alias in alias_norm:
                if norm == alias:
                    score = 1.0
                elif alias in norm or norm in alias:
                    score = 0.90
                else:
                    a = set(norm.split())
                    b = set(alias.split())
                    if not a or not b:
                        continue
                    score = len(a & b) / len(a | b)

                if score > best_score:
                    best_score = score
                    best = original

        if best is not None and best_score >= 0.50:
            mapping[target] = best
            confidence[target] = round(best_score, 2)
            used_sources.add(best)

    return mapping, confidence

def normalize_financial_data(raw_df):
    if raw_df is None or raw_df.empty:
        raise ValueError("The uploaded file contains no data.")

    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    mapping, confidence = infer_mapping(df.columns)

    missing_required = [
        field for field in REQUIRED_FIELDS
        if field not in mapping
    ]

    if missing_required:
        return {
            "status": "INVALID",
            "data": None,
            "mapping": mapping,
            "confidence": confidence,
            "missing_required": missing_required,
            "optional_available": [
                x for x in OPTIONAL_FIELDS if x in mapping
            ],
            "extra_columns": [
                c for c in df.columns if c not in mapping.values()
            ],
            "data_quality": None,
        }

    normalized = pd.DataFrame(index=df.index)

    for target, source in mapping.items():
        normalized[target] = df[source]

    numeric_fields = [
        x for x in REQUIRED_FIELDS + OPTIONAL_FIELDS
        if x not in ["year", "company", "category"]
        and x in normalized.columns
    ]

    numeric_report = {}

    for field in numeric_fields:
        original = _strip_placeholders(normalized[field])
        converted = pd.to_numeric(original, errors="coerce")

        invalid_mask = (
            original.notna() &
            converted.isna()
        )

        numeric_report[field] = {
            "invalid_count": int(invalid_mask.sum()),
            "missing_count": int(converted.isna().sum()),
        }

        normalized[field] = converted

    if "year" in normalized.columns:
        normalized["year"] = pd.to_numeric(
            normalized["year"], errors="coerce"
        )

    normalized["company"] = normalized["company"].astype("string").str.strip()

    if "category" in normalized.columns:
        normalized["category"] = (
            normalized["category"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    # Snapshot of provided (non-derived) columns BEFORE any derived
    # features are added. Missing-value statistics are computed over
    # these only: derived YoY/growth columns are NaN for each company's
    # first year by construction and must not inflate the count.
    base_columns = list(normalized.columns)

    # ------------------------------------------------------------------
    # DERIVED FEATURES
    # Calculate only features that can be mathematically derived from
    # reliable source fields. Never fabricate unavailable values.
    # ------------------------------------------------------------------

    def safe_divide(numerator, denominator):
        denominator = denominator.replace(0, pd.NA)
        return numerator / denominator

    # Cash Flow-Profit Gap:
    # Difference between operating cash flow growth and net income growth.
    if (
        "cash_flow_from_operating" in normalized.columns
        and "net_income" in normalized.columns
    ):
        normalized = normalized.sort_values(
            ["company", "year"]
        ).reset_index(drop=True)

        ocf_growth = (
            normalized
            .groupby("company")["cash_flow_from_operating"]
            .pct_change() * 100
        )

        profit_growth = (
            normalized
            .groupby("company")["net_income"]
            .pct_change() * 100
        )

        normalized["cash_flow_profit_gap"] = (
            ocf_growth - profit_growth
        )

    # Cash Flow to Debt:
    # Operating cash flow relative to total debt.
    if (
        "cash_flow_from_operating" in normalized.columns
        and "total_debt" in normalized.columns
    ):
        normalized["cash_flow_to_debt"] = (
            safe_divide(
                normalized["cash_flow_from_operating"],
                normalized["total_debt"]
            ) * 100
        )

    # Employee Growth Rate:
    # Year-over-year change in employee count.
    if (
        "number_of_employees" in normalized.columns
    ):
        normalized["employee_growth_rate"] = (
            normalized
            .groupby("company")["number_of_employees"]
            .pct_change() * 100
        )

    # Equity to Assets:
    # Shareholder equity relative to total assets.
    if (
        "share_holder_equity" in normalized.columns
        and "total_assets" in normalized.columns
    ):
        normalized["equity_asset_ratio"] = (
            safe_divide(
                normalized["share_holder_equity"],
                normalized["total_assets"]
            ) * 100
        )

    # Liabilities to Assets:
    # Total liabilities relative to total assets.
    if (
        "total_liabilities" in normalized.columns
        and "total_assets" in normalized.columns
    ):
        normalized["liabilities_asset_ratio"] = (
            safe_divide(
                normalized["total_liabilities"],
                normalized["total_assets"]
            ) * 100
        )

    # Margin Expansion:
    # Change in net profit margin.
    if (
        "net_profit_margin" in normalized.columns
    ):
        normalized["margin_expansion"] = (
            normalized
            .groupby("company")["net_profit_margin"]
            .diff()
        )

    # Payable Turnover:
    # Cost of revenue relative to accounts payable.
    if (
        "cost_of_revenue" in normalized.columns
        and "accounts_payable" in normalized.columns
    ):
        normalized["payable_turnover"] = safe_divide(
            normalized["cost_of_revenue"],
            normalized["accounts_payable"]
        )

    # Profit Growth Rate:
    # Year-over-year change in net income.
    if (
        "net_income" in normalized.columns
    ):
        normalized["profit_growth_rate"] = (
            normalized
            .groupby("company")["net_income"]
            .pct_change() * 100
        )

    # Real Profit Growth:
    # Profit growth adjusted for inflation.
    if (
        "profit_growth_rate" in normalized.columns
        and "inflation_ratein_us" in normalized.columns
    ):
        normalized["real_profit_growth"] = (
            normalized["profit_growth_rate"]
            - normalized["inflation_ratein_us"]
        )

    # Receivable Turnover:
    # Revenue relative to accounts receivable.
    if (
        "revenue" in normalized.columns
        and "accounts_receivable" in normalized.columns
    ):
        normalized["receivable_turnover"] = safe_divide(
            normalized["revenue"],
            normalized["accounts_receivable"]
        )

    # Effective Tax Rate:
    # Income tax as a percentage of pre-tax income.
    if (
        "income_tax" in normalized.columns
        and "income_before_tax" in normalized.columns
    ):
        normalized["tax_rate"] = (
            safe_divide(
                normalized["income_tax"],
                normalized["income_before_tax"]
            ) * 100
        )

    # The following features are intentionally NOT fabricated:
    # interest_income
    # retained_earnings
    # dividend_yield
    # return_on_capital

    # ------------------------------------------------------------------
    # END DERIVED FEATURES
    # ------------------------------------------------------------------

    # keep="first" counts only redundant copies (keep=False would count
    # both rows of each duplicated pair, overstating the problem).
    duplicate_count = int(
        normalized.duplicated(
            subset=["company", "year"],
            keep="first"
        ).sum()
    )

    duplicate_groups = int(
        normalized.loc[
            normalized.duplicated(subset=["company", "year"], keep=False),
            ["company", "year"],
        ].drop_duplicates().shape[0]
    ) if duplicate_count else 0

    derived_columns = [
        col for col in normalized.columns if col not in base_columns
    ]

    missing_by_field = {
        col: int(normalized[col].isna().sum())
        for col in base_columns
    }

    # Structural NaNs from derived YoY/growth columns (first year per
    # company has no prior period) are reported separately so they do
    # not inflate the missing-values count for provided data.
    derived_missing_by_field = {
        col: int(normalized[col].isna().sum())
        for col in derived_columns
    }

    invalid_numeric_total = sum(
        x["invalid_count"] for x in numeric_report.values()
    )

    quality_issues = []

    if duplicate_count:
        quality_issues.append(
            f"{duplicate_count} duplicate company/year records "
            f"across {duplicate_groups} group(s)"
        )

    if invalid_numeric_total:
        quality_issues.append(
            f"{invalid_numeric_total} invalid numeric values"
        )

    missing_total = sum(missing_by_field.values())

    if missing_total:
        quality_issues.append(
            f"{missing_total} missing values"
        )

    quality_status = "PASS"

    if quality_issues:
        quality_status = "REVIEW"

    unsupported = [
        c for c in df.columns
        if c not in mapping.values()
    ]

    result = {
        "status": "VALID",
        "data": normalized,
        "mapping": mapping,
        "confidence": confidence,
        "missing_required": [],
        "optional_available": [
            x for x in OPTIONAL_FIELDS if x in mapping
        ],
        "extra_columns": unsupported,
        "numeric_report": numeric_report,
        "duplicate_count": duplicate_count,
        "duplicate_groups": duplicate_groups,
        "missing_by_field": missing_by_field,
        "derived_columns": derived_columns,
        "derived_missing_by_field": derived_missing_by_field,
        "invalid_numeric_total": invalid_numeric_total,
        "quality_status": quality_status,
        "quality_issues": quality_issues,
        "row_count": len(normalized),
        "column_count": len(normalized.columns),
    }

    return result

def build_data_quality_report(result):
    if result is None:
        return {
            "status": "INVALID",
            "message": "No data result was provided."
        }

    if result.get("status") != "VALID":
        return {
            "status": "INVALID",
            "missing_required": result.get("missing_required", []),
            "message": "Required financial fields could not be mapped."
        }

    return {
        "status": result.get("quality_status", "REVIEW"),
        "rows": result.get("row_count", 0),
        "columns": result.get("column_count", 0),
        "duplicates": result.get("duplicate_count", 0),
        "duplicate_groups": result.get("duplicate_groups", 0),
        "invalid_numeric_values": result.get(
            "invalid_numeric_total", 0
        ),
        "missing_values": sum(
            result.get("missing_by_field", {}).values()
        ),
        "derived_columns": result.get("derived_columns", []),
        "derived_missing_values": sum(
            result.get("derived_missing_by_field", {}).values()
        ),
        "quality_issues": result.get("quality_issues", []),
        "mapped_fields": result.get("mapping", {}),
        "optional_fields_available": result.get(
            "optional_available", []
        ),
        "extra_columns": result.get("extra_columns", []),
    }


# ============================================================
# UI COMPATIBILITY PATCH
# Allow load_raw_file() to safely accept either:
# 1. Streamlit UploadedFile
# 2. pandas DataFrame
# ============================================================

_original_load_raw_file = load_raw_file

def load_raw_file(uploaded_file):

    # Case 1: Feature Intelligence receives an already-loaded DataFrame
    if isinstance(uploaded_file, pd.DataFrame):

        if uploaded_file.empty:
            raise ValueError(
                "The uploaded dataset contains no readable rows."
            )

        return uploaded_file.copy(), {
            "file_type": "DataFrame",
            "sheet_names": [],
            "selected_sheet": None,
            "sheet_count": 0,
        }

    # Case 2: Normal Streamlit UploadedFile / file-like object
    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
    except Exception:
        pass

    return _original_load_raw_file(uploaded_file)

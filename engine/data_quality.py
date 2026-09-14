
import pandas as pd
import numpy as np

# Placeholder tokens that spreadsheets / manual uploads use for "no value".
# pandas.isna() does NOT catch these, so they must be normalized first.
# NOTE: bare "-" / "--" are treated as missing only when they are the
# entire cell value (negative numbers such as "-5" are unaffected).
MISSING_PLACEHOLDERS = {
    "",
    "-",
    "--",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "?",
    "#n/a",
    "#value!",
    "#div/0!",
    "#ref!",
    "#null!",
}


def standardize_missing(df):
    """
    Return a copy of `df` with placeholder strings converted to NA.

    - Object/string cells are stripped; if the stripped, lower-cased
      value is in MISSING_PLACEHOLDERS it becomes pd.NA.
    - Nothing else is modified: real zeros, negative numbers and
      genuine text (e.g. company names) are preserved.
    """
    cleaned = df.copy()

    for col in cleaned.columns:
        series = cleaned[col]

        if series.dtype == object or pd.api.types.is_string_dtype(series):
            stripped = series.astype("string").str.strip()
            mask = stripped.str.lower().isin(MISSING_PLACEHOLDERS)
            series = series.mask(mask.fillna(False).to_numpy(), pd.NA)
            cleaned[col] = series

    return cleaned


def analyze_data_quality(df):
    if df is None or df.empty:
        return {
            "status": "INVALID",
            "rows": 0,
            "columns": 0,
            "missing_values": 0,
            "duplicate_records": 0,
            "duplicate_groups": 0,
            "numeric_anomalies": 0,
            "invalid_numeric_values": 0,
            "issues": ["No data available."]
        }

    issues = []

    # --- Missing values (NaN + placeholder strings) -------------------
    cleaned = standardize_missing(df)
    missing = int(cleaned.isna().sum().sum())

    # --- Duplicates ----------------------------------------------------
    # keep="first" counts only the *extra* copies; keep=False would
    # double-count every duplicated pair. We report both the number of
    # redundant rows and the number of affected company/year groups.
    duplicate_subset = [
        x for x in ["company", "year"]
        if x in df.columns
    ]

    if duplicate_subset:
        dup_mask_first = df.duplicated(subset=duplicate_subset, keep="first")
        dup_mask_all = df.duplicated(subset=duplicate_subset, keep=False)
        duplicates = int(dup_mask_first.sum())
        duplicate_groups = int(
            df.loc[dup_mask_all, duplicate_subset]
            .drop_duplicates()
            .shape[0]
        ) if duplicates else 0
    else:
        duplicates = int(df.duplicated(keep="first").sum())
        duplicate_groups = duplicates

    # --- Numeric anomalies ---------------------------------------------
    # 1. Infinite values in numeric columns.
    # 2. Non-numeric strings inside columns that are otherwise numeric
    #    ("numeric-like" columns: at least one value coerces to a number).
    #    Pure-text columns (company, category, ...) are never flagged.
    numeric_anomalies = 0
    invalid_numeric_values = 0

    for col in cleaned.columns:
        series = cleaned[col]

        if pd.api.types.is_numeric_dtype(series):
            numeric_anomalies += int(np.isinf(pd.to_numeric(series, errors="coerce")).sum())
            continue

        coerced = pd.to_numeric(series, errors="coerce")

        if coerced.notna().any():
            bad = int((series.notna() & coerced.isna()).sum())
            invalid_numeric_values += bad

    numeric_anomalies += invalid_numeric_values

    if missing:
        issues.append(f"{missing} missing values detected.")

    if duplicates:
        issues.append(
            f"{duplicates} duplicate company/year records detected "
            f"across {duplicate_groups} company/year group(s)."
        )

    if numeric_anomalies:
        issues.append(
            f"{numeric_anomalies} invalid numeric values detected."
        )

    return {
        "status": "PASS" if not issues else "REVIEW",
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": missing,
        "duplicate_records": duplicates,
        "duplicate_groups": duplicate_groups,
        "numeric_anomalies": numeric_anomalies,
        "invalid_numeric_values": invalid_numeric_values,
        "issues": issues,
    }

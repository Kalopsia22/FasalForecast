"""Shared config and helpers used by the pipeline scripts and the Streamlit app."""
import numpy as np
import pandas as pd

N_MONTHS = 6
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# max_yield (kg/ha) = plausibility cap used in 00_prepare_yield.py to drop data-entry errors.
# Every crop gets a 6-month window starting at `start_month` of the ICRISAT year.
# ASSUMPTION: ICRISAT year Y = agricultural year Y-(Y+1); verify in their docs.
CROPS = {
    "wheat":    {"label": "Wheat",    "icrisat": "WHEAT",    "start_month": 11, "max_yield": 6500,
                 "season": "Rabi: sown Nov, harvested Mar-Apr"},
    "chickpea": {"label": "Chickpea", "icrisat": "CHICKPEA", "start_month": 10, "max_yield": 3500,
                 "season": "Rabi: sown Oct, harvested Feb-Mar"},
    "rice":     {"label": "Rice",     "icrisat": "RICE",     "start_month": 6, "max_yield": 6500,
                 "season": "Kharif: Jun-Nov (monsoon clouds make NDVI noisy)"},
    "maize":    {"label": "Maize",    "icrisat": "MAIZE",    "start_month": 6, "max_yield": 9000,
                 "season": "Kharif: Jun-Nov (monsoon clouds make NDVI noisy)"},
}

# Names as they appear in FAO GAUL 2015. Run 01_... --list-states to verify.
STATES_GAUL = ["Andhra Pradesh", "Bihar", "Chhattisgarh", "Gujarat", "Haryana",
               "Himachal Pradesh", "Jharkhand", "Karnataka", "Madhya Pradesh",
               "Maharashtra", "Orissa", "Punjab", "Rajasthan", "Uttar Pradesh",
               "Uttaranchal", "West Bengal", "Telangana"]
GAUL_STATE_ALIAS = {"Uttaranchal": "Uttarakhand"}          # GAUL -> ICRISAT
STATE_FALLBACK = {"Telangana": ["Telangana", "Andhra Pradesh"]}  # pre-2014 GAUL


def season_months(crop):
    """[(year_offset, calendar_month), ...] for the 6 season months."""
    sm, out = CROPS[crop]["start_month"], []
    for i in range(N_MONTHS):
        m = sm + i
        out.append(((m - 1) // 12, (m - 1) % 12 + 1))
    return out


def month_labels(crop):
    return [MONTH_ABBR[m - 1] for _, m in season_months(crop)]


def ndvi_cols(k):
    return [f"ndvi_s{i}" for i in range(1, k + 1)]


def make_features(df, k):
    """Model features using only the first k season months (mid-season forecasting)."""
    X = pd.DataFrame(index=df.index)
    nd = df[ndvi_cols(k)]
    for c in nd.columns:
        X[c] = df[c]
    X["ndvi_max"] = nd.max(axis=1)
    X["ndvi_mean"] = nd.mean(axis=1)
    X["rain_cum"] = df[[f"rain_s{i}" for i in range(1, k + 1)]].sum(axis=1, min_count=1)
    X["tmean_avg"] = df[[f"tmean_s{i}" for i in range(1, k + 1)]].mean(axis=1)
    X["tmean_last"] = df[f"tmean_s{k}"]
    X["year"] = df["year"]
    return X


def bundle_predict(bundle, df, k):
    """Predict with a saved model bundle. Needs df['district_key'] and feature cols.
    Returns DataFrame[pred, lo, hi, base]; NaN where the district has no history."""
    b = bundle[k]
    out = pd.DataFrame(np.nan, index=df.index, columns=["pred", "lo", "hi", "base"])
    base = df["district_key"].map(b["district_mean"])
    ok = base.notna() & df[ndvi_cols(k)].notna().all(axis=1)
    if ok.any():
        X = make_features(df, k).loc[ok, b["features"]]
        m = b["mean"].predict(X)
        q = b["q"].predict(X)
        lo, hi = np.minimum(q.min(axis=1), m), np.maximum(q.max(axis=1), m)
        bb = base[ok].values
        out.loc[ok, "pred"] = m + bb
        out.loc[ok, "lo"] = lo + bb
        out.loc[ok, "hi"] = hi + bb
        out.loc[ok, "base"] = bb
    return out

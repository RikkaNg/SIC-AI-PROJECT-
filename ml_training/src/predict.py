# -*- coding: utf-8 -*-
"""
predict.py — v3 (khớp contract v2.1 của train.py)

Tự chọn chế độ theo ensemble_meta.json["min_lag"]:
  - min_lag = 1   → RECURSIVE: mirror đúng backtest (window (d-90, d+2],
                    ffill transactions_lag1 per store, blend LOG-SPACE,
                    ghi ngược dự báo làm lag cho ngày sau).
  - min_lag >= 16 → DIRECT one-shot: mọi feature nhìn ≥16 ngày về quá khứ
                    → 16 ngày test dùng toàn bộ lag thật, KHÔNG đệ quy.

Hợp đồng feature (bắt buộc khớp train):
  history : streaming train.csv → family-level → complete_panel() → raw cols
  serving : add_extra_features(engineer_features(window), min_lag) →
            cluster_engineer.transform → preprocessor.transform (LGBM) +
            Pool + cat_features (CatBoost)
  blend   : pred = clip(expm1(w*lgb_raw + (1-w)*cat_raw), 0, None)  ← LOG-SPACE

Lưu ý: streaming history không qua remove_data_errors của data_loader
(vài dòng item âm cực đoan) — sai lệch không đáng kể, chấp nhận.
"""

import sys, json, time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor, Pool

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

BASE_DIR = SRC_DIR.parent
PROJECT_ROOT = BASE_DIR.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml_service" / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = BASE_DIR / "models"

# ====================== CẤU HÌNH ======================
ARTIFACT_SUFFIX = ""        # "" = model đệ quy (RUN 1); "_direct16" = model direct (RUN 2)
HISTORY_START = "2016-01-01"   # khớp cửa sổ dữ liệu lúc train
WINDOW_DAYS = 90               # mirror BACKTEST_WINDOW_DAYS / HISTORY_LOOKBACK
LEAD_BUFFER_DAYS = 2           # is_holiday_lead1/2
LOOKBACK_SHARE = 45
LGBM_UNKNOWN_SENTINEL = 9999

CAT_FEATURES = ["store_nbr", "family", "city", "state", "type", "holiday_type"]

from data_loader import (
    prepare_oil, prepare_holidays, prepare_transactions_lag, add_earthquake_flag,
)
from preprocessor import engineer_features
from features_ext import add_extra_features, complete_panel
from cluster_features import ClusterFeatureEngineer  # noqa: F401


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} [PREDICT] {msg}", flush=True)


# ====================== HOLIDAY 3-CẤP (mirror data_loader) ======================
def merge_holidays_only(df, holidays):
    h_nat = holidays[holidays["locale"] == "National"].drop_duplicates(subset=["date"])
    h_reg = holidays[holidays["locale"] == "Regional"].drop_duplicates(subset=["date", "locale_name"])
    h_loc = holidays[holidays["locale"] == "Local"].drop_duplicates(subset=["date", "locale_name"])

    before = len(df)
    df = df.merge(h_nat[["date", "holiday_type", "holiday_description"]], on="date", how="left")
    df = df.merge(
        h_reg[["date", "locale_name", "holiday_type", "holiday_description"]].rename(columns={"locale_name": "state"}),
        on=["date", "state"], how="left", suffixes=("", "_reg"))
    df["holiday_type"] = df["holiday_type"].fillna(df["holiday_type_reg"])
    df.drop(columns=["holiday_type_reg", "holiday_description_reg"], inplace=True, errors="ignore")
    df = df.merge(
        h_loc[["date", "locale_name", "holiday_type", "holiday_description"]].rename(columns={"locale_name": "city"}),
        on=["date", "city"], how="left", suffixes=("", "_loc"))
    df["holiday_type"] = df["holiday_type"].fillna(df["holiday_type_loc"])
    df.drop(columns=["holiday_type_loc", "holiday_description_loc"], inplace=True, errors="ignore")
    assert len(df) == before, "Số dòng đổi sau khi merge holidays"
    df["holiday_type"] = df["holiday_type"].fillna("Normal Day")
    return df


# ====================== HISTORY (streaming + complete_panel) ======================
def build_family_history(items, stores, oil, holidays, transactions, test_max):
    log(f"Streaming train.csv từ {HISTORY_START}...")
    item_family = dict(zip(items["item_nbr"], items["family"]))
    fam_perishable = items.groupby("family")["perishable"].max()
    # onpromotion BẮT BUỘC khai báo dtype nullable: cột mixed True/False/NaN khi
    # đọc chunked chạm bug DtypeWarning của pandas 3.0 (IndexError trong
    # _concatenate_chunks) — đọc đầy đủ file (data_loader) không gặp vì không chunk.
    dtypes = {"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32",
              "onpromotion": "boolean"}
    parts, scanned = [], 0
    for chunk in pd.read_csv(RAW_DIR / "train.csv",
                             usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
                             dtype=dtypes, parse_dates=["date"], chunksize=2_000_000):
        scanned += len(chunk)
        c = chunk[chunk["date"] >= HISTORY_START]
        if c.empty:
            continue
        c = c.copy()
        c["family"] = c["item_nbr"].map(item_family)
        c = c.dropna(subset=["family"])
        parts.append(c.groupby(["store_nbr", "family", "date"], as_index=False).agg(
            unit_sales=("unit_sales", "sum"), onpromotion=("onpromotion", "any")))
    log(f"Đã quét {scanned:,} dòng")

    hist = pd.concat(parts, ignore_index=True)
    # Groupby theo chunk tách đôi những nhóm (store, family, date) nằm ở ranh giới
    # 2 chunk (~427 key) → phải gộp lại trước khi complete_panel.
    hist = hist.groupby(["store_nbr", "family", "date"], as_index=False).agg(
        unit_sales=("unit_sales", "sum"), onpromotion=("onpromotion", "any"))
    hist["perishable"] = hist["family"].map(fam_perishable)
    hist = hist.merge(stores, on="store_nbr", how="left")
    oil_prepared = prepare_oil(oil, hist["date"].min(), test_max)
    hist = hist.merge(oil_prepared, on="date", how="left")
    hist = merge_holidays_only(hist, holidays)
    hist = add_earthquake_flag(hist)
    hist = hist.merge(prepare_transactions_lag(transactions, 1), on=["store_nbr", "date"], how="left")
    hist["target"] = hist["unit_sales"].clip(lower=0).astype("float64")

    # BẮT BUỘC khớp train: lag theo NGÂY + model biết ngày bán 0
    hist = complete_panel(hist)
    log(f"History: {len(hist):,} dòng ({hist['date'].min().date()} → {hist['date'].max().date()})")
    return hist


def build_test_skeleton(test, items, stores, oil_prepared, holidays, transactions):
    test_items = test.merge(items[["item_nbr", "family"]], on="item_nbr", how="left")
    if test_items["family"].isna().any():
        log(f"CẢNH BÁO: {int(test_items['family'].isna().sum())} dòng test không map family → 0")

    skel = test_items.groupby(["store_nbr", "family", "date"], as_index=False).agg(
        onpromotion=("onpromotion", "any"))
    skel["perishable"] = skel["family"].map(items.groupby("family")["perishable"].max())
    skel = skel.merge(stores, on="store_nbr", how="left")
    skel = skel.merge(oil_prepared[["date", "oil_price"]], on="date", how="left")
    skel = merge_holidays_only(skel, holidays)
    skel = add_earthquake_flag(skel)
    # trans_lag ngày ĐẦU test = giá trị thật (lag từ ngày cuối history);
    # các ngày sau NaN → ffill per store (mirror backtest [F1])
    skel = skel.merge(prepare_transactions_lag(transactions, 1), on=["store_nbr", "date"], how="left")
    skel["target"] = np.nan
    return test_items[["id", "store_nbr", "item_nbr", "date", "family"]], skel


# ====================== PREDICT HELPERS (mirror train.py) ======================
def _fix_unknown_categorical(X):
    cat_cols = [c for c in CAT_FEATURES if c in X.columns]
    if not cat_cols or not bool((X[cat_cols] < 0).any().any()):
        return X
    for col in cat_cols:
        neg = X[col] < 0
        if neg.any():
            X.loc[neg, col] = LGBM_UNKNOWN_SENTINEL
    return X


def _predict_catboost(cat_model, X):
    """Đường predict CatBoost DUY NHẤT — copy nguyên văn train.py::_predict_catboost."""
    feature_names = list(cat_model.feature_names_ or [])
    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        raise RuntimeError(f"CatBoost thiếu feature {missing[:8]} — contract train/serve lệch!")
    X_cat = X.reindex(columns=feature_names).copy()
    if "store_nbr" in X_cat.columns and not pd.api.types.is_integer_dtype(X_cat["store_nbr"]):
        X_cat["store_nbr"] = X_cat["store_nbr"].round().astype("int64")
    for col in CAT_FEATURES:
        if col not in X_cat.columns or col == "store_nbr":
            continue
        if col == "holiday_type":
            X_cat[col] = X_cat[col].fillna("Normal Day")
        X_cat[col] = X_cat[col].astype(str)
    return cat_model.predict(Pool(data=X_cat, cat_features=[c for c in CAT_FEATURES if c in X_cat.columns]))


def _blend(day, prep, lgbm, cat, w_lgbm):
    X = _fix_unknown_categorical(prep.transform(day))
    lgb_raw = lgbm.predict(X)
    cat_raw = _predict_catboost(cat, day)
    return np.clip(np.expm1(w_lgbm * lgb_raw + (1 - w_lgbm) * cat_raw), 0, None)  # LOG-SPACE


# ====================== 2 CHẾ ĐỘ DỰ BÁO ======================
def recursive_forecast(combined, test_dates, lgbm, prep, ce, cat, w_lgbm, min_lag):
    """Mirror recursive_backtest: window (d-90, d+2], ghi ngược dự báo làm lag."""
    log(f"RECURSIVE forecast (min_lag={min_lag})...")
    combined = combined.copy()
    combined["target"] = combined["target"].astype("float64")
    preds_out = []
    for i, d in enumerate(sorted(test_dates), 1):
        d = pd.Timestamp(d)
        window = combined[(combined["date"] > d - pd.Timedelta(days=WINDOW_DAYS))
                          & (combined["date"] <= d + pd.Timedelta(days=LEAD_BUFFER_DAYS))].copy()
        featured = ce.transform(add_extra_features(engineer_features(window), min_lag=min_lag))
        day = featured[featured["date"] == d]
        if day.empty:
            log(f"  {d.date()}: không có dòng — skip")
            continue
        pred = _blend(day, prep, lgbm, cat, w_lgbm)

        mask = combined["date"] == d
        key = combined.loc[mask, "store_nbr"].astype(str) + "|" + combined.loc[mask, "family"].astype(str)
        pkey = day["store_nbr"].astype(str) + "|" + day["family"].astype(str)
        combined.loc[mask, "target"] = key.map(pd.Series(pred, index=pkey.values)).astype("float64").values

        out = day[["date", "store_nbr", "family"]].copy()
        out["predicted_target"] = pred
        preds_out.append(out)
        log(f"  Ngày {i}/{len(test_dates)}: {d.date()} → {len(pred)} điểm | mean={pred.mean():.1f}")
    return pd.concat(preds_out, ignore_index=True)


def direct_forecast(combined, test_dates, lgbm, prep, ce, cat, w_lgbm, min_lag):
    """Model direct (min_lag>=16): one-shot, mọi lag là số thật — không đệ quy."""
    log(f"DIRECT one-shot forecast (min_lag={min_lag})...")
    lo = pd.Timestamp(test_dates[0]) - pd.Timedelta(days=WINDOW_DAYS)
    hi = pd.Timestamp(test_dates[-1]) + pd.Timedelta(days=LEAD_BUFFER_DAYS)
    temp = combined[(combined["date"] >= lo) & (combined["date"] <= hi)].copy()
    featured = ce.transform(add_extra_features(engineer_features(temp), min_lag=min_lag))
    day = featured[featured["date"].isin(pd.DatetimeIndex(test_dates))].copy()
    pred = _blend(day, prep, lgbm, cat, w_lgbm)
    out = day[["date", "store_nbr", "family"]].copy()
    out["predicted_target"] = pred
    log(f"  One-shot {len(test_dates)} ngày × {len(day):,} điểm family")
    return out


# ====================== ITEM SHARE ======================
def compute_item_share(max_date, lookback=LOOKBACK_SHARE):
    log(f"Item shares (recent {lookback}d)...")
    start = (max_date - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")
    dtypes = {"date": "str", "store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"}
    agg = None
    for chunk in pd.read_csv(RAW_DIR / "train.csv",
                             usecols=["date", "store_nbr", "item_nbr", "unit_sales"],
                             dtype=dtypes, chunksize=2_000_000):
        m = chunk["date"] >= start
        if m.any():
            c = chunk[m].copy()
            c["unit_sales"] = c["unit_sales"].clip(lower=0)
            g = c.groupby(["store_nbr", "item_nbr"])["unit_sales"].agg(["sum", "count"])
            agg = g if agg is None else agg.add(g, fill_value=0)
    recent = (agg["sum"] / agg["count"].replace(0, np.nan)).rename("unit_sales").reset_index()
    items = pd.read_csv(RAW_DIR / "items.csv", dtype={"item_nbr": "int32"})
    recent = recent.merge(items[["item_nbr", "family"]], on="item_nbr", how="left")
    fam = recent.groupby(["store_nbr", "family"])["unit_sales"].transform("sum")
    recent["share"] = (recent["unit_sales"] / fam.replace(0, np.nan)).fillna(0).clip(upper=1.0)
    return recent[["store_nbr", "item_nbr", "family", "share"]]


# ====================== MAIN ======================
def main():
    t0 = time.time()
    s = ARTIFACT_SUFFIX
    log(f"Loading artifacts (suffix='{s or None}')...")
    lgbm = joblib.load(MODELS_DIR / f"lgbm_model{s}.pkl")
    prep = joblib.load(MODELS_DIR / f"preprocessor{s}.pkl")
    ce = joblib.load(MODELS_DIR / f"cluster_engineer{s}.pkl")
    cat = CatBoostRegressor()
    cat.load_model(str(MODELS_DIR / f"catboost_model{s}.cbm"))
    with open(MODELS_DIR / f"ensemble_meta{s}.json", encoding="utf-8") as f:
        meta = json.load(f)
    w_lgbm = float(meta["lgbm_weight"])
    min_lag = int(meta.get("min_lag", 1))
    mode = "direct" if min_lag >= 16 else "recursive"
    log(f"w_LGBM={w_lgbm:.3f} | min_lag={min_lag} → chế độ {mode.upper()}")

    test = pd.read_csv(RAW_DIR / "test.csv", parse_dates=["date"])
    test_dates = sorted(test["date"].unique())
    test_max = pd.Timestamp(test_dates[-1])

    stores = pd.read_csv(RAW_DIR / "stores.csv")
    items = pd.read_csv(RAW_DIR / "items.csv")
    oil = pd.read_csv(RAW_DIR / "oil.csv", parse_dates=["date"])
    holidays = prepare_holidays(pd.read_csv(RAW_DIR / "holidays_events.csv", parse_dates=["date"]))
    transactions = pd.read_csv(RAW_DIR / "transactions.csv", parse_dates=["date"])

    hist = build_family_history(items, stores, oil, holidays, transactions, test_max)
    oil_prepared = prepare_oil(oil, hist["date"].min(), test_max)
    test_items, skel = build_test_skeleton(test, items, stores, oil_prepared, holidays, transactions)

    combined = pd.concat([hist, skel], ignore_index=True)
    # ffill transactions_lag1 per store cho ngày tương lai (mirror backtest [F1])
    if "transactions_lag1" in combined.columns:
        combined["transactions_lag1"] = combined.groupby("store_nbr", observed=True)["transactions_lag1"].ffill()

    if mode == "direct":
        family_preds = direct_forecast(combined, test_dates, lgbm, prep, ce, cat, w_lgbm, min_lag)
    else:
        family_preds = recursive_forecast(combined, test_dates, lgbm, prep, ce, cat, w_lgbm, min_lag)

    shares = compute_item_share(max_date=hist["date"].max())
    merged = test_items.merge(family_preds, on=["store_nbr", "family", "date"], how="left")
    n_miss = int(merged["predicted_target"].isna().sum())
    if n_miss:
        log(f"CẢNH BÁO: {n_miss:,} dòng không có dự báo family → 0")
    merged = merged.merge(shares, on=["store_nbr", "item_nbr", "family"], how="left")
    merged["unit_sales"] = np.clip(
        merged["predicted_target"].fillna(0) * merged["share"].fillna(0), 0, None)

    out_path = OUTPUT_DIR / "submission_ensemble.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged[["id", "unit_sales"]].sort_values("id").to_csv(out_path, index=False)

    meta_out = {
        "mode": mode, "min_lag": min_lag, "artifact_suffix": s,
        "lgbm_weight": w_lgbm, "blend_space": "log",
        "val_reference": meta.get("product_quality", {}),
        "rows": int(len(merged)), "nonzero_pct": round(float((merged["unit_sales"] > 0).mean() * 100), 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUTPUT_DIR / "submission_meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False))
    log(f"XONG sau {(time.time()-t0)/60:.1f} phút → {out_path}")
    log(f"Meta: {meta_out}")


if __name__ == "__main__":
    main()
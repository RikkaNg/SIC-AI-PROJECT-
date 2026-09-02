# -*- coding: utf-8 -*-
"""
predict_global_lgbm.py
Dự báo test.csv bằng GLOBAL LIGHTGBM ONLY (không cần CatBoost / local models /
cluster_engineer.pkl - các file này không có trong môi trường hiện tại).

Sai khác so với predict.py gốc:
- Bỏ nhánh CatBoost (dùng 100% trọng số LGBM; RMSLE ~0.36 so với blend 0.357).
- Tự fit ClusterFeatureEngineer trên lịch sử gần đây (thay vì load pkl thiếu).
- Dựng lịch sử family-level bằng STREAMING train.csv lọc theo mốc thời gian
  (không load toàn bộ 4.7GB vào RAM như load_data()).

Output: ml_training/data/processed/submission_ensemble.csv
"""

import sys
import json
import time
from pathlib import Path

# Console Windows mặc định cp1252 - ép UTF-8 để log tiếng Việt không crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import joblib

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

BASE_DIR = SRC_DIR.parent                      # ml_training/
PROJECT_ROOT = BASE_DIR.parent                 # SIC-AI-PROJECT-/
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_SUBMISSION_PATH = OUTPUT_DIR / "submission_ensemble.csv"

MODELS_DIR = PROJECT_ROOT / "ml_service" / "models"
LGBM_MODEL_PATH = MODELS_DIR / "lgbm_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

# Đủ cho lag28 + rolling7 + holiday lead/lag2, dư địa lớn
HISTORY_START = "2017-04-15"

from data_loader import prepare_oil, prepare_holidays, prepare_transactions_lag, build_target
from preprocessor import engineer_features
from cluster_features import ClusterFeatureEngineer


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} [PREDICT] {msg}", flush=True)


# Cột thuộc nhánh num_zero của ColumnTransformer - phải ép về số trước khi transform
NUM_ZERO_COLS = [
    "transactions_lag1", "sales_lag7", "sales_lag14", "sales_rolling_mean7",
    "cluster_mean_sales", "cluster_median_sales", "cluster_std_sales",
    "cluster_family_mean_sales", "cluster_promo_mean_sales", "cluster_promo_lift",
]


def patch_preprocessor_dtypes(preprocessor):
    """
    Pickle tạo bằng sklearn 1.9 nhưng chạy trên 1.7: SimpleImputer(strategy='constant')
    đôi khi lưu statistics_ dạng object -> transform trên cột float64 bị từ chối.
    Ép lại statistics_ về float64 (fill_value gốc là 0 nên mất mát = 0).
    """
    for name in ("num_zero",):
        tr = getattr(preprocessor, "named_transformers_", {}).get(name)
        imp = getattr(tr, "named_steps", {}).get("impute") if tr is not None else None
        stats = getattr(imp, "statistics_", None)
        if stats is not None and stats.dtype == object:
            try:
                imp.statistics_ = stats.astype("float64")
                log(f"Đã ép statistics_ của imputer '{name}' về float64")
            except Exception as e:
                log(f"KHÔNG ép được statistics_ ('{name}'): {e}")


def sanitize_numeric(day):
    """Ép các cột num_zero về float64 để khớp dtype lúc fit."""
    for c in NUM_ZERO_COLS:
        if c in day.columns and not pd.api.types.is_float_dtype(day[c]):
            day[c] = pd.to_numeric(day[c], errors="coerce").astype("float64")
    return day


def merge_holidays_only(df, holidays):
    """Merge chi tiết National/Regional/Local - bản của merge_dimensions trừ stores/items/oil."""
    h_nat = holidays[holidays["locale"] == "National"].drop_duplicates(subset=["date"])
    h_reg = holidays[holidays["locale"] == "Regional"].drop_duplicates(subset=["date", "locale_name"])
    h_loc = holidays[holidays["locale"] == "Local"].drop_duplicates(subset=["date", "locale_name"])

    before = len(df)
    df = df.merge(h_nat[["date", "holiday_type", "holiday_description"]], on="date", how="left")
    df = df.merge(
        h_reg[["date", "locale_name", "holiday_type", "holiday_description"]].rename(columns={"locale_name": "state"}),
        on=["date", "state"], how="left", suffixes=("", "_reg"),
    )
    df["holiday_type"] = df["holiday_type"].fillna(df["holiday_type_reg"])
    df["holiday_description"] = df["holiday_description"].fillna(df["holiday_description_reg"])
    df.drop(columns=["holiday_type_reg", "holiday_description_reg"], inplace=True)

    df = df.merge(
        h_loc[["date", "locale_name", "holiday_type", "holiday_description"]].rename(columns={"locale_name": "city"}),
        on=["date", "city"], how="left", suffixes=("", "_loc"),
    )
    df["holiday_type"] = df["holiday_type"].fillna(df["holiday_type_loc"])
    df["holiday_description"] = df["holiday_description"].fillna(df["holiday_description_loc"])
    df.drop(columns=["holiday_type_loc", "holiday_description_loc"], inplace=True)

    assert len(df) == before, "Số dòng đổi sau khi merge holidays"
    df["holiday_type"] = df["holiday_type"].fillna("Normal Day")
    df["holiday_description"] = df["holiday_description"].fillna("None")
    return df


def build_family_history(items, stores, oil, holidays, transactions):
    """Streaming train.csv -> khung family-level (store, family, date) từ HISTORY_START."""
    log(f"Streaming train.csv từ {HISTORY_START}...")
    item_family = dict(zip(items["item_nbr"], items["family"]))
    fam_perishable = items.groupby("family")["perishable"].max()
    dtypes = {"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"}
    parts, total_scanned = [], 0

    for chunk in pd.read_csv(
        RAW_DIR / "train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        dtype=dtypes, parse_dates=["date"], chunksize=2_000_000,
    ):
        total_scanned += len(chunk)
        c = chunk[chunk["date"] >= HISTORY_START]
        if c.empty:
            continue
        c = c.copy()
        c["family"] = c["item_nbr"].map(item_family)
        c = c.dropna(subset=["family"])
        g = c.groupby(["store_nbr", "family", "date"], as_index=False).agg(
            unit_sales=("unit_sales", "sum"),
            onpromotion=("onpromotion", "any"),
        )
        parts.append(g)
    log(f"Đã quét {total_scanned:,} dòng train.csv")

    hist = pd.concat(parts, ignore_index=True)
    hist["perishable"] = hist["family"].map(fam_perishable)
    hist = hist.merge(stores, on="store_nbr", how="left")

    oil_prepared = prepare_oil(oil, hist["date"].min(), pd.Timestamp("2017-08-31"))
    hist = hist.merge(oil_prepared, on="date", how="left")
    hist = merge_holidays_only(hist, holidays)
    from data_loader import add_earthquake_flag
    hist = add_earthquake_flag(hist)

    trans_lag = prepare_transactions_lag(transactions, lag_days=1)
    hist = hist.merge(trans_lag, on=["store_nbr", "date"], how="left")
    hist = build_target(hist)
    log(f"Lịch sử family-level: {len(hist):,} dòng ({hist['date'].min().date()} → {hist['date'].max().date()})")
    return hist, oil_prepared


def build_test_skeleton(items, stores, oil_prepared, holidays, transactions):
    """Khung (store, family, date) cho 16 ngày test + map item-level để phân rã."""
    test = pd.read_csv(RAW_DIR / "test.csv", parse_dates=["date"])
    test_items = test.merge(items[["item_nbr", "family"]], on="item_nbr", how="left")
    if test_items["family"].isna().any():
        n = int(test_items["family"].isna().sum())
        log(f"CẢNH BÁO: {n} dòng test không map được family -> sẽ có dự báo 0")

    skel = test_items.groupby(["store_nbr", "family", "date"], as_index=False).agg(
        onpromotion=("onpromotion", "any"),
    )
    fam_perishable = items.groupby("family")["perishable"].max()
    skel["perishable"] = skel["family"].map(fam_perishable)
    skel = skel.merge(stores, on="store_nbr", how="left")
    skel = merge_holidays_only(skel, holidays)
    from data_loader import add_earthquake_flag
    skel = add_earthquake_flag(skel)
    skel["transactions_lag1"] = np.nan
    skel["target"] = np.nan

    keep = ["id", "store_nbr", "item_nbr", "date", "family"]
    return test_items[keep], skel


def recursive_forecast(combined, test_dates, lgbm_model, preprocessor, ce):
    log("Bắt đầu recursive forecast (LGBM only)...")
    predictions = []
    for i, current_date in enumerate(sorted(test_dates), 1):
        temp = combined[combined["date"] <= current_date]
        featured = engineer_features(temp, cluster_engineer=ce)
        day = featured[featured["date"] == current_date]
        if day.empty:
            continue
        day = sanitize_numeric(day)
        X_day = preprocessor.transform(day)
        # Lưới an toàn cuối: mọi cột ra khỏi preprocessor phải là số trước khi predict
        for c in X_day.columns:
            if not pd.api.types.is_numeric_dtype(X_day[c]):
                X_day[c] = pd.to_numeric(X_day[c], errors="coerce").fillna(0)
        preds = np.clip(np.expm1(lgbm_model.predict(X_day)), 0, None)

        pred_df = day[["store_nbr", "family", "date"]].copy()
        pred_df["predicted_target"] = preds
        predictions.append(pred_df)

        pred_map = day[["store_nbr", "family"]].copy()
        pred_map["target"] = preds
        mask = combined["date"] == current_date
        t = combined[mask].drop(columns=["target"], errors="ignore").merge(
            pred_map, on=["store_nbr", "family"], how="left")
        combined.loc[mask, "target"] = t["target"].values
        log(f"  Ngày {i}/{len(test_dates)}: {current_date.date()} -> {len(preds)} điểm family")
    return pd.concat(predictions, ignore_index=True)


def compute_item_share(lookback_recent=45):
    """Tỷ trọng item trong family theo 45 ngày gần nhất (streaming train.csv)."""
    log(f"Computing item shares (recent {lookback_recent}d)...")
    max_date = pd.Timestamp("2017-08-15")
    start_str = (max_date - pd.Timedelta(days=lookback_recent)).strftime("%Y-%m-%d")
    dtypes = {"date": "str", "store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"}
    agg = None
    for chunk in pd.read_csv(RAW_DIR / "train.csv",
                             usecols=["date", "store_nbr", "item_nbr", "unit_sales"],
                             dtype=dtypes, chunksize=2_000_000):
        m = chunk["date"] >= start_str
        if m.any():
            cc = chunk[m].copy()
            cc["unit_sales"] = cc["unit_sales"].clip(lower=0)
            g = cc.groupby(["store_nbr", "item_nbr"])["unit_sales"].agg(["sum", "count"])
            agg = g if agg is None else agg.add(g, fill_value=0)

    recent = (agg["sum"] / agg["count"].replace(0, np.nan)).rename("unit_sales").reset_index()
    items = pd.read_csv(RAW_DIR / "items.csv", dtype={"item_nbr": "int32"})
    recent = recent.merge(items[["item_nbr", "family"]], on="item_nbr", how="left")
    fam_total = recent.groupby(["store_nbr", "family"])["unit_sales"].transform("sum")
    recent["share"] = (recent["unit_sales"] / fam_total.replace(0, np.nan)).fillna(0).clip(upper=1.0)
    return recent[["store_nbr", "item_nbr", "family", "share"]]


def main():
    t0 = time.time()
    log("Loading models...")
    lgbm_model = joblib.load(LGBM_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    patch_preprocessor_dtypes(preprocessor)

    stores = pd.read_csv(RAW_DIR / "stores.csv")
    items = pd.read_csv(RAW_DIR / "items.csv")
    oil = pd.read_csv(RAW_DIR / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(RAW_DIR / "holidays_events.csv", parse_dates=["date"])
    holidays = prepare_holidays(holidays)
    transactions = pd.read_csv(RAW_DIR / "transactions.csv", parse_dates=["date"])

    hist, oil_prepared = build_family_history(items, stores, oil, holidays, transactions)

    ce = ClusterFeatureEngineer(smoothing=10.0)
    # Chuẩn hóa bool -> int TRƯỚC khi fit để khóa merge của CFE cùng dtype
    # với lúc transform (ngược lại pandas đẩy onpromotion về object -> LGBM từ chối)
    hist["onpromotion"] = hist["onpromotion"].fillna(False).astype("int8")
    hist["is_earthquake_period"] = hist["is_earthquake_period"].fillna(False).astype("int8")
    ce.fit(hist, target_col="target")
    log("ClusterFeatureEngineer đã fit trên lịch sử gần đây")

    test_items, skel = build_test_skeleton(items, stores, oil_prepared, holidays, transactions)
    test_dates = sorted(pd.read_csv(RAW_DIR / "test.csv", parse_dates=["date"])["date"].unique())

    combined = pd.concat([hist, skel], ignore_index=True)
    family_preds = recursive_forecast(combined, test_dates, lgbm_model, preprocessor, ce)

    shares = compute_item_share(lookback_recent=45)
    log("Phân rã family -> SKU...")
    merged = test_items.merge(family_preds, on=["store_nbr", "family", "date"], how="left")
    merged = merged.merge(shares, on=["store_nbr", "item_nbr", "family"], how="left")
    merged["unit_sales"] = np.clip(merged["predicted_target"] * merged["share"].fillna(0), 0, None)

    submission = merged[["id", "unit_sales"]].sort_values("id").reset_index(drop=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)

    meta = {
        "method": "global_lgbm_only (no CatBoost, CFE refit on recent history)",
        "rows": int(len(submission)),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nonzero_pct": round(float((submission["unit_sales"] > 0).mean() * 100), 1),
    }
    (OUTPUT_DIR / "submission_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"XONG sau {(time.time()-t0)/60:.1f} phút -> {OUTPUT_SUBMISSION_PATH}")
    log(f"Meta: {meta}")
    print(submission.head(10))


if __name__ == "__main__":
    main()

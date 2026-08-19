"""
predict.py
Dự báo trên test.csv bằng Ensemble LGBM + CatBoost.
Blend: w_lgbm * LGBM_pred + w_cat * CatBoost_pred
Tích hợp Cluster Features, xử lý recursive forecasting và disaggregation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import json
import logging
import warnings
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

# ====================== CẤU HÌNH IMPORT & ĐƯỜNG DẪN ======================

# 1. Định vị các cấp thư mục
SRC_DIR = Path(__file__).resolve().parent               # ml_training/src/
BASE_DIR = SRC_DIR.parent                              # ml_training/
PROJECT_ROOT = BASE_DIR.parent                         # SIC-AI-PROJECT-/

# Thêm src vào sys.path để import các module nội bộ
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# 2. Thư mục Models (Ưu tiên lấy từ ml_service/models, tự động fallback về ml_training/models)
MODELS_DIR = PROJECT_ROOT / "ml_service" / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = BASE_DIR / "models"

LGBM_MODEL_PATH = MODELS_DIR / "lgbm_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
CLUSTER_ENGINEER_PATH = MODELS_DIR / "cluster_engineer.pkl"
CATBOOST_MODEL_PATH = MODELS_DIR / "catboost_model.cbm"
ENSEMBLE_META_PATH = MODELS_DIR / "ensemble_meta.json"

# 3. Thư mục Dữ liệu đầu ra (Submission)
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_SUBMISSION_PATH = OUTPUT_DIR / "submission_ensemble.csv"

from data_loader import (
    RAW_DIR, load_data, load_raw_files, prepare_oil, prepare_holidays,
    merge_dimensions, add_earthquake_flag, aggregate_to_family_level,
)
from preprocessor import engineer_features

try:
    from cluster_features import ClusterFeatureEngineer
except ImportError:
    ClusterFeatureEngineer = None

# ====================== CẤU HÌNH LOGGING ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ====================== CÁC HÀM XỬ LÝ ======================
def build_test_family_skeleton():
    """
    test.csv ở mức item — merge đúng các bước như train rồi gộp lên family.
    Đảm bảo test_family có cột 'cluster' để dùng cluster_engineer.
    """
    logger.info(">>> Building test skeleton...")
    test = pd.read_csv(RAW_DIR / "test.csv", parse_dates=["date"])
    _, stores, items, oil, holidays, _ = load_raw_files()

    oil = prepare_oil(oil, test["date"].min(), test["date"].max())
    holidays = prepare_holidays(holidays)

    test = merge_dimensions(test, stores, items, oil, holidays)
    test = add_earthquake_flag(test)

    test_family = aggregate_to_family_level(
        test.assign(unit_sales=0, onpromotion=test["onpromotion"].fillna(False))
    )

    if "cluster" not in test_family.columns:
        test_family = test_family.merge(
            stores[["store_nbr", "cluster"]].drop_duplicates(),
            on="store_nbr", how="left"
        )

    test_family["transactions"] = np.nan
    return test.drop(columns=["unit_sales"], errors="ignore"), test_family


def compute_item_share(lookback_recent: int = 45, yoy_weight: float = 0) -> pd.DataFrame:
    print(f">>> Computing hybrid item shares (Recent: {lookback_recent}d, YoY weight: {yoy_weight})...")

    max_date = pd.Timestamp("2017-08-15")
    recent_start_str = (max_date - pd.Timedelta(days=lookback_recent)).strftime("%Y-%m-%d")
    yoy_start_str = (max_date - pd.Timedelta(days=365 + lookback_recent)).strftime("%Y-%m-%d")
    yoy_end_str = (max_date - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

    dtypes = {"date": "str", "store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"}
    recent_agg = None
    yoy_agg = None

    print(">>> Streaming train.csv in chunks...")
    for chunk in pd.read_csv(RAW_DIR / "train.csv", usecols=["date", "store_nbr", "item_nbr", "unit_sales"], dtype=dtypes, chunksize=2_000_000):
        mask_recent = chunk["date"] >= recent_start_str
        if mask_recent.any():
            c_recent = chunk[mask_recent].copy()
            c_recent["unit_sales"] = c_recent["unit_sales"].clip(lower=0)
            grp_r = c_recent.groupby(["store_nbr", "item_nbr"])["unit_sales"].agg(['sum', 'count'])
            recent_agg = grp_r if recent_agg is None else recent_agg.add(grp_r, fill_value=0)

        mask_yoy = (chunk["date"] >= yoy_start_str) & (chunk["date"] <= yoy_end_str)
        if mask_yoy.any():
            c_yoy = chunk[mask_yoy].copy()
            c_yoy["unit_sales"] = c_yoy["unit_sales"].clip(lower=0)
            grp_y = c_yoy.groupby(["store_nbr", "item_nbr"])["unit_sales"].agg(['sum', 'count'])
            yoy_agg = grp_y if yoy_agg is None else yoy_agg.add(grp_y, fill_value=0)

    recent_avg = (recent_agg['sum'] / recent_agg['count'].replace(0, np.nan)).rename("unit_sales").reset_index()
    items = pd.read_csv(RAW_DIR / "items.csv", dtype={"item_nbr": "int32", "family": "category"})
    recent_avg = recent_avg.merge(items[["item_nbr", "family"]], on="item_nbr", how="left")

    fam_recent = recent_avg.groupby(["store_nbr", "family"])["unit_sales"].transform("sum")
    recent_avg["recent_share"] = (recent_avg["unit_sales"] / fam_recent.replace(0, np.nan)).fillna(0)

    if yoy_agg is not None:
        yoy_avg = (yoy_agg['sum'] / yoy_agg['count'].replace(0, np.nan)).rename("unit_sales").reset_index()
        yoy_avg = yoy_avg.merge(items[["item_nbr", "family"]], on="item_nbr", how="left")
        fam_yoy = yoy_avg.groupby(["store_nbr", "family"])["unit_sales"].transform("sum")
        yoy_avg["yoy_share"] = (yoy_avg["unit_sales"] / fam_yoy.replace(0, np.nan)).fillna(0)

        merged_share = recent_avg[["store_nbr", "item_nbr", "family", "recent_share"]].merge(
            yoy_avg[["store_nbr", "item_nbr", "yoy_share"]], on=["store_nbr", "item_nbr"], how="left")
        merged_share["yoy_share"] = merged_share["yoy_share"].fillna(merged_share["recent_share"])
        merged_share["share"] = ((1 - yoy_weight) * merged_share["recent_share"] + yoy_weight * merged_share["yoy_share"])
    else:
        merged_share = recent_avg.rename(columns={"recent_share": "share"})

    merged_share["share"] = merged_share["share"].clip(upper=1.0)
    return merged_share[["store_nbr", "item_nbr", "family", "share"]]


def recursive_forecast_ensemble(history_family, test_dates, lgbm_model, preprocessor, cluster_engineer, cat_model, w_lgbm, w_cat):
    logger.info(">>> Starting recursive forecast (Ensemble)...")
    combined = history_family.copy()
    predictions = []
    cat_feature_names = cat_model.feature_names_

    for current_date in sorted(test_dates):
        temp_combined = combined[combined["date"] <= current_date].copy()
        featured = engineer_features(temp_combined)

        if cluster_engineer is not None:
            featured = cluster_engineer.transform(featured)

        day_featured = featured[featured["date"] == current_date].copy()
        if day_featured.empty:
            continue

        # --- 1. LGBM predict ---
        X_day_lgb = preprocessor.transform(day_featured)
        preds_lgb = np.clip(np.expm1(lgbm_model.predict(X_day_lgb)), 0, None)

        # --- 2. CatBoost predict ---
        X_day_cat = day_featured.reindex(columns=cat_feature_names, fill_value=0)
        cat_cols = ['family', 'city', 'state', 'type', 'holiday_type', 'cluster_family_id']
        for col in cat_cols:
            if col in X_day_cat.columns:
                X_day_cat[col] = X_day_cat[col].astype(str)

        preds_cat = np.clip(np.expm1(cat_model.predict(X_day_cat)), 0, None)

        # --- 3. Ensemble blend ---
        preds = np.clip(w_lgbm * preds_lgb + w_cat * preds_cat, 0, None)

        pred_df = day_featured[["store_nbr", "family", "date"]].copy()
        pred_df["predicted_target"] = preds
        predictions.append(pred_df)

        # Cập nhật target an toàn
        pred_map = day_featured[["store_nbr", "family"]].copy()
        pred_map["target"] = preds

        mask = combined["date"] == current_date
        temp_df = combined[mask].drop(columns=["target"], errors="ignore").merge(pred_map, on=["store_nbr", "family"], how="left")
        combined.loc[mask, "target"] = temp_df["target"].values

    logger.info(">>> Recursive forecast complete!")
    return pd.concat(predictions, ignore_index=True)


# ====================== HÀM CHÍNH ======================
def build_submission_ensemble():
    logger.info(">>> Loading models, preprocessor, cluster engineer, and weights...")
    lgbm_model = joblib.load(LGBM_MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    if CLUSTER_ENGINEER_PATH.exists() and ClusterFeatureEngineer is not None:
        cluster_engineer = joblib.load(CLUSTER_ENGINEER_PATH)
    else:
        cluster_engineer = None

    cat_model = CatBoostRegressor()
    cat_model.load_model(str(CATBOOST_MODEL_PATH))

    with open(ENSEMBLE_META_PATH) as f:
        meta = json.load(f)
    w_lgbm = meta["lgbm_weight"]
    w_cat = meta["catboost_weight"]
    logger.info(f">>> Ensemble weights: LGBM={w_lgbm:.3f}, CatBoost={w_cat:.3f}")

    logger.info(">>> Loading train history...")
    train_family = load_data()
    recent_history = train_family[train_family["date"] >= train_family["date"].max() - pd.Timedelta(days=90)].copy()

    test_item, test_family_skeleton = build_test_family_skeleton()
    test_dates = sorted(test_item["date"].unique())
    test_family_skeleton["target"] = np.nan

    combined_history = pd.concat([recent_history, test_family_skeleton], ignore_index=True)

    family_preds = recursive_forecast_ensemble(
        combined_history, test_dates, lgbm_model, preprocessor, cluster_engineer, cat_model, w_lgbm, w_cat
    )

    logger.info(">>> Disaggregating predictions to item level...")
    item_share = compute_item_share(lookback_recent=45, yoy_weight=0)

    merged = test_item.merge(family_preds, on=["store_nbr", "family", "date"], how="left")
    merged = merged.merge(item_share, on=["store_nbr", "item_nbr", "family"], how="left")

    merged["unit_sales"] = merged["predicted_target"] * merged["share"].fillna(0)
    merged["unit_sales"] = np.clip(merged["unit_sales"], 0, None)

    submission = merged[["id", "unit_sales"]].sort_values("id").reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)

    logger.info(f"\n>>> Submission saved to {OUTPUT_SUBMISSION_PATH}")
    logger.info(f">>> Shape: {submission.shape}")
    print(submission.head(10))

    return submission


if __name__ == "__main__":
    sub = build_submission_ensemble()
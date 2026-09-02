"""
predict_local.py (v2.4 - Production Ready)
Dự báo trên test.csv bằng kiến trúc Smart Routing:
- Dùng Local Model cho toàn bộ 33 ngành hàng (Family Models).
- Tự động Fallback về Global Ensemble (LGBM + CatBoost) nếu phát hiện ngành hàng mới.
- Khắc phục triệt để lỗi NoneType Cluster Transformer và lỗi Indexing trong Recursive Loop.
- Disaggregation tỷ trọng Top-Down từ cấp Family xuống SKU (item_nbr).
"""

import sys
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ====================== CẤU HÌNH IMPORT & ĐƯỜNG DẪN ======================
SRC_DIR = Path(__file__).resolve().parent               # ml_training/src/
BASE_DIR = SRC_DIR.parent                              # ml_training/
PROJECT_ROOT = BASE_DIR.parent                         # Root Project

for p in [str(SRC_DIR), str(BASE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.append(p)

MODELS_DIR = PROJECT_ROOT / "ml_service" / "models"
if not MODELS_DIR.exists():
    MODELS_DIR = BASE_DIR / "models"

# Đường dẫn Global Models (Dự phòng)
LGBM_MODEL_PATH = MODELS_DIR / "lgbm_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
CLUSTER_ENGINEER_PATH = MODELS_DIR / "cluster_engineer.pkl"
CATBOOST_MODEL_PATH = MODELS_DIR / "catboost_model.cbm"
ENSEMBLE_META_PATH = MODELS_DIR / "ensemble_meta.json"

# Đường dẫn Local Models (Ưu tiên)
LOCAL_MODELS_PATH = MODELS_DIR / "local_lgbm_models.pkl"

# Đường dẫn Output
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_SUBMISSION_PATH = OUTPUT_DIR / "submission_local.csv"

# Cửa sổ trượt cho vòng lặp đệ quy (§3.2): feature chỉ cần lag 28 + rolling 7/shift 1
# => 36 ngày; giữ 60 ngày làm biên an toàn thay vì quét toàn bộ lịch sử mỗi ngày.
RECURSION_WINDOW_DAYS = 60

from data_loader import (
    RAW_DIR,
    load_data,
    load_raw_files,
    prepare_oil,
    prepare_holidays,
    merge_dimensions,
    add_earthquake_flag,
    aggregate_to_family_level,
)
from preprocessor import engineer_features

# ====================== CẤU HÌNH LOGGING ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ====================== CÁC HÀM XỬ LÝ ======================
def build_test_family_skeleton():
    """Tạo khung dữ liệu kiểm thử cấp Family kết hợp đầy đủ Store, Oil, Holiday."""
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
            on="store_nbr",
            how="left",
        )

    test_family["transactions"] = np.nan
    return test.drop(columns=["unit_sales"], errors="ignore"), test_family


def compute_item_share(lookback_recent: int = 45, yoy_weight: float = 0.0) -> pd.DataFrame:
    """Tính toán ma trận tỷ trọng đóng góp của từng Item trong Family."""
    logger.info(f">>> Computing hybrid item shares (Recent: {lookback_recent}d, YoY weight: {yoy_weight})...")

    max_date = pd.Timestamp("2017-08-15")
    recent_start_str = (max_date - pd.Timedelta(days=lookback_recent)).strftime("%Y-%m-%d")
    yoy_start_str = (max_date - pd.Timedelta(days=365 + lookback_recent)).strftime("%Y-%m-%d")
    yoy_end_str = (max_date - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

    dtypes = {"date": "str", "store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"}
    recent_agg = None
    yoy_agg = None

    logger.info(">>> Streaming train.csv in chunks...")
    for chunk in pd.read_csv(
        RAW_DIR / "train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales"],
        dtype=dtypes,
        chunksize=2_000_000,
    ):
        mask_recent = chunk["date"] >= recent_start_str
        if mask_recent.any():
            c_recent = chunk[mask_recent].copy()
            c_recent["unit_sales"] = c_recent["unit_sales"].clip(lower=0)
            grp_r = c_recent.groupby(["store_nbr", "item_nbr"])["unit_sales"].agg(["sum", "count"])
            recent_agg = grp_r if recent_agg is None else recent_agg.add(grp_r, fill_value=0)

        mask_yoy = (chunk["date"] >= yoy_start_str) & (chunk["date"] <= yoy_end_str)
        if mask_yoy.any():
            c_yoy = chunk[mask_yoy].copy()
            c_yoy["unit_sales"] = c_yoy["unit_sales"].clip(lower=0)
            grp_y = c_yoy.groupby(["store_nbr", "item_nbr"])["unit_sales"].agg(["sum", "count"])
            yoy_agg = grp_y if yoy_agg is None else yoy_agg.add(grp_y, fill_value=0)

    recent_avg = (
        (recent_agg["sum"] / recent_agg["count"].replace(0, np.nan))
        .rename("unit_sales")
        .reset_index()
    )
    items = pd.read_csv(RAW_DIR / "items.csv", dtype={"item_nbr": "int32", "family": "category"})
    recent_avg = recent_avg.merge(items[["item_nbr", "family"]], on="item_nbr", how="left")

    fam_recent = recent_avg.groupby(["store_nbr", "family"])["unit_sales"].transform("sum")
    recent_avg["recent_share"] = (recent_avg["unit_sales"] / fam_recent.replace(0, np.nan)).fillna(0)

    if yoy_agg is not None and yoy_weight > 0:
        yoy_avg = (
            (yoy_agg["sum"] / yoy_agg["count"].replace(0, np.nan))
            .rename("unit_sales")
            .reset_index()
        )
        yoy_avg = yoy_avg.merge(items[["item_nbr", "family"]], on="item_nbr", how="left")
        fam_yoy = yoy_avg.groupby(["store_nbr", "family"])["unit_sales"].transform("sum")
        yoy_avg["yoy_share"] = (yoy_avg["unit_sales"] / fam_yoy.replace(0, np.nan)).fillna(0)

        merged_share = recent_avg[["store_nbr", "item_nbr", "family", "recent_share"]].merge(
            yoy_avg[["store_nbr", "item_nbr", "yoy_share"]],
            on=["store_nbr", "item_nbr"],
            how="left",
        )
        merged_share["yoy_share"] = merged_share["yoy_share"].fillna(merged_share["recent_share"])
        merged_share["share"] = (
            (1 - yoy_weight) * merged_share["recent_share"] + yoy_weight * merged_share["yoy_share"]
        )
    else:
        merged_share = recent_avg.rename(columns={"recent_share": "share"})

    merged_share["share"] = merged_share["share"].clip(lower=0.0, upper=1.0)
    return merged_share[["store_nbr", "item_nbr", "family", "share"]]


def recursive_forecast_mixed(
    combined_history: pd.DataFrame,
    test_dates: list,
    global_lgbm,
    global_prep,
    global_cat,
    w_lgbm: float,
    w_cat: float,
    local_models: dict,
    cluster_engineer=None,
    quality_filter=None,
) -> pd.DataFrame:
    """Thực hiện dự báo đệ quy từng ngày với cơ chế Smart Routing.

    quality_filter: callable(family) -> bool. Family có local model nhưng filter
    trả False sẽ rơi về Global Ensemble - khớp với định tuyến live của ml_service.
    """
    logger.info(">>> Starting recursive forecast (Smart Routing)...")
    combined = combined_history.copy()
    predictions = []
    cat_feature_names = getattr(global_cat, "feature_names_", None) if global_cat else None

    for current_date in sorted(test_dates):
        current_date_ts = pd.to_datetime(current_date)
        # Cửa sổ trượt: chỉ giữ 60 ngày gần nhất cho lag/rolling thay vì toàn bộ lịch sử (§3.2)
        window_start = current_date_ts - pd.Timedelta(days=RECURSION_WINDOW_DAYS)
        temp_combined = combined[
            (combined["date"] <= current_date_ts)
            & (combined["date"] > window_start)
        ].copy()

        # Tiền xử lý feature (truyền an toàn cluster_engineer)
        featured = engineer_features(
            temp_combined,
            cluster_engineer=cluster_engineer,
            fit_cluster=False,
        )

        day_featured = featured[featured["date"] == current_date_ts].copy()
        if day_featured.empty:
            logger.warning(f"    [WARN] No features generated for {current_date_ts.date()}. Skipping!")
            continue

        day_featured["predicted_target"] = 0.0

        for family, group_df in day_featured.groupby("family"):
            group_idx = group_df.index

            if family in local_models and (quality_filter is None or quality_filter(family)):
                # Tuyến 1: Local Model
                local_prep = local_models[family]["preprocessor"]
                local_model = local_models[family]["model"]

                X_local = local_prep.transform(group_df)
                preds = np.expm1(local_model.predict(X_local))
                day_featured.loc[group_idx, "predicted_target"] = np.clip(preds, 0, None)
            else:
                # Tuyến 2: Fallback Global Ensemble
                if global_lgbm is not None and global_prep is not None:
                    X_lgb = global_prep.transform(group_df)
                    preds_lgb = np.expm1(global_lgbm.predict(X_lgb))

                    if global_cat is not None and cat_feature_names:
                        X_cat = group_df.reindex(columns=cat_feature_names, fill_value=0)
                        cat_cols = ["family", "city", "state", "type", "holiday_type", "cluster_family_id"]
                        for col in cat_cols:
                            if col in X_cat.columns:
                                X_cat[col] = X_cat[col].astype(str)
                        preds_cat = np.expm1(global_cat.predict(X_cat))
                        preds = w_lgbm * preds_lgb + w_cat * preds_cat
                    else:
                        preds = preds_lgb

                    day_featured.loc[group_idx, "predicted_target"] = np.clip(preds, 0, None)
                else:
                    logger.error(f"    [ERROR] No model found for family: {family}")

        pred_df = day_featured[["store_nbr", "family", "date", "predicted_target"]].copy()
        predictions.append(pred_df)

        # Cập nhật kết quả dự báo vào lịch sử để làm lag cho các ngày kế tiếp
        pred_map = day_featured.set_index(["store_nbr", "family"])["predicted_target"]
        mask = combined["date"] == current_date_ts

        idx_tuples = list(zip(combined.loc[mask, "store_nbr"], combined.loc[mask, "family"]))
        combined.loc[mask, "target"] = pred_map.reindex(idx_tuples).values

    logger.info(">>> Recursive forecast complete!")

    if not predictions:
        logger.error(">>> ERROR: predictions list is empty!")
        return pd.DataFrame(columns=["store_nbr", "family", "date", "predicted_target"])

    return pd.concat(predictions, ignore_index=True)


# ====================== HÀM THỰC THI CHÍNH ======================
def build_submission_local():
    logger.info(">>> 1. Loading Models & Metadata...")

    # Nạp Local Models
    if not LOCAL_MODELS_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {LOCAL_MODELS_PATH}. Hãy chạy train_local_models.py trước!")
    local_models = joblib.load(LOCAL_MODELS_PATH)
    logger.info(f">>> Loaded {len(local_models)} local family models.")

    # Nạp Cluster Engineer (Kiểm tra an toàn trạng thái fit)
    cluster_engineer = None
    if CLUSTER_ENGINEER_PATH.exists():
        try:
            loaded_cluster = joblib.load(CLUSTER_ENGINEER_PATH)
            if getattr(loaded_cluster, "cluster_stats", None) is not None:
                cluster_engineer = loaded_cluster
                logger.info(">>> Cluster Feature Engineer loaded successfully.")
        except Exception as e:
            logger.warning(f">>> Could not load cluster_engineer: {e}. Running without it.")

    # Nạp Global Models phục vụ Fallback
    global_lgbm = joblib.load(LGBM_MODEL_PATH) if LGBM_MODEL_PATH.exists() else None
    global_prep = joblib.load(PREPROCESSOR_PATH) if PREPROCESSOR_PATH.exists() else None
    global_cat = None
    w_lgbm, w_cat = 1.0, 0.0

    ensemble_meta = {}
    if ENSEMBLE_META_PATH.exists():
        try:
            with open(ENSEMBLE_META_PATH) as f:
                ensemble_meta = json.load(f)
            w_lgbm = float(ensemble_meta.get("lgbm_weight", 0.5))
            w_cat = float(ensemble_meta.get("catboost_weight", 0.5))
        except Exception as e:
            logger.warning(f">>> Could not read {ENSEMBLE_META_PATH.name}: {e}")

    if CATBOOST_MODEL_PATH.exists():
        try:
            from catboost import CatBoostRegressor
            global_cat = CatBoostRegressor()
            global_cat.load_model(str(CATBOOST_MODEL_PATH))
        except Exception as e:
            logger.warning(f">>> Could not load CatBoost model: {e}")

    # Định tuyến theo chất lượng (§3.3): local model có RMSLE validation vượt
    # ngưỡng (1.3 × RMSLE ensemble tham chiếu) sẽ rơi về Global Ensemble -
    # đồng bộ với smart routing của ml_service.
    local_metrics: dict = {}
    metrics_csv = MODELS_DIR / "local_models_metrics.csv"
    if metrics_csv.exists():
        try:
            df_m = pd.read_csv(metrics_csv)
            local_metrics = dict(zip(df_m["family"], df_m["rmsle"].astype(float)))
        except Exception as e:
            logger.warning(f">>> Could not read {metrics_csv.name}: {e}")

    ref_rmsle = float(ensemble_meta.get("avg_blend_rmsle", 0.357))

    def _quality_filter(family: str) -> bool:
        rmsle = local_metrics.get(family)
        if rmsle is None:
            return True
        return rmsle <= 1.3 * ref_rmsle

    logger.info(">>> 2. Loading Train History & Test Skeleton...")
    train_family = load_data()
    recent_history = train_family[
        train_family["date"] >= train_family["date"].max() - pd.Timedelta(days=90)
    ].copy()

    test_item, test_family_skeleton = build_test_family_skeleton()
    test_dates = sorted(test_item["date"].unique())
    test_family_skeleton["target"] = np.nan

    combined_history = pd.concat([recent_history, test_family_skeleton], ignore_index=True)

    # 3. Chạy Recursive Forecast cấp Family
    family_preds = recursive_forecast_mixed(
        combined_history=combined_history,
        test_dates=test_dates,
        global_lgbm=global_lgbm,
        global_prep=global_prep,
        global_cat=global_cat,
        w_lgbm=w_lgbm,
        w_cat=w_cat,
        local_models=local_models,
        cluster_engineer=cluster_engineer,
        quality_filter=_quality_filter,
    )

    if family_preds.empty:
        logger.error(">>> Failed to generate predictions. Aborting!")
        return None

    # 4. Phân rã xuống cấp Item (Disaggregation)
    logger.info(">>> 3. Disaggregating predictions to Item Level...")
    item_share = compute_item_share(lookback_recent=45, yoy_weight=0.0)

    merged = test_item.merge(family_preds, on=["store_nbr", "family", "date"], how="left")
    merged = merged.merge(item_share, on=["store_nbr", "item_nbr", "family"], how="left")

    merged["unit_sales"] = (merged["predicted_target"] * merged["share"].fillna(0)).clip(lower=0)

    # 5. Xuất File Submission
    submission = merged[["id", "unit_sales"]].sort_values("id").reset_index(drop=True)
    submission["unit_sales"] = submission["unit_sales"].round(4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_SUBMISSION_PATH, index=False)

    logger.info("\n" + "=" * 60)
    logger.info(f">>> [SUCCESS] Submission saved to: {OUTPUT_SUBMISSION_PATH}")
    logger.info(f">>> Submission shape: {submission.shape}")
    logger.info("=" * 60)
    print(submission.head(10))

    return submission


if __name__ == "__main__":
    build_submission_local()
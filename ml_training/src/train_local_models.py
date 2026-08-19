"""
train_local_models.py (v2.2 - Production Ready)
Train Local Models (LightGBM) cho toàn bộ 33 ngành hàng (Family).

Cải tiến:
- Khớp tuyệt đối với preprocessor.py (tích hợp Cluster Features toàn cục).
- Tự động hạ `min_child_samples` linh hoạt theo dung lượng từng ngành hàng.
- Retrain on Full Data với best_iteration + buffer.
- Đóng gói toàn bộ artifacts nén (.pkl), bảng metrics (.csv) và manifest (.json).
"""

import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ====================== CẤU HÌNH ĐƯỜNG DẪN ======================
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
PROJECT_ROOT = BASE_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from data_loader import load_data
from preprocessor import engineer_features, build_preprocessor
from cluster_features import ClusterFeatureEngineer

MODELS_DIR = PROJECT_ROOT / "ml_service" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ====================== CẤU HÌNH LOGGING ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ====================== THAM SỐ HUẤN LUYỆN ======================
MIN_ROWS_TO_TRAIN = 1000    # Ngưỡng tối thiểu để train model riêng
VAL_DAYS = 28              # 28 ngày validation (Walk-forward)
RETRAIN_ON_FULL_DATA = True
RETRAIN_ITER_BUFFER = 1.1

BASE_LGBM_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def split_walk_forward(df_family: pd.DataFrame, val_days: int):
    """Chia tập walk-forward: val_days ngày cuối làm validation."""
    max_date = df_family["date"].max()
    cutoff = max_date - pd.Timedelta(days=val_days)
    train_df = df_family[df_family["date"] <= cutoff].copy()
    val_df = df_family[df_family["date"] > cutoff].copy()
    return train_df, val_df


def train_one_family(family: str, df_family: pd.DataFrame):
    """Huấn luyện mô hình cho 1 ngành hàng cụ thể."""
    train_df, val_df = split_walk_forward(df_family, VAL_DAYS)
    if train_df.empty or val_df.empty:
        raise ValueError("Tập Train hoặc Validation rỗng sau khi chia Walk-forward")

    # 1. Tiền xử lý dữ liệu độc lập (Gọi thẳng build_preprocessor)
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)

    # 2. Xử lý target log1p
    y_train = np.log1p(train_df["target"].clip(lower=0))
    y_val_log = np.log1p(val_df["target"].clip(lower=0))

    # 3. Tinh chỉnh tham số linh hoạt theo độ lớn dữ liệu
    params = BASE_LGBM_PARAMS.copy()
    if len(train_df) < 5000:
        params["min_child_samples"] = 5
        params["num_leaves"] = 15

    # 4. Tìm số vòng lặp tối ưu qua Early Stopping
    model = LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val_log)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )
    best_iter = model.best_iteration_ or params["n_estimators"]

    # 5. Đo lường RMSLE
    val_preds = model.predict(X_val)
    rmsle = float(np.sqrt(mean_squared_error(y_val_log, val_preds)))

    # 6. Retrain trên 100% dữ liệu (train + val)
    if RETRAIN_ON_FULL_DATA:
        final_iter = max(10, int(best_iter * RETRAIN_ITER_BUFFER))
        full_preprocessor = build_preprocessor()
        X_full = full_preprocessor.fit_transform(df_family)
        y_full = np.log1p(df_family["target"].clip(lower=0))

        final_model = LGBMRegressor(**{**params, "n_estimators": final_iter})
        final_model.fit(X_full, y_full)
        model, preprocessor = final_model, full_preprocessor

    artifact = {
        "model": model,
        "preprocessor": preprocessor,
        "best_iteration": int(best_iter),
        "trained_until": str(df_family["date"].max().date()),
    }
    metrics = {
        "family": family,
        "n_rows": len(df_family),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "best_iteration": int(best_iter),
        "rmsle": round(rmsle, 5),
        "retrained_on_full": RETRAIN_ON_FULL_DATA,
    }
    return artifact, metrics


def train_local_models():
    logger.info(">>> Loading data...")
    df = load_data()
    
    # Feature Engineering toàn cục 1 lần (bao gồm Cluster Features)
    logger.info(">>> Engineering features (Including Cluster Features)...")
    cluster_eng = ClusterFeatureEngineer(smoothing=10.0)
    df = engineer_features(df, cluster_engineer=cluster_eng, fit_cluster=True)

    all_families = sorted(df["family"].dropna().unique().tolist())
    logger.info(f">>> Found {len(all_families)} families. Starting training loop...\n")

    local_models = {}
    metrics_rows = []
    skipped_families = []
    failed_families = []

    for idx, family in enumerate(all_families, 1):
        logger.info(f"[{idx}/{len(all_families)}] Processing Family: {family}")
        df_family = df[df["family"] == family].copy()

        if len(df_family) < MIN_ROWS_TO_TRAIN:
            logger.warning(f"    [SKIP] Data too small ({len(df_family)} rows). Will fallback to Global Model.")
            skipped_families.append(family)
            continue

        try:
            artifact, metrics = train_one_family(family, df_family)
            local_models[family] = artifact
            metrics_rows.append(metrics)
            logger.info(
                f"    [SUCCESS] RMSLE: {metrics['rmsle']:.5f} | "
                f"Best Iter: {metrics['best_iteration']}\n"
            )
        except Exception:
            logger.exception(f"    [FAILED] Error training {family}")
            failed_families.append(family)
            continue

    # ====================== LƯU ARTIFACTS ======================
    save_path = MODELS_DIR / "local_lgbm_models.pkl"
    joblib.dump(local_models, save_path, compress=3)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows).sort_values("rmsle")
        metrics_path = MODELS_DIR / "local_models_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Metrics saved to: {metrics_path}")

    metadata = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "val_days": VAL_DAYS,
        "min_rows_to_train": MIN_ROWS_TO_TRAIN,
        "retrain_on_full_data": RETRAIN_ON_FULL_DATA,
        "total_families": len(all_families),
        "trained_count": len(local_models),
        "trained_families": sorted(local_models.keys()),
        "skipped_families": sorted(skipped_families),
        "failed_families": sorted(failed_families),
    }
    meta_path = MODELS_DIR / "local_models_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # ====================== BÁO CÁO ======================
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING LOCAL MODELS COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Successfully trained: {len(local_models)} / {len(all_families)} families")
    logger.info(f"Skipped: {len(skipped_families)} families {skipped_families}")
    logger.info(f"Failed: {len(failed_families)} families {failed_families}")
    logger.info(f"Artifacts: {save_path}")
    logger.info(f"Metadata:  {meta_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    train_local_models()
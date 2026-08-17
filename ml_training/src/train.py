"""
src/train.py
Huấn luyện mô hình Ensemble (LGBM + CatBoost) dự báo Sales (family-level).
- Walk-forward validation trên dữ liệu chuỗi thời gian.
- Tích hợp ClusterFeatureEngineer (chống rò rỉ dữ liệu / data leakage).
- Tìm trọng số tối ưu (Blending weight) tối thiểu hóa RMSLE.
- Huấn luyện Final Models trên toàn bộ dữ liệu & tự động export sang ml_service.
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_absolute_error, mean_squared_log_error

# ====================== 1. THIẾT LẬP IMPORT & ĐƯỜNG DẪN ======================
SRC_DIR = Path(__file__).resolve().parent               # ml_training/src/
BASE_DIR = SRC_DIR.parent                              # ml_training/
PROJECT_ROOT = BASE_DIR.parent                         # SIC-AI-PROJECT-/

# Thêm SRC_DIR vào sys.path ĐẦU TIÊN để nhận diện các module nội bộ
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import các module nội bộ sau khi sys.path đã sẵn sàng
from cluster_features import ClusterFeatureEngineer
from data_loader import load_data
from preprocessor import build_preprocessor, engineer_features

# Danh sách các thư mục lưu trữ Model (Lưu song song cả offline và service inference)
LOCAL_MODELS_DIR = BASE_DIR / "models"
SERVICE_MODELS_DIR = PROJECT_ROOT / "ml_service" / "models"

LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
SERVICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ====================== 2. THAM SỐ HUẤN LUYỆN ======================
TOTAL_DAYS = 730
VAL_DAYS = 28
N_SPLITS = 3
MIN_TRAIN_DAYS = 365

LGBM_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

CATBOOST_PARAMS = {
    "iterations": 3000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": False,
    "loss_function": "RMSE",
    "early_stopping_rounds": 50,
    "task_type": "CPU",
}

CAT_FEATURES = ["store_nbr", "family", "city", "state", "type", "holiday_type"]
NUM_FEATURES = [
    "transactions_lag1",
    "sales_lag7",
    "sales_lag14",
    "sales_lag28",
    "sales_rolling_mean7",
    "sales_rolling_mean14",
    "sales_rolling_mean30",
    "dayofweek",
    "month",
    "is_weekend",
    "oil_price",
    "cluster",
    "perishable",
    "is_holiday_lead1",
    "is_holiday_lead2",
    "onpromotion",
    "is_earthquake_period",
    "is_holiday",
]
CAT_ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


# ====================== 3. HÀM PHỤ TRỢ ======================
def prepare_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    print(">>> Loading data...")
    df = load_data(force_rebuild=force_rebuild)

    print(">>> Engineering base features...")
    df = engineer_features(df)
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    date_range = (df["date"].max() - df["date"].min()).days + 1
    print(f">>> Data date range: {df['date'].min().date()} → {df['date'].max().date()} ({date_range} days)")
    if date_range < TOTAL_DAYS - 30:
        print(f"Warning: Data has only {date_range} days, expected ~{TOTAL_DAYS} days")

    return df


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    val_days: int = 28,
    min_train_days: int = 365,
):
    dates = sorted(df["date"].unique())
    if not dates:
        raise ValueError("No dates found in dataframe")

    max_date, min_date = dates[-1], dates[0]
    total_days = (max_date - min_date).days + 1

    print(f">>> Total data span: {total_days} days ({min_date.date()} → {max_date.date()})")
    print(f">>> Walk-forward config: {n_splits} folds, val={val_days} days, min_train={min_train_days} days")

    temp_splits = []
    for i in range(n_splits):
        val_end = max_date - pd.Timedelta(days=i * val_days)
        val_start = val_end - pd.Timedelta(days=val_days - 1)
        train_end = val_start - pd.Timedelta(days=1)

        train_days = (train_end - min_date).days + 1
        if train_days < min_train_days:
            print(f"⚠️  Skip fold {i+1}: train period too short ({train_days} days < {min_train_days})")
            break

        train_mask = (df["date"] >= min_date) & (df["date"] <= train_end)
        val_mask = (df["date"] >= val_start) & (df["date"] <= val_end)

        fold_info = {
            "train_start": min_date.date(),
            "train_end": train_end.date(),
            "train_days": train_days,
            "val_start": val_start.date(),
            "val_end": val_end.date(),
            "val_days": val_days,
        }
        temp_splits.append((train_mask, val_mask, fold_info))

    temp_splits = list(reversed(temp_splits))

    splits = []
    for idx, (train_mask, val_mask, fold_info) in enumerate(temp_splits, start=1):
        fold_info["fold"] = idx
        splits.append((train_mask, val_mask, fold_info))
        print(
            f"    Fold {idx}: train {fold_info['train_start']} → {fold_info['train_end']} "
            f"({fold_info['train_days']}d) | val {fold_info['val_start']} → {fold_info['val_end']} ({fold_info['val_days']}d)"
        )

    if not splits:
        raise ValueError("No valid walk-forward splits could be created.")

    return splits


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.maximum(y_true, 0)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmsle": float(rmsle), "mae": float(mae)}


def find_best_weight(y_true: np.ndarray, lgbm_pred: np.ndarray, cat_pred: np.ndarray) -> tuple[float, float]:
    def loss_func(w):
        blend = w * lgbm_pred + (1 - w) * cat_pred
        return evaluate(y_true, blend)["rmsle"]

    res = minimize_scalar(loss_func, bounds=(0.0, 1.0), method="bounded")
    return float(res.x), float(res.fun)


def save_artifacts(models_dir: Path, final_lgbm, final_cat, final_prep, final_cluster, meta_dict):
    """Lưu toàn bộ artifacts vào thư mục chỉ định."""
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_lgbm, models_dir / "lgbm_model.pkl")
    final_cat.save_model(str(models_dir / "catboost_model.cbm"))
    joblib.dump(final_prep, models_dir / "preprocessor.pkl")
    joblib.dump(final_cluster, models_dir / "cluster_engineer.pkl")
    with open(models_dir / "ensemble_meta.json", "w") as f:
        json.dump(meta_dict, f, indent=2, default=str)


# ====================== 4. LUỒNG HUẤN LUYỆN CHÍNH ======================
def train_ensemble(force_rebuild: bool = False):
    df = prepare_dataset(force_rebuild=force_rebuild)

    splits = walk_forward_splits(
        df, n_splits=N_SPLITS, val_days=VAL_DAYS, min_train_days=MIN_TRAIN_DAYS
    )
    print(f">>> Số fold walk-forward thực tế: {len(splits)}\n")

    fold_summaries = []
    fold_weights = []
    all_lgbm_iters = []
    all_cat_iters = []

    for train_mask, val_mask, fold_info in splits:
        fold_num = fold_info["fold"]
        print(f"===== Fold {fold_num}/{len(splits)} | Val: {fold_info['val_start']} → {fold_info['val_end']} =====")

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()

        print(f"    Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")

        # CLUSTER FEATURES: Fit CHỈ trên train để chống rò rỉ dữ liệu
        cluster_engineer = ClusterFeatureEngineer(smoothing=100.0)
        train_df = cluster_engineer.fit_transform(train_df, target_col="target")
        val_df = cluster_engineer.transform(val_df)

        y_train_log = np.log1p(train_df["target"].values)
        y_val_log = np.log1p(val_df["target"].values)
        y_val_true = val_df["target"].values

        # LIGHTGBM TRAINING
        fold_preprocessor = build_preprocessor()
        X_train_lgb = fold_preprocessor.fit_transform(train_df)
        X_val_lgb = fold_preprocessor.transform(val_df)

        lgbm = LGBMRegressor(**LGBM_PARAMS)
        lgbm.fit(
            X_train_lgb,
            y_train_log,
            eval_set=[(X_val_lgb, y_val_log)],
            eval_metric="rmse",
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(period=0),
            ],
        )
        lgbm_pred = np.clip(np.expm1(lgbm.predict(X_val_lgb)), 0, None)
        lgb_best_iter = lgbm.best_iteration_ if lgbm.best_iteration_ else LGBM_PARAMS["n_estimators"]
        all_lgbm_iters.append(lgb_best_iter)

        # CATBOOST TRAINING
        cat_cols = [c for c in CAT_FEATURES if c in train_df.columns]
        all_features = [c for c in CAT_ALL_FEATURES if c in train_df.columns]

        train_pool = Pool(data=train_df[all_features], label=y_train_log, cat_features=cat_cols)
        val_pool = Pool(data=val_df[all_features], label=y_val_log, cat_features=cat_cols)

        cat = CatBoostRegressor(**CATBOOST_PARAMS)
        cat.fit(train_pool, eval_set=val_pool, verbose=False)
        cat_pred = np.clip(np.expm1(cat.predict(val_df[all_features])), 0, None)

        cat_best_iter = (
            (cat.get_best_iteration() + 1)
            if cat.get_best_iteration() is not None
            else CATBOOST_PARAMS["iterations"]
        )
        all_cat_iters.append(cat_best_iter)

        # TRỌNG SỐ BLEND TỐI ƯU
        best_w, blend_rmsle = find_best_weight(y_val_true, lgbm_pred, cat_pred)
        fold_weights.append(best_w)

        lgb_metrics = evaluate(y_val_true, lgbm_pred)
        cat_metrics = evaluate(y_val_true, cat_pred)
        blend_mae = evaluate(y_val_true, best_w * lgbm_pred + (1 - best_w) * cat_pred)["mae"]

        fold_record = {
            "fold": fold_num,
            "lgbm_rmsle": lgb_metrics["rmsle"],
            "lgbm_mae": lgb_metrics["mae"],
            "lgbm_best_iter": lgb_best_iter,
            "catboost_rmsle": cat_metrics["rmsle"],
            "catboost_mae": cat_metrics["mae"],
            "catboost_best_iter": cat_best_iter,
            "blend_weight_lgbm": round(best_w, 4),
            "blend_rmsle": round(blend_rmsle, 5),
            "blend_mae": round(blend_mae, 2),
        }
        fold_summaries.append(fold_record)

        print(f"    LGBM     RMSLE: {lgb_metrics['rmsle']:.5f} | Best Iter: {lgb_best_iter}")
        print(f"    CatBoost RMSLE: {cat_metrics['rmsle']:.5f} | Best Iter: {cat_best_iter}")
        print(f"    Blend    RMSLE: {blend_rmsle:.5f} | Weight LGBM: {best_w:.3f}")

    # ====================== TỔNG KẾT VALIDATION ======================
    avg_weight = float(np.mean(fold_weights))
    avg_lgb_iter = max(1, int(np.mean(all_lgbm_iters)))
    avg_cat_iter = max(1, int(np.mean(all_cat_iters)))
    avg_blend_rmsle = float(np.mean([f["blend_rmsle"] for f in fold_summaries]))

    print("\n" + "=" * 65)
    print("ENSEMBLE WALK-FORWARD SUMMARY")
    print("=" * 65)
    for f in fold_summaries:
        print(
            f"  Fold {f['fold']}: LGBM={f['lgbm_rmsle']:.5f} | "
            f"Cat={f['catboost_rmsle']:.5f} | Blend={f['blend_rmsle']:.5f} (w_LGBM={f['blend_weight_lgbm']:.2f})"
        )
    print("-" * 65)
    print(f"  Average Optimal LGBM Weight : {avg_weight:.3f} (CatBoost: {1 - avg_weight:.3f})")
    print(f"  Average Blend RMSLE        : {avg_blend_rmsle:.5f}")
    print(f"  Average LGBM Iterations    : {avg_lgb_iter}")
    print(f"  Average CatBoost Iterations: {avg_cat_iter}")
    print("=" * 65)

    # ====================== TRAIN FINAL MODELS ======================
    print("\n>>> Fitting Cluster Features on FULL data...")
    cluster_engineer_final = ClusterFeatureEngineer(smoothing=10.0)
    df_full = cluster_engineer_final.fit_transform(df, target_col="target")
    y_full_log = np.log1p(df_full["target"].values)

    # 1. Final LightGBM
    print(f"\n>>> Training FINAL LightGBM on {len(df_full):,} samples (n_estimators={avg_lgb_iter})...")
    final_preprocessor = build_preprocessor()
    X_full_lgb = final_preprocessor.fit_transform(df_full)

    final_lgb_params = LGBM_PARAMS.copy()
    final_lgb_params["n_estimators"] = avg_lgb_iter
    final_lgbm = LGBMRegressor(**final_lgb_params)
    final_lgbm.fit(X_full_lgb, y_full_log)

    # 2. Final CatBoost
    print(f">>> Training FINAL CatBoost on {len(df_full):,} samples (iterations={avg_cat_iter})...")
    cat_cols_full = [c for c in CAT_FEATURES if c in df_full.columns]
    all_features_full = [c for c in CAT_ALL_FEATURES if c in df_full.columns]

    full_pool = Pool(data=df_full[all_features_full], label=y_full_log, cat_features=cat_cols_full)
    final_cat_params = CATBOOST_PARAMS.copy()
    final_cat_params["iterations"] = avg_cat_iter
    final_cat_params["early_stopping_rounds"] = None

    final_cat = CatBoostRegressor(**final_cat_params)
    final_cat.fit(full_pool, verbose=False)

    # ====================== XUẤT ARTIFACTS SANG CẢ 2 NƠI ======================
    ensemble_meta = {
        "lgbm_weight": round(avg_weight, 4),
        "catboost_weight": round(1.0 - avg_weight, 4),
        "avg_blend_rmsle": round(avg_blend_rmsle, 5),
        "avg_lgbm_iteration": avg_lgb_iter,
        "avg_catboost_iteration": avg_cat_iter,
        "folds": fold_summaries,
    }

    print(f"\n>>> Exporting artifacts to {LOCAL_MODELS_DIR}...")
    save_artifacts(LOCAL_MODELS_DIR, final_lgbm, final_cat, final_preprocessor, cluster_engineer_final, ensemble_meta)

    if SERVICE_MODELS_DIR != LOCAL_MODELS_DIR:
        print(f">>> Exporting artifacts to {SERVICE_MODELS_DIR}...")
        save_artifacts(SERVICE_MODELS_DIR, final_lgbm, final_cat, final_preprocessor, cluster_engineer_final, ensemble_meta)

    print("\n>>> All models trained and artifacts exported successfully!")
    return final_lgbm, final_cat, avg_weight


if __name__ == "__main__":
    train_ensemble(force_rebuild=False)
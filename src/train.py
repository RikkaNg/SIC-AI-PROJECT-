"""
train.py
Huấn luyện model dự báo sales (family-level) với Walk-forward validation trên 2 năm dữ liệu.
Tích hợp trực tiếp với data_loader.py và preprocessor.py hiện có.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import json
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

from src.data_loader import load_data
from src.preprocessor import engineer_features, build_preprocessor

# ====================== CẤU HÌNH ======================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dữ liệu 2 năm = ~730 ngày
TOTAL_DAYS = 730
VAL_DAYS = 28          # 4 tuần validation mỗi fold
N_SPLITS = 3           # 3 folds walk-forward
MIN_TRAIN_DAYS = 365   # Tối thiểu 1 năm train

# LightGBM hyperparameters
LGBM_PARAMS = {
    "n_estimators": 3500,
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


# ====================== HELPER FUNCTIONS ======================
def prepare_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    """Load data + feature engineering."""
    print(">>> Loading data...")
    df = load_data(force_rebuild=force_rebuild)

    print(">>> Engineering features...")
    df = engineer_features(df)
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Đảm bảo dữ liệu đủ 2 năm
    date_range = (df["date"].max() - df["date"].min()).days + 1
    print(f">>> Data date range: {df['date'].min().date()} → {df['date'].max().date()} ({date_range} days)")
    if date_range < TOTAL_DAYS - 30:  # Cho phép sai số ~1 tháng
        print(f"Warning: Data has only {date_range} days, expected ~{TOTAL_DAYS} days (2 years)")

    return df


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    val_days: int = 28,
    min_train_days: int = 365,
):
    """
    Tạo các split walk-forward trên dữ liệu 2 năm.
    Trả về danh sách các tuple (train_mask, val_mask, fold_info) theo thứ tự thờigian
    từ cũ nhất đến gần nhất (fold 1 = cũ nhất, fold cuối = gần nhất).
    """
    dates = sorted(df["date"].unique())
    if not dates:
        raise ValueError("No dates found in dataframe")

    max_date = dates[-1]
    min_date = dates[0]
    total_days = (max_date - min_date).days + 1

    print(f">>> Total data span: {total_days} days ({min_date.date()} → {max_date.date()})")
    print(f">>> Walk-forward config: {n_splits} folds, val={val_days} days, min_train={min_train_days} days")

    # Tạo các fold từ gần nhất lùi về quá khứ, sau đó đảo ngược để có thứ tự cũ → mới
    temp_splits = []
    for i in range(n_splits):
        # Validation window: lùi dần từ max_date
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

    # Đảo ngược để có thứ tự cũ nhất → gần nhất
    temp_splits = list(reversed(temp_splits))

    # Gán số fold theo thứ tự mới
    splits = []
    for idx, (train_mask, val_mask, fold_info) in enumerate(temp_splits, start=1):
        fold_info["fold"] = idx
        splits.append((train_mask, val_mask, fold_info))
        print(f"    Fold {idx}: train {fold_info['train_start']}→{fold_info['train_end']} "
              f"({fold_info['train_days']}d) | val {fold_info['val_start']}→{fold_info['val_end']} ({fold_info['val_days']}d)")

    if not splits:
        raise ValueError("No valid walk-forward splits could be created. Increase data size or reduce min_train_days.")

    return splits


def evaluate(y_true, y_pred) -> dict:
    """
    Tính RMSLE và MAE.

    RMSLE = sqrt(mean((log(1+y_true) - log(1+y_pred))^2))
    sklearn.mean_squared_log_error đã tự động cộng 1 bên trong,
    nên KHÔNG cần truyền y_true+1, y_pred+1.
    """
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.maximum(y_true, 0)

    # sklearn tính: mean((log(1 + y_true) - log(1 + y_pred))^2)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmsle": rmsle, "mae": mae}


# ====================== MAIN TRAINING ======================
def train(force_rebuild: bool = False):
    df = prepare_dataset(force_rebuild=force_rebuild)

    print("\n>>> Building preprocessor...")
    preprocessor = build_preprocessor()

    splits = walk_forward_splits(
        df,
        n_splits=N_SPLITS,
        val_days=VAL_DAYS,
        min_train_days=MIN_TRAIN_DAYS
    )
    print(f">>> Số fold walk-forward thực tế: {len(splits)}\n")

    all_metrics = []
    best_iterations = []

    for train_mask, val_mask, fold_info in splits:
        fold_num = fold_info["fold"]
        print(f"===== Fold {fold_num}/{len(splits)} | Val: {fold_info['val_start']} → {fold_info['val_end']} =====")

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()

        print(f"    Train samples: {len(train_df):,} | Val samples: {len(val_df):,}")

        # Fit preprocessor CHỈ trên train → tránh leakage
        X_train = preprocessor.fit_transform(train_df)
        X_val = preprocessor.transform(val_df)

        # Log-transform target (phù hợp với sales right-skewed)
        y_train = np.log1p(train_df["target"].values)
        y_val_log = np.log1p(val_df["target"].values)

        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val_log)],
            eval_metric="rmse",
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(period=0),  # Tắt log để gọn
            ],
        )
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)

        metrics = evaluate(val_df["target"].values, y_pred)
        metrics["best_iteration"] = model.best_iteration_
        metrics["fold"] = fold_num
        all_metrics.append(metrics)
        best_iterations.append(model.best_iteration_)

        print(f"    RMSLE: {metrics['rmsle']:.5f} | MAE: {metrics['mae']:.2f} | Best iter: {model.best_iteration_}")

    # Tóm tắt kết quả walk-forward
    avg_rmsle = np.mean([m["rmsle"] for m in all_metrics])
    avg_mae = np.mean([m["mae"] for m in all_metrics])
    avg_best_iter = int(np.mean(best_iterations))

    print("\n" + "=" * 60)
    print("WALK-FORWARD SUMMARY (2-Year Data)")
    print("=" * 60)
    for m in all_metrics:
        print(f"  Fold {m['fold']}: RMSLE={m['rmsle']:.5f} | MAE={m['mae']:.2f} | BestIter={m['best_iteration']}")
    print(f"\n  Average RMSLE : {avg_rmsle:.5f}")
    print(f"  Average MAE   : {avg_mae:.2f}")
    print(f"  Average Best Iteration: {avg_best_iter}")
    print("=" * 60)

    # ===== Train FINAL model trên TOÀN BỘ 2 NĂM =====
    print("\n>>> Training FINAL model on FULL 2 YEARS of data...")

    # Fit preprocessor trên toàn bộ dữ liệu cho final model
    X_full = preprocessor.fit_transform(df)
    y_full = np.log1p(df["target"].values)  # Log transform target giống như lúc walk-forward

    # Cập nhật params: Khóa cứng n_estimators = avg_best_iter từ walk-forward
    final_params = LGBM_PARAMS.copy()
    final_params["n_estimators"] = avg_best_iter

    print(f"    Training on {len(df):,} samples with n_estimators={avg_best_iter}")
    print(f"    Date range: {df['date'].min().date()} → {df['date'].max().date()}")

    final_model = LGBMRegressor(**final_params)
    # Train trên toàn bộ dữ liệu, không dùng eval_set để tránh lãng phí dữ liệu
    final_model.fit(X_full, y_full)

    # Lưu model + preprocessor + metrics
    model_path = MODELS_DIR / "lgbm_model.pkl"
    prep_path = MODELS_DIR / "preprocessor.pkl"
    metrics_path = MODELS_DIR / "metrics.json"

    joblib.dump(final_model, model_path)
    joblib.dump(preprocessor, prep_path)

    # Lưu metrics walk-forward
    metrics_summary = {
        "avg_rmsle": float(avg_rmsle),
        "avg_mae": float(avg_mae),
        "avg_best_iteration": avg_best_iter,
        "folds": all_metrics
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2, default=str)

    print(f"\n>>> Saved model         → {model_path}")
    print(f">>> Saved preprocessor  → {prep_path}")
    print(f">>> Saved metrics       → {metrics_path}")
    print(f">>> Final model n_estimators: {avg_best_iter} (from {LGBM_PARAMS['n_estimators']} max)")
    print(">>> Training complete!")

    return final_model, preprocessor, all_metrics


if __name__ == "__main__":
    train(force_rebuild=False)
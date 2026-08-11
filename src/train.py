"""
train.py
Huấn luyện model dự báo sales (family-level) với Walk-forward validation.
Tích hợp trực tiếp với data_loader.py và preprocessor.py hiện có.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_log_error, mean_absolute_error

from src.data_loader import load_data
from src.preprocessor import engineer_features, build_preprocessor

# ====================== CẤU HÌNH ======================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Walk-forward settings
N_SPLITS = 3
VAL_DAYS = 28
MIN_TRAIN_DAYS = 90

# LightGBM hyperparameters
LGBM_PARAMS = {
    "n_estimators": 2000,
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
    return df


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 3,
    val_days: int = 28,
    min_train_days: int = 90,
):
    """
    Tạo các fold theo thời gian (không shuffle).
    Fold cuối cùng là fold gần hiện tại nhất.
    """
    dates = sorted(df["date"].unique())
    max_date = dates[-1]

    splits = []
    for i in range(n_splits):
        val_end = max_date - pd.Timedelta(days=i * val_days)
        val_start = val_end - pd.Timedelta(days=val_days - 1)
        train_end = val_start - pd.Timedelta(days=1)

        if (train_end - dates[0]).days < min_train_days:
            break

        train_mask = (df["date"] >= dates[0]) & (df["date"] <= train_end)
        val_mask = (df["date"] >= val_start) & (df["date"] <= val_end)

        splits.append((train_mask, val_mask, val_start, val_end))

    # Đảo ngược: fold 0 = cũ nhất, fold cuối = gần nhất
    return list(reversed(splits))


def evaluate(y_true, y_pred) -> dict:
    """Tính RMSLE và MAE. Sales không được âm."""
    y_pred = np.clip(y_pred, 0, None)
    # Tránh log(0)
    y_true = np.maximum(y_true, 0)

    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return {"rmsle": rmsle, "mae": mae}


# ====================== MAIN TRAINING ======================
def train(force_rebuild: bool = False):
    df = prepare_dataset(force_rebuild=force_rebuild)

    print(">>> Building preprocessor...")
    preprocessor = build_preprocessor()

    splits = walk_forward_splits(
        df, n_splits=N_SPLITS, val_days=VAL_DAYS, min_train_days=MIN_TRAIN_DAYS
    )
    print(f">>> Số fold walk-forward: {len(splits)}")

    all_metrics = []

    for fold, (train_mask, val_mask, val_start, val_end) in enumerate(splits):
        print(f"\n===== Fold {fold + 1}/{len(splits)} | Val: {val_start.date()} → {val_end.date()} =====")

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()

        # Fit preprocessor CHỈ trên train → tránh leakage
        X_train = preprocessor.fit_transform(train_df)
        X_val = preprocessor.transform(val_df)

        y_train = train_df["target"]
        y_val = val_df["target"]

        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train,
            y_train,
            eval_X=X_val, 
            eval_y=y_val,
            eval_metric="rmse",
            callbacks=[
                early_stopping(stopping_rounds=50, verbose=False),
                log_evaluation(period=100),
            ],
        )

        y_pred = model.predict(X_val)
        metrics = evaluate(y_val, y_pred)
        all_metrics.append(metrics)

        print(f"RMSLE: {metrics['rmsle']:.5f} | MAE: {metrics['mae']:.2f}")
        print(f"Best iteration: {model.best_iteration_}")

    # Tóm tắt kết quả walk-forward
    avg_rmsle = np.mean([m["rmsle"] for m in all_metrics])
    avg_mae = np.mean([m["mae"] for m in all_metrics])
    print("\n" + "=" * 50)
    print(f">>> Average RMSLE across folds: {avg_rmsle:.5f}")
    print(f">>> Average MAE   across folds: {avg_mae:.2f}")
    print("=" * 50)

    # ===== Train final model =====
    # Dùng toàn bộ data trừ 28 ngày cuối (giữ lại để đánh giá cuối)
    print("\n>>> Training final model on full data (except last 28 days)...")
    cutoff = df["date"].max() - pd.Timedelta(days=VAL_DAYS)
    final_train = df[df["date"] <= cutoff].copy()
    final_val = df[df["date"] > cutoff].copy()

    X_final = preprocessor.fit_transform(final_train)
    y_final = final_train["target"]

    final_model = LGBMRegressor(**LGBM_PARAMS)
    final_model.fit(
        X_final,
        y_final,
        eval_set=[(preprocessor.transform(final_val), final_val["target"])],
        eval_metric="rmse",
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=100),
        ],
    )

    # Đánh giá trên tập hold-out cuối
    y_pred_final = final_model.predict(preprocessor.transform(final_val))
    final_metrics = evaluate(final_val["target"], y_pred_final)
    print(f"\n>>> Final hold-out RMSLE: {final_metrics['rmsle']:.5f}")
    print(f">>> Final hold-out MAE   : {final_metrics['mae']:.2f}")

    # Lưu model + preprocessor
    model_path = MODELS_DIR / "lgbm_model.pkl"
    prep_path = MODELS_DIR / "preprocessor.pkl"

    joblib.dump(final_model, model_path)
    joblib.dump(preprocessor, prep_path)

    print(f"\n>>> Saved model         → {model_path}")
    print(f">>> Saved preprocessor  → {prep_path}")

    return final_model, preprocessor, all_metrics


if __name__ == "__main__":
    train(force_rebuild=False)
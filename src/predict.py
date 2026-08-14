"""
predict.py
Dự báo trên test.csv — xử lý recursive forecasting và disaggregation.
Tích hợp Cluster Features từ cluster_engineer.pkl.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

from src.data_loader import (
    RAW_DIR, load_data, load_raw_files, prepare_oil, prepare_holidays,
    merge_dimensions, add_earthquake_flag,
    aggregate_to_family_level,
)
from src.preprocessor import engineer_features
from src.cluster_features import ClusterFeatureEngineer

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "lgbm_model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"
CLUSTER_ENGINEER_PATH = BASE_DIR / "models" / "cluster_engineer.pkl"


def build_test_family_skeleton():
    """
    test.csv ở mức item — merge đúng các bước như train rồi gộp lên family.
    Đảm bảo test_family có cột 'cluster' để dùng cluster_engineer.
    """
    test = pd.read_csv(RAW_DIR / "test.csv", parse_dates=["date"])
    _, stores, items, oil, holidays, _ = load_raw_files()

    oil = prepare_oil(oil, test["date"].min(), test["date"].max())
    holidays = prepare_holidays(holidays)

    test = merge_dimensions(test, stores, items, oil, holidays)
    test = add_earthquake_flag(test)

    test_family = aggregate_to_family_level(
        test.assign(unit_sales=0, onpromotion=test["onpromotion"].fillna(False))
    )

    # Đảm bảo có cột cluster (từ stores) để dùng cluster_engineer
    if "cluster" not in test_family.columns:
        test_family = test_family.merge(
            stores[["store_nbr", "cluster"]].drop_duplicates(),
            on="store_nbr", how="left"
        )

    # Thêm cột transactions NaN để engineer_features không crash khi shift(1)
    test_family["transactions"] = np.nan

    return test.drop(columns=["unit_sales"], errors="ignore"), test_family


def compute_item_share(lookback_days: int = 90) -> pd.DataFrame:
    """Tỷ trọng doanh số trung bình của mỗi item trong family (cùng cửa hàng)."""
    print(">>> Computing item shares...")
    train_raw = pd.read_csv(
        RAW_DIR / "train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales"],
        parse_dates=["date"],
    )
    cutoff = train_raw["date"].max() - pd.Timedelta(days=lookback_days)
    recent = train_raw[train_raw["date"] >= cutoff]

    item_avg = recent.groupby(["store_nbr", "item_nbr"])["unit_sales"].mean().reset_index()
    item_avg = item_avg.rename(columns={"unit_sales": "item_avg"})

    items = pd.read_csv(RAW_DIR / "items.csv")
    item_avg = item_avg.merge(items[["item_nbr", "family"]], on="item_nbr", how="left")

    family_avg = item_avg.groupby(["store_nbr", "family"])["item_avg"].transform("sum")
    item_avg["share"] = (item_avg["item_avg"] / family_avg.replace(0, np.nan)).fillna(0)
    return item_avg[["store_nbr", "item_nbr", "family", "share"]]


def recursive_forecast(history_family: pd.DataFrame, test_dates, model, preprocessor, cluster_engineer) -> pd.DataFrame:
    """
    Dự báo tuần tự từng ngày. Cập nhật target dự báo để tính lag cho ngày tiếp theo.
    Thêm cluster features sau mỗi lần engineer_features.
    """
    print(">>> Starting recursive forecast...")
    combined = history_family.copy()
    predictions = []

    for current_date in sorted(test_dates):
        # Chỉ lấy dữ liệu <= current_date để tiết kiệm RAM
        temp_combined = combined[combined["date"] <= current_date].copy()
        featured = engineer_features(temp_combined)

        # === THÊM CLUSTER FEATURES ===
        featured = cluster_engineer.transform(featured)
        # ==============================

        day_featured = featured[featured["date"] == current_date].copy()

        if day_featured.empty:
            continue

        # Preprocessor transform
        X_day = preprocessor.transform(day_featured)

        # Dự báo (model train trên log1p -> expm1 về scale gốc)
        preds_log = model.predict(X_day)
        preds = np.expm1(preds_log)
        preds = np.clip(preds, 0, None)

        # Lưu kết quả
        pred_df = day_featured[["store_nbr", "family", "date"]].copy()
        pred_df["predicted_target"] = preds
        predictions.append(pred_df)

        # Cập nhật target an toàn bằng merge
        pred_map = day_featured[["store_nbr", "family"]].copy()
        pred_map["target"] = preds

        mask = combined["date"] == current_date
        combined_idx = combined[mask].merge(pred_map, on=["store_nbr", "family"], how="left").index
        combined.loc[combined_idx, "target"] = pred_map["target"].values

    print(">>> Recursive forecast complete!")
    return pd.concat(predictions, ignore_index=True)


def build_submission():
    print(">>> Loading model, preprocessor, and cluster engineer...")
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    cluster_engineer = joblib.load(CLUSTER_ENGINEER_PATH)

    # 1. Lấy lịch sử family-level gần nhất
    print(">>> Loading train history...")
    train_family = load_data()
    recent_history = train_family[
        train_family["date"] >= train_family["date"].max() - pd.Timedelta(days=90)
    ].copy()

    # 2. Khung family-level cho test
    print(">>> Building test skeleton...")
    test_item, test_family_skeleton = build_test_family_skeleton()
    test_dates = sorted(test_item["date"].unique())
    test_family_skeleton["target"] = np.nan

    # Nối lịch sử và test
    combined_history = pd.concat([recent_history, test_family_skeleton], ignore_index=True)

    # 3. Dự báo tuần tự
    family_preds = recursive_forecast(
        combined_history, test_dates, model, preprocessor, cluster_engineer
    )

    # 4. Disaggregation: phân bổ family → item
    print(">>> Disaggregating predictions to item level...")
    item_share = compute_item_share()

    merged = test_item.merge(
        family_preds, on=["store_nbr", "family", "date"], how="left"
    )
    merged = merged.merge(
        item_share, on=["store_nbr", "item_nbr", "family"], how="left"
    )

    merged["unit_sales"] = merged["predicted_target"] * merged["share"].fillna(0)
    merged["unit_sales"] = np.clip(merged["unit_sales"], 0, None)

    # 5. Submission
    submission = merged[["id", "unit_sales"]].sort_values("id").reset_index(drop=True)

    output_dir = BASE_DIR / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission.csv"
    submission.to_csv(output_path, index=False)

    print(f"\n>>> Submission saved to {output_path}")
    print(f">>> Shape: {submission.shape}")
    print(submission.head(10))

    return submission


if __name__ == "__main__":
    sub = build_submission()
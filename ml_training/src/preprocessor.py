"""
src/preprocessor.py
Feature engineering (pandas: groupby/shift/rolling) + encoding/imputation (sklearn Pipeline).
Tích hợp cluster features: Target Encoding + Interactions.

HỢP ĐỒNG SỬ DỤNG (chống data leakage):
- engineer_features() chỉ tạo feature TĨNH (lag, rolling, calendar, holiday, mùa vụ).
  Cluster target-encoding là feature CÓ STATE → hàm này KHÔNG BAO GIỜ fit.
- 2 cách gọi hợp lệ:
    1) train.py (walk-forward): df = engineer_features(df)
       → cluster features được fit/transform per-fold bằng ClusterFeatureEngineer
         (fit chỉ trên train của từng fold) — train.py đang làm đúng rồi.
    2) ml_service (inference): eng = joblib.load(".../cluster_engineer.pkl")
       → df = engineer_features(df, cluster_engineer=eng)
- Truyền engineer CHƯA fit → RuntimeError rõ ràng (thay vì TypeError mù mờ).
"""

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# ====================== CẤU HÌNH ĐƯỜNG DẪN ======================
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from cluster_features import ClusterFeatureEngineer

set_config(transform_output="pandas")

# ====================== CẤU HÌNH CỘT ======================
BOOL_COLS = ["onpromotion", "is_earthquake_period", "is_holiday", "is_back_to_school"]

# Cột số fill 0 nếu thiếu (lag/rolling đầu chuỗi + cluster features).
# ĐÃ BỔ SUNG sales_lag28, sales_rolling_mean14/30 để khớp NUM_FEATURES trong train.py:
# trước đây LGBM thiếu sales_lag28, CatBoost thiếu rolling14/30
# → 2 model học trên tập feature KHÁC NHAU.
COLS_FILL_ZERO = [
    "transactions_lag1",
    "sales_lag7", "sales_lag14", "sales_lag28",
    "sales_rolling_mean7", "sales_rolling_mean14", "sales_rolling_mean30",
    "sales_std7", "sales_std28",
    "cluster_mean_sales",
    "cluster_median_sales",
    "cluster_std_sales",
    "cluster_family_mean_sales",
    "cluster_promo_mean_sales",
    "cluster_promo_lift",
]

COLS_CATEGORICAL = ["store_nbr", "family", "city", "state", "type", "holiday_type"]

COLS_PASSTHROUGH = [
    "dayofweek", "month", "is_weekend",
    "oil_price", "cluster", "perishable",
    "is_holiday_lag1", "is_holiday_lag2",
    "is_holiday_lead1", "is_holiday_lead2",
    "is_tier1_cluster",
    "is_payday", "is_day_after_payday",
    "days_from_payday", "days_to_payday",
    "dayofmonth", "weekofmonth", "dayofyear", "is_month_start", "is_month_end",
] + BOOL_COLS

NORMAL_DAY = "Normal Day"

# ====================== BƯỚC 1: FEATURE ENGINEERING ======================
def add_date_features(df):
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df


def add_transactions_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo transactions_lag1 nếu data_loader chưa cung cấp.
    Lag tính theo (store, date) — phải dedup trước khi shift, vì df đang
    xếp theo (store, family, date): shift trực tiếp sẽ nhảy sang family khác.
    """
    if "transactions_lag1" in df.columns or "transactions" not in df.columns:
        return df
    trans = (
        df[["store_nbr", "date", "transactions"]]
        .drop_duplicates(["store_nbr", "date"])
        .sort_values(["store_nbr", "date"])
    )
    trans["transactions_lag1"] = trans.groupby("store_nbr")["transactions"].shift(1)
    df = df.merge(
        trans[["store_nbr", "date", "transactions_lag1"]],
        on=["store_nbr", "date"],
        how="left",
    )
    return df


def add_holiday_effects(df):
    """Tạo feature cho hiệu ứng trước/sau ngày lễ."""
    df = df.sort_values(["store_nbr", "family", "date"]).copy()
    df["is_holiday"] = (df["holiday_type"] != NORMAL_DAY).astype(int)

    df["is_holiday_lag1"] = df.groupby(["store_nbr", "family"])["is_holiday"].shift(1)
    df["is_holiday_lag2"] = df.groupby(["store_nbr", "family"])["is_holiday"].shift(2)
    df["is_holiday_lead1"] = df.groupby(["store_nbr", "family"])["is_holiday"].shift(-1)
    df["is_holiday_lead2"] = df.groupby(["store_nbr", "family"])["is_holiday"].shift(-2)

    holiday_cols = ["is_holiday_lag1", "is_holiday_lag2", "is_holiday_lead1", "is_holiday_lead2"]
    df[holiday_cols] = df[holiday_cols].fillna(0).astype(int)
    return df


def add_lag_features(df, lags=(7, 14, 28)):
    df = df.sort_values(["store_nbr", "family", "date"])
    for lag in lags:
        df[f"sales_lag{lag}"] = df.groupby(["store_nbr", "family"])["target"].shift(lag)
    return df


def add_rolling_features(df, windows=(7, 14, 30)):
    df = df.sort_values(["store_nbr", "family", "date"])
    for w in windows:
        df[f"sales_rolling_mean{w}"] = (
            df.groupby(["store_nbr", "family"])["target"]
            .transform(lambda s: s.shift(1).rolling(w).mean())
        )
    return df


def add_rolling_std_features(df, windows=(7, 28)):
    """Độ lệch chuẩn doanh số shift(1) — đo biến động ngắn/dài hạn cho reorder."""
    df = df.sort_values(["store_nbr", "family", "date"])
    for w in windows:
        df[f"sales_std{w}"] = (
            df.groupby(["store_nbr", "family"])["target"]
            .transform(lambda s: s.shift(1).rolling(w).std())
        )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lịch chi tiết + chu kỳ ngày lương Ecuador (trả lương ngày 15 và ngày cuối tháng).
    Cùng công thức với features_ext.add_extra_features — features_ext sẽ
    drop-then-recreate các cột trùng tên nên không xảy ra nhân đôi.
    """
    d = df["date"].dt
    df["dayofmonth"] = d.day
    df["weekofmonth"] = ((d.day - 1) // 7 + 1).astype("int16")
    df["dayofyear"] = d.dayofyear
    days_in_month = d.days_in_month
    df["is_payday"] = ((d.day == 15) | (d.day == days_in_month)).astype("int8")
    df["is_day_after_payday"] = ((d.day == 16) | (d.day == 1)).astype("int8")
    prev_pay = np.where(d.day >= 15, 15, 0)
    df["days_from_payday"] = (d.day - prev_pay).astype("int16")
    next_pay = np.where(d.day < 15, 15, days_in_month)
    df["days_to_payday"] = (next_pay - d.day).astype("int16")
    df["is_month_start"] = (d.day <= 3).astype("int8")
    df["is_month_end"] = (days_in_month - d.day <= 2).astype("int8")
    return df


def add_back_to_school_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo biến flag mùa tựu trường (Back-to-school) theo đặc thù 2 vùng của Ecuador:
    - Vùng Duyên hải (Costa): Cao điểm tháng 4 và 5.
    - Vùng Núi / Nội địa (Sierra / Oriente): Cao điểm tháng 8 và 9.
    """
    df = df.copy()

    COSTA_STATES = {
        "Guayas", "Manabi", "Los Rios", "El Oro",
        "Santa Elena", "Esmeraldas", "Santo Domingo de los Tsachilas", "Santo Domingo"
    }

    month = df["date"].dt.month if "month" not in df.columns else df["month"]

    if "state" in df.columns:
        is_costa = df["state"].isin(COSTA_STATES)
    else:
        is_costa = False

    costa_season = is_costa & month.isin([4, 5])
    sierra_season = (~is_costa) & month.isin([8, 9])

    df["is_back_to_school"] = (costa_season | sierra_season).astype(int)
    return df


def add_cluster_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo interaction features đơn giản từ cluster (không cần fit)."""
    df = df.copy()
    if "cluster_promo_mean_sales" in df.columns and "cluster_mean_sales" in df.columns:
        df["cluster_promo_lift"] = df["cluster_promo_mean_sales"] / (df["cluster_mean_sales"] + 1e-8)
    if "cluster" in df.columns:
        df["is_tier1_cluster"] = df["cluster"].isin({5, 11}).astype(int)
    return df


def engineer_features(
    df: pd.DataFrame,
    cluster_engineer: Optional[ClusterFeatureEngineer] = None,
) -> pd.DataFrame:
    """
    Feature engineering tĩnh + (tùy chọn) cluster features.

    Args:
        df: DataFrame từ load_data (dữ liệu lịch sử có cột `target`).
        cluster_engineer:
            - None (mặc định): bỏ qua cluster features — train.py sẽ
              fit/transform per-fold sau khi chia tập (chống leakage).
            - Instance ĐÃ FIT (vd load từ cluster_engineer.pkl ở ml_service):
              transform được áp dụng ngay tại đây.

    Lưu ý: hàm này KHÔNG tự fit ClusterFeatureEngineer. Fit trên toàn bộ df
    (trước khi chia fold) khiến target encoding chứa thông tin từ các khoảng
    validation tương lai → data leakage.
    """
    df = df.copy()
    df["onpromotion"] = df["onpromotion"].fillna(False).astype(int)
    df["is_earthquake_period"] = df["is_earthquake_period"].fillna(False).astype(int)

    df = add_date_features(df)
    df = add_calendar_features(df)
    df = add_transactions_lag(df)
    df = add_lag_features(df)
    df = add_back_to_school_feature(df)
    df = add_rolling_features(df, windows=(7, 14, 30))
    df = add_rolling_std_features(df, windows=(7, 28))
    df = add_holiday_effects(df)

    # ---- Payday flags gọn (bản đầy đủ ở add_calendar_features) ----
    # is_payday/is_day_after_payday đã tạo trong add_calendar_features.

    # ---- Cluster features: CHỈ transform, KHÔNG fit ----
    if cluster_engineer is not None:
        if getattr(cluster_engineer, "cluster_stats", None) is None:
            raise RuntimeError(
                "cluster_engineer truyền vào chưa được fit(). Hãy fit trên tập TRAIN "
                "của fold (hoặc load cluster_engineer.pkl), hoặc truyền None để "
                "train.py tự xử lý per-fold."
            )
        df = cluster_engineer.transform(df)

    df = add_cluster_interactions(df)
    df = df.reset_index(drop=True)
    return df


# ====================== BƯỚC 2: COLUMN TRANSFORMER ======================
def build_preprocessor() -> ColumnTransformer:
    """Xây dựng Pipeline xử lý dữ liệu đầu vào cho Model."""
    num_zero_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
    ])

    cat_pipe = Pipeline(steps=[
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_zero", num_zero_pipe, COLS_FILL_ZERO),
            ("cat", cat_pipe, COLS_CATEGORICAL),
            ("passthrough", "passthrough", COLS_PASSTHROUGH),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


# ====================== TEST RUN ======================
if __name__ == "__main__":
    from data_loader import load_data

    print(">>> Loading data...")
    raw = load_data()

    print(">>> Engineering static features (lag/rolling chỉ nhìn về quá khứ)...")
    df = engineer_features(raw)  # KHÔNG truyền cluster_engineer

    # Chia theo thời gian TRƯỚC khi fit cluster engineer — tránh leakage
    cutoff = df["date"].max() - pd.Timedelta(days=28)
    train_df = df[df["date"] <= cutoff].copy()
    val_df = df[df["date"] > cutoff].copy()
    print(f">>> Train: {train_df['date'].min().date()} → {train_df['date'].max().date()} ({len(train_df):,} rows)")
    print(f">>> Val  : {val_df['date'].min().date()} → {val_df['date'].max().date()} ({len(val_df):,} rows)")

    print(">>> Fit cluster features CHỈ trên train...")
    cluster_eng = ClusterFeatureEngineer(smoothing=10.0)
    train_df = cluster_eng.fit_transform(train_df, target_col="target")
    val_df = cluster_eng.transform(val_df)

    # Kiểm tra hợp đồng cột với build_preprocessor (chặn KeyError mù mờ)
    expected = COLS_FILL_ZERO + COLS_CATEGORICAL + COLS_PASSTHROUGH
    missing = [c for c in expected if c not in train_df.columns]
    if missing:
        raise ValueError(f"Thiếu cột sau khi engineering: {missing}")

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)

    y_train = train_df["target"]
    y_val = val_df["target"]

    print(f"\nKích thước X_train: {X_train.shape}, X_val: {X_val.shape}")
    print("Các cột sau khi transform:")
    print(X_train.columns.tolist())
    print(f"\nNaN sau transform — train: {int(X_train.isna().sum().sum())}, val: {int(X_val.isna().sum().sum())}")
    print("\n5 dòng đầu tiên:")
    print(X_train.head())
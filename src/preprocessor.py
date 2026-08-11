"""
preprocessor.py
Feature engineering (pandas, cần groupby/shift/rolling nên KHÔNG thể làm bằng
ColumnTransformer) + encoding/imputation (sklearn Pipeline, fit trên train,
transform trên train/val/test — tránh leakage thống kê giữa các tập).
"""

import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

set_config(transform_output="pandas")

# Thêm store_nbr vào biến phân loại vì dữ liệu đã được aggregate xuống mức family
BOOL_COLS = ["onpromotion", "is_earthquake_period"]
COLS_FILL_ZERO = ["transactions_lag1", "sales_lag7", "sales_lag14", "sales_rolling_mean7"]
COLS_CATEGORICAL = ["store_nbr", "family", "city", "state", "type", "holiday_type"]
COLS_PASSTHROUGH = ["dayofweek", "month", "is_weekend", "oil_price", "cluster"] + BOOL_COLS

# ---------- Bước 1: tạo feature bằng pandas ----------

def add_date_features(df):
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df

def add_lag_features(df, lags=(7, 14)):
    df = df.sort_values(["store_nbr", "family", "date"])
    # Lag cho target (sales)
    for lag in lags:
        df[f"sales_lag{lag}"] = df.groupby(["store_nbr", "family"])["target"].shift(lag)

    #Thêm lag cho transactions (như đã khai báo trong COLS_FILL_ZERO)
    #df["transactions_lag1"] = df.groupby(["store_nbr", "family"])["transactions"].shift(1)
    return df

def add_rolling_features(df, window=7):
    df = df.sort_values(["store_nbr", "family", "date"])
    # Viết gọn và tối ưu hơn bằng transform
    df[f"sales_rolling_mean{window}"] = (
        df.groupby(["store_nbr", "family"])["target"]
        .transform(lambda s: s.shift(1).rolling(window).mean())
    )
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Chỉ TẠO cột mới — chưa encode/impute gì cả. Dùng chung cho train và predict."""
    df = df.copy()
    df["onpromotion"] = df["onpromotion"].fillna(False).astype(int)
    df["is_earthquake_period"] = df["is_earthquake_period"].fillna(False).astype(int)

    df = add_date_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)

    # --- SỬA LỖI 5: Reset index để đảm bảo tính đồng nhất ---
    df = df.reset_index(drop=True)

    return df

# ---------- Bước 2: ColumnTransformer — encode/impute ----------

def build_preprocessor() -> ColumnTransformer:
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
        remainder="drop", # Bỏ cột 'date', 'target', 'item_nbr'... không dùng cho model
        verbose_feature_names_out=False,
    )
    return preprocessor


if __name__ == "__main__":
    from src.data_loader import load_data

    train_raw = load_data()
    train_raw = engineer_features(train_raw)

    # Walk-forward split — giữ 28 ngày cuối làm validation
    cutoff = train_raw["date"].max() - pd.Timedelta(days=28)
    train_df = train_raw[train_raw["date"] <= cutoff]
    val_df = train_raw[train_raw["date"] > cutoff]

    preprocessor = build_preprocessor()

    # fit CHỈ trên train — val chỉ transform
    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)

    y_train = train_df["target"]
    y_val = val_df["target"]

    print(f"Kích thước X_train: {X_train.shape}, X_val: {X_val.shape}")
    print("\nCác cột sau khi transform:")
    print(X_train.columns.tolist())
    print("\n5 dòng đầu tiên của X_train:")
    print(X_train.head())
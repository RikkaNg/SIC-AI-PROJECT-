"""
preprocessor.py
Feature engineering (pandas, cần groupby/shift/rolling) + encoding/imputation (sklearn Pipeline).
Tích hợp cluster features: Target Encoding + Interactions.
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

# ====================== CẤU HÌNH ĐƯỜNG DẪN ======================
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Import ClusterFeatureEngineer (Vì đã có file sẵn)
from cluster_features import ClusterFeatureEngineer

set_config(transform_output="pandas")

# ====================== CẤU HÌNH CỘT ======================
BOOL_COLS = ["onpromotion", "is_earthquake_period", "is_holiday","is_back_to_school"]

# Các cột dạng số cần điền 0 nếu bị thiếu (bao gồm cả Cluster Features)
COLS_FILL_ZERO = [
    "transactions_lag1", "sales_lag7", "sales_lag14", "sales_rolling_mean7",
    "cluster_mean_sales", "cluster_median_sales", "cluster_std_sales",
    "cluster_family_mean_sales", "cluster_promo_mean_sales", "cluster_promo_lift",
]

COLS_CATEGORICAL = ["store_nbr", "family", "city", "state", "type", "holiday_type"]

COLS_PASSTHROUGH = [
    "dayofweek", "month", "is_weekend",
    "oil_price", "cluster", "perishable",
    "is_holiday_lag1", "is_holiday_lag2",
    "is_holiday_lead1", "is_holiday_lead2",
    "is_tier1_cluster",
] + BOOL_COLS


# ====================== BƯỚC 1: FEATURE ENGINEERING ======================
def add_date_features(df):
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df

def add_holiday_effects(df):
    """Tạo feature cho hiệu ứng trước/sau ngày lễ."""
    df = df.sort_values(["store_nbr", "family", "date"]).copy()
    df['is_holiday'] = (df['holiday_type'] != 'Normal Day').astype(int)

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

def add_back_to_school_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo biến flag mùa tựu trường (Back-to-school) theo đặc thù 2 vùng của Ecuador:
    - Vùng Duyên hải (Costa): Cao điểm tháng 4 và 5.
    - Vùng Núi / Nội địa (Sierra / Oriente): Cao điểm tháng 8 và 9.
    """
    df = df.copy()
    
    # Danh sách các tỉnh thuộc vùng Duyên hải (Costa) trong tập Favorita
    COSTA_STATES = {
        "Guayas", "Manabi", "Los Rios", "El Oro", 
        "Santa Elena", "Esmeraldas", "Santo Domingo de los Tsachilas", "Santo Domingo"
    }

    # Đảm bảo đã có cột month
    month = df["date"].dt.month if "month" not in df.columns else df["month"]

    if "state" in df.columns:
        is_costa = df["state"].isin(COSTA_STATES)
    else:
        # Fallback nếu không có cột state: mặc định bắt cả 2 đợt
        is_costa = False

    # Logic kích hoạt cờ tựu trường
    costa_season = is_costa & month.isin([4, 5])
    sierra_season = (~is_costa) & month.isin([8, 9])

    df["is_back_to_school"] = (costa_season | sierra_season).astype(int)
    return df

def add_rolling_features(df, window=7):
    df = df.sort_values(["store_nbr", "family", "date"])
    df[f"sales_rolling_mean{window}"] = (
        df.groupby(["store_nbr", "family"])["target"]
        .transform(lambda s: s.shift(1).rolling(window).mean())
    )
    return df

def add_cluster_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo interaction features đơn giản từ cluster."""
    df = df.copy()
    if "cluster_promo_mean_sales" in df.columns and "cluster_mean_sales" in df.columns:
        df["cluster_promo_lift"] = df["cluster_promo_mean_sales"] / (df["cluster_mean_sales"] + 1e-8)
    if "cluster" in df.columns:
        df["is_tier1_cluster"] = df["cluster"].isin({5, 11}).astype(int)
    return df



def engineer_features(df: pd.DataFrame, cluster_engineer: ClusterFeatureEngineer = None, fit_cluster: bool = False) -> pd.DataFrame:
    """
    Tạo cột mới. 
    Args:
        cluster_engineer: Instance của ClusterFeatureEngineer. Nếu None, sẽ tự động tạo mới.
        fit_cluster: True nếu đang ở tập train, False nếu val/test.
    """
    df = df.copy()
    df["onpromotion"] = df["onpromotion"].fillna(False).astype(int)
    df["is_earthquake_period"] = df["is_earthquake_period"].fillna(False).astype(int)

    df = add_date_features(df)
    df = add_lag_features(df)
    df = add_back_to_school_feature(df)
    df = add_rolling_features(df)
    df = add_holiday_effects(df)

    # Nếu không truyền cluster_engineer vào, tự khởi tạo 1 cái mới (mặc định)
    if cluster_engineer is None:
        cluster_engineer = ClusterFeatureEngineer(smoothing=10.0)

    # Fit trên train, transform trên val/test
    if fit_cluster:
        df = cluster_engineer.fit_transform(df, target_col="target")
    else:
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
    train_raw = load_data()
    
    print(">>> Engineering features...")
    # Khởi tạo Cluster Engineer
    cluster_eng = ClusterFeatureEngineer(smoothing=10.0)
    train_raw = engineer_features(train_raw, cluster_engineer=cluster_eng, fit_cluster=True)

    # Walk-forward split
    cutoff = train_raw["date"].max() - pd.Timedelta(days=28)
    train_df = train_raw[train_raw["date"] <= cutoff]
    val_df = train_raw[train_raw["date"] > cutoff]

    preprocessor = build_preprocessor()

    X_train = preprocessor.fit_transform(train_df)
    X_val = preprocessor.transform(val_df)

    y_train = train_df["target"]
    y_val = val_df["target"]

    print(f"\nKích thước X_train: {X_train.shape}, X_val: {X_val.shape}")
    print("Các cột sau khi transform:")
    print(X_train.columns.tolist())
    print("\n5 dòng đầu tiên:")
    print(X_train.head())
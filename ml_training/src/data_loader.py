"""
data_loader.py
Đọc và chuẩn bị dữ liệu Favorita — dùng chung cho train.py và predict.py.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TRAIN_DTYPES = {
    "store_nbr": "int8",
    "item_nbr": "int32",
    "unit_sales": "float32",
    "onpromotion": "boolean",
}


def load_raw_files():
    train = pd.read_csv(RAW_DIR / "train.csv", parse_dates=["date"], dtype=TRAIN_DTYPES)
    stores = pd.read_csv(RAW_DIR / "stores.csv")
    items = pd.read_csv(RAW_DIR / "items.csv")
    oil = pd.read_csv(RAW_DIR / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(RAW_DIR / "holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv(RAW_DIR / "transactions.csv", parse_dates=["date"])
    return train, stores, items, oil, holidays, transactions

def filter_date_range(df, start_date="2016-01-01"):
    return df[df["date"] >= start_date].copy()

def fill_onpromotion(df):
    df["onpromotion"] = df["onpromotion"].fillna(False)
    return df

def remove_data_errors(df):
    """So sánh theo phân phối của từng item_nbr — chạy TRƯỚC khi merge/aggregate."""
    stats = df.groupby("item_nbr")["unit_sales"].agg(["max"]).rename(columns={"max": "item_max"})
    df = df.merge(stats, on="item_nbr", how="left")

    is_extreme_negative = (df["unit_sales"] < 0) & (df["unit_sales"].abs() > df["item_max"])
    print(f"Loại {is_extreme_negative.sum()} dòng lỗi nhập liệu cực đoan")

    df = df[~is_extreme_negative].drop(columns=["item_max"])
    return df

def prepare_oil(oil, date_min, date_max):
    all_dates = pd.DataFrame(pd.date_range(date_min, date_max), columns=["date"])
    oil = all_dates.merge(oil, on="date", how="left")
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    oil = oil.rename(columns={"dcoilwtico": "oil_price"})
    return oil

def prepare_holidays(holidays):
    # Chỉ bỏ các ngày bị transferred, giữ lại tất cả các cấp độ (National, Regional, Local)
    holidays = holidays[holidays["transferred"] == False].copy()
    holidays = holidays.rename(columns={
        "type": "holiday_type",
        "description": "holiday_description"
    })
    # Giữ lại cột locale và locale_name để phục vụ việc merge chi tiết
    return holidays[["date", "locale", "locale_name", "holiday_type", "holiday_description"]]

def prepare_transactions_lag(transactions, lag_days=1):
    """
    Tính lag TRƯỚC khi merge vào bảng chính — transactions vốn là chuỗi theo
    (store_nbr, date), không phụ thuộc item/family, nên tính lag ở đây rồi
    merge vào bảng đã gộp family sẽ đúng và nhẹ hơn nhiều so với tính lag
    sau khi đã nhân bản ra hàng nghìn item.
    """
    transactions = transactions.sort_values(["store_nbr", "date"]).copy()
    col_name = f"transactions_lag{lag_days}"
    transactions[col_name] = transactions.groupby("store_nbr")["transactions"].shift(lag_days)
    return transactions[["store_nbr", "date", col_name]]

def merge_dimensions(df, stores, items, oil, holidays):
    """Ghép các bảng dimension ở mức item — Xử lý holidays chi tiết theo National/Regional/Local."""
    df = df.merge(stores, on="store_nbr", how="left")
    df = df.merge(items, on="item_nbr", how="left")
    df = df.merge(oil, on="date", how="left")

    # --- XỬ LÝ HOLIDAYS CHI TIẾT THEO CẤP ĐỘ ---
    # Chia nhỏ holidays thành 3 bảng để xử lý riêng, tránh duplicate dòng khi merge
    h_nat = holidays[holidays["locale"] == "National"].drop_duplicates(subset=["date"])
    h_reg = holidays[holidays["locale"] == "Regional"].drop_duplicates(subset=["date", "locale_name"])
    h_loc = holidays[holidays["locale"] == "Local"].drop_duplicates(subset=["date", "locale_name"])

    before = len(df)

    # 1. Merge National theo date
    df = df.merge(
        h_nat[["date", "holiday_type", "holiday_description"]],
        on="date", how="left"
    )

    # 2. Merge Regional theo date và state (locale_name của Regional trùng với state)
    df = df.merge(
        h_reg[["date", "locale_name", "holiday_type", "holiday_description"]].rename(columns={"locale_name": "state"}),
        on=["date", "state"], how="left", suffixes=("", "_reg")
    )
    # Nếu National bị NaN, lấy giá trị của Regional đè lên
    df["holiday_type"] = df["holiday_type"].fillna(df["holiday_type_reg"])
    df["holiday_description"] = df["holiday_description"].fillna(df["holiday_description_reg"])
    df.drop(columns=["holiday_type_reg", "holiday_description_reg"], inplace=True)

    # 3. Merge Local theo date và city (locale_name của Local trùng với city)
    df = df.merge(
        h_loc[["date", "locale_name", "holiday_type", "holiday_description"]].rename(columns={"locale_name": "city"}),
        on=["date", "city"], how="left", suffixes=("", "_loc")
    )
    # Nếu vẫn NaN, lấy giá trị của Local đè lên
    df["holiday_type"] = df["holiday_type"].fillna(df["holiday_type_loc"])
    df["holiday_description"] = df["holiday_description"].fillna(df["holiday_description_loc"])
    df.drop(columns=["holiday_type_loc", "holiday_description_loc"], inplace=True)

    # Kiểm tra an toàn
    assert len(df) == before, "Số dòng thay đổi sau khi join holidays — kiểm tra trùng ngày"

    # Những ngày thực sự không có lễ gì mới là 'Normal Day'
    df["holiday_type"] = df["holiday_type"].fillna("Normal Day")
    df["holiday_description"] = df["holiday_description"].fillna("None")

    return df

def add_earthquake_flag(df):
    df["is_earthquake_period"] = df["holiday_description"].str.contains(
        "Terremoto", case=False, na=False
    )
    return df

def aggregate_to_family_level(df):
    """Gộp từ item_nbr (hàng nghìn chuỗi) lên family (33 chuỗi/cửa hàng) — đúng scope 1 tháng."""
    df_family = (
        df.groupby(["store_nbr", "family", "date"])
        .agg(
            unit_sales=("unit_sales", "sum"),
            onpromotion=("onpromotion", "any"),
            oil_price=("oil_price", "first"),
            is_earthquake_period=("is_earthquake_period", "any"),
            holiday_type=("holiday_type", "first"),
            city=("city", "first"),
            state=("state", "first"),
            type=("type", "first"),
            cluster=("cluster", "first"),
            perishable=("perishable", "max"),
        )
        .reset_index()
    )
    return df_family

def build_target(df):
    df["target"] = df["unit_sales"].clip(lower=0)
    return df

def load_data(force_rebuild: bool = False, transactions_lag_days: int = 1) -> pd.DataFrame:
    """
    Entry point duy nhất — gọi hàm này ở notebook/train.py/predict.py.
    Tự cache ra parquet, chỉ build lại khi force_rebuild=True hoặc chưa có cache.
    """
    cache_path = PROCESSED_DIR / "train_merged.parquet"

    if cache_path.exists() and not force_rebuild:
        return pd.read_parquet(cache_path)

    train, stores, items, oil, holidays, transactions = load_raw_files()
    train = filter_date_range(train)
    train = fill_onpromotion(train)
    train = remove_data_errors(train)

    oil = prepare_oil(oil, train["date"].min(), train["date"].max())
    holidays = prepare_holidays(holidays)
    train = merge_dimensions(train, stores, items, oil, holidays)
    train = add_earthquake_flag(train)
    train = aggregate_to_family_level(train)

    transactions_lag = prepare_transactions_lag(transactions, lag_days=transactions_lag_days)
    train = train.merge(transactions_lag, on=["store_nbr", "date"], how="left")

    train = build_target(train)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(cache_path, index=False)
    return train

if __name__ == "__main__":
    df = load_data(force_rebuild=True)
    print(df.shape)
    print(df.info(memory_usage="deep"))
    print(df.head())
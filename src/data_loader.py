"""
read_data.py
Đọc và chuẩn bị dữ liệu Favorita — dùng chung cho cả train.py và predict.py
để tránh train-serving skew (logic đọc/xử lý chỉ viết 1 lần duy nhất).
"""

import pandas as pd
from pathlib import Path

BASE_DIR =Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TRAIN_DTYPES = {
    "store_nbr": "int8",
    "item_nbr": "int32",
    "unit_sales": "float32",
    "onpromotion": "str",  # Đọc thô dạng str trước, xử lý phía dưới
}


def load_raw_files():
    """Đọc toàn bộ file gốc, chỉ lấy cột cần thiết (usecols) để tiết kiệm RAM."""
    train = pd.read_csv(
        RAW_DIR / "train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        dtype=TRAIN_DTYPES,
        # Bỏ parse_dates ở đây
    )
    stores = pd.read_csv(RAW_DIR / "stores.csv")
    oil = pd.read_csv(RAW_DIR / "oil.csv")
    holidays = pd.read_csv(RAW_DIR / "holidays_events.csv")
    items = pd.read_csv(RAW_DIR / "items.csv")  

    # Chuyển đổi cột 'date' sang datetime sau khi đã đọc xong
    # Dùng errors='coerce' để những dòng bị lỗi format sẽ thành NaT (Null) thay vì văng lỗi
    train["date"] = pd.to_datetime(train["date"], format="%Y-%m-%d", errors="coerce")
    oil["date"] = pd.to_datetime(oil["date"], format="%Y-%m-%d", errors="coerce")
    holidays["date"] = pd.to_datetime(holidays["date"], format="%Y-%m-%d", errors="coerce")
    
    return train, stores, oil, holidays, items

def filter_date_range(df, start_date="2016-01-01"):
    """Giới hạn khoảng thời gian — giữ ~1.5-2 năm gần nhất."""
    return df[df["date"] >= start_date].copy()


def remove_data_errors(df):
    """
    Loại các lỗi nhập liệu rõ ràng (đã xác nhận qua EDA — không liên quan sự kiện
    thật nào trong holidays_events, độ lớn vượt xa item_max của chính item đó).
    KHÔNG dùng ngưỡng cứng cố định — so sánh theo phân phối của từng item_nbr.
    """
    stats = df.groupby("item_nbr")["unit_sales"].agg(["max"]).rename(
        columns={"max": "item_max"}
    )
    df = df.merge(stats, on="item_nbr", how="left")

    is_extreme_negative = (df["unit_sales"] < 0) & (
        df["unit_sales"].abs() > df["item_max"]
    )
    print(f"Loại {is_extreme_negative.sum()} dòng lỗi nhập liệu cực đoan")

    df = df[~is_extreme_negative].drop(columns=["item_max"])
    return df


def merge_dimensions(df, stores, oil, holidays, items):
    """Join các bảng dimension vào bảng giao dịch chính. Luôn dùng how='left'."""
    df = df.merge(items[["item_nbr", "family"]], on="item_nbr", how="left")
    df = df.merge(stores, on="store_nbr", how="left")
    
    # --- SỬA 1: Điền giá trị dầu (ffill) cho các ngày cuối tuần bị thiếu ---
    oil = oil.sort_values("date")
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
    df = df.merge(oil, on="date", how="left")

    # --- SỬA 2: Xử lý holidays để tránh lỗi duplicate dòng ---
    holidays_clean = holidays[holidays["transferred"] == False].copy()
    # Ưu tiên lấy ngày lễ Quốc gia (National), nếu không có thì lấy Local/Regional
    # Sau đó drop_duplicates để đảm bảo 1 ngày chỉ có 1 record lớn nhất
    holidays_clean = holidays_clean.sort_values(['date', 'locale'], ascending=[True, False])
    holidays_clean = holidays_clean.drop_duplicates(subset=['date'], keep='first')
    
    holidays_clean = holidays_clean.rename(
        columns={"type": "holiday_type", "description": "holiday_description"}
    )
    
    before = len(df)
    df = df.merge(
        holidays_clean[["date", "holiday_type", "holiday_description"]],
        on="date",
        how="left",
    )
    assert len(df) == before, "Số dòng thay đổi sau khi join holidays — kiểm tra trùng ngày"

    return df


def add_earthquake_flag(df):
    """
    Đánh dấu giai đoạn động đất Manabi (16/4/2016) — không lặp lại ở test set,
    nhưng giúp model không học nhầm sụt giảm doanh số 2016 thành seasonality chung.
    """
    df["is_earthquake_period"] = df["holiday_description"].str.contains(
        "Terremoto", case=False, na=False
    )
    return df


def aggregate_to_family_level(df):
    """Gộp từ item_nbr (hàng nghìn chuỗi) lên family (33 chuỗi/cửa hàng)."""
    
    # --- SỬA 3: Ép onpromotion về boolean an toàn trước khi agg ---
    df['onpromotion'] = df['onpromotion'].fillna('False').astype(str)
    df['onpromotion'] = df['onpromotion'].str.strip().str.lower().map(
        {'true': True, 'false': False, '1': True, '0': False}
    ).fillna(False).astype(bool)

    df_family = (
        df.groupby(["store_nbr", "family", "date"])
        .agg(
            unit_sales=("unit_sales", "sum"),
            onpromotion=("onpromotion", "any"),  # Nếu 1 item trong family có khuyến mãi -> True
            oil_price=("dcoilwtico", "first"),
            is_earthquake_period=("is_earthquake_period", "any"),
            holiday_type=("holiday_type", "first"),
        )
        .reset_index()
    )
    return df_family


def build_target(df):
    """RMSLE không nhận số âm — clip target về 0, KHÔNG xoá dòng gốc."""
    df["target"] = df["unit_sales"].clip(lower=0)
    return df


def load_data(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Entry point duy nhất — gọi hàm này ở cả notebook/train.py/predict.py.
    Tự cache ra parquet, chỉ build lại khi force_rebuild=True hoặc chưa có cache.
    """
    cache_path = PROCESSED_DIR / "train_merged.parquet"

    if cache_path.exists() and not force_rebuild:
        return pd.read_parquet(cache_path)

    print("Bắt đầu đọc dữ liệu thô...")
    train, stores, oil, holidays, items = load_raw_files()
    
    print("Lọc thời gian...")
    train = filter_date_range(train)
    
    print("Xử lý lỗi dữ liệu...")
    train = remove_data_errors(train)
    
    print("Gộp dữ liệu...")
    train = merge_dimensions(train, stores, oil, holidays, items)
    
    print("Tạo cột đặc trưng...")
    train = add_earthquake_flag(train)
    
    print("Gộp xuống cấp độ Family...")
    train = aggregate_to_family_level(train)
    
    print("Tạo biến mục tiêu...")
    train = build_target(train)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(cache_path, index=False)
    print(f"Đã lưu cache tại: {cache_path}")
    return train


if __name__ == "__main__":
    df = load_data(force_rebuild=True)
    print(df.shape)
    print(df.head())
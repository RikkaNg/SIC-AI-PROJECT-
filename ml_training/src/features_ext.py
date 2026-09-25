"""
src/features_ext.py — MODULE DÙNG CHUNG (train + serving)

Lý do tồn tại: 3 nơi (train.py, predict.py, ml_service/app/inference.py) phải tạo
feature GIỐNG HỆT nhau. Thay vì copy-paste logic, tất cả gọi 2 hàm trong file này:

    df = complete_panel(df)                 # chỉ ở tầng load/backfill dữ liệu lịch sử
    df = add_extra_features(df, min_lag=1)  # sau engineer_features()

RÀNG BUỘC SERVING (đã kiểm tra với cửa sổ đệ quy (d-60, d+2]):
- Lag tối đa 28 ngày, rolling tối đa 28 ngày, days_since_sale chặn ở 56
  → mọi feature đều tính được trong cửa sổ 60 ngày, KHÔNG cần đổi window.
- Không dùng thông tin tương lai quá +2 ngày (giữ nguyên hợp đồng lead buffer).
- Mọi thống kê đều shift(min_lag) → không rò rỉ giá trị của chính ngày dự báo.

min_lag = "khoảng cách an toàn tới ngày dự báo":
    min_lag=1  → model dùng cho re-forecast HẰNG NGÀY (lag1 là số thật) hoặc cho
                 vòng đệ quy (lag1 là số model tự dự báo).
    min_lag=7/14/28 → model DIRECT cho horizon 8-14 / 15-21 / 22-28 ngày:
                 không cần đệ quy, không tích lũy sai số.

FIX LOG v2.1:
[W1] FutureWarning pandas ≥2.1 ("Downcasting object dtype arrays on .fillna",
     dòng observed & is_earthquake_period) → opt-in hành vi tương lai bằng
     set_option, bọc try/except cho pandas cũ. Giá trị KHÔNG đổi.
[W2] is_day_after_payday: train.py v2 khai báo trong PASSTHROUGH_FEATURES nhưng
     KHÔNG module nào tạo → bị _present() lọc mất im lặng (đúng lớp bug "drop
     âm thầm" mà v2 ra đời để diệt). Giờ được tạo thật tại đây; 3 nơi dùng chung
     hàm này nên train/serve contract tự khớp. Bỏ nếu không muốn: xóa 2 dòng
     đánh dấu [W2] và 1 tên trong extra_numeric_features.
[W3] complete_panel: guard trùng (store_nbr, family, date) — fail loud thay vì
     merge nổ dòng âm thầm khi dữ liệu vào chưa aggregate.
[W4] Chuẩn hóa indentation block cross-series (2-space lẫn 6-space → 4-space).
[W5] FEATURE_SPEC_VERSION: v2.0 → v2.1 (spec đổi → bump để meta/serve đối chiếu).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# [W1] pandas ≥ 2.1 cảnh báo khi fillna(False) trên cột object (downcast ngầm).
# Opt-in hành vi tương lai: downcast thành tường minh, giá trị giữ nguyên.
# pandas cũ không có option này → try/except (bản cũ vốn không cảnh báo).
try:
    pd.set_option("future.no_silent_downcasting", True)
except (KeyError, TypeError, ValueError):
    pass

set_config(transform_output="pandas")

KEYS = ["store_nbr", "family"]

LAGS = (1, 2, 3, 7, 14, 21, 28)
ROLL_WINDOWS = (7, 14, 28)
MAX_LOOKBACK = 56          # trần của days_since_sale (phải < 60 = window serving)

# Feature CẮT NGANG nhiều chuỗi: cần lịch sử của CẢ cửa hàng (nhiều family) trong
# cùng một dataframe. ml_service/inference.py::predict_recursive chỉ nạp lịch sử
# của ĐÚNG MỘT cặp (store, family) → tính ra sẽ khác lúc train (skew).
# Chỉ bật khi backend gửi đủ lịch sử toàn cửa hàng.
CROSS_SERIES_FEATURES = ["store_promo_share", "store_slog_mean_lag", "family_slog_mean_lag"]
FEATURE_SPEC_VERSION = "v2.1"   # [W5]

# Cột tĩnh theo (store_nbr, family) — dùng để vá khi bơm dòng 0 vào panel
STATIC_STORE_COLS = ["city", "state", "type", "cluster"]
STATIC_SF_COLS = ["perishable"]


# ============================================================================
# 1. HOÀN THIỆN PANEL (quan trọng nhất)
# ============================================================================
def complete_panel(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Favorita train.csv KHÔNG lưu dòng có doanh số 0. Sau khi gộp lên family,
    mỗi (store, family) là một chuỗi THỦNG LỖ CHỖ. Hậu quả:

      1) groupby().shift(7) dịch 7 DÒNG, không phải 7 NGÀY → "sales_lag7" thực tế
         có thể là doanh số của 9, 12, 20 ngày trước. Toàn bộ lag/rolling bị lệch.
      2) Model không bao giờ nhìn thấy ngày bán = 0 → luôn dự báo dương cho các
         family hiếm bán. RMSLE phạt rất nặng lỗi này (log1p(0)=0).

    Hàm này bơm lại các ngày thiếu với target=0, tính từ ngày ĐẦU TIÊN mà chuỗi
    đó xuất hiện (không bịa ra 0 cho giai đoạn cửa hàng/ngành hàng chưa tồn tại).

    LƯU Ý ĐO LƯỜNG: sau khi bật hàm này, RMSLE không so sánh trực tiếp được với
    số cũ vì tập đánh giá đã thay đổi (có thêm các ngày 0). Số đáng tin để so
    sánh là điểm Kaggle / NWRMSLE, vì tập test của Kaggle vốn có đủ các dòng 0.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # [W3] Panel phải unique theo (store, family, date). Nếu vào item-level
    # (chưa aggregate) thì merge bên dưới sẽ NHÂN dòng âm thầm — phải báo chết.
    dup = int(df.duplicated(subset=KEYS + ["date"]).sum())
    if dup:
        raise ValueError(
            f"complete_panel: dữ liệu vào có {dup:,} dòng trùng (store_nbr, family, date). "
            "Hàm này chỉ dành cho panel family-level — hãy aggregate_to_family_level() trước."
        )

    n_before = len(df)

    max_date = df["date"].max()
    starts = df.groupby(KEYS, observed=True)["date"].min().reset_index(name="_start")

    all_dates = pd.DataFrame({"date": pd.date_range(df["date"].min(), max_date, freq="D")})
    grid = starts.merge(all_dates, how="cross")
    grid = grid[grid["date"] >= grid["_start"]].drop(columns="_start")

    df["observed"] = True
    out = grid.merge(df, on=KEYS + ["date"], how="left")
    # [W1] với option đã bật: object → bool tường minh, không còn FutureWarning
    out["observed"] = out["observed"].fillna(False).astype(bool)

    # --- cột tĩnh theo store / store-family ---
    for col in STATIC_STORE_COLS:
        if col in out.columns:
            lut = df.dropna(subset=[col]).groupby("store_nbr", observed=True)[col].first()
            out[col] = out[col].fillna(out["store_nbr"].map(lut))
    for col in STATIC_SF_COLS:
        if col in out.columns:
            lut = df.dropna(subset=[col]).groupby(KEYS, observed=True)[col].first()
            idx = pd.MultiIndex.from_arrays([out["store_nbr"], out["family"]])
            out[col] = out[col].fillna(pd.Series(idx.map(lut), index=out.index))

    # --- cột theo ngày (oil, động đất) ---
    if "oil_price" in out.columns:
        lut = df.groupby("date")["oil_price"].first()
        out["oil_price"] = out["oil_price"].fillna(out["date"].map(lut))
        out = out.sort_values("date")
        out["oil_price"] = out["oil_price"].ffill().bfill()
    if "is_earthquake_period" in out.columns:
        lut = df.groupby("date")["is_earthquake_period"].max()
        out["is_earthquake_period"] = (
            out["is_earthquake_period"].fillna(out["date"].map(lut))
            .fillna(False).astype(bool)          # [W1] + chốt dtype bool
        )

    # --- holiday_type theo (store, date), fallback theo date ---
    if "holiday_type" in out.columns:
        sd = df.dropna(subset=["holiday_type"]).groupby(["store_nbr", "date"], observed=True)["holiday_type"].first()
        idx = pd.MultiIndex.from_arrays([out["store_nbr"], out["date"]])
        out["holiday_type"] = out["holiday_type"].fillna(pd.Series(idx.map(sd), index=out.index))
        nat = (
            df[df["holiday_type"] != "Normal Day"].groupby("date")["holiday_type"].first()
            if (df["holiday_type"] != "Normal Day").any() else pd.Series(dtype=object)
        )
        out["holiday_type"] = out["holiday_type"].fillna(out["date"].map(nat)).fillna("Normal Day")

    # --- cột theo (store, date): transactions ---
    for col in ("transactions", "transactions_lag1"):
        if col in out.columns:
            lut = df.dropna(subset=[col]).groupby(["store_nbr", "date"], observed=True)[col].first()
            idx = pd.MultiIndex.from_arrays([out["store_nbr"], out["date"]])
            out[col] = out[col].fillna(pd.Series(idx.map(lut), index=out.index))

    # --- target & promo cho ngày vừa bơm ---
    out["target"] = out["target"].fillna(0.0).clip(lower=0).astype("float64")
    if "unit_sales" in out.columns:
        out["unit_sales"] = out["unit_sales"].fillna(0.0)
    if "onpromotion" in out.columns:
        out["onpromotion"] = out["onpromotion"].fillna(False)

    out = out.sort_values(KEYS + ["date"], ignore_index=True)
    if verbose:
        added = len(out) - n_before
        print(f">>> complete_panel: {n_before:,} → {len(out):,} dòng "
              f"(+{added:,} ngày doanh số 0 được bơm lại, {added / max(len(out), 1):.1%})")
    return out


# ============================================================================
# 2. FEATURE MỚI
# ============================================================================
def extra_numeric_features(min_lag: int = 1, has_transactions: bool = True,
                           include_cross_series: bool = False) -> list[str]:
    """Danh sách tên feature do add_extra_features() tạo ra (đúng thứ tự)."""
    L = int(min_lag)
    lags = sorted({max(l, L) for l in LAGS})
    names = [f"slog_lag{l}" for l in lags]
    names += [f"slog_roll_mean{w}" for w in ROLL_WINDOWS]
    names += ["slog_roll_std7", "slog_roll_std28", "slog_dow_mean4",
              "slog_trend_7_28", "slog_ratio_7_28",
              "zero_rate_7", "zero_rate_28", "days_since_sale",
              "promo_rate_7", "promo_rate_28",
              "dayofmonth", "weekofmonth", "dayofyear",
              "is_payday", "is_day_after_payday",                      # [W2]
              "days_from_payday", "days_to_payday",
              "is_month_start", "is_month_end"]
    if has_transactions:
        names.append("trans_roll_mean7")
    if include_cross_series:
        names += CROSS_SERIES_FEATURES
    return names


def _grouped_roll(df: pd.DataFrame, col: str, window: int, stat: str, min_periods: int) -> pd.Series:
    r = df.groupby(KEYS, observed=True)[col].rolling(window, min_periods=min_periods)
    s = getattr(r, stat)()
    return s.reset_index(level=list(range(len(KEYS))), drop=True)


def add_extra_features(df: pd.DataFrame, min_lag: int = 1,
                       include_cross_series: bool = False) -> pd.DataFrame:
    """
    Thêm feature nâng cao. PHẢI gọi sau engineer_features() và trên dữ liệu đã
    complete_panel() (nếu không, mọi lag vẫn bị lệch ngày).

    Ở serving, gọi trên đúng cửa sổ (d-60, d+2] rồi lấy dòng của ngày d.
    """
    # [W3] fail loud nếu caller truyền thiếu cột lõi — thay vì KeyError mơ hồ
    # nổ giữa chừng khi groupby/shift bên dưới
    _need = ["date", "target"] + KEYS
    _missing = [c for c in _need if c not in df.columns]
    if _missing:
        raise ValueError(f"add_extra_features: thiếu cột bắt buộc {_missing}")

    L = int(min_lag)
    df = df.sort_values(KEYS + ["date"], ignore_index=True).copy()
    df["date"] = pd.to_datetime(df["date"])

    # QUAN TRỌNG cho vòng đệ quy: cửa sổ được cắt ra từ dataframe ĐÃ có sẵn các cột
    # này. Không xoá trước thì merge bên dưới sinh ra hậu tố _x/_y và preprocessor
    # sẽ báo "columns are missing".
    created = set(extra_numeric_features(L, has_transactions=True, include_cross_series=True))
    created |= {f"slog_lag{l}" for l in LAGS}
    df = df.drop(columns=[c for c in created if c in df.columns], errors="ignore")

    df["_slog"] = np.log1p(df["target"].clip(lower=0).astype("float64"))
    g = df.groupby(KEYS, observed=True)

    # ---- lag doanh số (log-space) ----
    for l in sorted({max(l, L) for l in LAGS}):
        df[f"slog_lag{l}"] = g["_slog"].shift(l)

    # ---- rolling trên chuỗi đã shift(L) ----
    df["_s"] = g["_slog"].shift(L)
    for w in ROLL_WINDOWS:
        df[f"slog_roll_mean{w}"] = _grouped_roll(df, "_s", w, "mean", max(2, w // 3))
    df["slog_roll_std7"] = _grouped_roll(df, "_s", 7, "std", 3)
    df["slog_roll_std28"] = _grouped_roll(df, "_s", 28, "std", 7)

    df["slog_trend_7_28"] = df["slog_roll_mean7"] - df["slog_roll_mean28"]
    df["slog_ratio_7_28"] = df["slog_roll_mean7"] / (df["slog_roll_mean28"] + 1e-3)

    # ---- trung bình cùng thứ trong tuần (4 tuần gần nhất) ----
    df["_dow"] = df["date"].dt.dayofweek
    dow_keys = KEYS + ["_dow"]
    df["_sdow"] = df.groupby(dow_keys, observed=True)["_slog"].shift(int(np.ceil(L / 7)))
    df["slog_dow_mean4"] = (
        df.groupby(dow_keys, observed=True)["_sdow"]
        .rolling(4, min_periods=1).mean()
        .reset_index(level=list(range(len(dow_keys))), drop=True)
    )

    # ---- cấu trúc ngày không bán ----
    df["_zero"] = (df["target"] <= 0).astype("float64")
    df["_z"] = g["_zero"].shift(L)
    df["zero_rate_7"] = _grouped_roll(df, "_z", 7, "mean", 2)
    df["zero_rate_28"] = _grouped_roll(df, "_z", 28, "mean", 7)

    df["_last_pos"] = df["date"].where(df["target"] > 0)
    df["_last_pos"] = df.groupby(KEYS, observed=True)["_last_pos"].ffill()
    df["_last_pos"] = df.groupby(KEYS, observed=True)["_last_pos"].shift(L)
    df["days_since_sale"] = (
        (df["date"] - df["_last_pos"]).dt.days.fillna(MAX_LOOKBACK).clip(0, MAX_LOOKBACK)
    )

    # ---- khuyến mãi ----
    if "onpromotion" in df.columns:
        df["_promo"] = df["onpromotion"].fillna(0).astype("float64")
        df["_p"] = g["_promo"].shift(L)
        df["promo_rate_7"] = _grouped_roll(df, "_p", 7, "mean", 2)
        df["promo_rate_28"] = _grouped_roll(df, "_p", 28, "mean", 7)
        # tỉ lệ ngành hàng đang khuyến mãi trong CÙNG cửa hàng, CÙNG ngày:
        # đây là kế hoạch promo → biết trước, không cần lag.
        if include_cross_series:
            df["store_promo_share"] = df.groupby(["store_nbr", "date"], observed=True)["_promo"].transform("mean")
    else:
        for c in ("promo_rate_7", "promo_rate_28"):
            df[c] = np.nan
        if include_cross_series:
            df["store_promo_share"] = np.nan

    # ---- tổng hợp cấp cửa hàng / cấp ngành hàng (bắt xu hướng chung) ----
    # [W4] indentation chuẩn hóa 4-space (bản cũ trộn 2/6-space)
    if include_cross_series:
        store_day = (
            df.groupby(["store_nbr", "date"], observed=True)["_slog"].mean()
            .reset_index(name="_store_mean").sort_values(["store_nbr", "date"])
        )
        store_day["store_slog_mean_lag"] = store_day.groupby("store_nbr", observed=True)["_store_mean"].shift(L)
        df = df.merge(store_day[["store_nbr", "date", "store_slog_mean_lag"]],
                      on=["store_nbr", "date"], how="left")

        fam_day = (
            df.groupby(["family", "date"], observed=True)["_slog"].mean()
            .reset_index(name="_fam_mean").sort_values(["family", "date"])
        )
        fam_day["family_slog_mean_lag"] = fam_day.groupby("family", observed=True)["_fam_mean"].shift(L)
        df = df.merge(fam_day[["family", "date", "family_slog_mean_lag"]],
                      on=["family", "date"], how="left")

    # ---- transactions cấp cửa hàng ----
    if "transactions" in df.columns:
        tr = (
            df.groupby(["store_nbr", "date"], observed=True)["transactions"].first()
            .reset_index().sort_values(["store_nbr", "date"])
        )
        tr["_t"] = tr.groupby("store_nbr", observed=True)["transactions"].shift(L)
        tr["trans_roll_mean7"] = (
            tr.groupby("store_nbr", observed=True)["_t"].rolling(7, min_periods=2).mean()
            .reset_index(level=0, drop=True)
        )
        df = df.merge(tr[["store_nbr", "date", "trans_roll_mean7"]],
                      on=["store_nbr", "date"], how="left")

    # ---- lịch / ngày lương (Ecuador trả lương ngày 15 và ngày cuối tháng) ----
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

    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")

    # float32 cho ~30 cột mới: giảm một nửa RAM, cây không mất độ chính xác đáng kể
    for c in extra_numeric_features(L, has_transactions=True, include_cross_series=True):
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype("float32")

    return df.sort_values(KEYS + ["date"], ignore_index=True)


# ============================================================================
# 3. PREPROCESSOR V2 (không còn hard-code, không âm thầm bỏ feature)
# ============================================================================
def build_preprocessor_v2(num_zero_cols, num_sentinel_cols, cat_cols, passthrough_cols):

    zero_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=0))])
    sent_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value=-1))])
    cat_pipe = Pipeline([("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])

    transformers = []
    if num_zero_cols:
        transformers.append(("num_zero", zero_pipe, list(num_zero_cols)))
    if num_sentinel_cols:
        transformers.append(("num_sent", sent_pipe, list(num_sentinel_cols)))
    if cat_cols:
        transformers.append(("cat", cat_pipe, list(cat_cols)))
    if passthrough_cols:
        transformers.append(("passthrough", "passthrough", list(passthrough_cols)))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
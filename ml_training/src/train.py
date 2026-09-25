"""
src/train.py — v2.4 (PRODUCT-ORIENTED, tối ưu điểm số)

ĐIỂM KHÁC BẢN CŨ (theo thứ tự tác động đến RMSLE):

[1] complete_panel() — bơm lại ngày 0; lag dịch theo NGÀY chứ không phải dòng.
[2] Preprocessor động — không còn drop feature âm thầm.
[3] ~30 feature mới (features_ext.py) + Cluster TE.
[4] Blend weight tối ưu trên OOF gộp 3 fold.
[5] Bộ metric đầy đủ + NWRMSLE (perishable ×1.25) bám sát Kaggle.
[6] Tùy chọn: seed averaging, recency weight, thành viên Tweedie.

FIX LOG:
v2.1: [F1] backtest crash transactions_lag1. [F2] history chưa cắt warm-up.
      [F3] parity báo cột thiếu 1 phía. [F4] window 60→90. [F5] drop stale cols.
v2.2: [A] Cluster TE + is_tier1 vào model (khai báo tĩnh + assert).
      [B] ARTIFACT_SUFFIX; warmup tự nâng. [C] per-family degradation.
v2.3: [D] LEGACY_LAG_DAYS — lọc cột legacy lag-thật < MIN_LAG (model DIRECT
      thấy giá trị thật của ngày trong val = optimistic giả). [D-guard]
      audit leak tự động (mask target val → feature phải bất biến).
v2.4: [E1] Blend RMSLE honest bằng LOO weight (w chấm fold i được fit CHỈ trên
      OOF các fold khác) + lượng hóa % lạc quan của số fold-tuned.
      [E2] ES_MODE="retrain": inner-ES tìm iterations → retrain FULL train với
      iterations đó → eval val (mirror đúng deploy; số trung thực nhất cho
      single-model). Mặc định "val" để so với run cũ.
      [E3] CatBoost iterations 4000→6000 (chạm trần 2/3 fold; ES tự chặn).
      [E4] OOF deploy metrics (RMSLE/NWRMSLE với w_deploy) trong summary+meta.

SYNC CONTRACT (4 nơi — sửa 1 phải sửa cả 4):
  1) train.py (file này)
  2) predict.py v3 (tự chọn recursive/direct theo meta["min_lag"]; WINDOW_DAYS=90)
  3) ml_service/app/inference.py::predict_recursive
  4) features_ext.py  ← df = add_extra_features(engineer_features(window), min_lag)

CÁCH ĐỌC SỐ (quan trọng cho báo cáo):
  • "fold-tuned"   : RMSLE blend của fold — hơi LẠC QUAN (ES + w chọn trên chính val)
  • "honest (LOO)" : w từ fold khác → TRUNG THỰC — dùng cho Bảng 4.x
  • "deploy OOF"   : toàn OOF với w deploy — gần nhất với hành vi submission
"""

import inspect
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_absolute_error

# ====================== 1. THIẾT LẬP IMPORT & ĐƯỜNG DẪN ======================
SRC_DIR = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
PROJECT_ROOT = BASE_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cluster_features import ClusterFeatureEngineer
from data_loader import load_data
from features_ext import (
    FEATURE_SPEC_VERSION,
    add_extra_features,
    build_preprocessor_v2,
    complete_panel,
    extra_numeric_features,
)
from preprocessor import engineer_features

LOCAL_MODELS_DIR = BASE_DIR / "models"
SERVICE_MODELS_DIR = PROJECT_ROOT / "ml_service" / "models"
MODEL_DIRS = [LOCAL_MODELS_DIR] + (
    [SERVICE_MODELS_DIR] if SERVICE_MODELS_DIR != LOCAL_MODELS_DIR else []
)
for d in MODEL_DIRS:
    d.mkdir(parents=True, exist_ok=True)


# ====================== 2. THAM SỐ ======================
TOTAL_DAYS = 730
N_SPLITS = 3
MIN_TRAIN_DAYS = 365

# ====================== CHỌN RUN ======================
# RUN 1 (re-forecast hằng ngày / ml_service):
#     MIN_LAG=1,  VAL_DAYS=28, RUN_RECURSIVE_BACKTEST=True,  ARTIFACT_SUFFIX=""
# RUN 2 (DIRECT cho Kaggle 16 ngày — sau [D], val RMSLE là one-shot trung thực):
#     MIN_LAG=16, VAL_DAYS=16, RUN_RECURSIVE_BACKTEST=False, ARTIFACT_SUFFIX="_direct16"
MIN_LAG = 1
VAL_DAYS = 28
RUN_RECURSIVE_BACKTEST = True
ARTIFACT_SUFFIX = ""

# --- [1][3] dữ liệu & feature ---
COMPLETE_PANEL = True
USE_CROSS_SERIES = False
WARMUP_DAYS = 28             # tự nâng max(WARMUP, MIN_LAG+28)
EVAL_ON_OBSERVED_ONLY = False
RUN_LEAK_AUDIT = True        # [D-guard]

CLUSTER_SMOOTHING = 10.0
BLEND_WEIGHT_STRATEGY = "oof"   # "oof" | "median" | "mean" | "recent"

# --- [E2] early stopping ---
#   "val"      : ES trên fold-val (số đẹp hơn nhưng lạc quan; mặc định để so run cũ)
#   "inner"    : tách INNER_ES_DAYS cuối train làm tập dừng (trung thực, mất dữ liệu gần)
#   "retrain"  : inner-ES tìm iterations → RETRAIN toàn train với iterations đó →
#                eval val. Mirror đúng deploy — số trung thực NHẤT cho single-model
#                (tốn ~gấp đôi thời gian train mỗi fold).
ES_MODE = "val"
INNER_ES_DAYS = 14
RETRAIN_ITER_BUFFER = 1.10   # [E2] buffer khi retrain phase-2

# --- backtest đệ quy (mirror predict.py; chỉ RUN 1) ---
BACKTEST_WINDOW_DAYS = 90
BACKTEST_LEAD_BUFFER_DAYS = 2
DEBUG_PER_FAMILY = True

# --- MENU THỬ NGHIỆM ---
N_SEEDS = 2                   # trung bình 2 seed LGBM — giảm phương sai
RECENCY_HALF_LIFE_DAYS = 180  # ưu tiên dữ liệu gần đây (thích nghi drift)
USE_PERISHABLE_WEIGHT = False
USE_TWEEDIE_MEMBER = False
FINAL_ITER_SCALE = 1.10

# --- [H] Tự động dò hyperparameter (Optuna, offline only) ---
RUN_HYPERPARAM_SEARCH = True
OPTUNA_N_TRIALS = 25
OPTUNA_TIMEOUT_SEC = 45 * 60          # chặn thời gian tối đa cho study
OPTUNA_MIN_IMPROVEMENT = 0.01         # chấp nhận khi fold-3 RMSLE tốt hơn ≥1%

CATBOOST_LOSS = "RMSE"

LGBM_PARAMS = {
    "n_estimators": 4000,
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

TWEEDIE_PARAMS = {
    **LGBM_PARAMS,
    "objective": "tweedie",
    "tweedie_variance_power": 1.2,
    "n_estimators": 3000,
    "random_state": 202,
}

CATBOOST_PARAMS = {
    # [E3] 4000 → 6000: chạm trần ở 2/3 fold run trước (3996/4000, 4000/4000).
    # ES=100 tự chặn khi đã đủ — fold 2 (2975) không tốn thêm giây nào.
    "iterations": 6000,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
    "verbose": False,
    "loss_function": CATBOOST_LOSS,
    "early_stopping_rounds": 100,
    "task_type": "CPU",
}

CAT_FEATURES = ["store_nbr", "family", "city", "state", "type", "holiday_type"]

LEGACY_NUM_FEATURES = [
    "transactions_lag1",
    "sales_lag7", "sales_lag14", "sales_lag28",
    "sales_rolling_mean7", "sales_rolling_mean14", "sales_rolling_mean30",
    "sales_std7", "sales_std28",   # [F2] preprocessor v1/nhánh local
]

# [D] Độ trễ THẬT của cột legacy — preprocessor.py shift hard-code, không biết
# MIN_LAG. Không lọc thì model DIRECT thấy giá trị THẬT của ngày TRONG val window
# (serving chỉ có NaN→0) → val_rmsle optimistic giả.
LEGACY_LAG_DAYS = {
    "sales_lag7": 7,
    "sales_lag14": 14,
    "sales_lag28": 28,
    "sales_rolling_mean7": 1,   # shift(1).rolling(7)
    "sales_rolling_mean14": 1,  # shift(1).rolling(14)
    "sales_rolling_mean30": 1,  # shift(1).rolling(30)
    "sales_std7": 1,            # shift(1).rolling(7).std
    "sales_std28": 1,           # shift(1).rolling(28).std
    "transactions_lag1": 1,
}

PASSTHROUGH_FEATURES = [
    "dayofweek", "month", "is_weekend",
    "oil_price", "cluster", "perishable",
    "is_holiday_lag1", "is_holiday_lag2",
    "is_holiday_lead1", "is_holiday_lead2",
    "is_tier1_cluster", "is_back_to_school", "is_day_after_payday",
    "onpromotion", "is_earthquake_period", "is_holiday",
    # [F2] lịch/ngày lương — giờ tính trong preprocessor.engineer_features (dùng chung
    # cho cả 33 local models); features_ext drop-then-recreate nên không nhân đôi
    "is_payday", "days_from_payday", "days_to_payday",
    "dayofmonth", "weekofmonth", "dayofyear", "is_month_start", "is_month_end",
]
# [A] Khai báo tĩnh — transform() tạo cột SAU build_feature_lists
CLUSTER_OUTPUT_COLS = [
    "cluster_mean_sales", "cluster_median_sales", "cluster_std_sales",
    "cluster_family_mean_sales", "cluster_promo_mean_sales", "cluster_promo_lift",
]

LGBM_UNKNOWN_SENTINEL = 9999
REQUIRED_COLS = [
    "date", "target", "store_nbr", "family", "cluster", "onpromotion",
    "is_earthquake_period", "city", "state", "type", "holiday_type",
    "oil_price", "perishable",
]
FFILL_COLS = ["transactions", "transactions_lag1"]
KNOWN_SERVING_PROXIES = {"transactions_lag1"}

FEATURE_STATE: dict = {}


# ====================== 3. HÀM PHỤ TRỢ ======================
def _present(cols, df):
    return [c for c in cols if c in df.columns]


def build_feature_lists(df: pd.DataFrame) -> dict:
    extra = _present(
        extra_numeric_features(MIN_LAG, "transactions" in df.columns,
                               include_cross_series=USE_CROSS_SERIES), df)

    missing_lag = [c for c in LEGACY_NUM_FEATURES if c not in LEGACY_LAG_DAYS]
    if missing_lag:
        raise ValueError(f"LEGACY_LAG_DAYS thiếu độ trễ của: {missing_lag}")

    legacy_eligible = [c for c in LEGACY_NUM_FEATURES if LEGACY_LAG_DAYS[c] >= MIN_LAG]
    legacy = _present(legacy_eligible, df)
    legacy_dropped = [c for c in LEGACY_NUM_FEATURES if c not in legacy_eligible]

    passthrough = _present(PASSTHROUGH_FEATURES, df)
    cluster = list(CLUSTER_OUTPUT_COLS) + ["is_tier1_cluster"]
    cat = _present(CAT_FEATURES, df)

    seen = set()
    def _uniq(cols):
        return [c for c in cols if not (c in seen or seen.add(c))]

    state = {
        "num_zero_cols": _uniq(legacy + cluster),
        "num_sentinel_cols": _uniq(extra),
        "cat_cols": _uniq(cat),
        "passthrough_cols": _uniq(passthrough),
        "legacy_dropped_for_min_lag": legacy_dropped,
    }
    state["lgbm_input_cols"] = (
        state["num_zero_cols"] + state["num_sentinel_cols"]
        + state["cat_cols"] + state["passthrough_cols"]
    )
    state["catboost_cols"] = (
        state["cat_cols"] + state["num_zero_cols"]
        + state["num_sentinel_cols"] + state["passthrough_cols"]
    )
    return state


def audit_future_leakage(df_all: pd.DataFrame, anchor: pd.Timestamp,
                         cols_to_check: list) -> list[tuple[str, float]]:
    """
    [D-guard] Bắt mọi cột feature phụ thuộc TARGET của val window.

    Mask target = NaN cho ngày > anchor (đúng thứ serving biết lúc one-shot),
    tính lại pipeline, so từng cột trên dòng val. Cột ĐỔI GIÁ TRỊ = leak.

    PHẢI chạy trên df_all CHƯA CẮT warm-up (chuỗi trẻ đủ history — tránh
    false positive). Chỉ bắt leak từ TARGET; leak exogenous (transactions/oil
    tương lai) do proxy/ffill lo — xem KNOWN_SERVING_PROXIES.

    FIX: cols_to_check (lgbm_input_cols) chứa CẢ cat features, trong đó có
    store_nbr/family trùng với merge-key. Chọn key + [c] khi c ∈ key tạo frame
    2 cột trùng tên → merge crash "column label is not unique". Giống guard
    `c not in key` vốn đã có ở feature_parity_report — giờ áp cho cả 2 nơi.
    Cột key là identity (không phụ thuộc target) — bỏ qua là đúng ngữ nghĩa.
    """
    masked = df_all.copy()
    masked.loc[masked["date"] > anchor, "target"] = np.nan
    masked = add_extra_features(engineer_features(masked), min_lag=MIN_LAG,
                                include_cross_series=USE_CROSS_SERIES)

    val_a = df_all[df_all["date"] > anchor]
    val_b = masked[masked["date"] > anchor]
    key = ["store_nbr", "family", "date"]
    key_set = set(key)

    leaked = []
    for c in dict.fromkeys(cols_to_check):          # dedupe + giữ thứ tự
        if c in key_set:                            # ← FIX: bỏ cột identity trùng key
            continue
        if c not in val_a.columns or c not in val_b.columns:
            continue  # vd cluster TE — chưa tạo ở tầng này, audit ở tầng khác
        a = val_a[key + [c]]
        b = val_b[key + [c]]
        m = a.merge(b, on=key, suffixes=("_r", "_m"))
        diff = (m[f"{c}_r"].fillna(-9e9) != m[f"{c}_m"].fillna(-9e9))
        if diff.any():
            leaked.append((c, round(float(diff.mean()), 3)))
    return leaked


def prepare_dataset(force_rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trả (df đã cắt warm-up, backtest_hist chưa cắt) — như v2.3."""
    print(">>> Loading data...")
    df = load_data(force_rebuild=force_rebuild)

    if COMPLETE_PANEL:
        df = complete_panel(df)

    print(">>> Engineering base features...")
    df = engineer_features(df)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"load_data/engineer_features thiếu cột: {missing}")

    print(f">>> Adding extra features (min_lag={MIN_LAG})...")
    df_all = add_extra_features(df, min_lag=MIN_LAG, include_cross_series=USE_CROSS_SERIES)

    if "observed" not in df_all.columns:
        df_all["observed"] = True

    effective_warmup = max(WARMUP_DAYS, MIN_LAG + 28)
    if effective_warmup != WARMUP_DAYS:
        print(f">>> Warmup nâng {WARMUP_DAYS} → {effective_warmup} ngày "
              f"(MIN_LAG={MIN_LAG} + rolling 28)")
    if effective_warmup:
        first = df_all.groupby(["store_nbr", "family"], observed=True)["date"].transform("min")
        keep = df_all["date"] >= first + pd.Timedelta(days=effective_warmup)
        print(f">>> Bỏ {int((~keep).sum()):,} dòng warm-up ({effective_warmup} ngày đầu mỗi chuỗi)")
        df = df_all[keep].copy()
    else:
        df = df_all.copy()

    backtest_hist = df_all

    df = df.sort_values(["store_nbr", "family", "date"], ignore_index=True)
    date_range = (df["date"].max() - df["date"].min()).days + 1
    print(f">>> Data range: {df['date'].min().date()} → {df['date'].max().date()} "
          f"({date_range} days, {len(df):,} rows)")
    if date_range < TOTAL_DAYS - 30:
        print(f"Warning: Data has only {date_range} days, expected ~{TOTAL_DAYS} days")
    return df, backtest_hist


def walk_forward_splits(df, n_splits=3, val_days=28, min_train_days=365):
    dates = sorted(df["date"].unique())
    if not dates:
        raise ValueError("No dates found in dataframe")

    max_date, min_date = pd.Timestamp(dates[-1]), pd.Timestamp(dates[0])
    total_days = (max_date - min_date).days + 1
    print(f">>> Total data span: {total_days} days ({min_date.date()} → {max_date.date()})")
    print(f">>> Walk-forward: {n_splits} folds, val={val_days}d, min_train={min_train_days}d")

    temp = []
    for i in range(n_splits):
        val_end = max_date - pd.Timedelta(days=i * val_days)
        val_start = val_end - pd.Timedelta(days=val_days - 1)
        train_end = val_start - pd.Timedelta(days=1)
        train_days = (train_end - min_date).days + 1
        if train_days < min_train_days:
            print(f"⚠️  Skip fold {i+1}: train quá ngắn ({train_days}d < {min_train_days}d)")
            break
        temp.append((
            (df["date"] >= min_date) & (df["date"] <= train_end),
            (df["date"] >= val_start) & (df["date"] <= val_end),
            {"train_start": min_date.date(), "train_end": train_end.date(),
             "train_days": train_days, "val_start": val_start.date(),
             "val_end": val_end.date(), "val_days": val_days},
        ))

    splits = []
    for idx, (tm, vm, fi) in enumerate(list(reversed(temp)), start=1):
        fi["fold"] = idx
        splits.append((tm, vm, fi))
        print(f"    Fold {idx}: train → {fi['train_end']} ({fi['train_days']}d) "
              f"| val {fi['val_start']} → {fi['val_end']}")
    if not splits:
        raise ValueError("No valid walk-forward splits could be created.")
    return splits


# ---------------------- METRICS ----------------------
def evaluate(y_true, y_pred, perishable=None) -> dict:
    y_pred = np.clip(np.asarray(y_pred, dtype="float64"), 0, None)
    y_true = np.maximum(np.asarray(y_true, dtype="float64"), 0)
    err = y_pred - y_true
    log_err = np.log1p(y_pred) - np.log1p(y_true)

    pos = y_true > 0
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    # WMAPE: WAPE tổng quát có trọng số (perishable ×1.5 — chuẩn chấm Favorita);
    # không có perishable thì w=1 → WMAPE ≡ WAPE
    if perishable is not None:
        w_abs = np.where(np.asarray(perishable) > 0, 1.5, 1.0)
    else:
        w_abs = 1.0
    out = {
        "rmsle": float(np.sqrt(np.mean(log_err ** 2))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(np.mean(np.abs(err[pos]) / y_true[pos]) * 100) if pos.any() else None,
        "wape": float(np.abs(err).sum() / max(y_true.sum(), 1e-9) * 100),
        "wmape": float((w_abs * np.abs(err)).sum() / max((w_abs * y_true).sum(), 1e-9) * 100),
        "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else None,
        "zero_rate_true": float((y_true == 0).mean()),
    }
    if perishable is not None:
        w = np.where(np.asarray(perishable) > 0, 1.25, 1.0)
        out["nwrmsle"] = float(np.sqrt(np.sum(w * log_err ** 2) / w.sum()))
    return out


def sample_weights(df: pd.DataFrame, ref_date) -> np.ndarray | None:
    w = None
    if RECENCY_HALF_LIFE_DAYS:
        age = (pd.Timestamp(ref_date) - df["date"]).dt.days.clip(lower=0).to_numpy()
        w = 0.5 ** (age / float(RECENCY_HALF_LIFE_DAYS))
    if USE_PERISHABLE_WEIGHT and "perishable" in df.columns:
        pw = np.where(df["perishable"].to_numpy() > 0, 1.25, 1.0)
        w = pw if w is None else w * pw
    return None if w is None else w.astype("float64")


def find_best_weight(y_true, a_log, b_log, perishable=None) -> tuple[float, float]:
    """Tối ưu w trong LOG-SPACE + guard: nghiệm không được tệ hơn model đơn."""
    def loss(w):
        pred = np.expm1(w * a_log + (1 - w) * b_log)
        m = evaluate(y_true, pred, perishable)
        return m["nwrmsle"] if (USE_PERISHABLE_WEIGHT and perishable is not None) else m["rmsle"]

    res = minimize_scalar(loss, bounds=(0.0, 1.0), method="bounded")
    l0, l1 = loss(0.0), loss(1.0)
    best_end = min(l0, l1)
    if res.fun > best_end + 1e-6:
        w = 0.0 if l0 <= l1 else 1.0
        print(f"    ⚠️ Blend optimizer kém hơn điểm biên → w={w:.2f}")
        return w, best_end
    return float(res.x), float(res.fun)


# ---------------------- [H] OPTUNA HYPERPARAM SEARCH (offline only) ----------------------
def tune_lgbm_on_fold(df, splits) -> tuple[bool, dict]:
    """
    Dò hyperparameter LGBM trên fold MỚI NHẤT (val window cuối dữ liệu).

    Cổng chấp nhận: chỉ cập nhật LGBM_PARAMS khi blend RMSLE trên fold đó
    (tuned LGBM + CatBoost mặc định) tốt hơn baseline ≥ OPTUNA_MIN_IMPROVEMENT.
    Trả về (accepted, info) để ghi vào ensemble_meta.json.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_mask, val_mask, fold_info = splits[-1]
    train_df, val_df = df[train_mask], df[val_mask]
    ref_date = train_df["date"].max()

    cluster_engineer = ClusterFeatureEngineer(smoothing=CLUSTER_SMOOTHING)
    train_df = cluster_engineer.fit_transform(train_df, target_col="target")
    val_df = cluster_engineer.transform(val_df)

    S = FEATURE_STATE
    prep = build_preprocessor_v2(S["num_zero_cols"], S["num_sentinel_cols"],
                                 S["cat_cols"], S["passthrough_cols"])
    X_fit = _fix_unknown_categorical(prep.fit_transform(train_df))
    X_val = _fix_unknown_categorical(prep.transform(val_df))
    lgbm_cat_cols = [c for c in CAT_FEATURES if c in X_fit.columns]
    y_fit_log = np.log1p(train_df["target"].values)
    y_val_log = np.log1p(val_df["target"].values)
    y_val_true = val_df["target"].values
    peri_val = val_df["perishable"].values if "perishable" in val_df.columns else None
    w_fit = sample_weights(train_df, ref_date)

    def _lgbm_eval(extra_params: dict) -> np.ndarray:
        """Fit LGBM với params = hiện tại + extra_params, trả pred gốc (không log)."""
        m = _lgbm_fit(LGBMRegressor(**{**LGBM_PARAMS, **extra_params}),
                      X_fit, y_fit_log, w_fit, X_val, y_val_log, lgbm_cat_cols)
        return np.clip(np.expm1(m.predict(X_val)), 0, None)

    # ---- CatBoost mặc định fit 1 lần (dùng chung cho baseline & best) ----
    cat_cols = [c for c in CAT_FEATURES if c in train_df.columns]
    all_features = [c for c in S["catboost_cols"] if c in train_df.columns]
    cat = CatBoostRegressor(**CATBOOST_PARAMS)
    cat.fit(Pool(data=train_df[all_features], label=y_fit_log,
                 cat_features=cat_cols, weight=w_fit), verbose=False)
    cat_pred = np.clip(np.expm1(_predict_catboost(cat, val_df)), 0, None)

    def _blend_rmsle(lgbm_pred: np.ndarray) -> float:
        lgbm_log = np.log1p(lgbm_pred)
        cat_log = np.log1p(cat_pred)
        w, _ = find_best_weight(y_val_true, lgbm_log, cat_log, peri_val)
        return evaluate(y_val_true, np.clip(np.expm1(w * lgbm_log + (1 - w) * cat_log), 0, None),
                        peri_val)["rmsle"]

    base_pred = _lgbm_eval({})
    base_blend = _blend_rmsle(base_pred)
    print(f">>> [H] Baseline fold-{fold_info['fold']}: LGBM "
          f"{evaluate(y_val_true, base_pred, peri_val)['rmsle']:.5f} | blend {base_blend:.5f}")

    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 150, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }
        return evaluate(y_val_true, _lgbm_eval(params), peri_val)["rmsle"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS, timeout=OPTUNA_TIMEOUT_SEC,
                   show_progress_bar=False)
    best = study.best_trial

    best_pred = _lgbm_eval(best.params)
    best_blend = _blend_rmsle(best_pred)
    improvement = (base_blend - best_blend) / base_blend
    accepted = improvement >= OPTUNA_MIN_IMPROVEMENT
    if accepted:
        LGBM_PARAMS.update(best.params)
    verdict = ("CHẤP NHẬN, cập nhật LGBM_PARAMS" if accepted
               else f"GIỮ tham số cũ (không đạt ngưỡng ≥{OPTUNA_MIN_IMPROVEMENT:.0%})")
    print(f">>> [H] Best LGBM {best.value:.5f} | blend {best_blend:.5f} "
          f"({improvement:+.2%} so baseline {base_blend:.5f}) → {verdict}")

    info = {
        "enabled": True,
        "n_trials": len(study.trials),
        "fold": fold_info["fold"],
        "val_start": str(fold_info["val_start"]), "val_end": str(fold_info["val_end"]),
        "baseline_blend_rmsle": round(base_blend, 5),
        "best_blend_rmsle": round(best_blend, 5),
        "best_lgbm_rmsle": round(best.value, 5),
        "improvement_pct": round(improvement * 100, 3),
        "min_improvement_pct": OPTUNA_MIN_IMPROVEMENT * 100,
        "accepted": bool(accepted),
        "best_params": {k: (float(v) if isinstance(v, float) else int(v))
                        for k, v in best.params.items()},
    }
    return accepted, info


# ---------------------- [E1] LOO HONEST BLEND ----------------------
def loo_honest_blend(oof):
    """
    Đánh giá blend bằng weight leave-one-fold-out: w dùng để chấm fold i được
    tối ưu CHỈ trên OOF của các fold khác → không có fold nào được chấm bằng w
    fit trên chính nó. Đây là con số TRUNG THỰC cho báo cáo (đối trọng với
    blend_rmsle fold-tuned vốn hơi lạc quan vì w + ES chọn trên chính val).
    Trả (rows chi tiết, mean dict) — None nếu < 2 fold.
    """
    n = len(oof)
    if n < 2:
        return None, None

    rows = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        y = np.concatenate([oof[j]["y"] for j in idx])
        a = np.concatenate([oof[j]["lgb"] for j in idx])
        b = np.concatenate([oof[j]["cat"] for j in idx])
        peri = (np.concatenate([oof[j]["peri"] for j in idx])
                if all(oof[j]["peri"] is not None for j in idx) else None)
        w_i, _ = find_best_weight(y, a, b, peri)

        o = oof[i]
        pred = np.expm1(w_i * o["lgb"] + (1 - w_i) * o["cat"])
        m = evaluate(o["y"], pred, o["peri"])
        rows.append({
            "fold": i + 1,
            "w_loo": round(float(w_i), 4),
            "rmsle_honest": round(m["rmsle"], 5),
            "nwrmsle_honest": round(m["nwrmsle"], 5) if m.get("nwrmsle") else None,
        })

    mean_rmsle = round(float(np.mean([r["rmsle_honest"] for r in rows])), 5)
    nw_vals = [r["nwrmsle_honest"] for r in rows if r["nwrmsle_honest"] is not None]
    mean_nwrmsle = round(float(np.mean(nw_vals)), 5) if len(nw_vals) == len(rows) else None
    return rows, {"rmsle": mean_rmsle, "nwrmsle": mean_nwrmsle}


def oof_deploy_metrics(chosen_w: float, oof) -> dict:
    """[E4] Toàn OOF với w deploy — gần nhất hành vi submission (w được fit trên
    cùng OOF nên vẫn hơi lạc quan — LOO mới là số trung thực tuyệt đối)."""
    y = np.concatenate([o["y"] for o in oof])
    a = np.concatenate([o["lgb"] for o in oof])
    b = np.concatenate([o["cat"] for o in oof])
    peri = (np.concatenate([o["peri"] for o in oof])
            if all(o["peri"] is not None for o in oof) else None)
    pred = np.expm1(chosen_w * a + (1 - chosen_w) * b)
    return evaluate(y, pred, peri)


def _fix_unknown_categorical(X: pd.DataFrame) -> pd.DataFrame:
    cat_cols = [c for c in CAT_FEATURES if c in X.columns]
    if not cat_cols or not bool((X[cat_cols] < 0).any().any()):
        return X
    for col in cat_cols:
        neg = X[col] < 0
        if neg.any():
            X.loc[neg, col] = LGBM_UNKNOWN_SENTINEL
    return X


def _predict_catboost(cat_model, X: pd.DataFrame, raise_on_missing: bool = False) -> np.ndarray:
    """ĐƯỜNG PREDICT CATBOOST DUY NHẤT — dùng chung cho fold, backtest, serving."""
    feature_names = list(cat_model.feature_names_ or [])
    if not feature_names:
        raise ValueError("CatBoost model không có feature_names_")

    missing = [c for c in feature_names if c not in X.columns]
    if missing:
        msg = f"    CatBoost thiếu {len(missing)} feature → fill 0: {missing[:8]}{'...' if len(missing) > 8 else ''}"
        if raise_on_missing:
            raise ValueError(msg.strip())
        print(msg)

    X_cat = X.reindex(columns=feature_names, fill_value=0).copy()
    if "store_nbr" in X_cat.columns and not pd.api.types.is_integer_dtype(X_cat["store_nbr"]):
        X_cat["store_nbr"] = X_cat["store_nbr"].round().astype("int64")
    for col in CAT_FEATURES:
        if col not in X_cat.columns or col == "store_nbr":
            continue
        if col == "holiday_type":
            X_cat[col] = X_cat[col].fillna("Normal Day")
        X_cat[col] = X_cat[col].astype(str)

    cat_cols_present = [c for c in CAT_FEATURES if c in X_cat.columns]
    return cat_model.predict(Pool(data=X_cat, cat_features=cat_cols_present))


def _lgbm_fit(model, X, y, w, X_val, y_val, cat_cols, es_rounds=100):
    """Tương thích cả LightGBM dùng eval_set lẫn bản dùng eval_X/eval_y."""
    kw = dict(sample_weight=w, eval_metric="rmse",
              categorical_feature=cat_cols,
              callbacks=[early_stopping(es_rounds, verbose=False), log_evaluation(period=0)])
    params = inspect.signature(model.fit).parameters
    if "eval_X" in params:
        kw.update(eval_X=X_val, eval_y=y_val)
    else:
        kw.update(eval_set=[(X_val, y_val)])
    model.fit(X, y, **kw)
    return model


# ====================== 4. RECURSIVE BACKTEST (chỉ RUN 1) ======================
def feature_parity_report(day_featured: pd.DataFrame, ref_df: pd.DataFrame,
                          cols: list, top_n: int = 12) -> None:
    key = ["store_nbr", "family"]
    candidates = [c for c in dict.fromkeys(cols) if c not in key]

    missing_rec = [c for c in candidates if c in ref_df.columns and c not in day_featured.columns]
    missing_fold = [c for c in candidates if c in day_featured.columns and c not in ref_df.columns]
    if missing_rec or missing_fold:
        print(f"    [parity] ❌ cột chỉ có 1 phía — đệ quy thiếu: {missing_rec} | "
              f"fold thiếu: {missing_fold}")

    num_cols = [c for c in candidates
                if c in day_featured.columns and c in ref_df.columns
                and pd.api.types.is_numeric_dtype(day_featured[c])]
    a = day_featured[key + num_cols].copy()
    b = ref_df[key + num_cols].copy()
    m = a.merge(b, on=key, suffixes=("_rec", "_fold"))
    if m.empty:
        print("    [parity] không khớp được dòng nào để so sánh")
        return

    rows = []
    for c in num_cols:
        x = pd.to_numeric(m[f"{c}_rec"], errors="coerce")
        y = pd.to_numeric(m[f"{c}_fold"], errors="coerce")
        scale = float(np.nanmean(np.abs(y))) or 1.0
        rows.append({
            "feature": c,
            "rel_diff": float(np.nanmean(np.abs(x - y)) / scale),
            "nan_rec": float(x.isna().mean()),
            "nan_fold": float(y.isna().mean()),
        })
    rep = pd.DataFrame(rows).sort_values("rel_diff", ascending=False)
    bad = rep[(rep["rel_diff"] > 0.01) | ((rep["nan_rec"] - rep["nan_fold"]).abs() > 0.01)]

    print(f"    [parity] so {len(m):,} dòng ngày đầu backtest, {len(num_cols)} feature số")
    if bad.empty:
        print("    [parity] ✅ không cột nào lệch >1% — vấn đề KHÔNG nằm ở feature")
    else:
        print(f"    [parity] ⚠️ {len(bad)} cột lệch (đệ quy vs fold):")
        for _, r in bad.head(top_n).iterrows():
            tag = "  ← proxy serving (ffill), chấp nhận" if r["feature"] in KNOWN_SERVING_PROXIES else ""
            print(f"        {r['feature']:<26} rel_diff={r['rel_diff']:.3f}  "
                  f"NaN {r['nan_fold']:.0%}→{r['nan_rec']:.0%}{tag}")


def recursive_backtest(raw_train, raw_val, cluster_engineer, preprocessor, lgbm, cat, w_lgbm,
                       ref_val=None):
    """Mô phỏng /forecast one-shot: window (d-90, d+2], ffill transactions*,
    blend log-space, ghi ngược dự báo làm lag cho ngày sau."""
    try:
        hist = raw_train.copy()
        future = raw_val.copy()

        future["target"] = np.nan
        for col in ("transactions", "transactions_lag1"):
            if col in future.columns:
                future[col] = np.nan
        for col in CLUSTER_OUTPUT_COLS:
            hist = hist.drop(columns=col, errors="ignore")
            future = future.drop(columns=col, errors="ignore")

        combined = pd.concat([hist, future], ignore_index=True)
        combined["target"] = combined["target"].astype("float64")

        stale_cols = [c for c in FEATURE_STATE.get("num_sentinel_cols", [])
                      if c in combined.columns]
        if stale_cols:
            combined = combined.drop(columns=stale_cols)

        combined = combined.sort_values(["store_nbr", "date"], ignore_index=True)

        for col in FFILL_COLS:
            if col in combined.columns:
                combined[col] = combined.groupby("store_nbr", observed=True)[col].ffill()

        val_dates = sorted(future["date"].unique())
        h_map = {d: i + 1 for i, d in enumerate(val_dates)}
        pred_rows = []

        for d in val_dates:
            d = pd.Timestamp(d)
            window = combined[
                (combined["date"] > d - pd.Timedelta(days=BACKTEST_WINDOW_DAYS))
                & (combined["date"] <= d + pd.Timedelta(days=BACKTEST_LEAD_BUFFER_DAYS))
            ].copy()

            featured = add_extra_features(engineer_features(window), min_lag=MIN_LAG,
                                          include_cross_series=USE_CROSS_SERIES)
            if cluster_engineer is not None:
                featured = cluster_engineer.transform(featured)
            day_featured = featured[featured["date"] == d]

            if day_featured.empty:
                print(f"    !!! {d.date()}: day_featured empty — fill target=0")
                combined.loc[combined["date"] == d, "target"] = 0.0
                continue

            if d == pd.Timestamp(val_dates[0]) and ref_val is not None:
                ref_day = ref_val[ref_val["date"] == d]
                if not ref_day.empty:
                    feature_parity_report(
                        day_featured, ref_day,
                        FEATURE_STATE.get("lgbm_input_cols", list(day_featured.columns)),
                    )

            X_lgb = _fix_unknown_categorical(preprocessor.transform(day_featured))
            lgb_raw = lgbm.predict(X_lgb)
            cat_raw = _predict_catboost(cat, day_featured)
            pred = np.clip(np.expm1(w_lgbm * lgb_raw + (1 - w_lgbm) * cat_raw), 0, None)

            mask = combined["date"] == d
            key = combined.loc[mask, "store_nbr"].astype(str) + "|" + combined.loc[mask, "family"].astype(str)
            pred_key = day_featured["store_nbr"].astype(str) + "|" + day_featured["family"].astype(str)
            combined.loc[mask, "target"] = (
                key.map(pd.Series(pred, index=pred_key.values)).astype("float64").values
            )

            day_out = day_featured[["date", "store_nbr", "family"]].copy()
            day_out["pred"] = pred
            day_out["horizon_day"] = h_map[pd.Timestamp(d)]
            pred_rows.append(day_out)

        if not pred_rows:
            return None

        preds = pd.concat(pred_rows, ignore_index=True)
        cols = ["date", "store_nbr", "family", "target"]
        if "perishable" in raw_val.columns:
            cols.append("perishable")
        merged = preds.merge(raw_val[cols], on=["date", "store_nbr", "family"], how="inner")

        merged["week"] = ((merged["horizon_day"] - 1) // 7 + 1).astype(int)
        weekly = {
            f"week {w}": round(evaluate(g["target"].values, g["pred"].values)["rmsle"], 5)
            for w, g in merged.groupby("week")
        }

        if DEBUG_PER_FAMILY:
            rows_dbg = []
            for (fam, w), g in merged.groupby(["family", "week"]):
                rows_dbg.append((fam, w, evaluate(g["target"].values, g["pred"].values)["rmsle"]))
            per_fam = (pd.DataFrame(rows_dbg, columns=["family", "week", "rmsle"])
                       .pivot(index="family", columns="week", values="rmsle"))
            wk = sorted(per_fam.columns)
            if len(wk) >= 2:
                w1, w4 = per_fam[wk[0]], per_fam[wk[-1]]
                ratio = (w4 / w1.replace(0, np.nan)).sort_values(ascending=False)
                print("    [debug] family xấu đi nhất theo horizon "
                      f"(tuần cuối/tuần đầu, {wk[0]}→{wk[-1]}):")
                for fam, r in ratio.head(6).items():
                    print(f"        {fam:<32} w{wk[0]}={w1[fam]:.3f}  "
                          f"w{wk[-1]}={w4[fam]:.3f}  ratio={r:.2f}")

        full = evaluate(merged["target"].values, merged["pred"].values,
                        merged["perishable"].values if "perishable" in merged else None)
        return {"rmsle": full["rmsle"], "nwrmsle": full.get("nwrmsle"), "weekly": weekly}
    except Exception as e:
        import traceback
        print(f"    ⚠️ Recursive backtest lỗi (bỏ qua): {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        return None


# ====================== 5. TRAIN 1 FOLD ======================
def _train_fold(train_df: pd.DataFrame, val_df: pd.DataFrame,
                backtest_hist: pd.DataFrame | None = None) -> dict:
    raw_train, raw_val = train_df, val_df
    ref_date = train_df["date"].max()

    cluster_engineer = ClusterFeatureEngineer(smoothing=CLUSTER_SMOOTHING)
    train_df = cluster_engineer.fit_transform(train_df, target_col="target")
    val_df = cluster_engineer.transform(val_df)

    # [E2] chia tập theo chế độ ES
    if ES_MODE in ("inner", "retrain"):
        cut = ref_date - pd.Timedelta(days=INNER_ES_DAYS - 1)
        es_df = train_df[train_df["date"] >= cut]
        fit_df = train_df[train_df["date"] < cut]
    else:  # "val"
        fit_df, es_df = train_df, val_df

    y_train_log = np.log1p(train_df["target"].values)          # full train (phase-2 [E2])
    y_fit_log = np.log1p(fit_df["target"].values)
    y_es_log = np.log1p(es_df["target"].values)
    y_val_true = val_df["target"].values
    peri_val = val_df["perishable"].values if "perishable" in val_df.columns else None
    w_train = sample_weights(train_df, ref_date)
    w_fit = sample_weights(fit_df, ref_date)
    w_es = sample_weights(es_df, ref_date)

    S = FEATURE_STATE

    # ---------------- LIGHTGBM ----------------
    # Fit prep trên TOÀN BỘ phía train (deploy-correct; val không bị đụng)
    fold_prep = build_preprocessor_v2(
        S["num_zero_cols"], S["num_sentinel_cols"], S["cat_cols"], S["passthrough_cols"]
    )
    X_train_full = _fix_unknown_categorical(fold_prep.fit_transform(train_df))
    X_val = _fix_unknown_categorical(fold_prep.transform(val_df))
    if ES_MODE == "val":
        X_fit, X_es = X_train_full, X_val
    else:
        X_fit = _fix_unknown_categorical(fold_prep.transform(fit_df))
        X_es = _fix_unknown_categorical(fold_prep.transform(es_df))
    lgbm_cat_cols = [c for c in CAT_FEATURES if c in X_fit.columns]

    lgbm_logs, lgb_iters = [], []
    lgbm = None
    for s in range(N_SEEDS):
        p = {**LGBM_PARAMS, "random_state": LGBM_PARAMS["random_state"] + 1000 * s}
        if ES_MODE == "retrain":
            # [E2] Phase 1: inner-ES tìm iterations (KHÔNG đụng val)
            m1 = _lgbm_fit(LGBMRegressor(**p), X_fit, y_fit_log, w_fit,
                           X_es, y_es_log, lgbm_cat_cols)
            it1 = m1.best_iteration_ or LGBM_PARAMS["n_estimators"]
            # Phase 2: retrain TOÀN train với iterations đã chọn — mirror deploy
            m = LGBMRegressor(**{**p, "n_estimators": max(10, int(it1 * RETRAIN_ITER_BUFFER))})
            m.fit(X_train_full, y_train_log, sample_weight=w_train,
                  categorical_feature=lgbm_cat_cols)
            best_iter_s = int(it1)
        else:
            m = _lgbm_fit(LGBMRegressor(**p), X_fit, y_fit_log, w_fit,
                          X_es, y_es_log, lgbm_cat_cols)
            best_iter_s = m.best_iteration_ or LGBM_PARAMS["n_estimators"]
        lgbm_logs.append(m.predict(X_val))
        lgb_iters.append(int(best_iter_s))
        if s == 0:
            lgbm = m
    lgbm_log = np.mean(lgbm_logs, axis=0)
    lgbm_pred = np.clip(np.expm1(lgbm_log), 0, None)
    lgb_best_iter = int(np.mean(lgb_iters))

    # ---------------- CATBOOST ----------------
    cat_cols = [c for c in CAT_FEATURES if c in fit_df.columns]
    all_features = [c for c in S["catboost_cols"] if c in fit_df.columns]
    fit_pool = Pool(data=fit_df[all_features], label=y_fit_log, cat_features=cat_cols, weight=w_fit)
    es_pool = Pool(data=es_df[all_features], label=y_es_log, cat_features=cat_cols, weight=w_es)
    full_train_pool = None
    if ES_MODE == "retrain":
        full_train_pool = Pool(data=train_df[all_features], label=y_train_log,
                               cat_features=cat_cols, weight=w_train)

    cat_logs, cat_iters = [], []
    cat = None
    for s in range(N_SEEDS):
        p = {**CATBOOST_PARAMS, "random_seed": CATBOOST_PARAMS["random_seed"] + 1000 * s}
        if ES_MODE == "retrain":
            m1 = CatBoostRegressor(**p)
            m1.fit(fit_pool, eval_set=es_pool, verbose=False)
            bi = m1.get_best_iteration()
            it1 = (bi + 1) if bi is not None else p["iterations"]
            m = CatBoostRegressor(**{**p,
                                     "iterations": max(10, int(it1 * RETRAIN_ITER_BUFFER)),
                                     "early_stopping_rounds": None})
            m.fit(full_train_pool, verbose=False)
            best_iter_s = int(it1)
        else:
            m = CatBoostRegressor(**p)
            m.fit(fit_pool, eval_set=es_pool, verbose=False)
            bi = m.get_best_iteration()
            best_iter_s = (bi + 1) if bi is not None else p["iterations"]
        cat_logs.append(_predict_catboost(m, val_df))
        cat_iters.append(int(best_iter_s))
        if s == 0:
            cat = m
    cat_log = np.mean(cat_logs, axis=0)
    cat_pred = np.clip(np.expm1(cat_log), 0, None)
    cat_best_iter = int(np.mean(cat_iters))
    if ES_MODE != "retrain" and cat_best_iter >= CATBOOST_PARAMS["iterations"]:
        print(f"    ⚠️ CatBoost chạm trần {CATBOOST_PARAMS['iterations']} iterations")

    # ---------------- TWEEDIE (tùy chọn) ----------------
    tw_log = None
    if USE_TWEEDIE_MEMBER:
        tw = LGBMRegressor(**TWEEDIE_PARAMS)
        _lgbm_fit(tw, X_fit, np.expm1(y_fit_log), w_fit, X_es, np.expm1(y_es_log), lgbm_cat_cols)
        tw_log = np.log1p(np.clip(tw.predict(X_val), 0, None))

    # ---------------- BLEND (fold-tuned — số này hơi lạc quan, [E1] đối trọng) ----
    best_w, blend_loss = find_best_weight(y_val_true, lgbm_log, cat_log, peri_val)
    blend_log = best_w * lgbm_log + (1 - best_w) * cat_log
    if tw_log is not None:
        w2, _ = find_best_weight(y_val_true, blend_log, tw_log, peri_val)
        blend_log = w2 * blend_log + (1 - w2) * tw_log
    blend_pred = np.clip(np.expm1(blend_log), 0, None)

    m_lgb = evaluate(y_val_true, lgbm_pred, peri_val)
    m_cat = evaluate(y_val_true, cat_pred, peri_val)
    m_blend = evaluate(y_val_true, blend_pred, peri_val)
    best_single = min(m_lgb["rmsle"], m_cat["rmsle"])

    # ---------------- DIRECT weekly (RUN 2; sau [D] là degradation thật) ----
    direct_weekly = None
    if MIN_LAG >= VAL_DAYS:
        hz = (val_df["date"] - ref_date).dt.days.to_numpy()
        wk = (hz - 1) // 7 + 1
        direct_weekly = {
            f"week {w}": round(evaluate(y_val_true[wk == w], blend_pred[wk == w])["rmsle"], 5)
            for w in sorted(np.unique(wk))
        }

    recursive = None
    if RUN_RECURSIVE_BACKTEST:
        hist_src = backtest_hist if backtest_hist is not None else raw_train
        hist_src = hist_src[hist_src["date"] <= ref_date]
        recursive = recursive_backtest(hist_src, raw_val, cluster_engineer,
                                       fold_prep, lgbm, cat, best_w, ref_val=val_df)

    return {
        "es_mode": ES_MODE,                                  # [E2] truy vết
        "lgbm_rmsle": m_lgb["rmsle"], "lgbm_mae": m_lgb["mae"], "lgbm_wape": m_lgb["wape"],
        "lgbm_rmse": round(m_lgb["rmse"], 3), "lgbm_r2": round(m_lgb["r2"], 4) if m_lgb["r2"] is not None else None,
        "lgbm_wmape": round(m_lgb["wmape"], 3),
        "lgbm_best_iter": lgb_best_iter, "lgbm_n_features": int(X_fit.shape[1]),
        "catboost_rmsle": m_cat["rmsle"], "catboost_mae": m_cat["mae"], "catboost_wape": m_cat["wape"],
        "catboost_rmse": round(m_cat["rmse"], 3), "catboost_r2": round(m_cat["r2"], 4) if m_cat["r2"] is not None else None,
        "catboost_wmape": round(m_cat["wmape"], 3),
        "catboost_best_iter": cat_best_iter, "catboost_n_features": len(all_features),
        "blend_weight_lgbm": round(best_w, 4),
        "blend_rmsle": round(m_blend["rmsle"], 5),
        "blend_nwrmsle": round(m_blend["nwrmsle"], 5) if m_blend.get("nwrmsle") else None,
        "blend_mae": round(m_blend["mae"], 2),
        "blend_rmse": round(m_blend["rmse"], 2),
        "blend_wape": round(m_blend["wape"], 3),
        "blend_wmape": round(m_blend["wmape"], 3),
        "blend_r2": round(m_blend["r2"], 4) if m_blend["r2"] is not None else None,
        "zero_rate_true": round(m_blend["zero_rate_true"], 4),
        "blend_gain_pct": round((best_single - m_blend["rmsle"]) / best_single * 100, 3),
        "recursive_rmsle": round(recursive["rmsle"], 5) if recursive else None,
        "recursive_weekly": recursive["weekly"] if recursive else None,
        "direct_weekly": direct_weekly,
        "_oof": {"y": y_val_true, "lgb": lgbm_log, "cat": cat_log, "peri": peri_val},
    }


def select_blend_weight(fold_summaries, oof, strategy=BLEND_WEIGHT_STRATEGY):
    weights = [f["blend_weight_lgbm"] for f in fold_summaries]
    recent = max(fold_summaries, key=lambda f: f["val_end"])
    y = np.concatenate([o["y"] for o in oof])
    a = np.concatenate([o["lgb"] for o in oof])
    b = np.concatenate([o["cat"] for o in oof])
    peri = np.concatenate([o["peri"] for o in oof]) if oof[0]["peri"] is not None else None
    w_oof, _ = find_best_weight(y, a, b, peri)

    candidates = {
        "mean": float(np.mean(weights)),
        "median": float(np.median(weights)),
        "recent": float(recent["blend_weight_lgbm"]),
        "oof": float(w_oof),
    }
    if strategy not in candidates:
        raise ValueError(f"BLEND_WEIGHT_STRATEGY không hợp lệ: {strategy}")
    return candidates[strategy], candidates


# ====================== 6. ARTIFACTS ======================
def save_artifacts(models_dirs, final_lgbm, final_cat, final_prep, final_cluster, meta_dict):
    s = ARTIFACT_SUFFIX
    for models_dir in models_dirs:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_lgbm, models_dir / f"lgbm_model{s}.pkl")
        final_cat.save_model(str(models_dir / f"catboost_model{s}.cbm"))
        joblib.dump(final_prep, models_dir / f"preprocessor{s}.pkl")
        joblib.dump(final_cluster, models_dir / f"cluster_engineer{s}.pkl")
        with open(models_dir / f"ensemble_meta{s}.json", "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2, default=str, ensure_ascii=False)
        print(f"    Saved → {models_dir}  (suffix='{s or None}')")


def _smoke_test_artifacts(models_dir: Path, df_full: pd.DataFrame):
    try:
        s = ARTIFACT_SUFFIX
        lgbm = joblib.load(models_dir / f"lgbm_model{s}.pkl")
        prep = joblib.load(models_dir / f"preprocessor{s}.pkl")
        cluster = joblib.load(models_dir / f"cluster_engineer{s}.pkl")
        if getattr(cluster, "cluster_stats", None) is None:
            raise ValueError("cluster_engineer.pkl chưa fit")
        cat = CatBoostRegressor()
        cat.load_model(str(models_dir / f"catboost_model{s}.cbm"))

        sample = df_full.tail(2000)
        X = _fix_unknown_categorical(prep.transform(sample))
        missing_lgb = [c for c in lgbm.feature_name_ if c not in X.columns]
        assert not missing_lgb, f"Smoke test: LGBM thiếu feature {missing_lgb[:8]}"
        _ = lgbm.predict(X)

        missing_cat = [c for c in cat.feature_names_ if c not in sample.columns]
        assert not missing_cat, f"Smoke test: CatBoost thiếu feature {missing_cat[:8]}"
        _ = _predict_catboost(cat, sample, raise_on_missing=True)
        print(f"    ✅ Smoke test (suffix='{s or None}'): load + feature contract + predict OK")
    except Exception as e:
        print(f"    ⚠️ Smoke test FAILED — KHÔNG deploy trước khi fix: {e}")


# ====================== 7. LUỒNG CHÍNH ======================
def train_ensemble(force_rebuild: bool = False):
    global FEATURE_STATE
    t_start = time.perf_counter()

    df, backtest_hist = prepare_dataset(force_rebuild=force_rebuild)
    FEATURE_STATE = build_feature_lists(df)

    # [A] FAIL LOUD
    te_missing = [c for c in CLUSTER_OUTPUT_COLS if c not in FEATURE_STATE["num_zero_cols"]]
    assert not te_missing, f"Cluster TE vẫn bị drop: {te_missing}"
    assert "is_tier1_cluster" in FEATURE_STATE["num_zero_cols"], "is_tier1_cluster bị drop!"

    # [D] in cột legacy bị lọc
    if MIN_LAG > 1 and FEATURE_STATE["legacy_dropped_for_min_lag"]:
        print(f">>> [D] Loại khỏi model (lag thật < MIN_LAG={MIN_LAG}): "
              f"{FEATURE_STATE['legacy_dropped_for_min_lag']} — thay bằng "
              f"slog_lag/slog_roll_mean (features_ext, đã floor đúng MIN_LAG)")

    # [D-guard] audit leak
    if RUN_LEAK_AUDIT:
        anchor = df["date"].max() - pd.Timedelta(days=VAL_DAYS)
        print(f">>> [D-guard] Audit leak trên val window (sau {anchor.date()})...")
        leaked = audit_future_leakage(backtest_hist, anchor,
                                      FEATURE_STATE["lgbm_input_cols"])
        if leaked and MIN_LAG >= VAL_DAYS:
            raise RuntimeError(
                f"[D-guard] DIRECT model vẫn leak target-val: {leaked} — "
                "kiểm tra LEGACY_LAG_DAYS / feature mới chưa floor đúng MIN_LAG."
            )
        elif leaked:
            print(f"    [D-guard] cột phụ thuộc target-val (HỢP LỆ cho re-forecast "
                  f"hằng ngày — recursive feedback lo phần này): {[c for c, _ in leaked]}")
        else:
            print("    [D-guard] ✅ không cột nào phụ thuộc target của val window")

    print(f">>> Feature spec {FEATURE_SPEC_VERSION}: "
          f"{len(FEATURE_STATE['lgbm_input_cols'])} cột vào LGBM "
          f"(zero={len(FEATURE_STATE['num_zero_cols'])}, "
          f"sentinel={len(FEATURE_STATE['num_sentinel_cols'])}, "
          f"cat={len(FEATURE_STATE['cat_cols'])}, "
          f"pass={len(FEATURE_STATE['passthrough_cols'])}), "
          f"{len(FEATURE_STATE['catboost_cols'])} cột vào CatBoost")
    print(f">>> Chế độ: MIN_LAG={MIN_LAG} → "
          f"{'DIRECT one-shot (không đệ quy)' if MIN_LAG >= 16 else 'RECURSIVE (re-forecast hằng ngày)'}"
          f" | ES='{ES_MODE}' | artifact suffix='{ARTIFACT_SUFFIX or None}'\n")

    splits = walk_forward_splits(df, n_splits=N_SPLITS, val_days=VAL_DAYS,
                                 min_train_days=MIN_TRAIN_DAYS)
    print(f">>> Số fold thực tế: {len(splits)}\n")

    # [H] Dò hyperparameter trên fold mới nhất TRƯỚC khi chạy vòng fold chính
    hp_info = None
    if RUN_HYPERPARAM_SEARCH:
        try:
            _accepted, hp_info = tune_lgbm_on_fold(df, splits)
        except Exception as e:
            print(f">>> [H] Hyperparam search bỏ qua ({type(e).__name__}: {e})")
            hp_info = {"enabled": True, "error": f"{type(e).__name__}: {e}", "accepted": False}

    fold_summaries, oof = [], []
    for train_mask, val_mask, fold_info in splits:
        fold_num = fold_info["fold"]
        print(f"===== Fold {fold_num}/{len(splits)} | Val: {fold_info['val_start']} → {fold_info['val_end']} =====")
        t_fold = time.perf_counter()

        train_df = df[train_mask]
        val_df = df[val_mask]
        if EVAL_ON_OBSERVED_ONLY and "observed" in val_df.columns:
            val_df = val_df[val_df["observed"]]
        print(f"    Train: {len(train_df):,} | Val: {len(val_df):,}")

        record = _train_fold(train_df, val_df, backtest_hist=backtest_hist)
        oof.append(record.pop("_oof"))
        record = {"fold": fold_num, "val_start": fold_info["val_start"],
                  "val_end": fold_info["val_end"], "train_days": fold_info["train_days"],
                  **record, "duration_sec": round(time.perf_counter() - t_fold, 1)}
        fold_summaries.append(record)

        print(f"    LGBM     RMSLE : {record['lgbm_rmsle']:.5f} | iter {record['lgbm_best_iter']} | feats {record['lgbm_n_features']}")
        print(f"    CatBoost RMSLE : {record['catboost_rmsle']:.5f} | iter {record['catboost_best_iter']} | feats {record['catboost_n_features']}")
        print(f"    Blend    RMSLE : {record['blend_rmsle']:.5f} | w_LGBM {record['blend_weight_lgbm']:.3f} | gain {record['blend_gain_pct']:.2f}%")
        if record["blend_nwrmsle"]:
            print(f"    Blend  NWRMSLE : {record['blend_nwrmsle']:.5f}  (chuẩn chấm của Kaggle)")
        print(f"    WAPE {record['blend_wape']:.2f}% | MAE {record['blend_mae']:.1f} | tỉ lệ ngày 0 trong val: {record['zero_rate_true']:.1%}")
        if record["recursive_rmsle"] is not None:
            degr = record["recursive_rmsle"] / record["blend_rmsle"] - 1
            print(f"    ► RECURSIVE one-shot : {record['recursive_rmsle']:.5f} ({degr:+.1%})")
            print(f"      Theo tuần           : {record['recursive_weekly']}")
        if record["direct_weekly"]:
            print(f"    ► DIRECT theo horizon : {record['direct_weekly']}")
        print(f"    Fold duration : {record['duration_sec']}s\n")

    # ---------------- TỔNG KẾT ----------------
    avg_lgb_iter = max(1, int(np.mean([f["lgbm_best_iter"] for f in fold_summaries])))
    avg_cat_iter = max(1, int(np.mean([f["catboost_best_iter"] for f in fold_summaries])))
    med_lgb_iter = max(1, int(np.median([f["lgbm_best_iter"] for f in fold_summaries])))
    med_cat_iter = max(1, int(np.median([f["catboost_best_iter"] for f in fold_summaries])))
    avg_gain = float(np.mean([f["blend_gain_pct"] for f in fold_summaries]))
    chosen_w, weight_candidates = select_blend_weight(fold_summaries, oof)

    # [E1] honest LOO + [E4] OOF deploy
    loo_rows, loo_mean = loo_honest_blend(oof)
    oof_dep = oof_deploy_metrics(chosen_w, oof)

    # [M] Bộ 6 metrics pooled trên toàn OOF — đánh giá đầy đủ hơn RMSLE đơn lẻ
    y_pool = np.concatenate([o["y"] for o in oof])
    lgb_pool = np.concatenate([o["lgb"] for o in oof])
    cat_pool = np.concatenate([o["cat"] for o in oof])
    has_peri = oof and oof[0].get("peri") is not None
    peri_pool = np.concatenate([o["peri"] for o in oof]) if has_peri else None
    blend_pool_log = chosen_w * lgb_pool + (1 - chosen_w) * cat_pool

    def _slim(m: dict) -> dict:
        keep = ("rmsle", "nwrmsle", "mae", "rmse", "wape", "wmape", "r2")
        return {k: (round(v, 5) if isinstance(v, float) else v)
                for k, v in m.items() if k in keep and v is not None}

    validation_metrics = {
        "pooled_oof": {
            "lgbm": _slim(evaluate(y_pool, np.clip(np.expm1(lgb_pool), 0, None), peri_pool)),
            "catboost": _slim(evaluate(y_pool, np.clip(np.expm1(cat_pool), 0, None), peri_pool)),
            "blend": _slim(evaluate(y_pool, np.clip(np.expm1(blend_pool_log), 0, None), peri_pool)),
        },
        "per_fold": [
            {"fold": f["fold"], "val_start": str(f["val_start"]), "val_end": str(f["val_end"]),
             "lgbm": {"rmsle": f["lgbm_rmsle"], "mae": f["lgbm_mae"], "rmse": f["lgbm_rmse"],
                      "wape": f["lgbm_wape"], "wmape": f["lgbm_wmape"], "r2": f["lgbm_r2"]},
             "catboost": {"rmsle": f["catboost_rmsle"], "mae": f["catboost_mae"], "rmse": f["catboost_rmse"],
                          "wape": f["catboost_wape"], "wmape": f["catboost_wmape"], "r2": f["catboost_r2"]},
             "blend": {"rmsle": f["blend_rmsle"], "mae": f["blend_mae"], "rmse": f["blend_rmse"],
                       "wape": f["blend_wape"], "wmape": f["blend_wmape"], "r2": f["blend_r2"]}}
            for f in fold_summaries
        ],
        "definitions": {
            "rmsle": "sqrt(mean((log1p(ŷ)-log1p(y))^2))",
            "mae": "mean(|y-ŷ|)",
            "rmse": "sqrt(mean((y-ŷ)^2))",
            "wape": "Σ|y-ŷ|/Σ|y| ×100 (%)",
            "wmape": "Σw|y-ŷ|/Σw|y| ×100, w=1.5 nếu perishable (%)",
            "r2": "1 - SS_res/SS_tot",
            "note": "pooled OOF = ghép dự báo out-of-fold cả 3 fold, blend ở trọng số deploy",
        },
    }

    rec_folds = [f for f in fold_summaries if f["recursive_rmsle"] is not None]
    avg_val = float(np.mean([f["blend_rmsle"] for f in fold_summaries]))
    nw = [f["blend_nwrmsle"] for f in fold_summaries if f["blend_nwrmsle"]]
    avg_nw = float(np.mean(nw)) if nw else None
    std_val = float(np.std([f["blend_rmsle"] for f in fold_summaries]))

    if rec_folds:
        avg_rec = float(np.mean([f["recursive_rmsle"] for f in rec_folds]))
        avg_val_rec = float(np.mean([f["blend_rmsle"] for f in rec_folds]))
        degradation = (avg_rec / avg_val_rec - 1) * 100
    else:
        avg_rec = avg_val_rec = degradation = None

    def _avg_weekly(key):
        folds_w = [f[key] for f in fold_summaries if f.get(key)]
        if not folds_w:
            return {}
        keys = sorted({k for d in folds_w for k in d})
        return {k: round(float(np.mean([d[k] for d in folds_w if k in d])), 5)
                for k in keys}

    avg_weekly = _avg_weekly("recursive_weekly")
    avg_direct_weekly = _avg_weekly("direct_weekly")

    print("\n" + "=" * 70)
    print(f"ENSEMBLE WALK-FORWARD SUMMARY  (MIN_LAG={MIN_LAG}, ES='{ES_MODE}', "
          f"suffix='{ARTIFACT_SUFFIX or None}')")
    print("=" * 70)
    for f in fold_summaries:
        rec_s = f" | Recursive={f['recursive_rmsle']:.5f}" if f["recursive_rmsle"] else ""
        print(f"  Fold {f['fold']} (val → {f['val_end']}): LGBM={f['lgbm_rmsle']:.5f} | "
              f"Cat={f['catboost_rmsle']:.5f} | Blend={f['blend_rmsle']:.5f}{rec_s} | {f['duration_sec']}s")
    print("-" * 70)
    print(f"  Iterations LGBM/CatBoost : {avg_lgb_iter}/{avg_cat_iter} (median {med_lgb_iter}/{med_cat_iter})")
    print(f"  Blend weight ({BLEND_WEIGHT_STRATEGY})       : {chosen_w:.3f}")
    print("  Ứng viên weight          : " + ", ".join(f"{k}={v:.3f}" for k, v in weight_candidates.items()))
    print(f"  Gain vs model đơn tốt nhất: {avg_gain:.2f}%"
          + ("  → blend gần như vô ích, bật USE_TWEEDIE_MEMBER" if avg_gain < 0.5 else ""))
    print("-" * 70)
    # ----- [E1] SO SÁNH SỐ LẠC QUAN vs TRUNG THỰC (dùng cho Bảng 4.x) -----
    print("  METRIC REPORTING (đọc kỹ trước khi ghi báo cáo):")
    print(f"  • Blend RMSLE (fold-tuned) : {avg_val:.4f} ± {std_val:.4f}  "
          f"← hơi lạc quan (ES + w chọn trên chính val)")
    if loo_mean:
        gap = (avg_val / loo_mean["rmsle"] - 1) * 100
        print(f"  • Blend RMSLE honest (LOO) : {loo_mean['rmsle']:.4f}  "
              f"← SỐ DÙNG CHO BÁO CÁO (w từ fold khác)")
        print(f"      Độ lạc quan đã lượng hóa: {gap:+.2f}%  | chi tiết: "
              + ", ".join(f"f{r['fold']}: w={r['w_loo']:.3f}, rmsle={r['rmsle_honest']:.5f}"
                          for r in loo_rows))
    print(f"  • Deploy OOF (w={chosen_w:.3f})  : RMSLE {oof_dep['rmsle']:.4f}"
          + (f" | NWRMSLE {oof_dep['nwrmsle']:.4f}" if oof_dep.get("nwrmsle") else "")
          + "  ← gần nhất hành vi submission")
    print("-" * 70)
    print("PRODUCTION READINESS")
    if MIN_LAG >= VAL_DAYS:
        print(f"  • DIRECT one-shot {VAL_DAYS} ngày : RMSLE ≈ {avg_val:.4f} ± {std_val:.4f} "
              f"(n={len(fold_summaries)} folds — [D-guard] pass, trung thực)")
        if avg_direct_weekly:
            print("  • Theo horizon (val)    : " + ", ".join(f"{k}={v:.4f}" for k, v in avg_direct_weekly.items()))
            print("  → Fold gần nhất (val → 2017-08-15) là ước lượng tốt nhất cho submission Kaggle.")
    else:
        print(f"  • Re-forecast HẰNG NGÀY : RMSLE ≈ {avg_val:.4f} ± {std_val:.4f} (n={len(fold_summaries)} folds)")
        if avg_rec is not None:
            print(f"  • One-shot {VAL_DAYS} ngày      : RMSLE ≈ {avg_rec:.4f} ({degradation:+.1f}%, n={len(rec_folds)})")
        if avg_weekly:
            print("  • Theo horizon          : " + ", ".join(f"{k}={v:.4f}" for k, v in avg_weekly.items()))
            print("  → Tuần 3-4 xuống nhiều thì dùng model DIRECT (MIN_LAG=16) cho submission.")
    if avg_nw:
        print(f"  • NWRMSLE (chuẩn Kaggle): {avg_nw:.4f}")
    print("=" * 70)

    # ---------------- FINAL MODELS ----------------
    print("\n>>> Fitting Cluster Features on FULL data...")
    cluster_final = ClusterFeatureEngineer(smoothing=CLUSTER_SMOOTHING)
    df_full = cluster_final.fit_transform(df, target_col="target")
    y_full_log = np.log1p(df_full["target"].values)
    w_full = sample_weights(df_full, df_full["date"].max())

    n_lgb = max(1, int(avg_lgb_iter * FINAL_ITER_SCALE))
    n_cat = max(1, int(avg_cat_iter * FINAL_ITER_SCALE))

    print(f">>> FINAL LightGBM on {len(df_full):,} rows (n_estimators={n_lgb})...")
    final_prep = build_preprocessor_v2(
        FEATURE_STATE["num_zero_cols"], FEATURE_STATE["num_sentinel_cols"],
        FEATURE_STATE["cat_cols"], FEATURE_STATE["passthrough_cols"]
    )
    X_full = _fix_unknown_categorical(final_prep.fit_transform(df_full))
    lgbm_cat_full = [c for c in CAT_FEATURES if c in X_full.columns]
    final_lgbm = LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": n_lgb})
    final_lgbm.fit(X_full, y_full_log, sample_weight=w_full, categorical_feature=lgbm_cat_full)

    print(f">>> FINAL CatBoost on {len(df_full):,} rows (iterations={n_cat})...")
    cat_cols_full = [c for c in CAT_FEATURES if c in df_full.columns]
    feats_full = [c for c in FEATURE_STATE["catboost_cols"] if c in df_full.columns]
    full_pool = Pool(data=df_full[feats_full], label=y_full_log,
                     cat_features=cat_cols_full, weight=w_full)
    final_cat = CatBoostRegressor(**{**CATBOOST_PARAMS, "iterations": n_cat,
                                     "early_stopping_rounds": None})
    final_cat.fit(full_pool, verbose=False)

    ensemble_meta = {
        # ---- khoá ml_service đang đọc ----
        "lgbm_weight": round(chosen_w, 4),
        "catboost_weight": round(1.0 - chosen_w, 4),
        "avg_blend_rmsle": round(avg_val, 5),
        "avg_lgbm_iteration": n_lgb,
        "avg_catboost_iteration": n_cat,
        # ---- hợp đồng feature ----
        "feature_spec_version": FEATURE_SPEC_VERSION,
        "requires_features_ext": True,
        "min_lag": MIN_LAG,
        "serving_mode": "direct_oneshot" if MIN_LAG >= 16 else "recursive",
        "artifact_suffix": ARTIFACT_SUFFIX,
        "legacy_lag_filter": {
            "min_lag": MIN_LAG,
            "legacy_lag_days": LEGACY_LAG_DAYS,
            "dropped": FEATURE_STATE["legacy_dropped_for_min_lag"],
        },
        "use_cross_series": USE_CROSS_SERIES,
        "complete_panel": COMPLETE_PANEL,
        "serving_recipe": "df = add_extra_features(engineer_features(window), min_lag); "
                          "lịch sử nạp DB phải đi qua complete_panel(); "
                          "ffill transactions_lag1 per store cho ngày tương lai (RUN 1)",
        # ---- blend ----
        "blend_space": "log",
        "blend_formula": "pred = clip(expm1(w*lgbm_raw + (1-w)*catboost_raw), 0, None)",
        "blend_weight_strategy": BLEND_WEIGHT_STRATEGY,
        "blend_weight_candidates": {k: round(v, 4) for k, v in weight_candidates.items()},
        "ensemble_gain_pct": round(avg_gain, 3),
        "cluster_smoothing": CLUSTER_SMOOTHING,
        "cluster_features_in_model": FEATURE_STATE["num_zero_cols"],
        "catboost_loss": CATBOOST_LOSS,
        # ---- [M] bộ 6 metrics đánh giá đầy đủ ----
        "validation_metrics": validation_metrics,
        # ---- [H] kết quả dò hyperparameter ----
        "hyperparam_search": hp_info,
        # ---- [E1]/[E4] reporting trung thực ----
        "reporting": {
            "es_mode": ES_MODE,
            "blend_rmsle_fold_tuned": round(avg_val, 5),
            "blend_rmsle_fold_tuned_std": round(std_val, 5),
            "blend_rmsle_honest_loo": loo_mean["rmsle"] if loo_mean else None,
            "blend_nwrmsle_honest_loo": loo_mean["nwrmsle"] if loo_mean else None,
            "loo_detail": loo_rows,
            "optimism_gap_pct": round((avg_val / loo_mean["rmsle"] - 1) * 100, 3)
                                if loo_mean else None,
            "oof_deploy_rmsle": round(oof_dep["rmsle"], 5),
            "oof_deploy_nwrmsle": round(oof_dep["nwrmsle"], 5) if oof_dep.get("nwrmsle") else None,
            "note": "fold-tuned hơi lạc quan (ES+w trên val); honest=LOO (w từ fold khác); "
                    "deploy OOF = toàn OOF với w deploy",
        },
        "training_options": {
            "n_seeds": N_SEEDS,
            "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
            "perishable_weight": USE_PERISHABLE_WEIGHT,
            "tweedie_member": USE_TWEEDIE_MEMBER,
            "es_mode": ES_MODE,
            "warmup_days": max(WARMUP_DAYS, MIN_LAG + 28),
            "eval_on_observed_only": EVAL_ON_OBSERVED_ONLY,
            "leak_audit_passed": True,
        },
        "product_quality": {
            "val_rmsle": round(avg_val, 5),
            "val_rmsle_std": round(std_val, 5),
            "val_nwrmsle": round(avg_nw, 5) if avg_nw else None,
            "val_n_folds": len(fold_summaries),
            "val_is_one_shot_estimate": bool(MIN_LAG >= VAL_DAYS),
            "one_shot_rmsle": round(avg_rec, 5) if avg_rec else None,
            "one_shot_horizon_days": VAL_DAYS,
            "one_shot_n_folds": len(rec_folds),
            "one_shot_blend_rmsle_same_folds": round(avg_val_rec, 5) if avg_val_rec else None,
            "horizon_weekly_rmsle": avg_weekly,
            "direct_val_weekly_rmsle": avg_direct_weekly,
            "degradation_pct": round(degradation, 1) if degradation is not None else None,
            "measurement": "recursive backtest mirror predict.py (window (d-90, d+2], "
                           "log-space blend, ffill transactions_lag1)",
        },
        "iterations": {"lgbm_mean": avg_lgb_iter, "lgbm_median": med_lgb_iter,
                       "catboost_mean": avg_cat_iter, "catboost_median": med_cat_iter,
                       "final_scale": FINAL_ITER_SCALE},
        "lgbm_feature_names": list(final_prep.get_feature_names_out()),
        "lgbm_categorical_features": lgbm_cat_full,
        "catboost_features": feats_full,
        "catboost_cat_features": cat_cols_full,
        "walk_forward": {"n_folds": len(splits), "val_days": VAL_DAYS},
        "backtest_config": {"window_days": BACKTEST_WINDOW_DAYS,
                            "lead_buffer_days": BACKTEST_LEAD_BUFFER_DAYS,
                            "ffill_cols": FFILL_COLS},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "total_duration_sec": round(time.perf_counter() - t_start, 1),
        "folds": fold_summaries,
    }

    print(f"\n>>> Exporting artifacts to {len(MODEL_DIRS)} location(s)...")
    save_artifacts(MODEL_DIRS, final_lgbm, final_cat, final_prep, cluster_final, ensemble_meta)

    print("\n>>> Smoke test artifacts...")
    _smoke_test_artifacts(LOCAL_MODELS_DIR, df_full)

    print("\n>>> Done.")
    return final_lgbm, final_cat, chosen_w


if __name__ == "__main__":
    train_ensemble(force_rebuild=False)
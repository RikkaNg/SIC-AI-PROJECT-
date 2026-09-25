"""
src/evaluate_metrics.py
Tính bộ chỉ số đánh giá đầy đủ cho Ensemble (LGBM + CatBoost + Blend):
MAE, RMSE, WAPE, R², WMAPE (+ RMSLE để đối chiếu train.py).

- Import cấu hình từ train.py → luôn đồng bộ với pipeline huấn luyện hiện tại.
- Walk-forward validation giống train.py, thu prediction từng fold,
  tính chỉ số PER-FOLD + POOLED (gộp toàn bộ fold — bộ số liệu cho báo cáo).
- Tự mirror train.py: blend log/original-space, LGBM categorical native,
  eval_X/eval_y (LightGBM mới, fix deprecation eval_set).
- Lưu prediction ra CSV làm cache → chạy lại chỉ số KHÔNG cần train lại.

Cách chạy:
    python src/evaluate_metrics.py               # lần 1: train + tính (~30 phút)
    python src/evaluate_metrics.py               # lần sau: dùng cache, tức thì
    python src/evaluate_metrics.py --force       # bỏ cache, train lại
    python src/evaluate_metrics.py --n-splits 1  # chạy nhanh 1 fold

Kết quả: bảng console + ml_training/models/evaluation_metrics.json
         + ml_training/models/eval_predictions.csv (cache)
"""

import sys
import json
import argparse
import inspect
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_log_error

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import train  # chỉ chạy module-level mkdir (an toàn), không tự train
from cluster_features import ClusterFeatureEngineer
from preprocessor import build_preprocessor

# ============ 0. ĐỒNG BỘ CẤU HÌNH TỪ train.py ============
LGBM_PARAMS = train.LGBM_PARAMS
CATBOOST_PARAMS = train.CATBOOST_PARAMS
CAT_FEATURES = train.CAT_FEATURES
CAT_ALL_FEATURES = train.CAT_ALL_FEATURES
CLUSTER_SMOOTHING = getattr(train, "CLUSTER_SMOOTHING", 10.0)
LOCAL_MODELS_DIR = train.LOCAL_MODELS_DIR

# Phát hiện phiên bản train.py đang chạy để mirror đúng hành vi:
_blend_params = set(inspect.signature(train.find_best_weight).parameters)
BLEND_IS_LOG = "lgbm_log" in _blend_params           # blend log-space hay original-space
try:
    USE_CATEGORICAL = "categorical_feature" in inspect.getsource(train)
except (OSError, TypeError):
    USE_CATEGORICAL = False
USE_EVAL_XY = "eval_X" in inspect.signature(LGBMRegressor.fit).parameters  # LightGBM >= 4.6

PRED_CACHE = LOCAL_MODELS_DIR / "eval_predictions.csv"
METRICS_JSON = LOCAL_MODELS_DIR / "evaluation_metrics.json"
MODELS = ["lgbm", "catboost", "blend"]


# ============ 1. CÁC CHỈ SỐ (thuần numpy — import được ở nơi khác) ============
def mae(y_true, y_pred) -> float:
    """MAE = mean(|y − ŷ|) — sai số tuyệt đối trung bình (đơn vị sales)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """RMSE = sqrt(mean((y − ŷ)²)) — phạt mạnh lỗi lớn."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true, y_pred) -> float:
    """WAPE = Σ|y − ŷ| / Σ|y| — sai số theo quy mô doanh số thật (bền với y=0)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denom = float(np.abs(y_true).sum())
    return float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else float("nan")


def r2_score_manual(y_true, y_pred) -> float:
    """R² = 1 − SS_res / SS_tot."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def wmape(y_true, y_pred, weights=None) -> float:
    """
    WMAPE = Σ(w·|y − ŷ|) / Σ(w·|y|) — WAPE tổng quát có trọng số.
    - weights=None → w=1 → WMAPE ≡ WAPE (2 tên gọi phổ biến của cùng công thức).
    - Script mặc định truyền w = 1.5 nếu perishable, ngược lại 1.0
      (trọng số của metric gốc cuộc thi Favorita).
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    w = np.ones_like(y_true) if weights is None else np.asarray(weights, float)
    denom = float(np.sum(w * np.abs(y_true)))
    return float(np.sum(w * np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")


def rmsle(y_true, y_pred) -> float:
    """RMSLE — giữ để đối chiếu với train.py."""
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)
    y_true = np.maximum(np.asarray(y_true, float), 0)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


def compute_all_metrics(y_true, y_pred, weights=None) -> dict:
    """Tính đủ 6 chỉ số cho 1 cặp (y_true, y_pred)."""
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "r2": r2_score_manual(y_true, y_pred),
        "wmape": wmape(y_true, y_pred, weights),
        "rmsle": rmsle(y_true, y_pred),
    }


DEFINITIONS = {
    "MAE": "mean(|y - yhat|)",
    "RMSE": "sqrt(mean((y - yhat)^2))",
    "WAPE": "sum(|y - yhat|) / sum(|y|)",
    "R2": "1 - sum((y-yhat)^2) / sum((y - mean(y))^2)",
    "WMAPE": "sum(w*|y-yhat|) / sum(w*|y|); w=1.5 if perishable else 1.0; w=1 => WAPE",
    "RMSLE": "sqrt(mean((log1p(y) - log1p(yhat))^2))",
}


# ============ 2. HUẤN LUYỆN TỪNG FOLD (mirror train.py) ============
def _fix_unknown_categorical(X: pd.DataFrame) -> pd.DataFrame:
    """LightGBM không chấp nhận categorical âm; mã unknown -1 → sentinel 9999."""
    cat_cols = [c for c in CAT_FEATURES if c in X.columns]
    if not cat_cols or not bool((X[cat_cols] < 0).any().any()):
        return X
    for col in cat_cols:
        neg = X[col] < 0
        if neg.any():
            X.loc[neg, col] = 9999
    return X


def fold_predictions(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Train 1 fold (đúng logic train.py) và trả về predictions thô."""
    cluster_engineer = ClusterFeatureEngineer(smoothing=CLUSTER_SMOOTHING)
    train_df = cluster_engineer.fit_transform(train_df, target_col="target")
    val_df = cluster_engineer.transform(val_df)

    y_train_log = np.log1p(train_df["target"].values)
    y_val_log = np.log1p(val_df["target"].values)
    y_val_true = val_df["target"].values
    perishable_w = (
        np.where(val_df["perishable"].values == 1, 1.5, 1.0)
        if "perishable" in val_df.columns else None
    )

    # ---- LGBM ----
    fold_preprocessor = build_preprocessor()
    X_train_lgb = _fix_unknown_categorical(fold_preprocessor.fit_transform(train_df))
    X_val_lgb = _fix_unknown_categorical(fold_preprocessor.transform(val_df))

    fit_kwargs = {}
    if USE_EVAL_XY:                                  # fix deprecation eval_set
        fit_kwargs["eval_X"], fit_kwargs["eval_y"] = X_val_lgb, y_val_log
    else:
        fit_kwargs["eval_set"] = [(X_val_lgb, y_val_log)]
    if USE_CATEGORICAL:                              # mirror train.py
        fit_kwargs["categorical_feature"] = [c for c in CAT_FEATURES if c in X_train_lgb.columns]

    lgbm = LGBMRegressor(**LGBM_PARAMS)
    lgbm.fit(
        X_train_lgb, y_train_log,
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=0)],
        **fit_kwargs,
    )
    lgbm_log = lgbm.predict(X_val_lgb)
    lgbm_pred = np.clip(np.expm1(lgbm_log), 0, None)

    # ---- CatBoost ----
    cat_cols = [c for c in CAT_FEATURES if c in train_df.columns]
    all_features = [c for c in CAT_ALL_FEATURES if c in train_df.columns]
    train_pool = Pool(train_df[all_features], label=y_train_log, cat_features=cat_cols)
    val_pool = Pool(val_df[all_features], label=y_val_log, cat_features=cat_cols)

    cat = CatBoostRegressor(**CATBOOST_PARAMS)
    cat.fit(train_pool, eval_set=val_pool, verbose=False)
    cat_log = cat.predict(val_df[all_features])
    cat_pred = np.clip(np.expm1(cat_log), 0, None)

    # ---- Blend: mirror ĐÚNG không gian blend của train.py ----
    if BLEND_IS_LOG:
        best_w, _ = train.find_best_weight(y_val_true, lgbm_log, cat_log)
        blend_pred = np.clip(np.expm1(best_w * lgbm_log + (1 - best_w) * cat_log), 0, None)
    else:
        best_w, _ = train.find_best_weight(y_val_true, lgbm_pred, cat_pred)
        blend_pred = np.clip(best_w * lgbm_pred + (1 - best_w) * cat_pred, 0, None)

    return y_val_true, lgbm_pred, cat_pred, blend_pred, float(best_w), perishable_w


# ============ 3. LUỒNG ĐÁNH GIÁ ============
def _collect_predictions(n_splits: int) -> pd.DataFrame:
    df = train.prepare_dataset(force_rebuild=False)
    splits = train.walk_forward_splits(
        df, n_splits=n_splits, val_days=train.VAL_DAYS, min_train_days=train.MIN_TRAIN_DAYS
    )
    print(f">>> Folds: {len(splits)} | blend={'log-space' if BLEND_IS_LOG else 'original-space'} "
          f"| LGBM categorical={'on' if USE_CATEGORICAL else 'off'} "
          f"| smoothing={CLUSTER_SMOOTHING}\n")

    frames = []
    for train_mask, val_mask, fold_info in splits:
        print(f"===== Fold {fold_info['fold']}/{len(splits)} | "
              f"Val {fold_info['val_start']} → {fold_info['val_end']} =====")
        train_df, val_df = df[train_mask], df[val_mask]
        print(f"    Train: {len(train_df):,} | Val: {len(val_df):,} — đang huấn luyện...")
        y, lgb, cat, blend, w, pw = fold_predictions(train_df, val_df)
        frames.append(pd.DataFrame({
            "fold": fold_info["fold"],
            "val_start": str(fold_info["val_start"]),
            "val_end": str(fold_info["val_end"]),
            "y_true": y,
            "lgbm_pred": lgb,
            "catboost_pred": cat,
            "blend_pred": blend,
            "blend_weight": w,
            # 1 = perishable (w=1.5), 0 = thường (w=1.0); 0 toàn bộ nếu không có cột
            "perishable": (pw == 1.5).astype(int) if pw is not None else 0,
        }))
        print(f"    Xong (w_LGBM={w:.3f})\n")

    cache = pd.concat(frames, ignore_index=True)
    cache.to_csv(PRED_CACHE, index=False)
    print(f">>> Đã lưu cache prediction → {PRED_CACHE}")
    return cache


def _metrics_for(cache: pd.DataFrame, pred_col: str, fold=None) -> dict:
    sub = cache if fold is None else cache[cache["fold"] == fold]
    weights = np.where(sub["perishable"].values == 1, 1.5, 1.0)
    return compute_all_metrics(sub["y_true"].values, sub[pred_col].values, weights)


HEADER = (f"  {'Model':<9}| {'MAE':>9} | {'RMSE':>9} | {'WAPE':>8} | "
          f"{'R2':>8} | {'WMAPE*':>8} | {'RMSLE':>8}")
SEP = "  " + "-" * (len(HEADER) - 2)


def _row(name: str, m: dict) -> str:
    return (f"  {name:<9}| {m['mae']:>9.2f} | {m['rmse']:>9.2f} | {m['wape']*100:>7.2f}% | "
            f"{m['r2']:>8.4f} | {m['wmape']*100:>7.2f}% | {m['rmsle']:>8.5f}")


def build_report(cache: pd.DataFrame, config: dict) -> dict:
    per_fold = []
    for fold in sorted(cache["fold"].unique()):
        sub = cache[cache["fold"] == fold]
        per_fold.append({
            "fold": int(fold),
            "val_start": str(sub["val_start"].iloc[0]),
            "val_end": str(sub["val_end"].iloc[0]),
            "blend_weight_lgbm": float(sub["blend_weight"].iloc[0]),
            "models": {
                "lgbm": _metrics_for(cache, "lgbm_pred", fold),
                "catboost": _metrics_for(cache, "catboost_pred", fold),
                "blend": _metrics_for(cache, "blend_pred", fold),
            },
        })

    pooled = {m: _metrics_for(cache, f"{m}_pred") for m in MODELS}
    avg = {
        m: {k: float(np.mean([f["models"][m][k] for f in per_fold])) for k in pooled[m]}
        for m in MODELS
    }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "definitions": DEFINITIONS,
        "per_fold": per_fold,
        "pooled": pooled,          # <-- bộ số liệu dùng cho báo cáo
        "average_per_fold": avg,
    }


def main():
    parser = argparse.ArgumentParser(description="Tính MAE/RMSE/WAPE/R2/WMAPE cho ensemble")
    parser.add_argument("--force", action="store_true", help="Bỏ cache, train lại từ đầu")
    parser.add_argument("--n-splits", type=int, default=None,
                        help=f"Số fold (mặc định train.N_SPLITS={train.N_SPLITS})")
    args = parser.parse_args()

    if PRED_CACHE.exists() and not args.force:
        print(f">>> Dùng cache prediction: {PRED_CACHE} (dùng --force để chạy lại)\n")
        cache = pd.read_csv(PRED_CACHE)
    else:
        cache = _collect_predictions(args.n_splits or train.N_SPLITS)

    config = {
        "n_folds": int(cache["fold"].nunique()),
        "val_days": train.VAL_DAYS,
        "blend_space": "log" if BLEND_IS_LOG else "original",
        "lgbm_categorical_native": bool(USE_CATEGORICAL),
        "cluster_smoothing": CLUSTER_SMOOTHING,
        "wmape_weights": "perishable x1.5 (metric goc Favorita)",
    }
    report = build_report(cache, config)

    for f in report["per_fold"]:
        print(f"\nFold {f['fold']} | val {f['val_start']} → {f['val_end']} | w_LGBM={f['blend_weight_lgbm']:.3f}")
        print(HEADER); print(SEP)
        print(_row("LGBM", f["models"]["lgbm"]))
        print(_row("CatBoost", f["models"]["catboost"]))
        print(_row("Blend", f["models"]["blend"]))

    print("\n" + "=" * 80)
    print("POOLED (gộp toàn bộ fold) — bộ số liệu dùng cho báo cáo")
    print("=" * 80)
    print(HEADER); print(SEP)
    print(_row("LGBM", report["pooled"]["lgbm"]))
    print(_row("CatBoost", report["pooled"]["catboost"]))
    print(_row("Blend", report["pooled"]["blend"]))

    print("\nAverage per-fold:")
    print(HEADER); print(SEP)
    print(_row("LGBM", report["average_per_fold"]["lgbm"]))
    print(_row("CatBoost", report["average_per_fold"]["catboost"]))
    print(_row("Blend", report["average_per_fold"]["blend"]))

    print("\n(*) WMAPE dùng trọng số perishable x1.5 (metric gốc cuộc thi Favorita).")
    print("    WMAPE KHÔNG trọng số (w=1) ≡ WAPE — nếu template của bạn định nghĩa")
    print("    WMAPE = WAPE thì lấy số ở cột WAPE.")

    with open(METRICS_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str, ensure_ascii=False)
    print(f"\n>>> Đã lưu: {METRICS_JSON}")


if __name__ == "__main__":
    main()
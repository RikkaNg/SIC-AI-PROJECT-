"""
ml_service/app/inference.py (v3.0 — SYNC train.py v2 + features_ext.py)

SYNC CONTRACT (4 nơi PHẢI giữ cùng logic — sửa 1 nơi thì sửa cả 4):
  1) ml_training/src/train.py::recursive_backtest             (đo chất lượng)
  2) ml_training/src/predict.py::recursive_forecast_ensemble  (submission Kaggle)
  3) ml_service/app/inference.py::predict_recursive           (phục vụ) ← file này
  4) ml_training/src/features_ext.py                          (logic feature)
Verify: grep -n "NORMAL_DAY\\|LEAD_BUFFER_DAYS\\|ffill\\|add_extra_features" inference.py

THAY ĐỔI vs v2.8 — 3 lỗi đầu là lỗi CHẶN, service không chạy hoặc chạy sai:

[H1] engineer_features(..., fit_cluster=False) → TypeError.
     preprocessor.py bản mới đã BỎ tham số fit_cluster (chỉ còn cluster_engineer).
     Mọi request /forecast sẽ ném TypeError: unexpected keyword argument.

[H2] Thiếu _fix_unknown_categorical trước khi LGBM predict.
     train.py và predict.py đều remap -1 → 9999. Service thì không: gặp city/
     family lạ, OrdinalEncoder trả -1, LightGBM đọc categorical âm → rơi vào
     bucket sai, im lặng, không có exception.

[H3] Thiếu add_extra_features → model v2 nhận thiếu ~29 cột.
     LGBM: preprocessor.transform ném KeyError. CatBoost: reindex fill 0 → dự báo
     ra số rác mà KHÔNG báo lỗi. Đây là kiểu hỏng nguy hiểm nhất.

[H4] Feature cắt ngang cửa hàng (store_slog_mean_lag, family_slog_mean_lag,
     store_promo_share) KHÔNG tính được ở đây: history_df chỉ có MỘT cặp
     (store, family). Engine kiểm tra meta và từ chối khởi động nếu model được
     train với use_cross_series=True mà service chỉ nạp 1 chuỗi.

[H5] Kiểm tra feature_spec_version lúc __init__ — lệch version thì fail fast,
     thay vì trả về số dự báo sai cho khách hàng.

GIỮ NGUYÊN từ v2.8: NORMAL_DAY dùng chung, LEAD_BUFFER_DAYS=2, ffill transactions
theo store, target float64, sort trước khi lấy last_row, copy future_exog,
check w_lgbm + w_cat ≈ 1, Smart Routing local → global.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# ====================== CẤU HÌNH ĐƯỜNG DẪN ======================
CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent
ML_SERVICE_DIR = APP_DIR.parent
PROJECT_ROOT = CURRENT_FILE.parents[2]
for _parent in CURRENT_FILE.parents:
    if (_parent / "ml_training" / "src" / "preprocessor.py").exists():
        PROJECT_ROOT = _parent
        break
ML_TRAINING_DIR = PROJECT_ROOT / "ml_training"
ML_TRAINING_SRC = ML_TRAINING_DIR / "src"

for p in [str(APP_DIR), str(ML_TRAINING_SRC), str(ML_TRAINING_DIR),
          str(ML_SERVICE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.append(p)

try:
    from preprocessor import engineer_features, NORMAL_DAY
except ImportError:
    from preprocessor import engineer_features
    NORMAL_DAY = "Normal Day"

# [H3] logic feature dùng chung với train.py
from features_ext import FEATURE_SPEC_VERSION, add_extra_features

logger = logging.getLogger(__name__)

# Cửa sổ trượt: lag xa nhất 28 + rolling 30 (shift 1) + days_since_sale (trần 56).
RECURSION_WINDOW_DAYS = 60
LEAD_BUFFER_DAYS = 2          # cho is_holiday_lead1/lead2
# => Window hiệu dụng 62 ngày. History gửi vào nên ≥ 90 ngày.

LGBM_UNKNOWN_SENTINEL = 9999                                   # [H2] SYNC train.py
CAT_FEATURES = ["store_nbr", "family", "city", "state", "type", "holiday_type"]


def _fix_unknown_categorical(X: pd.DataFrame) -> pd.DataFrame:
    """[H2] Remap -1 của OrdinalEncoder → sentinel. Bản sao từ train.py."""
    cat_cols = [c for c in CAT_FEATURES if c in X.columns]
    if not cat_cols or not bool((X[cat_cols] < 0).any().any()):
        return X
    for col in cat_cols:
        neg = X[col] < 0
        if neg.any():
            X.loc[neg, col] = LGBM_UNKNOWN_SENTINEL
    return X


class RetailInferenceEngine:
    """Engine quản lý luồng dự báo chuỗi thời gian đệ quy và định tuyến mô hình."""

    def __init__(
        self,
        global_lgbm=None,
        global_prep=None,
        global_cat=None,
        local_models: Optional[Dict[str, Any]] = None,
        cluster_engineer=None,
        w_lgbm: float = 0.5,
        w_cat: float = 0.5,
        quality_filter: Optional[Any] = None,
        ensemble_meta: Optional[Dict[str, Any]] = None,
    ):
        self.global_lgbm = global_lgbm
        self.global_prep = global_prep
        self.global_cat = global_cat
        self.local_models = local_models or {}
        self.cluster_engineer = cluster_engineer
        self.w_lgbm = w_lgbm
        self.w_cat = w_cat
        self.quality_filter = quality_filter

        if abs(self.w_lgbm + self.w_cat - 1.0) > 1e-6:
            logger.warning(
                f"⚠️ w_lgbm + w_cat = {self.w_lgbm + self.w_cat:.4f} ≠ 1 — "
                "kiểm tra ensemble_meta.json / main.py!"
            )

        # ---- [H5] hợp đồng feature với model đang load ----
        meta = ensemble_meta or self._load_meta_fallback()
        self.meta = meta or {}
        self.min_lag = int(self.meta.get("min_lag", 1))
        self.use_cross_series = bool(self.meta.get("use_cross_series", False))
        self.requires_features_ext = bool(self.meta.get("requires_features_ext", False))

        spec = self.meta.get("feature_spec_version")
        if self.requires_features_ext and spec != FEATURE_SPEC_VERSION:
            raise RuntimeError(
                f"Model train bằng feature spec {spec}, service đang chạy "
                f"{FEATURE_SPEC_VERSION}. Deploy lại features_ext.py cho khớp."
            )
        # [H4] service chỉ nạp lịch sử 1 cặp (store, family)
        if self.use_cross_series:
            raise RuntimeError(
                "Model train với use_cross_series=True nhưng predict_recursive chỉ "
                "nhận lịch sử của MỘT cặp (store, family) → store_slog_mean_lag / "
                "family_slog_mean_lag / store_promo_share sẽ sai. Hoặc train lại với "
                "USE_CROSS_SERIES=False, hoặc sửa backend gửi lịch sử cả cửa hàng."
            )
        if not self.requires_features_ext:
            logger.warning(
                "ensemble_meta.json không có requires_features_ext — model cũ (v1). "
                "Service vẫn chạy nhưng nên train lại bằng train.py v2."
            )

        self.cat_feature_names = (
            getattr(self.global_cat, "feature_names_", []) if self.global_cat else []
        )

        required_cols: set = set()
        for prep in [self.global_prep] + [
            art.get("preprocessor") for art in self.local_models.values()
        ]:
            names = getattr(prep, "feature_names_in_", None)
            if names is not None:
                required_cols.update(str(c) for c in names)
        self._required_feature_cols = required_cols

    @staticmethod
    def _load_meta_fallback() -> Optional[dict]:
        for base in (ML_SERVICE_DIR / "models", ML_TRAINING_DIR / "models"):
            path = base / "ensemble_meta.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        return None

    # ------------------------------------------------------------------
    def predict_recursive(
        self,
        history_df: pd.DataFrame,
        future_dates: List[str],
        future_exog: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dự báo đệ quy từng ngày cho một cặp (store_nbr, family) — SYNC CONTRACT.

        history_df cần ≥ 90 ngày và các cột: date, store_nbr, family, target,
        transactions, holiday_type, oil_price, onpromotion, city/state/type/cluster.
        Lịch sử càng thủng ngày thì lag càng lệch: DB nên lưu đủ cả ngày bán 0
        (tầng nạp dữ liệu phải chạy features_ext.complete_panel()).
        """
        if history_df.empty:
            raise ValueError("history_df cannot be empty.")

        combined = history_df.copy()
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.sort_values(["store_nbr", "date"], ignore_index=True)
        if "target" in combined.columns:
            combined["target"] = combined["target"].astype("float64")

        # cảnh báo chuỗi thủng ngày → lag lệch (shift dịch theo DÒNG, không theo NGÀY)
        span = (combined["date"].max() - combined["date"].min()).days + 1
        n_pairs = combined.groupby(["store_nbr", "family"], observed=True).ngroups
        if n_pairs == 1 and len(combined) < span * 0.95:
            logger.warning(
                f"history_df thiếu {span - len(combined)} ngày trong khoảng {span} ngày — "
                "lag/rolling sẽ lệch. Nạp DB qua complete_panel()."
            )

        last_row = combined.iloc[-1]
        store_nbr = int(last_row["store_nbr"])
        family = str(last_row["family"])

        # ---- 1. Khung dữ liệu tương lai ----
        future_df = pd.DataFrame({"date": pd.to_datetime(future_dates)})
        future_df["target"] = np.nan

        static_cols = ["store_nbr", "family", "city", "state", "type",
                       "cluster", "cluster_family_id", "perishable"]
        for col in static_cols:
            if col in combined.columns:
                future_df[col] = last_row[col]

        if future_exog is not None and not future_exog.empty:
            future_exog = future_exog.copy()
            future_exog["date"] = pd.to_datetime(future_exog["date"])
            future_df = future_df.merge(future_exog, on="date", how="left", suffixes=("", "_exog"))

        if "onpromotion" not in future_df.columns or future_df["onpromotion"].isna().all():
            future_df["onpromotion"] = 0

        for col in combined.columns:
            if col not in future_df.columns:
                future_df[col] = np.nan

        if "holiday_type" in future_df.columns:
            future_df["holiday_type"] = future_df["holiday_type"].fillna(NORMAL_DAY)
        else:
            future_df["holiday_type"] = NORMAL_DAY

        combined = pd.concat([combined, future_df], ignore_index=True)

        # transactions tương lai: ffill theo store. KHÔNG ffill transactions_lag1 —
        # engineer_features::add_transactions_lag tự tái tạo từ transactions.
        if "transactions" not in combined.columns:
            logger.warning("history_df thiếu cột transactions — tín hiệu store-traffic chết.")
        else:
            if "transactions_lag1" in combined.columns:
                combined = combined.drop(columns="transactions_lag1")
            n_nan = int(combined["transactions"].isna().sum())
            combined["transactions"] = combined.groupby("store_nbr", observed=True)["transactions"].ffill()
            if n_nan:
                logger.debug(f"ffill 'transactions': {n_nan} NaN → giá trị thật cuối của store")

        predictions = []

        # ---- 2. Vòng lặp đệ quy ----
        for current_date in sorted(pd.to_datetime(pd.Series(future_dates)).unique()):
            d = pd.Timestamp(current_date)
            temp_combined = combined[
                (combined["date"] > d - pd.Timedelta(days=RECURSION_WINDOW_DAYS))
                & (combined["date"] <= d + pd.Timedelta(days=LEAD_BUFFER_DAYS))
            ].copy()

            # [H1] signature mới: KHÔNG còn fit_cluster
            # [H3] add_extra_features — cùng logic với train.py
            featured = engineer_features(temp_combined, cluster_engineer=self.cluster_engineer)
            featured = add_extra_features(
                featured, min_lag=self.min_lag, include_cross_series=False
            )

            day_featured = featured[featured["date"] == d].copy()
            if day_featured.empty:
                logger.warning(f"No features generated for date: {d.date()}")
                continue

            for col in self._required_feature_cols:
                if col not in day_featured.columns:
                    day_featured[col] = np.nan

            current_family = day_featured["family"].iloc[0]
            used_model = "None"

            # --- SMART ROUTING ---
            if current_family in self.local_models and (
                self.quality_filter is None or self.quality_filter(current_family)
            ):
                local_prep = self.local_models[current_family]["preprocessor"]
                local_model = self.local_models[current_family]["model"]

                if hasattr(local_prep, "feature_names_in_"):
                    X_local = local_prep.transform(day_featured[list(local_prep.feature_names_in_)])
                else:
                    X_local = local_prep.transform(day_featured)
                X_local = _fix_unknown_categorical(X_local)           # [H2]

                pred = np.expm1(local_model.predict(X_local))
                used_model = f"Local LGBM ({current_family})"

            elif self.global_lgbm is not None and self.global_prep is not None:
                X_lgb = _fix_unknown_categorical(self.global_prep.transform(day_featured))  # [H2]

                if self.global_cat is not None and len(self.cat_feature_names) > 0:
                    missing = [c for c in self.cat_feature_names if c not in day_featured.columns]
                    if missing:
                        # reindex fill 0 sẽ im lặng trả số rác → phải nói ra
                        logger.error(
                            f"CatBoost thiếu {len(missing)} feature: {missing[:8]} — "
                            "features_ext.py ở service không khớp model."
                        )
                    X_cat = day_featured.reindex(columns=self.cat_feature_names, fill_value=0)

                    if "store_nbr" in X_cat.columns and not pd.api.types.is_integer_dtype(X_cat["store_nbr"]):
                        X_cat["store_nbr"] = X_cat["store_nbr"].round().astype("int64")

                    for col in ["family", "city", "state", "type", "holiday_type", "cluster_family_id"]:
                        if col in X_cat.columns:
                            if col == "holiday_type":
                                X_cat[col] = X_cat[col].fillna(NORMAL_DAY)
                            X_cat[col] = X_cat[col].astype(str)

                    pred_lgb_raw = self.global_lgbm.predict(X_lgb)   # KHÔNG expm1
                    pred_cat_raw = self.global_cat.predict(X_cat)    # KHÔNG expm1
                    pred = np.expm1(self.w_lgbm * pred_lgb_raw + self.w_cat * pred_cat_raw)
                    used_model = "Global Ensemble (LGBM + CatBoost)"
                else:
                    pred = np.expm1(self.global_lgbm.predict(X_lgb))
                    used_model = "Global LGBM"
            else:
                pred = np.array([0.0])
                used_model = "Zero-Fallback (No Model)"

            pred_val = float(np.clip(pred, 0, None).ravel()[0])

            predictions.append({
                "date": str(d.date()),
                "store_nbr": store_nbr,
                "family": current_family,
                "predicted_sales": round(pred_val, 4),
                "used_model": used_model,
            })

            combined.loc[combined["date"] == d, "target"] = pred_val

        return predictions
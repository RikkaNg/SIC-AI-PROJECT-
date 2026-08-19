"""
ml_service/app/inference.py (v2.7 - Production Ready)
Inference Engine: Xử lý logic Recursive Forecasting và Smart Routing.
- Bảo toàn đầy đủ metadata (store_nbr, family, cluster, location) xuyên suốt tương lai.
- Tích hợp Smart Routing: Ưu tiên Local Family Model -> Fallback Global Ensemble (LGBM + CatBoost).
- Cập nhật đệ quy kết quả dự báo làm biến trễ (lag/rolling) cho các ngày tiếp theo.
"""

import sys
import logging
from pathlib import Path
import joblib
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# ====================== CẤU HÌNH ĐƯỜNG DẪN ======================
CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent
ML_SERVICE_DIR = APP_DIR.parent
PROJECT_ROOT = ML_SERVICE_DIR.parent
ML_TRAINING_SRC = PROJECT_ROOT / "ml_training" / "src"


for p in [str(ML_TRAINING_SRC), str(ML_SERVICE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.append(p)

from preprocessor import engineer_features

logger = logging.getLogger(__name__)


class RetailInferenceEngine:
    """Engine quản lý toàn bộ luồng dự báo chuỗi thời gian đệ quy và định tuyến mô hình."""

    def __init__(
        self,
        global_lgbm=None,
        global_prep=None,
        global_cat=None,
        local_models: Optional[Dict[str, Any]] = None,
        cluster_engineer=None,
        w_lgbm: float = 0.5,
        w_cat: float = 0.5,
    ):
        self.global_lgbm = global_lgbm
        self.global_prep = global_prep
        self.global_cat = global_cat
        self.local_models = local_models or {}
        self.cluster_engineer = cluster_engineer
        self.w_lgbm = w_lgbm
        self.w_cat = w_cat

        self.cat_feature_names = (
            getattr(self.global_cat, "feature_names_", []) if self.global_cat else []
        )

    def predict_recursive(
        self, 
        history_df: pd.DataFrame, 
        future_dates: List[str],
        future_exog: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """
        Dự báo đệ quy từng ngày cho một cặp (store_nbr, family).

        Args:
            history_df: Dữ liệu lịch sử (tối thiểu 60-90 ngày) có đủ cột date, store_nbr, family, target...
            future_dates: Danh sách ngày cần dự báo (VD: ['2017-08-16', '2017-08-17', ...])
            future_exog: (Tùy chọn) Bảng biến ngoại sinh tương lai (onpromotion, oil, holiday...)

        Returns:
            Danh sách dict gồm {date, store_nbr, family, predicted_sales, used_model}
        """
        if history_df.empty:
            raise ValueError("history_df cannot be empty.")

        combined = history_df.copy()
        combined["date"] = pd.to_datetime(combined["date"])
        
        # Lấy thông tin định danh tĩnh từ dòng lịch sử gần nhất
        last_row = history_df.iloc[-1]
        store_nbr = int(last_row["store_nbr"])
        family = str(last_row["family"])

        # 1. Khởi tạo khung dữ liệu tương lai
        future_df = pd.DataFrame({"date": pd.to_datetime(future_dates)})
        future_df["target"] = np.nan

        # Kế thừa toàn bộ metadata cố định của cửa hàng và ngành hàng
        static_cols = [
            "store_nbr", "family", "city", "state", "type", 
            "cluster", "cluster_family_id", "perishable"
        ]
        for col in static_cols:
            if col in history_df.columns:
                future_df[col] = last_row[col]

        # Ghép biến ngoại sinh tương lai nếu có (hoặc điền mặc định an toàn)
        if future_exog is not None and not future_exog.empty:
            future_exog["date"] = pd.to_datetime(future_exog["date"])
            future_df = future_df.merge(future_exog, on="date", how="left", suffixes=("", "_exog"))
        
        if "onpromotion" not in future_df.columns or future_df["onpromotion"].isna().all():
            future_df["onpromotion"] = 0

        # Đảm bảo đồng nhất tất cả các cột còn lại giữa lịch sử và tương lai
        for col in history_df.columns:
            if col not in future_df.columns:
                future_df[col] = np.nan

        combined = pd.concat([combined, future_df], ignore_index=True)
        predictions = []

        # 2. Vòng lặp đệ quy qua từng ngày
        for current_date in sorted(future_dates):
            current_date_ts = pd.to_datetime(current_date)
            temp_combined = combined[combined["date"] <= current_date_ts].copy()

            # Tạo feature kỹ thuật (lag, rolling, calendar)
            featured = engineer_features(
                temp_combined,
                cluster_engineer=self.cluster_engineer,
                fit_cluster=False,
            )

            day_featured = featured[featured["date"] == current_date_ts].copy()
            if day_featured.empty:
                logger.warning(f"No features generated for date: {current_date_ts.date()}")
                continue

            current_family = day_featured["family"].iloc[0]
            used_model = "None"

            # --- SMART ROUTING INFERENCE ---
            # Tuyến 1: Local Model chuyên biệt
            if current_family in self.local_models:
                local_prep = self.local_models[current_family]["preprocessor"]
                local_model = self.local_models[current_family]["model"]

                X_local = local_prep.transform(day_featured)
                pred = np.expm1(local_model.predict(X_local))
                used_model = f"Local LGBM ({current_family})"

            # Tuyến 2: Fallback Global Ensemble
            elif self.global_lgbm is not None and self.global_prep is not None:
                X_lgb = self.global_prep.transform(day_featured)
                pred_lgb = np.expm1(self.global_lgbm.predict(X_lgb))

                if self.global_cat is not None and len(self.cat_feature_names) > 0:
                    X_cat = day_featured.reindex(columns=self.cat_feature_names, fill_value=0)
                    cat_cols = ["family", "city", "state", "type", "holiday_type", "cluster_family_id"]
                    for col in cat_cols:
                        if col in X_cat.columns:
                            X_cat[col] = X_cat[col].astype(str)
                    pred_cat = np.expm1(self.global_cat.predict(X_cat))
                    pred = self.w_lgbm * pred_lgb + self.w_cat * pred_cat
                    used_model = "Global Ensemble (LGBM + CatBoost)"
                else:
                    pred = pred_lgb
                    used_model = "Global LGBM"
            else:
                pred = np.array([0.0])
                used_model = "Zero-Fallback (No Model)"

            # Chống âm và ép kiểu float tiêu chuẩn
            pred_val = float(np.clip(pred, 0, None).ravel()[0])

            predictions.append({
                "date": str(current_date_ts.date()),
                "store_nbr": store_nbr,
                "family": current_family,
                "predicted_sales": round(pred_val, 4),
                "used_model": used_model,
            })

            # Cập nhật dự báo vào combined làm giá trị trễ (lag) cho ngày tiếp theo
            combined.loc[combined["date"] == current_date_ts, "target"] = pred_val

        return predictions
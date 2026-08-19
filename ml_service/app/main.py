"""
ml_service/app/main.py (v2.6 - Production Ready)
FastAPI Inference Server cho ML Service:
- Nạp toàn bộ models (33 Local Models + Global Ensemble LGBM/CatBoost) vào RAM lúc startup qua lifespan.
- Cung cấp API /predict (dự báo đơn lẻ) và /predict/batch (dự báo hàng loạt tốc độ cao).
- Tích hợp Smart Routing: Ưu tiên Local Family Model -> Fallback Global Ensemble.
- Endpoint /health kiểm tra tình trạng nạp model.
"""

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# ====================== CẤU HÌNH ĐƯỜNG DẪN ======================
CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent                          # ml_service/app/
ML_SERVICE_DIR = APP_DIR.parent                        # ml_service/
PROJECT_ROOT = ML_SERVICE_DIR.parent                   # SIC-AI-PROJECT-/
ML_TRAINING_SRC = PROJECT_ROOT / "ml_training" / "src"

for p in [str(ML_TRAINING_SRC), str(ML_SERVICE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.append(p)

# Ưu tiên ml_service/models, nếu chưa có thì đọc từ ml_training/models
MODELS_DIR = ML_SERVICE_DIR / "models"
if not MODELS_DIR.exists() or not any(MODELS_DIR.iterdir()):
    MODELS_DIR = PROJECT_ROOT / "ml_training" / "models"

# ====================== LOGGING ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ml_service")

# ====================== IN-MEMORY MODEL REGISTRY ======================
models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Nạp models vào RAM khi server khởi động và giải phóng khi tắt."""
    logger.info(">>> [LIFESPAN] Loading ML models into memory...")

    # 1. Nạp Local Models (Ưu tiên)
    local_path = MODELS_DIR / "local_lgbm_models.pkl"
    if local_path.exists():
        try:
            models["local"] = joblib.load(local_path)
            logger.info(f"    Loaded {len(models['local'])} Local Family Models.")
        except Exception as e:
            logger.error(f"    Failed to load local models: {e}")

    # 2. Nạp Global LGBM & Preprocessor
    lgbm_path = MODELS_DIR / "lgbm_model.pkl"
    prep_path = MODELS_DIR / "preprocessor.pkl"

    if lgbm_path.exists():
        models["global_lgbm"] = joblib.load(lgbm_path)
        logger.info("    Loaded Global LightGBM Model.")
    if prep_path.exists():
        models["global_prep"] = joblib.load(prep_path)
        logger.info("    Loaded Global Preprocessor.")

    # 3. Nạp Global CatBoost (nếu có)
    cat_path = MODELS_DIR / "catboost_model.cbm"
    if cat_path.exists():
        try:
            from catboost import CatBoostRegressor
            cat_model = CatBoostRegressor()
            cat_model.load_model(str(cat_path))
            models["global_cat"] = cat_model
            logger.info("    Loaded Global CatBoost Model.")
        except Exception as e:
            logger.warning(f"    Could not load CatBoost: {e}")

    logger.info(f">>> [LIFESPAN] ML Service ready. Loaded keys: {list(models.keys())}")
    yield

    models.clear()
    logger.info(">>> [LIFESPAN] Models cleared from memory.")


# ====================== FASTAPI APP ======================
app = FastAPI(
    title="Retail AI - ML Inference Service",
    description="Microservice dự báo nhu cầu bán lẻ thời gian thực",
    version="2.6.0",
    lifespan=lifespan
)


# ====================== SCHEMAS ======================
class SinglePredictRequest(BaseModel):
    store_nbr: int
    family: str
    features: Dict[str, Any]


class SinglePredictResponse(BaseModel):
    store_nbr: int
    family: str
    predicted_sales: float
    used_model: str


class BatchPredictRequest(BaseModel):
    items: List[SinglePredictRequest]


class BatchPredictResponse(BaseModel):
    total_records: int
    predictions: List[SinglePredictResponse]


# ====================== CORE INFERENCE LOGIC ======================
def predict_single_row(store_nbr: int, family: str, feature_dict: Dict[str, Any]) -> SinglePredictResponse:
    df_row = pd.DataFrame([feature_dict])
    if "target" in df_row.columns:
        df_row = df_row.drop(columns=["target"])

    # TUYẾN 1: Local Model
    if "local" in models and family in models["local"]:
        local_artifact = models["local"][family]
        local_model = local_artifact["model"]
        local_prep = local_artifact["preprocessor"]

        X = local_prep.transform(df_row)
        pred = np.expm1(local_model.predict(X)[0])
        return SinglePredictResponse(
            store_nbr=store_nbr,
            family=family,
            predicted_sales=float(max(0.0, round(pred, 4))),
            used_model=f"Local LGBM ({family})"
        )

    # TUYẾN 2: Fallback Global Model
    elif "global_lgbm" in models and "global_prep" in models:
        X = models["global_prep"].transform(df_row)
        pred_lgb = np.expm1(models["global_lgbm"].predict(X)[0])

        if "global_cat" in models:
            cat_feature_names = getattr(models["global_cat"], "feature_names_", None)
            if cat_feature_names:
                X_cat = df_row.reindex(columns=cat_feature_names, fill_value=0)
                pred_cat = np.expm1(models["global_cat"].predict(X_cat)[0])
                pred = 0.5 * pred_lgb + 0.5 * pred_cat
            else:
                pred = pred_lgb
        else:
            pred = pred_lgb

        return SinglePredictResponse(
            store_nbr=store_nbr,
            family=family,
            predicted_sales=float(max(0.0, round(pred, 4))),
            used_model="Global Ensemble Fallback"
        )

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="No trained model available to handle this request."
    )


# ====================== ENDPOINTS ======================
@app.get("/health")
def health_check():
    return {
        "status": "healthy" if models else "degraded",
        "models_in_memory": list(models.keys()),
        "total_local_models": len(models.get("local", {}))
    }


@app.post("/predict", response_model=SinglePredictResponse)
def predict(req: SinglePredictRequest):
    return predict_single_row(req.store_nbr, req.family, req.features)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    results = [
        predict_single_row(item.store_nbr, item.family, item.features)
        for item in req.items
    ]
    return BatchPredictResponse(
        total_records=len(results),
        predictions=results
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
"""
ml_service/app/main.py (v3.0 - Production Ready)
FastAPI Inference Server cho ML Service:
- Nạp toàn bộ models (33 Local Models + Global Ensemble LGBM/CatBoost) vào RAM lúc startup qua lifespan.
- Cung cấp API /predict (dự báo đơn lẻ), /predict/batch (dự báo hàng loạt vector hóa)
  và /forecast (chuỗi dự báo đệ quy nhiều ngày qua RetailInferenceEngine).
- Tích hợp Smart Routing có kiểm soát chất lượng: Local Family Model (nếu RMSLE validation
  của family đó đạt ngưỡng) -> Fallback Global Ensemble với trọng số tuned từ ensemble_meta.json.
- Endpoint /health trả kèm metadata chất lượng model phục vụ giám sát.
"""

import sys
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# ====================== CẤU HÌNH ĐƯỜNG DẪN ======================
# Dò project root theo anchor file để khớp CẢ 2 layout:
#   local:     .../SIC-AI-PROJECT-/ml_service/app/main.py  -> root = SIC-AI-PROJECT-/
#   container: /app/app/main.py (WORKDIR /app)             -> root = /app/
CURRENT_FILE = Path(__file__).resolve()
APP_DIR = CURRENT_FILE.parent                          # ml_service/app/ | /app/app/
PROJECT_ROOT = CURRENT_FILE.parents[2]                 # fallback
for _parent in CURRENT_FILE.parents:
    if (_parent / "ml_training" / "src" / "preprocessor.py").exists():
        PROJECT_ROOT = _parent
        break
ML_SERVICE_DIR = APP_DIR.parent                        # ml_service/ | /app/
ML_TRAINING_DIR = PROJECT_ROOT / "ml_training"         # để unpickle artifact tham chiếu src.cluster_features
ML_TRAINING_SRC = ML_TRAINING_DIR / "src"

for p in [str(APP_DIR), str(ML_TRAINING_SRC), str(ML_TRAINING_DIR), str(ML_SERVICE_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.append(p)

# Ưu tiên ml_service/models, nếu chưa có thì đọc từ ml_training/models
MODELS_DIR = PROJECT_ROOT / "ml_service" / "models"
if not MODELS_DIR.exists() or not any(MODELS_DIR.iterdir()):
    MODELS_DIR = PROJECT_ROOT / "ml_training" / "models"

# ====================== CẤU HÌNH HÀNH VI (env-overridable) ======================
# Định tuyến theo chất lượng: family có RMSLE validation vượt
# LOCAL_RMSLE_MAX_FACTOR × RMSLE ensemble tham chiếu -> rơi về Global Ensemble.
LOCAL_RMSLE_MAX_FACTOR = float(os.environ.get("ML_LOCAL_RMSLE_MAX_FACTOR", "1.3"))
# Giới hạn horizont của /forecast để chặn payload quá lớn.
MAX_FORECAST_HORIZON = int(os.environ.get("ML_MAX_FORECAST_HORIZON", "60"))
# Số ngày lịch sử tối đa /forecast chấp nhận (chỉ giữ đuôi để giới hạn bộ nhớ xử lý).
MAX_HISTORY_ROWS = int(os.environ.get("ML_MAX_HISTORY_ROWS", "365"))

# ====================== LOGGING ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ml_service")

# ====================== IN-MEMORY MODEL REGISTRY ======================
models: Dict[str, Any] = {}


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"    Could not read {path.name}: {e}")
        return None


def _local_rmsle_threshold() -> float:
    """Ngưỡng RMSLE tối đa để một local model được phép phục vụ family của nó."""
    meta = models.get("ensemble_meta") or {}
    ref = float(meta.get("avg_blend_rmsle", 0.357))
    return LOCAL_RMSLE_MAX_FACTOR * ref


def _local_model_acceptable(family: str) -> bool:
    """
    Kiểm soát chất lượng định tuyến (§3.3): local model chỉ được dùng khi RMSLE
    validation của family đó <= ngưỡng. Thiếu thông tin metrics -> cho phép
    (giữ tương thích khi chưa chạy train_local_models.py mới).
    """
    local_metrics: Dict[str, float] = models.get("local_metrics", {})
    rmsle = local_metrics.get(family)
    if rmsle is None:
        return True
    return float(rmsle) <= _local_rmsle_threshold()


def _warm_up_models() -> None:
    """
    Làm ấm model bằng 1 lệnh predict giả mỗi tuyến (best-effort) để request
    đầu tiên của user không phải chịu chi phí khởi tạo nội bộ thư viện.
    """
    warmed: List[str] = []

    def _warm_with_prep(name: str, prep, model) -> None:
        cols = getattr(prep, "feature_names_in_", None)
        cols = list(cols) if cols is not None else []
        if not cols:
            return
        try:
            sample = pd.DataFrame([[0] * len(cols)], columns=cols)
            model.predict(prep.transform(sample))
            warmed.append(name)
        except Exception as e:
            logger.warning(f"    Warm-up {name} bỏ qua (dữ liệu giả không hợp lệ): {e}")

    if "global_prep" in models and "global_lgbm" in models:
        _warm_with_prep("global_lgbm", models["global_prep"], models["global_lgbm"])

    # Chỉ làm ấm 1 local model đại diện để không kéo dài startup
    local = models.get("local") or {}
    if local:
        fam, art = next(iter(local.items()))
        _warm_with_prep(f"local[{fam}]", art["preprocessor"], art["model"])

    cat = models.get("global_cat")
    if cat is not None:
        cat_names = getattr(cat, "feature_names_", None) or []
        if cat_names:
            for fill in (0, ""):
                try:
                    cat.predict(pd.DataFrame([{name: fill for name in cat_names}]))
                    warmed.append("global_cat")
                    break
                except Exception:
                    continue

    logger.info(f">>> [LIFESPAN] Warm-up hoàn tất: {warmed or 'không model nào'}")


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

    # 4. Nạp Cluster Feature Engineer (dùng cho /forecast đệ quy)
    cluster_path = MODELS_DIR / "cluster_engineer.pkl"
    if cluster_path.exists():
        try:
            loaded_cluster = joblib.load(cluster_path)
            if getattr(loaded_cluster, "cluster_stats", None) is not None:
                models["cluster_engineer"] = loaded_cluster
                logger.info("    Loaded Cluster Feature Engineer.")
        except Exception as e:
            logger.warning(f"    Could not load cluster_engineer: {e}")

    # 5. Trọng số ensemble đã tune từ cross-validation (thay cho 0.5/0.5 cứng)
    ensemble_meta = _load_json(MODELS_DIR / "ensemble_meta.json") or {}
    if ensemble_meta:
        models["ensemble_meta"] = ensemble_meta
        models["ensemble_w"] = (
            float(ensemble_meta.get("lgbm_weight", 0.5)),
            float(ensemble_meta.get("catboost_weight", 0.5)),
        )
        logger.info(f"    Ensemble weights (tuned): lgbm={models['ensemble_w'][0]}, "
                    f"catboost={models['ensemble_w'][1]}")
    else:
        models["ensemble_w"] = (0.5, 0.5)

    # 6. Metrics chất lượng từng local model (định tuyến theo chất lượng)
    metrics_path = MODELS_DIR / "local_models_metrics.csv"
    if metrics_path.exists():
        try:
            df_m = pd.read_csv(metrics_path)
            models["local_metrics"] = dict(zip(df_m["family"], df_m["rmsle"].astype(float)))
            threshold = _local_rmsle_threshold()
            fallback_fams = sorted(
                fam for fam, rmsle in models["local_metrics"].items()
                if rmsle > threshold
            )
            if fallback_fams:
                logger.info(f"    Local models vượt ngưỡng RMSLE {threshold:.4f} "
                            f"(factor {LOCAL_RMSLE_MAX_FACTOR}) -> sẽ fallback Global: {fallback_fams}")
        except Exception as e:
            logger.warning(f"    Could not load local model metrics: {e}")

    # 7. Metadata lần train gần nhất (phục vụ /health)
    local_meta = _load_json(MODELS_DIR / "local_models_metadata.json")
    if local_meta:
        models["local_meta"] = local_meta

    # 8. Khởi tạo RetailInferenceEngine cho /forecast (dự báo đệ quy live)
    if "global_lgbm" in models and "global_prep" in models:
        try:
            from inference import RetailInferenceEngine
            models["engine"] = RetailInferenceEngine(
                global_lgbm=models["global_lgbm"],
                global_prep=models["global_prep"],
                global_cat=models.get("global_cat"),
                local_models=models.get("local", {}),
                cluster_engineer=models.get("cluster_engineer"),
                w_lgbm=models["ensemble_w"][0],
                w_cat=models["ensemble_w"][1],
                quality_filter=_local_model_acceptable,
            )
            logger.info("    RetailInferenceEngine ready (endpoint /forecast).")
        except Exception as e:
            logger.error(f"    Could not initialize RetailInferenceEngine: {e}")

    # 9. Warm-up chống cold-start
    _warm_up_models()

    logger.info(f">>> [LIFESPAN] ML Service ready. Loaded keys: {list(models.keys())}")
    yield

    models.clear()
    logger.info(">>> [LIFESPAN] Models cleared from memory.")


# ====================== FASTAPI APP ======================
app = FastAPI(
    title="Retail AI - ML Inference Service",
    description="Microservice dự báo nhu cầu bán lẻ thời gian thực",
    version="3.0.0",
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


class ForecastSeriesRequest(BaseModel):
    store_nbr: int
    family: str
    history: List[Dict[str, Any]] = Field(
        ...,
        description="Lịch sử >= 60 ngày gần nhất, mỗi dòng có date/store_nbr/family/target + metadata.",
    )
    future_dates: List[str] = Field(..., description="Danh sách ngày cần dự báo (<= ML_MAX_FORECAST_HORIZON).")
    future_exog: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="(Tùy chọn) biến ngoại sinh tương lai: onpromotion, oil, holiday..."
    )


class ForecastSeriesResponse(BaseModel):
    store_nbr: int
    family: str
    predictions: List[Dict[str, Any]]


# ====================== CORE INFERENCE LOGIC ======================
def _global_ensemble_predict(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """Tuyến fallback: Global LGBM (+ CatBoost với trọng số tuned) trên cả lô dòng."""
    X = models["global_prep"].transform(df)
    preds = np.expm1(models["global_lgbm"].predict(X))
    used_model = "Global LGBM"

    cat = models.get("global_cat")
    if cat is not None:
        cat_feature_names = getattr(cat, "feature_names_", None)
        if cat_feature_names:
            X_cat = df.reindex(columns=cat_feature_names, fill_value=0)
            preds_cat = np.expm1(cat.predict(X_cat))
            w_lgbm, w_cat = models.get("ensemble_w", (0.5, 0.5))
            preds = w_lgbm * preds + w_cat * preds_cat
            used_model = "Global Ensemble Fallback"
    return preds, used_model


def _route_family_predict(family: str, df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    """
    Smart Routing cho cả lô dòng cùng family:
    Tuyến 1 - Local LGBM của family (nếu đạt ngưỡng chất lượng);
    Tuyến 2 - Global Ensemble fallback.
    """
    local = models.get("local") or {}
    if family in local and _local_model_acceptable(family):
        art = local[family]
        X = art["preprocessor"].transform(df)
        return np.expm1(art["model"].predict(X)), f"Local LGBM ({family})"
    if "global_lgbm" in models and "global_prep" in models:
        return _global_ensemble_predict(df)
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="No trained model available to handle this request."
    )


def _to_response(store_nbr: int, family: str, pred: float, used_model: str) -> SinglePredictResponse:
    return SinglePredictResponse(
        store_nbr=store_nbr,
        family=family,
        predicted_sales=float(max(0.0, round(float(np.clip(pred, 0, None)), 4))),
        used_model=used_model,
    )


def predict_single_row(store_nbr: int, family: str, feature_dict: Dict[str, Any]) -> SinglePredictResponse:
    df_row = pd.DataFrame([feature_dict])
    if "target" in df_row.columns:
        df_row = df_row.drop(columns=["target"])

    preds, used_model = _route_family_predict(family, df_row)
    return _to_response(store_nbr, family, preds[0], used_model)


def predict_batch_vectorized(items: List[SinglePredictRequest]) -> List[SinglePredictResponse]:
    """
    Vector hóa batch (§2.2): gom items theo family rồi transform/predict mỗi lần
    cho cả lô thay vì từng dòng - nhanh hơn bậc độ lớn với batch lớn.
    Kết quả giữ đúng thứ tự request và khớp với /predict từng dòng.
    """
    results: List[Optional[SinglePredictResponse]] = [None] * len(items)
    idxs_by_family: Dict[str, List[int]] = {}
    for idx, item in enumerate(items):
        idxs_by_family.setdefault(item.family, []).append(idx)

    for family, idxs in idxs_by_family.items():
        df = pd.DataFrame([items[i].features for i in idxs])
        if "target" in df.columns:
            df = df.drop(columns=["target"])

        preds, used_model = _route_family_predict(family, df)
        preds = np.clip(np.asarray(preds, dtype=float).ravel(), 0, None)
        for j, i in enumerate(idxs):
            results[i] = SinglePredictResponse(
                store_nbr=items[i].store_nbr,
                family=family,
                predicted_sales=float(max(0.0, round(float(preds[j]), 4))),
                used_model=used_model,
            )
    return results  # type: ignore[return-value]


# ====================== ENDPOINTS ======================
@app.get("/health")
def health_check():
    ensemble_meta = models.get("ensemble_meta") or {}
    local_meta = models.get("local_meta") or {}
    threshold = _local_rmsle_threshold() if models else None
    local_metrics: Dict[str, float] = models.get("local_metrics", {})
    return {
        "status": "healthy" if models else "degraded",
        "models_in_memory": list(models.keys()),
        "total_local_models": len(models.get("local", {})),
        "ensemble": {
            "weights": {
                "lgbm": models.get("ensemble_w", (None, None))[0],
                "catboost": models.get("ensemble_w", (None, None))[1],
            },
            "avg_blend_rmsle": ensemble_meta.get("avg_blend_rmsle"),
        },
        "local_routing": {
            "rmsle_threshold": round(threshold, 5) if threshold else None,
            "fallback_families": sorted(
                fam for fam, rmsle in local_metrics.items() if rmsle > threshold
            ) if threshold else [],
        },
        "trained_at": local_meta.get("trained_at"),
    }


@app.post("/predict", response_model=SinglePredictResponse)
def predict(req: SinglePredictRequest):
    return predict_single_row(req.store_nbr, req.family, req.features)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest):
    if not req.items:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="items không được rỗng.")
    results = predict_batch_vectorized(req.items)
    return BatchPredictResponse(
        total_records=len(results),
        predictions=results
    )


@app.post("/forecast", response_model=ForecastSeriesResponse)
def forecast_series(req: ForecastSeriesRequest):
    """
    Chuỗi dự báo đệ quy nhiều ngày cho 1 cặp (store_nbr, family) qua
    RetailInferenceEngine: smart routing + cập nhật lag bằng chính dự báo.
    Lịch sử khuyến nghị >= 60 ngày để đủ lag/rolling (sales_lag28 + rolling 7).
    """
    engine = models.get("engine")
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecast engine chưa sẵn sàng (thiếu Global models lúc startup).",
        )
    if not req.history:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="history không được rỗng.")
    if not req.future_dates:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="future_dates không được rỗng.")
    if len(req.future_dates) > MAX_FORECAST_HORIZON:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"future_dates tối đa {MAX_FORECAST_HORIZON} ngày mỗi request.",
        )

    history_df = pd.DataFrame(req.history)
    if "date" not in history_df.columns or "target" not in history_df.columns:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="history phải có tối thiểu các cột 'date' và 'target'.",
        )
    history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce")
    if history_df["date"].isna().all():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Không parse được 'date' nào trong history.")

    history_df = history_df.sort_values("date").tail(MAX_HISTORY_ROWS)

    last_family = str(history_df["family"].iloc[-1]) if "family" in history_df.columns else req.family
    if last_family != req.family:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"family trong history ({last_family}) không khớp với family yêu cầu ({req.family}).",
        )

    try:
        predictions = engine.predict_recursive(
            history_df=history_df,
            future_dates=req.future_dates,
            future_exog=pd.DataFrame(req.future_exog) if req.future_exog else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception:
        logger.exception("Lỗi dự báo đệ quy /forecast")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Lỗi nội bộ khi dự báo chuỗi. Vui lòng thử lại sau.")

    return ForecastSeriesResponse(
        store_nbr=req.store_nbr,
        family=req.family,
        predictions=predictions,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)

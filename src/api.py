from typing import List, Optional
 
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
 
from src.data_loader import load_data
from src.predict import predict, RAW_DIR
 
app = FastAPI(title="Sales Forecast API", version="1.0")
 
# Cho phép frontend (kể cả chạy trên domain/port khác, hoặc trong artifact
# preview của Claude) gọi API này. Khi deploy thật, nên giới hạn allow_origins
# về đúng domain frontend thay vì "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Cache đơn giản trong bộ nhớ — tránh chạy lại predict() (tốn thời gian vì
# phải build lag/rolling) ở mỗi request. Xoá cache bằng POST /api/refresh.
_predictions_cache: Optional[pd.DataFrame] = None
_history_cache: Optional[pd.DataFrame] = None
 
 
class PredictionOut(BaseModel):
    store_nbr: int
    family: str
    date: str
    predicted_sales: float
 
 
class AlertOut(BaseModel):
    store_nbr: int
    family: str
    predicted_avg: float
    historical_avg: float
    change_pct: float
    level: str  # "low_stock_risk" (nên nhập thêm) | "overstock_risk" (nên giảm nhập)
    message: str
 
 
def _get_predictions() -> pd.DataFrame:
    global _predictions_cache
    if _predictions_cache is None:
        test_path = RAW_DIR / "test.csv"
        if not test_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Không tìm thấy data/raw/test.csv để dự đoán.",
            )
        test_raw = pd.read_csv(test_path, parse_dates=["date"])
        _predictions_cache = predict(test_raw)
    return _predictions_cache
 
 
def _get_history() -> pd.DataFrame:
    global _history_cache
    if _history_cache is None:
        _history_cache = load_data()
    return _history_cache
 
 
@app.post("/api/refresh")
def refresh_cache():
    """Xoá cache, buộc lần gọi tiếp theo tính lại dự đoán từ đầu."""
    global _predictions_cache, _history_cache
    _predictions_cache = None
    _history_cache = None
    return {"status": "cache cleared"}
 
 
@app.get("/api/stores")
def get_stores() -> List[int]:
    df = _get_predictions()
    return sorted(df["store_nbr"].unique().tolist())
 
 
@app.get("/api/families")
def get_families() -> List[str]:
    df = _get_predictions()
    return sorted(df["family"].unique().tolist())
 
 
@app.get("/api/predictions", response_model=List[PredictionOut])
def get_predictions(
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
):
    df = _get_predictions().copy()
    if store_nbr is not None:
        df = df[df["store_nbr"] == store_nbr]
    if family is not None:
        df = df[df["family"] == family]
    if start_date is not None:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["date"] <= pd.Timestamp(end_date)]
 
    df = df.sort_values(["store_nbr", "family", "date"])
    df = df.copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.to_dict(orient="records")
 
 
@app.get("/api/kpi")
def get_kpi():
    df = _get_predictions()
    total_predicted = float(df["predicted_sales"].sum()) if len(df) else 0.0
    top_family = None
    if len(df):
        top_family = (
            df.groupby("family")["predicted_sales"].sum().sort_values(ascending=False).index[0]
        )
    return {
        "total_predicted_sales": total_predicted,
        "top_family": top_family,
        "forecast_days": int(df["date"].nunique()) if len(df) else 0,
        "stores_covered": int(df["store_nbr"].nunique()) if len(df) else 0,
    }
 
 
@app.get("/api/alerts", response_model=List[AlertOut])
def get_alerts(threshold_pct: float = 30.0):
    """
    So sánh trung bình dự báo (predicted_sales) với trung bình 28 ngày lịch
    sử gần nhất, cùng (store_nbr, family). Lệch quá threshold_pct% thì gắn
    cảnh báo cần nhập thêm hàng (dự báo tăng mạnh) hoặc giảm nhập (dự báo
    giảm mạnh, tránh tồn kho).
    """
    preds = _get_predictions()
    history = _get_history()
 
    if not len(preds):
        return []
 
    pred_avg = preds.groupby(["store_nbr", "family"])["predicted_sales"].mean().reset_index()
    pred_avg.columns = ["store_nbr", "family", "predicted_avg"]
 
    recent_cutoff = history["date"].max() - pd.Timedelta(days=28)
    recent_history = history[history["date"] >= recent_cutoff]
    hist_avg = recent_history.groupby(["store_nbr", "family"])["target"].mean().reset_index()
    hist_avg.columns = ["store_nbr", "family", "historical_avg"]
 
    merged = pred_avg.merge(hist_avg, on=["store_nbr", "family"], how="left")
    merged["historical_avg"] = merged["historical_avg"].fillna(0)
 
    def _classify(row):
        hist = row["historical_avg"]
        pred = row["predicted_avg"]
        change_pct = 0.0 if hist <= 0 else (pred - hist) / hist * 100
 
        if change_pct >= threshold_pct:
            level = "low_stock_risk"
            message = f"Dự báo tăng {change_pct:.0f}% so với trung bình gần đây — nên chuẩn bị nhập thêm hàng."
        elif change_pct <= -threshold_pct:
            level = "overstock_risk"
            message = f"Dự báo giảm {abs(change_pct):.0f}% so với trung bình gần đây — cân nhắc giảm nhập, tránh tồn kho."
        else:
            level = "normal"
            message = "Dự báo ổn định, không cần điều chỉnh."
        return pd.Series([change_pct, level, message])
 
    merged[["change_pct", "level", "message"]] = merged.apply(_classify, axis=1)
    merged = merged[merged["level"] != "normal"].sort_values(
        "change_pct", key=lambda s: s.abs(), ascending=False
    )
 
    return merged.to_dict(orient="records")
 
 
@app.get("/health")
def health():
    return {"status": "ok"}
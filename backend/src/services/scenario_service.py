# backend/src/services/scenario_service.py
"""
Scenario Lab - dịch vụ chạy lại dự báo với số liệu do user chỉnh (what-if).

Luồng: build history (90 ngày) từ retail.db + CSV tham chiếu -> gọi ml_service
/forecast (dự báo đệ quy 16 ngày với future_exog đã chỉnh) -> phân rã family
xuống SKU theo tỷ trọng bảng forecasts -> phân tích deterministic so với
baseline (bảng forecasts precomputed).

Điểm quan trọng về tính công bằng so sánh: baseline trong bảng `forecasts`
được sinh từ predict_local.py với exog thật của kỳ test (onpromotion lấy từ
test.csv, oil từ oil.csv, lịch nghỉ từ holidays_events.csv). Do đó exog
"trung lập" của kịch bản phải tái tạo đúng các nguồn đó; chỉ khi user chỉnh
thì giá trị mới đổi.
"""
import json
import logging
import math
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

# ====================== ĐƯỜNG DẪN & HẰNG SỐ ======================
_ENV_DB_PATH = os.getenv("DB_PATH")
if _ENV_DB_PATH:
    DB_PATH = Path(_ENV_DB_PATH)
else:
    DB_PATH = Path(__file__).resolve().parents[1] / "database" / "retail.db"

# CSV tham chiếu (oil, holidays, test) - trong container được mount từ
# ./ml_training/data/raw (xem docker-compose.yml). Có thể override bằng env.
RAW_DATA_DIR = Path(os.getenv(
    "SCENARIO_RAW_DATA_DIR",
    str(Path(__file__).resolve().parents[3] / "ml_training" / "data" / "raw"),
))

HISTORY_DAYS = 90          # đủ cho lag 14 + rolling 7 + cửa sổ đệ quy 60 ngày
HISTORY_TTL_SECONDS = 600  # cache history theo (store, family) 10 phút
FALLBACK_UNIT_PRICE = 5.0  # giá tham chiếu khi family_prices thiếu family
OVERSTOCK_DAYS_THRESHOLD = 30.0  # khớp quy ước overstock trong system prompt

ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://localhost:8001")
ML_SERIES_TIMEOUT = float(os.environ.get("ML_SERIES_TIMEOUT_SECONDS", "90"))


class ScenarioError(Exception):
    """Lỗi nghiệp vụ của Scenario Lab (trả 502/503 cho route biến thành HTTP)."""


# ====================== KẾT NỐI DB (read-only) ======================
def _ro_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# ====================== DỮ LIỆU THAM CHIẾU (cache tiến trình) ======================
_NAT_HOLIDAYS: Dict[str, Tuple[str, str]] = {}
_REG_HOLIDAYS: Dict[Tuple[str, str], Tuple[str, str]] = {}
_LOC_HOLIDAYS: Dict[Tuple[str, str], Tuple[str, str]] = {}
_OIL_SERIES: Dict[str, float] = {}
_FUTURE_PROMO: Dict[Tuple[str, int, str], int] = {}  # (date, store, family) -> 0/1
_FUTURE_PROMO_DAYS: Dict[Tuple[int, str], int] = {}  # (store, family) -> số ngày có KM
_calendar_loaded = False
_calendar_lock = threading.Lock()


def _load_calendar() -> None:
    """Nạp oil.csv + holidays_events.csv + test.csv vào cache tiến trình (1 lần).

    Tất cả đều best-effort: thiếu file thì degrad xuống giá trị mặc định
    (oil rỗng, không có ngày lễ, không có khuyến mãi baseline) thay vì crash.
    """
    global _calendar_loaded, _NAT_HOLIDAYS, _REG_HOLIDAYS, _LOC_HOLIDAYS, _OIL_SERIES
    with _calendar_lock:
        if _calendar_loaded:
            return
        _calendar_loaded = True  # đánh dấu trước để các lần gọi sau không chờ lock

        # --- Giá dầu: chuỗi ngày liên tục, ffill/bfill như prepare_oil() ---
        try:
            oil = pd.read_csv(RAW_DATA_DIR / "oil.csv", parse_dates=["date"])
            all_dates = pd.date_range(oil["date"].min(), oil["date"].max())
            oil = oil.set_index("date").reindex(all_dates)
            oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()
            _OIL_SERIES = {
                d.strftime("%Y-%m-%d"): round(float(v), 4)
                for d, v in oil["dcoilwtico"].items() if pd.notna(v)
            }
            logger.info(f"[scenario] oil.csv nạp {len(_OIL_SERIES)} ngày.")
        except Exception as e:
            logger.warning(f"[scenario] Không đọc được oil.csv ({e}) - dùng mặc định rỗng.")

        # --- Ngày lễ: National/Regional/Local, bỏ transferred (như data_loader) ---
        try:
            hol = pd.read_csv(RAW_DATA_DIR / "holidays_events.csv", parse_dates=["date"])
            hol = hol[hol["transferred"] == False]  # noqa: E712 - khớp data_loader
            for _, r in hol.iterrows():
                d = r["date"].strftime("%Y-%m-%d")
                entry = (str(r["type"]), str(r.get("description", "")))
                if r["locale"] == "National":
                    _NAT_HOLIDAYS.setdefault(d, entry)
                elif r["locale"] == "Regional":
                    _REG_HOLIDAYS.setdefault((d, str(r["locale_name"])), entry)
                elif r["locale"] == "Local":
                    _LOC_HOLIDAYS.setdefault((d, str(r["locale_name"])), entry)
            logger.info(
                f"[scenario] holidays nạp: nat={len(_NAT_HOLIDAYS)} "
                f"reg={len(_REG_HOLIDAYS)} loc={len(_LOC_HOLIDAYS)}."
            )
        except Exception as e:
            logger.warning(f"[scenario] Không đọc được holidays_events.csv ({e}).")

        # --- Khuyến mãi kỳ tương lai: onpromotion thật từ test.csv (như baseline) ---
        # test.csv nặng ~126MB nên aggregate 1 lần rồi cache ra file nhỏ cạnh DB;
        # xóa file cache này để buộc đọc lại từ test.csv.
        promo_cache_path = DB_PATH.parent / "scenario_future_promo.csv"
        try:
            agg_rows: List[Tuple[str, int, str, int]] = []
            if promo_cache_path.exists():
                cached = pd.read_csv(promo_cache_path)
                agg_rows = list(cached.itertuples(index=False, name=None))
                logger.info(f"[scenario] promo cache: {len(agg_rows)} dòng từ {promo_cache_path.name}.")
            else:
                test = pd.read_csv(
                    RAW_DATA_DIR / "test.csv",
                    usecols=["date", "store_nbr", "item_nbr", "onpromotion"],
                    parse_dates=["date"],
                )
                conn = _ro_conn()
                try:
                    items = pd.read_sql_query("SELECT item_nbr, family FROM items", conn)
                finally:
                    conn.close()
                test = test.merge(items, on="item_nbr", how="left")
                test["onpromotion"] = test["onpromotion"].fillna(False).astype(bool)
                agg = (
                    test.groupby([test["date"].dt.strftime("%Y-%m-%d"),
                                  "store_nbr", "family"])["onpromotion"]
                    .any().astype(int)
                )
                agg_rows = [(d, int(s), str(f), int(v)) for (d, s, f), v in agg.items()]
                try:
                    promo_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame(
                        agg_rows, columns=["date", "store_nbr", "family", "onpromotion"]
                    ).to_csv(promo_cache_path, index=False)
                    logger.info(f"[scenario] Đã ghi promo cache: {promo_cache_path.name}.")
                except OSError as e:
                    logger.warning(f"[scenario] Không ghi được promo cache ({e}).")
            for d, s, f, v in agg_rows:
                _FUTURE_PROMO[(str(d), int(s), str(f))] = int(v)
                if v:
                    _FUTURE_PROMO_DAYS[(int(s), str(f))] = \
                        _FUTURE_PROMO_DAYS.get((int(s), str(f)), 0) + 1
            logger.info(f"[scenario] future promo: {len(_FUTURE_PROMO)} nhóm (date,store,family).")
        except Exception as e:
            logger.warning(f"[scenario] Không nạp được promo baseline ({e}) - mặc định 0.")


def _holiday_for_date(date_str: str, city: str, state: str) -> Tuple[str, str, bool]:
    """Ưu tiên National > Regional > Local như merge_dimensions(); trả (type, description, is_earthquake)."""
    entry = _NAT_HOLIDAYS.get(date_str)
    if entry is None:
        entry = _REG_HOLIDAYS.get((date_str, state)) or _LOC_HOLIDAYS.get((date_str, city))
    if entry is None:
        return "Normal Day", "", False
    htype, desc = entry
    return htype, desc, "terremoto" in desc.lower()


def _oil_for_date(date_str: str) -> float:
    return _OIL_SERIES.get(date_str, float("nan"))


# ====================== HISTORY 90 NGÀY ======================
_history_cache: Dict[Tuple[int, str], Tuple[float, List[Dict[str, Any]]]] = {}
_history_lock = threading.Lock()
_hist_max_date_cache: Optional[str] = None


def _hist_max_date(conn: sqlite3.Connection) -> str:
    """Ngày cuối cùng có dữ liệu lịch sử. Suy ra nhanh từ MIN(date) của bảng
    forecasts (trừ 1 ngày) thay vì quét full 59 triệu dòng historical_sales
    (6.3s mỗi lần); chỉ fallback quét khi forecasts trống."""
    global _hist_max_date_cache
    if _hist_max_date_cache:
        return _hist_max_date_cache
    row = conn.execute("SELECT MIN(date) AS d FROM forecasts").fetchone()
    if row and row["d"]:
        _hist_max_date_cache = (pd.Timestamp(str(row["d"])) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return _hist_max_date_cache
    row = conn.execute("SELECT MAX(date) AS d FROM historical_sales").fetchone()
    if not row or not row["d"]:
        raise ScenarioError("Bảng historical_sales trống - hãy chạy init_database.py.")
    _hist_max_date_cache = str(row["d"])
    return _hist_max_date_cache


def build_history(store_nbr: int, family: str, history_days: int = HISTORY_DAYS) -> List[Dict[str, Any]]:
    """Lịch sử bán hàng cấp family theo ngày cho 1 cửa hàng, đủ cột cho
    engineer_features() bên ml_service. Cache TTL 10 phút theo (store, family)."""
    key = (int(store_nbr), str(family))
    with _history_lock:
        hit = _history_cache.get(key)
        if hit and time.monotonic() - hit[0] < HISTORY_TTL_SECONDS:
            return hit[1]

    conn = _ro_conn()
    try:
        end_date = _hist_max_date(conn)
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=history_days - 1)).strftime("%Y-%m-%d")
        lag_start = (pd.Timestamp(start_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # Aggregation qua TEMP table: nếu JOIN thẳng items, SQLite quét toàn bảng
        # items cho TỪNG dòng lịch sử (51s cho 90 ngày); TEMP table với PK lookup
        # chạy ~0.06s (đã đo).
        conn.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _fam_items "
            "(item_nbr INTEGER PRIMARY KEY, perishable INTEGER)"
        )
        conn.execute("DELETE FROM _fam_items")
        conn.execute(
            "INSERT INTO _fam_items SELECT item_nbr, perishable FROM items WHERE family = ?",
            (str(family),),
        )
        rows = conn.execute(
            """
            SELECT h.date AS date,
                   SUM(h.unit_sales) AS unit_sales,
                   MAX(COALESCE(h.onpromotion, 0)) AS onpromotion,
                   MAX(t.perishable) AS perishable
            FROM historical_sales h
            JOIN _fam_items t ON h.item_nbr = t.item_nbr
            WHERE h.store_nbr = ? AND h.date >= ?
            GROUP BY h.date
            ORDER BY h.date
            """,
            (int(store_nbr), start_date),
        ).fetchall()
        if not rows:
            raise ScenarioError(f"Không có lịch sử bán cho family '{family}' tại cửa hàng {store_nbr}.")

        store = conn.execute(
            "SELECT city, state, type, cluster FROM stores WHERE store_nbr = ?",
            (int(store_nbr),),
        ).fetchone()
        if not store:
            raise ScenarioError(f"Không tìm thấy cửa hàng {store_nbr}.")

        tx = conn.execute(
            "SELECT date, n_invoices FROM daily_transactions "
            "WHERE store_nbr = ? AND date >= ? ORDER BY date",
            (int(store_nbr), lag_start),
        ).fetchall()
    finally:
        conn.close()

    # transactions lag-1: tính trên chuỗi theo store (như prepare_transactions_lag)
    tx_map: Dict[str, float] = {}
    prev: Optional[float] = None
    for r in tx:
        tx_map[str(r["date"])] = float(prev) if prev is not None else float("nan")
        prev = float(r["n_invoices"])

    _load_calendar()
    city, state = str(store["city"]), str(store["state"])
    history: List[Dict[str, Any]] = []
    for r in rows:
        d = str(r["date"])
        htype, _desc, is_quake = _holiday_for_date(d, city, state)
        history.append({
            "date": d,
            "store_nbr": int(store_nbr),
            "family": str(family),
            "target": round(max(0.0, float(r["unit_sales"])), 4),
            "onpromotion": int(r["onpromotion"] or 0),
            "oil_price": _oil_for_date(d),
            "is_earthquake_period": int(is_quake),
            "holiday_type": htype,
            "city": city,
            "state": state,
            "type": str(store["type"]),
            "cluster": int(store["cluster"]),
            "perishable": int(r["perishable"] or 0),
            "transactions_lag1": tx_map.get(d, float("nan")),
        })

    with _history_lock:
        _history_cache[key] = (time.monotonic(), history)
    return history


# ====================== EXOG TƯƠNG LAI ======================
def build_future_exog(
    store_nbr: int,
    family: str,
    future_dates: List[str],
    promo_days: Optional[int] = None,
    oil_price: Optional[float] = None,
    traffic_change_pct: Optional[float] = None,
    event_type: str = "none",
    event_days: int = 0,
    last_transactions: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Exog từng ngày tương lai. Mặc định tái tạo đúng điều kiện baseline
    (promo từ test.csv, lịch thật từ holidays_events.csv, oil thật từ oil.csv);
    chỉ các tham số user truyền vào mới ghi đè N ngày đầu của kỳ."""
    _load_calendar()
    conn = _ro_conn()
    try:
        store = conn.execute(
            "SELECT city, state FROM stores WHERE store_nbr = ?", (int(store_nbr),)
        ).fetchone()
    finally:
        conn.close()
    city, state = str(store["city"]), str(store["state"])

    traffic_factor = 1.0
    if traffic_change_pct is not None:
        traffic_factor = max(0.0, 1.0 + float(traffic_change_pct) / 100.0)

    exog: List[Dict[str, Any]] = []
    for i, d in enumerate(future_dates):
        htype, _desc, is_quake = _holiday_for_date(d, city, state)

        # Khuyến mãi: mặc định theo lịch thật (test.csv); nếu user đặt promo_days
        # thì N ngày đầu = 1, phần còn lại = 0.
        if promo_days is not None:
            promo = 1 if i < int(promo_days) else 0
        else:
            promo = _FUTURE_PROMO.get((d, int(store_nbr), str(family)), 0)

        # Sự kiện bất ngờ: đè lên lịch thật trong N ngày đầu
        if event_type == "holiday" and i < int(event_days):
            htype, is_quake = "Holiday", False
        elif event_type == "earthquake" and i < int(event_days):
            htype, is_quake = "Event", True

        row = {
            "date": d,
            "onpromotion": int(promo),
            "oil_price": float(oil_price) if oil_price is not None else _oil_for_date(d),
            "holiday_type": htype,
            "is_earthquake_period": int(is_quake),
        }
        if traffic_change_pct is not None:
            base_tx = float(last_transactions) if last_transactions else 0.0
            row["transactions_lag1"] = round(max(0.0, base_tx * traffic_factor), 2)
        exog.append(row)
    return exog


# ====================== GỌI ML SERVICE ======================
def get_ml_forecast_series(store_nbr: int, family: str,
                           history: List[Dict[str, Any]],
                           future_dates: List[str],
                           future_exog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """POST /forecast sang ml_service (dự báo đệ quy). Sync vì được gọi từ
    threadpool của FastAPI và từ LLM tool. Raises ScenarioError khi lỗi."""
    payload = {
        "store_nbr": int(store_nbr),
        "family": str(family),
        "history": history,
        "future_dates": future_dates,
        "future_exog": future_exog,
    }
    try:
        resp = httpx.post(
            f"{ML_SERVICE_URL}/forecast", json=payload, timeout=ML_SERIES_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json().get("predictions", [])
    except httpx.HTTPStatusError as e:
        raise ScenarioError(f"ML Service trả lỗi {e.response.status_code}: {e.response.text[:200]}") from e
    except httpx.RequestError as e:
        raise ScenarioError(f"Không kết nối được ML Service ({ML_SERVICE_URL}): {e}") from e


# ====================== DỮ LIỆU PHỤ TRỢ (baseline, SKU, giá) ======================
def _baseline_family_series(conn: sqlite3.Connection, store_nbr: int,
                            family: str, future_dates: List[str]) -> List[Dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT f.date AS date, ROUND(SUM(f.predicted_sales), 4) AS predicted_sales
        FROM forecasts f JOIN items i ON f.item_nbr = i.item_nbr
        WHERE f.store_nbr = ? AND i.family = ? AND f.date BETWEEN ? AND ?
        GROUP BY f.date ORDER BY f.date
        """,
        (int(store_nbr), str(family), future_dates[0], future_dates[-1]),
    ).fetchall()
    return [{"date": str(r["date"]), "predicted_sales": float(r["predicted_sales"] or 0.0)} for r in rows]


def _item_shares(conn: sqlite3.Connection, store_nbr: int, family: str,
                 future_dates: List[str]) -> Dict[int, float]:
    """Tỷ trọng từng SKU trong family, lấy từ chính bảng forecasts của kỳ."""
    rows = conn.execute(
        """
        SELECT f.item_nbr AS item_nbr, SUM(f.predicted_sales) AS total
        FROM forecasts f JOIN items i ON f.item_nbr = i.item_nbr
        WHERE f.store_nbr = ? AND i.family = ? AND f.date BETWEEN ? AND ?
        GROUP BY f.item_nbr
        """,
        (int(store_nbr), str(family), future_dates[0], future_dates[-1]),
    ).fetchall()
    grand = sum(float(r["total"] or 0.0) for r in rows)
    if grand <= 0:
        return {}
    return {int(r["item_nbr"]): float(r["total"]) / grand for r in rows}


def _family_inventory(conn: sqlite3.Connection, store_nbr: int, family: str) -> Dict[str, Any]:
    row = conn.execute(
        """
        SELECT SUM(i2.current_stock) AS stock, AVG(i2.lead_time_days) AS lead_time,
               COUNT(*) AS sku_count
        FROM inventory i2 JOIN items i ON i2.item_nbr = i.item_nbr
        WHERE i2.store_nbr = ? AND i.family = ?
        """,
        (int(store_nbr), str(family)),
    ).fetchone()
    return {
        "stock": float(row["stock"] or 0.0),
        "lead_time": round(float(row["lead_time"]), 1) if row["lead_time"] is not None else 3.0,
        "sku_count": int(row["sku_count"] or 0),
    }


def _unit_price(conn: sqlite3.Connection, family: str) -> float:
    row = conn.execute(
        "SELECT unit_price FROM family_prices WHERE UPPER(family) = ?",
        (str(family).upper(),),
    ).fetchone()
    return float(row["unit_price"]) if row and row["unit_price"] else FALLBACK_UNIT_PRICE


def _future_window(conn: sqlite3.Connection, horizon_days: int) -> List[str]:
    row = conn.execute("SELECT MIN(date) AS dmin, MAX(date) AS dmax FROM forecasts").fetchone()
    if not row or not row["dmin"]:
        raise ScenarioError("Bảng forecasts trống - hãy chạy init_database.py.")
    dates = pd.date_range(row["dmin"], row["dmax"]).strftime("%Y-%m-%d").tolist()
    return dates[: int(horizon_days)]


# ====================== PHÂN TÍCH DETERMINISTIC ======================
def _round2(x: float) -> float:
    return round(float(x), 2)


def _days_cover(stock: float, daily_demand: float) -> float:
    if daily_demand <= 0:
        return 999.0
    return min(999.0, stock / daily_demand)


def analyze_scenario(
    family: str,
    baseline_series: List[Dict[str, Any]],
    scenario_series: List[Dict[str, Any]],
    stock: float,
    lead_time: float,
    unit_price: float,
    horizon_days: int,
) -> Dict[str, Any]:
    """So baseline vs kịch bản -> KPI + kết luận + đề xuất (tiếng Việt làm sẵn
    để LLM/UI chỉ trình bày, không tự tính)."""
    old_total = sum(p["predicted_sales"] for p in baseline_series)
    new_total = sum(p["predicted_sales"] for p in scenario_series)
    delta = new_total - old_total
    delta_pct = (delta / old_total * 100.0) if old_total > 0 else None

    daily_new = new_total / horizon_days if horizon_days > 0 else 0.0
    cover = _days_cover(stock, daily_new)
    shortfall = max(0.0, new_total - stock)
    excess = max(0.0, stock - new_total)

    will_stockout = shortfall > 0
    overstock = (not will_stockout) and cover > OVERSTOCK_DAYS_THRESHOLD and new_total > 0

    if will_stockout:
        ratio = shortfall / max(stock, 1.0)
        if ratio > 2.0:
            risk = "Rất cao"
        elif ratio > 1.0:
            risk = "Cao"
        elif ratio > 0.3:
            risk = "Trung bình"
        else:
            risk = "Thấp"
        verdict = "SẮP ĐỨT HÀNG"
    elif overstock:
        risk = "Dư tồn"
        verdict = "DƯ TỒN"
    else:
        risk = "An toàn"
        verdict = "ĐỦ HÀNG"

    revenue_new = new_total * unit_price
    revenue_old = old_total * unit_price
    lost_revenue = shortfall * unit_price

    # --- Văn bản phân tích (UI/LLM trình bày nguyên trạng) ---
    delta_txt = f"{delta_pct:+.1f}%" if delta_pct is not None else "không xác định (baseline = 0)"
    lines = [
        f"• Dự báo tổng {horizon_days} ngày cho ngành {family}: "
        f"{_round2(old_total)} → {_round2(new_total)} đơn vị ({delta_txt}).",
        f"• Tồn kho hiện tại: {_round2(stock)} đơn vị "
        f"(~{_round2(cover)} ngày bán ở mức nhu cầu mới).",
    ]
    if will_stockout:
        lines.append(
            f"• KẾT LUẬN: {verdict} — thiếu ~{_round2(shortfall)} đơn vị "
            f"trong kỳ nếu không nhập thêm (mức rủi ro: {risk})."
        )
        lines.append(
            f"• Doanh thu kỳ vọng kỳ mới ≈ ${_round2(revenue_new):,.2f}; "
            f"doanh thu mất nếu không nhập đủ ≈ ${_round2(lost_revenue):,.2f} "
            f"(giá tham chiếu ${_round2(unit_price)}/đơn vị)."
        )
    elif overstock:
        lines.append(
            f"• KẾT LUẬN: {verdict} — dư ~{_round2(excess)} đơn vị "
            f"({risk}), chiếm vốn khoảng ${_round2(excess * unit_price):,.2f}."
        )
        lines.append(f"• Doanh thu kỳ vọng kỳ mới ≈ ${_round2(revenue_new):,.2f}.")
    else:
        lines.append(
            f"• KẾT LUẬN: {verdict} — nhu cầu mới nằm trong khả năng đáp ứng của tồn kho."
        )
        lines.append(f"• Doanh thu kỳ vọng kỳ mới ≈ ${_round2(revenue_new):,.2f}.")

    # --- Đề xuất hành động ---
    if will_stockout:
        recommendation = (
            f"Cần nhập tối thiểu {_round2(shortfall)} đơn vị trong {horizon_days} ngày tới "
            f"(≈ {_round2(shortfall / max(horizon_days, 1))} đơn vị/ngày). "
            f"Đặt hàng ngay hôm nay vì lead_time trung bình {lead_time} ngày."
        )
    elif overstock:
        recommendation = (
            f"Không nhập thêm trong kỳ. Cân nhắc khuyến mãi/giảm giá để giải phóng "
            f"~{_round2(excess)} đơn vị tồn (≈ {_round2(cover)} ngày bán)."
        )
    else:
        recommendation = (
            "Duy trì kế hoạch nhập hiện tại; theo dõi reorder point "
            f"(lead_time {lead_time} ngày) để đặt hàng đúng thời điểm."
        )

    return {
        "kpi": {
            "baseline_total": _round2(old_total),
            "scenario_total": _round2(new_total),
            "delta": _round2(delta),
            "delta_pct": round(delta_pct, 2) if delta_pct is not None else None,
            "stock": _round2(stock),
            "days_of_cover": _round2(cover),
            "shortfall": _round2(shortfall),
            "excess": _round2(excess),
            "will_stockout": bool(will_stockout),
            "overstock": bool(overstock),
            "risk_level": risk,
            "verdict": verdict,
            "unit_price": _round2(unit_price),
            "expected_revenue_usd": _round2(revenue_new),
            "baseline_revenue_usd": _round2(revenue_old),
            "lost_revenue_usd": _round2(lost_revenue),
        },
        "analysis": "\n".join(lines),
        "recommendation": recommendation,
    }


# ====================== ĐIỂM VÀO CHÍNH ======================
def _json_safe(value: Any) -> Any:
    """Đảm bảo mọi số đều hữu hạn (agent/UI nhận JSON hợp lệ)."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def run_scenario(
    store_nbr: int,
    family: str,
    horizon_days: int = 16,
    demand_multiplier: float = 1.0,
    promo_days: Optional[int] = None,
    oil_price: Optional[float] = None,
    traffic_change_pct: Optional[float] = None,
    event_type: str = "none",
    event_days: int = 0,
    stock_override: Optional[float] = None,
    lead_time_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Chạy 1 kịch bản what-if trọn vẹn. Raises ScenarioError khi dữ liệu/ML lỗi."""
    conn = _ro_conn()
    try:
        future_dates = _future_window(conn, horizon_days)
        baseline_series = _baseline_family_series(conn, store_nbr, family, future_dates)
        shares = _item_shares(conn, store_nbr, family, future_dates)
        inv = _family_inventory(conn, store_nbr, family)
    finally:
        conn.close()
    if not baseline_series:
        raise ScenarioError(
            f"Không có baseline dự báo cho family '{family}' tại cửa hàng {store_nbr}."
        )

    history = build_history(store_nbr, family)

    # Số giao dịch ngày cuối history làm mốc cho biến % lưu lượng khách
    last_tx = history[-1].get("transactions_lag1") if history else None
    if last_tx is None or (isinstance(last_tx, float) and not math.isfinite(last_tx)):
        last_tx = 0.0

    exog = build_future_exog(
        store_nbr, family, future_dates,
        promo_days=promo_days, oil_price=oil_price,
        traffic_change_pct=traffic_change_pct,
        event_type=event_type, event_days=event_days,
        last_transactions=float(last_tx),
    )

    try:
        predictions = get_ml_forecast_series(store_nbr, family, history, future_dates, exog)
        source = "ml_service"
    except ScenarioError:
        if demand_multiplier == 1.0 and promo_days is None and oil_price is None \
                and traffic_change_pct is None and event_type == "none":
            raise  # kịch bản trung lập vẫn fail thì không có gì để trả
        # Graceful degradation: nhân hệ số lên baseline thay vì bỏ cuộc
        predictions = [
            {"date": b["date"], "predicted_sales": b["predicted_sales"] * float(demand_multiplier)}
            for b in baseline_series
        ]
        source = "approximate_fallback"

    scenario_series = [
        {"date": str(p["date"]),
         "predicted_sales": round(max(0.0, float(p["predicted_sales"]) * float(demand_multiplier)), 4)}
        for p in predictions
    ]

    stock = float(stock_override) if stock_override is not None else inv["stock"]
    lead_time = float(lead_time_override) if lead_time_override is not None else inv["lead_time"]
    conn = _ro_conn()
    try:
        price = _unit_price(conn, family)
    finally:
        conn.close()

    analysis = analyze_scenario(
        family, baseline_series, scenario_series,
        stock=stock, lead_time=lead_time, unit_price=price,
        horizon_days=len(future_dates),
    )

    # --- Phân rã xuống SKU + top biến động ---
    per_sku: List[Dict[str, Any]] = []
    base_by_date = {b["date"]: b["predicted_sales"] for b in baseline_series}
    if shares:
        conn = _ro_conn()
        try:
            sku_stocks = {
                int(r["item_nbr"]): float(r["stock"] or 0.0)
                for r in conn.execute(
                    """
                    SELECT i2.item_nbr AS item_nbr, i2.current_stock AS stock
                    FROM inventory i2 JOIN items i ON i2.item_nbr = i.item_nbr
                    WHERE i2.store_nbr = ? AND i.family = ?
                    """,
                    (int(store_nbr), str(family)),
                ).fetchall()
            }
        finally:
            conn.close()
        for item_nbr, share in shares.items():
            sku_base = sum(base_by_date[d] * share for d in future_dates)
            sku_scen = sum(s["predicted_sales"] * share for s in scenario_series)
            sku_stock = sku_stocks.get(item_nbr, 0.0)
            per_sku.append({
                "item_nbr": int(item_nbr),
                "baseline_total": _round2(sku_base),
                "scenario_total": _round2(sku_scen),
                "delta": _round2(sku_scen - sku_base),
                "delta_pct": round((sku_scen - sku_base) / sku_base * 100.0, 2) if sku_base > 0 else None,
                "stock": _round2(sku_stock),
                "shortfall": _round2(max(0.0, sku_scen - sku_stock)),
            })
        per_sku.sort(key=lambda r: abs(r["delta"]), reverse=True)

    return _json_safe({
        "status": "success",
        "source": source,
        "store_nbr": int(store_nbr),
        "family": str(family),
        "horizon_days": len(future_dates),
        "future_dates": future_dates,
        "baseline_series": baseline_series,
        "scenario_series": scenario_series,
        "sku_count": len(shares),
        "inventory": {"stock": inv["stock"], "lead_time": inv["lead_time"],
                      "sku_count": inv["sku_count"], "stock_used": _round2(stock),
                      "lead_time_used": lead_time},
        "overrides_applied": {
            "demand_multiplier": demand_multiplier,
            "promo_days": promo_days,
            "oil_price": oil_price,
            "traffic_change_pct": traffic_change_pct,
            "event_type": event_type,
            "event_days": event_days,
            "stock_override": stock_override,
            "lead_time_override": lead_time_override,
        },
        **analysis,
        "top_sku_movements": per_sku[:8],
    })


def get_scenario_meta(store_nbr: int, family: str) -> Dict[str, Any]:
    """Giá trị mặc định để form UI prefill (tồn, lead time, giá, promo baseline...)."""
    conn = _ro_conn()
    try:
        future_dates = _future_window(conn, 16)
        inv = _family_inventory(conn, store_nbr, family)
        price = _unit_price(conn, family)
    finally:
        conn.close()
    _load_calendar()
    oil_now = next(
        (v for d, v in sorted(_OIL_SERIES.items(), reverse=True) if d <= future_dates[0]),
        None,
    )
    history = build_history(store_nbr, family)
    last_tx = history[-1].get("transactions_lag1") if history else None

    return _json_safe({
        "store_nbr": int(store_nbr),
        "family": str(family),
        "future_dates": future_dates,
        "default_stock": round(inv["stock"], 2),
        "default_lead_time": inv["lead_time"],
        "sku_count": inv["sku_count"],
        "unit_price": round(price, 2),
        "baseline_promo_days": _FUTURE_PROMO_DAYS.get((int(store_nbr), str(family)), 0),
        "current_oil_price": oil_now,
        "last_transactions": round(float(last_tx), 0) if last_tx and math.isfinite(float(last_tx)) else 0.0,
    })

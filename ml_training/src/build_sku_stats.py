"""
build_sku_stats.py (chạy độc lập hoặc từ init_database.py)
Quét historical_sales MỘT LƯỢT để dựng bảng cache `sku_stats` + `model_meta` trong retail.db:

- sold_2016        : tổng unit_sales năm 2016 theo SKU (toàn chuỗi)
- hist_total       : tổng toàn kỳ lịch sử có trong DB (>= 2016-01-01)
- avg_daily_45d    : trung bình ngày trong 45 ngày cuối (2017-07-02 .. 2017-08-15)
- w0_total/w1_total: hai cửa sổ 45 ngày liên tiếp để tính validation proxy
- fc_total_16d     : tổng dự báo 16 ngày (từ bảng forecasts)
- abc_class        : A/B/C theo tỷ trọng cộng dồn hist_total (80%/95%)
- family_rmsle     : RMSLE walk-forward của local model family (từ local_models_metrics.csv)
- model_meta       : global_rmsle, built_at, sku_validation (JSON theo lớp ABC)

Chạy: python ml_training/src/build_sku_stats.py
"""

import csv
import json
import logging
import sqlite3
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_sku_stats")

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent
DB_PATH = PROJECT_ROOT / "backend" / "src" / "database" / "retail.db"
METRICS_CSV = PROJECT_ROOT / "ml_service" / "models" / "local_models_metrics.csv"

GLOBAL_RMSLE_DEFAULT = 0.357
W1_START = "2017-07-02"          # 45 ngày cuối lịch sử
W0_START = "2017-05-18"          # 45 ngày liền trước w1 (cửa sổ so sánh)
HIST_START = "2016-01-01"
Y2016_END = "2016-12-31"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sku_stats (
    item_nbr      INTEGER PRIMARY KEY,
    family        TEXT,
    class_code    INTEGER,
    perishable    INTEGER,
    sold_2016     REAL DEFAULT 0,
    hist_total    REAL DEFAULT 0,
    avg_daily_45d REAL DEFAULT 0,
    w0_total      REAL DEFAULT 0,
    w1_total      REAL DEFAULT 0,
    fc_total_16d  REAL DEFAULT 0,
    abc_class     TEXT,
    family_rmsle  REAL
);
CREATE TABLE IF NOT EXISTS model_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def load_family_rmsle() -> dict:
    rmsle = {}
    if METRICS_CSV.exists():
        with open(METRICS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    rmsle[row["family"]] = float(row["rmsle"])
                except (KeyError, ValueError):
                    continue
    logger.info(f"Loaded family RMSLE cho {len(rmsle)} family từ {METRICS_CSV.name}")
    return rmsle


def scan_history(conn: sqlite3.Connection) -> dict:
    """Một lượt quét historical_sales -> {item_nbr: [s2016, s_all, s_w0, s_w1]}."""
    agg: dict[int, list[float]] = {}
    cur = conn.execute(
        "SELECT item_nbr, date, unit_sales FROM historical_sales ORDER BY rowid"
    )
    n = 0
    t0 = time.time()
    while True:
        batch = cur.fetchmany(500_000)
        if not batch:
            break
        for item_nbr, date, sales in batch:
            s = agg.get(item_nbr)
            if s is None:
                s = agg[item_nbr] = [0.0, 0.0, 0.0, 0.0]
            v = sales or 0.0
            s[1] += v                                   # hist_total
            if HIST_START <= date <= Y2016_END:
                s[0] += v                               # sold_2016
            if date >= W1_START:
                s[3] += v                               # w1 (45 ngày cuối)
            elif date >= W0_START:
                s[2] += v                               # w0 (45 ngày liền trước)
        n += len(batch)
        logger.info(f"  ...đã quét {n:,} dòng ({time.time()-t0:.0f}s)")
    logger.info(f"Quét xong {n:,} dòng trong {time.time()-t0:.0f}s -> {len(agg):,} SKU")
    return agg


def classify_abc(hist_totals: dict[int, float]) -> dict[int, str]:
    total = sum(hist_totals.values()) or 1.0
    classes: dict[int, str] = {}
    cum = 0.0
    for it, v in sorted(hist_totals.items(), key=lambda kv: kv[1], reverse=True):
        cum += v
        classes[it] = "A" if cum <= 0.80 * total else ("B" if cum <= 0.95 * total else "C")
    return classes


def validation_proxy(agg: dict, classes: dict) -> dict:
    """Độ lệch w1 so với w0 theo lớp ABC (proxy chất lượng SKU-level)."""
    out = {}
    for cls in ("A", "B", "C"):
        devs = []
        for it, s in agg.items():
            if classes.get(it) != cls or s[2] <= 0:
                continue
            devs.append((s[3] - s[2]) / s[2] * 100.0)
        if not devs:
            out[cls] = {"n": 0}
            continue
        devs.sort()
        m = len(devs)
        out[cls] = {
            "n": m,
            "median_dev_pct": round(devs[m // 2], 1),
            "mean_abs_dev_pct": round(sum(abs(d) for d in devs) / m, 1),
            "pct_within_30": round(100 * sum(1 for d in devs if abs(d) <= 30) / m),
        }
    return out


def build_sku_stats(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)

    items = {
        r[0]: (r[1], r[2], r[3])
        for r in conn.execute("SELECT item_nbr, family, class, perishable FROM items")
    }
    logger.info(f"items catalog: {len(items):,} SKU")

    logger.info(">>> Đang quét historical_sales (một lượt duy nhất)...")
    agg = scan_history(conn)

    logger.info(">>> Phân loại ABC...")
    classes = classify_abc({it: s[1] for it, s in agg.items()})

    logger.info(">>> Tổng hợp dự báo 16 ngày theo SKU...")
    fc = {r[0]: r[1] for r in conn.execute(
        "SELECT item_nbr, SUM(predicted_sales) FROM forecasts GROUP BY item_nbr")}

    rmsle_map = load_family_rmsle()

    logger.info(">>> Ghi bảng sku_stats...")
    conn.execute("DELETE FROM sku_stats")
    rows = []
    for it, (family, class_code, perishable) in items.items():
        s = agg.get(it, [0.0, 0.0, 0.0, 0.0])
        rows.append((
            int(it), family, class_code, perishable,
            round(s[0], 2), round(s[1], 2), round(s[3] / 45.0, 4),
            round(s[2], 2), round(s[3], 2), round(fc.get(it, 0.0), 2),
            classes.get(it, "C"),
            rmsle_map.get(family, GLOBAL_RMSLE_DEFAULT),
        ))
        if len(rows) >= 20_000:
            conn.executemany(
                "INSERT OR REPLACE INTO sku_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            rows.clear()
    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO sku_stats VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()

    counts = {c: conn.execute("SELECT COUNT(*) FROM sku_stats WHERE abc_class=?", (c,)).fetchone()[0]
              for c in ("A", "B", "C")}
    logger.info(f"ABC: A={counts['A']}, B={counts['B']}, C={counts['C']}")

    meta_rows = [
        ("global_rmsle", str(GLOBAL_RMSLE_DEFAULT)),
        ("built_at", time.strftime("%Y-%m-%dT%H:%M:%S")),
        ("sku_validation", json.dumps(validation_proxy(agg, classes), ensure_ascii=False)),
    ]
    conn.executemany("INSERT OR REPLACE INTO model_meta VALUES (?,?)", meta_rows)
    conn.commit()

    n_final = conn.execute("SELECT COUNT(*) FROM sku_stats").fetchone()[0]
    logger.info(f"Hoàn tất: sku_stats={n_final:,} dòng.")


if __name__ == "__main__":
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy {DB_PATH}")
    c = sqlite3.connect(str(DB_PATH))
    try:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA mmap_size=2147483648")
        c.execute("PRAGMA cache_size=-131072")
        build_sku_stats(c)
    finally:
        c.close()

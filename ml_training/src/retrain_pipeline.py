# -*- coding: utf-8 -*-
"""
retrain_pipeline.py (v1.0)
Job retraining định kỳ (§3.4): đóng gói toàn bộ chuỗi làm mới model + dự báo.

Chuỗi tương đương chạy tay 3 script:
  1. train_local_models.py        -> ml_service/models/local_lgbm_models.pkl
                                     (+ local_models_metrics.csv, metadata.json)
  2. predict_local.py             -> ml_training/data/processed/submission_local.csv
                                     (dự báo đệ quy smart routing, ghi đè file cũ)
  3. load_forecasts_inventory.py  -> backend/src/database/retail.db
                                     (nạp lại bảng forecasts + sinh inventory + sku_stats)

Chạy thủ công:
    python ml_training/src/retrain_pipeline.py

Chạy bằng Docker (one-shot, không ảnh hưởng service đang chạy):
    docker compose --profile retrain run --rm ml-retrain

Lên lịch định kỳ: cron host chạy lệnh Docker ở trên (VD hàng tuần), hoặc scheduler
bên ngoài gọi endpoint/job này. Sau khi xong, restart backend/ml_service để nạp
model + dữ liệu mới (bảng forecasts được backend đọc live nên không cần restart).
"""

import sys
import logging
import time
from pathlib import Path

# Console Windows mặc định cp1252 - ép UTF-8 để log tiếng Việt không crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def main() -> int:
    started = time.monotonic()
    logger.info("=" * 60)
    logger.info(">>> RETRAIN PIPELINE START")
    logger.info("=" * 60)

    # Bước 1: train lại 33 local family models (fallback Global không cần train lại)
    logger.info(">>> [1/3] Training local family models...")
    import train_local_models
    train_local_models.train_local_models()

    # Bước 2: dự báo đệ quy trên test horizon -> submission CSV
    logger.info(">>> [2/3] Running recursive forecast -> submission_local.csv...")
    import predict_local
    submission = predict_local.build_submission_local()
    if submission is None:
        logger.error(">>> Dự báo thất bại (submission rỗng) - DỪNG, không đụng vào DB.")
        return 1

    # Bước 3: nạp forecasts + inventory + sku_stats vào retail.db của backend
    logger.info(">>> [3/3] Reloading forecasts/inventory vào retail.db...")
    import load_forecasts_inventory
    load_forecasts_inventory.main()

    elapsed = time.monotonic() - started
    logger.info("=" * 60)
    logger.info(f">>> RETRAIN PIPELINE DONE in {elapsed / 60:.1f} min - DB đã được làm mới.")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

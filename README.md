# SIC-AI-PROJECT

SIC-AI-PROJECT/
│
├── backend/                         # API Gateway + Orchestrator (FastAPI)
│   ├── src/
│   │   ├── database/                # Nơi lưu trữ DB SQLite & kết nối
│   │   │   ├── connection.py        # Quản lý connection SQLite WAL mode
│   │   │   └── retail.db            # File SQLite Database
│   │   ├── llm_agent/               # Toàn bộ logic LLM Agent (Groq Qwen 3 - qwen/qwen3-32b)
│   │   ├── security.py              # Auth JWT/API-key + Row-Level Isolation theo cửa hàng
│   │   ├── scripts/init_auth.py     # Tạo auth.db + seed user (admin / manager1 / manager2)
│   │   └── database/
│   │       ├── retail.db            # Dữ liệu bán lẻ (SQLite WAL)
│   │       └── auth.db              # User + quyền store_nbr (tách riêng để không mất khi re-init)
│   │   │   ├── config.py            # Khởi tạo Groq Client
│   │   │   ├── prompts.py           # Quản lý System Prompts chuỗi cung ứng
│   │   │   ├── tools.py             # Function calling (Truy vấn DB, tính toán tồn kho)
│   │   │   └── agent.py             # Bộ não Qwen Agent
│   │   ├── services/                # Các client giao tiếp nội bộ
│   │   │   └── ml_client.py         # Code httpx gọi sang ml_service (:8001)
│   │   ├── routes/                  # API endpoints cho Web UI
│   │   │   ├── forecast_routes.py   # API lấy số liệu dự báo
│   │   │   ├── inventory_routes.py  # API quản lý tồn kho, đặt hàng
│   │   │   └── chat_routes.py       # API tương tác với LLM Assistant
│   │   └── main.py                  # Entrypoint khởi động FastAPI Gateway (:8000)
│   ├── requirements.txt
│   └── Dockerfile
│
├── ml_service/                      # Microservice dự báo ML (Port 8001)
│   ├── models/                      # Chứa các file trọng số (Mount vào container)
│   │   ├── local_lgbm_models.pkl    # 33 Local Models
│   │   ├── lgbm_model.pkl           # Global LGBM Fallback
│   │   ├── catboost_model.cbm       # Global CatBoost Fallback
│   │   ├── cluster_engineer.pkl     # Bộ mã hóa Cluster / Store
│   │   └── preprocessor.pkl         # Scaler / Target Encoder
│   ├── app/
│   │   ├── main.py                  # FastAPI server: /health, /predict
│   │   └── inference.py             # Smart Routing & Recursive Forecasting
│   ├── requirements.txt
│   └── Dockerfile
│
├── ml_training/                     # Môi trường huấn luyện Offline (Chạy trên máy cục bộ)
│   ├── data/
│   │   ├── raw/                     # Dữ liệu gốc Kaggle (train.csv, items.csv,...)
│   │   └── processed/               # Feature store sau khi xử lý
│   ├── notebooks/
│   │   └── 01_EDA.ipynb
│   ├── src/
│   │   ├── data_loader.py           # Đọc và ghép nối dữ liệu
│   │   ├── preprocessor.py          # Feature Engineering (Lags, Rolling, Date)
│   │   ├── train.py                 # Huấn luyện 33 local + Global models
│   │   └── init_database.py         # Script nạp dữ liệu lịch sử & dự báo vào retail.db
│   └── requirements.txt
│
├── frontend/                        # Giao diện người dùng (React/Vite + Tailwind, build bằng Dockerfile & nginx)
│   ├── src/
│   ├── requirements.txt (hoặc package.json)
│   └── Dockerfile
│
├── docker-compose.yml               # Điều phối 3 services: ml_service, backend, frontend
├── .env                             # Chứa GROQ_API_KEY (Không commit lên Git)
├── .gitignore
└── README.md

## API sản phẩm (SKU cụ thể - dữ liệu thật từ retail.db)

| Endpoint | Mô tả |
|---|---|
| `GET /api/products` | Danh mục 4.100 sản phẩm thật: tồn kho, tổng bán 2016, trạng thái; hỗ trợ `search`, `family`, `status`, `sort`, phân trang server-side |
| `GET /api/top-products` | Top sản phẩm bán chạy theo **item_nbr cụ thể** (doanh số 2016), lọc theo cửa hàng |
| `GET /api/family-mix` | Thị phần doanh số theo nhóm hàng (pie chart) |
| `GET /api/family-trend` | Chuỗi dự báo theo ngày × nhóm hàng từ model LightGBM |
| `GET /api/product-families` | Danh sách nhóm hàng cho dropdown lọc |

Tất cả đều áp Row-Level Isolation theo phạm vi cửa hàng của user.

### Bảng tổng hợp cache (BẮT BUỘC chạy 1 lần sau khi init DB)

Các endpoint trên đọc bảng tổng hợp để phản hồi <1s thay vì quét ~59 triệu dòng historical_sales:

```bash
python backend/scripts/build_sales_cache.py   # dựng agg_item_store_sales + agg_forecast_date_family (~60s)
```

`ml_training/src/init_database.py` cũng tự dựng 2 bảng này mỗi lần init DB.

### LLM Chatbot

- Model mặc định đã đổi sang **`qwen/qwen3.6-27b`** vì `qwen/qwen3-32b` bị Groq ngừng phục vụ
  (API trả 404 `model_not_found` → backend 500 → frontend báo lỗi fetch). Đổi model khác qua `LLM_MODEL_NAME` trong `.env`.
- Lỗi LLM giờ trả về HTTP 502 kèm `detail` tiếng Việt rõ ràng thay vì 500 trống không thông tin.
- Sau khi đổi `.env`, cần **restart/recreate container backend** (`docker compose up -d --force-recreate backend`)
  hoặc khởi động lại uvicorn để nhận model mới và các route sản phẩm mới.

### Scenario Lab — kịch bản What-if (mới)

Chỉnh số liệu → dự báo lại bằng mô hình thật → phân tích tự động → xem kết quả, ở **2 kênh**:

- **Web**: view "Kịch bản What-if" trong dashboard — chọn cửa hàng + ngành hàng, chỉnh 6 nhóm số liệu
  (hệ số nhu cầu 0.5–2×, khuyến mãi, giá dầu, lưu lượng khách, sự kiện bất ngờ ngày lễ/thiên tai,
  tồn kho + lead time), bấm "Chạy kịch bản" → biểu đồ so sánh trước/sau + KPI + kết luận + đề xuất nhập + Top SKU biến động.
- **Chatbot**: hỏi "giả lập tăng 30% nhu cầu ngành BEVERAGES tại cửa hàng 1" — LLM gọi tool
  `run_scenario_analysis` (backend tự chạy toàn bộ phân tích, LLM chỉ trình bày lại).

API: `POST /api/scenario/run` (RLS theo cửa hàng) + `GET /api/scenario/meta` (giá trị prefill).

Lưu ý vận hành:

- Lần chạy đầu mỗi tiến trình backend cần đọc `test.csv` (126 MB) để nạp lịch khuyến mãi baseline →
  ~1-2 phút, sau đó tự cache ra `backend/src/database/scenario_future_promo.csv` (xóa file này để build lại).
- 1 lần chạy kịch bản mất ~30-45s do dự báo đệ quy 16 ngày bên ml_service.
- Lịch/khuyến mãi baseline lấy từ `ml_training/data/raw` (oil.csv, holidays_events.csv, test.csv);
  docker đã mount read-only vào backend.

### Chạy test (backend/tests)

```bash
# Từ thư mục gốc project (cần backend :8000 + ml_service :8001 đang chạy)
python -m pytest                        # toàn bộ 113 test
python -m pytest -m "not llm and not slow"   # bỏ qua các test gọi Groq API thật
python -m pytest backend/tests/test_api.py -v
```

Lưu ý quan trọng:

- Test dùng `http://127.0.0.1:8000/8001` (không phải `localhost`): trên máy dev,
  `wslrelay.exe` chiếm `[::1]:8000/8001` (service cũ trong WSL không có model, trả
  `degraded`) nên resolve `localhost` sang IPv6 sẽ trúng nhầm service đó.
- Các test chat (`llm`, `slow` + chat trong test_api/test_security/test_edge_cases) gọi
  **Groq API thật**. Free tier chỉ có 8.000 TPM / 200.000 TPD, mỗi lời gọi chat tốn
  ~5.000 token → chạy cả suite một mạch sẽ bị 429 → backend trả 502 → test fail.
  Hãy chạy test LLM tách lẻ (cách nhau ~45-60s) hoặc nâng tier Groq.
- Cần Python có: fastapi, uvicorn, groq, pyjwt, httpx, pandas (backend) và
  lightgbm, catboost, scikit-learn (ml_service), cùng pytest cho test.
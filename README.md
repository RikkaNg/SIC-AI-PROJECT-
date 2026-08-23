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
# SIC-AI-PROJECT

SIC-AI-PROJECT/
│
├── backend/                         # API Gateway + Orchestrator (FastAPI)
│   ├── src/
│   │   ├── database/                # Nơi lưu trữ DB SQLite & kết nối
│   │   │   ├── connection.py        # Quản lý connection SQLite WAL mode
│   │   │   └── retail.db            # File SQLite Database
│   │   ├── llm_agent/               # Toàn bộ logic LLM Agent (Groq Qwen 3.6)
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
├── frontend/                        # Giao diện người dùng (Streamlit hoặc React/Vue)
│   ├── src/
│   ├── requirements.txt (hoặc package.json)
│   └── Dockerfile
│
├── docker-compose.yml               # Điều phối 3 services: ml_service, backend, frontend
├── .env                             # Chứa GROQ_API_KEY (Không commit lên Git)
├── .gitignore
└── README.md
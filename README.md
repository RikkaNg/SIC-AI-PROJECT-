# SIC-AI-PROJECT

SIC-AI-PROJECT/
│
├── backend/                       # Lớp API Gateway (Node.js hoặc Python FastAPI)
│   ├── src/
│   │   ├── database/               # Đặt file retail.db ở đây
│   │   │   └── retail.db
│   │   ├── agent-tools/            # Code Function Calling gọi sang ml_service
│   │   │   └── supply_chain.py
│   │   ├── routes/                 # Các route API cho frontend
│   │   └── main.py                 # Entry point FastAPI của Backend
│   ├── requirements.txt
│   └── Dockerfile
│
├── ml_service/                    # Lớp Inference (Chỉ chứa model và API dự báo)
│   ├── models/                     # Các file .pkl, .cbm đã train xong
│   │   ├── lgbm_model.pkl
│   │   ├── catboost_model.cbm
│   │   └── preprocessor.pkl
│   ├── app/
│   │   ├── main.py                 # FastAPI: POST /predict, POST /simulate
│   │   └── inference.py            # Code chạy recursive forecast
│   ├── requirements.txt
│   └── Dockerfile
│
├── ml_training/                   # Lớp Offline (Chạy trên máy bạn, không đưa lên Docker)
│   ├── data/
│   │   ├── raw/
│   │   └── processed/
│   ├── notebooks/
│   │   └── 01_EDA.ipynb
│   ├── src/                        # Tất cả code training ở đây
│   │   ├── train.py
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   └── init_database.py        # Script tạo DB dời về đây
│   └── requirements.txt
│
├── llm_service/                   # Lớp điều phối AI
│   ├── prompts/
│   │   └── system_prompt.txt
│   ├── tools/
│   │   └── schema.json             # JSON Schema khai báo tools cho LLM
│   └── Modelfile                   # Cấu hình Ollama (nếu chạy local)
│
├── frontend/                      # Giao diện Web (React/Vue)
│   └── src/
│
├── docker-compose.yml             # File kết nối các service
├── .gitignore
└── README.md
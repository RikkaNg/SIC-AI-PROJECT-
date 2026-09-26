# SIC-AI-PROJECT

Hệ thống **AI quản trị chuỗi cung ứng bán lẻ**: dự báo nhu cầu bằng Machine Learning,
AI Agent trả lời câu hỏi kinh doanh bằng tiếng Việt tự nhiên (28 tools function-calling),
dashboard quản trị với Scenario Lab kịch bản what-if — tất cả chạy local bằng Docker.

> Dataset: bán lẻ Ecuador (54 cửa hàng, 2013–2017, ~59 triệu dòng giao dịch lịch sử).
> Dự báo: 33 local models theo ngành hàng + global ensemble (LightGBM + CatBoost).

---

## Kiến trúc

```
┌─────────────┐   /api/*    ┌──────────────────┐   /predict   ┌──────────────┐
│  Frontend   │ ──────────> │  Backend Gateway │ ───────────> │  ML Service  │
│ React+Vite  │   :8501→80  │  FastAPI  :8000  │  :8001       │  FastAPI     │
│ (nginx)     │             │  + LLM Agent     │ <──────────  │  33 models   │
└─────────────┘             │  + RLS + JWT/API │   SQLite WAL └──────────────┘
                            └────────┬─────────┘   (read-only)
                                     │ Groq API (Qwen 3.8, function calling)
                                     ▼
                              retail.db (SQLite ~4GB, không commit)
```

| Service | Port | Vai trò |
|---|---|---|
| `frontend` | 8501→80 | Dashboard React/Vite qua nginx, healthcheck riêng |
| `backend` | 8000 | API Gateway: auth JWT/X-API-Key, RLS theo cửa hàng, LLM Agent, Scenario API |
| `ml_service` | 8001 | Dự báo: smart routing 33 local model per-family + global LGBM/CatBoost ensemble, dự báo đệ quy 16 ngày |
| `ml-retrain` | one-shot | Profile compose (`--profile retrain`): train lại model → nạp forecasts/inventory/sku_stats vào DB |

## Cấu trúc thư mục

```
├── backend/
│   ├── src/
│   │   ├── llm_agent/            # AI Agent: agent.py (điều phối + chống bịa + retry),
│   │   │                         #   tools.py (28 tools), prompts.py, config.py
│   │   ├── routes/               # auth, chat, dashboard, forecast, inventory,
│   │   │                         #   product, scenario (kịch bản what-if)
│   │   ├── services/             # ml_client.py, scenario_service.py
│   │   ├── security.py           # JWT HS256 + X-API-Key + Row-Level Isolation
│   │   └── database/             # retail.db (~4GB, ignored) + auth.db (ignored)
│   ├── scripts/                  # init_auth, build_sales_cache, build_business_cache,
│   │                             #   build_promo_cache, load_daily_transactions, gen_jwt_secret
│   ├── tests/                    # 143 test pytest (9 file)
│   └── Dockerfile
├── ml_service/
│   ├── app/                      # main.py (routes), inference.py (smart routing + recursive)
│   ├── models/                   # lgbm_model.pkl + preprocessor.pkl (được commit),
│   │                             #   local_lgbm_models.pkl (retrain sinh, ignored)
│   └── Dockerfile
├── ml_training/
│   ├── data/raw|processed/       # Dataset gốc + feature store (ignored, có .gitkeep)
│   ├── src/                      # init_database, train, build_sku_stats, retrain_pipeline...
│   └── notebook/01_EDA.ipynb
├── frontend/src/app/App.tsx      # Toàn bộ dashboard (single-file app)
├── docker-compose.yml            # 3 service + profile "retrain"
├── pytest.ini                    # markers: llm, slow
└── .envexamble                   # Template env (commit) — copy thành .env rồi điền key
```

## Quick start

### 1. Cấu hình môi trường

```bash
cp .envexamble .env        # rồi điền GROQ_API_KEY (lấy tại console.groq.com/keys)
```

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `GROQ_API_KEY` | — | API key Groq (bắt buộc cho AI Agent) |
| `GROQ_BASE_URL` | (trống) | Endpoint OpenAI-compatible khác (VD OrcaRouter) — trống = Groq |
| `LLM_MODEL_NAME` | `qwen/qwen3.8-27b` | Model Groq bất kỳ hỗ trợ function calling |
| `LLM_REASONING_EFFORT` | `none` | Tắt thinking của Qwen 3 (trống = không gửi tham số) |
| `LLM_HISTORY_TOKEN_BUDGET` | `1200` | Ngân sách token cho lịch sử hội thoại (backend tự trim) |
| `LLM_HISTORY_MAX_MESSAGES` / `_MAX_CHARS_PER_MSG` | `40` / `4000` | Giới hạn số tin / độ dài mỗi tin |
| `LLM_GROUNDING_MIN_MATCH` / `LLM_GROUNDING_RETRIES` | `0.6` / `1` | Chống bịa số liệu (0 = tắt verifier) |
| `JWT_SECRET` | auto | Khóa ký JWT — sinh bằng `python backend/scripts/gen_jwt_secret.py` |

Khởi động bằng Docker (khuyến nghị):

```bash
docker compose up -d --build          # 3 service + healthcheck tự động
```

Hoặc không Docker (3 cửa sổ PowerShell): `powershell -ExecutionPolicy Bypass -File run_local.ps1`.
Đăng nhập: `admin/admin123` (toàn chuỗi) · `manager1|manager2/manager123` (RLS theo cửa hàng).

### 2. Dựng database + cache (chạy 1 lần)

```bash
python ml_training/src/init_database.py            # retail.db: schema + dữ liệu + 2 bảng agg cơ bản
python backend/scripts/build_business_cache.py     # agg_daily_business + family_prices + sku_stats (doanh thu USD tham chiếu)
python backend/scripts/build_promo_cache.py        # agg_promo_family_stats (quét 59M dòng ~20 phút, 1 lần duy nhất)
python backend/scripts/load_daily_transactions.py  # daily_transactions (lượt khách theo ngày)
python backend/scripts/init_auth.py                # auth.db: admin + manager theo cửa hàng
```

`retail.db` (~4GB) và `auth.db` bị gitignore — ai clone repo phải tự dựng từ `ml_training/data/raw`.

---

## AI Agent — 28 tools function calling

Agent trả lời câu hỏi tiếng Việt tự nhiên (không cần cú pháp), chỉ được dùng số liệu
từ tool — hỏi ngoài phạm vi thì từ chối có hướng dẫn thay vì bịa.

| Nhóm | Tools |
|---|---|
| Dự báo & tồn kho | `get_sales_summary` (kèm `forecast_window` mốc ngày thật), `get_family_forecast`, `check_stockout_risk`, `check_perishable_risk`, `calculate_reorder_point`, `calculate_purchase_target`, `evaluate_stockout_loss`, `find_cross_sell_items`, `recommend_slow_mover_strategy` |
| Doanh thu thực tế | `get_monthly_revenue`, `compare_stores_revenue`, `get_top_selling_items`, `get_store_traffic`, `get_store_profile`, `get_item_profile` |
| Phân tích tài chính (FP&A) | `analyze_gross_margin`, `analyze_revenue_change` (phân tách lượt khách × giá trị hóa đơn, bridge đóng về 0), `benchmark_store_vs_peers`, `analyze_reorder_profitability` |
| Quản trị bán lẻ | `analyze_inventory_health` (DOH/vòng quay/overstock), `find_dead_stock`, `get_abc_analysis` (A≤80%/B≤95%), `analyze_weekly_pattern`, `compare_family_mix` |
| Kịch bản | `run_scenario_analysis` (ML what-if theo ngành), `simulate_demand_multiplier`, `compare_cluster_trends`, `evaluate_promotion_impact` |

### Chống bịa số liệu (grounding check)

- **Đường A** (có gọi tool): mọi số trong câu trả lời được đối chiếu với payload tool
  (hiểu cả định dạng `239.626,92` VN lẫn `239,626.92` US, bỏ qua số đếm nhỏ, chấp nhận
  số derived ≤ ngưỡng). Khớp < `LLM_GROUNDING_MIN_MATCH` → tự retry 1 lần ép dùng số từ tool.
- **Đường B** (không gọi tool nào nhưng reply chứa số) → retry với tools vẫn cấp:
  hoặc gọi tool, hoặc thừa nhận chưa có dữ liệu.
- System prompt có danh mục NĂNG LỰC / KHÔNG CÓ (giá bán thật từng SKU, chi phí cố định,
  dòng tiền, lợi nhuận ròng, đối thủ...).

### Ngữ cảnh hội thoại & độ trễ

- Lịch sử do backend làm chủ: sanitize role (chặn client chèn `system`/`tool` giả),
  trim theo ngân sách token, không cắt giữa cặp hỏi–đáp.
- Retry 429 thông minh: đọc thời lượng Groq khuyên chờ; hết hạn mức **ngày** (TPD 200K
  free tier) thì báo lỗi rõ thay vì treo. Mỗi LLM call đều log latency + token usage.
- Trần thực tế của gói free Groq (TPM 8K): 1 câu hỏi ~6K token → câu hỏi đơn lẻ ~1–5s;
  hỏi liên tục trong cùng một phút sẽ chờ cửa sổ hạn mức (~30–45s). Nâng tier là cách
  xử lý triệt để duy nhất.

---

## API Gateway (:8000, tất cả áp RLS)

| Endpoint | Mô tả |
|---|---|
| `POST /api/auth/login` | JWT (admin/manager) hoặc `X-API-Key` cho ERP/POS |
| `POST /api/chat` | AI Agent (lịch sử hội thoại được backend sanitize + trim) |
| `GET /api/products` · `/api/top-products` · `/api/product-families` | Danh mục SKU, top bán chạy, ngành hàng |
| `GET /api/family-mix` · `/api/family-trend` | Thị phần theo ngành, chuỗi dự báo ngày × ngành |
| `GET /api/dashboard/*` · `/api/forecast` · `/api/inventory/*` | KPI, dự báo, tồn kho/đặt hàng |
| `POST /api/scenario/run` · `GET /api/scenario/meta` | Scenario Lab what-if (dự báo lại bằng model thật) |

Lỗi LLM/Gateway trả HTTP 502 kèm `detail` tiếng Việt rõ ràng.

## Scenario Lab

Chỉnh số liệu (hệ số cầu 0.5–2×, khuyến mãi, giá dầu, lưu lượng khách, sự kiện
ngày lễ/thiên tai, tồn kho + lead time) → dự báo lại bằng mô hình thật 16 ngày →
so trước/sau + KPI + kết luận. Hai kênh dùng chung một engine: view "Kịch bản What-if"
trên dashboard, hoặc hỏi trực tiếp chatbot. Lần chạy đầu backend đọc `test.csv` (126MB)
để nạp lịch khuyến mãi baseline (~1–2 phút, tự cache). Một lần chạy ~30–45s.

## ML Service (:8001) & Training

- `GET /health` · `POST /predict` · `POST /predict/batch` · `POST /forecast`
- Smart routing: local model theo ngành nếu có, fallback global LGBM → CatBoost ensemble;
  dự báo đệ quy 16 ngày cho chuỗi dài.
- 6 chỉ số đánh giá sau mỗi lần train (RMSLE · MAE · RMSE · WAPE · WMAPE trọng số
  perishable ×1.5 · R²): `ml_service/models/ensemble_meta.json` (global + Optuna) và
  `local_models_metrics.csv` (33 local). Unit test công thức: `backend/tests/test_train_metrics.py`.
- Retrain toàn bộ: `docker compose --profile retrain run --rm ml-retrain`
  (train → dự báo → nạp lại `forecasts`/`inventory`/`sku_stats` vào retail.db).

## Chạy test

```bash
python -m pytest                        # toàn bộ 143 test
python -m pytest -m "not llm and not slow"   # bỏ qua test gọi Groq API thật
python -m pytest backend/tests/test_llm_history.py -v   # chạy không cần server
```

- Test integration trỏ `http://127.0.0.1:8000` (dùng IP thay vì `localhost` để tránh
  `wslrelay.exe` chiếm IPv6 trên máy dev).
- Test chat (`llm`, `slow`) gọi **Groq thật** — free tier 8K TPM / 200K TPD, chạy tách
  lẻ hoặc nâng tier, nếu không sẽ 429 → backend 502 → test fail.
- Test grounding/context (`test_llm_history.py`) dùng logic thuần, không cần backend.

## Bảo mật

- Mật khẩu PBKDF2-HMAC-SHA256 (200K iterations), JWT HS256 hết hạn theo `TOKEN_EXPIRE_MINUTES`.
- **Row-Level Isolation**: manager chỉ thấy cửa hàng được gán — áp cho mọi API route
  VÀ mọi tool của AI Agent (validate trước khi thực thi, kể cả so sánh 2 cửa hàng).
- `retail.db`, `auth.db`, `.env` đều gitignore. `JWT_SECRET` sinh tự động, đừng dùng giá trị mặc định khi triển khai thật.

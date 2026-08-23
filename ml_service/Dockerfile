# Sử dụng Python 3.9 slim
FROM python:3.9-slim

# Bắt buộc: Cài đặt libgomp1 (OpenMP cho LightGBM), build-essential và curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập biến môi trường
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:/app/ml_training/src"

WORKDIR /app

# Cài đặt Python dependencies
COPY ml_service/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn API của ml_service
COPY ml_service/app/ ./app/

# Copy toàn bộ thư mục src của ml_training (gồm preprocessor.py, data_loader.py,...)
COPY ml_training/src/ ./ml_training/src/

# Tạo sẵn thư mục mount models
RUN mkdir -p /app/ml_service/models /app/ml_training/models

EXPOSE 8001

# Chạy Uvicorn Server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
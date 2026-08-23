# Sử dụng image Python nhẹ để tối ưu dung lượng
FROM python:3.11-slim

# Cài đặt thư viện hệ thống (bắt buộc để chạy pandas/numpy)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Set biến môi trường để code chạy được (như bên ta làm trên localhost)
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy file requirements.txt vào container trước
COPY backend/requirements.txt /app/backend/requirements.txt

# Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy toàn bộ source code của backend vào container
COPY backend/ /app/backend/

# Mở cổng 8000
EXPOSE 8000

# Lệnh khởi chạy Uvicorn
CMD ["uvicorn", "backend.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
# frontend/Dockerfile - Build React (Vite) và serve bằng nginx
# Lưu ý: build context là GỐC dự án (xem docker-compose.yml), nên mọi COPY phải
# dẫn đường từ gốc, VD: frontend/...

FROM node:20-alpine AS build

WORKDIR /app

# Cài dependencies trước để tận dụng layer cache
COPY frontend/package.json frontend/package-lock.json frontend/pnpm-workspace.yaml ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund

# Copy mã nguồn frontend rồi build production bundle
COPY frontend/ .
RUN npm run build

# Serve bằng nginx với SPA fallback
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

# backend/src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.src.routes import chat_routes, forecast_routes, inventory_routes, dashboard_routes, auth_routes, product_routes

app = FastAPI(title="Retail API Gateway", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký Routes
app.include_router(auth_routes.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(chat_routes.router, prefix="/api", tags=["Chat Assistant"])
app.include_router(forecast_routes.router, prefix="/api", tags=["Forecasting"])
app.include_router(inventory_routes.router, prefix="/api", tags=["Inventory"])
app.include_router(dashboard_routes.router, prefix="/api", tags=["Dashboard"])
app.include_router(product_routes.router, prefix="/api", tags=["Products"])

@app.get("/")
def health_check():
    return {"status": "API Gateway is running!"}
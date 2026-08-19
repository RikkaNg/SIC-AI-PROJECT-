# backend/src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.src.routes import chat_routes, forecast_routes, inventory_routes

app = FastAPI(title="Retail API Gateway", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký Routes
app.include_router(chat_routes.router, prefix="/api", tags=["Chat Assistant"])
app.include_router(forecast_routes.router, prefix="/api", tags=["Forecasting"])
app.include_router(inventory_routes.router, prefix="/api", tags=["Inventory"])

@app.get("/")
def health_check():
    return {"status": "API Gateway is running!"}
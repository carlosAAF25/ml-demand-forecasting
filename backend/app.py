# backend/app.py
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="Agrochemical Demand Forecasting API",
    description="API for weekly multi-branch agrochemical demand forecasting.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_BRANCHES = ["Sucursal_Norte", "Sucursal_Sur", "Sucursal_Central"]
VALID_PRODUCTS = ["PROD_GLIFO", "PROD_PARAQUAT", "PROD_ATRAZINA", "PROD_CLORPI", "PROD_MANCO", "PROD_UREA_FOL"]

try:
    model_bundle = joblib.load("demand_forecaster.joblib")
    pipeline = model_bundle["pipeline"]
    model_name = model_bundle.get("model_name", "RandomForest")
except Exception as exc:
    pipeline = None
    model_name = "None"
    print(f"Model artifact not loaded: {exc}")

class PredictionRequest(BaseModel):
    branch_id: str = Field(..., example="Sucursal_Norte")
    product_id: str = Field(..., example="PROD_GLIFO")
    product_category: str = Field(..., example="Herbicida")
    week_of_year: int = Field(..., ge=1, le=52, example=44)
    is_rainy_season: int = Field(..., ge=0, le=1, example=1)
    promo_applied: int = Field(..., ge=0, le=1, example=0)
    lag_1_demand: float = Field(..., ge=0.0, example=115.0)
    lag_2_demand: float = Field(..., ge=0.0, example=108.0)
    lag_4_demand: float = Field(..., ge=0.0, example=95.0)
    rolling_mean_4w: float = Field(..., ge=0.0, example=105.0)

class PredictionResponse(BaseModel):
    estimated_demand_liters: float
    status: str
    model_version: str

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": pipeline is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict_demand(payload: PredictionRequest):
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo predictivo no está cargado en el servidor."
        )

    if payload.branch_id not in VALID_BRANCHES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Sucursal no válida. Opciones: {VALID_BRANCHES}"
        )
    if payload.product_id not in VALID_PRODUCTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Producto no válido. Opciones: {VALID_PRODUCTS}"
        )

    input_data = pd.DataFrame([{
        "branch_id": payload.branch_id,
        "product_id": payload.product_id,
        "product_category": payload.product_category,
        "week_of_year": payload.week_of_year,
        "is_rainy_season": payload.is_rainy_season,
        "promo_applied": payload.promo_applied,
        "lag_1_demand": payload.lag_1_demand,
        "lag_2_demand": payload.lag_2_demand,
        "lag_4_demand": payload.lag_4_demand,
        "rolling_mean_4w": payload.rolling_mean_4w
    }])

    prediction = float(pipeline.predict(input_data)[0])
    prediction = max(0.0, round(prediction, 2))

    return PredictionResponse(
        estimated_demand_liters=prediction,
        status="success",
        model_version=f"{model_name}-v1"
    )
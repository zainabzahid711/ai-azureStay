from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from demand.routers import router as demand_router
from demand.models.prophet_model import DemandProphet
import logging

# Initialize FastAPI app
app = FastAPI(title="AzureStay AI Services")

# Include routers
app.include_router(demand_router)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

# This makes the app variable available for ASGI
__all__ = ["app"]

router = APIRouter(prefix="/api/demand", tags=["demand"])
logger = logging.getLogger(__name__)

class ForecastRequest(BaseModel):
    booking_dates: List[str]
    room_capacity: int = 10

@router.post("/forecast")
async def get_forecast(request: ForecastRequest):
    """Simplified and reliable forecast endpoint"""
    try:
        prophet = DemandProphet()
        return {
            "forecast": prophet.forecast(
                booking_dates=request.booking_dates,
                room_capacity=request.room_capacity
            )
        }
    except Exception as e:
        logger.error(f"API Error: {e}")
        raise HTTPException(
            status_code=400,
            detail="Forecast service is temporarily unavailable"
        )
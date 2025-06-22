from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from demand.models.prophet_model import DemandProphet
import logging

router = APIRouter(prefix="/api/demand", tags=["demand"])
logger = logging.getLogger(__name__)

class ForecastRequest(BaseModel):
    booking_dates: List[str]
    room_capacity: int = 10

@router.post("/forecast")
async def get_forecast(request: ForecastRequest):
    try:
        prophet = DemandProphet()
        forecast = prophet.forecast(
            booking_dates=request.booking_dates,
            room_capacity=request.room_capacity
        )
        return {"forecast": forecast}
    except Exception as e:
        logger.error(f"Forecast failed: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail="Could not generate forecast. Please check your input data."
        )
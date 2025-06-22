from fastapi import APIRouter, HTTPException
from .schemas import ForecastRequest
from .services import DemandService

router = APIRouter(prefix="/demand", tags=["demand"])
service = DemandService()

@router.post("/forecast")
async def demand_forecast(request: ForecastRequest):
    try:
        return await service.get_forecast(
            room_type=request.room_type,
            booking_dates=request.booking_dates
        )
    except Exception as e:
        raise HTTPException(500, f"Forecast failed: {str(e)}")
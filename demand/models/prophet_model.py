import pandas as pd
from prophet import Prophet
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# new_model=Prophet()
# new_model.fit(data)
# forcast = new_model.make_future_dataframe(periods=...)

app = FastAPI()

# Add CORS middleware at the application level
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1337, *"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class ForecastRequest(BaseModel):
    room_type: str
    booking_dates: List[str]

class ForecastItem(BaseModel):
    date: str
    demand: float
    confidence_min: Optional[float] = None
    confidence_max: Optional[float] = None

class ForecastResponse(BaseModel):
    room_type: str
    forecast: List[ForecastItem]

class DemandProphet:
    def __init__(self):
         self.config = {
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'holidays': self._get_holidays(),
            'seasonality_mode': 'multiplicative'
        }
    
    
    def _get_holidays(self) -> pd.DataFrame:
        """Define hotel-relevant holidays"""
        holidays = pd.DataFrame({
            'holiday': 'peak_season',
            'ds': pd.to_datetime(['2025-07-04', '2025-12-25']),  # Add more dates
            'lower_window': -2,
            'upper_window': 2
        })
        return holidays

    def forecast(self, booking_dates: List[str], periods: int = 30) -> List[Dict]:
        """Generate demand forecast with confidence intervals"""
        # Convert and validate dates first
        try:
            df = pd.DataFrame({
                'ds': pd.to_datetime(booking_dates),
                'y': 1
            }).groupby('ds').sum().reset_index()
        except ValueError as e:
            raise ValueError(f"Invalid date format in booking dates: {str(e)}")
        
        if len(df) < 7:  # Minimum one week of data
            return []
            
       # Create NEW model instance for each forecast
        model = Prophet(**self.config)
        model.fit(df)
        future = model.make_future_dataframe(periods=periods)
        forecast = model.predict(future)
        
        
        return [{
            'date': row['ds'].date().isoformat(),
            'demand': round(float(row['yhat']), 2),
            'confidence_min': round(float(row['yhat_lower']), 2),
            'confidence_max': round(float(row['yhat_upper']), 2)
        } for _, row in forecast.tail(periods).iterrows()]

@app.post("/demand/forecast", response_model=ForecastResponse)
async def get_forecast(request: ForecastRequest):
    try:
        # Input validation
        if not request.booking_dates:
            return {
                "room_type": request.room_type,
                "forecast": []
            }
            
        # Validate date formats
        for date_str in request.booking_dates:
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD"
                )
        
        print(f"Received forecast request for room: {request.room_type}")
        print(f"Number of booking dates received: {len(request.booking_dates)}")
        
        prophet = DemandProphet()
        forecast_data = prophet.forecast(request.booking_dates)
        
        print(f"Generated forecast with {len(forecast_data)} items")
        return {
            "room_type": request.room_type,
            "forecast": forecast_data
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Forecast error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail={
                "message": "Forecast generation failed",
                "error": str(e)
            }
        )
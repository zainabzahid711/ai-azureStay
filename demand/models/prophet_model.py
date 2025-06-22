import pandas as pd
from prophet import Prophet
from typing import List, Dict

class DemandProphet:
    def __init__(self):
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            holidays=self._get_holidays(),
            seasonality_mode='multiplicative'
        )
    
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
        df = pd.DataFrame({
            'ds': pd.to_datetime(booking_dates),
            'y': 1
        }).groupby('ds').sum().reset_index()
        
        if len(df) < 7:  # Minimum one week of data
            return []
            
        self.model.fit(df)
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        
        return [{
            'date': row['ds'].date().isoformat(),
            'demand': round(float(row['yhat']), 2),
            'confidence_min': round(float(row['yhat_lower']), 2),
            'confidence_max': round(float(row['yhat_upper']), 2)
        } for _, row in forecast.tail(periods).iterrows()]
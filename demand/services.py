from .models.prophet_model import DemandProphet
from typing import List, Dict
from demand.models.prophet_model import DemandProphet

class DemandService:
    def __init__(self):
        self.forecaster = DemandProphet()
    
    async def get_forecast(self, room_type: str, booking_dates: List[str]) -> Dict:
        """Get forecast with room-type specific adjustments"""
        base_forecast = self.forecaster.forecast(booking_dates)
        
        # Apply room-type multipliers
        multipliers = {
            'presidential': 1.3,
            'executive': 1.1,
            'deluxe': 1.0
        }
        
        return {
            'room_type': room_type,
            'forecast': [
                {
                    **day,
                    'demand': round(day['demand'] * multipliers.get(room_type.lower(), 1.0), 2)
                } 
                for day in base_forecast
            ]
        }
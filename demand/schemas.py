from pydantic import BaseModel

class ForecastRequest(BaseModel):
    room_type: str
    booking_dates: list[str]
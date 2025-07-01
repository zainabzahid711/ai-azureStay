# models/prophet_train.py
from prophet import Prophet
from pathlib import Path
import os
import pandas as pd
import joblib



BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "hotel_bookings.csv"



if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")


# Time Series Forecasting
df = pd.read_csv(DATA_PATH, 
                 parse_dates=['startDate'] )
df['ds'] = pd.to_datetime(df['startDate'])
daily = df.groupby('ds').size().reset_index(name='y')

model = Prophet(weekly_seasonality=True)
model.fit(daily)



# Save model in same directory as script
MODEL_PATH = Path(__file__).parent / "prophet_forecaster.joblib"
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
from fastapi import FastAPI, HTTPException, Query
from datetime import datetime
import joblib
import pandas as pd
from pathlib import Path
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1337","*"],  # For development only
    allow_methods=["*"],
    allow_headers=["*"],
)


# ABSOLUTE PATH to models - CORRECTED
MODEL_DIR = Path("C:/Users/zaina/ai-azureStay/demand/models")

try:
    # Verify model directory exists
    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model directory not found at {MODEL_DIR}")

    # Verify model file exists
    clf_path = MODEL_DIR / "cancellation_predictor.joblib"
    if not clf_path.exists():
        available_files = "\n".join(os.listdir(MODEL_DIR))
        raise RuntimeError(f"Model file not found. Available files:\n{available_files}")

    # Load model
    clf_data = joblib.load(clf_path)
    clf = clf_data['model']
    room_mapping = {v: k for k, v in clf_data['room_mapping'].items()}

    print(f"✅ Successfully loaded model from {clf_path}")

except Exception as e:
    print(f"❌ Critical error loading model: {str(e)}")
    print(f"Current working directory: {os.getcwd()}")
    raise

@app.post("/predict-cancellation")
async def predict(booking: dict):
    try:
        input_data = pd.DataFrame([{
            'room_encoded': room_mapping[booking['room']],
            'guest': booking['guest'],
            'stay_duration': (datetime.strptime(booking['endDate'], "%Y-%m-%d") - 
                            datetime.strptime(booking['startDate'], "%Y-%m-%d")).days,
            'totalPrice': float(booking['totalPrice'])
        }])
        proba = clf.predict_proba(input_data)[0][1]
        return {
            "cancel_probability": round(proba, 4),
            "interpretation": "High risk" if proba > 0.7 else "Medium risk" if proba > 0.3 else "Low risk"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/forecast-demand")
async def forecast_demand(
    days_ahead: int = Query(30, description="Number of days to forecast"),
    frequency: str = Query("daily", description="Aggregation: 'daily', 'weekly', or 'monthly'"),
    top_n: int = Query(None, description="Return only top N peak days")
):
    try:
        prophet_model = joblib.load(MODEL_DIR / "prophet_forecaster.joblib")
        
        future = prophet_model.make_future_dataframe(periods=days_ahead)
        future['is_weekend'] = future['ds'].dt.day_name().isin(['Friday', 'Saturday', 'Sunday'])
        forecast = prophet_model.predict(future)
        
        future_forecast = forecast[forecast['ds'] > datetime.now()].copy()
        
        if frequency != "daily":
            df = future_forecast.set_index('ds').resample('W-Mon' if frequency == "weekly" else 'M').mean().reset_index()
        else:
            df = future_forecast
        
        if top_n:
            df = df.nlargest(top_n, 'yhat')
        
        return {
            "forecast_period": f"next {days_ahead} days",
            "aggregation": frequency,
            "max_capacity": 100,  # Replace with your actual max capacity
            "data": [
                {
                    "date": row['ds'].strftime('%Y-%m-%d'),
                    "day_of_week": row['ds'].strftime('%A'),
                    "forecasted_bookings": round(row['yhat'], 1),
                    "confidence_interval": {
                        "upper": round(row['yhat_upper'], 1),
                        "lower": round(row['yhat_lower'], 1)
                    },
                    "is_weekend": row['ds'].strftime('%A') in ['Friday', 'Saturday', 'Sunday']
                }
                for _, row in df.iterrows()
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
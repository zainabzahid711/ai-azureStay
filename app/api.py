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

# Initialize models as None at the top level
prophet_model = None
xgb_model = None
scaler = None
historical_data = None

# ABSOLUTE PATH to models
MODEL_DIR = Path("C:/Users/zaina/ai-azureStay/demand/models")



# ABSOLUTE PATH to models - CORRECTED
MODEL_DIR = Path("C:/Users/zaina/ai-azureStay/demand/models")



def load_models():
    """Load all models at startup"""
    global prophet_model, xgb_model, scaler, historical_data
    
    try:
        # Load demand forecasting models
        prophet_model = joblib.load(MODEL_DIR / "prophet_forecaster.joblib")
        
        # Load XGBoost components - handle both cases (dictionary or raw model)
        xgb_data = joblib.load(MODEL_DIR / "xgb_residual_model.joblib")
        
        if isinstance(xgb_data, dict):  # If saved as dictionary
            xgb_model = xgb_data['model']
            scaler = xgb_data.get('scaler')
        else:  # If saved as raw model
            xgb_model = xgb_data
            scaler = joblib.load(MODEL_DIR / "feature_scaler.joblib")
            
        historical_data = pd.read_pickle(MODEL_DIR / "historical_data.pkl")
        print("✅ All models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {str(e)}")
        raise

# Load models when starting up
load_models()

# Load cancellation prediction model
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

    print(f"✅ Successfully loaded cancellation model from {clf_path}")

except Exception as e:
    print(f"❌ Critical error loading cancellation model: {str(e)}")
    print(f"Current working directory: {os.getcwd()}")
    raise

# Load demand forecasting models
try:
    prophet_train = joblib.load(MODEL_DIR / "prophet_forecaster.joblib")
    xgb_model = joblib.load(MODEL_DIR / "xgb_residual_model.joblib")
    historical_data = pd.read_pickle(MODEL_DIR / "historical_data.pkl")
    print("✅ Successfully loaded demand forecasting models")
except Exception as e:
    print(f"❌ Error loading demand forecasting models: {str(e)}")
    # You might want to handle this differently if demand forecasting is optional

def create_future_features(future_dates):
    """Generate features for future dates based on historical patterns"""

    global historical_data

    if historical_data is None:
        raise ValueError("Historical data not loaded")
    # Convert to DataFrame if needed
    if not isinstance(future_dates, pd.DataFrame):
        future = pd.DataFrame({'ds': future_dates})
    else:
        future = future_dates.copy()
    
    # Basic temporal features
    future['month'] = future['ds'].dt.month
    future['day_of_week_num'] = future['ds'].dt.day_name()
    future['is_weekend'] = future['ds'].dt.dayofweek.isin([4,5,6]).astype(int)

    # Room type features - using historical averages
    room_cols = [col for col in historical_data.columns if col.startswith('room_') and not col.endswith('_7day_avg')]
    
    for room in room_cols:
        # Use the average value for that day of week in historical data
        day_avg = historical_data.groupby(historical_data.index.day_name())[room].mean()
        future[room] = future['day_of_week_num'].map(day_avg)
        
        # Calculate rolling averages (using last available historical data)
        last_7days = historical_data[room].iloc[-7:].mean()
        future[f'{room}_7day_avg'] = last_7days
    
    future = future.drop(columns=['day_of_week_num'], errors='ignore')
    
    room_features = [col for col in future.columns if col.startswith('room_')]
    return future[['ds'] + room_features]

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
    days_ahead: int = Query(30, gt=1, le=365, description="Number of days to forecast"),
    top_n: int = Query(None, ge=1, description="Return only top N peak days"),
    frequency: str = Query("daily", description="Aggregation: 'daily', 'weekly', or 'monthly'"),
):
    try:
        if prophet_model is None or xgb_model is None or historical_data is None:
            raise HTTPException(status_code=503, detail="Models not loaded")

        # Create future dataframe
        future = prophet_model.make_future_dataframe(periods=days_ahead)
        
        # Generate features for future dates
        future_features = create_future_features(future)
        
        # Prophet prediction
        prophet_forecast = prophet_model.predict(future_features)

        # Prepare features for XGBoost
        if isinstance(xgb_model, dict):  # If model was saved as dictionary
            xgb = xgb_model['model']
            scaler = xgb_model.get('scaler')
            feature_names = xgb_model.get('feature_names', [])
        else:  # If model was saved directly
            xgb = xgb_model
            scaler = None
            # Try to get feature names from booster
            try:
                feature_names = xgb.get_booster().feature_names
            except:
                feature_names = historical_data.columns.tolist()

        # Select only the features the model expects
        xgb_input = future_features.set_index('ds')[feature_names].copy()

        # Scale if we have a scaler
        if scaler:
            xgb_input = scaler.transform(xgb_input)
        
        # Get residuals prediction
        residuals = xgb.predict(xgb_input)
        
        # Combined prediction
        future['yhat'] = prophet_forecast['yhat'] + residuals
        future['yhat_upper'] = prophet_forecast['yhat_upper'] + residuals
        future['yhat_lower'] = prophet_forecast['yhat_lower'] + residuals
        
        # Filter future dates
        future_forecast = future[future['ds'] > datetime.now()].copy()
        
        # Apply frequency aggregation
        if frequency != "daily":
            freq = 'W-Mon' if frequency == "weekly" else 'M'
            df = future_forecast.set_index('ds').resample(freq).mean().reset_index()
        else:
            df = future_forecast
        
        # Apply top_n filter if requested
        if top_n:
            df = df.nlargest(top_n, 'yhat')
        
        # Prepare response
        return {
            "forecast_period": f"next {days_ahead} days",
            "aggregation": frequency,
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
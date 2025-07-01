from fastapi import FastAPI, HTTPException
from datetime import datetime
import joblib
import pandas as pd
from pathlib import Path
import os

app = FastAPI()

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
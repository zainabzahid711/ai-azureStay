from prophet import Prophet
from pathlib import Path
import pandas as pd
import joblib
from prophet.plot import plot_components
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "hotel_bookings.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

# Load and prepare data
df = pd.read_csv(DATA_PATH, parse_dates=['startDate'])
df['ds'] = pd.to_datetime(df['startDate'])
daily = df.groupby('ds').size().reset_index(name='y')

# Analyze historical patterns
print("\n=== Data Analysis ===")
print("Most common booking days:")
print(df['ds'].dt.day_name().value_counts())
print("\nTop 10 booking dates:")
print(daily.sort_values('y', ascending=False).head(10))
print(f"\nMaximum daily bookings: {daily['y'].max()}")

# Configure Prophet with custom seasonality
model = Prophet(
    weekly_seasonality=True,
    seasonality_prior_scale=0.1,
    yearly_seasonality=True,
    daily_seasonality=False
)

# Add custom weekend seasonality
model.add_seasonality(
    name='weekend',
    period=7,
    fourier_order=3,
    condition_name='is_weekend'
)

# Prepare data with weekend indicator
daily['is_weekend'] = daily['ds'].dt.day_name().isin(['Friday', 'Saturday', 'Sunday'])

# Fit model
model.fit(daily)

# Generate forecast for diagnostics
future = model.make_future_dataframe(periods=365)
future['is_weekend'] = future['ds'].dt.day_name().isin(['Friday', 'Saturday', 'Sunday'])
forecast = model.predict(future)

# Save diagnostic plots
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
fig1.savefig('forecast_plot.png')
fig2.savefig('components_plot.png')

# Save model
MODEL_PATH = Path(__file__).parent / "prophet_forecaster.joblib"
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
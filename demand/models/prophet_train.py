import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def prepare_features(df):
    """Feature engineering pipeline"""
    df['stay_duration'] = (df['endDate'] - df['startDate']).dt.days
    df['cancelled'] = (df['status'] == 'canceled').astype(int)
    
    # Temporal features
    df['month'] = df['startDate'].dt.month
    df['day_of_week'] = df['startDate'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin(['Friday', 'Saturday', 'Sunday'])
    
    # Room type demand features
    room_demand = df.groupby(['startDate', 'room']).size().unstack().fillna(0)
    room_demand.columns = [f"room_{col.lower().replace(' ','_')}" for col in room_demand.columns]
    
    # Create target variable and merge features
    daily = df.groupby('startDate').size().rename('total_bookings').to_frame()
    daily = daily.join(room_demand)
    
    # Add rolling features with proper NaN handling
    for col in room_demand.columns:
        daily[f'{col}_7day_avg'] = daily[col].rolling(7, min_periods=1).mean().fillna(0)
    
    return daily

def train_prophet(daily):
    """Train Prophet model with room type regressors"""
    model = Prophet(
        yearly_seasonality=10,
        weekly_seasonality=15,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    # Add all room types as regressors
    for col in daily.columns:
        if col.startswith('room_'):
            model.add_regressor(col)
    
    prophet_df = daily.reset_index().rename(columns={'startDate': 'ds', 'total_bookings': 'y'})
    prophet_df = prophet_df.fillna(0)  # Ensure no NaN values
    
    model.fit(prophet_df)
    return model, prophet_df

def train_xgboost_residuals(prophet_df, prophet_train):
    """Train XGBoost to predict Prophet's residuals"""
    # Get Prophet predictions
    forecast = prophet_train.predict(prophet_df)
    prophet_pred = forecast.set_index('ds')['yhat']
    
    # Calculate residuals
    residuals = prophet_df.set_index('ds')['y'] - prophet_pred
    
    # Prepare features for XGBoost - ensure we keep feature names
    room_features = sorted([col for col in prophet_df.columns if col.startswith('room_') or col.endswith('_7day_avg')])
    X = prophet_df.set_index('ds')[room_features]
    y = residuals
    
    # Train-test split
    split_date = prophet_df['ds'].quantile(0.8)
    X_train, X_test = X[X.index <= split_date], X[X.index > split_date]
    y_train, y_test = y[y.index <= split_date], y[y.index > split_date]
    
    # Get feature names before scaling
    feature_names = X_train.columns.tolist()
    
    # Create separate scaler to preserve feature names
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), 
                                columns=feature_names, 
                                index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), 
                               columns=feature_names, 
                               index=X_test.index)
    
    # Train XGBoost directly (not in pipeline) to preserve feature names
    xgb_model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        max_depth=5,
        early_stopping_rounds=20
    )

    xgb_model.feature_names = feature_names  # Store feature names in model
    xgb_model.get_booster().feature_names = feature_names  # Ensure booster has feature names
    
    # Fit with validation set
    xgb_model.fit(X_train_scaled, y_train,
                 eval_set=[(X_test_scaled, y_test)],
                 verbose=True)
    
    # Return both the model and scaler
    return {'model': xgb_model, 'scaler': scaler,
            'feature_names': feature_names, 'feature_order': feature_names}

def save_models(prophet_train, xgb_dict, prophet_df):
    """Save trained models"""
    models_dir = Path(__file__).parent
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(prophet_train, models_dir / "prophet_forecaster.joblib")
    joblib.dump({
        'model': xgb_dict['model'],
        'scaler': xgb_dict['scaler'],
        'feature_names': xgb_dict['feature_names']
    }, models_dir / "xgb_residual_model.joblib")
    joblib.dump(xgb_dict['scaler'], models_dir / "feature_scaler.joblib")
    # joblib.dump(xgb_model, models_dir / "xgb_residual_model.joblib")
    prophet_df.set_index('ds').to_pickle(models_dir / "historical_data.pkl")

if __name__ == "__main__":
    try:
        # Load data
        data_path = Path(__file__).parent.parent.parent / "data" / "hotel_bookings.csv"
        print(f"Loading data from: {data_path}")
        
        df = pd.read_csv(data_path, parse_dates=['startDate', 'endDate'])
        df['day_of_week_num'] = df['startDate'].dt.dayofweek
        print("Data loaded successfully. Sample:")
        print(df.head())
        
        # Feature engineering
        daily = prepare_features(df)
        print("\nFeatures prepared. Sample:")
        print(daily.head())
        
        # Train models
        prophet_train, prophet_df = train_prophet(daily)
        print("\nProphet model trained successfully")
        
        xgb_dict = train_xgboost_residuals(prophet_df, prophet_train)
        print("XGBoost model trained successfully")
        
        # Save models
        save_models(prophet_train, xgb_dict, prophet_df)
        print("Models saved successfully")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise
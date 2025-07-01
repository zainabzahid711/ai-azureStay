# models/cancellation_train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import joblib
from pathlib import Path


# Get absolute path to data
BASE_DIR = Path(__file__).parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "hotel_bookings.csv"


# Classification Model
df = pd.read_csv(DATA_PATH)
df['stay_duration'] = (pd.to_datetime(df['endDate']) - pd.to_datetime(df['startDate'])).dt.days
df['is_canceled'] = (df['status'] == 'canceled').astype(int)
df['room_encoded'] = df['room'].astype('category').cat.codes

X = df[['room_encoded', 'guest', 'stay_duration', 'totalPrice']]
y = df['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))

MODEL_PATH = Path(__file__).parent / "cancellation_predictor.joblib"
joblib.dump({
    'model': clf,
    'room_mapping': dict(enumerate(df['room'].astype('category').cat.categories)),
    'feature_names': list(X.columns)
}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

joblib.dump({
    'model': clf,
    'room_mapping': dict(enumerate(df['room'].astype('category').cat.categories)),
    'feature_names': list(X.columns)
}, "cancellation_predictor.joblib")  # Now includes metadata
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests
import os
from typing import Dict, Any

app = FastAPI(title="Prediction Module")
DATA_MANAGER_URL = os.environ.get("DATA_MANAGER_URL", "http://localhost:8002")
MODEL_DIR = "/app/models"
os.makedirs(MODEL_DIR, exist_ok=True)

class TrainRequest(BaseModel):
    user_id: str
    habit_name: str

class PredictionRequest(BaseModel):
    user_id: str
    habit_name: str
    day_of_week: int
    hour: int

def get_model_path(user_id: str, habit_name: str) -> str:
    return os.path.join(MODEL_DIR, f"model_{user_id}_{habit_name}.joblib")

def get_scaler_path(user_id: str, habit_name: str) -> str:
    return os.path.join(MODEL_DIR, f"scaler_{user_id}_{habit_name}.joblib")

def get_user_data_from_manager(user_id: str) -> pd.DataFrame:
    """Fetches user data from the user-data-manager service."""
    try:
        response = requests.get(f"{DATA_MANAGER_URL}/get_data/{user_id}")
        response.raise_for_status()
        data = response.json().get("data", [])
        if not data:
            return pd.DataFrame(columns=["day_of_week", "hour", "habit_name", "done"])
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error fetching data from manager: {e}")
        return pd.DataFrame(columns=["day_of_week", "hour", "habit_name", "done"])

# API Endpoints
@app.post("/train")
async def train_user_model(req: TrainRequest):
    """Trains and saves a predictive model for a specific user and habit."""
    print(f"Attempting to train model for {req.user_id} on {req.habit_name}...")
    
    df = get_user_data_from_manager(req.user_id)
    if df.empty:
        return {"status": "failed", "message": "No data available to train."}

    habit_df = df[df["habit_name"] == req.habit_name].copy()
    
    if len(habit_df) < 5 or habit_df["done"].nunique() < 2:
        return {"status": "failed", "message": f"Not enough diverse data for {req.habit_name}."}

    features = habit_df[["day_of_week", "hour"]]
    labels = habit_df["done"]
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    
    model = LogisticRegression(random_state=42, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler to the persistent volume
    joblib.dump(model, get_model_path(req.user_id, req.habit_name))
    joblib.dump(scaler, get_scaler_path(req.user_id, req.habit_name))
    
    print(f"Successfully trained model for {req.user_id} on {req.habit_name}.")
    return {"status": "success", "message": "Model trained successfully."}

@app.post("/predict")
async def predict_habit(req: PredictionRequest):
    """Predicts the likelihood of a user completing a habit."""
    model_path = get_model_path(req.user_id, req.habit_name)
    scaler_path = get_scaler_path(req.user_id, req.habit_name)

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return {"prediction": None, "probability": 0.5, "message": "No model trained yet."}

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        feature_df = pd.DataFrame([[req.day_of_week, req.hour]], columns=["day_of_week", "hour"])
        features_scaled = scaler.transform(feature_df)
        
        probability = model.predict_proba(features_scaled)[0][1]
        prediction = 1 if probability > 0.5 else 0
        
        return {
            "prediction": "done" if prediction == 1 else "skip",
            "probability": f"{probability:.2f}",
            "message": "Prediction successful."
        }
    except Exception as e:
        return {"status": "error", "message": f"Prediction failed: {e}"}

@app.get("/")
def read_root():
    return {"message": "Prediction Module is running."}

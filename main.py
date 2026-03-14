import os
import joblib
import numpy as np
import datetime
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="OncoAI API", version="2026.1")

# CORS Setup - Flutter ke liye zaroori hai
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model Loading Logic (Safe Loading)
model = None
scaler = None

# Files ke naam check karlein ke 'stable_model.pkl' hi hain
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, 'stable_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'stable_scaler.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Model and Scaler loaded successfully from path!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

class PatientInput(BaseModel):
    age: float
    gender: int
    smoking: float
    radon: float
    asbestos: float
    alcohol: float

@app.get("/")
def read_root():
    return {
        "status": "Online", 
        "server_time": str(datetime.datetime.now()),
        "message": "Welcome to OncoAI Prediction API"
    }

@app.post("/predict")
async def predict(data: PatientInput):
    if model is None or scaler is None:
        return {"error": "Model not loaded on server"}
        
    try:
        # Features array banana
        features = np.array([[
            data.age, data.gender, data.smoking, 
            data.radon, data.asbestos, data.alcohol
        ]])
        
        # Scaling
        scaled_features = scaler.transform(features)
        
        # Prediction
        prob = float(model.predict_proba(scaled_features)[0][1])
        
        # Logic match with Flutter (0.55 threshold)
        risk_level = "High Risk" if prob >= 0.55 else "Low Risk"
        
        return {
            "risk": risk_level,
            "probability": round(prob * 100, 2),
            "confidence_score": prob,
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Render port automatically handle karta hai, local ke liye 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
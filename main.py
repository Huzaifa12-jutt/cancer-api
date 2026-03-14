import os
import joblib
import numpy as np
import uvicorn
import xgboost  # <--- Ye lazmi hai XGBoost model ke liye
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None

# Model loading logic
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(BASE_DIR, 'stable_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'stable_scaler.pkl'))
    print("✅ Model & Scaler Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

class PatientInput(BaseModel):
    age: float
    gender: int
    smoking: float
    radon: float
    asbestos: float
    alcohol: float

@app.post("/predict")
async def predict(data: PatientInput):
    if model is None:
        return {"error": "Model not loaded. Check Railway logs for XGBoost error."}
    
    try:
        features = np.array([[data.age, data.gender, data.smoking, data.radon, data.asbestos, data.alcohol]])
        
        # Scaling
        scaled_features = scaler.transform(features)
        
        # Prediction
        prob = float(model.predict_proba(scaled_features)[0][1])
        
        return {
            "risk": "High Risk" if prob >= 0.55 else "Low Risk",
            "probability": round(prob * 100, 2),
            "confidence_score": prob
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

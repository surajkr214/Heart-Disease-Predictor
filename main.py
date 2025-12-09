from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Heart Disease API", description="Production API")

# ==========================================
# LOAD YOUR EXISTING MODEL
# ==========================================
# We use os.path to safely find the file in Docker/Render
curr_dir = os.path.dirname(os.path.realpath(__file__))

# CHANGED: Now pointing to your original files
model_path = os.path.join(curr_dir, 'knn_model.pkl')
scaler_path = os.path.join(curr_dir, 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print("✅ Original Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Did you forget to upload knn_model.pkl?")

# ==========================================
# INPUT DATA STRUCTURE
# ==========================================
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    return {"status": "Active", "model": "Using User's Custom Model"}

@app.post("/predict")
def predict(data: PatientData):
    # Convert input to 2D array
    input_data = np.array([[
        data.age, data.sex, data.cp, data.trestbps, data.chol, 
        data.fbs, data.restecg, data.thalach, data.exang, 
        data.oldpeak, data.slope, data.ca, data.thal
    ]])
    
    # Scale and Predict
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)
    
    class_name = "Disease Detected" if prediction[0] == 1 else "Healthy"
    # Get probability of the predicted class
    conf = probability[0][1] if prediction[0] == 1 else probability[0][0]
    
    return {"prediction": class_name, "confidence": float(conf)}
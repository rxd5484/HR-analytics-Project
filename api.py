"""
FastAPI Application for Employee Attrition Prediction
Serves ML models via REST API
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="Employee Attrition Prediction API",
    description="ML-powered API for predicting employee turnover risk",
    version="1.0.0"
)

# Load models
try:
    xgb_model = joblib.load('models/xgboost.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    print("✅ Models loaded successfully")
except:
    print("⚠️ Models not found. Run train_models.py first.")
    xgb_model = None

# Request model
class EmployeeData(BaseModel):
    Age: int
    Gender: str
    MaritalStatus: str
    Education: str
    Department: str
    JobRole: str
    YearsAtCompany: float
    YearsInCurrentRole: float
    YearsSinceLastPromotion: int
    MonthlyIncome: int
    JobSatisfaction: int
    EnvironmentSatisfaction: int
    RelationshipSatisfaction: int
    WorkLifeBalance: int
    PerformanceRating: int
    OverTime: str
    DistanceFromHome: float
    NumCompaniesWorked: int
    TrainingTimesLastYear: int

# Response model
class PredictionResponse(BaseModel):
    attrition_risk: float
    risk_level: str
    confidence: float
    top_risk_factors: List[str]

@app.get("/")
def root():
    return {
        "message": "Employee Attrition Prediction API",
        "version": "1.0.0",
        "endpoints": [
            "/predict",
            "/predict/batch",
            "/health"
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_attrition(employee: EmployeeData):
    """Predict attrition risk for a single employee"""
    
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dataframe
        data = employee.dict()
        df = pd.DataFrame([data])
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform(df[col])
        
        # Calculate derived features
        df['TenureToIncomeRatio'] = df['YearsAtCompany'] / (df['MonthlyIncome'] / 1000)
        df['SatisfactionScore'] = (
            df['JobSatisfaction'] + 
            df['EnvironmentSatisfaction'] + 
            df['RelationshipSatisfaction'] + 
            df['WorkLifeBalance']
        ) / 4
        
        # Scale features
        X = scaler.transform(df)
        
        # Predict
        attrition_prob = xgb_model.predict_proba(X)[0][1]
        
        # Determine risk level
        if attrition_prob >= 0.7:
            risk_level = "HIGH"
        elif attrition_prob >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Identify top risk factors
        risk_factors = []
        if employee.JobSatisfaction <= 2:
            risk_factors.append("Low job satisfaction")
        if employee.WorkLifeBalance <= 2:
            risk_factors.append("Poor work-life balance")
        if employee.YearsAtCompany < 2:
            risk_factors.append("Short tenure (< 2 years)")
        if employee.OverTime == "Yes":
            risk_factors.append("Frequent overtime")
        if employee.YearsSinceLastPromotion > 4:
            risk_factors.append("No recent promotion (> 4 years)")
        if not risk_factors:
            risk_factors.append("Multiple minor factors")
        
        return {
            "attrition_risk": round(float(attrition_prob), 3),
            "risk_level": risk_level,
            "confidence": round(max(attrition_prob, 1 - attrition_prob), 3),
            "top_risk_factors": risk_factors[:3]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": xgb_model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
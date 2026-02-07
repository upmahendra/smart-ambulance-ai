from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np

app = FastAPI(title="Smart Ambulance ML Service")

WINDOW = 60

# -------------------------
# Input schema (STRICT)
# -------------------------
class VitalInput(BaseModel):
    hr: List[float]
    spo2: List[float]
    bp_sys: List[float]
    motion: List[float]

# -------------------------
# Root endpoint (health)
# -------------------------
@app.get("/")
def root():
    return {"status": "Smart Ambulance ML API running"}

# -------------------------
# Analyze endpoint
# -------------------------
@app.post("/analyze")
def analyze_vitals(data: VitalInput):
    try:
        # Ensure enough data
        if len(data.hr) < WINDOW:
            return {
                "anomaly": False,
                "risk_score": 0.0,
                "confidence": 0.0,
                "error": "Not enough data (need 60 samples)"
            }

        # Convert to numpy arrays
        hr = np.array(data.hr[-WINDOW:], dtype=float)
        spo2 = np.array(data.spo2[-WINDOW:], dtype=float)
        bp = np.array(data.bp_sys[-WINDOW:], dtype=float)
        motion = np.array(data.motion[-WINDOW:], dtype=float)

        # Cleaning (median over window)
        hr_clean = float(np.median(hr))
        spo2_clean = float(np.median(spo2))
        bp_clean = float(np.median(bp))

        # Risk components
        hr_risk = max(0.0, (hr_clean - 100.0) / 40.0)
        spo2_risk = max(0.0, (94.0 - spo2_clean) / 10.0)
        bp_risk = max(0.0, (100.0 - bp_clean) / 40.0)

        risk_score = min(
            0.4 * hr_risk +
            0.4 * spo2_risk +
            0.2 * bp_risk,
            1.0
        )

        # Confidence (motion-aware)
        motion_ratio = float(np.mean(motion > 1.5))
        confidence = max(0.0, 1.0 - motion_ratio)

        anomaly = (risk_score > 0.6) and (confidence > 0.6)

        return {
            "anomaly": anomaly,
            "risk_score": round(risk_score, 2),
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        # Safe debug response (NO crash)
        return {
            "anomaly": False,
            "risk_score": 0.0,
            "confidence": 0.0,
            "error": str(e)
        }

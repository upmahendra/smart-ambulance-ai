import pandas as pd
import numpy as np

df = pd.read_csv("data/patient_with_anomalies.csv")

WINDOW = 60

risk_scores = []
confidence_scores = []
alerts = []

for i in range(len(df)):
    if i < WINDOW:
        risk_scores.append(0)
        confidence_scores.append(1)
        alerts.append(False)
        continue

    window = df.iloc[i-WINDOW:i]

    # Risk components
    hr_risk = max(0, (window["hr_clean"].iloc[-1] - 100) / 40)
    spo2_risk = max(0, (94 - window["spo2_clean"].iloc[-1]) / 10)
    bp_risk = max(0, (100 - window["bp_sys_clean"].iloc[-1]) / 40)

    risk = 0.4 * hr_risk + 0.4 * spo2_risk + 0.2 * bp_risk
    risk = min(risk, 1.0)

    # Confidence (motion-aware)
    motion_ratio = (window["motion"] > 1.5).mean()
    confidence = max(0, 1 - motion_ratio)

    # Alert logic
    alert = (risk > 0.65) and (confidence > 0.6)

    risk_scores.append(risk)
    confidence_scores.append(confidence)
    alerts.append(alert)

df["risk_score"] = risk_scores
df["confidence"] = confidence_scores
df["alert"] = alerts

print("Total alerts triggered:", df["alert"].sum())

df.to_csv("data/patient_with_risk.csv", index=False)

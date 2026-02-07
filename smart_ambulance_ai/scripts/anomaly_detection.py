import pandas as pd
import numpy as np
from scipy.stats import zscore

# Load cleaned data
df = pd.read_csv("data/patient_001.csv")

# Recompute cleaned signals (simple import for now)
df["spo2_clean"] = df["spo2"].rolling(5, min_periods=1).median()
df["hr_clean"] = df["hr"].rolling(5, min_periods=1).median()
df["bp_sys_clean"] = df["bp_sys"].rolling(5, min_periods=1).median()

WINDOW = 60  # seconds

anomaly_scores = []

for i in range(WINDOW, len(df)):
    window = df.iloc[i-WINDOW:i]

    # Trends (slope approximation)
    hr_trend = window["hr_clean"].iloc[-1] - window["hr_clean"].iloc[0]
    spo2_trend = window["spo2_clean"].iloc[-1] - window["spo2_clean"].iloc[0]
    bp_trend = window["bp_sys_clean"].iloc[-1] - window["bp_sys_clean"].iloc[0]

    # Normalize risk
    hr_risk = max(0, hr_trend / 20)
    spo2_risk = max(0, -spo2_trend / 5)
    bp_risk = max(0, -bp_trend / 20)

    score = 0.4 * hr_risk + 0.4 * spo2_risk + 0.2 * bp_risk
    anomaly_scores.append(score)

# Pad initial window
anomaly_scores = [0]*WINDOW + anomaly_scores
df["anomaly_score"] = anomaly_scores
df["anomaly_flag"] = df["anomaly_score"] > 0.6

# Summary
print("Total anomaly points detected:", df["anomaly_flag"].sum())

# Save for next steps
df.to_csv("data/patient_with_anomalies.csv", index=False)

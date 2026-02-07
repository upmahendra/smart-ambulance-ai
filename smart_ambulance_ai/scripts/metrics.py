import pandas as pd
import numpy as np

df = pd.read_csv("data/patient_with_risk.csv")

# Define ground truth distress window
df["ground_truth"] = False
df.loc[600:1500, "ground_truth"] = True

# Confusion matrix components
TP = ((df["alert"]) & (df["ground_truth"])).sum()
FP = ((df["alert"]) & (~df["ground_truth"])).sum()
FN = ((~df["alert"]) & (df["ground_truth"])).sum()

precision = TP / (TP + FP + 1e-6)
recall = TP / (TP + FN + 1e-6)

false_alert_rate = FP / (len(df) / 60)  # per minute

# Alert latency
alert_times = df.index[df["alert"] & df["ground_truth"]]
latency = alert_times.min() - 600 if len(alert_times) > 0 else None

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"False Alert Rate (per min): {false_alert_rate:.2f}")
print(f"Alert Latency (sec): {latency}")

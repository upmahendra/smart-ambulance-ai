import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/patient_001.csv")

# ----------------------------
# Artifact flags
# ----------------------------
df["spo2_artifact"] = False
df["hr_artifact"] = False
df["bp_artifact"] = False

# ----------------------------
# 1. Motion-induced SpO2 drops
# ----------------------------
motion_threshold = 1.5
spo2_drop_threshold = 3

spo2_diff = df["spo2"].diff()

df.loc[
    (df["motion"] > motion_threshold) &
    (spo2_diff < -spo2_drop_threshold),
    "spo2_artifact"
] = True

# ----------------------------
# 2. Short HR spikes (bumps)
# ----------------------------
hr_diff = df["hr"].diff()

spike_indices = df.index[hr_diff > 20]

for idx in spike_indices:
    df.loc[idx:idx+3, "hr_artifact"] = True

# ----------------------------
# 3. BP flatline detection
# ----------------------------
bp_diff = df["bp_sys"].diff().abs()
df.loc[bp_diff < 0.1, "bp_artifact"] = True

# ----------------------------
# Cleaning logic
# ----------------------------
df["spo2_clean"] = df["spo2"].mask(df["spo2_artifact"]).interpolate()
df["hr_clean"] = (
    df["hr"]
    .mask(df["hr_artifact"])
    .rolling(5, min_periods=1)
    .median()
)
df["bp_sys_clean"] = df["bp_sys"].mask(df["bp_artifact"]).interpolate()

# ----------------------------
# Plot before vs after
# ----------------------------
plt.figure(figsize=(14, 8))

plt.subplot(3, 1, 1)
plt.plot(df["spo2"], label="Raw SpO2", alpha=0.4)
plt.plot(df["spo2_clean"], label="Cleaned SpO2")
plt.legend()
plt.title("SpO2 Artifact Handling")

plt.subplot(3, 1, 2)
plt.plot(df["hr"], label="Raw HR", alpha=0.4)
plt.plot(df["hr_clean"], label="Cleaned HR")
plt.legend()
plt.title("HR Artifact Handling")

plt.subplot(3, 1, 3)
plt.plot(df["bp_sys"], label="Raw BP Sys", alpha=0.4)
plt.plot(df["bp_sys_clean"], label="Cleaned BP Sys")
plt.legend()
plt.title("BP Artifact Handling")

plt.tight_layout()
plt.show()

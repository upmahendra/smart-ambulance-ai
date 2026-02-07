import numpy as np
import pandas as pd

np.random.seed(42)

SECONDS = 1800  # 30 minutes
t = np.arange(SECONDS)

# -------------------
# Motion / Vibration
# -------------------
motion = np.random.normal(0.2, 0.05, SECONDS)

# Motion-heavy road segment
motion[1080:1320] += np.random.normal(1.2, 0.3, 240)

# Random bumps
for i in np.random.choice(SECONDS, 25, replace=False):
    motion[i:i+3] += np.random.uniform(1.5, 2.5)

motion = np.clip(motion, 0, None)

# -------------------
# Heart Rate (HR)
# -------------------
hr = 75 + np.random.normal(0, 1.5, SECONDS)

# Gradual deterioration
hr[600:1080] += np.linspace(0, 25, 480)

# Acute event
hr[1320:1500] += 35

# Motion-induced spikes
hr += motion * np.random.uniform(2, 4)

# -------------------
# SpO2
# -------------------
spo2 = 98 + np.random.normal(0, 0.3, SECONDS)

# Gradual hypoxia
spo2[600:1080] -= np.linspace(0, 6, 480)

# Acute drop
spo2[1320:1500] -= 8

# Motion artifacts (false drops)
spo2[motion > 1.5] -= np.random.uniform(4, 8)

spo2 = np.clip(spo2, 70, 100)

# -------------------
# Blood Pressure
# -------------------
bp_sys = 120 + np.random.normal(0, 3, SECONDS)
bp_dia = 80 + np.random.normal(0, 2, SECONDS)

# Shock-like drop
bp_sys[600:1080] -= np.linspace(0, 25, 480)
bp_dia[600:1080] -= np.linspace(0, 15, 480)

# Sensor flatline
bp_sys[1550:1580] = bp_sys[1549]
bp_dia[1550:1580] = bp_dia[1549]

# -------------------
# Build DataFrame
# -------------------
df = pd.DataFrame({
    "time_sec": t,
    "hr": hr,
    "spo2": spo2,
    "bp_sys": bp_sys,
    "bp_dia": bp_dia,
    "motion": motion
})

df.to_csv("data/patient_001.csv", index=False)
print("Synthetic patient data generated successfully.")

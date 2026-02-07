# Smart Ambulance AI – Real-Time Patient Monitoring & Decision Support System
## Overview

Smart Ambulance AI is a safety-critical, end-to-end Machine Learning system designed to monitor real-time patient vitals during ambulance transport and generate early warning alerts for medical deterioration.

The system simulates realistic physiological time-series data, handles sensor artifacts caused by vehicle motion, detects early anomalies beyond simple thresholds, and exposes predictions via a REST API built using FastAPI.

This project was developed as part of an AI/ML Engineer Internship assignment, focusing on:
Noisy time-series data
Safety-critical ML reasoning
Engineering judgment
Reproducible ML systems

## Problem Statement

### During ambulance transport, patient vitals are affected by:

Vehicle motion and vibrations
Sensor dropouts and artifacts
Delayed clinical interpretation

### This system aims to:

Detect early deterioration, not just threshold breaches 
Reduce false alarms
Provide risk scores with confidence
Support paramedics and clinicians in real time

### Vitals Monitored

Heart Rate (HR)
SpO₂ (Oxygen Saturation)
Blood Pressure (Systolic)
Motion / Vibration Signal

### Data is sampled at 1-second resolution over a 30-minute transport window.

## System Workflow

### Synthetic Data Generation
Normal transport
Medical distress scenarios
Motion-induced sensor artifacts

### Artifact Detection & Cleaning
Motion-related SpO₂ drops
HR spikes due to road bumps
Missing data handling

### Anomaly Detection
Sliding window analysis
Trend-based deviation detection
Reduced false positives

### Risk Scoring
Multi-vital fusion
Trend severity weighting
Confidence estimation

### API Deployment
FastAPI REST service
JSON input → anomaly + risk score output

## Project Structure
smart_ambulance_ai/
│
├── api/
│   └── main.py                 # FastAPI service
│
├── data/
│   ├── patient_001.csv         # Raw synthetic data
│   ├── patient_with_anomalies.csv
│   └── patient_with_risk.csv
│
├── scripts/
│   ├── generate_data.py        # Synthetic data generator
│   ├── artifact_detection.py   # Artifact handling logic
│   ├── anomaly_detection.py    # Anomaly detection model
│   ├── risk_scoring.py         # Risk scoring & alert logic
│   └── metrics.py              # Evaluation metrics
│
├── requirements.txt
├── .gitignore
└── README.md

## Anomaly Detection & Risk Logic

### Anomaly Detection
Window-based statistical deviation
Trend-aware detection (early warning)
Artifact-aware suppression

### Risk Scoring
Combines:
HR trend
SpO₂ decline
BP deterioration
Motion confidence

### Outputs:
anomaly (True / False)
risk_score (0–1)
confidence (0–1)

## API Usage
### Start the API Server
uvicorn api.main:app --reload


### Open browser:

http://127.0.0.1:8000/docs

### Sample API Request
{
  "hr": [90, 95, 100, 110, 120],
  "spo2": [98, 97, 95, 92, 88],
  "bp_sys": [120, 115, 110, 105, 95],
  "motion": [0.2, 0.3, 0.4, 0.2, 0.1]
}

### Sample API Response
{
  "anomaly": true,
  "risk_score": 1,
  "confidence": 1
}

## Evaluation Metrics

Precision
Recall
False Alert Rate
Alert Latency

### In medical systems:

False negatives are more dangerous than false positives
Alert latency is critical during patient transport

### Safety-Critical Considerations

Explicit artifact handling before ML
Conservative alerting logic
Confidence-aware decision support
No full automation of medical decisions

This system is intended to assist clinicians, not replace them.

### Tech Stack

Python
NumPy, Pandas, SciPy
Matplotlib
FastAPI
Uvicorn
REST API (JSON)

### Author

## U P Mahendra
Bachelor of Engineering – Artificial Intelligence & Machine Learning

Project developed as part of a Real-World AI/ML Internship Assignment focused on safety-critical systems.

### Final Notes

This project emphasizes:
ML reasoning over raw accuracy
Robust handling of noisy real-world data
Clean, reproducible engineering practices
Responsible medical AI design

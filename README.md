# GlucoTwin: AI-Powered Digital Twin for Early Prediction of Glucose Spikes

## Project Overview

GlucoTwin is an AI-powered healthcare Digital Twin prototype designed to simulate a patient's health state by combining historical Electronic Health Record (EHR) data with dynamic wearable and IoT data.

The system uses machine learning to estimate the likelihood of a glucose spike approximately 2 hours ahead.

This project is developed as a research prototype for the Happiest Health Digital Twin Challenge 2026.

**Disclaimer:** GlucoTwin is a research prototype using synthetic data. It is not intended for medical diagnosis, treatment, or clinical decision-making.



## Problem Statement

Glucose levels can change because of multiple factors such as:

- Current glucose level
- Physical activity
- Heart rate
- Heart-rate variability (HRV)
- Sleep duration
- HbA1c
- Diabetes history
- Medication
- Family history

GlucoTwin combines historical and dynamic health data into a simulated Digital Twin and applies machine learning to identify potential glucose spikes in advance.



## Proposed Solution

GlucoTwin follows a data-fusion approach:
text
Historical EHR Data
        +
Wearable / IoT Data
        |
        v
Data Preprocessing
        |
        v
Digital Twin Data Fusion
        |
        v
Feature Engineering
        |
        v
Random Forest Model
        |
        v
2-Hour Glucose Spike Prediction
        |
        v
Streamlit Dashboard
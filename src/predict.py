import pandas as pd
import joblib

# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

model = joblib.load(
    "models/glucose_prediction_model.pkl"
)

# ==========================================
# 2. SELECT A PATIENT
# ==========================================

ehr = pd.read_csv("data/ehr_data.csv")
wearable = pd.read_csv("data/wearable_data.csv")

patient_id = "P001"

# Get patient EHR information
patient_ehr = ehr[
    ehr["Patient_ID"] == patient_id
].iloc[0]

# Get latest wearable information
patient_wearable = wearable[
    wearable["Patient_ID"] == patient_id
].iloc[-1]

# ==========================================
# 3. PREPARE INPUT
# ==========================================

input_data = pd.DataFrame([{
    "Age": patient_ehr["Age"],
    "Diabetes": patient_ehr["Diabetes"],
    "HbA1c": patient_ehr["HbA1c"],
    "Previous_Glucose": patient_ehr["Previous_Glucose"],
    "Medication": patient_ehr["Medication"],
    "Family_History": patient_ehr["Family_History"],
    "Glucose": patient_wearable["Glucose"],
    "Heart_Rate": patient_wearable["Heart_Rate"],
    "HRV": patient_wearable["HRV"],
    "Steps": patient_wearable["Steps"],
    "Sleep_Hours": patient_wearable["Sleep_Hours"]
}])

# ==========================================
# 4. PREDICTION
# ==========================================

prediction = model.predict(input_data)[0]

probability = model.predict_proba(
    input_data
)[0][1]

# ==========================================
# 5. DISPLAY RESULT
# ==========================================

print("======================================")
print("GLUCOTWIN PATIENT PREDICTION")
print("======================================")

print("\nPatient ID:", patient_id)

print("\nCurrent Patient Data:")
print(input_data.to_string(index=False))

print("\n--------------------------------------")

if prediction == 1:
    print("⚠️ GLUCOSE SPIKE LIKELY")
else:
    print("✅ LOW GLUCOSE SPIKE RISK")

print(
    "Spike Probability:",
    round(probability * 100, 2),
    "%"
)

print("--------------------------------------")

print(
    "\nPrediction Horizon: Approximately 2 hours"
)

print(
    "\nNote: This is a research prototype using"
    " synthetic data and is not a medical diagnosis."
)
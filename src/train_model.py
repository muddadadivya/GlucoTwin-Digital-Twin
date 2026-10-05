import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. LOAD DATA
# ==========================================

ehr = pd.read_csv("data/ehr_data.csv")
wearable = pd.read_csv("data/wearable_data.csv")

# ==========================================
# 2. MERGE EHR + WEARABLE DATA
# ==========================================

data = wearable.merge(
    ehr,
    on="Patient_ID",
    how="left"
)

# Sort data correctly
data["Timestamp"] = pd.to_datetime(data["Timestamp"])

data = data.sort_values(
    ["Patient_ID", "Timestamp"]
).reset_index(drop=True)

# ==========================================
# 3. CREATE 2-HOUR FUTURE GLUCOSE
# ==========================================

data["Future_Glucose_2H"] = (
    data.groupby("Patient_ID")["Glucose"]
    .shift(-2)
)

# Remove last 2 records of each patient
data = data.dropna(
    subset=["Future_Glucose_2H"]
).copy()

# ==========================================
# 4. CREATE GLUCOSE SPIKE TARGET
# ==========================================

# A spike means glucose increases by 30 or more
# within the next 2 hours.

data["Glucose_Spike"] = (
    data["Future_Glucose_2H"] - data["Glucose"] >= 30
).astype(int)

# ==========================================
# 5. FEATURES
# ==========================================

features = [
    "Age",
    "Diabetes",
    "HbA1c",
    "Previous_Glucose",
    "Medication",
    "Family_History",
    "Glucose",
    "Heart_Rate",
    "HRV",
    "Steps",
    "Sleep_Hours"
]

X = data[features]
y = data["Glucose_Spike"]

# ==========================================
# 6. CHECK TARGET DISTRIBUTION
# ==========================================

print("======================================")
print("GLUCOTWIN GLUCOSE SPIKE PREDICTION")
print("======================================")

print("\nTotal records:", len(data))

print("\nTarget distribution:")
print(y.value_counts())

# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ==========================================
# 8. TRAIN RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ==========================================
# 9. PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# 10. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n======================================")
print("MODEL RESULTS")
print("======================================")

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# ==========================================
# 11. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance)

# ==========================================
# 12. SAVE MODEL
# ==========================================

joblib.dump(
    model,
    "models/glucose_prediction_model.pkl"
)

print("\n======================================")
print("MODEL SAVED SUCCESSFULLY!")
print("======================================")

print(
    "\nSaved as: models/glucose_prediction_model.pkl"
)
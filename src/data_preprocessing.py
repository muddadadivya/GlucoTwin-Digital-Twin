import pandas as pd

# Load EHR data
ehr = pd.read_csv("data/ehr_data.csv")

# Load wearable data
wearable = pd.read_csv("data/wearable_data.csv")

# Merge EHR and wearable data
digital_twin = wearable.merge(
    ehr,
    on="Patient_ID",
    how="left"
)

print("======================================")
print("GLUCOTWIN DIGITAL TWIN DATA")
print("======================================")

print("\nEHR records:", len(ehr))
print("Wearable records:", len(wearable))
print("Merged records:", len(digital_twin))

print("\nMerged columns:")
print(digital_twin.columns.tolist())

print("\nFirst 5 merged records:")
print(digital_twin.head())

print("\nMissing values:")
print(digital_twin.isnull().sum())
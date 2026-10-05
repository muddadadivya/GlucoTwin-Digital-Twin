import pandas as pd
import random

random.seed(42)

rows = []

for i in range(1, 101):

    patient_id = f"P{i:03d}"
    age = random.randint(30, 70)
    diabetes = random.choice([0, 1])

    if diabetes == 1:
        hba1c = round(random.uniform(6.5, 9.0), 1)
        previous_glucose = random.randint(130, 200)
        medication = random.choice([0, 1])
    else:
        hba1c = round(random.uniform(4.8, 6.4), 1)
        previous_glucose = random.randint(80, 125)
        medication = 0

    family_history = random.choice([0, 1])

    rows.append([
        patient_id,
        age,
        diabetes,
        hba1c,
        previous_glucose,
        medication,
        family_history
    ])

columns = [
    "Patient_ID",
    "Age",
    "Diabetes",
    "HbA1c",
    "Previous_Glucose",
    "Medication",
    "Family_History"
]

df = pd.DataFrame(rows, columns=columns)

df.to_csv("data/ehr_data.csv", index=False)

print("EHR CSV created successfully!")
print("Number of patients:", len(df))
print()
print(df.head())
import pandas as pd
import random
from datetime import datetime, timedelta

random.seed(42)

rows = []

for patient_num in range(1, 101):

    patient_id = f"P{patient_num:03d}"

    # Start time for each patient
    start_time = datetime(2026, 10, 1, 6, 0)

    # Generate 24 hours of wearable data
    for hour in range(24):

        timestamp = start_time + timedelta(hours=hour)

        # Simulated glucose
        glucose = random.randint(80, 180)

        # Simulated heart rate
        heart_rate = random.randint(60, 105)

        # Simulated HRV
        hrv = random.randint(30, 90)

        # Simulated steps
        steps = random.randint(0, 800)

        # Simulated sleep duration
        sleep_hours = round(random.uniform(5.0, 8.5), 1)

        rows.append([
            patient_id,
            timestamp,
            glucose,
            heart_rate,
            hrv,
            steps,
            sleep_hours
        ])

columns = [
    "Patient_ID",
    "Timestamp",
    "Glucose",
    "Heart_Rate",
    "HRV",
    "Steps",
    "Sleep_Hours"
]

df = pd.DataFrame(rows, columns=columns)

# Save the CSV
df.to_csv("data/wearable_data.csv", index=False)

print("Wearable CSV created successfully!")
print("Number of records:", len(df))
print()
print(df.head())
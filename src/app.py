import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="GlucoTwin",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# LOAD DATA
# ==========================================

ehr = pd.read_csv("data/ehr_data.csv")
wearable = pd.read_csv("data/wearable_data.csv")

wearable["Timestamp"] = pd.to_datetime(
    wearable["Timestamp"]
)

model = joblib.load(
    "models/glucose_prediction_model.pkl"
)

# ==========================================
# TITLE
# ==========================================

st.title("🩺 GlucoTwin")
st.subheader(
    "AI-Powered Digital Twin for 2-Hour Glucose Spike Prediction"
)

st.info(
    "Research Prototype | Uses synthetic patient data | "
    "Not a medical diagnostic system"
)

# ==========================================
# PATIENT SELECTION
# ==========================================

patient_ids = sorted(
    wearable["Patient_ID"].unique()
)

selected_patient = st.selectbox(
    "Select Patient",
    patient_ids
)

# ==========================================
# GET PATIENT DATA
# ==========================================

patient_ehr = ehr[
    ehr["Patient_ID"] == selected_patient
].iloc[0]

patient_wearable = wearable[
    wearable["Patient_ID"] == selected_patient
].sort_values("Timestamp")

latest = patient_wearable.iloc[-1]

# ==========================================
# PATIENT PROFILE
# ==========================================

st.header("👤 Patient Profile")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Patient ID",
        selected_patient
    )

with col2:
    st.metric(
        "Age",
        int(patient_ehr["Age"])
    )

with col3:
    st.metric(
        "Diabetes",
        "Yes" if patient_ehr["Diabetes"] == 1 else "No"
    )

with col4:
    st.metric(
        "HbA1c",
        patient_ehr["HbA1c"]
    )

# ==========================================
# CURRENT VITALS
# ==========================================

st.header("📡 Current Wearable Data")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Glucose",
        f'{int(latest["Glucose"])} mg/dL'
    )

with col2:
    st.metric(
        "Heart Rate",
        f'{int(latest["Heart_Rate"])} bpm'
    )

with col3:
    st.metric(
        "HRV",
        f'{int(latest["HRV"])} ms'
    )

with col4:
    st.metric(
        "Steps",
        int(latest["Steps"])
    )

with col5:
    st.metric(
        "Sleep",
        f'{latest["Sleep_Hours"]} hrs'
    )

# ==========================================
# AI PREDICTION
# ==========================================

st.header("🤖 AI Risk Prediction")

input_data = pd.DataFrame([{
    "Age": patient_ehr["Age"],
    "Diabetes": patient_ehr["Diabetes"],
    "HbA1c": patient_ehr["HbA1c"],
    "Previous_Glucose": patient_ehr["Previous_Glucose"],
    "Medication": patient_ehr["Medication"],
    "Family_History": patient_ehr["Family_History"],
    "Glucose": latest["Glucose"],
    "Heart_Rate": latest["Heart_Rate"],
    "HRV": latest["HRV"],
    "Steps": latest["Steps"],
    "Sleep_Hours": latest["Sleep_Hours"]
}])

prediction = model.predict(input_data)[0]

probability = model.predict_proba(
    input_data
)[0][1]

col1, col2 = st.columns(2)

with col1:

    if prediction == 1:
        st.error(
            "⚠️ HIGHER GLUCOSE SPIKE RISK"
        )
    else:
        st.success(
            "✅ LOW GLUCOSE SPIKE RISK"
        )

with col2:

    st.metric(
        "Spike Probability",
        f"{probability * 100:.1f}%"
    )

st.caption(
    "Prediction horizon: approximately 2 hours"
)

# ==========================================
# GLUCOSE TREND
# ==========================================

st.header("📈 Glucose Trend")

glucose_chart = px.line(
    patient_wearable,
    x="Timestamp",
    y="Glucose",
    markers=True,
    title="Patient Glucose Over Time"
)

glucose_chart.update_layout(
    xaxis_title="Time",
    yaxis_title="Glucose (mg/dL)"
)

st.plotly_chart(
    glucose_chart,
    use_container_width=True
)

# ==========================================
# HEART RATE TREND
# ==========================================

st.header("❤️ Heart Rate Trend")

heart_chart = px.line(
    patient_wearable,
    x="Timestamp",
    y="Heart_Rate",
    markers=True,
    title="Patient Heart Rate Over Time"
)

heart_chart.update_layout(
    xaxis_title="Time",
    yaxis_title="Heart Rate (bpm)"
)

st.plotly_chart(
    heart_chart,
    use_container_width=True
)

# ==========================================
# HRV + STEPS
# ==========================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📊 HRV Trend")

    hrv_chart = px.line(
        patient_wearable,
        x="Timestamp",
        y="HRV",
        markers=True
    )

    st.plotly_chart(
        hrv_chart,
        use_container_width=True
    )

with col2:

    st.subheader("🚶 Activity Trend")

    steps_chart = px.bar(
        patient_wearable,
        x="Timestamp",
        y="Steps"
    )

    st.plotly_chart(
        steps_chart,
        use_container_width=True
    )

# ==========================================
# EHR INFORMATION
# ==========================================

st.header("📋 Historical EHR Information")

ehr_display = pd.DataFrame({
    "Parameter": [
        "Age",
        "Diabetes",
        "HbA1c",
        "Previous Glucose",
        "Medication",
        "Family History"
    ],
    "Value": [
        patient_ehr["Age"],
        patient_ehr["Diabetes"],
        patient_ehr["HbA1c"],
        patient_ehr["Previous_Glucose"],
        patient_ehr["Medication"],
        patient_ehr["Family_History"]
    ]
})

st.dataframe(
    ehr_display,
    use_container_width=True,
    hide_index=True
)

# ==========================================
# DIGITAL TWIN EXPLANATION
# ==========================================

st.header("🔗 Digital Twin Data Fusion")

st.write(
    """
    GlucoTwin combines historical EHR information with
    dynamic wearable data to create a simulated digital
    representation of the patient.

    Historical Data:
    Age, diabetes status, HbA1c, medication and family history.

    Dynamic Data:
    Glucose, heart rate, HRV, steps and sleep.

    The Random Forest model uses these combined features
    to estimate the probability of a glucose spike
    approximately two hours ahead.
    """
)

# ==========================================
# DISCLAIMER
# ==========================================

st.warning(
    "⚠️ Disclaimer: GlucoTwin is a research prototype "
    "developed using synthetic data. It is not intended "
    "for medical diagnosis, treatment or clinical decision-making."
)
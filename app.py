import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.title("🏥 Smart Hospital Emergency Bed Demand Predictor")

# -----------------------------
# 1️⃣ SAMPLE DATASET (Synthetic)
# -----------------------------

complaints = [
    "Severe chest pain",
    "Mild headache",
    "Difficulty breathing",
    "High fever and vomiting",
    "Minor cut on hand",
    "Severe accident injury",
    "Cold and cough",
    "Stomach pain"
]

urgency_labels = [
    "High",
    "Low",
    "High",
    "Medium",
    "Low",
    "High",
    "Low",
    "Medium"
]

df_text = pd.DataFrame({
    "Complaint": complaints,
    "Urgency": urgency_labels
})

# -----------------------------
# 2️⃣ NLP MODEL
# -----------------------------

vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(df_text["Complaint"])
y_text = df_text["Urgency"]

model_text = LogisticRegression()
model_text.fit(X_text, y_text)

# -----------------------------
# 3️⃣ BED DEMAND DATASET
# -----------------------------

np.random.seed(42)
data_size = 100

crowd_level = np.random.randint(1, 4, data_size)  # 1=Low, 2=Medium, 3=High
urgent_cases = np.random.randint(1, 20, data_size)
beds_used = crowd_level * 5 + urgent_cases * 2 + np.random.randint(0, 10, data_size)

df_bed = pd.DataFrame({
    "Crowd_Level": crowd_level,
    "Urgent_Cases": urgent_cases,
    "Beds_Used": beds_used
})

X = df_bed[["Crowd_Level", "Urgent_Cases"]]
y = df_bed["Beds_Used"]

model_bed = RandomForestRegressor()
model_bed.fit(X, y)

# -----------------------------
# 4️⃣ USER INPUT SECTION
# -----------------------------

st.header("Enter Patient Complaint")
user_complaint = st.text_input("Complaint")

st.header("Select ER Crowd Level")
crowd = st.selectbox("Crowd Level", ["Low", "Medium", "High"])

crowd_mapping = {"Low":1, "Medium":2, "High":3}

if st.button("Predict"):

    # NLP Prediction
    user_vec = vectorizer.transform([user_complaint])
    predicted_urgency = model_text.predict(user_vec)[0]

    # Convert urgency to number
    urgency_number = {"Low":2, "Medium":5, "High":10}
    urgent_cases_input = urgency_number[predicted_urgency]

    # Bed Prediction
    input_data = pd.DataFrame({
        "Crowd_Level":[crowd_mapping[crowd]],
        "Urgent_Cases":[urgent_cases_input]
    })

    predicted_beds = model_bed.predict(input_data)[0]

    st.subheader("Results")
    st.write("Predicted Urgency Level:", predicted_urgency)
    st.write("Predicted Beds Required:", int(predicted_beds))

    # Simple Visualization
    fig, ax = plt.subplots()
    ax.bar(["Predicted Beds"], [predicted_beds])
    st.pyplot(fig)
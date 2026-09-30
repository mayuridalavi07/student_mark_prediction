import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("student_marks_model.joblib")

# Page settings
st.set_page_config(
    page_title="Student Marks Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Marks Prediction")
st.write("Enter student academic and study-related details to predict final exam marks.")

st.divider()

# Student inputs
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=5.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance Percentage",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)

previous_marks = st.number_input(
    "Previous Exam Marks",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=1.0
)

assignment_marks = st.number_input(
    "Assignment Marks",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

internal_marks = st.number_input(
    "Internal Marks",
    min_value=0.0,
    max_value=100.0,
    value=65.0,
    step=1.0
)

practice_test_scores = st.number_input(
    "Practice Test Scores",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=1.0
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0,
    step=0.5
)

st.divider()

# Prediction button
if st.button("Predict Final Marks", type="primary"):

    input_data = np.array([[
        study_hours,
        attendance,
        previous_marks,
        assignment_marks,
        internal_marks,
        practice_test_scores,
        sleep_hours
    ]])

    prediction = model.predict(input_data)[0]

    # Keep prediction between 0 and 100
    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Final Exam Marks: {prediction:.2f} / 100")

    st.subheader("Input Summary")

    st.write(f"📚 Study Hours: {study_hours}")
    st.write(f"📅 Attendance: {attendance}%")
    st.write(f"📝 Previous Exam Marks: {previous_marks}")
    st.write(f"📄 Assignment Marks: {assignment_marks}")
    st.write(f"📊 Internal Marks: {internal_marks}")
    st.write(f"✍️ Practice Test Scores: {practice_test_scores}")
    st.write(f"😴 Sleep Hours: {sleep_hours}")

st.divider()

st.caption("Student Marks Prediction using Machine Learning")
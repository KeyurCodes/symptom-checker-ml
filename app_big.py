import streamlit as st
import joblib
import numpy as np

# Title
st.title("🩺 Symptom Checker – Medical Assistant (Demo)")
st.write("This tool predicts the possible disease category based on symptoms.")

# Load model & vectorizer
clf = joblib.load("symptom_clf_big.joblib")
vect = joblib.load("tfidf_vect_big.joblib")

# Input area
symptoms = st.text_area("Enter symptoms:", height=150, placeholder="Example: fever, cough, sore throat")

if st.button("Predict"):
    if symptoms.strip() == "":
        st.error("Please enter symptoms.")
    else:
        text = symptoms.lower()
        X = vect.transform([text])
        pred = clf.predict(X)[0]
        probs = clf.predict_proba(X)[0]

        labels = clf.classes_

        st.success(f"Predicted Category: **{pred}**")

        # Show top probabilities
        st.write("### Probability Breakdown:")
        top_idx = np.argsort(probs)[::-1][:5]

        for i in top_idx:
            st.write(f"- **{labels[i]}** : {probs[i]:.2f}")

        st.write("---")
        st.write("Developed by KEYUR BHAVESH TRIVEDI.")
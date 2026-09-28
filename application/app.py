"""Streamlit UI for the fruit quality/ripeness/defect classifier FastAPI backend."""

import pandas as pd
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Fruit Quality Classifier", page_icon="🍎")
st.title("🍎 Fruit Quality, Ripeness & Defect Classifier")
st.write("Upload a photo of a single fruit and the model will classify its type and quality.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    if st.button("Classify"):
        with st.spinner("Running inference..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                response = requests.post(API_URL, files=files, timeout=30)
                response.raise_for_status()
            except requests.RequestException as e:
                st.error(f"Could not reach the API at {API_URL}. Is it running? ({e})")
                st.stop()

        result = response.json()
        st.success(f"Prediction: **{result['predicted_class']}** ({result['confidence']:.1%} confidence)")

        probs = pd.Series(result["all_probabilities"]).sort_values(ascending=False)
        st.bar_chart(probs)
